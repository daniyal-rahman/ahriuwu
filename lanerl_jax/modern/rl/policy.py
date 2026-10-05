"""Entity-transformer policy with screen-click heads (button, screen_x, screen_y) and a value head.

Trunk: entity attention blocks pooled max+mean, a context layer over the self/global vectors, an MLP,
then an optional GRU core. Initialisation is the standard PPO recipe: orthogonal(sqrt 2) trunk,
orthogonal(0.01) action heads (a near-uniform start), orthogonal(1) value head.
"""
from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..actions import MODERN_BUTTONS
from ..obs import MODERN_ENTITY_DIM, MODERN_WORLD_SELF_DIM, N_SLOTS
from ..screen import N_SCREEN_X, N_SCREEN_Y


class PolicyConfig(NamedTuple):
    n_slots: int = N_SLOTS
    entity_dim: int = MODERN_ENTITY_DIM
    self_dim: int = MODERN_WORLD_SELF_DIM
    global_dim: int = 6
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 256
    ctx_dim: int = 256
    core_dim: int = 512
    mlp_hidden: int = 1024
    mlp_layers: int = 4
    n_buttons: int = len(MODERN_BUTTONS)
    n_screen_x: int = N_SCREEN_X
    n_screen_y: int = N_SCREEN_Y
    core: str = "mlp"              # "mlp" or "gru" (a GRUCell of core_dim on top of the trunk)
    core_norm: bool = False        # LayerNorm the GRU input
    core_residual: bool = False    # heads read trunk + GRU output
    detach_critic: bool = False    # the value head does not train the trunk


class ActionLogits(NamedTuple):
    button: jax.Array
    screen_x: jax.Array
    screen_y: jax.Array
    value: jax.Array


TRUNK = dict(kernel_init=nn.initializers.orthogonal(2.0 ** 0.5), bias_init=nn.initializers.zeros)
HEAD = dict(kernel_init=nn.initializers.orthogonal(0.01), bias_init=nn.initializers.zeros)
VALUE = dict(kernel_init=nn.initializers.orthogonal(1.0), bias_init=nn.initializers.zeros)


class _Block(nn.Module):
    cfg: PolicyConfig

    @nn.compact
    def __call__(self, x, mask):
        h = nn.LayerNorm()(x)
        x = x + nn.MultiHeadDotProductAttention(num_heads=self.cfg.n_heads, qkv_features=self.cfg.d_model)(
            h, h, mask=mask)
        h = nn.relu(nn.Dense(self.cfg.ffn_dim, **TRUNK)(nn.LayerNorm()(x)))
        return x + nn.Dense(self.cfg.d_model, **TRUNK)(h)


class LanePolicy(nn.Module):
    """``__call__(entities, pad_mask, self_vec, global_vec[, carry])`` -> logits (and the new carry for
    the GRU core). Buttons in ``button_off`` get logit -1e4 (zero probability, finite entropy)."""
    cfg: PolicyConfig = PolicyConfig()
    button_off: tuple = ()

    @nn.compact
    def __call__(self, entities, pad_mask, self_vec, global_vec, carry=None):
        c = self.cfg
        got = (tuple(entities.shape[-2:]), tuple(self_vec.shape[-1:]), tuple(global_vec.shape[-1:]))
        if got != ((c.n_slots, c.entity_dim), (c.self_dim,), (c.global_dim,)):
            raise ValueError(f"observation widths {got} do not match PolicyConfig")
        tokens = nn.Dense(c.d_model, **TRUNK)(entities)
        attn_mask = nn.make_attention_mask(~pad_mask, ~pad_mask)
        for _ in range(c.n_layers):
            tokens = _Block(c)(tokens, attn_mask)
        keep = (~pad_mask)[..., None]
        masked = jnp.where(keep, tokens, -jnp.inf)
        pooled_max = jnp.max(jnp.where(jnp.isfinite(masked), masked, -1e30), axis=-2)
        pooled_max = jnp.where(jnp.any(keep, axis=-2), pooled_max, 0.0)
        pooled_mean = jnp.sum(jnp.where(keep, tokens, 0.0), axis=-2) / jnp.maximum(jnp.sum(keep, axis=-2), 1.0)
        ctx = nn.relu(nn.Dense(c.ctx_dim, **TRUNK)(jnp.concatenate([self_vec, global_vec], axis=-1)))
        h = jnp.concatenate([pooled_max, pooled_mean, ctx], axis=-1)
        for _ in range(c.mlp_layers):
            h = nn.relu(nn.Dense(c.mlp_hidden, **TRUNK)(h))
        h = nn.Dense(c.core_dim, **TRUNK)(h)
        new_carry = None
        if c.core == "gru":
            if carry is None:
                raise ValueError("gru core needs a carry; use initial_carry")
            x = nn.LayerNorm(name="core_norm")(h) if c.core_norm else h
            new_carry, g = nn.GRUCell(features=c.core_dim, name="core_gru")(carry, x)
            h = (h + g) if c.core_residual else g
        elif c.core != "mlp":
            raise ValueError(f"unknown core {c.core!r}")
        button = nn.Dense(c.n_buttons, **HEAD)(h)
        if self.button_off:
            keep_b = jnp.asarray([i not in self.button_off for i in range(c.n_buttons)])
            button = jnp.where(keep_b, button, -1e4)
        logits = ActionLogits(button=button, screen_x=nn.Dense(c.n_screen_x, **HEAD)(h),
                              screen_y=nn.Dense(c.n_screen_y, **HEAD)(h),
                              value=nn.Dense(1, name="value_head", **VALUE)(
                                  jax.lax.stop_gradient(h) if c.detach_critic else h)[..., 0])
        return (logits, new_carry) if c.core == "gru" else logits

    def initial_carry(self, batch_shape=()):
        """Zero GRU state ``batch_shape + (core_dim,)``, also the reset value at episode ends."""
        return jnp.zeros(tuple(batch_shape) + (self.cfg.core_dim,), jnp.float32)
