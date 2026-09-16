"""The policy, in flax -- dimensions matched to the production PyTorch model.

Why it exists before the trainer
--------------------------------
The J1 throughput gate is written as "measured **with the real policy in the
loop**, not the sim alone", because the entity transformer may well dominate.
A sim-only number is the one that flatters and the one that misleads.

Dimensions are `runs/rl-league-0915e/resolved_config.json` verbatim: d_model
128, 2 layers, 4 heads, ffn 256, 32 entity slots, core 512. The action heads are
`constants.BUTTONS` (8), `N_SCREEN_X` (96), `N_SCREEN_Y` (54) and a target head
over the 32 slots.

The core is the MLP, not the GRU
--------------------------------
`ModelConfig.core` already offers both, and `lanerl_rl/model.py` argues for the
ablation on its own merits: GT Sophy reached superhuman Gran Turismo with a
4x2048 MLP and no recurrence, and the long-horizon memory in this stack lives
in the observation builder rather than the core. Starting on the MLP removes
BPTT, burn-in staleness, stored hidden state and chunked sequence minibatching
from the first JAX trainer in one go. The GRU comes back once the loop is
trusted -- as an ablation that was wanted anyway.

The target head reads the entity tokens
---------------------------------------
`softmax(FC(h) . tokens^T)` -- a pointer over slots rather than a fixed 32-way
classifier, so it stays permutation-equivariant within a block and does not
learn slot indices. That property is the reason `LAST_HIT_SORT_K` is 0.
"""
from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

__all__ = ["PolicyConfig", "LanePolicy", "ActionLogits"]


class PolicyConfig(NamedTuple):
    n_slots: int = 32
    entity_dim: int = 16
    self_dim: int = 16
    global_dim: int = 6
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 256
    ctx_dim: int = 256
    core_dim: int = 512
    mlp_hidden: int = 1024
    mlp_layers: int = 4
    frame_stack: int = 4
    n_buttons: int = 8
    n_screen_x: int = 96
    n_screen_y: int = 54


class ActionLogits(NamedTuple):
    button: jax.Array
    screen_x: jax.Array
    screen_y: jax.Array
    target: jax.Array
    value: jax.Array


class _Block(nn.Module):
    cfg: PolicyConfig

    @nn.compact
    def __call__(self, x, mask):
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.cfg.n_heads, qkv_features=self.cfg.d_model
        )(h, h, mask=mask)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.cfg.ffn_dim)(h)
        h = nn.relu(h)
        h = nn.Dense(self.cfg.d_model)(h)
        return x + h


class LanePolicy(nn.Module):
    cfg: PolicyConfig = PolicyConfig()

    @nn.compact
    def __call__(self, entities, pad_mask, self_vec, global_vec):
        c = self.cfg
        tokens = nn.Dense(c.d_model)(entities)
        # `key_padding_mask=~valid` in the PyTorch model: masked slots must not
        # be attended to. An empty slot is all-zero, which is NOT the same as
        # absent -- a zero row still moves an unmasked mean.
        attn_mask = nn.make_attention_mask(~pad_mask, ~pad_mask)
        for _ in range(c.n_layers):
            tokens = _Block(c)(tokens, attn_mask)

        keep = (~pad_mask)[..., None]
        masked = jnp.where(keep, tokens, -jnp.inf)
        pooled_max = jnp.max(jnp.where(jnp.isfinite(masked), masked, -1e30), axis=-2)
        pooled_max = jnp.where(jnp.any(keep, axis=-2), pooled_max, 0.0)
        denom = jnp.maximum(jnp.sum(keep, axis=-2), 1.0)
        pooled_mean = jnp.sum(jnp.where(keep, tokens, 0.0), axis=-2) / denom
        ent = jnp.concatenate([pooled_max, pooled_mean], axis=-1)

        ctx = nn.Dense(c.ctx_dim)(jnp.concatenate([self_vec, global_vec], axis=-1))
        ctx = nn.relu(ctx)
        h = jnp.concatenate([ent, ctx], axis=-1)

        for _ in range(c.mlp_layers):
            h = nn.relu(nn.Dense(c.mlp_hidden)(h))
        h = nn.Dense(c.core_dim)(h)

        # target head: a POINTER over slots, not a 32-way classifier
        q = nn.Dense(c.d_model)(h)
        target = jnp.einsum("...d,...sd->...s", q, tokens)
        target = jnp.where(pad_mask, -1e30, target)

        return ActionLogits(
            button=nn.Dense(c.n_buttons)(h),
            screen_x=nn.Dense(c.n_screen_x)(h),
            screen_y=nn.Dense(c.n_screen_y)(h),
            target=target,
            value=nn.Dense(1)(h)[..., 0],
        )
