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

Initialisation is the standard PPO recipe, and it needed both halves
--------------------------------------------------------------------
Trunk layers use ``orthogonal(sqrt(2))`` with zero bias; the action heads use
``orthogonal(0.01)``; the value head ``orthogonal(1.0)``.

Small heads alone are not enough, which is worth stating because it is the
version I shipped first. With flax's default ``lecun_normal`` trunk, four
1024-wide layers grow the activation magnitude enough that even 0.01-scaled
heads produce structured logits: measured **10.63 nats against a 14.099
uniform maximum** on real observations, i.e. a policy that starts 25% peaked in
a direction the PRNG chose. The unit test missed it because it feeds zero
observations, where the trunk output is small and the heads look uniform.
Scaling the trunk is what actually makes the start uniform.

The value head keeps a full-scale initialisation: it is a regression, not a
distribution, and shrinking it only makes the critic start further from useful.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from lanerl_rl.constants import (
    ENTITY_DIM,
    GLOBAL_DIM,
    N_BUTTONS,
    N_SCREEN_X,
    N_SCREEN_Y,
    N_SLOTS,
    SELF_DIM,
)

__all__ = ["PolicyConfig", "LanePolicy", "ActionLogits", "apply_flattened_batch"]


class PolicyConfig(NamedTuple):
    n_slots: int = N_SLOTS
    entity_dim: int = ENTITY_DIM
    self_dim: int = SELF_DIM
    global_dim: int = GLOBAL_DIM
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 256
    ctx_dim: int = 256
    core_dim: int = 512
    mlp_hidden: int = 1024
    mlp_layers: int = 4
    frame_stack: int = 4
    n_buttons: int = N_BUTTONS
    n_screen_x: int = N_SCREEN_X
    n_screen_y: int = N_SCREEN_Y


class ActionLogits(NamedTuple):
    button: jax.Array
    screen_x: jax.Array
    screen_y: jax.Array
    target: jax.Array
    value: jax.Array


#: the standard PPO recipe -- see the module docstring
TRUNK = dict(kernel_init=nn.initializers.orthogonal(2.0 ** 0.5),
             bias_init=nn.initializers.zeros)
HEAD = dict(kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros)
VALUE = dict(kernel_init=nn.initializers.orthogonal(1.0),
             bias_init=nn.initializers.zeros)


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
        h = nn.Dense(self.cfg.ffn_dim, **TRUNK)(h)
        h = nn.relu(h)
        h = nn.Dense(self.cfg.d_model, **TRUNK)(h)
        return x + h


class LanePolicy(nn.Module):
    cfg: PolicyConfig = PolicyConfig()

    @nn.compact
    def __call__(self, entities, pad_mask, self_vec, global_vec):
        c = self.cfg
        tokens = nn.Dense(c.d_model, **TRUNK)(entities)
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

        ctx = nn.Dense(c.ctx_dim, **TRUNK)(
            jnp.concatenate([self_vec, global_vec], axis=-1))
        ctx = nn.relu(ctx)
        h = jnp.concatenate([ent, ctx], axis=-1)

        for _ in range(c.mlp_layers):
            h = nn.relu(nn.Dense(c.mlp_hidden, **TRUNK)(h))
        h = nn.Dense(c.core_dim, **TRUNK)(h)

        # target head: a POINTER over slots, not a 32-way classifier
        q = nn.Dense(c.d_model, **HEAD)(h)
        target = jnp.einsum("...d,...sd->...s", q, tokens)
        # -1e9, not -1e30: the softmax is insensitive to the magnitude (it
        # subtracts the max) but -1e30 SQUARED overflows float32, so any
        # variance or norm computed over these logits goes to inf.
        target = jnp.where(pad_mask, -1e9, target)

        return ActionLogits(
            button=nn.Dense(c.n_buttons, **HEAD)(h),
            screen_x=nn.Dense(c.n_screen_x, **HEAD)(h),
            screen_y=nn.Dense(c.n_screen_y, **HEAD)(h),
            target=target,
            value=nn.Dense(1, **VALUE)(h)[..., 0],
        )


def apply_flattened_batch(policy: LanePolicy, variables, entities, pad_mask,
                          self_vec, global_vec) -> ActionLogits:
    """Apply an independent policy batch as one leading axis.

    The policy has no interaction across leading examples. Flattening them
    therefore preserves its architecture and probability distribution while
    allowing XLA to use one larger matmul layout instead of a nested batch
    layout. Floating-point reduction order may differ at ordinary GPU roundoff
    level; the numerical contract is covered by ``test_policy``.
    """
    leading = entities.shape[:-2]
    n = math.prod(leading)
    flat_entities = entities.reshape((n,) + entities.shape[-2:])
    flat_mask = pad_mask.reshape((n,) + pad_mask.shape[-1:])
    flat_self = self_vec.reshape((n,) + self_vec.shape[-1:])
    flat_global = global_vec.reshape((n,) + global_vec.shape[-1:])
    flat = policy.apply(variables, flat_entities, flat_mask, flat_self, flat_global)
    return jax.tree.map(lambda a: a.reshape(leading + a.shape[1:]), flat)
