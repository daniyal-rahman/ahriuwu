"""The policy, in flax -- dimensions matched to the production PyTorch model.

Why it exists before the trainer
--------------------------------
The J1 throughput gate is written as "measured **with the real policy in the
loop**, not the sim alone", because the entity transformer may well dominate.
A sim-only number is the one that flatters and the one that misleads.

Dimensions are `runs/rl-league-0915e/resolved_config.json` verbatim: d_model
128, 2 layers, 4 heads, ffn 256, 32 entity slots, core 512. The action heads are
`constants.BUTTONS` (8), `N_SCREEN_X` (96), `N_SCREEN_Y` (54).

The core is the MLP, not the GRU
--------------------------------
`ModelConfig.core` already offers both, and `lanerl_rl/model.py` argues for the
ablation on its own merits: GT Sophy reached superhuman Gran Turismo with a
4x2048 MLP and no recurrence, and the long-horizon memory in this stack lives
in the observation builder rather than the core. Starting on the MLP removes
BPTT, burn-in staleness, stored hidden state and chunked sequence minibatching
from the first JAX trainer in one go. The GRU comes back once the loop is
trusted -- as an ablation that was wanted anyway.

Actions are screen clicks
------------------------
The default actor uses factored button, screen_x and screen_y logits.
Experimental ``click_proposals`` adds learned attention over observed entities
to propose screen cells, mixed with the original coordinate distribution.
The actor still emits only button/x/y; the environment/server resolves what
lies under the cursor. No entity identity is an executed action argument.

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

Removed for the baseline (2026-09-23)
-------------------------------------
``PolicyConfig.frame_stack`` (4): declared, recorded, read by nothing -- the
policy has never stacked frames (`PPO-03`). A memory mechanism is an
experiment with its own contract, not a dormant field. Recoverable from
commit ``490bb38``. The input widths (``n_slots``/``entity_dim``/
``self_dim``/``global_dim``) were in the same state and are now checked
against the observation instead of removed, because they are the contract
with `obs/builder.py`.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

OWN_ACTION_INTERFACE = "viewport-structured-v4-own-action"


def expand_own_action_inputs(variables):
    """Preserve a v3 policy while inserting four zero-weight self features.

    Context input order is self then global; insert before the six global
    rows. This deliberately rejects other schemas instead of guessing.
    """
    result = jax.tree.map(lambda x: x, variables)
    kernel = result["params"]["Dense_1"]["kernel"]
    if kernel.shape[0] == 26:
        return result
    if kernel.shape[0] != 22:
        raise ValueError("own-action migration requires 16 self + 6 global inputs")
    result["params"]["Dense_1"]["kernel"] = jnp.concatenate([
        kernel[:16], jnp.zeros((4, kernel.shape[1]), kernel.dtype), kernel[16:]
    ])
    return result

from lanerl_rl.constants import (
    ENTITY_DIM,
    GLOBAL_DIM,
    N_BUTTONS,
    N_SCREEN_X,
    N_SCREEN_Y,
    N_SLOTS,
    SELF_DIM,
)
from ..obs.visible_history import VISIBLE_HISTORY_INTERFACE, HISTORY_ENTITY_DIM
from ..obs.combat_features import COMBAT_INTERFACE, COMBAT_ENTITY_DIM, COMBAT_SELF_DIM

__all__ = ["PolicyConfig", "LanePolicy", "ActionLogits", "apply_flattened_batch"]


class PolicyConfig(NamedTuple):
    #: The input widths are CHECKED against the observation on every call
    #: (`LanePolicy.__call__`), not merely recorded: flax infers them from
    #: the arrays, so without the check these four were declared, written
    #: into manifests and read by nothing (`RL-004` class).
    action_interface: str = "screen-click-v2"
    # v3 keeps tensor widths but requires authoritative death, independent of HP.
    observation_interface: str = "viewport-structured-v3"
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
    n_buttons: int = N_BUTTONS
    n_screen_x: int = N_SCREEN_X
    n_screen_y: int = N_SCREEN_Y
    #: "mlp": feed-forward (E01-E04). "gru": a GRUCell of `core_dim` on top of
    #: the trunk, the standard PPO-LSTM recipe (CleanRL ppo_lstm, OpenAI Five):
    #: memory is LEARNED, not hand-built from windup flags or frame stacks.
    #: `__call__` then takes and returns the carry.
    core: str = "mlp"
    #: GRU variants (default False keeps E06-era checkpoints bit-identical):
    #: `core_norm` LayerNorms the GRU input; `core_residual` feeds the heads
    #: trunk + GRU output instead of the GRU output alone. Motivated by the
    #: GRU clone that could not memorise 16 sequences (probes/gru_bc_*).
    core_norm: bool = False
    core_residual: bool = False
    #: INT-001 principled fix: the click is sampled from the masked joint
    #: distribution over WALKABLE screen cells (`ppo.joint_click_logits`), and
    #: the loss uses the same masked log-prob. No parameters change; the
    #: collector supplies the mask per step from the champion's position.
    click_mask: bool = False
    #: Stop the critic's gradient at the shared trunk: the value head trains
    #: only its own weights (E12a's regime), the actor trains the trunk. The
    #: decisive test of "the critic drags the trunk" (PPO-16 follow-up).
    detach_critic: bool = False
    #: Experimental observation-derived screen-cell proposals; physical
    #: screen-click-v2 protocol unchanged. Initial proposal mass is 10%.
    click_proposals: bool = False


class ActionLogits(NamedTuple):
    button: jax.Array
    screen_x: jax.Array
    screen_y: jax.Array
    value: jax.Array


class JointActionLogits(NamedTuple):
    button: jax.Array
    screen_x: jax.Array
    screen_y: jax.Array
    value: jax.Array
    click_logits: jax.Array
    proposal_mass: jax.Array | None = None


def merge_click_proposal_params(initialized, source):
    """Keep every checkpoint weight; initialise only the two new named heads."""
    result = jax.tree.map(lambda x: x, initialized)
    new, old = result['params'], source['params']
    extra = set(new) - set(old)
    if not extra <= {'click_proposal_query', 'click_proposal_gate'} or set(old)-set(new):
        raise ValueError('proposal migration may only add the two click heads')
    for name, value in old.items():
        if (jax.tree.structure(value) != jax.tree.structure(new[name]) or
            any(a.shape != b.shape for a,b in zip(jax.tree.leaves(value), jax.tree.leaves(new[name])))):
            raise ValueError(f'incompatible checkpoint parameter: {name}')
        new[name] = value
    return result


def merge_visible_history_params(initialized, source):
    """Add only the zero-weight history projection; preserve all v3 weights."""
    result = jax.tree.map(lambda x: x, initialized)
    new, old = result['params'], source['params']
    if set(new)-set(old) != {'visible_history'} or set(old)-set(new):
        raise ValueError('visible-history migration requires a v3 source and one new projection')
    for name, value in old.items():
        if (jax.tree.structure(value) != jax.tree.structure(new[name]) or
            any(a.shape != b.shape for a,b in zip(jax.tree.leaves(value), jax.tree.leaves(new[name])))):
            raise ValueError(f'incompatible checkpoint parameter: {name}')
        new[name] = value
    if any(bool(jnp.any(x != 0)) for x in jax.tree.leaves(new['visible_history'])):
        raise ValueError('history projection must start at zero')
    return result


def merge_combat_params(initialized, source):
    """Keep the v3 checkpoint exact; only two zero-weight projections are new."""
    result = jax.tree.map(lambda x: x, initialized)
    new, old = result['params'], source['params']
    extra = {'combat_entities', 'combat_self'}
    if set(new)-set(old) != extra or set(old)-set(new):
        raise ValueError('combat migration requires a v3 source and two new projections')
    for name, value in old.items():
        if (jax.tree.structure(value) != jax.tree.structure(new[name]) or
            any(a.shape != b.shape for a,b in zip(jax.tree.leaves(value), jax.tree.leaves(new[name])))):
            raise ValueError(f'incompatible checkpoint parameter: {name}')
        new[name] = value
    if any(bool(jnp.any(x != 0)) for name in extra for x in jax.tree.leaves(new[name])):
        raise ValueError('combat projections must start at zero')
    return result


#: the standard PPO recipe -- see the module docstring
TRUNK = dict(kernel_init=nn.initializers.orthogonal(2.0 ** 0.5),
             bias_init=nn.initializers.zeros)
HEAD = dict(kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros)
VALUE = dict(kernel_init=nn.initializers.orthogonal(1.0),
             bias_init=nn.initializers.zeros)
#: Stable value-head name for checkpoint compatibility and trunk diagnostics.
#: A positional Dense_N name would move when a preceding layer is inserted.
VALUE_HEAD_NAME = "value_head"


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
    def __call__(self, entities, pad_mask, self_vec, global_vec, carry=None):
        c = self.cfg
        if c.action_interface != "screen-click-v2":
            raise ValueError("policy requires screen-click-v2; pointer checkpoints need retraining")
        if c.observation_interface not in ("viewport-structured-v3", OWN_ACTION_INTERFACE, VISIBLE_HISTORY_INTERFACE, COMBAT_INTERFACE):
            raise ValueError("unsupported observation interface")
        if c.observation_interface == OWN_ACTION_INTERFACE and c.self_dim != 20:
            raise ValueError("own-action observations require self_dim=20")
        history = c.observation_interface == VISIBLE_HISTORY_INTERFACE
        combat = c.observation_interface == COMBAT_INTERFACE
        if combat and (c.entity_dim != COMBAT_ENTITY_DIM or c.self_dim != COMBAT_SELF_DIM):
            raise ValueError('combat observation dimensions do not match the declared feature contract')
        if history and (c.entity_dim != HISTORY_ENTITY_DIM or c.self_dim != 16):
            raise ValueError('visible-history observations require 76 entity and 16 self features')
        # Static shapes, so this is a trace-time check with no runtime cost.
        got = (tuple(entities.shape[-2:]), tuple(pad_mask.shape[-1:]),
               tuple(self_vec.shape[-1:]), tuple(global_vec.shape[-1:]))
        want = ((c.n_slots, c.entity_dim), (c.n_slots,), (c.self_dim,),
                (c.global_dim,))
        if got != want:
            raise ValueError(
                f"observation widths {got} do not match PolicyConfig {want} "
                "(entities, pad_mask, self_vec, global_vec)")
        tokens = nn.Dense(c.d_model, **TRUNK)(entities[..., :16] if history or combat else entities)
        if combat:
            tokens = tokens + nn.Dense(c.d_model, use_bias=False,
                kernel_init=nn.initializers.zeros, name='combat_entities')(entities[..., 16:])
        if history:
            # Separate projection preserves the original GEMM shape and all
            # checkpoint weights. No bias: unknown/ablated history is inert.
            tokens = tokens + nn.Dense(c.d_model, use_bias=False,
                kernel_init=nn.initializers.zeros, name='visible_history')(entities[..., 16:])
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
            jnp.concatenate([self_vec[..., :16] if combat else self_vec, global_vec], axis=-1))
        if combat:
            ctx = ctx + nn.Dense(c.ctx_dim, use_bias=False,
                kernel_init=nn.initializers.zeros, name='combat_self')(self_vec[..., 16:])
        ctx = nn.relu(ctx)
        h = jnp.concatenate([ent, ctx], axis=-1)

        for _ in range(c.mlp_layers):
            h = nn.relu(nn.Dense(c.mlp_hidden, **TRUNK)(h))
        h = nn.Dense(c.core_dim, **TRUNK)(h)
        new_carry = None
        if c.core == "gru":
            if carry is None:
                raise ValueError("gru core needs a carry; use LanePolicy.initial_carry")
            x = nn.LayerNorm(name="core_norm")(h) if c.core_norm else h
            new_carry, g = nn.GRUCell(features=c.core_dim, name="core_gru")(carry, x)
            h = (h + g) if c.core_residual else g
        elif c.core != "mlp":
            raise ValueError(f"unknown core {c.core!r}")

        logits = ActionLogits(
            button=nn.Dense(c.n_buttons, **HEAD)(h),
            screen_x=nn.Dense(c.n_screen_x, **HEAD)(h),
            screen_y=nn.Dense(c.n_screen_y, **HEAD)(h),
            # Stable name also lets learner diagnostics exclude this head.
            value=nn.Dense(1, name=VALUE_HEAD_NAME, **VALUE)(
                jax.lax.stop_gradient(h) if c.detach_critic else h)[..., 0],
        )
        if c.click_proposals:
            from .click_proposals import proposal_cells, mixture_click_logits
            query = nn.Dense(c.d_model, name='click_proposal_query',
                kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(h)
            scores = jnp.sum(tokens * query[..., None, :], axis=-1) / math.sqrt(c.d_model)
            gate = nn.Dense(1, name='click_proposal_gate',
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.constant(math.log(.1/.9)))(h)[..., 0]
            cells, valid = proposal_cells(entities, pad_mask, c.n_screen_x, c.n_screen_y)
            joint = mixture_click_logits(logits.screen_x, logits.screen_y, scores, gate, cells, valid)
            logits = JointActionLogits(*logits, joint, jax.nn.sigmoid(gate)*jnp.any(valid,axis=-1))
        return (logits, new_carry) if c.core == "gru" else logits

    def initial_carry(self, batch_shape=()):
        """Zero GRU state, ``batch_shape + (core_dim,)``; the reset value at
        every episode boundary."""
        return jnp.zeros(tuple(batch_shape) + (self.cfg.core_dim,), jnp.float32)


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
