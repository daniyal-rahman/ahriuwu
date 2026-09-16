"""The policy's shape contract with the production PyTorch model.

Dimensions come from `runs/rl-league-0915e/resolved_config.json` and the action
heads from `lanerl_rl/constants.py`. A policy whose heads disagree with the
action space is not a smaller bug than a wrong mechanic -- it is the same class
of silent failure, and `constants.py` records what it costs: a skill order that
differed between server and observation made the action mask forbid a spell the
champion had and offer one it did not.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.train.policy import LanePolicy, PolicyConfig


@pytest.fixture(scope="module")
def built():
    cfg = PolicyConfig()
    p = LanePolicy(cfg)
    b = 3
    ent = jnp.zeros((b, cfg.n_slots, cfg.entity_dim))
    mask = jnp.zeros((b, cfg.n_slots), bool)
    sv = jnp.zeros((b, cfg.self_dim))
    gv = jnp.zeros((b, cfg.global_dim))
    v = p.init(jax.random.key(0), ent, mask, sv, gv)
    return p, v, (ent, mask, sv, gv), cfg


def test_head_widths_match_the_action_space(built):
    """8 buttons x 96 screen_x x 54 screen_y x 32 targets."""
    from lanerl_rl import constants as C

    p, v, args, cfg = built
    out = p.apply(v, *args)
    assert out.button.shape[-1] == len(C.BUTTONS) == cfg.n_buttons
    assert out.screen_x.shape[-1] == C.N_SCREEN_X
    assert out.screen_y.shape[-1] == C.N_SCREEN_Y
    assert out.target.shape[-1] == C.N_SLOTS
    assert out.value.shape == (3,)


def test_dimensions_match_the_production_config():
    """`resolved_config.json` from the last league run."""
    cfg = PolicyConfig()
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.ffn_dim) == (128, 2, 4, 256)
    assert (cfg.core_dim, cfg.mlp_hidden, cfg.mlp_layers) == (512, 1024, 4)
    assert cfg.n_slots == 32


def test_masked_slots_cannot_be_targeted(built):
    """A padded slot must be unreachable, not merely unlikely."""
    p, v, (ent, _, sv, gv), cfg = built
    mask = np.zeros((3, cfg.n_slots), bool)
    mask[:, 5:] = True
    out = p.apply(v, ent, jnp.asarray(mask), sv, gv)
    probs = jax.nn.softmax(out.target, axis=-1)
    assert float(np.asarray(probs)[:, 5:].max()) < 1e-12


def test_padded_slots_do_not_change_the_output(built):
    """An empty slot is all-zero, and a zero row is NOT the same as absent --
    it still moves an unmasked mean. This is what `key_padding_mask` is for."""
    p, v, (_, _, sv, gv), cfg = built
    rng = np.random.default_rng(0)
    ent = rng.normal(size=(1, cfg.n_slots, cfg.entity_dim)).astype(np.float32)
    mask = np.zeros((1, cfg.n_slots), bool)
    mask[0, 4:] = True
    a = p.apply(v, jnp.asarray(ent), jnp.asarray(mask), sv[:1], gv[:1])
    # scribble over the masked rows; nothing may move
    ent2 = ent.copy()
    ent2[0, 4:] = rng.normal(size=ent2[0, 4:].shape)
    b = p.apply(v, jnp.asarray(ent2), jnp.asarray(mask), sv[:1], gv[:1])
    for x, y in ((a.button, b.button), (a.screen_x, b.screen_x),
                 (a.value, b.value)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-5)


def test_the_target_head_is_a_pointer_not_a_classifier(built):
    """`softmax(FC(h) . tokens^T)`: permutation-equivariant within a block.

    Permuting two valid slots must permute their logits and leave the others
    alone. A fixed 32-way classifier would not do this, and the property is why
    `LAST_HIT_SORT_K` can be 0 -- the head finds the weak minion instead of
    learning that it lives at index 13.
    """
    p, v, (_, _, sv, gv), cfg = built
    rng = np.random.default_rng(1)
    ent = rng.normal(size=(1, cfg.n_slots, cfg.entity_dim)).astype(np.float32)
    mask = np.zeros((1, cfg.n_slots), bool)
    base = p.apply(v, jnp.asarray(ent), jnp.asarray(mask), sv[:1], gv[:1])
    swapped = ent.copy()
    swapped[0, [2, 7]] = swapped[0, [7, 2]]
    out = p.apply(v, jnp.asarray(swapped), jnp.asarray(mask), sv[:1], gv[:1])
    t0 = np.asarray(base.target)[0]
    t1 = np.asarray(out.target)[0]
    assert t1[2] == pytest.approx(t0[7], abs=1e-4)
    assert t1[7] == pytest.approx(t0[2], abs=1e-4)
    others = [i for i in range(cfg.n_slots) if i not in (2, 7)]
    np.testing.assert_allclose(t1[others], t0[others], atol=1e-4)


def test_it_jits_and_vmaps(built):
    p, v, args, cfg = built
    f = jax.jit(lambda *a: p.apply(v, *a))
    assert f(*args).button.shape == (3, cfg.n_buttons)
    batched = jax.vmap(f)(*[jnp.broadcast_to(a, (4,) + a.shape) for a in args])
    assert batched.button.shape == (4, 3, cfg.n_buttons)


def test_parameter_count_is_in_the_right_ballpark(built):
    """The production model is ~32M agent-block params against a frozen
    backbone; this is the standalone lane policy and should be far smaller."""
    _, v, _, _ = built
    n = sum(x.size for x in jax.tree.leaves(v))
    assert 1e6 < n < 2e7, f"{n:,} parameters"
