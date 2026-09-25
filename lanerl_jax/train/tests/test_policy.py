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

from lanerl_jax.train.policy import LanePolicy, PolicyConfig, apply_flattened_batch


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
    """8 buttons x 96 screen_x x 54 screen_y; no actor entity target."""
    from lanerl_rl import constants as C

    p, v, args, cfg = built
    out = p.apply(v, *args)
    assert out.button.shape[-1] == len(C.BUTTONS) == cfg.n_buttons
    assert out.screen_x.shape[-1] == C.N_SCREEN_X
    assert out.screen_y.shape[-1] == C.N_SCREEN_Y
    assert out._fields == ("button", "screen_x", "screen_y", "value")
    assert out.value.shape == (3,)


def test_dimensions_match_the_production_config():
    """`resolved_config.json` from the last league run."""
    cfg = PolicyConfig()
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.ffn_dim) == (128, 2, 4, 256)
    assert (cfg.core_dim, cfg.mlp_hidden, cfg.mlp_layers) == (512, 1024, 4)
    assert cfg.n_slots == 32


def test_input_widths_are_checked_against_the_config(built):
    """`n_slots`/`entity_dim`/`self_dim`/`global_dim` were declared, recorded
    and read by nothing (flax infers widths from the arrays), so an
    observation that drifted from the config was silently accepted
    (`RL-004` class). They are now a checked contract."""
    p, v, (ent, mask, sv, gv), cfg = built
    with pytest.raises(ValueError, match="PolicyConfig"):
        p.apply(v, ent, mask, jnp.zeros((3, cfg.self_dim + 1)), gv)
    wide = LanePolicy(cfg._replace(entity_dim=cfg.entity_dim + 1))
    with pytest.raises(ValueError, match="PolicyConfig"):
        wide.init(jax.random.key(0), ent, mask, sv, gv)


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
                 (a.screen_y, b.screen_y), (a.value, b.value)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-5)


def test_entity_permutation_preserves_screen_policy(built):
    """Slot identity cannot act as an implicit target selector."""
    p, v, (_, _, sv, gv), cfg = built
    rng = np.random.default_rng(1)
    ent = rng.normal(size=(1, cfg.n_slots, cfg.entity_dim)).astype(np.float32)
    mask = np.zeros((1, cfg.n_slots), bool)
    base = p.apply(v, jnp.asarray(ent), jnp.asarray(mask), sv[:1], gv[:1])
    swapped = ent.copy()
    swapped[0, [2, 7]] = swapped[0, [7, 2]]
    out = p.apply(v, jnp.asarray(swapped), jnp.asarray(mask), sv[:1], gv[:1])
    for a, b in zip(base, out):
        np.testing.assert_allclose(a, b, atol=1e-4)
    names = [jax.tree_util.keystr(path) for path, _ in
             jax.tree_util.tree_flatten_with_path(v)[0]]
    assert not any("target" in name for name in names)


def test_it_jits_and_vmaps(built):
    p, v, args, cfg = built
    f = jax.jit(lambda *a: p.apply(v, *a))
    assert f(*args).button.shape == (3, cfg.n_buttons)
    batched = jax.vmap(f)(*[jnp.broadcast_to(a, (4,) + a.shape) for a in args])
    assert batched.button.shape == (4, 3, cfg.n_buttons)


def test_flattened_batch_preserves_policy_distribution(built):
    """A layout-only batch flatten may differ only by GPU roundoff."""
    p, v, args, _ = built
    nested = tuple(jnp.broadcast_to(a, (4, 2) + a.shape) for a in args)
    ordinary = p.apply(v, *nested)
    flat = apply_flattened_batch(p, v, *nested)
    for a, b in zip(ordinary, flat):
        delta = np.abs(np.asarray(a) - np.asarray(b))
        assert float(delta.max()) < 2e-5
    # Action heads are the behavioural surface.  Check their distributions,
    # rather than only raw logit scale.
    for a, b in zip(ordinary[:3], flat[:3]):
        pa = np.asarray(jax.nn.softmax(a, axis=-1))
        pb = np.asarray(jax.nn.softmax(b, axis=-1))
        kl = np.sum(pa * (np.log(np.maximum(pa, 1e-30))
                          - np.log(np.maximum(pb, 1e-30))), axis=-1)
        assert float(kl.max()) < 1e-7


def test_parameter_count_is_in_the_right_ballpark(built):
    """The production model is ~32M agent-block params against a frozen
    backbone; this is the standalone lane policy and should be far smaller."""
    _, v, _, _ = built
    n = sum(x.size for x in jax.tree.leaves(v))
    assert 1e6 < n < 2e7, f"{n:,} parameters"


def test_each_head_starts_uniform_over_its_OWN_support(built):
    """Real observations should start each head near ln(its support)."""
    import numpy as _np

    from lanerl_jax.obs.builder import build_observation
    from lanerl_jax.obs.frame import make_lane_frame
    from lanerl_jax.sim.init import TOP_OUTER_TURRET, init_lane, lane_params
    from lanerl_jax.sim.state import Team
    from lanerl_jax.train.ppo import factored_entropy

    p, v, _, _ = built
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], (1131.8, 1426.3))
    ob = build_observation(init_lane(), 0, frame, params=lane_params())
    out = p.apply(v, ob.entities[None], ob.entity_pad_mask[None],
                  ob.self_vec[None], ob.global_vec[None])

    for name, logits, support in (
        ("button", out.button, 8),
        ("screen_x", out.screen_x, 96),
        ("screen_y", out.screen_y, 54),
    ):
        h = float(factored_entropy([logits])[0])
        assert h == pytest.approx(float(_np.log(support)), abs=0.01), (
            f"{name}: {h:.4f} vs uniform {_np.log(support):.4f}")



def test_fully_padded_observation_has_finite_outputs(built):
    """An empty visible entity set must not cause attention/gradient NaNs."""
    p, v, (ent, _, sv, gv), cfg = built
    out = p.apply(v, ent, jnp.ones((3, cfg.n_slots), bool), sv, gv)
    for logits in out:
        assert np.isfinite(np.asarray(logits)).all()
