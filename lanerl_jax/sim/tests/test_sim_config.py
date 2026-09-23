"""`STRUCT-003`: one step configuration object, shared by trainer and gates.

The trainer's deferred-terrain mode was run by nothing else, which is how the
fountain turrets walked for five days (`COLL-004`). These pin: the trainer
steps `SimConfig.training`; the gate's config differs from it only in an
allow-list whose every entry cites a live ledger row; `apply_orders` has no
params-free fallback; and the manifest summary is JSON.
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.sim.config import DEFAULT_ROUTE_ARTIFACT, SimConfig
from lanerl_jax.sim.init import TOP_LANE_PATH, init_lane
from lanerl_jax.sim.orders import OrderKind, Orders, apply_orders

ROOT = Path(__file__).resolve().parents[3]

#: The ONLY fields in which the parity gate may step the sim differently from
#: training, each with the ledger row that justifies it. A new entry needs a
#: row; a stale one fails `test_the_gate_config_is_the_training_config...`.
GATE_VS_TRAINING_ALLOWED = {
    # The gate diffs EVERY server tick, so it steps one tick per call and
    # applies orders before the first of each decision's two ticks.
    # `step_decision` is a `lax.scan` of the same `tick`, so this is
    # granularity, not a different computation.
    "step_ticks": "PARITY-001",
}

_SMALL = dict(n_envs=2, rollout_steps=2, n_updates=1)


def _ledger_ids() -> set:
    text = (ROOT / "docs" / "JAX_FIDELITY_LEDGER.md").read_text()
    return set(re.findall(r"^\| ([A-Z][A-Z0-9_-]+) \|", text, re.M))


def test_training_config_is_what_the_trainer_steps():
    """(a) `make_train` exposes the object its env step closes over, and it
    is `SimConfig.training` -- the deferred-terrain mode, TOP lane waves."""
    from lanerl_jax.train.trainer import TrainConfig, make_train

    built = make_train(TrainConfig(**_SMALL))
    cfg = built.sim_config
    assert SimConfig.training(route_artifact=None).differing_fields(cfg) == set()
    assert (cfg.collision_terrain, cfg.defer_collision_terrain) == (False, True)
    assert (cfg.enable_collision, cfg.enable_call_for_help) == (True, True)
    assert cfg.step_ticks == 2
    assert np.array_equal(np.asarray(cfg.lane_path),
                          np.asarray(TOP_LANE_PATH, np.float32))

    # an explicit config is passed through as the SAME object
    mine = SimConfig.training(route_artifact=None)
    assert make_train(TrainConfig(**_SMALL), sim_config=mine).sim_config is mine
    # and a loaded route table reaches it untouched
    rt, ter = object(), object()
    routed = make_train(TrainConfig(**_SMALL), route_table=rt, terrain=ter).sim_config
    assert routed.route_table is rt and routed.terrain is ter
    with pytest.raises(ValueError):
        make_train(TrainConfig(**_SMALL), sim_config=mine, route_table=rt, terrain=ter)


def test_the_gate_config_is_the_training_config_except_the_allow_list():
    """(b) Every allowed difference cites a ledger row that exists, and the
    allow-list is exact (a stale entry fails too)."""
    ids = _ledger_ids()
    for field, row in GATE_VS_TRAINING_ALLOWED.items():
        assert row in ids, f"{field} cites {row}, which has no ledger row"
    gate = SimConfig.gate(route_artifact=None)
    train = SimConfig.training(route_artifact=None)
    assert gate.differing_fields(train) == set(GATE_VS_TRAINING_ALLOWED)
    # both default to the SAME route artifact, the one `run_train` loads
    for ctor in (SimConfig.training, SimConfig.gate):
        assert inspect.signature(ctor).parameters["route_artifact"].default \
            == DEFAULT_ROUTE_ARTIFACT
    from lanerl_jax.train.run_train import DEFAULT_ROUTE_ARTIFACT as TRAIN_DEFAULT
    assert TRAIN_DEFAULT == DEFAULT_ROUTE_ARTIFACT


def test_scripted_drivers_are_named_as_not_the_training_mode():
    """The scripted parity drivers run the INLINE terrain repair. That is a
    real difference from training, made visible by name rather than hidden
    in a call site."""
    diff = SimConfig.scripted().differing_fields(SimConfig.training(route_artifact=None))
    assert diff == {"collision_terrain", "defer_collision_terrain"}


def test_apply_orders_without_params_raises():
    """(c) The `_ad_placeholder` fallback is gone."""
    s = init_lane()
    o = Orders(kind=jnp.asarray([OrderKind.NOOP, OrderKind.NOOP], jnp.int8),
               x=jnp.zeros(2), y=jnp.zeros(2),
               target=jnp.asarray([-1, -1], jnp.int8))
    with pytest.raises(TypeError):
        apply_orders(s, o)
    with pytest.raises(TypeError):
        apply_orders(s, o, None)
    import lanerl_jax.sim.orders as orders_mod
    assert not hasattr(orders_mod, "_ad_placeholder")


def test_describe_is_json_and_fingerprint_tracks_behaviour():
    """(d)"""
    cfgs = [SimConfig.training(route_artifact=None), SimConfig.gate(route_artifact=None),
            SimConfig.scripted(), SimConfig.unit_test()]
    for c in cfgs:
        d = json.loads(json.dumps(c.describe()))
        assert d["name"] == c.name
    fps = {c.fingerprint() for c in cfgs}
    assert len(fps) == len(cfgs), "distinct behaviour must fingerprint distinctly"
    assert SimConfig.training(route_artifact=None).fingerprint() == cfgs[0].fingerprint()
    assert cfgs[0].describe()["lane_path"] == "TOP_LANE_PATH"


def test_env_step_is_bit_identical_to_the_trainers_old_hand_assembled_step():
    """The plumbing changed nothing: `env_step` under `SimConfig.training`
    equals the call `trainer.py` made by hand before `STRUCT-003`."""
    from lanerl_jax.sim.step import env_step, step_decision

    cfg = SimConfig.training(route_artifact=None)
    params = cfg.params
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))

    new = jax.jit(lambda s, o: env_step(s, o, cfg))
    old = jax.jit(lambda s, o: step_decision(
        apply_orders(s, o, params), params, lane_path=path,
        collision_terrain=False, defer_collision_terrain=True))

    o = Orders(kind=jnp.asarray([OrderKind.MOVE, OrderKind.CAST_E], jnp.int8),
               x=jnp.asarray([3000.0, 0.0]), y=jnp.asarray([3000.0, 0.0]),
               target=jnp.asarray([-1, -1], jnp.int8))
    a = b = init_lane()
    for _ in range(4):
        a, b = new(a, o), old(b, o)
    for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        if jnp.issubdtype(la.dtype, jax.dtypes.prng_key):
            la, lb = jax.random.key_data(la), jax.random.key_data(lb)
        xa, xb = np.asarray(la), np.asarray(lb)
        assert np.array_equal(xa, xb, equal_nan=xa.dtype.kind == "f")
