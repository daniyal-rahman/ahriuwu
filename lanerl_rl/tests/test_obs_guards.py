"""Runtime guards on the observation tensors.

Every test corrupts a REAL observation, built by the real builder, and asserts
the guard both catches it and says where.  A guard whose message is "assertion
failed" is only marginally better than the NaN it caught: the point is that the
error names the array, the slot and the field.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from lanerl_rl import constants as C
from lanerl_rl.frame import decode_frame
from lanerl_rl.obs import (
    OBS_VALUE_LIMIT,
    ObservationBuilder,
    ObservationError,
    check_observation,
    obs_checks_enabled,
)


def raw_frame(t_ms: int = 0, n_minions: int = 14) -> dict:
    """A two-champion lane with a wave and both outer turrets."""
    u = [
        {"id": 1, "k": "Champion", "tm": 100, "x": 1200.0, "y": 11800.0, "hp": 560,
         "mhp": 620, "vb": 1, "vr": 1, "gold": 650, "xp": 400, "lvl": 3, "cs": 12,
         "cd0": 0, "cd1": 2.0, "cd2": -1, "cd3": -1},
        {"id": 2, "k": "Champion", "tm": 200, "x": 2600.0, "y": 12600.0, "hp": 480,
         "mhp": 620, "vb": 1, "vr": 1, "gold": 700, "xp": 420, "lvl": 3, "cs": 15,
         "cd0": -1, "cd1": -1, "cd2": -1, "cd3": -1},
        {"id": 50, "k": "Turret", "tm": 100, "x": 574.0, "y": 10220.0, "hp": 3000,
         "mhp": 3000, "vb": 1, "vr": 1},
        {"id": 51, "k": "Turret", "tm": 200, "x": 4318.0, "y": 13875.0, "hp": 3000,
         "mhp": 3000, "vb": 1, "vr": 1},
    ]
    for i in range(n_minions):
        u.append({
            "id": 100 + i, "k": "LaneMinion", "tm": 100 if i % 2 == 0 else 200,
            "x": 1500.0 + 60 * i, "y": 12000.0 + 40 * i,
            "hp": max(1, 477 - 20 * i), "mhp": 477, "vb": 1, "vr": 1,
        })
    return {"t": int(t_ms), "u": u}


@pytest.fixture
def obs():
    b = ObservationBuilder(C.TEAM_BLUE)
    o = None
    for t in range(0, 66 * 6, 66):
        o = b.build(decode_frame(raw_frame(t)))
    return o


# -- the guard is on, and a clean observation passes -----------------------


def test_a_real_observation_passes_and_the_guard_is_on_by_default(obs):
    assert obs_checks_enabled(), "the guard must not be opt-in"
    check_observation(obs)  # must not raise


def test_the_env_switch_turns_the_guard_off(monkeypatch, obs):
    obs.self_vec[C.S_HP_FRAC] = np.nan
    with pytest.raises(ObservationError):
        check_observation(obs)
    monkeypatch.setenv("LANERL_OBS_STRICT", "0")
    assert not obs_checks_enabled()
    check_observation(obs)  # the same corrupt observation now sails through


def test_the_switch_is_read_per_call_not_at_import(monkeypatch):
    monkeypatch.setenv("LANERL_OBS_STRICT", "0")
    assert not obs_checks_enabled()
    monkeypatch.setenv("LANERL_OBS_STRICT", "1")
    assert obs_checks_enabled()


# -- finiteness ------------------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_entity_field_names_the_slot_and_the_field(obs, bad):
    obs.entities[13, C.E_HP_FRAC] = bad
    with pytest.raises(ObservationError) as exc:
        check_observation(obs)
    msg = str(exc.value)
    assert "entities[slot 13]" in msg
    assert "hp_frac" in msg
    assert f"field {C.E_HP_FRAC}" in msg


def test_a_nan_in_self_vec_names_the_field_not_just_the_array(obs):
    obs.self_vec[C.S_LANE_S] = np.nan
    with pytest.raises(ObservationError, match=r"self_vec\.lane_s"):
        check_observation(obs)


def test_a_nan_in_the_critic_only_arrays_is_caught_too(obs):
    """priv_* never reaches the actor, but it reaches the value head and GAE."""
    obs.priv_vec[C.P_GOLD_DIFF] = np.nan
    with pytest.raises(ObservationError, match="priv_vec.gold_diff"):
        check_observation(obs)


def test_the_count_of_bad_values_is_reported_not_just_the_first(obs):
    obs.entities[0, C.E_DS] = np.nan
    obs.entities[5, C.E_DN] = np.nan
    obs.entities[7, C.E_DIST] = np.nan
    with pytest.raises(ObservationError, match="3 non-finite"):
        check_observation(obs)


# -- range -----------------------------------------------------------------


def test_a_raw_world_coordinate_in_a_normalised_field_is_caught(obs):
    """The real bug shape: a position that skipped its /NORM_XY."""
    obs.entities[0, C.E_DS] = 12000.0
    with pytest.raises(ObservationError) as exc:
        check_observation(obs)
    assert "entities[slot 0].lane_ds" in str(exc.value)
    assert "12000" in str(exc.value)


def test_the_limit_has_real_headroom_over_what_the_builder_produces(obs):
    """A guard that fires on healthy data gets switched off within a week."""
    worst = max(
        float(np.abs(getattr(obs, n)).max())
        for n in ("entities", "self_vec", "global_vec", "priv_entities", "priv_vec")
    )
    assert worst < OBS_VALUE_LIMIT / 4.0, f"largest healthy value {worst} is too near the limit"


def test_a_value_just_inside_the_limit_passes(obs):
    obs.priv_vec[C.P_XP_DIFF] = OBS_VALUE_LIMIT - 0.1
    check_observation(obs)
    obs.priv_vec[C.P_XP_DIFF] = OBS_VALUE_LIMIT + 0.1
    with pytest.raises(ObservationError, match="beyond the normalised range"):
        check_observation(obs)


# -- shape -----------------------------------------------------------------


def test_a_wrong_shape_is_a_layout_change_and_fails_loudly(obs):
    obs.self_vec = np.zeros(C.SELF_DIM - 1, dtype=np.float32)
    with pytest.raises(ObservationError) as exc:
        check_observation(obs)
    assert "self_vec has shape" in str(exc.value)
    assert str(C.SELF_DIM) in str(exc.value)


def test_a_transposed_entity_table_is_caught(obs):
    obs.entities = obs.entities.T.copy()
    with pytest.raises(ObservationError, match="entities has shape"):
        check_observation(obs)


# -- mask consistency ------------------------------------------------------


def test_a_pad_mask_that_disagrees_with_valid_is_caught(obs):
    """Attending to an invalid slot is the "enemy on top of me" hallucination."""
    obs.entity_pad_mask[0] = not bool(obs.entity_pad_mask[0])
    with pytest.raises(ObservationError) as exc:
        check_observation(obs)
    assert "entity_pad_mask[0]" in str(exc.value)
    assert "key_padding_mask" in str(exc.value)


def test_an_all_masked_categorical_is_caught(obs):
    obs.action_mask.target[:] = False
    with pytest.raises(ObservationError, match="action_mask.target is all False"):
        check_observation(obs)


def test_the_builder_never_emits_an_all_masked_target():
    """Even with nothing in memory: _build_action_mask force-enables slot 0."""
    b = ObservationBuilder(C.TEAM_BLUE)
    empty = {"t": 0, "u": [
        {"id": 1, "k": "Champion", "tm": 100, "x": 1200.0, "y": 11800.0, "hp": 560,
         "mhp": 620, "vb": 1, "vr": 0, "gold": 500, "xp": 0, "lvl": 1, "cs": 0},
    ]}
    o = b.build(decode_frame(empty))
    assert o.action_mask.target.any()
    check_observation(o)


# -- the guard is actually wired into build() ------------------------------


def test_build_itself_runs_the_guard(monkeypatch):
    """Otherwise the guard is a function nobody calls on the hot path."""
    calls = []
    import lanerl_rl.obs as obs_mod

    real = obs_mod.check_observation
    monkeypatch.setattr(obs_mod, "check_observation",
                        lambda o, *a, **k: (calls.append(o), real(o, *a, **k))[1])
    b = ObservationBuilder(C.TEAM_BLUE)
    b.build(decode_frame(raw_frame(0)))
    assert len(calls) == 1


def test_build_raises_when_the_builder_would_emit_a_nan():
    """A corrupt input must die at build(), not three layers into the policy."""
    b = ObservationBuilder(C.TEAM_BLUE)
    bad = raw_frame(0)
    bad["u"][4]["x"] = float("nan")  # one junk coordinate off the wire
    with pytest.raises(ObservationError, match=r"entities\[slot \d+\]"):
        b.build(decode_frame(bad))


def test_a_clock_that_rewinds_does_not_put_a_huge_negative_dt_in_the_policy(caplog):
    """A reset that never reached the builder used to emit dt_norm = -75."""
    b = ObservationBuilder(C.TEAM_BLUE)
    for t in range(0, 66 * 40, 66):
        b.build(decode_frame(raw_frame(t)))
    with caplog.at_level("ERROR"):
        o = b.build(decode_frame(raw_frame(0)))  # the clock rewinds
    assert o.global_vec[C.G_DT_NORM] == pytest.approx(1.0)
    assert any("clock went backwards" in r.message for r in caplog.records)
    check_observation(o)


def test_a_long_stall_is_capped_rather_than_left_unbounded():
    b = ObservationBuilder(C.TEAM_BLUE)
    b.build(decode_frame(raw_frame(0)))
    o = b.build(decode_frame(raw_frame(60_000)))  # a minute-long gap
    assert o.global_vec[C.G_DT_NORM] == pytest.approx(C.DT_NORM_CAP)
    check_observation(o)


# -- cost ------------------------------------------------------------------


def test_the_guard_stays_inside_its_measured_budget():
    """Measured on danilogin: 535 us off, 601 us on -- 12.3%.

    The bound here is deliberately loose (50%) so ordinary machine noise cannot
    fail the build, while a guard that starts copying arrays or goes quadratic
    still does.
    """
    frames = [decode_frame(raw_frame(t)) for t in range(0, 66 * 120, 66)]

    def run(strict: str) -> float:
        import os

        prev = os.environ.get("LANERL_OBS_STRICT")
        os.environ["LANERL_OBS_STRICT"] = strict
        try:
            b = ObservationBuilder(C.TEAM_BLUE)
            for f in frames[:20]:  # warm
                b.build(f)
            ts = []
            for f in frames:
                t0 = time.perf_counter()
                b.build(f)
                ts.append(time.perf_counter() - t0)
            return statistics.median(ts)
        finally:
            if prev is None:
                os.environ.pop("LANERL_OBS_STRICT", None)
            else:
                os.environ["LANERL_OBS_STRICT"] = prev

    off = run("0")
    on = run("1")
    assert on < off * 1.5, f"guards cost {100 * (on / off - 1):.0f}% of build(), budget 50%"


def test_the_disabled_path_costs_one_environment_lookup(monkeypatch, obs):
    """If disabling it still scanned the arrays, the switch would be pointless."""
    monkeypatch.setenv("LANERL_OBS_STRICT", "0")
    scanned = []
    real_isfinite = np.isfinite
    monkeypatch.setattr(np, "isfinite", lambda *a, **k: (scanned.append(1), real_isfinite(*a, **k))[1])
    for _ in range(50):
        check_observation(obs)
    assert scanned == []
