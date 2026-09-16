"""The assembled lane: profiles, init, wave spawning, and a full-length run.

The population comparison here is the project's first **Tier-3** measurement --
distributional rather than a diff -- and it came with a lesson. See
``test_minion_population_is_close_to_the_server``.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

import jax.numpy as jnp

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch  # noqa: E402
from lanerl_jax.sim.init import (  # noqa: E402
    MINION_SPAWN,
    RUNE_HP_BONUS,
    TOP_LANE_PATH,
    TOP_OUTER_TURRET,
    init_lane,
    lane_params,
)
from lanerl_jax.sim.profiles import (  # noqa: E402
    N_PROFILES,
    build_profile_tables,
    profile_id,
)
from lanerl_jax.sim.state import Kind, TU_SLICE, Team  # noqa: E402
from lanerl_jax.sim.step import step_decision  # noqa: E402
from lanerl_jax.sim.targeting import MinionType  # noqa: E402
from lanerl_jax.sim.waves import spawn_schedule  # noqa: E402
from lanerl_jax.sim.waves_jax import step_waves_jax  # noqa: E402

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)

#: Server, `bots off`, champions never ordered, 600 s (2026-09-16). The matching
#: conditions for the sim: no champion is farming in either.
SERVER_IDLE = {"median": 21, "p95": 27, "max": 30}
#: Live-minion type mix over the same window, recovered from the dump's max-HP
#: histogram (the wire calls every one of them `LaneMinion`).
SERVER_TYPE_MIX = {"melee": 0.416, "caster": 0.536, "cannon": 0.048}
#: Fraction of live minions at full health -- an engagement measure.
SERVER_FULL_HP_FRACTION = 0.78


@pytest.fixture(scope="module")
def patch():
    return load_patch()


@pytest.fixture(scope="module")
def tables(patch):
    return build_profile_tables(patch)


def test_profiles_keep_the_blue_red_asymmetry(tables):
    """Per-(kind, type, **team**) rows, because the sides genuinely differ."""
    b = profile_id(Kind.LANE_MINION, MinionType.CANNON, Team.BLUE)
    r = profile_id(Kind.LANE_MINION, MinionType.CANNON, Team.RED)
    assert float(tables["attack_range"][b]) == 300.0
    assert float(tables["attack_range"][r]) == 280.0
    assert float(tables["gold_on_death"][b]) == pytest.approx(35.0)
    assert float(tables["gold_on_death"][r]) == pytest.approx(30.0)


#: Live-minion max-HP histogram from a 600 s server dump: the RAW Content
#: values, not the modified ones.
OBSERVED_MINION_MAX_HP = {"melee": 455.0, "caster": 290.0, "cannon": 700.0}


def test_levelscript_minion_modifiers_are_NOT_applied(tables):
    """**The server declares them and never uses them**, and I applied them.

    ``LevelScript.Init`` populates a ``MinionModifiers`` dictionary (+20 HP,
    +1 AD, +3 armour on melee, and so on) whose own declaration says::

        //These minion modifiers will remain unused for the moment, untill i
        //pull the spawning systems to MapScripts

    I read that comment and applied them anyway. The dump settles it: live
    minions have max HP **455 / 290 / 700**, the raw Content values.

    The cost was not the HP alone -- the extra armour compounds with it, so
    minions were both tankier and harder to hurt, and the sim's steady-state
    minion population ran +19% above the server's. That gap had already been
    written down as "real and unexplained"; this was the explanation.
    """
    for name, mt in (("melee", MinionType.MELEE), ("caster", MinionType.CASTER),
                     ("cannon", MinionType.CANNON)):
        r = profile_id(Kind.LANE_MINION, mt, Team.BLUE)
        assert float(tables["max_hp"][r]) == pytest.approx(
            OBSERVED_MINION_MAX_HP[name]), name
    m = profile_id(Kind.LANE_MINION, MinionType.MELEE, Team.BLUE)
    assert float(tables["attack_damage"][m]) == pytest.approx(12.0)
    assert float(tables["armor"][m]) == pytest.approx(0.0)


def test_an_unknown_profile_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="no stat profile"):
        profile_id(Kind.LANE_MINION, 99, Team.BLUE)


def test_init_places_the_measured_geometry(patch):
    s = init_lane(patch)
    assert int(s.alive.sum()) == 4          # two champions, two top turrets
    assert float(s.x[0]) == pytest.approx(26.0)
    assert float(s.hp[0]) == pytest.approx(
        patch.champion.hp_at_level(1) + RUNE_HP_BONUS)
    assert float(s.x[TU_SLICE.start]) == pytest.approx(TOP_OUTER_TURRET[Team.BLUE][0])
    assert float(s.next_spawn_ms) == 90_000.0


def test_the_lane_path_starts_at_the_measured_barracks():
    """Cross-check on two independent extractions: the lane polyline's first
    vertex (from `LanerlLane.TopLaneDefault`) against the first full-health
    minion sighting in a dump."""
    px, py = TOP_LANE_PATH[0]
    bx, by = MINION_SPAWN[Team.BLUE]
    assert abs(px - bx) < 10 and abs(py - by) < 10


def test_the_jax_spawner_matches_the_python_reference():
    """Same loop, one traceable and one readable; the readable one is what was
    validated against the recording, so they must not drift."""
    t = jnp.float32(0.0)
    ns = jnp.float32(90_000.0)
    mn = jnp.int32(0)
    cc = jnp.int32(0)
    dt = 1000.0 / 60.0
    got = []
    f = jax.jit(step_waves_jax, static_argnums=(4,))
    for _ in range(int(300_000 / dt)):
        mt, ns, mn, cc = f(t, ns, mn, cc, 2)
        if int(mt) >= 0:
            got.append((float(t), int(mt)))
        t = t + jnp.float32(dt)
    ref = spawn_schedule(300_000)
    assert [m for _, m in got] == [m for _, m in ref]
    for (a, _), (b, _) in zip(got, ref):
        assert a == pytest.approx(b, abs=20.0)


@pytest.mark.slow
def test_minion_population_is_close_to_the_server(patch):
    """**Tier 3: distributional, and the baseline has to match.**

    I first compared this against a recording in which the scripted bots were
    *farming*, and the sim looked 60% heavy (median 24 vs 15). Two champions
    removing minions is a large effect on the population, and the sim had no
    champions acting at all. Re-recorded with ``bot_teams="none"`` and
    champions never ordered -- the sim's actual conditions -- and the gap fell
    to ~15%.

    The lesson is cheap to state and was nearly expensive: **match the
    conditions before comparing distributions.** A Tier-1 diff cannot make this
    mistake because it injects state; a Tier-3 comparison can, and silently.

    Of the gap this test originally allowed, roughly half turned out to be the
    unapplied `LevelScript` minion modifiers (see
    `test_levelscript_minion_modifiers_are_NOT_applied`) -- extra HP *and*
    armour on every minion. Removing them took the population gap from +19% to
    **+10%** and moved the melee/caster mix toward the server's
    (caster 60.9% -> 59.9% against the server's 53.6%).

    The residual +10% is real and unexplained. The leading suspect is the
    absence of **minion collision**: without it casters never get pushed into
    melee reach, which is consistent with casters being the type that
    over-survives.
    """
    from lanerl_jax.sim.profiles import PROFILES

    s = init_lane(patch)
    params = lane_params(patch)
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))
    f = jax.jit(lambda st: step_decision(st, params, lane_path=path))
    mtype = np.array([q[1] for q in PROFILES])
    counts, mix, hp_frac = [], [], []
    for k in range(1, 18_001):            # 600 s of game time
        s = f(s)
        if k % 300 == 0:
            m = np.asarray((s.kind == Kind.LANE_MINION) & s.alive)
            counts.append(int(m.sum()))
            if float(s.t_ms) > 120_000 and m.any():
                mix += list(mtype[np.asarray(s.model)[m]])
                hp_frac += list(np.asarray(s.hp)[m]
                                / np.maximum(np.asarray(s.max_hp)[m], 1))
    after = counts[int(len(counts) * 0.2):]
    mix = np.array(mix)
    hp_frac = np.array(hp_frac)

    # --- the strong assertions: WHO is alive and HOW HURT they are ----------
    # These are better evidence of fidelity than the raw count, because they say
    # the right units are fighting the right amount. Both became tight only
    # after collision and the corrected radii went in.
    for name, idx, want in (("melee", 0, SERVER_TYPE_MIX["melee"]),
                            ("caster", 1, SERVER_TYPE_MIX["caster"]),
                            ("cannon", 2, SERVER_TYPE_MIX["cannon"])):
        got = float((mix == idx).mean())
        assert got == pytest.approx(want, abs=0.06), f"{name}: {got:.3f} vs {want}"
    assert float((hp_frac > 0.99).mean()) == pytest.approx(
        SERVER_FULL_HP_FRACTION, abs=0.06)

    # --- the weak one: how MANY are alive ----------------------------------
    median = float(np.median(after))
    assert median == pytest.approx(SERVER_IDLE["median"], rel=0.25), (
        f"sim median {median} vs server {SERVER_IDLE['median']}")
    assert int(s.alive.sum()) >= 4, "champions and turrets must survive"


def test_a_lane_runs_and_waves_arrive_on_schedule(patch):
    """Short smoke test: the first wave exists by 100 s and not before 90 s."""
    s = init_lane(patch)
    params = lane_params(patch)
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))
    f = jax.jit(lambda st: step_decision(st, params, lane_path=path))
    seen_before, seen_after = 0, 0
    for k in range(1, 3001):              # 100 s
        s = f(s)
        n = int(((s.kind == Kind.LANE_MINION) & s.alive).sum())
        if float(s.t_ms) < 89_000:
            seen_before = max(seen_before, n)
        else:
            seen_after = max(seen_after, n)
    assert seen_before == 0, "minions before the 90 s first wave"
    assert seen_after >= 10, f"only {seen_after} minions by 100 s"
