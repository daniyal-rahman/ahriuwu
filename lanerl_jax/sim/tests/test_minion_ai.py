"""`LaneMinionAI` behaviours, each pinned to the rule it comes from.

Minion health trajectories are what a last hit is timed against, so an error
here does not present as "minions behave oddly" -- it presents as the agent
failing to farm, with a plausible learning curve. Hence a test per rule rather
than one end-to-end check.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

import jax.numpy as jnp

from lanerl_jax.sim.minion_ai import (  # noqa: E402
    ACTION_TIMER_MS,
    GIVE_UP_MS,
    IGNORE_MS,
    step_minion_ai,
)
from lanerl_jax.sim.state import Kind, MoveOrder, Team  # noqa: E402
from lanerl_jax.sim.targeting import (  # noqa: E402
    ClassifyUnit,
    MinionType,
    base_priority,
)

N = 6


def _world(xs, kinds, teams, minion_types=None, **over):
    n = len(xs)
    kind = jnp.asarray(kinds, jnp.int8)
    mt = jnp.asarray(minion_types if minion_types is not None else [0] * n, jnp.int8)
    kw = dict(
        kind=kind, alive=jnp.ones(n, bool),
        x=jnp.asarray(xs, jnp.float64), y=jnp.zeros(n),
        team=jnp.asarray(teams, jnp.int8),
        targetable=jnp.ones(n, bool), visible=jnp.ones(n, bool),
        is_attacking=jnp.zeros(n, bool),
        acquisition_range=jnp.full(n, 800.0),
        base_prio=base_priority(kind, mt),
        help_priority=jnp.full((n, n), ClassifyUnit.DEFAULT, jnp.int8),
        target=jnp.full(n, -1, jnp.int8),
        target_priority=jnp.full(n, ClassifyUnit.DEFAULT, jnp.int8),
        ai_timer=jnp.full(n, ACTION_TIMER_MS),
        ai_local_time=jnp.zeros(n),
        time_since_attack=jnp.zeros(n),
        ignore_until=jnp.zeros((n, n)),
        had_target=jnp.zeros(n, bool),
        move_order=jnp.full(n, MoveOrder.NONE, jnp.int8),
    )
    kw.update(over)
    return kw


M, C = Kind.LANE_MINION, Kind.CHAMPION
B, R = Team.BLUE, Team.RED


def test_priority_beats_distance():
    """A distant minion outranks an adjacent enemy champion.

    `ClassifyTarget` gives MELEE_MINION 9 and CHAMPION 11, lower being higher
    priority, and the minion sort is priority-first. So a minion standing next
    to an enemy champion still walks past it to hit the wave -- which is why an
    agent is not automatically punished for standing in the enemy wave, and
    *is* punished the moment it attacks (see the call-for-help test).
    """
    kw = _world([0.0, 300.0, 10.0], [M, M, C], [B, R, R],
                [MinionType.MELEE, MinionType.MELEE, 0])
    out = step_minion_ai(**kw)
    assert int(out.target[0]) == 1, "should target the far minion, not the near champion"
    assert int(out.target_priority[0]) == ClassifyUnit.MELEE_MINION


def test_the_250ms_sweep_gates_reevaluation():
    kw = _world([0.0, 300.0], [M, M], [B, R], [MinionType.MELEE] * 2,
                ai_timer=jnp.zeros(2))
    assert not bool(step_minion_ai(**kw).reevaluated[0])
    kw["ai_timer"] = jnp.full(2, ACTION_TIMER_MS)
    assert bool(step_minion_ai(**kw).reevaluated[0])


def test_a_call_for_help_reevaluates_immediately():
    """The 250 ms sweep is a ceiling, not a period.

    ``FoundNewTarget(true)`` runs every tick against the call-for-help map, so a
    minion reacts within one tick of an ally being attacked. This is the
    mechanic that punishes auto-attacking an enemy champion next to their wave.
    """
    kw = _world([0.0, 300.0, 10.0], [M, M, C], [B, R, R],
                [MinionType.MELEE, MinionType.MELEE, 0], ai_timer=jnp.zeros(3))
    assert not bool(step_minion_ai(**kw).reevaluated[0])
    # the enemy champion (index 2) is now flagged as attacking an ally of 0
    help_p = np.full((3, 3), ClassifyUnit.DEFAULT, np.int8)
    help_p[0, 2] = ClassifyUnit.CHAMPION_ATTACKING_MINION      # 5
    kw["help_priority"] = jnp.asarray(help_p)
    out = step_minion_ai(**kw)
    assert bool(out.reevaluated[0])
    assert int(out.target[0]) == 2, "should switch onto the champion that attacked"
    assert int(out.target_priority[0]) == ClassifyUnit.CHAMPION_ATTACKING_MINION


def test_an_equal_priority_target_cannot_displace_the_incumbent():
    """The hysteresis rule, quoted in the server: "The minion cannot acquire a
    new target that has the same priority as their current target".

    Seeded by setting the incumbent's ``distanceSquared`` to -1. Without it,
    equal-priority minions thrash -- the C# comment records swaps at 16 ms.
    """
    # minion 0 already holds minion 1; minion 2 is the same kind but closer
    kw = _world([0.0, 400.0, 100.0], [M, M, M], [B, R, R],
                [MinionType.MELEE] * 3,
                target=jnp.asarray([1, -1, -1], jnp.int8),
                target_priority=jnp.asarray(
                    [ClassifyUnit.MELEE_MINION, ClassifyUnit.DEFAULT,
                     ClassifyUnit.DEFAULT], jnp.int8),
                had_target=jnp.asarray([True, False, False]))
    out = step_minion_ai(**kw)
    assert int(out.target[0]) == 1, "a closer EQUAL-priority target must not steal aggro"


def test_a_better_priority_does_NOT_displace_a_valid_incumbent():
    """**The wiki's priority list describes acquisition, not re-selection.**

    ``ReevaluateBehavior`` returns ``AttackTo`` the instant it sees
    ``targetIsStillValid``, so the full ``FoundNewTarget()`` scan below it is
    unreachable while the current target holds. A minion chewing on a melee
    minion will *not* switch to an adjacent cannon even though the cannon
    outranks it.

    Written the other way round first, from the wiki's description, and the test
    failed -- correctly. Only the call-for-help scan re-targets a live
    incumbent (see the test below).
    """
    kw = _world([0.0, 400.0, 500.0], [M, M, M], [B, R, R],
                [MinionType.MELEE, MinionType.MELEE, MinionType.CANNON],
                target=jnp.asarray([1, -1, -1], jnp.int8),
                target_priority=jnp.asarray(
                    [ClassifyUnit.MELEE_MINION, ClassifyUnit.DEFAULT,
                     ClassifyUnit.DEFAULT], jnp.int8),
                had_target=jnp.asarray([True, False, False]))
    out = step_minion_ai(**kw)
    assert int(out.target[0]) == 1, "a valid incumbent is not re-prioritised"
    assert int(out.target_priority[0]) == ClassifyUnit.MELEE_MINION


def test_a_call_for_help_DOES_displace_a_valid_incumbent():
    """The one path that re-targets a live incumbent, and it needs a strictly
    better priority. This is why attacking beside the enemy wave is punished."""
    kw = _world([0.0, 400.0, 60.0], [M, M, C], [B, R, R],
                [MinionType.MELEE, MinionType.MELEE, 0],
                target=jnp.asarray([1, -1, -1], jnp.int8),
                target_priority=jnp.asarray(
                    [ClassifyUnit.MELEE_MINION, ClassifyUnit.DEFAULT,
                     ClassifyUnit.DEFAULT], jnp.int8),
                had_target=jnp.asarray([True, False, False]),
                ai_timer=jnp.zeros(3))
    help_p = np.full((3, 3), ClassifyUnit.DEFAULT, np.int8)
    help_p[0, 2] = ClassifyUnit.CHAMPION_ATTACKING_MINION       # 5 < 9
    kw["help_priority"] = jnp.asarray(help_p)
    out = step_minion_ai(**kw)
    assert int(out.target[0]) == 2
    assert int(out.target_priority[0]) == ClassifyUnit.CHAMPION_ATTACKING_MINION


def test_a_call_for_help_at_worse_priority_does_not_displace():
    kw = _world([0.0, 400.0, 500.0], [M, M, M], [B, R, R],
                [MinionType.MELEE] * 3,
                target=jnp.asarray([1, -1, -1], jnp.int8),
                target_priority=jnp.asarray(
                    [ClassifyUnit.MINION_ATTACKING_MINION, ClassifyUnit.DEFAULT,
                     ClassifyUnit.DEFAULT], jnp.int8),        # incumbent at 3
                had_target=jnp.asarray([True, False, False]),
                ai_timer=jnp.zeros(3))
    help_p = np.full((3, 3), ClassifyUnit.DEFAULT, np.int8)
    help_p[0, 2] = ClassifyUnit.CHAMPION_ATTACKING_MINION       # 5 > 3, worse
    kw["help_priority"] = jnp.asarray(help_p)
    assert int(step_minion_ai(**kw).target[0]) == 1


def test_four_seconds_without_landing_a_hit_gives_up_and_ignores():
    """``timeSinceLastAttack >= 4000`` -> ``Ignore(target, 500)``.

    Note it is time since *attacking*, not since acquiring, and it resets
    whenever the minion is attacking or has no target -- so this fires for a
    minion chasing something it cannot reach.
    """
    kw = _world([0.0, 400.0], [M, M], [B, R], [MinionType.MELEE] * 2,
                target=jnp.asarray([1, -1], jnp.int8),
                target_priority=jnp.asarray(
                    [ClassifyUnit.MELEE_MINION, ClassifyUnit.DEFAULT], jnp.int8),
                time_since_attack=jnp.asarray([GIVE_UP_MS, 0.0]),
                had_target=jnp.asarray([True, False]),
                ai_local_time=jnp.asarray([10_000.0, 10_000.0]))
    out = step_minion_ai(**kw)
    # `localTime += delta` happens at the TOP of OnUpdate (LaneMinionAI.cs:61)
    # and `Ignore` uses that already-advanced value, so the deadline is one tick
    # later than a naive reading gives.
    tick = 1000.0 / 60.0
    assert float(out.ignore_until[0, 1]) == pytest.approx(
        10_000.0 + tick + IGNORE_MS)


def test_time_since_attack_resets_while_attacking():
    kw = _world([0.0, 100.0], [M, M], [B, R], [MinionType.MELEE] * 2,
                target=jnp.asarray([1, -1], jnp.int8),
                is_attacking=jnp.asarray([True, False]),
                time_since_attack=jnp.asarray([3000.0, 0.0]),
                had_target=jnp.asarray([True, False]))
    assert float(step_minion_ai(**kw).time_since_attack[0]) == 0.0


def test_an_ignored_unit_is_not_reacquired():
    kw = _world([0.0, 300.0], [M, M], [B, R], [MinionType.MELEE] * 2,
                ignore_until=jnp.asarray([[0.0, 5000.0], [0.0, 0.0]]),
                ai_local_time=jnp.asarray([1000.0, 1000.0]))
    assert int(step_minion_ai(**kw).target[0]) == -1


def test_champions_do_not_run_the_minion_controller():
    kw = _world([0.0, 300.0], [C, M], [B, R], [0, MinionType.MELEE])
    out = step_minion_ai(**kw)
    assert not bool(out.reevaluated[0])
    assert int(out.target[0]) == -1
    assert int(out.move_order[0]) == MoveOrder.NONE


def test_dead_minions_are_untouched():
    kw = _world([0.0, 300.0], [M, M], [B, R], [MinionType.MELEE] * 2,
                alive=jnp.asarray([False, True]))
    out = step_minion_ai(**kw)
    assert not bool(out.reevaluated[0])
    assert float(out.ai_local_time[0]) == 0.0


def test_it_vmaps_and_jits():
    kw = _world([0.0, 300.0, 10.0], [M, M, C], [B, R, R],
                [MinionType.MELEE, MinionType.MELEE, 0])
    f = jax.jit(lambda xx: step_minion_ai(**{**kw, "x": xx}).target)
    out = jax.vmap(f)(jnp.tile(kw["x"], (16, 1)))
    assert out.shape == (16, 3)
    assert (np.asarray(out)[0] == np.asarray(out)[-1]).all()


def test_time_since_attack_resets_when_a_new_target_is_acquired():
    """``FoundNewTarget`` sets ``timeSinceLastAttack = 0f`` alongside
    ``SetTargetUnit`` -- the third reset condition, and the one I missed.

    Without it the timer never clears for a minion that keeps switching, so
    every minion permanently believes it has failed to attack for 4 seconds.
    Caught by an end-to-end run in which `time_since_attack` read 10,017 ms
    after ten seconds of minions visibly killing each other.
    """
    kw = _world([0.0, 300.0], [M, M], [B, R], [MinionType.MELEE] * 2,
                time_since_attack=jnp.asarray([3500.0, 0.0]))
    out = step_minion_ai(**kw)
    assert int(out.target[0]) == 1, "should have acquired"
    assert float(out.time_since_attack[0]) == 0.0
