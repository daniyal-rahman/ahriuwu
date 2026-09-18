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
    advance_lane_waypoints,
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


def _lane_waypoint_world(xs, *, snapshot_x=None, radius=None,
                         lane_key=None, route_end=None):
    """Small source-shaped fixture for ``LaneMinionAI.WaypointReached``."""
    n = len(xs)
    radius = np.asarray(radius if radius is not None else [40.0] * n, np.float32)
    snapshot_x = np.asarray(snapshot_x if snapshot_x is not None else xs, np.float32)
    lane_key = np.asarray(lane_key if lane_key is not None else [0] * n, np.int8)
    route_end = np.asarray(route_end if route_end is not None else [999.0] * n, np.float32)
    # all units share a short artificial immutable lane; this makes the test
    # geometry legible while exercising the production batched gather/scan.
    lane = np.broadcast_to(np.asarray([[210.0, 0.0], [500.0, 0.0]], np.float32),
                           (n, 2, 2)).copy()
    routes = np.zeros((n, 3, 2), np.float32)
    routes[:, 0, 0] = xs
    routes[:, 1, 0] = route_end
    return dict(
        kind=jnp.full(n, Kind.LANE_MINION, jnp.int8),
        alive=jnp.ones(n, bool),
        x=jnp.asarray(xs, jnp.float32), y=jnp.zeros(n, jnp.float32),
        collision_x=jnp.asarray(snapshot_x), collision_y=jnp.zeros(n, jnp.float32),
        collision_present=jnp.ones(n, bool),
        spawn_seq=jnp.arange(n, dtype=jnp.int32),
        collision_radius=jnp.asarray(radius),
        acquisition_range=jnp.full(n, 800.0, jnp.float32),
        lane_waypoints=jnp.asarray(lane), lane_waypoint_key=jnp.asarray(lane_key),
        waypoints=jnp.asarray(routes), n_waypoints=jnp.full(n, 2, jnp.int8),
        reevaluated=jnp.ones(n, bool), has_target=jnp.zeros(n, bool),
    )


def test_waypoint_reached_merges_a_collision_chain_before_applying_margin():
    """Literal `WaypointReached` geometry from LaneMinionAI.cs:276-313.

    The lone minion at 0 is 210 units from waypoint 0, well outside its
    40+25 radius. Two packed 40-radius minions at 70 and 140 are processed in
    the source OrderBy order, progressively shifting the virtual centre to 80
    and growing its radius to 120. The final 130-unit waypoint distance is
    then inside 120+25 and advances the AI's *persistent* lane cursor.
    """
    out = advance_lane_waypoints(**_lane_waypoint_world([0.0, 70.0, 140.0]))
    assert int(out.key[0]) == 1
    assert bool(out.reset_path[0])
    np.testing.assert_allclose(np.asarray(out.destination[0]), [500.0, 0.0])


def test_waypoint_cluster_uses_frozen_collision_tree_for_membership():
    """EnumerateUnitsInRange sees CollisionHandler's pre-move node positions.

    The live neighbour at 70 would make the same collision chain as above,
    but its rebuilt node was at 2,000 -- outside 0's acquisition query circle.
    It is absent from the IEnumerable entirely and cannot expand arrival.
    """
    out = advance_lane_waypoints(**_lane_waypoint_world(
        [0.0, 70.0], snapshot_x=[0.0, 2000.0]))
    assert int(out.key[0]) == 0
    np.testing.assert_allclose(np.asarray(out.destination[0]), [210.0, 0.0])


def test_lane_cursor_survives_chase_and_resets_a_route_to_the_resume_waypoint():
    """A combat route must not overwrite `currentWaypointIndex`.

    The old transient route ends at 999 (a former target). Once the target is
    gone and ReevaluateBehavior reaches its lane tail, the stored lane index
    is advanced and `SetWaypoints([Position, PathingWaypoints[index]])` is
    requested. This is the state distinction the previous single waypoint key
    could not represent.
    """
    kw = _lane_waypoint_world([150.0], lane_key=[0], route_end=[999.0])
    # Make immutable lane waypoint 0 be the one reached while coming back from
    # combat; its successor is 500, the route that must replace target=999.
    lane = np.asarray(kw["lane_waypoints"]).copy()
    lane[0] = np.asarray([[200.0, 0.0], [500.0, 0.0]], np.float32)
    kw["lane_waypoints"] = jnp.asarray(lane)
    out = advance_lane_waypoints(**kw)
    assert int(out.key[0]) == 1
    assert bool(out.reset_path[0])
    np.testing.assert_allclose(np.asarray(out.destination[0]), [500.0, 0.0])


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


def test_a_minion_holding_the_idle_champion_is_NOT_displaced_by_a_fresh_minion():
    """Direct test of the exact mechanism behind J1 gate 3's excess sim deaths
    (5 vs the server's 0-1, ``lanerl_jax/parity/tests/test_last_hit_gate.py``):
    a minion that has validly acquired the IDLE champion (``ClassifyUnit.
    CHAMPION`` = 11, worse priority than any minion's own 6-9) does NOT
    release him just because a minion later wanders into its acquisition
    range -- even though 6-9 < 11 would make that minion a strictly better
    candidate if it were ever compared.

    Read directly off ``LaneMinionAI.cs``, not assumed:
    ``ReevaluateBehavior`` (``:239-250``)::

        if (targetIsStillValid) {
            if (timeSinceLastAttack >= 4000f) { Ignore(...); targetIsStillValid = false; }
            else return OrderType.AttackTo;
        }
        if (FoundNewTarget()) { return OrderType.AttackTo; }

    returns ``AttackTo`` and never reaches the unrestricted ``FoundNewTarget()``
    call at all while ``targetIsStillValid`` -- so the "only a strictly better
    priority displaces the incumbent" comparison inside ``FoundNewTarget``
    (``:157-274``) never even runs against a live target. The ONLY path that
    can pull a minion off a live incumbent is ``FoundNewTarget(true)`` -- the
    call-for-help-restricted scan, called every tick as part of ``OnUpdate``'s
    own trigger condition, not this one -- and that needs an actual
    call-for-help event (some ally taking damage nearby), not merely "a
    minion is now in range". This is the SAME rule
    ``test_a_better_priority_does_NOT_displace_a_valid_incumbent`` already
    pins for a minion-vs-minion incumbent; checked again here with a
    CHAMPION incumbent specifically, because that is gate 3's scenario, and
    because it settles (in the sim's favour) the question of whether this is
    a targeting bug: it is not -- the sim already matches the server here.
    """
    kw = _world([0.0, 60.0, 400.0], [M, C, M], [B, R, R],
                [MinionType.MELEE, 0, MinionType.MELEE],
                target=jnp.asarray([1, -1, -1], jnp.int8),
                target_priority=jnp.asarray(
                    [ClassifyUnit.CHAMPION, ClassifyUnit.DEFAULT,
                     ClassifyUnit.DEFAULT], jnp.int8),
                had_target=jnp.asarray([True, False, False]))
    out = step_minion_ai(**kw)
    assert int(out.target[0]) == 1, (
        "a minion that already validly holds the champion must not switch "
        "to a fresh minion just because one entered range -- that requires "
        "a call for help, not mere proximity"
    )
    assert int(out.target_priority[0]) == ClassifyUnit.CHAMPION


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


def test_target_death_short_circuits_the_restricted_call_for_help_scan():
    """C# evaluates ``TargetJustDied() || FoundNewTarget(true)`` left-to-right.

    A pending help call may still win the unrestricted scan that follows, but
    it must not be reported or handled as the restricted incumbent-displacing
    path on the same tick the incumbent dies.
    """
    kw = _world([0.0, 100.0, 300.0], [M, M, C], [B, R, R],
                [MinionType.MELEE, MinionType.MELEE, 0],
                alive=jnp.asarray([True, False, True]),
                target=jnp.asarray([1, -1, -1], jnp.int8),
                target_priority=jnp.asarray(
                    [ClassifyUnit.MELEE_MINION, ClassifyUnit.DEFAULT,
                     ClassifyUnit.DEFAULT], jnp.int8),
                had_target=jnp.asarray([True, False, False]),
                ai_timer=jnp.zeros(3))
    help_p = np.full((3, 3), ClassifyUnit.DEFAULT, np.int8)
    help_p[0, 2] = ClassifyUnit.CHAMPION_ATTACKING_MINION
    kw["help_priority"] = jnp.asarray(help_p)

    out = step_minion_ai(**kw)
    assert int(out.target[0]) == 2, "unrestricted scan may still select the caller"
    assert not bool(out.cfh_switch[0]), "restricted scan was short-circuited"


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
