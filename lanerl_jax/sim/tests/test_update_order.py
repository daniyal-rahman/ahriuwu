"""``ObjAIBase.Update``'s *order*, pinned -- not just its outcomes.

Three gate-1 residuals (`AA-001`, `TURRET-001`, and the move-order Hold) were
all the same species of bug: every quantity was right and every step was
present, but two of them ran in the wrong order relative to each other. None of
the existing tests could see it, because each asserts an outcome -- the swing
period, the acquisition radius, the fact that a unit in range ends up on Hold
-- and an outcome survives a reordering that a phase does not.

So the assertions here are deliberately about *when*::

    UpdateBuffs / Move                      # AttackableUnit.Update
    AIScript.OnUpdate(diff)                 # TurretAI / LaneMinionAI
    spells.Update(diff)                     # windup advances, swings resolve
    UpdateTarget()                          # RefreshWaypoints + the swing gate
    _autoAttackCurrentCooldown -= diff/1000 # LAST

Each test below names the two neighbouring steps it separates and fails if they
swap, which is the only shape of test that would have caught these.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch
from lanerl_jax.sim.autoattack import step_autoattack
from lanerl_jax.sim.init import (TOP_OUTER_TURRET, init_lane, lane_params,
                                 spawn_minion)
from lanerl_jax.sim.minion_ai import ACTION_TIMER_MS
from lanerl_jax.sim.profiles import profile_id
from lanerl_jax.sim.state import Kind, MI_SLICE, MoveOrder, TU_SLICE, Team
from lanerl_jax.sim.step import tick
from lanerl_jax.sim.targeting import MinionType, turret_acquire

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)

DT_MS = 1000.0 / 60.0


def _aa(cd, *, windup=0.0, attacking=False, period=0.8, windup_time=0.3,
        in_range=True, has_target=True):
    """One unit, one tick of :func:`step_autoattack`, float32 like the server."""
    f32 = np.float32
    return step_autoattack(
        np.asarray([cd], f32), np.asarray([windup], f32),
        np.asarray([attacking]), np.asarray([False]),
        in_range=np.asarray([in_range]), can_attack=np.asarray([True]),
        has_target=np.asarray([has_target]),
        attack_period=np.asarray([period], f32),
        windup_time=np.asarray([windup_time], f32),
        attack_damage=np.asarray([50.0], f32),
        target_resist=np.zeros(1, f32), delta_ms=DT_MS)


def _one_tick_of_cooldown() -> np.float32:
    """``diff / 1000.0f`` in the server's own float32, not a Python float."""
    return np.float32(np.float32(DT_MS) / np.float32(1000.0))


# --------------------------------------------------------------------------
# AA-001: UpdateTarget() before `_autoAttackCurrentCooldown -= diff/1000`
# --------------------------------------------------------------------------

def test_a_unit_cannot_swing_on_the_tick_its_cooldown_expires():
    """`ObjAIBase.cs:1101-1105`: ``UpdateTarget()`` runs, and the decrement is
    the statement *after* it. So the gate reads the cooldown as it stood on
    entry, and a cooldown that only reaches zero because of **this** tick's
    subtraction buys no swing until the next tick.

    This is the whole of `AA-001`. The reversed order -- decrement, then gate
    -- is self-consistent, keeps the 97-tick period, and passes every timing
    test in `parity/tests/test_autoattack.py`; it just runs the entire sim one
    tick early. 2,824 of 2,880 minion `aa_fire` disagreements had exactly this
    shape: sim fires, server does not, targets agree.
    """
    step = _one_tick_of_cooldown()

    expiring = _aa(step)            # exactly one tick of cooldown left
    assert not bool(expiring.is_attacking[0]), (
        "swung on the tick the cooldown expired -- the decrement is running "
        "before the gate again")
    assert float(expiring.aa_cooldown[0]) == 0.0, "but it did drain to zero"

    expired = _aa(float(expiring.aa_cooldown[0]))
    assert bool(expired.is_attacking[0]), "and swings on the NEXT tick"


def test_a_swing_tick_ends_one_tick_into_its_own_new_cooldown():
    """The dumped observable that settles the order, from minion net
    1073743551: ``aacd`` 17 -> 0 -> 0 -> **802** with ``attacking`` 0,0,0,1,
    and 802/1024 = 0.7832 = 0.8 - 1/60.

    The server sets ``_autoAttackCurrentCooldown = 1/AttackSpeed`` inside
    ``UpdateTarget`` and then immediately falls into the ``-= diff/1000``
    below it, so a swing tick can never end on a whole period. Gate-then-
    decrement reproduces 802; decrement-then-gate reproduces 819, which is
    what the parity drill actually observed the sim emitting.
    """
    out = _aa(0.0, period=0.8)
    assert bool(out.is_attacking[0])
    cd = float(out.aa_cooldown[0])
    assert cd == pytest.approx(0.8 - DT_MS / 1000.0, abs=1e-6)
    assert int(round(cd * 1024)) == 802, "the dump's own quantisation"


def test_the_swing_period_is_unchanged_by_the_reordering():
    """The control that explains why this went unnoticed for so long.

    Both orderings see ``period - k*step`` at the gate of the k-th tick after a
    swing, so the 97-tick drain of a 1.6 s cooldown -- the one auto-attack fact
    with independent server measurement behind it -- is identical either way.
    `AA-001` is a phase error, not a rate error, and a rate test cannot see it.
    """
    cd, wu, atk, had = (np.zeros(1, np.float32), np.zeros(1, np.float32),
                        np.zeros(1, bool), np.zeros(1, bool))
    starts = []
    for t in range(400):
        out = step_autoattack(
            cd, wu, atk, had,
            in_range=np.ones(1, bool), can_attack=np.ones(1, bool),
            has_target=np.ones(1, bool),
            attack_period=np.full(1, 1.6, np.float32),
            windup_time=np.full(1, 0.3, np.float32),
            attack_damage=np.full(1, 50.0, np.float32),
            target_resist=np.zeros(1, np.float32), delta_ms=DT_MS)
        if bool(out.is_attacking[0]) and not bool(atk[0]):
            starts.append(t)
        cd, wu, atk, had = (out.aa_cooldown, out.aa_windup,
                            out.is_attacking, out.has_auto_attacked)
    gaps = {starts[i + 1] - starts[i] for i in range(len(starts) - 1)}
    assert gaps == {97}, f"swing spacing changed: {sorted(gaps)}"


# --------------------------------------------------------------------------
# TURRET-001: CheckForTargets() before the retention test, and a wider radius
# --------------------------------------------------------------------------

def _turret_scene(distance: float, minion_radius: float = 40.0):
    """A 750-range blue turret in slot 0 and one red melee minion in slot 1."""
    x = jnp.asarray([0.0, distance])
    y = jnp.zeros(2)
    team = jnp.asarray([Team.BLUE, Team.RED], dtype=jnp.int8)
    alive = jnp.ones(2, dtype=bool)
    kind = jnp.asarray([Kind.TURRET, Kind.LANE_MINION], dtype=jnp.int8)
    mtype = jnp.asarray([MinionType.MELEE] * 2, dtype=jnp.int8)
    return turret_acquire(
        x, y, team, alive, alive, kind, mtype,
        jnp.full(2, 750.0), jnp.full(2, -1, dtype=jnp.int8),
        jnp.full(2, -1, dtype=jnp.int8), jnp.full(2, 750.0),
        jnp.asarray([0, 1], jnp.int32),
        collision_radius=jnp.asarray([88.4, minion_radius]))


def test_turret_acquisition_reaches_further_than_its_attack_range():
    """`TURRET-001`, first half. ``GetUnitsInRange`` is a quadtree
    circle-vs-circle query (`QuadTree.cs:61-64` through
    `CollisionHandler.cs:69-72` and `ApiFunctionManager.cs:601-604`), so the
    candidate test is ``dist^2 < (Range + candidate.CollisionRadius)^2``, and
    the inequality is strict. For a 40-radius lane minion against a 750-range
    turret that is **790**, not 750.
    """
    assert int(_turret_scene(700.0)[0]) == 1, "plainly inside, a candidate"
    assert int(_turret_scene(770.0)[0]) == 1, (
        "inside 790 but outside 750 -- the quadtree still returns it")
    assert int(_turret_scene(789.9)[0]) == 1
    assert int(_turret_scene(790.0)[0]) == -1, "the C# test is strict `<`"
    assert int(_turret_scene(800.0)[0]) == -1

    # and the radius is the CANDIDATE's own, not a constant
    assert int(_turret_scene(770.0, minion_radius=10.0)[0]) == -1


def test_a_minion_in_the_annulus_is_acquired_and_dropped_on_the_same_tick():
    """`TURRET-001`, second half -- and the half that makes it a bug rather
    than a tolerance.

    ``TurretAI.OnUpdate`` picks (`:22-26`) and then drops (`:28-32`) inside one
    update, and selection between the two is priority-only with distance never
    consulted (`:43-62`). A best-priority minion sitting between 750 and 790
    is therefore chosen by ``CheckForTargets`` and nulled four lines later,
    every tick, while closer minions go unattacked -- measured at 664
    consecutive idle ticks on a real turret.

    The ordering is the test, and this is specifically the guard against
    fixing only half of `TURRET-001`: score the retention check against the
    target the turret holds *after* acquisition and the trap fires; score it
    against the one it entered the tick with -- as this sim did -- and a
    widened acquisition radius makes turret targeting strictly **worse**,
    because the sim then simply keeps the annulus minion the server threw
    away. (Before either half, the 770 case passed here for the wrong reason:
    the sim never saw the minion at all.)
    """
    patch = load_patch()
    params = lane_params(patch)
    s = init_lane(patch, include_all_turrets=False)
    turret = TU_SLICE.start
    tx, ty = TOP_OUTER_TURRET[Team.BLUE]

    def _one(distance):
        st = spawn_minion(
            s, Team.RED, profile_id(Kind.LANE_MINION, MinionType.MELEE,
                                    Team.RED),
            patch.minions["melee_red"].hp_at_level(1),
            jnp.asarray(np.asarray([[tx + distance, ty]], np.float32)),
            spawn_xy=(tx + distance, ty))
        out = tick(st, params)
        return int(out.target[turret]), int(out.is_attacking[turret])

    inside, firing = _one(700.0)
    assert inside > 0, "a minion at 700 is held and shot -- the positive control"
    assert firing == 1

    annulus, firing = _one(770.0)
    assert annulus == -1, (
        "a minion at 770 is acquired by the quadtree and dropped by the flat "
        "750 retention test in the same tick: the turret holds nothing")
    assert firing == 0, "and therefore does not swing"


# --------------------------------------------------------------------------
# The move-order Hold: UpdateTarget() returns early while a swing is in flight
# --------------------------------------------------------------------------

def _champ_pair(is_attacking: bool, windup: float, order: int):
    """Two champions 100 units apart, each targeting the other."""
    patch = load_patch()
    params = lane_params(patch)
    s = init_lane(patch, include_all_turrets=False)
    s = s.replace(
        x=s.x.at[0].set(1000.0).at[1].set(1100.0),
        y=s.y.at[0].set(1000.0).at[1].set(1000.0),
        target=s.target.at[0].set(1).at[1].set(0),
        move_order=s.move_order.at[0].set(order).at[1].set(order),
        is_attacking=s.is_attacking.at[0].set(is_attacking)
                                  .at[1].set(is_attacking),
        aa_windup=s.aa_windup.at[0].set(windup).at[1].set(windup),
        aa_cooldown=s.aa_cooldown.at[0].set(0.0).at[1].set(0.0),
    )
    return tick(s, params)


def test_hold_is_not_written_while_a_swing_is_winding_up():
    """`ObjAIBase.cs:1192-1205`. ``UpdateTarget`` hits an early ``return`` on
    the ``IsAttacking`` the unit carried into the tick, so ``RefreshWaypoints``
    -- and therefore ``UpdateMoveOrder(OrderType.Hold)`` at `:651-655` -- is
    never reached during a windup. The order the unit keeps is whatever its AI
    script wrote earlier in the same tick, which for a lane minion is the
    250 ms timer's ``AttackTo``.

    This was the largest gate-1 residual by a wide margin: 34,295 move-order
    disagreements, of which 5,390 of the 5,423 drilled cases were the single
    shape ``sim=HOLD, server=ATTACK_TO``. Writing ``hold = has_tgt & in_rng``
    unconditionally gates neither the windup nor the early return.
    """
    out = _champ_pair(is_attacking=True, windup=0.3,
                      order=MoveOrder.ATTACK_TO)
    assert int(out.move_order[0]) == MoveOrder.ATTACK_TO, (
        "Hold written during a windup -- UpdateTarget's early return is gone")
    assert int(out.move_order[1]) == MoveOrder.ATTACK_TO


def test_hold_is_written_on_the_tick_the_swing_actually_starts():
    """The other side of the same gate, so this is a phase pin and not a
    licence to stop writing Hold at all.

    ``RefreshWaypoints`` runs *before* the swing gate inside the same
    ``UpdateTarget`` (`:1239-1242` calls it, then falls into ``CanAttack()``),
    so the firing tick is exactly the tick Hold appears -- and the unit is
    ``IsAttacking`` only from the END of that tick onwards.
    """
    out = _champ_pair(is_attacking=False, windup=0.0,
                      order=MoveOrder.ATTACK_TO)
    assert bool(out.is_attacking[0]), "cooldown 0 and in range: it swings"
    assert int(out.move_order[0]) == MoveOrder.HOLD, (
        "Hold is still written on the tick RefreshWaypoints does run")


def test_a_turrets_move_order_is_never_touched_by_refresh_waypoints():
    """``BaseTurret.RefreshWaypoints`` (`BaseTurret.cs:108-110`) is an empty
    override -- "Overridden function unused by turrets" -- so a turret holding
    a target in range must NOT be moved onto Hold by this path; its order
    simply persists. One-step parity injects the server's own order every
    tick, so persisting it is what agreement looks like.
    """
    patch = load_patch()
    params = lane_params(patch)
    s = init_lane(patch, include_all_turrets=False)
    turret = TU_SLICE.start
    tx, ty = TOP_OUTER_TURRET[Team.BLUE]
    s = spawn_minion(
        s, Team.RED,
        profile_id(Kind.LANE_MINION, MinionType.MELEE, Team.RED),
        patch.minions["melee_red"].hp_at_level(1),
        jnp.asarray(np.asarray([[tx + 400.0, ty]], np.float32)),
        spawn_xy=(tx + 400.0, ty))
    s = s.replace(move_order=s.move_order.at[turret].set(MoveOrder.MOVE_TO))
    out = tick(s, params)
    assert int(out.target[turret]) > 0, "it did acquire the minion"
    assert int(out.move_order[turret]) == MoveOrder.MOVE_TO, (
        "a turret's move order was rewritten by RefreshWaypoints")


def _champ_with_route(*, is_attacking=False, windup=0.0,
                      order=MoveOrder.ATTACK_TO, gap=100.0):
    """Champion 0 chasing champion 1 with a LIVE two-point route in hand.

    ``gap`` is the centre-to-centre distance, so the caller chooses whether
    ``RefreshWaypoints`` takes its in-range branch or its re-path branch.
    """
    patch = load_patch()
    params = lane_params(patch)
    s = init_lane(patch, include_all_turrets=False)
    route = np.zeros((s.waypoints.shape[1], 2), np.float32)
    route[0] = (1000.0, 1000.0)
    route[1] = (1000.0 + gap, 1000.0)
    s = s.replace(
        x=s.x.at[0].set(1000.0).at[1].set(1000.0 + gap),
        y=s.y.at[0].set(1000.0).at[1].set(1000.0),
        target=s.target.at[0].set(1).at[1].set(-1),
        move_order=s.move_order.at[0].set(order).at[1].set(MoveOrder.NONE),
        is_attacking=s.is_attacking.at[0].set(is_attacking),
        aa_windup=s.aa_windup.at[0].set(windup),
        aa_cooldown=s.aa_cooldown.at[0].set(0.0),
        waypoints=s.waypoints.at[0].set(jnp.asarray(route)),
        waypoint_key=s.waypoint_key.at[0].set(1),
        n_waypoints=s.n_waypoints.at[0].set(2),
    )
    return s, params


def test_hold_destroys_the_route_and_does_not_merely_block_it():
    """`HOLD-001`. ``UpdateMoveOrder(OrderType.Hold, true)`` is not an order
    write: `ObjAIBase.cs:1362-1366` calls ``StopMovement()``, i.e.
    ``AttackableUnit.ResetWaypoints`` (`AttackableUnit.cs:996-1002`), which
    replaces the whole list with ``[Position]`` and sets
    ``CurrentWaypointKey = 1``.

    Asserting only ``move_order == HOLD`` -- which the test above this one
    does -- passes with the route left intact, because ``_can_move`` blocks
    Hold anyway. That is exactly how this survived: the outcome is identical
    for as long as the order stays Hold.
    """
    s, params = _champ_with_route(gap=100.0)
    out = tick(s, params)
    assert int(out.move_order[0]) == MoveOrder.HOLD, "in range: it holds"
    assert int(out.n_waypoints[0]) == 1, (
        "Hold left the route in place; StopMovement/ResetWaypoints is missing")
    assert int(out.waypoint_key[0]) == 1
    assert float(out.waypoints[0, 0, 0]) == pytest.approx(float(out.x[0]))
    assert float(out.waypoints[0, 0, 1]) == pytest.approx(float(out.y[0]))


def test_a_unit_stopped_by_hold_stays_stopped_when_the_order_flips_back():
    """The consequence, and the reason the field that caught this was
    *position* rather than any controller-state field.

    ``LaneMinionAI.ReevaluateBehavior`` (`LaneMinionAI.cs:321-331`) returns
    ``AttackTo`` for a still-valid target on its next 250 ms sweep, and
    ``UpdateMoveOrder(AttackTo)`` touches no waypoints. With the route
    destroyed the unit stays where it stopped; with the route intact it walks
    on, agreeing on order, target, ``is_attacking`` and the auto-attack clock
    the whole way. Tier 1.5 measured exactly that: minion 1073744061 separated
    5.427 u at tick 34 and 48.8 u by tick 42 with no other field disagreeing.
    """
    s, params = _champ_with_route(gap=100.0)
    held = tick(s, params)
    x_held, y_held = float(held.x[0]), float(held.y[0])
    # the AI script writes the order back; it does NOT re-path.
    resumed = held.replace(
        move_order=held.move_order.at[0].set(MoveOrder.ATTACK_TO))
    after = tick(resumed, params)
    assert float(after.x[0]) == pytest.approx(x_held), (
        "the unit resumed a route the server had already destroyed")
    assert float(after.y[0]) == pytest.approx(y_held)


def test_the_out_of_range_repath_still_produces_a_two_point_route():
    """The control. `HOLD-001`'s reset must not leak into the branch next to
    it: `:657-669` re-paths a unit whose target is out of range, and that
    branch still has to hand back a route the unit can walk.
    """
    s, params = _champ_with_route(gap=600.0)
    out = tick(s, params)
    assert int(out.move_order[0]) == MoveOrder.ATTACK_TO
    assert int(out.n_waypoints[0]) == 2, "the chase route was reset too"
    assert float(out.x[0]) > 1000.0, "it should have walked toward the target"


# --------------------------------------------------------------------------
# ORDER-004: on a tick a minion enters mid-swing, the 250 ms controller is the
# only move-order writer there is
# --------------------------------------------------------------------------
#
# `ORDER-002` established that `UpdateTarget` early-returns on the
# `IsAttacking` a unit carried into the tick, so `RefreshWaypoints` -- and
# with it both `UpdateMoveOrder(OrderType.Hold)` (`ObjAIBase.cs:655`) and the
# forced `UpdateMoveOrder(OrderType.AttackTo)` (`:604`) -- is unreachable.
# A lane minion has exactly two move-order writers in the whole server
# (`LaneMinionAI.cs:96` and those two lines), so on such a tick the order is
# whatever `LaneMinionAI.OnUpdate` wrote earlier in the SAME tick, or the
# order the unit already had.
#
# That is not a detail: it is 91.8% of the whole-corpus move-order residual.
# Drilled over 19,800 tick-pairs, 4,396 of the 4,791 LaneMinion move-order
# disagreements sit on ticks where the server's own dump has `attacking=1`,
# i.e. on ticks where NEITHER engine can reach `RefreshWaypoints` and the only
# thing that can differ is whether each side's controller swept. The three
# tests below pin all three arms of that: the sweep writes during a windup,
# nothing writes without a sweep, and an event trigger counts as a sweep.


def _minion_pair(*, gap, is_attacking=False, windup=0.0, ai_timer=0.0,
                 order=MoveOrder.HOLD, target_alive=True):
    """Two enemy lane minions ``gap`` apart, blue targeting red.

    Deliberately built with ``lane_path=None`` (the default ``tick`` takes):
    the lane-walk tail of ``ReevaluateBehavior`` is a separate mechanism with
    its own tests, and leaving it out keeps these assertions about the order
    writer and nothing else.
    """
    patch = load_patch()
    params = lane_params(patch)
    s = init_lane(patch, include_all_turrets=False)
    blue, red = MI_SLICE.start, MI_SLICE.start + 1
    for team, key, px in ((Team.BLUE, "melee_blue", 1000.0),
                          (Team.RED, "melee_red", 1000.0 + gap)):
        s = spawn_minion(
            s, team, profile_id(Kind.LANE_MINION, MinionType.MELEE, team),
            patch.minions[key].hp_at_level(1),
            jnp.asarray(np.asarray([[px, 1000.0]], np.float32)),
            spawn_xy=(px, 1000.0))
    s = s.replace(
        alive=s.alive.at[red].set(target_alive),
        target=s.target.at[blue].set(red),
        move_order=s.move_order.at[blue].set(order),
        is_attacking=s.is_attacking.at[blue].set(is_attacking),
        aa_windup=s.aa_windup.at[blue].set(windup),
        aa_cooldown=s.aa_cooldown.at[blue].set(1.0),
        ai_timer=s.ai_timer.at[blue].set(ai_timer),
    )
    return s, params, blue


def test_the_250ms_sweep_writes_attack_to_during_a_windup():
    """`LaneMinionAI.cs:59-100`. ``AIScript.OnUpdate`` runs *before*
    ``UpdateTarget`` inside ``ObjAIBase.Update`` and is not gated on
    ``IsAttacking`` at all, so a minion whose swing is in flight still sweeps
    on schedule and ``ReevaluateBehavior`` still returns ``AttackTo`` for its
    still-valid target (`:321-331`).

    The pairing with ``test_hold_is_not_written_while_a_swing_is_winding_up``
    is the point. That test proves ``RefreshWaypoints`` does *not* run here;
    this one proves something else still writes. Without both, "the order
    during a windup" is pinned in one direction only, and the residual this
    guards is symmetric -- 2,392 corpus ticks of `sim=HOLD, server=ATTACK_TO`
    against 2,085 of the exact mirror.
    """
    s, params, blue = _minion_pair(gap=100.0, is_attacking=True, windup=0.3,
                                   ai_timer=ACTION_TIMER_MS,
                                   order=MoveOrder.HOLD)
    out = tick(s, params)
    assert bool(out.is_attacking[blue]), "the swing is still in flight"
    assert int(out.move_order[blue]) == MoveOrder.ATTACK_TO, (
        "the 250 ms sweep was suppressed during a windup; only "
        "RefreshWaypoints is, and the AI script runs before it")


def test_a_mid_swing_minion_whose_timer_is_not_due_keeps_the_order_it_had():
    """The control, and the reason this residual is a *phase* residual.

    Same state, timer nowhere near due. Now neither writer runs -- the
    controller because 250 ms has not elapsed, ``RefreshWaypoints`` because of
    `ORDER-002`'s early return -- so the order is simply carried. A move order
    that is carried rather than recomputed is what turns a one-tick difference
    in *when* each side sweeps into a visible label disagreement, which is
    what the whole-corpus drill measures: 2,021 of the 4,791 disagreements sit
    on ticks where the server's own ``aitimer`` shows it did not sweep at all.
    """
    s, params, blue = _minion_pair(gap=100.0, is_attacking=True, windup=0.3,
                                   ai_timer=0.0, order=MoveOrder.HOLD)
    out = tick(s, params)
    assert int(out.move_order[blue]) == MoveOrder.HOLD, (
        "something wrote the order on a tick with no sweep and no "
        "RefreshWaypoints")


def test_a_target_dying_re_evaluates_the_move_order_with_the_timer_not_due():
    """``TargetJustDied()`` is the first term of the trigger
    (`LaneMinionAI.cs:83-93`), so a minion re-evaluates the moment its target
    stops being a valid one -- 250 ms timer or no 250 ms timer.

    Measured on the corpus dump alone: the server makes **9,277** sweeps whose
    ``minionActionTimer`` was not due (34.96% of all 26,537 minion sweeps),
    against **zero** timer-due ticks on which it did not sweep. Those event
    sweeps are the only remaining source of move-order phase difference
    between the two engines, so a regression that dropped ``just_died`` from
    the trigger would not fail any target test here -- the target ends up
    ``-1`` either way once the sweep does run -- but it would move this row.
    """
    s, params, blue = _minion_pair(gap=100.0, is_attacking=False,
                                   ai_timer=0.0, order=MoveOrder.HOLD,
                                   target_alive=False)
    out = tick(s, params)
    assert int(out.target[blue]) == -1, "the dead target is dropped"
    assert int(out.move_order[blue]) == MoveOrder.MOVE_TO, (
        "the death did not trigger a re-evaluation: the order is still the "
        "one the minion was holding, so only the 250 ms timer can be firing")


def test_finishing_an_auto_attack_holds_and_destroys_the_route():
    """`ORDER-005`. ``Spell.FinishCasting`` (`Spell.cs:1051-1065`) ends with

        if (SpellData.Flags.HasFlag(SpellDataFlags.InstantCast)) { ... }
        else { CastInfo.Owner.UpdateMoveOrder(OrderType.Hold, true); }

    for **every** completed cast, and an auto-attack is one. The lane minion's
    ``SRU_OrderMinionMeleeBasicAttack`` has ``Flags = 232448``, whose bit 2
    (``InstantCast``) is clear, so the ``else`` is what a minion takes.

    Both halves are asserted because `HOLD-001` is the precedent for exactly
    this being half-ported: ``UpdateMoveOrder(Hold, true)`` is ``StopMovement``
    is ``ResetWaypoints``, so the route dies with the order. Asserting only the
    label passes on a tree that leaves a live chase route behind, and that is
    the shape that took eight ticks to separate two otherwise bit-identical
    engines in Tier 1.5.
    """
    s, params, blue = _minion_pair(gap=100.0, is_attacking=True,
                                   windup=DT_MS / 2000.0,
                                   ai_timer=0.0, order=MoveOrder.ATTACK_TO)
    # a live two-point chase route, so "the route was destroyed" is observable
    route = np.zeros((s.waypoints.shape[1], 2), np.float32)
    route[0] = (1000.0, 1000.0)
    route[1] = (1100.0, 1000.0)
    s = s.replace(waypoints=s.waypoints.at[blue].set(jnp.asarray(route)),
                  waypoint_key=s.waypoint_key.at[blue].set(1),
                  n_waypoints=s.n_waypoints.at[blue].set(2))
    out = tick(s, params)
    assert not bool(out.is_attacking[blue]), "the wind-up ran out this tick"
    assert bool(out.has_auto_attacked[blue]), "FinishCasting ran"
    assert int(out.move_order[blue]) == MoveOrder.HOLD, (
        "FinishCasting's UpdateMoveOrder(Hold) is not ported: the order is "
        "still whatever the unit was carrying")
    assert int(out.n_waypoints[blue]) == 1, (
        "the order was written but StopMovement/ResetWaypoints was not")
    assert int(out.waypoint_key[blue]) == 1


def test_a_swing_still_winding_up_does_not_get_finish_castings_hold():
    """The control against over-fixing `ORDER-005` into "mid-swing = Hold".

    ``FinishCasting`` runs on exactly one tick of a swing -- the one where
    ``CurrentDelayTime`` reaches ``DesignerCastTime`` -- and on every other
    mid-swing tick the order is the 250 ms controller's, per `ORDER-002`. The
    corpus split is 2,050 completion ticks against 2,345 mid-wind-up ticks, so
    a fix that fired on both would trade one half of the residual for the
    other and look like progress.
    """
    s, params, blue = _minion_pair(gap=100.0, is_attacking=True, windup=0.3,
                                   ai_timer=0.0, order=MoveOrder.ATTACK_TO)
    out = tick(s, params)
    assert bool(out.is_attacking[blue]), "still winding up"
    assert int(out.move_order[blue]) == MoveOrder.ATTACK_TO, (
        "Hold was written on a tick FinishCasting does not run")
