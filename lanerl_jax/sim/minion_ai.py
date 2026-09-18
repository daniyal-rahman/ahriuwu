"""``LaneMinionAI``: the hardest mechanic here, and the one last-hitting rests on.

Why it is the hard one
----------------------
Everything else in the lane is a formula.  This is a stateful controller with
its own clock, a hysteresis rule, a 4-second give-up timer, a 500 ms ignore
list and a call-for-help channel -- and minion health trajectories are what a
last hit is timed against, so an error here does not show up as "minions behave
oddly", it shows up as the agent failing to farm.

The controller, from ``LaneMinionAI.OnUpdate``::

    localTime += delta;
    if (IsAttacking || TargetUnit == null) timeSinceLastAttack = 0;
    else                                   timeSinceLastAttack += delta;
    minionActionTimer += delta;
    if (MovementParameters == null
        && (TargetJustDied() || FoundNewTarget(true) || minionActionTimer >= 250f)) {
        UpdateMoveOrder(ReevaluateBehavior(delta));
        minionActionTimer = 0;
    }
    if (callsForHelpMayBeCleared) { callsForHelpMayBeCleared = false;
                                    unitsAttackingAllies.Clear(); }

and ``ReevaluateBehavior``::

    if (targetIsStillValid) {
        if (timeSinceLastAttack >= 4000f) { Ignore(TargetUnit); targetIsStillValid = false; }
        else return AttackTo;
    }
    if (FoundNewTarget())      return AttackTo;
    else if (TargetUnit != null) { CancelAutoAttack(false, true); SetTargetUnit(null); }
    while (currentWaypointIndex < PathingWaypoints.Count && WaypointReached())
        currentWaypointIndex++;
    if (notYetOutOfRange) { ...SetWaypoints...; return MoveTo; }
    return Stop;

Three details that are easy to lose
-----------------------------------
**The 250 ms sweep is a ceiling, not a period.** Two events re-evaluate
immediately: the current target dying, and a call for help. So a minion can
react within one tick of an ally being attacked -- which is exactly the
mechanic that punishes an agent for auto-attacking an enemy champion next to
their wave.

**The hysteresis lives in `FoundNewTarget`, not here.** A still-valid incumbent
is seeded with ``distanceSquared = -1``, so only a *strictly better priority*
displaces it. Without that, equal-priority minions thrash; the server's own
comment records swaps as fast as 16 ms.

**A valid incumbent is never re-prioritised by the regular sweep.**
``ReevaluateBehavior`` returns ``AttackTo`` the moment it sees
``targetIsStillValid``; the full ``FoundNewTarget()`` scan below it is
unreachable while the current target holds. So a minion chewing on a melee
minion will *not* switch to an adjacent cannon, even though the cannon outranks
it -- the wiki's priority list describes acquisition, not continuous
re-selection. The **only** thing that re-targets a live incumbent is
``FoundNewTarget(true)``, the call-for-help scan in the trigger condition, and
it too requires a strictly better priority.

That asymmetry is the whole reason attacking an enemy champion next to their
wave is punished while merely standing there is not: standing generates no call
for help, so committed minions keep hitting what they were hitting.

**The 4 s rule uses time-since-*attack*, not time-since-acquire**, and
``timeSinceLastAttack`` resets on three separate events: the minion
``IsAttacking``, it has **no target at all**, or ``FoundNewTarget`` **acquires a
new one** (the tail of that method sets ``timeSinceLastAttack = 0f`` alongside
``SetTargetUnit``). Miss the third and the timer never clears for a minion that
keeps switching targets, so every minion in the sim permanently believes it has
failed to attack for 4 seconds -- which is what happened here before it was
fixed: `time_since_attack` read 10,017 ms after ten seconds of a fight in which
minions were visibly killing each other.

One server omission remains
---------------------------
``attackers`` in the target sort is absent, because
   ``CountUnitsAttackingUnit`` is commented out in the server itself ("First
   Wave Behaviour is unfinished"). That is parity, not a simplification.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .state import Kind, MoveOrder
from .targeting import ClassifyUnit, minion_acquire

__all__ = [
    "ACTION_TIMER_MS", "GIVE_UP_MS", "IGNORE_MS", "WAYPOINT_MARGIN",
    "MinionAIOut", "LaneWaypointOut", "step_minion_ai", "advance_lane_waypoints",
]

#: ``minionActionTimer >= 250.0f`` -- the regular sweep of the priority list.
ACTION_TIMER_MS = 250.0
#: ``timeSinceLastAttack >= 4000f`` -- give up on an unreachable target.
GIVE_UP_MS = 4000.0
#: ``Ignore(unit, time = 500)``.
IGNORE_MS = 500.0
#: ``float margin = 25f`` in ``WaypointReached``.
WAYPOINT_MARGIN = 25.0


class MinionAIOut(NamedTuple):
    target: Any
    target_priority: Any
    move_order: Any
    ai_timer: Any
    ai_local_time: Any
    time_since_attack: Any
    ignore_until: Any
    had_target: Any
    reevaluated: Any        # True where the controller ran this tick
    #: True where `FoundNewTarget(true)` (the call-for-help scan) itself pulled
    #: a unit off a STILL-VALID incumbent this tick -- i.e. server `cfh=1` in
    #: `LANERL_AGGRO_TRACE`'s `MRT` line, not "any switch that happened to
    #: involve a boosted priority". Diagnostic only: nothing in `step.py`
    #: reads this field, and adding it changes no behaviour (`MinionAIOut` is
    #: consumed by attribute, so a new trailing field is inert to every
    #: existing caller). See `docs/TARGET_ACQUISITION_DIFF.md` and
    #: `docs/CALL_FOR_HELP_SWITCH_RATE.md` for why `cfh=1` and "toprio in the
    #: call-for-help range" are NOT the same set -- the ordinary unrestricted
    #: scan can also land on a boosted-priority candidate.
    cfh_switch: Any


class LaneWaypointOut(NamedTuple):
    """Result of the lane-only tail of ``ReevaluateBehavior``.

    ``reset_path`` means the server's current movement destination differed
    from the newly selected immutable lane waypoint, so the caller must issue
    ``SetWaypoints([Position, destination])``. ``stop`` is the exhausted
    ``PathingWaypoints`` branch.
    """

    key: Any
    destination: Any              # (N, 2), gathered with a safe key
    reset_path: Any               # (N,) bool
    stop: Any                     # (N,) bool


def advance_lane_waypoints(
    *,
    kind: jax.Array,
    alive: jax.Array,
    x: jax.Array,
    y: jax.Array,
    collision_x: jax.Array,
    collision_y: jax.Array,
    collision_present: jax.Array,
    spawn_seq: jax.Array,
    collision_radius: jax.Array,
    acquisition_range: jax.Array,
    lane_waypoints: jax.Array,
    lane_waypoint_key: jax.Array,
    waypoints: jax.Array,
    n_waypoints: jax.Array,
    reevaluated: jax.Array,
    has_target: jax.Array,
) -> LaneWaypointOut:
    """Port ``LaneMinionAI.WaypointReached`` and its enclosing while-loop.

    ``collision_x/y/present`` are the frozen CollisionHandler nodes rebuilt
    before movement.  ``EnumerateUnitsInRange`` queries those nodes with this
    minion's *live* post-movement centre, then the LINQ sort and the expanding
    collision-cluster geometry use live positions. This split is observable in
    the server and is why both coordinate pairs are explicit inputs.

    The function only acts on ``ReevaluateBehavior`` calls that reached the
    no-target lane-walk tail. A minion with a live/new target never evaluates
    a lane waypoint, exactly as the early returns in the C# method require.
    """
    n = x.shape[0]
    width = lane_waypoints.shape[1]
    idx = jnp.arange(n)
    is_minion = (kind == Kind.LANE_MINION) & alive
    active = is_minion & reevaluated & ~has_target

    # `EnumerateUnitsInRange` asks CollisionHandler's (pre-move) tree whether
    # its node circle intersects a query circle of AcquisitionRange.  LINQ's
    # `OfType<LaneMinion>()` occurs before OrderBy, so champions/turrets never
    # participate in, or terminate, the cluster walk.
    sx = collision_x[None, :] - x[:, None]
    sy = collision_y[None, :] - y[:, None]
    membership = collision_present[None, :] & (
        sx * sx + sy * sy
        < (acquisition_range[:, None] + collision_radius[None, :]) ** 2)
    candidates = membership & is_minion[None, :]

    # OrderBy is stable. CollisionHandler's flat root-list has creation order,
    # so `spawn_seq` supplies the tie break for equal float sort keys.
    live_d2 = (x[None, :] - x[:, None]) ** 2 + (y[None, :] - y[:, None]) ** 2
    score = live_d2 - collision_radius[None, :]
    score = jnp.where(candidates, score, jnp.inf)
    order = jnp.lexsort(
        (jnp.broadcast_to(spawn_seq[None, :], (n, n)), score), axis=-1)

    def ordered(a, rank):
        return jnp.take_along_axis(
            jnp.broadcast_to(a[None, :], (n, n)), order[:, rank, None],
            axis=1)[:, 0]

    # The C# loop turns a chain of overlapping minions into one progressively
    # shifted circle, stopping at the FIRST sorted non-collider. A scan keeps
    # its fixed JIT shape while `open_cluster` reproduces that `break`.
    def cluster_body(carry, rank):
        cx, cy, radius, open_cluster = carry
        j = ordered(idx, rank)
        valid = jnp.take_along_axis(
            candidates, order[:, rank, None], axis=1)[:, 0]
        ox, oy = ordered(x, rank), ordered(y, rank)
        oradius = ordered(collision_radius, rank)
        other = j != idx
        test = open_cluster & valid & other
        dx, dy = ox - cx, oy - cy
        dist = jnp.sqrt(dx * dx + dy * dy)
        collides = dist <= radius + oradius
        safe = jnp.where(dist > 0, dist, jnp.ones_like(dist))
        grow = test & collides
        cx = jnp.where(grow, cx + dx / safe * oradius, cx)
        cy = jnp.where(grow, cy + dy / safe * oradius, cy)
        radius = jnp.where(grow, radius + oradius, radius)
        # Self is explicitly skipped by C# and cannot break the loop. Invalid
        # entries are not in the IEnumerable and likewise cannot break it.
        open_cluster = open_cluster & (~test | collides)
        return (cx, cy, radius, open_cluster), None

    (center_x, center_y, final_radius, _), _ = jax.lax.scan(
        cluster_body, (x, y, collision_radius, jnp.ones((n,), dtype=bool)),
        jnp.arange(n))

    def waypoint_body(key, _):
        safe_key = jnp.clip(key.astype(jnp.int32), 0, width - 1)
        wx = jnp.take_along_axis(lane_waypoints[..., 0], safe_key[:, None], 1)[:, 0]
        wy = jnp.take_along_axis(lane_waypoints[..., 1], safe_key[:, None], 1)[:, 0]
        in_path = key.astype(jnp.int32) < width
        reached = active & in_path & (
            (wx - center_x) ** 2 + (wy - center_y) ** 2
            <= (final_radius + WAYPOINT_MARGIN) ** 2)
        return jnp.where(reached, key + jnp.int8(1), key), None

    key, _ = jax.lax.scan(waypoint_body, lane_waypoint_key, None, length=width)
    safe_key = jnp.clip(key.astype(jnp.int32), 0, width - 1)
    destination = jnp.take_along_axis(
        lane_waypoints, safe_key[:, None, None], axis=1)[:, 0, :]
    walking = active & (key.astype(jnp.int32) < width)
    stop = active & ~walking
    last_idx = jnp.clip(n_waypoints.astype(jnp.int32) - 1, 0, waypoints.shape[1] - 1)
    current_destination = jnp.take_along_axis(
        waypoints, last_idx[:, None, None], axis=1)[:, 0, :]
    reset_path = walking & jnp.any(current_destination != destination, axis=-1)
    return LaneWaypointOut(key=key.astype(lane_waypoint_key.dtype),
                           destination=destination, reset_path=reset_path,
                           stop=stop)


def step_minion_ai(
    *,
    kind: jax.Array,
    alive: jax.Array,
    x: jax.Array,
    y: jax.Array,
    team: jax.Array,
    targetable: jax.Array,
    visible: jax.Array,
    is_attacking: jax.Array,
    acquisition_range: jax.Array,
    base_prio: jax.Array,          # (N,) ClassifyTarget(u) with no victim
    help_priority: jax.Array,      # (N, N) call-for-help overrides, 14 = none
    target: jax.Array,
    target_priority: jax.Array,
    ai_timer: jax.Array,
    ai_local_time: jax.Array,
    time_since_attack: jax.Array,
    ignore_until: jax.Array,
    had_target: jax.Array,
    move_order: jax.Array,
    spawn_seq: jax.Array | None = None,
    delta_ms: float = 1000.0 / 60.0,
) -> MinionAIOut:
    """One tick of ``LaneMinionAI.OnUpdate`` for every unit at once.

    Units that are not live lane minions are left untouched; the caller does not
    have to pre-filter.
    """
    n = x.shape[0]
    dt = jnp.asarray(delta_ms, ai_local_time.dtype)
    me = (kind == Kind.LANE_MINION) & alive

    local_time = jnp.where(me, ai_local_time + dt, ai_local_time)

    has_target = target >= 0
    tsa = jnp.where(me,
                    jnp.where(is_attacking | ~has_target,
                              jnp.zeros_like(time_since_attack),
                              time_since_attack + dt),
                    time_since_attack)
    timer = jnp.where(me, ai_timer + dt, ai_timer)

    ignored = local_time[:, None] < ignore_until

    # priority matrix: call-for-help value where registered, ClassifyTarget else
    prio = jnp.where(help_priority < jnp.int8(ClassifyUnit.DEFAULT),
                     help_priority,
                     jnp.broadcast_to(base_prio[None, :], (n, n)))

    # `TargetJustDied()`: it also REFRESHES targetIsStillValid as a side effect,
    # which is why validity is computed here and reused below.
    cur = jnp.clip(target, 0, n - 1)
    cur_ok = (
        has_target & alive[cur] & (team[cur] != team) & targetable[cur]
        & visible[cur]
        & (((x[cur] - x) ** 2 + (y[cur] - y) ** 2)
           <= acquisition_range * acquisition_range)
        & ~jnp.take_along_axis(ignored, cur[:, None], axis=1)[:, 0]
    )
    just_died = had_target & ~cur_ok

    # ``FoundNewTarget(true)`` -- the call-for-help scan, run as part of the
    # TRIGGER condition and therefore BEFORE ReevaluateBehavior. It is the only
    # path that can re-target a minion whose current target is still valid, and
    # it too needs a strictly better priority (the incumbent is seeded with
    # distanceSquared = -1). Restricted to units that actually registered a call.
    cfh_mask = help_priority < jnp.int8(ClassifyUnit.DEFAULT)
    cfh_any = jnp.any(cfh_mask, axis=-1)
    cfh_pick = minion_acquire(
        x, y, team, alive, targetable, visible,
        jnp.where(cfh_mask, help_priority, jnp.int8(ClassifyUnit.DEFAULT)),
        acquisition_range,
        jnp.where(cur_ok, target, jnp.int8(-1)), target_priority,
        ignored | ~cfh_mask,
        # The incumbent's protection comes from IsValidTarget, which does NOT
        # consult the call-for-help map -- see minion_acquire's docstring.
        incumbent_valid=cur_ok,
        spawn_seq=spawn_seq,
    )
    # `FoundNewTarget` refuses outright while the minion is already on a
    # turret::
    #
    #     if (targetIsStillValid && LaneMinion.TargetUnit is BaseTurret)
    #         return false;
    #
    # Structures are not in the published priority list at all, so nothing
    # outranks a turret a minion has committed to -- it leaves only by the
    # normal exits (the turret dies, it falls out of range or vision, or the
    # 4 s failed-to-attack ignore fires). Without this a call for help drags a
    # sieging wave off the turret it is hitting, which is precisely when a
    # wave is most crowded and calls are loudest.
    on_turret = cur_ok & (kind[cur] == Kind.TURRET)

    # The server spells the trigger as a short-circuiting ``A || B || C``:
    # ``TargetJustDied() || FoundNewTarget(true) || timer >= 250``.  When the
    # incumbent died, ``FoundNewTarget(true)`` is therefore not called at all;
    # ReevaluateBehavior performs the unrestricted scan below instead.  This
    # matters when a call-for-help candidate and an unrelated, better ordinary
    # candidate are both present on the death tick.
    cfh_switch = (me & ~just_died & cfh_any & (cfh_pick >= 0) & (cfh_pick != target)
                  & ~on_turret)
    target = jnp.where(cfh_switch, cfh_pick, target)
    target_priority = jnp.where(
        cfh_switch,
        jnp.take_along_axis(help_priority, jnp.clip(cfh_pick, 0, n - 1)[:, None],
                            axis=1)[:, 0],
        target_priority)
    # the switch also refreshes validity for the block below
    cur = jnp.clip(target, 0, n - 1)
    cur_ok = cur_ok | cfh_switch

    run = me & (just_died | cfh_switch | (timer >= ACTION_TIMER_MS))

    # ---- ReevaluateBehavior -------------------------------------------------
    # give up on a target we have not managed to hit for 4 s, and ignore it
    give_up = run & cur_ok & (tsa >= GIVE_UP_MS)
    new_ignore = ignore_until.at[jnp.arange(n), cur].set(
        jnp.where(give_up, local_time + IGNORE_MS,
                  ignore_until[jnp.arange(n), cur]))
    valid_after = cur_ok & ~give_up

    keep = run & valid_after            # "return OrderType.AttackTo"

    picked = minion_acquire(
        x, y, team, alive, targetable, visible, prio, acquisition_range,
        jnp.where(valid_after, target, jnp.int8(-1)),
        target_priority,
        ignored | (jnp.arange(n)[None, :] == cur[:, None]) & give_up[:, None],
        spawn_seq=spawn_seq,
    )
    took_new = run & ~keep & (picked >= 0)

    new_target = jnp.where(keep, target,
                           jnp.where(took_new, picked,
                                     jnp.where(run, jnp.int8(-1), target)))
    safe_new = jnp.clip(new_target, 0, n - 1)
    new_prio = jnp.where(
        took_new, jnp.take_along_axis(prio, safe_new[:, None], axis=1)[:, 0],
        jnp.where(keep, target_priority,
                  jnp.where(run, jnp.int8(ClassifyUnit.DEFAULT), target_priority)))

    attacking_order = keep | took_new
    new_order = jnp.where(
        run,
        jnp.where(attacking_order, jnp.int8(MoveOrder.ATTACK_TO),
                  jnp.int8(MoveOrder.MOVE_TO)),   # lane walk; Stop only at path end
        move_order)

    # `FoundNewTarget` resets the clock when it acquires -- see the docstring.
    acquired = took_new | cfh_switch
    tsa = jnp.where(acquired, jnp.zeros_like(tsa), tsa)

    return MinionAIOut(
        target=new_target.astype(target.dtype),
        target_priority=new_prio.astype(target_priority.dtype),
        move_order=new_order.astype(move_order.dtype),
        ai_timer=jnp.where(run, jnp.zeros_like(timer), timer),
        ai_local_time=local_time,
        time_since_attack=tsa,
        ignore_until=new_ignore,
        had_target=jnp.where(me, new_target >= 0, had_target),
        reevaluated=run,
        cfh_switch=cfh_switch,
    )
