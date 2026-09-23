"""Target acquisition: three different rules, ported exactly.

The server does not have *a* targeting function.  It has three, and they
disagree with each other in ways that matter:

``ObjAIBase.UpdateTarget`` (champions on attack-move)
    Nearest enemy inside ``AcquisitionRange``.  **No priority at all** -- a
    minion one unit closer outranks the enemy champion.
``AIScripts/TurretAI.CheckForTargets``
    Priority first, distance never.  Plus an override: a turret already holding
    a non-champion target switches to an enemy *champion* who is attacking an
    allied champion in range.
``AIScripts/LaneMinionAI.FoundNewTarget``
    Lexicographic on ``(attackers, priority, distance^2)``, with the incumbent
    made unbeatable on distance so only a **strictly better priority** can
    displace it.

Ties, and why creation order is a parity surface
------------------------------------------------
All three iterate and keep the first strictly-better candidate, so ties go to
the first candidate in the server collection.  On Map1 the collision quadtree
degenerates to its root list (see :mod:`sim.collision`), whose traversal is
object-add order.  Array slots are *not* that order: turrets are created before
champions, and dead minion slots are recycled.  Callers therefore pass
``LaneState.spawn_seq`` as the explicit tie key.  Slot order remains only as a
standalone-test fallback.

Lexicographic without a packed key
----------------------------------
The minion rule is a two-level sort.  Packing ``priority`` and ``distance^2``
into one float loses the tie-break: priorities run 1..14 and squared distances
reach ~1.2e6, so a packed key needs more mantissa than float32 has, and a
silently-lost tie-break is a targeting bug that only shows up as a strange
last-hit. So it is done as two masked reductions instead -- exact, and the cost
is one extra pass over a 66-element axis.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .state import Kind, Team

__all__ = [
    "ClassifyUnit",
    "MinionType",
    "base_priority",
    "help_priority_for",
    "call_for_help_map",
    "nearest_enemy",
    "turret_acquire",
    "minion_acquire",
]


class ClassifyUnit:
    """``GameServerCore.Enums.ClassifyUnit``. **Lower is higher priority.**"""

    CHAMPION_ATTACKING_CHAMPION = 1
    MINION_ATTACKING_CHAMPION = 2
    MINION_ATTACKING_MINION = 3
    TURRET_ATTACKING_MINION = 4
    CHAMPION_ATTACKING_MINION = 5
    MINION = 6
    SUPER_OR_CANNON_MINION = 7
    CASTER_MINION = 8
    MELEE_MINION = 9
    TURRET = 10
    CHAMPION = 11
    INHIBITOR = 12
    NEXUS = 13
    DEFAULT = 14


class MinionType:
    """``MinionSpawnType``, in the order ``MinionWaveTypes`` uses."""

    MELEE = 0
    CASTER = 1
    CANNON = 2
    SUPER = 3


def base_priority(kind: jax.Array, minion_type: jax.Array) -> jax.Array:
    """``ObjAIBase.ClassifyTarget(target)`` with no victim.

    Note the ordering the server chose: a **melee** minion (9) is *lower*
    priority than a caster (8), which is lower than a cannon/super (7). So a
    minion wave prefers to hit cannons, then casters, then melee -- and all of
    them before a turret (10) or a champion (11).
    """
    p = jnp.full(kind.shape, ClassifyUnit.DEFAULT, dtype=jnp.int8)
    minion_p = jnp.select(
        [minion_type == MinionType.MELEE,
         minion_type == MinionType.CASTER,
         (minion_type == MinionType.CANNON) | (minion_type == MinionType.SUPER)],
        [jnp.int8(ClassifyUnit.MELEE_MINION),
         jnp.int8(ClassifyUnit.CASTER_MINION),
         jnp.int8(ClassifyUnit.SUPER_OR_CANNON_MINION)],
        default=jnp.int8(ClassifyUnit.MINION),
    )
    p = jnp.where(kind == Kind.LANE_MINION, minion_p, p)
    p = jnp.where(kind == Kind.TURRET, jnp.int8(ClassifyUnit.TURRET), p)
    p = jnp.where(kind == Kind.CHAMPION, jnp.int8(ClassifyUnit.CHAMPION), p)
    return p


def help_priority_for(attacker_kind: jax.Array, victim_kind: jax.Array,
                      attacker_minion_type: jax.Array) -> jax.Array:
    """``ClassifyTarget(target, victim)`` -- the call-for-help table.

    ``attacker_kind`` and ``victim_kind`` broadcast against each other, so
    passing ``(N,1)`` and ``(1,N)`` yields the full ``(N,N)`` matrix the state
    carries as ``help_priority``; ``attacker_minion_type`` broadcasts with
    ``attacker_kind``.

    `ENT-06`: a pair with no "ally in distress" case is NOT "no call".
    ``ObjAIBase.ClassifyTarget`` (`ObjAIBase.cs:388-448`) tries the
    ``victium != null`` switch first and, when no case matches, FALLS THROUGH
    to the attacker's own base class -- the same value as
    :func:`base_priority`. ``LaneMinionAI.OnCallForHelp`` stores whatever that
    returns (`Math.Min(existing, ClassifyTarget(attacker, victim))`), so a
    turret shooting a champion registers at ``TURRET`` (10), a champion
    hitting a turret at ``CHAMPION`` (11), and a minion hitting a turret at
    its own minion class (7/8/9). This used to return ``DEFAULT`` for all
    three, i.e. no call at all.
    """
    ak, vk = attacker_kind, victim_kind
    is_minionish = (vk == Kind.LANE_MINION)
    fallback = base_priority(ak, attacker_minion_type)
    return jnp.select(
        [(ak == Kind.CHAMPION) & (vk == Kind.CHAMPION),
         (ak == Kind.CHAMPION) & is_minionish,
         (ak == Kind.LANE_MINION) & (vk == Kind.CHAMPION),
         (ak == Kind.LANE_MINION) & is_minionish,
         (ak == Kind.TURRET) & is_minionish],
        [jnp.int8(ClassifyUnit.CHAMPION_ATTACKING_CHAMPION),
         jnp.int8(ClassifyUnit.CHAMPION_ATTACKING_MINION),
         jnp.int8(ClassifyUnit.MINION_ATTACKING_CHAMPION),
         jnp.int8(ClassifyUnit.MINION_ATTACKING_MINION),
         jnp.int8(ClassifyUnit.TURRET_ATTACKING_MINION)],
        default=jnp.broadcast_to(
            fallback, jnp.broadcast_shapes(jnp.shape(ak), jnp.shape(vk))),
    )


def call_for_help_map(*, damage_ij, x, y, alive, kind, team,
                      acquisition_range, minion_type) -> jax.Array:
    """``(N, N)`` ``[listener, attacker]`` call-for-help priorities.

    This is the signal that was MISSING. ``help_priority`` was declared, was
    initialised to all-DEFAULT, was read by ``step_minion_ai`` -- and was never
    written anywhere in the simulation loop, only in unit tests. So the
    machinery was verified in isolation and fed nothing in production.

    It is not a cosmetic gap. Per ``LaneMinionAI``, a call for help is the ONLY
    channel besides the target dying or leaving range that can pull a minion
    off a target it already holds: ``ReevaluateBehavior`` short-circuits to
    ``AttackTo`` before it ever re-scans. Without it, a minion that once
    acquired the champion attacks the champion until one of them dies.

    Measured against the server, with a champion standing still in lane:
    acquisition rates were close (sim 61 per 600 s, server 47), but the server
    RELEASED -- 28 of 28 observed departures from the champion went straight
    back to a minion, median hold 2.5 s -- while the sim released never. Lock-ons
    piled up instead of recycling: sim mean 4.14 and max 12 simultaneous
    attackers against the server's bounded exposure, turning a 1.3x acquisition
    gap into a 4x death gap.

    ``ObjAIBase.TakeDamage`` broadcasts the call::

        u != this && !u.IsDead && u.Team == Team
          && u.AIScript.AIScriptMetaData.HandlesCallsForHelp
          && DistanceSquared(u.Position, Position)          <= acqRange^2
          && DistanceSquared(u.Position, attacker.Position) <= acqRange^2

    Three things there are easy to get wrong and are reproduced exactly:
    ``acqRange`` is the **victim's** acquisition range, not the listener's; the
    listener must be in range of BOTH the victim and the attacker; and only
    scripts with ``HandlesCallsForHelp`` listen, which among these units is
    lane minions alone.

    ``LaneMinionAI.OnCallForHelp`` then keeps the BEST priority seen::

        priority = Math.Min(existing, ClassifyTarget(attacker, victim))

    The map is one-shot: ``FoundNewTarget`` sets ``callsForHelpMayBeCleared``
    and the map is wiped at the end of that same update, so a call raised by
    this tick's damage is consumed by the next AI pass and discarded. Hence
    this returns a fresh map each tick rather than accumulating into one.
    """
    n = x.shape[0]
    d2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    hit = damage_ij > 0                                   # [attacker, victim]
    r2 = acquisition_range * acquisition_range            # per VICTIM

    listens = alive & (kind == Kind.LANE_MINION)          # HandlesCallsForHelp
    # [listener, victim]: ally, not self, and inside the victim's range
    uv = (listens[:, None] & (team[:, None] == team[None, :])
          & ~jnp.eye(n, dtype=bool) & (d2 <= r2[None, :]))
    # [listener, attacker, victim]: also inside that same range of the attacker
    ua_v = d2[:, :, None] <= r2[None, None, :]

    valid = hit[None, :, :] & uv[:, None, :] & ua_v
    cls = help_priority_for(kind[:, None], kind[None, :],
                            minion_type[:, None])          # [attacker, victim]
    prio = jnp.where(valid, cls[None, :, :],
                     jnp.int8(ClassifyUnit.DEFAULT))
    return jnp.min(prio, axis=2).astype(jnp.int8)          # min over victims


def _first_argmin(value: jax.Array, valid: jax.Array, axis: int = -1,
                  tie_key: jax.Array | None = None) -> jax.Array:
    """Index of the minimum, ties going to the lowest ``tie_key``.

    ``tie_key`` is a one-dimensional candidate creation-rank vector.  When it
    is omitted, array index is used for compact standalone tests.  Production
    passes ``spawn_seq`` so recycled slots cannot silently change target ties.
    """
    big = jnp.asarray(jnp.inf, value.dtype) if jnp.issubdtype(value.dtype, jnp.floating) \
        else jnp.asarray(jnp.iinfo(value.dtype).max, value.dtype)
    masked = jnp.where(valid, value, big)
    best = jnp.min(masked, axis=axis, keepdims=True)
    tied = valid & (value == best)
    if tie_key is None:
        keys = jnp.arange(value.shape[axis], dtype=jnp.int32)
    else:
        keys = tie_key.astype(jnp.int32)
    shape = [1] * value.ndim
    shape[axis] = value.shape[axis]
    keys = jnp.broadcast_to(keys.reshape(shape), value.shape)
    key_max = jnp.iinfo(jnp.int32).max
    chosen_key = jnp.min(jnp.where(tied, keys, key_max), axis=axis,
                         keepdims=True)
    chosen = tied & (keys == chosen_key)
    idx = jnp.argmax(chosen.astype(jnp.int8), axis=axis)
    any_valid = jnp.any(valid, axis=axis)
    return jnp.where(any_valid, idx, -1).astype(jnp.int8)


def _pairwise_dist2(x: jax.Array, y: jax.Array) -> jax.Array:
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    return dx * dx + dy * dy


def nearest_enemy(x: jax.Array, y: jax.Array, team: jax.Array, alive: jax.Array,
                  targetable: jax.Array, acquisition_range: jax.Array,
                  spawn_seq: jax.Array | None = None) -> jax.Array:
    """``ObjAIBase.UpdateTarget``'s attack-move scan: closest enemy, no priority.

    The server's loop rejects on ``u.IsDead``, ``u.Team == Team``,
    ``DistanceSquared > range*range`` and ``!Targetable``, then keeps the
    strictly closest. ``range`` is ``Stats.AcquisitionRange.Total`` -- the
    comment in the C# notes that using ``max(Range, AcquisitionRange)`` was the
    previous, incorrect version.

    Returns ``(N,)`` unit indices, -1 where nothing qualifies.
    """
    d2 = _pairwise_dist2(x, y)
    r = acquisition_range[:, None]
    valid = (
        alive[None, :]
        & targetable[None, :]
        & (team[None, :] != team[:, None])
        & (d2 <= r * r)
        & ~jnp.eye(x.shape[0], dtype=bool)
    )
    return _first_argmin(d2, valid, tie_key=spawn_seq)


def turret_acquire(x: jax.Array, y: jax.Array, team: jax.Array, alive: jax.Array,
                   targetable: jax.Array, kind: jax.Array, minion_type: jax.Array,
                   turret_range: jax.Array, current_target: jax.Array,
                   target_of: jax.Array, attack_range: jax.Array,
                   spawn_seq: jax.Array | None = None, *,
                   collision_radius: jax.Array) -> jax.Array:
    """``TurretAI.CheckForTargets``.

    Two regimes, and the server really does branch on whether it already holds
    a target:

    * **no current target** -> lowest ``ClassifyTarget`` priority wins, distance
      never consulted, ties to the first-added unit.
    * **holding a target** -> if that target is a champion, keep it. Otherwise
      look *only* for an enemy champion who is attacking an allied champion,
      where both the attacker's target is in the attacker's own attack range and
      that victim is inside turret range. First such champion wins outright
      (the C# ``break``s), no priority involved.

    Acquisition is WIDER than retention, and that is not a rounding detail
    -- it is `TURRET-001`
    ------------------------------------------------------------------------
    The candidate list comes from ``GetUnitsInRange(Position, Range.Total,
    true)``, which is a **circle-vs-circle** quadtree query, not a
    centre-to-centre distance test.  ``ApiFunctionManager.cs:601-604`` builds
    a ``Circle(pos, range)``; ``CollisionHandler.GetNearestObjects`` hands it
    to ``QuadTree.GetNodesInside``; and every stored node is itself a
    ``Circle(obj.Position, Math.Max(0.5f, obj.CollisionRadius))``
    (``CollisionHandler.cs:69-72``).  The leaf test is
    ``QuadTree.cs:61-64``::

        public bool IntersectsWith(Circle circle)
            => Vector2.DistanceSquared(Position, circle.Position)
               < (Radius + circle.Radius) * (Radius + circle.Radius);

    So a unit is a **candidate** at ``dist < Range + its own CollisionRadius``
    -- 790 for a 40-radius lane minion against a 750-range turret -- and the
    inequality is strict.  Retention four lines later in the same
    ``TurretAI.OnUpdate`` is a flat centre-to-centre ``Range`` (see
    ``step.py``'s turret block), and selection is priority-only with distance
    never consulted (``:43-62``).

    The three together are a trap, not a tolerance: a best-priority minion
    standing in that 40-unit annulus is **picked** by ``CheckForTargets`` and
    **dropped** by the retention test in the same update, every tick, while
    better-placed lower-priority minions sit unattacked.  One blue turret was
    measured idle at cooldown 0 for 664 consecutive ticks (11 s) with melee
    minions at 196 units, because a caster sat at 764.  Using one radius for
    both -- which this function did until `TURRET-001` -- makes that
    unreachable, so the sim acquires and fires where the server stands still.

    ``collision_radius`` is therefore required rather than defaulted: a
    silently-zero radius is exactly the old behaviour, and it is wrong in a
    direction no test would notice.
    """
    n = x.shape[0]
    d2 = _pairwise_dist2(x, y)
    rng2 = (turret_range * turret_range)[:, None]
    # `Math.Max(0.5f, obj.CollisionRadius)` -- `CollisionHandler.GetBounds`.
    cand_r = turret_range[:, None] + jnp.maximum(collision_radius, 0.5)[None, :]
    in_range = (
        alive[None, :] & targetable[None, :]
        & (team[None, :] != team[:, None]) & (d2 < cand_r * cand_r)
        & ~jnp.eye(n, dtype=bool)
    )

    # regime 1: priority only
    prio = jnp.broadcast_to(base_priority(kind, minion_type)[None, :], (n, n))
    by_priority = _first_argmin(prio, in_range, tie_key=spawn_seq)

    # regime 2: an enemy champion diving an allied champion
    tgt = target_of                                   # (N,) each unit's target
    safe_tgt = jnp.clip(tgt, 0, n - 1)
    victim_kind = kind[safe_tgt]
    victim_x, victim_y = x[safe_tgt], y[safe_tgt]
    atk_to_victim2 = (x - victim_x) ** 2 + (y - victim_y) ** 2
    attacker_ok = (
        (kind == Kind.CHAMPION) & (tgt >= 0) & (victim_kind == Kind.CHAMPION)
        & (atk_to_victim2 <= attack_range * attack_range)
    )
    victim_in_turret_range = (
        (x[None, safe_tgt] - x[:, None]) ** 2
        + (y[None, safe_tgt] - y[:, None]) ** 2
    ) <= rng2
    diving = in_range & attacker_ok[None, :] & victim_in_turret_range
    # "no priority required ... break" -> the first qualifying index
    by_dive = _first_argmin(jnp.zeros((n, n), dtype=jnp.int8), diving,
                            tie_key=spawn_seq)

    holding = current_target >= 0
    holding_champ = holding & (kind[jnp.clip(current_target, 0, n - 1)] == Kind.CHAMPION)
    out = jnp.where(holding, jnp.where(by_dive >= 0, by_dive, current_target),
                    by_priority)
    return jnp.where(holding_champ, current_target, out).astype(jnp.int8)


def minion_acquire(x: jax.Array, y: jax.Array, team: jax.Array, alive: jax.Array,
                   targetable: jax.Array, visible: jax.Array,
                   priority: jax.Array, acquisition_range: jax.Array,
                   current_target: jax.Array, current_priority: jax.Array,
                   ignored: jax.Array,
                   incumbent_valid: jax.Array | None = None,
                   spawn_seq: jax.Array | None = None) -> jax.Array:
    """``LaneMinionAI.FoundNewTarget``, lexicographic on ``(priority, dist^2)``.

    ``priority`` is ``(N, N)``: row *i* is what unit *i* thinks each candidate is
    worth, which is ``ClassifyTarget(u)`` normally and the call-for-help value
    where one is registered. ``ignored`` is ``(N, N)`` bool from
    ``temporaryIgnored``.

    The incumbent's protection is the subtle part. The server seeds
    ``nextTargetDistanceSquared = -1`` for a still-valid current target, making
    it unbeatable on the distance tie-break, so only a **strictly better
    priority** displaces it -- the quoted rule is *"The minion cannot acquire a
    new target that has the same priority as their current target"*. Without
    that, two equal-priority minions swap the moment one steps closer, and the
    C# comment records swaps as fast as 16 ms.

    ``incumbent_valid`` exists because the two things are **independent** in the
    server and conflating them is a real bug I shipped once here. In
    ``FoundNewTarget(true)`` -- the call-for-help scan -- the *candidate set* is
    restricted to ``unitsAttackingAllies.Keys``, but the incumbent's protection
    comes from ``IsValidTarget(TargetUnit)``, which does not consult that map at
    all. Deriving validity from the (restricted) candidate mask makes the
    incumbent look invalid, drops its priority floor to DEFAULT, and lets a
    *worse* priority steal the target. Pass it explicitly; ``None`` means "derive
    it from ``ignored``", which is right only for the unrestricted scan.

    Not modelled: the ``attackers`` term. ``CountUnitsAttackingUnit`` is
    commented out in the server ("First Wave Behaviour is unfinished"), so it is
    constant 0 there and absent here. Reproducing the server, not the wiki.
    """
    n = x.shape[0]
    d2 = _pairwise_dist2(x, y)
    r = acquisition_range[:, None]
    valid = (
        alive[None, :] & targetable[None, :] & visible[None, :]
        & (team[None, :] != team[:, None])
        & (d2 <= r * r) & ~ignored
        & ~jnp.eye(n, dtype=bool)
    )

    # incumbent: still valid, and unbeatable on distance
    has_cur = current_target >= 0
    cur = jnp.clip(current_target, 0, n - 1)
    if incumbent_valid is None:
        cur_valid = has_cur & jnp.take_along_axis(valid, cur[:, None], axis=1)[:, 0]
    else:
        cur_valid = has_cur & incumbent_valid
    best_prio_floor = jnp.where(cur_valid, current_priority,
                                jnp.int8(ClassifyUnit.DEFAULT))

    # level 1: the best priority available, but strictly better than the incumbent
    cand = valid & (priority < best_prio_floor[:, None])
    # a minion with no valid incumbent may take anything up to DEFAULT
    cand = jnp.where(cur_valid[:, None], cand,
                     valid & (priority <= jnp.int8(ClassifyUnit.DEFAULT)))
    best_p = jnp.min(jnp.where(cand, priority, jnp.int8(ClassifyUnit.DEFAULT)),
                     axis=-1, keepdims=True)
    # level 2: nearest among those, ties to the lowest index
    at_best = cand & (priority == best_p)
    pick = _first_argmin(d2, at_best, tie_key=spawn_seq)

    keep = cur_valid & (pick < 0)
    return jnp.where(keep, current_target, pick).astype(jnp.int8)
