"""Tier 1, finally built: inject a server snapshot into a :class:`LaneState`.

``lanerl_jax.parity.sim_vs_server`` explains why nobody had done this yet: the
state dump (``LanerlStateDump.Describe``) does not carry ``TargetUnit``, the
auto-attack clock, waypoint *positions*, the minion AI's own timers, its
ignore list, or the wave spawner's internal counters.  Injecting the
observable subset and leaving the rest at ``empty_state`` defaults does not
produce "the server's state, one step on" -- it produces a state that shares
some fields with it and is silently wrong in the rest.

This module does it anyway, on the theory that a **precisely scoped** wrong
answer beats no answer, provided every approximation is named where it is
made and the resulting harness reports its own blind spots rather than
hiding them.  Three kinds of field, by provenance:

* **injected** -- read straight off the dump (position, HP, alive, kind,
  team, level, gold, CS, deaths, move order, champion spell level/cooldown).
* **derived** -- not in the dump verbatim, but recoverable *exactly* from
  what is, with no guessing:

  - ``model`` (the profile row that drives every per-tick stat gather) from
    ``(kind, team, max_hp)``: the four minion max-HP values (290/455/700/1500)
    are unique per type and identical across teams, so max HP alone
    disambiguates melee/caster/cannon/super with no ambiguity. Champions and
    turrets need only ``team``.
  - the wave-spawner scalars (``next_spawn_ms``, ``minion_number``,
    ``cannon_count``) are not per-entity at all -- they are not in the dump
    in any form -- but ``LevelScript.Update`` is a **deterministic, RNG-free**
    function of game time alone (``sim/waves.py``), so replaying it from
    ``t=0`` using the trace's own recorded tick times reproduces them exactly,
    with no independent clock to drift against the server's.
  - a marching minion's waypoints: ``Waypoints.Count`` is in the dump but the
    vertices are not.  A minion with no target walks the same static lane
    corridor every episode (``TOP_LANE_PATH``, forward for blue / reversed for
    red -- see ``sim/init.py``), so a position that lies on that corridor
    pins down the corridor *and* which vertex is next.  This is exact when it
    is on the corridor and refused otherwise (see ``reconstruct_waypoints``).

* **unrecoverable** -- defaulted to the same values :func:`empty_state` uses,
  and named here so a downstream report can say which mechanics they poison
  rather than let a default read as an assertion:

  - ``target`` (all units): minion-target identity is not observable at all
    (see ``parity.diff.UNOBSERVABLE``); champion/turret target identity
    *would* be recoverable from the observation wire / turret trace, but this
    injector does not consume those streams (the recorded fixture is a pure
    no-orders idle trace, so no champion ever has a target and this
    limitation happens not to bite it -- it would for a driven fixture).
    Defaulting to ``-1`` plus ``ai_timer=250`` (forces immediate
    re-evaluation, see below) means every unit re-acquires its target from
    scratch on every injected tick. That is *not* the same experiment as
    "did the server's held target survive this tick" -- it tests "does a
    from-scratch acquisition agree with the server's *actual* target", which
    conflates a wrong pick with the server's hysteresis (a still-valid
    incumbent is protected from a same-priority closer unit; a fresh
    acquisition is not, see ``sim/minion_ai.py``). Any target-selection
    disagreement this harness finds is real evidence of a difference, but a
    *lack* of disagreement does not confirm the hysteresis rule is right.
  - ``aa_cooldown`` / ``aa_windup`` / ``is_attacking`` / ``has_auto_attacked``:
    completely absent from the dump (measured in
    ``parity/tests/test_autoattack.py``: ``cast_spell`` is ``"-"`` on 100% of
    15,162 champion rows in a run with 32 real swings). Defaulted to
    "not attacking, cooldown ready". **This is the single biggest hole**: the
    plan's Tier-1 target for auto-attack is the exact fire tick, and this
    injector cannot pin the fire tick of any swing except the first one after
    two units meet (where "cooldown ready" is close to true) -- every swing in
    a sustained fight is tested against a fabricated clock. See
    ``docs/ONE_STEP_DIFFERENTIAL.md`` for how the harness works around this
    (event-level hit/no-hit agreement, which does not need the clock to be
    right, only the outcome).
  - the minion AI's own bookkeeping: ``target_priority`` (-> 14, "no
    incumbent"), ``ignore_until`` (-> 0, nothing ignored), ``help_priority``
    (-> 14, no call for help pending), ``ai_local_time`` (-> 0),
    ``time_since_attack`` (-> 0). None of these are in the dump. Consequence:
    the 4-second give-up rule and the 500 ms post-give-up ignore window can
    never fire in an injected tick (the clock that drives them is reset every
    time), and a call-for-help re-target can never be reproduced.
  - waypoints for a unit whose ``move_order`` is ``ATTACK_TO`` (chasing a
    target): these are set, on the server, to a straight line at the
    attacker's and target's *current* positions and recomputed every tick
    (confirmed by reading ``sim/step.py``'s own "3b. RefreshWaypoints" -- our
    sim does the same thing), so they would be exactly reconstructable *if*
    we knew the target's identity. We do not (see ``target`` above), so these
    are left empty (``n_waypoints=0``) rather than pointed at a guessed unit,
    and the unit is flagged ``movement_trustworthy=False`` for that tick.
  - waypoints for a minion that is marching (``move_order=MOVE_TO``) but is
    **not** on the known lane corridor within tolerance -- this happens to a
    minion that just gave up a chase (see ``sim/minion_ai.py``'s
    ``ReevaluateBehavior``: the server resumes the corridor from
    ``PathingWaypoints``, a *separate* list our own sim does not model at all
    -- ``sim/step.py``'s ai-driven ``MOVE_TO`` branch does not touch
    ``waypoints``/``n_waypoints``, so a minion that just lost a chase target
    keeps the stale two-point chase line in our own sim, a genuine
    unmodelled-mechanic gap, not just an injection one). Also flagged
    ``movement_trustworthy=False``.

Everything not listed above that the dump does not carry (``mr``,
``attack_speed``, ``skill_points``, ``buffs``, ``can_move``, ``cast_spell``,
``channel_spell``, spell levels/cooldowns beyond what ``ChampionBlock``
supplies) is simply not part of :class:`LaneState` and is excluded from every
comparison via ``sim_vs_server.NOT_MODELLED`` -- that list is unchanged by
this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..sim.init import TOP_LANE_PATH
from ..sim.state import (
    CH_SLICE,
    MI_SLICE,
    TU_SLICE,
    Kind,
    LaneState,
    MoveOrder,
    Team,
    empty_state,
)
from ..sim.waves import FIRST_WAVE_MS, WaveState, step_waves
from .recover_waypoints import point_line_distance
from .trace import Entity, Snapshot, StatQ

__all__ = [
    "KIND_NAME_TO_ID", "SERVER_TEAM_TO_ID",
    "UnitInjectionNote", "InjectionReport",
    "infer_minion_model", "reconstruct_waypoints",
    "replay_wave_states", "inject_snapshot",
]

KIND_NAME_TO_ID = {
    "Champion": Kind.CHAMPION, "LaneMinion": Kind.LANE_MINION,
    "LaneTurret": Kind.TURRET,
}
SERVER_TEAM_TO_ID = {100: Team.BLUE, 200: Team.RED, 300: Team.NEUTRAL}

#: perpendicular distance, in world units, within which a marching minion's
#: position is considered "on the known lane corridor". The corridor's
#: vertices are ~50 units apart and a base-speed minion moves 2 units/tick,
#: so this is loose enough to absorb quantisation (1/16 unit) and a tick or
#: two of numerical slop but tight enough that a genuinely off-corridor
#: minion (e.g. mid-collision-push, or resuming from a chase) is refused
#: rather than silently mis-reconstructed.
CORRIDOR_TOLERANCE = 8.0


@dataclass(slots=True)
class UnitInjectionNote:
    """What happened when injecting one entity, for the report."""

    kind: str
    team: int
    x: float
    y: float
    model_row: Optional[int]
    model_reason: str
    movement_trustworthy: bool
    movement_reason: str
    #: the LaneState slot this entity was placed in, and the source Entity
    #: (pre-tick, i.e. the server's ground truth at the injected tick) --
    #: both needed downstream to recover correspondence without re-deriving
    #: the slot-assignment rule a second time (a second implementation of
    #: "champions 0/1, then minions, then turrets" is exactly the kind of
    #: thing that silently drifts from this one).
    slot: int = -1
    entity: Optional[Entity] = None


@dataclass(slots=True)
class InjectionReport:
    """Everything about the injection of one snapshot, for auditing."""

    t_ms: int
    notes: List[UnitInjectionNote] = field(default_factory=list)
    dropped_unknown_kind: int = 0
    dropped_capacity: Dict[str, int] = field(default_factory=dict)

    @property
    def n_untrustworthy_movement(self) -> int:
        return sum(1 for n in self.notes if not n.movement_trustworthy)

    @property
    def n_units(self) -> int:
        return len(self.notes)


def infer_turret_model(x: float, y: float, team: int) -> Tuple[Optional[int], str]:
    """``(kind=TURRET, team, x, y)`` -> profile row, via nearest known position.

    Unlike a minion, a turret's TIER cannot be inferred from anything in the
    injected snapshot itself -- HP is a poor signal (a damaged inner turret and
    a fresh outer turret can read the same fraction, and the raw HP depends on
    which schedule has fired) and AD/armour are exactly what this is trying to
    recover in the first place, not something to read back out. What IS fixed
    and known in advance is WHERE each of the 24 turrets sits, at the same
    resolution `sim.init.ALL_TURRETS` was measured at -- see that table's
    docstring for how each entry's tier was cross-referenced against
    `LevelScriptObjects.GetTurretType`. So this matches on position instead,
    the same table `init_lane` places turrets from, and refuses (returns
    ``None``) rather than guessing a tier when nothing is close enough to
    trust -- consistent with `infer_minion_model`'s refusal contract.
    """
    from ..sim.init import ALL_TURRETS
    from ..sim.profiles import profile_id

    best_j, best_d2 = None, float("inf")
    for j, (t, tx, ty, _thp, _tier) in enumerate(ALL_TURRETS):
        if t != team:
            continue
        d2 = (tx - x) ** 2 + (ty - y) ** 2
        if d2 < best_d2:
            best_j, best_d2 = j, d2
    if best_j is None:
        return None, "no turret on this team in ALL_TURRETS"
    dist = best_d2 ** 0.5
    if dist > 2.0:
        return None, (
            f"nearest known turret is {dist:.2f} units away -- too far to "
            "trust, refusing rather than guessing a tier")
    tier = ALL_TURRETS[best_j][4]
    return (profile_id(Kind.TURRET, tier, team),
            f"matched ALL_TURRETS[{best_j}] (tier {tier}), {dist:.2f} units off")


def infer_minion_model(max_hp: float, team: int, params: dict,
                       profiles) -> Tuple[Optional[int], str]:
    """``(kind=LANE_MINION, team, max_hp)`` -> profile row, or ``None``.

    Exact match against the profile table's own ``max_hp`` column (built from
    the same patch table the sim runs on), not a hardcoded constant -- so a
    modern-patch swap does not silently break this. Melee/caster/cannon/super
    (290/455/700/1500 on this patch) never collide, so "nearest" and "exact"
    agree; nearest is used only to absorb the dump's 1/1024 quantisation.
    """
    best_row, best_err = None, float("inf")
    for row, (kind, _mtype, prof_team) in enumerate(profiles):
        if kind != Kind.LANE_MINION or prof_team != team:
            continue
        err = abs(float(params["max_hp"][row]) - max_hp)
        if err < best_err:
            best_row, best_err = row, err
    if best_row is None:
        return None, "no minion profile for this team"
    if best_err > 0.5:
        return best_row, (
            f"nearest match off by {best_err:.3f} HP -- suspicious, "
            f"minion types should match max_hp exactly")
    return best_row, "exact max_hp match"


def _project_to_polyline(x: float, y: float, path: np.ndarray
                         ) -> Tuple[int, float, float]:
    """Nearest segment of ``path`` to ``(x, y)`` -> ``(seg_index, dist, t)``.

    ``seg_index`` is the 0-based index of the segment's *start*; the vertex at
    its far end (``seg_index + 1``) is what ``CurrentWaypointKey`` should point
    at when the unit is anywhere on or before that segment.
    """
    best_i, best_d, best_t = 0, float("inf"), 0.0
    for i in range(len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        t = 0.0 if length2 == 0.0 else max(
            0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length2))
        d = point_line_distance((x, y), (ax, ay), (bx, by))
        if d < best_d:
            best_i, best_d, best_t = i, d, t
    return best_i, best_d, best_t


def reconstruct_waypoints(
    x: float, y: float, team: int, tol: float = CORRIDOR_TOLERANCE
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int], str]:
    """Best-effort recovery of a marching minion's waypoint array.

    Returns ``(waypoints (L,2) or None, waypoint_key or None, n_waypoints or
    None, reason)``. ``None`` means "refused" -- the caller must not invent a
    destination, see the module docstring.
    """
    path = np.asarray(TOP_LANE_PATH, np.float32)
    if team == Team.RED:
        path = path[::-1].copy()
    seg, dist, t = _project_to_polyline(x, y, path)
    if dist > tol:
        return None, None, None, (
            f"position is {dist:.1f} units off the known lane corridor "
            f"(tolerance {tol}) -- not walking the spawn path, or resuming "
            "from a lost target (PathingWaypoints is not modelled)")
    key = seg + 1
    if t > 0.999 and key + 1 < len(path):
        key += 1
    key = min(key, len(path) - 1)
    return path, key, len(path), f"on corridor, {dist:.2f} units off, key={key}"


def replay_wave_states(trace: Trace) -> List[WaveState]:
    """``WaveState`` to inject alongside each snapshot, replayed exactly.

    ``LevelScript.Update`` (ported in ``sim/waves.py``) is a deterministic,
    RNG-free function of game time and its own running counters -- nothing
    else -- so replaying it from the episode's true initial condition
    (``FIRST_WAVE_MS``, 0, 0, matching ``empty_state``/``init_lane``) using the
    trace's *own* recorded tick times reproduces the server's wave-spawner
    state exactly, with no independent clock to drift against it.

    Index ``i`` of the returned list is the state to inject **alongside**
    ``trace[i]`` -- i.e. the state ``tick()`` will consume when stepping from
    ``trace[i]`` to ``trace[i + 1]``. See the module docstring's derivation:
    the state paired with a snapshot at time ``T`` is the result of the
    *previous* tick's spawn decision, not one evaluated at ``T`` itself.
    """
    out: List[WaveState] = [WaveState(next_spawn_ms=FIRST_WAVE_MS,
                                      minion_number=0, cannon_count=0)]
    st = out[0]
    for i in range(len(trace) - 1):
        nxt = WaveState(next_spawn_ms=st.next_spawn_ms,
                        minion_number=st.minion_number,
                        cannon_count=st.cannon_count)
        step_waves(nxt, float(trace[i].t_ms))
        out.append(nxt)
        st = nxt
    return out


def inject_snapshot(
    snapshot: Snapshot, wave_state: WaveState, params: dict, profiles,
    dtype=None,
) -> Tuple[LaneState, InjectionReport]:
    """Build a :class:`LaneState` from one server :class:`Snapshot`.

    Slot assignment is ours to choose (champions -> ``CH_SLICE`` by team,
    minions/turrets -> the free slots of ``MI_SLICE``/``TU_SLICE`` in the
    order they appear in the snapshot, which is the dump's *content* sort
    order, not the server's true object-add order). That is fine for what
    this harness measures -- correspondence with the server's *next* snapshot
    is recovered by nearest position, not by slot identity, exactly as
    ``parity.diff`` already does for Tier 2 -- but it does mean acquisition
    **tie-breaks** (equal priority, equal distance) are not guaranteed to
    replicate the server's true add-order tie-break (``D5`` in the rewrite
    plan already names this as a general "silent-divergence source"; this
    injector does not attempt to solve it).
    """
    import jax.numpy as jnp

    dtype = dtype or jnp.float32
    report = InjectionReport(t_ms=snapshot.t_ms)
    s = empty_state(dtype=dtype, seed=0)

    ch_slot = {Team.BLUE: CH_SLICE.start, Team.RED: CH_SLICE.start + 1}
    next_minion = MI_SLICE.start
    next_turret = TU_SLICE.start

    # `np.array(..., copy=True)`, not `np.asarray`: a JAX device array's
    # zero-copy numpy view is read-only, and every field below is mutated
    # in place while building the injected state.
    arr = lambda a: np.array(a, copy=True)          # noqa: E731
    kind = arr(s.kind); team = arr(s.team)
    alive = arr(s.alive); model = arr(s.model)
    x = arr(s.x); y = arr(s.y)
    hp = arr(s.hp); max_hp = arr(s.max_hp)
    move_order = arr(s.move_order)
    n_waypoints = arr(s.n_waypoints)
    waypoint_key = arr(s.waypoint_key)
    waypoints = arr(s.waypoints)
    level = arr(s.level); gold = arr(s.gold)
    cs = arr(s.cs); deaths = arr(s.deaths)
    spell_level = arr(s.spell_level)
    spell_cooldown = arr(s.spell_cooldown)
    spawn_x = arr(s.spawn_x); spawn_y = arr(s.spawn_y)

    slot_of: Dict[int, int] = {}   # id(entity) -> slot, for debugging only

    for ent in snapshot.entities:
        if ent.kind not in KIND_NAME_TO_ID:
            continue
        if ent.dead:
            # A dead-but-still-present row (buildings keep existing after
            # death; a champion mid-respawn does too). LaneState has no
            # "dead object that still occupies a slot" representation for
            # minions/turrets other than `alive=False`, and a dead champion
            # needs its respawn timer, which the dump does not carry either
            # (see the module docstring) -- so a dead champion is injected
            # as alive=False and left to the sim's own (unknown) respawn
            # clock, which will be wrong. Rare in an idle trace (no kills).
            continue
        ek = KIND_NAME_TO_ID[ent.kind]
        et = SERVER_TEAM_TO_ID.get(ent.team, Team.NEUTRAL)

        if ek == Kind.CHAMPION:
            i = ch_slot[et]
        elif ek == Kind.LANE_MINION:
            if next_minion >= MI_SLICE.stop:
                report.dropped_capacity["LaneMinion"] = (
                    report.dropped_capacity.get("LaneMinion", 0) + 1)
                continue
            i = next_minion
            next_minion += 1
        else:  # TURRET
            if next_turret >= TU_SLICE.stop:
                report.dropped_capacity["LaneTurret"] = (
                    report.dropped_capacity.get("LaneTurret", 0) + 1)
                continue
            i = next_turret
            next_turret += 1

        kind[i] = ek
        team[i] = et
        alive[i] = True
        x[i], y[i] = ent.x, ent.y
        hp[i] = ent.hp
        max_hp[i] = ent.q_max_hp / StatQ
        move_order[i] = ent.ai.move_order if ent.ai else MoveOrder.NONE
        n_waypoints[i] = ent.ai.waypoints if ent.ai else 0

        # ---- model -------------------------------------------------------
        if ek == Kind.CHAMPION:
            model_row = next(r for r, (k, _, t) in enumerate(profiles)
                             if k == Kind.CHAMPION and t == et)
            model_reason = "champion team -> profile (one champion per team)"
        elif ek == Kind.TURRET:
            model_row, model_reason = infer_turret_model(
                float(x[i]), float(y[i]), et)
        else:
            model_row, model_reason = infer_minion_model(
                float(max_hp[i]), et, params, profiles)
        model[i] = model_row if model_row is not None else 0
        if model_row is None:
            key = "unmodelled_turret" if ek == Kind.TURRET else "unmodelled_minion"
            report.dropped_capacity[key] = report.dropped_capacity.get(key, 0) + 1

        # ---- champion-only fields -----------------------------------------
        if ek == Kind.CHAMPION and ent.champ is not None:
            level[i] = ent.champ.level
            gold[i] = ent.champ.q_gold / StatQ
            cs[i] = ent.champ.minions_killed
            deaths[i] = ent.champ.deaths
            for slot in range(4):
                lv, cd = ent.champ.spells[slot]
                spell_level[i, slot] = max(lv, 0)
                spell_cooldown[i, slot] = max(cd, 0) / StatQ
            # Champions never move in this fixture (no orders are issued),
            # so `spawn_x/spawn_y` (only consulted on respawn) are set to the
            # current position; this is a real gap for a driven fixture,
            # harmless for this one.
            spawn_x[i] = ent.x
            spawn_y[i] = ent.y

        # ---- movement / waypoints -------------------------------------
        trustworthy, reason = True, "movement blocked this tick (order not MOVE_TO/ATTACK_TO)"
        mo = move_order[i]
        if mo == MoveOrder.MOVE_TO:
            wp, key, n, reason = reconstruct_waypoints(ent.x, ent.y, et)
            if wp is None:
                trustworthy = False
                n_waypoints[i] = 0
            else:
                waypoints[i, :len(wp)] = wp
                waypoint_key[i] = key
                # Trust the reconstruction's own count over the dump's, but
                # note a mismatch -- it means this minion's history is not
                # "spawned once, walked the corridor ever since".
                if n_waypoints[i] not in (0, n):
                    reason += (f"; dump reports {n_waypoints[i]} waypoints, "
                              f"reconstruction found {n} -- history mismatch")
                    trustworthy = False
                n_waypoints[i] = n
        elif mo == MoveOrder.ATTACK_TO:
            trustworthy = False
            reason = ("chasing a target whose identity is unobservable "
                     "(see module docstring); waypoints left empty")
            n_waypoints[i] = 0

        report.notes.append(UnitInjectionNote(
            # NOTE: `team` here is the server's RAW id (100/200/300, same as
            # `Entity.team`), not the sim's compact `Team` enum (`et`) --
            # deliberately, so this note can be grouped/joined against
            # `Entity` objects from the trace (`one_step.py` does exactly
            # that) without a second translation table.
            kind=ent.kind, team=ent.team, x=ent.x, y=ent.y,
            model_row=int(model[i]), model_reason=model_reason,
            movement_trustworthy=trustworthy, movement_reason=reason,
            slot=i, entity=ent,
        ))

    s = s.replace(
        t_ms=jnp.asarray(snapshot.t_ms, dtype), tick=s.tick,
        kind=jnp.asarray(kind), team=jnp.asarray(team),
        alive=jnp.asarray(alive), model=jnp.asarray(model),
        x=jnp.asarray(x, dtype), y=jnp.asarray(y, dtype),
        waypoints=jnp.asarray(waypoints, dtype),
        waypoint_key=jnp.asarray(waypoint_key),
        n_waypoints=jnp.asarray(n_waypoints),
        move_order=jnp.asarray(move_order),
        hp=jnp.asarray(hp, dtype), max_hp=jnp.asarray(max_hp, dtype),
        spawn_x=jnp.asarray(spawn_x, dtype), spawn_y=jnp.asarray(spawn_y, dtype),
        level=jnp.asarray(level), gold=jnp.asarray(gold, dtype),
        cs=jnp.asarray(cs), deaths=jnp.asarray(deaths),
        spell_level=jnp.asarray(spell_level),
        spell_cooldown=jnp.asarray(spell_cooldown, dtype),
        # wave spawner: derived exactly, see replay_wave_states
        next_spawn_ms=jnp.asarray(wave_state.next_spawn_ms, dtype),
        minion_number=jnp.asarray(wave_state.minion_number, jnp.int32),
        cannon_count=jnp.asarray(wave_state.cannon_count, jnp.int32),
        # everything below is an UNRECOVERABLE default -- see module docstring
        # target=-1, target_priority=14, ai_timer=250, ignore_until=0,
        # help_priority=14, ai_local_time=0, time_since_attack=0,
        # aa_cooldown=0, aa_windup=0, is_attacking=False,
        # has_auto_attacked=False are all already `empty_state`'s defaults.
    )
    return s, report
