"""Tier 1, finally built: inject a server snapshot into a :class:`LaneState`.

``lanerl_jax.parity.archive.sim_vs_server`` explains why nobody had done this yet: the
canonical state dump (``LanerlStateDump.Describe``) does not carry
``TargetUnit``, the auto-attack clock, waypoint *positions*, the minion AI's
own timers/maps, missiles, or the collision-cache position. Its opt-in
diagnostic-internals stream does carry those one-step recovery fields, outside
the canonical hash. Injecting a canonical-only snapshot still does not produce
"the server's state, one step on"; diagnostic recovery must remain visibly
distinct from canonical reset recovery.

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
  - a marching minion's movement waypoints: ``Waypoints.Count`` is in the dump but the
    vertices are not.  A minion with no target walks the same static lane
    corridor every episode (``TOP_LANE_PATH``, forward for blue / reversed for
    red -- see ``sim/init.py``), so a position that lies on that corridor
    pins down the corridor *and* which vertex is next. The same projection
    initializes LaneMinionAI's separate private cursor: exact on-corridor and
    explicitly labelled nearest/upcoming guess off-corridor.

* **unrecoverable** -- defaulted to the same values :func:`empty_state` uses,
  and named here so a downstream report can say which mechanics they poison
  rather than let a default read as an assertion. One exception is folded in
  here rather than given its own top-level category: the MOVE_TO-off-corridor
  waypoints bullet below is not a default at all any more (it is a measured,
  injected guess) but it belongs next to the ATTACK_TO bullet it was tried
  and rejected alongside, not scattered away from that comparison:

  - ``target`` (all units) **without diagnostic internals**: minion-target
    identity is normally not observable
    (see ``parity.diff.UNOBSERVABLE``); champion/turret target identity
    *would* be recoverable from the observation wire / turret trace, but this
    injector does not consume those streams (the recorded fixture is a pure
    no-orders idle trace, so no champion ever has a target and this
    limitation happens not to bite it -- it would for a driven fixture).
    One narrow exception is a minion in ``ATTACK_TO`` with exactly one
    injected lane enemy within acquisition range: that identity is injected
    as a conditional constraint, explicitly labelled because a targetable
    object outside this simulator's lane kinds would invalidate it.  Every
    other case defaults to ``-1`` plus ``ai_timer=250`` (forces immediate
    re-evaluation, see below) means every unit re-acquires its target from
    scratch on every injected tick. That is *not* the same experiment as
    "did the server's held target survive this tick" -- it tests "does a
    from-scratch acquisition agree with the server's *actual* target", which
    conflates a wrong pick with the server's hysteresis (a still-valid
    incumbent is protected from a same-priority closer unit; a fresh
    acquisition is not, see ``sim/minion_ai.py``). Any target-selection
    disagreement this harness finds is real evidence of a difference, but a
    *lack* of disagreement does not confirm the hysteresis rule is right.
  - ``aa_cooldown`` / ``aa_windup`` / ``is_attacking`` / ``has_auto_attacked``
    **without diagnostic internals**:
    completely absent from the dump (measured in
    ``parity/tests/test_autoattack.py``: ``cast_spell`` is ``"-"`` on 100% of
    15,162 champion rows in a run with 32 real swings). Defaulted to
    "not attacking, cooldown ready", except that a minion's ``MOVE_TO`` or
    ``ATTACK_TO`` order rules out an in-flight windup and is injected as such.
    **This is the single biggest hole**: the
    plan's Tier-1 target for auto-attack is the exact fire tick, and this
    injector cannot pin the fire tick of any swing except the first one after
    two units meet (where "cooldown ready" is close to true) -- every swing in
    a sustained fight is tested against a fabricated clock. See
    ``docs/ONE_STEP_DIFFERENTIAL.md`` for how the harness works around this
    (event-level hit/no-hit agreement, which does not need the clock to be
    right, only the outcome).
  - the minion AI's own bookkeeping **without diagnostic internals**:
    ``target_priority`` (-> 14, "no
    incumbent"), ``ignore_until`` (-> 0, nothing ignored), ``help_priority``
    (-> 14, no call for help pending), ``ai_local_time`` (-> 0),
    ``time_since_attack`` (-> 0). None of these are in the dump. Consequence:
    the 4-second give-up rule and the 500 ms post-give-up ignore window can
    never fire in an injected tick (the clock that drives them is reset every
    time), and a call-for-help re-target can never be reproduced.
  - waypoints for a unit whose ``move_order`` is ``ATTACK_TO`` (chasing a
    target): these are set, on the server, to a straight line at the
    attacker's and target's *current* positions and recomputed every tick
    (confirmed by reading ``ObjAIBase.cs:595-671``'s ``RefreshWaypoints``,
    called from ``UpdateTarget`` every tick a target is held, not just on the
    minion AI's 250 ms sweep -- our sim's "3b. RefreshWaypoints" in
    ``sim/step.py`` does the same thing), so they would be exactly
    reconstructable *if* we knew the target's identity. We do not (see
    ``target`` above). A GUESS was tried anyway -- nearest strictly-best-
    priority enemy in acquisition range, i.e. a from-scratch
    ``minion_acquire`` -- and measured against real trace kinematics rather
    than assumed: 70.5% of guesses landed on an enemy already IN attack
    range, which is impossible for the server's true (still-chasing)
    incumbent, confirming ``sim/minion_ai.py``'s hysteresis warning above:
    the true target is usually a farther, hysteresis-protected unit the
    naive nearest-search never considers. Worse, "predict no movement at
    all" (freezing, i.e. what this injector already does) turned out to beat
    the guess outright: 86.3% of ATTACK_TO minions land within 0.5 units of
    their PRE-tick position one tick later regardless -- most of them are
    packed into a collision scrum and functionally stationary even while
    nominally still closing on a target -- so a moving guess is *net
    negative* against the honest default. Left empty (``n_waypoints=0``,
    i.e. frozen) and flagged ``movement_trustworthy=False``, as before, but
    now for a measured reason rather than an assumed one.
  - movement waypoints for a minion that is marching (``move_order=MOVE_TO``) but is
    **not** on the known lane corridor within tolerance -- this happens to a
    minion that just gave up a chase (see ``sim/minion_ai.py``'s
    ``ReevaluateBehavior``: the server resumes the corridor from its separate
    ``PathingWaypoints[currentWaypointIndex]``. The sim now represents that
    cursor and the injector initializes it by nearest/upcoming projection;
    the transient movement route below remains a one-step guess rather than a
    recovered server waypoint list. Unlike the ATTACK_TO case above, a guess
    HERE was measured to help, decisively: the same corridor projection
    ``reconstruct_waypoints`` uses, but without its perpendicular-distance
    gate (see ``reconstruct_waypoints_relaxed``), beats "predict no
    movement" on 87.1% of ticks (median predicted-position error 0.134
    units, 82.5% within 0.5 units) on a 6,000-tick-pair sample -- these
    minions are actually walking, unlike the ATTACK_TO scrum case, so
    freezing them is usually wrong and a corridor-ward guess is usually
    right. Injected as the fallback when the strict (exact) reconstruction
    refuses, flagged ``movement_trustworthy=True`` -- read that as "measured
    to help", not "recovered from the dump" the way the on-corridor case's
    is; see that function's docstring for the numbers before relying on this
    for anything past Tier 1.

The canonical dump's named buff identities and champion spell cooldowns are
injected where their meaning is known. Diagnostic internals additionally restore
target/AA/AI/missile/waypoint/collision-cache state for a one-step trace, while
finite-buff phase/power, generic cast/channel state and ``can_move`` remain
absent. The injector records those absences per unit rather than treating an
empty simulator field as agreement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..sim.combat import growth_sum
from ..sim.init import CHAMPION_SPAWN, TOP_LANE_PATH
from ..sim.movement_jax import TICK_MS
from ..sim.spells import (
    BUFF_NAMES, E_DURATION_S, Q_BUFF_DURATION, Slot, q_haste_duration_at_rank,
    w_duration_at_rank,
)
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
from .diff import _match_group
from .recover_waypoints import point_line_distance
from .trace import Entity, PosQ, Snapshot, StatQ

__all__ = [
    "KIND_NAME_TO_ID", "SERVER_TEAM_TO_ID",
    "UnitInjectionNote", "InjectionReport",
    "infer_minion_model", "reconstruct_waypoints", "reconstruct_waypoints_relaxed",
    "replay_wave_states", "xp_bounds_for_level", "inject_snapshot",
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

# The state dump prints names; the simulator keeps one typed record per buff
# kind (`sim.state.Buffs`, `STRUCT-001`) and `spells.BUFF_NAMES` maps one to
# the other.  Only names whose identity is documented by the content scripts
# are listed there.  In particular, GarenPassive/GarenPassiveHeal are
# deliberately *not* aliases for a simulated buff: their timers are not on the
# wire and their behaviour is represented elsewhere (regen.py).
_DUMP_BUFF_TO_SIM = BUFF_NAMES

#: A rank-timed buff's spell slot: its duration is a function of the rank it
#: was cast at, which the record carries instead of a duration.
_RANKED_BUFF_SPELL = {"w": Slot.W, "q_haste": Slot.Q}


def _buff_duration_s(field: str, rank: int) -> float:
    """The sim's duration for buff record ``field`` cast at ``rank``."""
    import jax.numpy as jnp
    if field == "e":
        return E_DURATION_S
    if field == "q":
        return Q_BUFF_DURATION
    fn = w_duration_at_rank if field == "w" else q_haste_duration_at_rank
    return float(fn(jnp.asarray(rank, jnp.int32)))


def _rank_for_duration(field: str, duration_s: float) -> int:
    """Invert ``_buff_duration_s`` for the rank-timed records: W lasts
    ``rank + 1`` s, Q-haste ``1.5 + 0.75*(rank-1)`` s."""
    if field == "w":
        r = round(duration_s - 1.0)
    else:
        r = round((duration_s - 1.5) / 0.75 + 1.0)
    return int(min(max(r, 1), 5))


def _set_buff(buffs, field: str, i: int, *, elapsed_s: float, rank: int = 0):
    """Mark buff ``field`` live on unit ``i`` in a numpy-leaved ``Buffs``."""
    if field == "w_passive":
        buffs.w_passive[i] = True
        return
    rec = getattr(buffs, field)
    rec.active[i] = True
    rec.elapsed_s[i] = elapsed_s
    if hasattr(rec, "rank"):
        rec.rank[i] = rank


def xp_bounds_for_level(level: int, xp_to_reach_level) -> Tuple[float, Optional[float]]:
    """Return the closed/open cumulative-XP interval visible as ``level``.

    ``xp_to_reach_level`` is `profiles`' level table (`STRUCT-005`): 19 rows,
    row ``L`` the cumulative XP at which a champion BECOMES level ``L``, row 0
    unused. The dump exposes level but not XP, so ``[table[L], table[L + 1])``
    is the *entire* recoverable fact; choosing the lower bound for
    ``LaneState.xp`` is a deterministic representative, not an observation.
    At the level cap (18, the last row) there is no finite upper bound.
    """
    n = len(xp_to_reach_level)
    L = max(1, min(int(level), n - 1))
    lower = float(xp_to_reach_level[L])
    upper = float(xp_to_reach_level[L + 1]) if L + 1 < n else None
    return lower, upper


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
    #: These are provenance labels, not confidence scores.  A report must say
    #: which state was actually reconstructed and which was merely defaulted.
    target_recovery: str = "not attempted"
    attack_recovery: str = "not attempted"
    collision_cache_recovery: str = "current-position fallback"
    position_recovery: str = "canonical 1/16-quantised position"
    cooldown_recovery: str = "not applicable"
    buff_recovery: str = "not applicable"
    cast_recovery: str = "not applicable"
    #: LaneMinionAI.currentWaypointIndex is not dumped. It is projected onto
    #: the team-relative immutable corridor; the label distinguishes an
    #: on-corridor recovery from the off-corridor nearest/upcoming guess.
    lane_waypoint_recovery: str = "not applicable"
    #: ``INJ-003``: whether ``minionActionTimer`` was recovered onto the
    #: server's own float32 tick grid or left at the dump's rounded value.
    #: A rounded value cannot cross the ``>= 250f`` trigger -- see the call
    #: site -- so this is the difference between the 250 ms sweep running and
    #: not running at all.
    ai_timer_recovery: str = "not applicable"
    #: ``(inclusive lower, exclusive upper)``; ``upper=None`` at level cap.
    xp_bounds: Optional[Tuple[float, Optional[float]]] = None


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

    def provenance_counts(self) -> Dict[str, int]:
        """Count recovery labels for a compact one-step uncertainty report."""
        out: Dict[str, int] = {}
        for note in self.notes:
            for field_name in (
                "target_recovery", "attack_recovery", "collision_cache_recovery",
                "position_recovery",
                "cooldown_recovery", "buff_recovery", "cast_recovery",
                "lane_waypoint_recovery",
            ):
                key = f"{field_name}={getattr(note, field_name)}"
                out[key] = out.get(key, 0) + 1
            if note.xp_bounds is not None:
                out["xp_bounds=within-level interval (lower bound injected)"] = (
                    out.get("xp_bounds=within-level interval (lower bound injected)", 0) + 1)
        return out


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


#: Any value in ``(0, dt]`` completes the swing on the very next tick, which is
#: what ``aa_state == STATE_CASTING`` plus a non-positive grid remainder means.
#: Kept far below the dump's own quantum so it can never be mistaken for a
#: recovered measurement.
_MIN_CASTING_WINDUP = 1.0 / 1_048_576.0


def _f32_from_bits(bits: int) -> float:
    """A float32 bit pattern as published by `Bits()` in the dump, back to a
    float. Same idiom `inject.py` already uses for the buff phase."""
    import numpy as _np
    return float(_np.asarray(_np.uint32(bits & 0xFFFFFFFF)).view(_np.float32).item())


def snap_windup_to_tick_grid(q_windup: float, model_row: int, unit_level: float,
                             params: dict) -> Tuple[float, str]:
    """``AA-002``: recover the exact remaining wind-up from the rounded dump.

    The server's cast clock is **not** a free-running float. ``Spell.Update``
    (`GameServerLib/GameObjects/Spell/Spell.cs:1643-1647`) advances an
    auto-attack by accumulating *upwards*::

        CurrentDelayTime += diff / 1000.0f;
        if (CurrentDelayTime >= CastInfo.DesignerCastTime / CastInfo.AttackSpeedModifier)
        {
            FinishCasting();
        }

    and ``CurrentDelayTime`` is zeroed at the swing's start
    (``AutoAttackSpell.ResetSpellCast()``, `ObjAIBase.cs:1250`). So the elapsed
    cast time is always an exact integer multiple of the tick and the remaining
    wind-up always lies on the grid ``W - k*dt`` for the unit's fixed threshold
    ``W``. ``LanerlStateDump`` publishes that remainder rounded to 1/1024 s,
    which loses the grid -- and losing it is the whole of the `AA-002`
    residual, because the sim's hit test is ``remaining <= dt`` and the
    rounding error (<= 1/2048 s) lands on the wrong side of ``dt`` whenever a
    profile's ``W/dt`` sits near an integer.

    That is not a uniform jitter, which is why `AA-002` measured an 18x
    melee/ranged asymmetry rather than noise: it is a property of each
    profile's ``W/dt``. The blue melee lane minion's is **20.0016**, so its
    final wind-up tick has 2.67e-5 s left -- 18x *below* the dump's 1/2048
    rounding threshold, hence dumped as a flat ``aawindup=0``.

    ``k`` itself is never ambiguous: the rounding error is 17x smaller than the
    tick stride, so ``round((W - q_windup) / dt)`` is exact. This recovers it
    and returns the true remainder. The recovery is rejected -- leaving the
    dump's own value untouched -- when the re-derived remainder disagrees with
    the dump by more than the dump could have rounded, because that means our
    ``W`` is not the server's threshold and snapping would be inventing phase
    rather than recovering it.
    """
    dt = TICK_MS / 1000.0
    asm = 1.0 + (float(params["attack_speed_per_level"][model_row]) / 100.0) * float(
        growth_sum(np.float32(unit_level), np))
    full = float(params["attack_windup"][model_row]) / asm
    if not full > 0.0:
        return q_windup, "no wind-up in this profile"
    k = round((full - q_windup) / dt)
    if k < 0:
        return q_windup, "dumped wind-up exceeds the profile threshold"
    exact = full - k * dt
    if abs(exact - q_windup) > 1.0 / (2.0 * StatQ) + 1e-6:
        return q_windup, (
            f"profile wind-up {full:.6f}s puts tick {k} at {exact:.6f}s, "
            f"{abs(exact - q_windup) * 1000:.3f} ms from the dumped "
            f"{q_windup:.6f}s -- more than rounding, so not snapped")
    if exact <= 0.0:
        return _MIN_CASTING_WINDUP, "snapped onto the server's cast-clock grid"
    # The sim's hit test is `windup - dt <= 0` in **float32**, and for a
    # profile whose `W/dt` is an integer the true remainder differs from `dt`
    # by less than a float32 ulp -- the recovery would survive the injector and
    # then be lost again one line later. So state the server's own comparison,
    # `(k+1)*dt >= W`, in a value float32 cannot round across. The magnitude is
    # preserved wherever it is representable; only a remainder that float32
    # cannot separate from `dt` is nudged, by one ulp.
    dt32 = np.float32(dt)
    out = np.float32(exact)
    if exact <= dt and not np.float32(0.0) < out <= dt32:
        out = np.float32(dt * 0.5)
    elif exact > dt and out <= dt32:
        out = np.nextafter(dt32, np.float32(np.inf))
    # Deliberately NOT `f"... tick {k} ..."`: the provenance census in
    # `one_step.py` groups by this string, and a per-tick label turns one line
    # into a 34-line histogram of the cast clock in every run's report.
    return float(out), "snapped onto the server's cast-clock grid"


def snap_cooldown_to_tick_grid(q_cooldown: float, model_row: int, unit_level: float,
                               params: dict) -> Tuple[float, str]:
    """Recover the exact remaining auto-attack cooldown, the way `AA-002`
    recovered wind-up -- same clock shape, same dump, same fix.

    ``ObjAIBase.Update`` decrements ``_autoAttackCurrentCooldown`` by exactly
    ``diff/1000`` every tick it is positive (`ObjAIBase.cs:1103-1105`), from a
    value set only at swing start, ``1.0f / Stats.GetTotalAttackSpeed()``
    (`:1263`). So -- for the identical reason `AA-002` gave for the wind-up
    clock -- the true remaining cooldown always lies on the grid
    ``period - k*dt`` for the unit's fixed period ``period``. The dump
    (`LanerlStateDump.cs:205`) publishes ``Q(Math.Max(0f, remaining), StatQ)``:
    the clamp happens BEFORE the quantisation, so this is not simply the
    wind-up fix with different constants -- every already-fired tick and
    every not-yet-fired tick within one rounding step of the gate collapse
    onto the same dumped ``0``, and no per-tick formula can tell those apart
    (doing so needs the previous tick's own dump, which is outside this
    function's contract; see the module docstring's discussion of the
    ``aa_fire`` timing question, which this does NOT attempt to resolve). A
    dumped ``0`` is therefore left alone.

    For every OTHER dumped value -- i.e. everywhere the cooldown is not
    within rounding distance of the gate -- ``k`` is exactly as recoverable as
    `AA-002`'s: ``round((period - q_cooldown) / dt)`` is unambiguous because
    the dump's rounding error (<= 1/2048 s) is 17x smaller than the
    17.0667-quantum tick stride. This is the mechanism behind the measured
    82.3%-of-misses "<=2 quanta" mode in `docs/JAX_FIDELITY_LEDGER.md`'s
    `aa_cooldown` row: two independent roundings (this tick's dump and the
    next tick's) of the SAME real-valued grid point, not two different real
    values. Recovering the grid point before the sim decrements it once
    removes the second rounding's chance to disagree with the first.
    """
    dt = TICK_MS / 1000.0
    if q_cooldown <= 0.0:
        return q_cooldown, "at the dump's gate clamp; no grid to recover"
    asm = 1.0 + (float(params["attack_speed_per_level"][model_row]) / 100.0) * float(
        growth_sum(np.float32(unit_level), np))
    period = float(params["attack_period"][model_row]) / asm
    if not period > 0.0:
        return q_cooldown, "no attack period in this profile"
    k = round((period - q_cooldown) / dt)
    if k < 0:
        return q_cooldown, "dumped cooldown exceeds the profile period"
    exact = period - k * dt
    if abs(exact - q_cooldown) > 1.0 / (2.0 * StatQ) + 1e-6:
        return q_cooldown, (
            f"profile period {period:.6f}s puts tick {k} at {exact:.6f}s, "
            f"{abs(exact - q_cooldown) * 1000:.3f} ms from the dumped "
            f"{q_cooldown:.6f}s -- more than rounding, so not snapped")
    if exact <= 0.0:
        return 0.0, "recovered tick is at or past the gate"
    return exact, "snapped onto the server's cooldown-clock grid"


#: ``LaneMinionAI.minionActionTimer`` is only ever ``= 0`` (at a sweep),
#: ``= 250f`` (at construction) or ``+= delta``, and the server's free-run
#: ``deltaTime`` is ``(float)REFRESH_RATE`` (`Game.cs:333`). So the timer is
#: always one of a short, exactly enumerable list of float32 accumulations.
_ACTION_TIMER_DT32 = np.float32(np.float32(1000.0) / np.float32(60.0))


def _action_timer_grid() -> np.ndarray:
    """Every value ``minionActionTimer`` can hold, in the server's float32.

    Two ladders, because there are two starting points: ``0`` after a sweep and
    the ``250f`` a freshly constructed ``LaneMinionAI`` carries (`:22`, which
    is what makes a new minion sweep on its very first update). The trigger
    fires at ``>= 250`` so nothing survives far past it, but the ladders are
    run out to 32 ticks so an unswept minion is covered too.
    """
    out = []
    for start in (np.float32(0.0), np.float32(250.0)):
        t = start
        out.append(float(t))
        for _ in range(32):
            t = np.float32(t + _ACTION_TIMER_DT32)
            out.append(float(t))
    return np.asarray(sorted(set(out)), dtype=np.float64)


_ACTION_TIMER_GRID = _action_timer_grid()


def snap_action_timer_to_tick_grid(q_timer: float) -> Tuple[float, str]:
    """``INJ-003``: recover ``minionActionTimer`` from its rounded dump.

    The trigger this value feeds is ``minionActionTimer >= 250.0f``
    (`LaneMinionAI.cs:90`) and the server crosses it by **0.0000305 ms**: the
    fifteenth float32 accumulation of ``1000f/60f`` is ``250.0000305``, not
    ``250``. `LanerlStateDump` publishes the timer rounded to 1/1024 ms, and
    the fourteenth accumulation -- ``233.3333588`` -- rounds to
    ``238933/1024 = 233.3330078``, i.e. **0.00035 ms low**, 11.5x the whole
    crossing margin. Adding one tick to the rounded value gives
    ``249.9996796``, which is below the threshold, so an injected minion
    **never** sweeps.

    That is not a near miss that shows up sometimes. Measured over the whole
    gate-1 corpus, the sim's own rule fires on **0 of 395,366** minion
    tick-pairs while the server's dumped ``aitimer`` resets on **26,537** of
    them. Every Tier-1 minion-controller number taken before this was measured
    with the regular sweep switched off; only ``TargetJustDied()`` and the
    call-for-help branch ever re-evaluated anything.

    The grid is unambiguous: neighbouring grid points are 16.67 ms apart and
    the dump's error is 0.0005 ms, so the nearest point is the server's value
    by a factor of 3.4e4. Anything further than the dump could have rounded is
    left alone rather than snapped, because that would mean the value is not
    on this grid at all and snapping would be inventing phase.
    """
    i = int(np.abs(_ACTION_TIMER_GRID - q_timer).argmin())
    exact = float(_ACTION_TIMER_GRID[i])
    err = abs(exact - q_timer)
    if err > 1.0 / (2.0 * StatQ) + 1e-9:
        return q_timer, (
            f"dumped {q_timer:.6f} ms is {err * 1000:.3f} us from the nearest "
            f"tick-grid value {exact:.6f} -- more than the dump could have "
            f"rounded, so not snapped")
    return exact, f"exact tick grid ({exact:.7f} ms, dumped {q_timer:.7f})"


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


def _corridor_path(team: int) -> np.ndarray:
    path = np.asarray(TOP_LANE_PATH, np.float32)
    if team == Team.RED:
        path = path[::-1].copy()
    return path


def _key_from_projection(path: np.ndarray, seg: int, t: float) -> int:
    key = seg + 1
    if t > 0.999 and key + 1 < len(path):
        key += 1
    return min(key, len(path) - 1)


def reconstruct_waypoints(
    x: float, y: float, team: int, tol: float = CORRIDOR_TOLERANCE
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[int], str]:
    """Best-effort recovery of a marching minion's waypoint array.

    Returns ``(waypoints (L,2) or None, waypoint_key or None, n_waypoints or
    None, reason)``. ``None`` means "refused" -- the caller must not invent a
    destination, see the module docstring.
    """
    path = _corridor_path(team)
    seg, dist, t = _project_to_polyline(x, y, path)
    if dist > tol:
        return None, None, None, (
            f"position is {dist:.1f} units off the known lane corridor "
            f"(tolerance {tol}) -- not walking the spawn path, or resuming "
            "from a lost target (PathingWaypoints is not modelled)")
    key = _key_from_projection(path, seg, t)
    return path, key, len(path), f"on corridor, {dist:.2f} units off, key={key}"


def reconstruct_waypoints_relaxed(
    x: float, y: float, team: int
) -> Tuple[np.ndarray, int, int, str]:
    """Fallback for a marching minion that :func:`reconstruct_waypoints`
    refuses: the same corridor projection, but with NO perpendicular-distance
    gate, so it always returns a guess rather than ``None``.

    This is a GUESS, not a recovery: the real cause of being off-corridor is
    almost always "resuming the corridor after losing a chase target" (see
    the module docstring's ``target`` section), where the server picks up
    from ``PathingWaypoints[currentWaypointIndex]`` -- state this injector
    does not have -- rather than the nearest point on the corridor to wherever
    combat left the minion. Nearest-point-on-corridor is the best available
    proxy for that index without it.

    Measured against real trace kinematics (no sim involved; same method as
    the chase-target guess this module's ATTACK_TO branch tried and rejected,
    see that branch's comment) on a 6,000-tick-pair sample: this beats
    "predict no movement" (the alternative -- refuse and let the unit freeze)
    on 87.1% of ticks, median predicted-position error 0.134 units, 82.5% of
    predictions within 0.5 units of the real one. Not exact -- do not read
    the resulting ``movement_trustworthy=True`` as "recovered from the dump"
    the way the on-corridor case's is -- but a validated, one-sided
    improvement over refusing, unlike the ATTACK_TO guess. See
    docs/TIER1_POST_REORDER.md for the measurement this docstring cites.
    """
    path = _corridor_path(team)
    seg, dist, t = _project_to_polyline(x, y, path)
    key = _key_from_projection(path, seg, t)
    return path, key, len(path), (
        f"RELAXED reconstruction (guess, not recovery): {dist:.1f} units off "
        f"the known lane corridor, nearest projection used anyway, key={key} "
        "-- see reconstruct_waypoints_relaxed's docstring for the measured "
        "accuracy before trusting this for anything beyond Tier 1")


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
    ``trace[i]`` to ``trace[i + 1]``.

    FIXED 2026-09-16 (was inverted): an earlier version paired ``trace[i]``
    with the counters as they stood *before* ``trace[i]``'s own tick, on the
    reasoning "the state paired with a snapshot at time T is the result of
    the previous tick's spawn decision, not one evaluated at T itself". That
    reasoning had it backwards. ``trace[i].t_ms`` is read from the dump,
    which ``LanerlHooks.OnUpdate`` emits AFTER tick i's own
    ``LevelScript.Update`` already ran (`parity/trace.py`'s module docstring)
    -- so if ``trace[i]`` is itself a spawn tick, the new minion is ALREADY in
    ``trace[i]``'s own entity list, and ``tick()`` (called on the injected
    state to step FROM ``trace[i]`` TO ``trace[i+1]``, checking
    ``state.t_ms >= next_spawn_ms + ...`` with ``state.t_ms = trace[i].t_ms``)
    must be given counters that already reflect that decision, or it
    re-evaluates the exact same threshold at the exact same game time and
    spawns AGAIN on top of the unit the dump already shows. Confirmed by
    replaying this trace's first wave: at the trace's first post-90s tick,
    both barracks already show their one minion (0 real deaths, 0 real
    arrivals -- an unchanged population), while the un-fixed pairing had the
    sim spawn a second one into the SAME tick's prediction. This -- not a
    fundamental modelling gap -- is what produced
    docs/TIER1_POST_REORDER.md's "1,665 of 1,667 spawn ticks disagreed on
    count": nearly every spawn-adjacent tick was off by this one index, in
    one direction or the other depending on which side of a threshold
    ``trace[i].t_ms`` fell on.
    """
    st = WaveState(next_spawn_ms=FIRST_WAVE_MS, minion_number=0, cannon_count=0)
    out: List[WaveState] = []
    for snap in trace:
        step_waves(st, float(snap.t_ms))
        out.append(WaveState(next_spawn_ms=st.next_spawn_ms,
                             minion_number=st.minion_number,
                             cannon_count=st.cannon_count))
    return out


def inject_snapshot(
    snapshot: Snapshot, wave_state: WaveState, params: dict, profiles,
    dtype=None, previous_snapshot: Optional[Snapshot] = None,
) -> Tuple[LaneState, InjectionReport]:
    """Build a :class:`LaneState` from one server :class:`Snapshot`.

    ``previous_snapshot``, when supplied, is the immediately preceding dump
    tick.  It is used only for a labelled collision-cache proxy; no future
    snapshot is consulted, so this remains a causal injection.

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
    lane_waypoint_key = arr(s.lane_waypoint_key)
    waypoints = arr(s.waypoints)
    level = arr(s.level); xp = arr(s.xp); gold = arr(s.gold)
    cs = arr(s.cs); deaths = arr(s.deaths)
    spell_level = arr(s.spell_level)
    spell_cooldown = arr(s.spell_cooldown)
    target = arr(s.target)
    is_attacking = arr(s.is_attacking)
    has_auto_attacked = arr(s.has_auto_attacked)
    aa_cooldown = arr(s.aa_cooldown)
    aa_windup = arr(s.aa_windup)
    # numpy copies of every buff record's arrays, same struct shape
    import jax
    buffs = jax.tree.map(arr, s.buffs)
    collision_x = arr(s.collision_x); collision_y = arr(s.collision_y)
    collision_present = arr(s.collision_present)
    spawn_x = arr(s.spawn_x); spawn_y = arr(s.spawn_y)
    spawn_seq = arr(s.spawn_seq)
    ai_timer = arr(s.ai_timer); target_priority = arr(s.target_priority)
    had_target = arr(s.had_target)
    ignore_until = arr(s.ignore_until); help_priority = arr(s.help_priority)
    ai_local_time = arr(s.ai_local_time)
    time_since_attack = arr(s.time_since_attack)
    missile_alive = arr(s.missile_alive); missile_x = arr(s.missile_x)
    missile_y = arr(s.missile_y); missile_tx = arr(s.missile_tx)
    missile_source = arr(s.missile_source)
    missile_damage = arr(s.missile_damage); missile_speed = arr(s.missile_speed)

    slot_of: Dict[int, int] = {}   # id(entity) -> slot, for temporal recovery
    note_of: Dict[int, UnitInjectionNote] = {}

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
        slot_of[id(ent)] = i

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
            # The lower bound is deliberately only a representative.  The
            # note records the full within-level interval below.
            xp[i], _ = xp_bounds_for_level(int(level[i]), params["xp_to_reach_level"])
            gold[i] = ent.champ.q_gold / StatQ
            cs[i] = ent.champ.minions_killed
            deaths[i] = ent.champ.deaths
            for slot in range(4):
                lv, cd = ent.champ.spells[slot]
                spell_level[i, slot] = max(lv, 0)
                spell_cooldown[i, slot] = max(cd, 0) / StatQ
            # `spawn_x/spawn_y` is the champion's FOUNTAIN, not wherever it
            # happens to be standing.
            #
            # This used to be `ent.x, ent.y`, with a comment saying that was a
            # "real gap for a driven fixture, harmless for this one" because
            # the only consumer was respawn. That was wrong: `step.tick`'s
            # fountain-heal block (`0b`) tests
            # `d_spawn2 <= _FOUNTAIN_RADIUS ** 2` against the SAME field, so an
            # injected champion was always standing in its own fountain and
            # took a **15% of max HP** pulse every second of simulated time.
            # Invisible in Tier 1 -- `fountain_heal_ms` starts at 0 and one
            # 16.667 ms tick can never reach the 1,000 ms period -- and
            # decisive in any free run: measured on `parity/tier15.py`, an
            # injected champion under six attacking minions LOST 11.6 HP over
            # 7.6 s where the server's lost 587.4, because it was being healed
            # ~100 HP/s by a fountain it was 12,000 units away from.
            spawn_x[i], spawn_y[i] = CHAMPION_SPAWN[et]

        # ---- movement / waypoints -------------------------------------
        trustworthy, reason = True, "movement blocked this tick (order not MOVE_TO/ATTACK_TO)"
        lane_key_reason = "not a LaneMinion"
        mo = move_order[i]
        if ek == Kind.LANE_MINION:
            # `currentWaypointIndex` is private AI state absent from the dump.
            # The immutable path and direction ARE known, so choose the
            # upcoming vertex of the nearest corridor segment. This is exact
            # on the corridor; after a chase it is an explicit nearest-path
            # guess, but never the silently catastrophic default-0 cursor.
            lane_path = _corridor_path(et)
            lane_seg, lane_dist, lane_t = _project_to_polyline(
                ent.x, ent.y, lane_path)
            lane_waypoint_key[i] = _key_from_projection(lane_path, lane_seg, lane_t)
            lane_key_reason = (
                f"on-corridor upcoming index={lane_waypoint_key[i]} "
                f"(perpendicular distance {lane_dist:.2f})"
                if lane_dist <= CORRIDOR_TOLERANCE else
                f"OFF-CORRIDOR nearest/upcoming projection guess index="
                f"{lane_waypoint_key[i]} (perpendicular distance {lane_dist:.1f}; "
                "private currentWaypointIndex unobservable)")
        if mo == MoveOrder.MOVE_TO:
            wp, key, n, reason = reconstruct_waypoints(ent.x, ent.y, et)
            if wp is None:
                # Strict (exact) reconstruction refused -- fall back to the
                # RELAXED guess rather than freezing the unit. Validated to
                # beat "predict no movement" on 87.1% of ticks (see
                # reconstruct_waypoints_relaxed's docstring); still a guess,
                # so `trustworthy=True` here means "measured to help", not
                # "recovered from the dump" the way the exact branch's does.
                wp, key, n, reason = reconstruct_waypoints_relaxed(
                    ent.x, ent.y, et)
                waypoints[i, :len(wp)] = wp
                waypoint_key[i] = key
                n_waypoints[i] = n
                trustworthy = True
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
            # A target-identity GUESS was tried here (nearest strictly-best-
            # priority enemy in acquisition range, i.e. `minion_acquire` with
            # no incumbent -- see `guess_chase_target`, kept below for the
            # record) and measured against real trace kinematics rather than
            # assumed: on a 6,000-tick-pair sample, 70.5% of guesses landed on
            # an enemy already WITHIN attack range, which is impossible for
            # the server's true held target (`ObjAIBase.cs`'s
            # `RefreshWaypoints` would already have set `Hold`, not
            # `AttackTo`, the moment that happened) -- i.e. most of the time
            # the true incumbent is a FARTHER, hysteresis-protected unit the
            # naive nearest-search never considers (`sim/minion_ai.py`'s own
            # docstring predicts exactly this). Restricting the candidate set
            # to enemies NOT already in range is the obvious next refinement
            # and was NOT validated in the time available -- see
            # docs/TIER1_POST_REORDER.md for exactly what was and was not
            # checked. Even on the SURVIVING (not-already-in-range) guesses,
            # predicted position missed the real one by a median 5.2 units --
            # essentially a full tick's travel, i.e. uncorrelated with the
            # truth, not a small residual error. Guessing does not clear the
            # bar this harness sets for calling something "recovered": refuse,
            # as before.
            trustworthy = False
            reason = ("chasing a target whose identity is unobservable "
                     "(see module docstring); waypoints left empty")
            n_waypoints[i] = 0

        note = UnitInjectionNote(
            # NOTE: `team` here is the server's RAW id (100/200/300, same as
            # `Entity.team`), not the sim's compact `Team` enum (`et`) --
            # deliberately, so this note can be grouped/joined against
            # `Entity` objects from the trace (`one_step.py` does exactly
            # that) without a second translation table.
            kind=ent.kind, team=ent.team, x=ent.x, y=ent.y,
            model_row=int(model[i]), model_reason=model_reason,
            movement_trustworthy=trustworthy, movement_reason=reason,
            slot=i, entity=ent,
            cooldown_recovery=(
                "exact dumped cooldowns" if ek == Kind.CHAMPION and ent.champ is not None
                else "not applicable"),
            cast_recovery=(
                "no active cast visible" if ent.ai is not None
                and ent.ai.cast_spell == "-" and ent.ai.channel_spell == "-"
                else "visible cast/channel has no LaneState representation"),
            lane_waypoint_recovery=lane_key_reason,
            xp_bounds=(xp_bounds_for_level(ent.champ.level, params["xp_to_reach_level"])
                       if ek == Kind.CHAMPION and ent.champ is not None else None),
        )
        report.notes.append(note)
        note_of[id(ent)] = note

    # ---- named buffs: identity can be recovered, timing cannot ------------
    # Buff.Update advances elapsed before applying its semantics.  For active
    # finite buffs the dump omits elapsed/duration/power, so do NOT invent a
    # remaining duration or a cast-time damage snapshot.  The buff is placed
    # AT ITS END (elapsed = its full duration): that preserves the incoming
    # identity for collision's pre-buff ghost check (E), then expires safely
    # at the next buff update -- what a zero duration did in the old lane
    # table, where duration was state.  (A consequence of durations no longer
    # being state, and of this phase being invented either way: an
    # identity-only E now reads as past `E_CANCEL_MIN_S`, i.e. cancellable,
    # where the zero-duration form read as 0 s into the spin.)  A rank-timed
    # buff (W, Q-haste) takes the unit's current rank in that spell.  The
    # permanent W passive needs no phase and is represented exactly.
    for note in report.notes:
        ent = note.entity
        if ent is None or ent.ai is None:
            continue
        known, unknown = [], []
        for name in ent.ai.buffs:
            bfield = _DUMP_BUFF_TO_SIM.get(name)
            if bfield is None:
                unknown.append(name)
                continue
            rank = 0
            if bfield in _RANKED_BUFF_SPELL:
                rank = int(min(max(spell_level[note.slot,
                                               _RANKED_BUFF_SPELL[bfield]], 1), 5))
            _set_buff(buffs, bfield, note.slot, rank=rank,
                      elapsed_s=(0.0 if bfield == "w_passive"
                                 else _buff_duration_s(bfield, rank)))
            known.append(name)
        if known:
            finite = [b for b in known if b != "GarenWPassive"]
            note.buff_recovery = (
                "exact permanent identity" if not finite else
                "identity only; finite-buff phase/power intentionally unresolved")
        elif ent.ai.buffs:
            note.buff_recovery = "dump buff names have no LaneState mapping"
        if unknown:
            note.buff_recovery += "; unmapped=" + "+".join(unknown)

    # The optional internal stream is excluded from the canonical hash, but
    # makes one-tick replay exact: NetIds recover identity, attack phase and
    # LaneMinionAI's private clocks without changing server behaviour.
    unmatched = list(snapshot.ai_internals)
    internal_for_note = {}
    note_by_slot = {n.slot: n for n in report.notes}
    for note in report.notes:
        candidates = [v for v in unmatched if v.kind == note.kind and v.team == note.team]
        if not candidates:
            continue
        qx, qy = round(note.x * 16), round(note.y * 16)
        internal = min(candidates, key=lambda v: (v.q_x - qx) ** 2 + (v.q_y - qy) ** 2)
        unmatched.remove(internal)
        internal_for_note[note.slot] = internal

    id_to_slot = {v.net_id: slot for slot, v in internal_for_note.items()}
    creation_rank = {net_id: rank for rank, net_id in enumerate(sorted(id_to_slot))}
    for slot, internal in internal_for_note.items():
        note = note_by_slot[slot]
        if internal.x_bits is not None and internal.y_bits is not None:
            x[slot] = np.asarray(
                np.uint32(internal.x_bits & 0xFFFFFFFF)).view(np.float32).item()
            y[slot] = np.asarray(
                np.uint32(internal.y_bits & 0xFFFFFFFF)).view(np.float32).item()
            note.position_recovery = "exact float32 bits from diagnostic stream"
        spawn_seq[slot] = creation_rank[internal.net_id]
        # `BUFF-001`: the buff PHASE, which the canonical row cannot carry.
        # The identity-only pass above places a finite buff at its END, and
        # the consequence is not cosmetic: it expires Garen E on the very tick
        # it is injected, `CanAttack` comes back, and a champion the server
        # has locked out of attacking for the rest of its 3.0 s spin swings in
        # the simulator -- measured at 258 of 647 sim-fires-early rows, 40%,
        # and previously charged to the simulator as a missing `CanAttack()`
        # gate. With `aibuffs` the phase is exact, so restore it. The dumped
        # DURATION is not state in the sim (`STRUCT-001`): E's and Q's are
        # constants, and W's and Q-haste's give back the rank they were cast
        # at, which is what the record carries.
        #
        # E's damage snapshot (`buffs.e.power`) is still NOT restored: the
        # dump does not publish it. Recovering phase while inventing power
        # would trade a visible residual for an invisible one.
        if internal.buffs_phase is not None:
            mismatched = []
            for bname, el_bits, du_bits in internal.buffs_phase:
                bfield = _DUMP_BUFF_TO_SIM.get(bname)
                if bfield is None:
                    continue
                el = np.asarray(
                    np.uint32(el_bits & 0xFFFFFFFF)).view(np.float32).item()
                du = np.asarray(
                    np.uint32(du_bits & 0xFFFFFFFF)).view(np.float32).item()
                rank = (_rank_for_duration(bfield, du)
                        if bfield in _RANKED_BUFF_SPELL else 0)
                if (bfield != "w_passive"
                        and np.float32(_buff_duration_s(bfield, rank))
                        != np.float32(du)):
                    mismatched.append(f"{bname}={du}")
                _set_buff(buffs, bfield, slot, rank=rank,
                          elapsed_s=0.0 if bfield == "w_passive" else el)
            note.buff_recovery = (
                "exact phase from diagnostic stream (power still unresolved)")
            if mismatched:
                note.buff_recovery += ("; dumped duration differs from the "
                                       "sim's: " + "+".join(mismatched))
        target[slot] = id_to_slot.get(internal.target_net_id, -1)
        # `aa_cooldown`'s own grid recovery -- same mechanism as the wind-up
        # snap just below, see `snap_cooldown_to_tick_grid`'s docstring.
        if internal.aa_cooldown_bits is not None:
            # EXACT and UNCLAMPED, straight off the server. `aacd=` publishes
            # `Q(Math.Max(0f, remaining), StatQ)` -- the clamp runs BEFORE the
            # quantisation, so a still-positive cooldown within one rounding
            # step of the gate publishes as a flat 0 and is indistinguishable
            # from genuinely ready. Injecting that 0 tells the simulator READY
            # when the server was not, and it swings one tick early.
            #
            # MEASURED on a 2,500-pair window of the canonical corpus: of the
            # 122 `sim fires, server does NOT` rows, **122 (100%) sit on a
            # dumped cooldown of 0, and 121 of those have a strictly POSITIVE
            # exact value** (residue 5.51e-07 s). Strictly negative -- the
            # server genuinely ready and something else stopping it, i.e. an
            # actual port bug -- is **0**.
            #
            # `snap_cooldown_to_tick_grid` below cannot fix this and is not at
            # fault: the clamp has already destroyed the SIGN before the
            # reconstruction sees the value, so no amount of grid snapping can
            # tell "+5.51e-07, do not fire" from "0, fire". Only the exact
            # bits can, which is why `aacdbits=` was added. It has been
            # published and parsed since 2026-09-21 and simply never consumed.
            aa_cooldown[slot] = _f32_from_bits(internal.aa_cooldown_bits)
            cd_why = "EXACT (aacdbits, unclamped)"
        else:
            cd_snapped, cd_why = snap_cooldown_to_tick_grid(
                internal.q_aa_cooldown / StatQ, int(model[slot]),
                float(level[slot]), params)
            aa_cooldown[slot] = cd_snapped
        is_attacking[slot] = internal.is_attacking
        has_auto_attacked[slot] = internal.has_auto_attacked
        note.target_recovery = "exact NetId from diagnostic internal stream"
        # `aa_state == 1` is `SpellState.STATE_CASTING`: the server saying this
        # unit's wind-up has NOT completed. `AA-002`: the dumped remainder is
        # rounded to 1/1024 s, so snap it back onto the server's own tick grid
        # before the sim's `remaining <= dt` hit test reads it.
        if internal.aa_state == 1 and internal.is_attacking:
            if internal.aa_windup_bits is not None:
                # EXACT, straight off the server. `aawindup=` publishes
                # `Q(max(0, W - CurrentDelayTime), StatQ)`, and the case that
                # matters is sub-quantum: measured on this corpus, **522 of
                # 53,673 mid-windup LaneMinion unit-ticks (0.973%) publish a
                # flat 0 while the true remainder is positive**, median
                # 2.676e-05 s = 0.0274 of a quantum. That is `AA-002`.
                #
                # `snap_windup_to_tick_grid` below exists to reconstruct that
                # residue by re-deriving `W - k*dt` from an attack-speed
                # multiplier it computes itself. It is a good reconstruction
                # and it is still a reconstruction: it can be REJECTED when
                # its derived `asm` disagrees with the server's, and then it
                # falls back to the same flat 0 it was built to avoid.
                # `aawindupbits=` removes the derivation rather than repairing
                # a term of it -- no `W`, no `asm`, no tick grid. It can only
                # be wrong if the server's own field is wrong.
                aa_windup[slot] = _f32_from_bits(internal.aa_windup_bits)
                note.attack_recovery = (
                    "exact attack flags from diagnostic stream; "
                    f"cooldown {cd_why}; wind-up EXACT (aawindupbits)")
                snapped = why = None
            else:
                snapped, why = snap_windup_to_tick_grid(
                    internal.q_aa_windup / StatQ, int(model[slot]),
                    float(level[slot]), params)
                aa_windup[slot] = snapped
            if why is not None:
                note.attack_recovery = (
                    "exact attack flags from diagnostic stream; "
                    f"cooldown {cd_why}; wind-up {why}")
        else:
            aa_windup[slot] = 0.0
            note.attack_recovery = (
                "exact windup/attack flags from diagnostic stream; "
                f"cooldown {cd_why}")
        if internal.waypoints:
            width = min(len(internal.waypoints), waypoints.shape[1])
            waypoints[slot] = 0.0
            waypoints[slot, :width] = np.asarray(
                internal.waypoints[:width], dtype=np.float32) / 16.0
            # `INJ-002`. The cap is `width`, NOT `width - 1`.
            # `CurrentWaypointKey == Waypoints.Count` is a legal, common and
            # load-bearing server state: it is exactly `IsPathEnded()`
            # (`AttackableUnit.cs:1008-1011`), and `ResetWaypoints`
            # (`:996-1002`) produces it on purpose -- `Waypoints =
            # [Position]`, `CurrentWaypointKey = 1` -- every time
            # `StopMovement()` runs. Clamping that to `width - 1 = 0` turned
            # "this unit is stopped" into "this unit has one waypoint left,
            # at the place it is already standing", and
            # `movement_jax.step_move_units`' `k < n` gate then let it walk
            # back to that waypoint. A minion's per-tick budget (325 u/s ->
            # 5.417 u) is just larger than a collision escape, so the walk
            # back landed exactly on the pre-collision position and CANCELLED
            # that tick's push-apart. Measured: that is the whole of the
            # 1,742-miss `LaneMinion.position_linf` gate-1 residual -- on
            # three drilled cases the error goes 5.398/5.380/5.410 u ->
            # 0.007/0.037/0.038 u, inside the dump's own 1/16 u quantisation,
            # from this one character. `step_move_units` already clamps the
            # gather index, so a key equal to the count is safe to inject.
            waypoint_key[slot] = min(internal.waypoint_key, width)
            n_waypoints[slot] = width
            note.movement_trustworthy = len(internal.waypoints) <= waypoints.shape[1]
            note.movement_reason = (
                "exact waypoint list/current key from diagnostic stream"
                if note.movement_trustworthy else
                "diagnostic waypoint list exceeded fixed LaneState capacity")
        if internal.q_ai_timer is not None:
            # `INJ-003`. Same species as the wind-up above, and it costs more.
            # `minionActionTimer` is zeroed at every sweep and then only ever
            # `+= delta` (`LaneMinionAI.cs:78,96`), so it is an exact
            # float32 accumulation of `(float)REFRESH_RATE` (`Game.cs:333`,
            # `deltaTime = (float)REFRESH_RATE` in the free-run path this
            # corpus was recorded with) and lies on a known grid.  The dump
            # rounds it to 1/1024 ms, and the trigger it feeds is a bare
            # `>= 250.0f` whose crossing margin is **0.0000305 ms** -- the
            # 15th accumulation is 250.0000305, not 250.  The rounding error
            # at the 14th (233.3333588 -> 238933/1024 = 233.3330078) is
            # 0.00035 ms, **11.5x larger** than that margin and in the wrong
            # direction, so the injected value misses the threshold every
            # single time.  Measured on the whole corpus before this: the
            # sim's rule `q/1024 + float32(1000/60) >= 250` fires on
            # **0 of 395,366** minion tick-pairs against the server's own
            # 26,537 sweeps.  The regular 250 ms sweep was switched off for
            # every Tier-1 minion number ever taken; only the event branches
            # (`TargetJustDied` / call-for-help) ever ran.
            snapped, why = snap_action_timer_to_tick_grid(
                internal.q_ai_timer / StatQ)
            ai_timer[slot] = snapped
            note.ai_timer_recovery = why
        if internal.q_ai_local is not None:
            ai_local_time[slot] = internal.q_ai_local / StatQ
        if internal.q_time_since_attack is not None:
            time_since_attack[slot] = internal.q_time_since_attack / StatQ
        if internal.target_priority is not None:
            target_priority[slot] = internal.target_priority
        if internal.had_target is not None:
            # `aihad=` is the server's own latch. Substituting
            # `target >= 0` for it disagrees on 556 of 395,486 scored
            # LaneMinion unit-ticks -- the same order as the whole
            # `target` residual (522) and in the direction that opens
            # the port's `TargetJustDied` arm when the server's stays
            # shut. Injecting it removes a reconstruction, which is the
            # cheapest class of fix: it changes no model, only what the
            # model is started from.
            had_target[slot] = bool(internal.had_target)
        if internal.lane_waypoint_key is not None:
            lane_waypoint_key[slot] = internal.lane_waypoint_key
            note.lane_waypoint_recovery = "exact private currentWaypointIndex"
        for other_id, q_until in internal.ignored:
            other = id_to_slot.get(other_id)
            if other is not None:
                ignore_until[slot, other] = q_until / StatQ
        for other_id, priority in internal.help:
            other = id_to_slot.get(other_id)
            if other is not None:
                help_priority[slot, other] = priority
        other = id_to_slot.get(internal.target_net_id)
        if (not internal.waypoints and other is not None
                and move_order[slot] == MoveOrder.ATTACK_TO):
            waypoints[slot, 0] = (x[slot], y[slot])
            waypoints[slot, 1] = (x[other], y[other])
            waypoint_key[slot] = 1
            n_waypoints[slot] = 2
            note.movement_trustworthy = True
            note.movement_reason = "exact target-derived ATTACK_TO segment"

    for mi, internal in enumerate(snapshot.missile_internals[:len(missile_alive)]):
        source = id_to_slot.get(internal.owner_net_id)
        dest = id_to_slot.get(internal.target_net_id)
        if source is None or dest is None:
            continue
        missile_alive[mi] = True
        if internal.x_bits is not None and internal.y_bits is not None:
            missile_x[mi] = np.asarray(
                np.uint32(internal.x_bits & 0xFFFFFFFF)).view(np.float32).item()
            missile_y[mi] = np.asarray(
                np.uint32(internal.y_bits & 0xFFFFFFFF)).view(np.float32).item()
        else:
            missile_x[mi], missile_y[mi] = internal.q_x / 16.0, internal.q_y / 16.0
        missile_source[mi], missile_tx[mi] = source, dest
        missile_speed[mi] = internal.q_speed / StatQ
        missile_damage[mi] = internal.q_damage / StatQ

    # ---- target / attack-state facts recoverable from the current dump -----
    # A live minion in ATTACK_TO has a held target outside ideal attack range.
    # If the dump leaves exactly one *modelled* enemy inside that minion's
    # acquisition range, its identity is constrained conditionally.  This intentionally does
    # not guess from nearest-target priority or use the result for champions /
    # turrets, whose held-target rules differ.
    live_notes = [n for n in report.notes if n.entity is not None and not n.entity.dead]
    for note in live_notes:
        ent = note.entity
        assert ent is not None
        if note.slot in internal_for_note:
            continue
        if ent.kind != "LaneMinion" or ent.ai is None:
            continue
        if ent.ai.move_order == MoveOrder.ATTACK_TO:
            is_attacking[note.slot] = False
            aa_windup[note.slot] = 0.0
            note.attack_recovery = (
                "windup ruled out by ATTACK_TO; cooldown/last-hit state unobservable")
            acq = float(params["acquisition_range"][int(model[note.slot])])
            candidates = [other for other in live_notes
                          if other.team != note.team
                          and math.hypot(other.x - note.x, other.y - note.y) <= acq]
            if len(candidates) == 1:
                target[note.slot] = candidates[0].slot
                note.target_recovery = (
                    "unique ATTACK_TO candidate among injected lane entities "
                    "(unmodelled targetable kinds remain a blind spot)")
            else:
                note.target_recovery = (
                    f"unresolved ATTACK_TO target ({len(candidates)} injected candidates)")
        elif ent.ai.move_order == MoveOrder.MOVE_TO:
            # LaneMinionAI reaches MOVE_TO only after it has no target.  This
            # excludes a windup without assigning an invisible target clock.
            is_attacking[note.slot] = False
            aa_windup[note.slot] = 0.0
            note.target_recovery = "no held target implied by minion MOVE_TO"
            note.attack_recovery = "windup ruled out by minion MOVE_TO; cooldown unobservable"
        else:
            note.target_recovery = "unobservable from dump"
            note.attack_recovery = "auto-attack clock unobservable from dump"

    # ---- collision cache ---------------------------------------------------
    # At N+1 collision queries the cache rebuilt before N's movement.  The N
    # dump is after movement, but N-1 is a materially better proxy than N for a
    # continuously tracked unit.  Matching is intentionally bounded to one
    # normal tick; crowded/teleport/death cases fall back to the observable N
    # position and are reported as such.
    collision_x[:] = x
    collision_y[:] = y
    collision_present[:] = alive & (kind != Kind.NONE)
    # Never turn an older, dropped-log snapshot into an apparently precise
    # quadtree cache.  A normal server interval is 16--17 ms; 34 permits a
    # single scheduler-sized long tick but refuses a genuine trace gap.
    previous_is_adjacent = (
        previous_snapshot is not None
        and 0 < snapshot.t_ms - previous_snapshot.t_ms <= 34)
    if previous_is_adjacent:
        previous_by_group: Dict[Tuple[str, int], List[Entity]] = {}
        current_by_group: Dict[Tuple[str, int], List[Entity]] = {}
        for n in report.notes:
            assert n.entity is not None
            current_by_group.setdefault((n.kind, n.team), []).append(n.entity)
        for old in previous_snapshot.entities:
            if old.kind in KIND_NAME_TO_ID and not old.dead and old.team is not None:
                previous_by_group.setdefault((old.kind, old.team), []).append(old)
        for key, cur_entities in current_by_group.items():
            matched, _old_only, _new_only = _match_group(
                previous_by_group.get(key, []), cur_entities, 16 * 8)
            for old, cur, _distance_q in matched:
                slot = slot_of[id(cur)]
                collision_x[slot], collision_y[slot] = old.x, old.y
                note_of[id(cur)].collision_cache_recovery = (
                    "preceding-position temporal proxy (pre-move cache unobservable)")

    # New diagnostic traces expose the actual quadtree position.  The stream
    # is deliberately outside the canonical hash, so this improves differential
    # injection without changing reset/determinism semantics.  Older traces
    # retain the explicitly labelled temporal proxy above.
    for slot, internal in internal_for_note.items():
        if not internal.collision_observed:
            continue
        if internal.collision_q_x is None or internal.collision_q_y is None:
            collision_present[slot] = False
            note_by_slot[slot].collision_cache_recovery = (
                "exact diagnostic absence (not present in server quadtree)")
        else:
            if (internal.collision_x_bits is not None
                    and internal.collision_y_bits is not None):
                collision_x[slot] = np.asarray(
                    np.uint32(internal.collision_x_bits & 0xFFFFFFFF)).view(np.float32).item()
                collision_y[slot] = np.asarray(
                    np.uint32(internal.collision_y_bits & 0xFFFFFFFF)).view(np.float32).item()
                cache_label = "exact cached position (float32 bits) from diagnostic stream"
            else:
                collision_x[slot] = internal.collision_q_x / PosQ
                collision_y[slot] = internal.collision_q_y / PosQ
                cache_label = "exact cached position (quantised) from diagnostic stream"
            collision_present[slot] = True
            note_by_slot[slot].collision_cache_recovery = cache_label

    s = s.replace(
        t_ms=jnp.asarray(snapshot.t_ms, dtype), tick=s.tick,
        kind=jnp.asarray(kind), team=jnp.asarray(team),
        alive=jnp.asarray(alive), model=jnp.asarray(model),
        spawn_seq=jnp.asarray(spawn_seq),
        x=jnp.asarray(x, dtype), y=jnp.asarray(y, dtype),
        # Diagnostic traces use the exact server cache.  Legacy traces use the
        # explicitly labelled temporal proxy above.
        collision_x=jnp.asarray(collision_x, dtype),
        collision_y=jnp.asarray(collision_y, dtype),
        collision_present=jnp.asarray(collision_present),
        waypoints=jnp.asarray(waypoints, dtype),
        waypoint_key=jnp.asarray(waypoint_key),
        lane_waypoint_key=jnp.asarray(lane_waypoint_key),
        n_waypoints=jnp.asarray(n_waypoints),
        move_order=jnp.asarray(move_order),
        hp=jnp.asarray(hp, dtype), max_hp=jnp.asarray(max_hp, dtype),
        spawn_x=jnp.asarray(spawn_x, dtype), spawn_y=jnp.asarray(spawn_y, dtype),
        level=jnp.asarray(level), xp=jnp.asarray(xp, dtype),
        gold=jnp.asarray(gold, dtype),
        cs=jnp.asarray(cs), deaths=jnp.asarray(deaths),
        spell_level=jnp.asarray(spell_level),
        spell_cooldown=jnp.asarray(spell_cooldown, dtype),
        target=jnp.asarray(target),
        is_attacking=jnp.asarray(is_attacking),
        has_auto_attacked=jnp.asarray(has_auto_attacked),
        aa_cooldown=jnp.asarray(aa_cooldown, dtype),
        aa_windup=jnp.asarray(aa_windup, dtype),
        buffs=jax.tree.map(
            lambda a: jnp.asarray(a, dtype if a.dtype.kind == "f" else a.dtype),
            buffs),
        ai_timer=jnp.asarray(ai_timer, dtype),
        target_priority=jnp.asarray(target_priority),
        had_target=jnp.asarray(had_target, dtype=bool),
        ignore_until=jnp.asarray(ignore_until, dtype),
        help_priority=jnp.asarray(help_priority),
        ai_local_time=jnp.asarray(ai_local_time, dtype),
        time_since_attack=jnp.asarray(time_since_attack, dtype),
        missile_alive=jnp.asarray(missile_alive),
        missile_x=jnp.asarray(missile_x, dtype),
        missile_y=jnp.asarray(missile_y, dtype),
        missile_tx=jnp.asarray(missile_tx),
        missile_source=jnp.asarray(missile_source),
        missile_damage=jnp.asarray(missile_damage, dtype),
        missile_speed=jnp.asarray(missile_speed, dtype),
        # wave spawner: derived exactly, see replay_wave_states
        next_spawn_ms=jnp.asarray(wave_state.next_spawn_ms, dtype),
        minion_number=jnp.asarray(wave_state.minion_number, jnp.int32),
        cannon_count=jnp.asarray(wave_state.cannon_count, jnp.int32),
        next_spawn_seq=jnp.asarray(len(creation_rank), jnp.int32),
        # Remaining minion bookkeeping and attack cooldown/last-hit state are
        # unrecoverable defaults.  UnitInjectionNote records the narrower
        # target/windup facts set above, so a default never reads as evidence.
    )
    return s, report
