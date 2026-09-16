"""Tier 1: the one-step differential itself.

For every recorded server tick N: inject the server's state at N into a
:class:`~lanerl_jax.sim.state.LaneState` (:mod:`lanerl_jax.parity.inject`),
step the sim exactly one tick, and diff the result against the server's own
state at N+1. No accumulation -- every comparison starts from ground truth,
so a disagreement is attributable to that tick's mechanics alone (see
``docs/ONE_STEP_DIFFERENTIAL.md`` for the full write-up and the honesty
caveats ``lanerl_jax.parity.inject`` documents in detail).

Recording
---------
:func:`record_idle_trace` boots one server, issues **no orders at all**
(``env.step([None])``) with bots off, and dumps full per-tick state. That is
deliberately a different fixture from :mod:`lanerl_jax.parity.record`'s
scripted drive: no orders means the trace is fully deterministic and
policy-free, and -- because champions never move or fight when nothing
controls them -- it isolates the minion/turret/wave-spawner mechanics this
module is built to test, with the champion side mostly reduced to "does HP
regen exist" (it does on the server, and not at all in the sim -- see
``docs/TICK_PARITY_AUDIT.md`` Gap 1).

Two matching passes, not one
-----------------------------
Entity correspondence has to be *recovered* (the dump strips NetId, see
``parity.diff``), and this module needs it twice, for two different
questions, and conflating them would blur a clean result:

1. **Death/spawn**: did the same units that existed at tick N still exist at
   N+1, on both sides? Matched by each entity's *pre-tick* (injected, ground
   truth) position against the server's real N+1 population. A unit with no
   match on the server side is dead there; whether the sim marked the same
   slot dead is read directly off its output, not by another position match.
2. **Field accuracy** (position, HP, move order, waypoint count): for units
   both sides agree survived, how far off is the sim's one-step prediction?
   Matched by *post-tick* position, sim's predicted output against the
   server's real N+1 entities.

Reusing one pass for both would silently score a death disagreement as a
"missing" position sample or vice versa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import numpy as np

from ..sim.init import TOP_LANE_PATH
from ..sim.profiles import PROFILES
from ..sim.state import Kind, LaneState
from ..sim.step import tick
from .diff import LANE_KINDS, _match_group
from .inject import UnitInjectionNote, inject_snapshot, replay_wave_states
from .sim_vs_server import state_to_snapshot
from .trace import Entity, PosQ, Snapshot, StatQ

__all__ = [
    "SEED", "GAME_SECONDS", "record_idle_trace",
    "MatchedPair", "DeathEvent", "SpawnEvent", "FieldStats",
    "OneStepResult", "run_one_step_differential",
]

SEED = 4242
GAME_SECONDS = 200.0
DECISIONS_PER_S = 30          # step_ticks=2 at 60 Hz -> 30 decisions/s
#: same match radius `diff.DEFAULT_TOLERANCE` uses: 8 world units, well above
#: one tick of travel for anything in this slice (a champion at base speed
#: moves 5.75 units/tick; nothing here is faster).
MATCH_RADIUS_Q = 16 * 8
#: HP quantisation unit, in real HP -- used to call a signed HP error
#: "float-rounding-sized" vs "a real disagreement".
HP_Q_UNIT = 1.0 / StatQ
POS_Q_UNIT = 1.0 / PosQ

#: `tick()` has fixed shapes throughout (LaneState's whole point, per D2 in
#: the rewrite plan), so it JIT-compiles once and every subsequent call in
#: the thousands-of-ticks loop below is a fast dispatch, not a retrace.
_tick_jit = jax.jit(tick)


def record_idle_trace(out_dir: Path, game_seconds: float = GAME_SECONDS,
                      port_base: int = 48765, seed: int = SEED) -> Path:
    """Boot one server, drive it with NO orders at all, return the log path.

    ``bot_teams="none"`` and every ``env.step`` call passes ``[None]`` --
    nothing ever issues a command, so the only things moving are the minion
    waves, the wave spawner's clock, and the turrets they eventually walk
    into range of. Fully deterministic and policy-free by construction.
    """
    from lanerl_train.ports import PortAllocator
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions = int(game_seconds * DECISIONS_PER_S)

    env = VecLaneEnv(
        1,
        spec=ServerLaunchSpec(
            toponly=True, bot_teams="none", bot_seed=seed, step_ticks=2,
            extra_env={"LANERL_STATE_DUMP": "1", "LANERL_STATE_DUMP_FULL": "1"},
        ),
        log_dir=out_dir / "server",
        ports=PortAllocator(base=port_base).allocate(1),
        step_timeout_s=180.0,
        auto_restart=False,
    )
    env.start()
    try:
        if not all(env.alive):
            raise RuntimeError(f"server failed to boot: {env.alive}")
        for _ in range(decisions):
            env.step([None])
        log = Path(env.handles[0].log_path)
    finally:
        env.close()
    return log


# --------------------------------------------------------------------------
# per-tick comparison
# --------------------------------------------------------------------------

@dataclass(slots=True)
class MatchedPair:
    """One entity alive on both sides after the step, matched by post-tick
    position."""

    kind: str
    team: int
    slot: int
    match_distance_q: float
    pred: Entity
    real: Entity
    movement_trustworthy: bool
    movement_reason: str
    pre_x: float
    pre_y: float


@dataclass(slots=True)
class DeathEvent:
    kind: str
    team: int
    slot: int
    sim_alive_after: bool
    server_alive_after: bool
    #: True when the server row still exists at N+1 but flagged dead (the
    #: removal-lag behaviour docs/TICK_PARITY_AUDIT.md Gap 4 describes),
    #: rather than fully absent.
    server_row_present_but_dead: bool = False


@dataclass(slots=True)
class SpawnEvent:
    kind: str
    team: int
    #: real entities at N+1 with no pre-tick predecessor anywhere nearby --
    #: i.e. genuinely new on the server this tick.
    n_real_new: int
    #: sim's own count of newly-alive slots of this (kind, team) this tick
    #: (population increase, not identity-matched -- see module docstring:
    #: identity for a brand-new spawn is not meaningfully recoverable, but
    #: the count is, and the wave schedule is deterministic).
    n_sim_new: int


@dataclass(slots=True)
class TickResult:
    t_ms: int
    matched: List[MatchedPair] = field(default_factory=list)
    deaths: List[DeathEvent] = field(default_factory=list)
    spawns: List[SpawnEvent] = field(default_factory=list)
    n_units_injected: int = 0
    n_untrustworthy_movement: int = 0


def _by_group(entities, key_fn) -> Dict[Tuple[str, int], List]:
    out: Dict[Tuple[str, int], List] = {}
    for e in entities:
        k = key_fn(e)
        if k is None:
            continue
        out.setdefault(k, []).append(e)
    return out


def compare_one_tick(state_n: LaneState, notes: List[UnitInjectionNote],
                     snap_n1: Snapshot, params: dict, lane_path) -> TickResult:
    """Step ``state_n`` (already injected from tick N) once and diff vs N+1."""
    out = TickResult(t_ms=int(state_n.t_ms))
    out.n_units_injected = len(notes)
    out.n_untrustworthy_movement = sum(1 for n in notes if not n.movement_trustworthy)
    note_by_slot = {n.slot: n for n in notes}

    pred_state = _tick_jit(state_n, params, lane_path=lane_path)
    pred_snapshot = state_to_snapshot(pred_state, t_ms=float(pred_state.t_ms),
                                      params=params)
    pred_kind = np.asarray(pred_state.kind)
    pred_alive = np.asarray(pred_state.alive)
    pred_slots = np.flatnonzero((pred_kind != Kind.NONE) & pred_alive)
    assert len(pred_slots) == len(pred_snapshot.entities)

    # ---- pass 1: death / spawn, matched by PRE-tick position ---------------
    real_all = _by_group(snap_n1.entities,
                         lambda e: (e.kind, e.team) if e.kind in LANE_KINDS
                         and e.team is not None else None)
    pre_by_group = _by_group(notes, lambda n: (n.kind, n.team))
    for key in set(pre_by_group) | set(real_all):
        pre_list = pre_by_group.get(key, [])
        real_list = real_all.get(key, [])
        left = [n.entity for n in pre_list]
        matched, only_l, only_r = _match_group(left, real_list, MATCH_RADIUS_Q)
        note_by_entity_id = {id(n.entity): n for n in pre_list}
        for a, b, _d in matched:
            n = note_by_entity_id[id(a)]
            server_alive = not b.dead
            out.deaths.append(DeathEvent(
                kind=key[0], team=key[1], slot=n.slot,
                sim_alive_after=bool(pred_alive[n.slot]),
                server_alive_after=server_alive,
                server_row_present_but_dead=b.dead,
            ))
        for a in only_l:
            n = note_by_entity_id[id(a)]
            out.deaths.append(DeathEvent(
                kind=key[0], team=key[1], slot=n.slot,
                sim_alive_after=bool(pred_alive[n.slot]),
                server_alive_after=False,
            ))
        out.spawns.append(SpawnEvent(
            kind=key[0], team=key[1], n_real_new=len(only_r),
            n_sim_new=0,   # filled in after pred_by_group is built, see below
        ))

    # sim's new-unit count per group: predicted alive count minus how many
    # pre-tick slots of that group survived (from the death pass above).
    pred_by_group_count = {}
    for slot, e in zip(pred_slots, pred_snapshot.entities):
        pred_by_group_count[(e.kind, e.team)] = pred_by_group_count.get(
            (e.kind, e.team), 0) + 1
    survivors_count: Dict[Tuple[str, int], int] = {}
    for d in out.deaths:
        if d.sim_alive_after:
            survivors_count[(d.kind, d.team)] = survivors_count.get(
                (d.kind, d.team), 0) + 1
    for sp in out.spawns:
        key = (sp.kind, sp.team)
        sp.n_sim_new = max(
            0, pred_by_group_count.get(key, 0) - survivors_count.get(key, 0))

    # ---- pass 2: field accuracy, matched by POST-tick position -------------
    # Real entities that are already dead-but-not-yet-swept (Gap 4) are
    # deliberately excluded here: pairing a surviving sim prediction against
    # a frozen dead row is not a position/HP sample, it is the death
    # disagreement pass 1 already recorded.
    real_alive_all = _by_group(
        snap_n1.entities,
        lambda e: (e.kind, e.team) if e.kind in LANE_KINDS
        and e.team is not None and not e.dead else None)
    pred_by_group: Dict[Tuple[str, int], List[Tuple[int, Entity]]] = {}
    for slot, e in zip(pred_slots, pred_snapshot.entities):
        pred_by_group.setdefault((e.kind, e.team), []).append((int(slot), e))
    for key in set(pred_by_group) | set(real_alive_all):
        plist = pred_by_group.get(key, [])
        rlist = real_alive_all.get(key, [])
        left = [e for _, e in plist]
        matched, _only_l, _only_r = _match_group(left, rlist, MATCH_RADIUS_Q)
        slot_by_id = {id(e): slot for slot, e in plist}
        for a, b, d in matched:
            slot = slot_by_id[id(a)]
            note = note_by_slot.get(slot)
            trust = True if note is None else note.movement_trustworthy
            reason = "freshly spawned this tick" if note is None else note.movement_reason
            pre_x = note.x if note is not None else float("nan")
            pre_y = note.y if note is not None else float("nan")
            out.matched.append(MatchedPair(
                kind=key[0], team=key[1], slot=slot, match_distance_q=d,
                pred=a, real=b, movement_trustworthy=trust,
                movement_reason=reason, pre_x=pre_x, pre_y=pre_y,
            ))
    return out


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

@dataclass(slots=True)
class FieldStats:
    name: str
    kind: str
    n_total: int = 0
    n_exact: int = 0
    errors: List[float] = field(default_factory=list)

    @property
    def n_diff(self) -> int:
        return self.n_total - self.n_exact

    @property
    def frac_exact(self) -> float:
        return self.n_exact / self.n_total if self.n_total else float("nan")

    def summary(self) -> str:
        if self.n_total == 0:
            return f"{self.kind}.{self.name}: no observations"
        s = (f"{self.kind}.{self.name}: {self.n_exact}/{self.n_total} exact "
            f"({100 * self.frac_exact:.2f}%)")
        if self.errors:
            e = np.asarray(self.errors)
            ae = np.abs(e)
            s += (f"; among the {len(e)} inexact, |error| median {np.median(ae):.4f} "
                 f"p95 {np.percentile(ae, 95):.4f} max {ae.max():.4f}"
                 f"; mean signed {e.mean():+.4f} (one-sided if far from 0)")
        return s


def _confusion_key(d: Dict[str, Dict[str, int]], kind: str) -> Dict[str, int]:
    return d.setdefault(kind, {"both_alive": 0, "sim_only_alive": 0,
                               "server_only_alive": 0, "both_dead": 0})


@dataclass(slots=True)
class OneStepResult:
    n_ticks: int = 0
    n_ticks_skipped: int = 0
    fields: Dict[Tuple[str, str], FieldStats] = field(default_factory=dict)
    death_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    hp_change_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    spawn_mismatches: Dict[str, int] = field(default_factory=dict)
    n_spawn_ticks: Dict[str, int] = field(default_factory=dict)
    n_untrustworthy_movement_ticks: int = 0
    n_units_seen: int = 0
    per_tick_worst_pos_error: List[Tuple[int, float]] = field(default_factory=list)
    gap4_candidates: int = 0
    #: MAJOR confound, see docs/ONE_STEP_DIFFERENTIAL.md: the dump carries a
    #: `SpellMissile` row (kind, x, y only -- no team/target/damage/speed), so
    #: a missile launched on an earlier tick and still in flight at the
    #: injected tick is invisible to the injector. `LaneState`'s own missile
    #: array is always injected empty. Every non-melee attack (caster/cannon
    #: minions, both outer turrets, `sim/missiles.py`) fires one, and a
    #: caster's is in flight ~0.85s (~51 ticks) -- more than half its attack
    #: cycle -- so this is not a rare edge case. Tabulated per tick so the
    #: confound is quantified, not just named.
    n_ticks_with_inflight_missile: int = 0
    n_ticks_with_inflight_missile_and_hp_diff: int = 0
    n_ticks_without_missile_and_hp_diff: int = 0

    def get(self, kind: str, name: str) -> FieldStats:
        key = (kind, name)
        if key not in self.fields:
            self.fields[key] = FieldStats(name=name, kind=kind)
        return self.fields[key]

    def rank_mechanics(self) -> List[Tuple[str, str, float, int, int]]:
        """Fields ranked by disagreement RATE (not absolute count), highest
        first: ``(kind, field, frac_diff, n_diff, n_total)``.

        Rate rather than raw count is the fair basis for ranking -- a field
        observed on 66 units/tick and one observed on 2 are not comparable by
        count alone. Fields excluded from movement's own trust gating
        (``position_untrustworthy_injection``) are kept in the ranking but
        should be read as "known-contaminated", not "mechanic X is broken";
        see ``docs/ONE_STEP_DIFFERENTIAL.md``.
        """
        rows = []
        for (kind, name), fs in self.fields.items():
            if fs.n_total == 0:
                continue
            rows.append((kind, name, fs.n_diff / fs.n_total, fs.n_diff, fs.n_total))
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows

    def report(self) -> str:
        lines = [
            f"{self.n_ticks} tick-pairs compared "
            f"({self.n_ticks_skipped} skipped for a time gap in the log)",
            f"units injected: {self.n_units_seen} total across all ticks; "
            f"{self.n_untrustworthy_movement_ticks} ticks had at least one unit "
            "whose movement injection was not trustworthy (see inject.py)",
            "",
            "-- field accuracy (post-tick position match; 'exact' = the dump's "
            "own quantised integers agree bit-for-bit) --",
        ]
        for key in sorted(self.fields):
            lines.append("  " + self.fields[key].summary())
        lines.append("")
        lines.append("-- death agreement (pre-tick position match) --")
        for kind, d in sorted(self.death_confusion.items()):
            total = sum(d.values())
            lines.append(
                f"  {kind}: both alive {d['both_alive']}, both dead "
                f"{d['both_dead']}, SIM ALIVE BUT SERVER DEAD "
                f"{d['sim_only_alive']}, SERVER ALIVE BUT SIM DEAD "
                f"{d['server_only_alive']}  (n={total})")
        lines.append(f"  server rows present-but-flagged-dead at N+1 "
                     f"(Gap 4 candidates): {self.gap4_candidates}")
        lines.append("")
        lines.append("-- wave spawning (population count per group, not identity) --")
        for kind, n in sorted(self.n_spawn_ticks.items()):
            mism = self.spawn_mismatches.get(kind, 0)
            lines.append(f"  {kind}: {n} ticks with a server spawn, "
                         f"{mism} where the sim's count disagreed")
        lines.append("")
        lines.append("-- HP-change (did it take damage this tick?) agreement --")
        for kind, d in sorted(self.hp_change_confusion.items()):
            total = sum(d.values())
            lines.append(
                f"  {kind}: both {d['both_alive']}, neither {d['both_dead']}, "
                f"sim-only {d['sim_only_alive']}, server-only "
                f"{d['server_only_alive']}  (n={total})")
        lines.append("")
        lines.append("-- missile confound (a SpellMissile row present at the "
                     "injected tick means a launch the injector cannot see) --")
        with_m = self.n_ticks_with_inflight_missile
        without_m = self.n_ticks - with_m
        rate_with = (self.n_ticks_with_inflight_missile_and_hp_diff / with_m
                    if with_m else float("nan"))
        rate_without = (self.n_ticks_without_missile_and_hp_diff / without_m
                        if without_m else float("nan"))
        lines.append(
            f"  ticks with >=1 missile in flight: {with_m}/{self.n_ticks} "
            f"({100 * with_m / self.n_ticks if self.n_ticks else 0:.1f}%)")
        lines.append(
            f"  of those, an HP disagreement also occurred: "
            f"{self.n_ticks_with_inflight_missile_and_hp_diff}/{with_m} "
            f"({100 * rate_with:.1f}%)")
        lines.append(
            f"  HP-disagreement rate on missile-free ticks: "
            f"{self.n_ticks_without_missile_and_hp_diff}/{without_m} "
            f"({100 * rate_without:.1f}%)")
        return "\n".join(lines)


def run_one_step_differential(
    trace, patch=None, max_pairs: Optional[int] = None,
) -> OneStepResult:
    """The whole Tier-1 pass over a recorded trace."""
    import jax.numpy as jnp

    from ..data.patch import load_patch
    from ..sim.init import lane_params

    patch = patch or load_patch()
    params = lane_params(patch)
    wave_states = replay_wave_states(trace)
    lane_path = jnp.asarray(np.asarray(TOP_LANE_PATH, np.float32))

    result = OneStepResult()
    n = len(trace)
    if max_pairs is not None:
        n = min(n, max_pairs + 1)

    for i in range(n - 1):
        snap_n, snap_n1 = trace[i], trace[i + 1]
        dt = snap_n1.t_ms - snap_n.t_ms
        if dt <= 0 or dt > 34:      # more than ~2 ticks: a gap in the log
            result.n_ticks_skipped += 1
            continue

        state_n, report = inject_snapshot(snap_n, wave_states[i], params, PROFILES)
        tr = compare_one_tick(state_n, report.notes, snap_n1, params, lane_path)
        result.n_ticks += 1
        result.n_units_seen += tr.n_units_injected
        if tr.n_untrustworthy_movement:
            result.n_untrustworthy_movement_ticks += 1

        worst_pos = 0.0
        for m in tr.matched:
            dx = m.pred.x - m.real.x
            dy = m.pred.y - m.real.y
            euclid = math.hypot(dx, dy)
            worst_pos = max(worst_pos, euclid)
            exact_pos = (m.pred.q_x == m.real.q_x) and (m.pred.q_y == m.real.q_y)

            field_name = "position" if m.movement_trustworthy else \
                "position_untrustworthy_injection"
            fs = result.get(m.kind, field_name)
            fs.n_total += 1
            fs.n_exact += int(exact_pos)
            if not exact_pos:
                fs.errors.append(euclid)

            if m.movement_trustworthy:
                mvx, mvy = m.pred.x - m.pre_x, m.pred.y - m.pre_y
                mag = math.hypot(mvx, mvy)
                if mag > 1e-6 and not math.isnan(m.pre_x):
                    hx, hy = mvx / mag, mvy / mag
                    # signed_along > 0: sim ended up further along its own
                    # heading than the server did this tick (sim "ahead");
                    # < 0: sim under-shot along that heading (sim "behind").
                    signed_along = dx * hx + dy * hy
                    fsl = result.get(m.kind, "position_along_heading")
                    fsl.n_total += 1
                    fsl.n_exact += int(abs(signed_along) < POS_Q_UNIT)
                    if abs(signed_along) >= POS_Q_UNIT:
                        fsl.errors.append(signed_along)

            fs_hp = result.get(m.kind, "hp")
            fs_hp.n_total += 1
            exact_hp = m.pred.q_hp == m.real.q_hp
            fs_hp.n_exact += int(exact_hp)
            if not exact_hp:
                fs_hp.errors.append((m.pred.hp or 0.0) - (m.real.hp or 0.0))

            if m.pred.ai is not None and m.real.ai is not None:
                fs_mo = result.get(m.kind, "move_order")
                fs_mo.n_total += 1
                fs_mo.n_exact += int(m.pred.ai.move_order == m.real.ai.move_order)
                if m.movement_trustworthy:
                    fs_wp = result.get(m.kind, "waypoints")
                    fs_wp.n_total += 1
                    fs_wp.n_exact += int(m.pred.ai.waypoints == m.real.ai.waypoints)

        # HP-change event agreement, using the injected (pre-tick) HP as the
        # shared baseline for both sides.
        pre_hp_by_slot = {n.slot: n.entity.hp for n in report.notes}
        any_hp_diff_this_tick = False
        for m in tr.matched:
            pre_hp = pre_hp_by_slot.get(m.slot)
            if pre_hp is None:
                continue
            sim_dmg = (m.pred.hp or 0.0) < pre_hp - HP_Q_UNIT
            srv_dmg = (m.real.hp or 0.0) < pre_hp - HP_Q_UNIT
            d = _confusion_key(result.hp_change_confusion, m.kind)
            if sim_dmg and srv_dmg:
                d["both_alive"] += 1
            elif sim_dmg and not srv_dmg:
                d["sim_only_alive"] += 1
                any_hp_diff_this_tick = True
            elif srv_dmg and not sim_dmg:
                d["server_only_alive"] += 1
                any_hp_diff_this_tick = True
            else:
                d["both_dead"] += 1

        # Missile confound: a `SpellMissile` row in the INJECTED snapshot
        # means a missile launched on an earlier tick is still in flight,
        # invisible to the injector (see OneStepResult's docstring comment).
        has_missile = any(e.kind == "SpellMissile" for e in snap_n.entities)
        if has_missile:
            result.n_ticks_with_inflight_missile += 1
            if any_hp_diff_this_tick:
                result.n_ticks_with_inflight_missile_and_hp_diff += 1
        elif any_hp_diff_this_tick:
            result.n_ticks_without_missile_and_hp_diff += 1

        for de in tr.deaths:
            d = _confusion_key(result.death_confusion, de.kind)
            if de.sim_alive_after and de.server_alive_after:
                d["both_alive"] += 1
            elif de.sim_alive_after and not de.server_alive_after:
                d["sim_only_alive"] += 1
            elif de.server_alive_after and not de.sim_alive_after:
                d["server_only_alive"] += 1
            else:
                d["both_dead"] += 1
            if de.server_row_present_but_dead:
                result.gap4_candidates += 1

        for sp in tr.spawns:
            if sp.n_real_new == 0 and sp.n_sim_new == 0:
                continue
            result.n_spawn_ticks[sp.kind] = result.n_spawn_ticks.get(sp.kind, 0) + 1
            if sp.n_real_new != sp.n_sim_new:
                result.spawn_mismatches[sp.kind] = (
                    result.spawn_mismatches.get(sp.kind, 0) + 1)

        result.per_tick_worst_pos_error.append((snap_n.t_ms, worst_pos))

    return result
