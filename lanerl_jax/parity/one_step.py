"""Tier 1: the one-step differential itself.

For every recorded server tick N: inject the server's state at N into a
:class:`~lanerl_jax.sim.state.LaneState` (:mod:`lanerl_jax.parity.inject`),
step the sim exactly one tick, and diff the result against the server's own
state at N+1. No accumulation -- every comparison starts from ground truth, so
a disagreement is attributable to that tick's mechanics when the required reset
fields are present (see ``docs/ONE_STEP_DIFFERENTIAL.md`` for the full write-up
and the honesty caveats ``lanerl_jax.parity.inject`` documents in detail).
Canonical-only traces retain labelled recovery gaps; `LANERL_STATE_DUMP_INTERNALS=1`
adds diagnostic target/AA/AI/missile/waypoint/collision-cache fields outside the
canonical hash.

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

Three passes, not one
---------------------
Entity correspondence has to be *recovered* (the dump strips NetId, see
``parity.diff``), and this module needs it three times, for three different
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
3. **Post-step controller state** (target identity, auto-attack fire tick):
   for units both sides kept alive, does the sim hold the same target NetId
   and swing on the same tick? Keyed on diagnostic NetId, never on proximity
   -- a target is an identity, and proximity cannot distinguish "kept the
   incumbent" from "acquired whatever is standing there now".

Reusing one pass for these would silently score a death disagreement as a
"missing" position sample or vice versa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import jax
import numpy as np

from ..sim.init import TOP_LANE_PATH
from ..sim.orders import Orders, apply_orders
from ..sim.profiles import PROFILES
from ..sim.state import Kind, LaneState
from ..sim.step import tick
from .diff import LANE_KINDS, _match_group
from .inject import UnitInjectionNote, inject_snapshot, replay_wave_states
from .sim_vs_server import state_to_snapshot
from .trace import Entity, PosQ, Snapshot, StatQ

__all__ = [
    "SEED", "GAME_SECONDS", "record_idle_trace",
    "MatchedPair", "DeathEvent", "SpawnEvent", "ControllerPair", "FieldStats",
    "OneStepResult", "merge_one_step_results", "run_one_step_differential",
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
#: J1 gate 1's own position target (`docs/JAX_REWRITE_PLAN.md` §3): "unit
#: position <= 1/16 unit (the dump's own quantisation) after one step" -- NOT
#: bit-exact. Bit-exact (`q_x == q_x and q_y == q_y`, tracked as the "position"
#: / "position_untrustworthy_injection" fields below) is a strictly stronger
#: criterion that the gate never asked for: the dump itself cannot represent a
#: difference finer than one quantum, so a one-quantum miss is not evidence of
#: anything the sim got wrong. `POS_GATE_SLACK` absorbs float32 rounding at
#: this coordinate magnitude (~1e4 units, float32 ULP there is ~1e-3) without
#: loosening the criterion by anything that would matter to a real miss.
POS_GATE_TOL = POS_Q_UNIT
POS_GATE_SLACK = 1e-2
#: how many concrete mismatching cases each controller comparison keeps.  A
#: rate says a field disagrees; only a case says what it disagreed about, and
#: these are cheap enough to retain unconditionally.
EXAMPLE_LIMIT = 12

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
            extra_env={"LANERL_STATE_DUMP": "1", "LANERL_STATE_DUMP_FULL": "1",
                       "LANERL_STATE_DUMP_INTERNALS": "1"},
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
class ControllerPair:
    """Post-step controller state for one NetId-identified unit.

    The injector restores target identity and the auto-attack clock from the
    diagnostic stream (RESET-001/RESET-002), but restoring a field and
    *scoring* it are different things: until this pair existed, a one-step run
    could retarget every unit wrongly and still report a clean sheet, because
    nothing compared the post-step values at all.

    Both sides are read against the SAME pre-tick baseline -- the injected
    ``LaneState``, whose AA clock and target came bit-for-bit off the server's
    tick-N internal line -- so "did this unit start a swing this tick" is the
    same question on both sides rather than two differently-derived events.
    """

    kind: str
    team: int
    slot: int
    net_id: int
    #: ``None`` when the sim targets a slot that has no tick-N NetId (a unit
    #: the sim spawned this tick).  Scored as unmappable, never as a miss.
    sim_target_net_id: Optional[int]
    server_target_net_id: int
    sim_target_label: str
    server_target_label: str
    #: a swing *started* this tick: the auto-attack cooldown was re-armed,
    #: which only ``AutoAttackSpell.Cast`` does (`ObjAIBase.UpdateTarget`).
    sim_fire: bool
    server_fire: bool
    #: the swing's damage *landed* this tick: ``HasAutoAttacked`` went false
    #: -> true, which only ``AutoAttackHit`` does.
    sim_hit: bool
    server_hit: bool
    sim_attacking: bool
    server_attacking: bool
    sim_has_auto_attacked: bool
    server_has_auto_attacked: bool
    sim_aa_cooldown_q: int
    server_aa_cooldown_q: int


@dataclass(slots=True)
class TickResult:
    t_ms: int
    matched: List[MatchedPair] = field(default_factory=list)
    deaths: List[DeathEvent] = field(default_factory=list)
    spawns: List[SpawnEvent] = field(default_factory=list)
    controller: List[ControllerPair] = field(default_factory=list)
    n_units_injected: int = 0
    n_untrustworthy_movement: int = 0
    identity_mode: str = "legacy proximity"


def _by_group(entities, key_fn) -> Dict[Tuple[str, int], List]:
    out: Dict[Tuple[str, int], List] = {}
    for e in entities:
        k = key_fn(e)
        if k is None:
            continue
        out.setdefault(k, []).append(e)
    return out


def _q_away_from_zero(v: float, scale: float) -> int:
    """The server's own quantiser (``LanerlStateDump.Q``).

    ``Math.Round(v * scale, MidpointRounding.AwayFromZero)`` -- NOT numpy's
    banker's rounding, which would disagree on exact halves and turn a
    matching clock into a one-quantum "disagreement".
    """
    x = float(v) * scale
    return int(math.floor(abs(x) + 0.5)) * (1 if x >= 0 else -1)


def _compare_controller(
    state_n: LaneState, pred_state: LaneState, snap_n1: Snapshot,
    real_by_net_id: Dict[int, Entity], pre_net_id_to_slot: Dict[int, int],
    note_by_slot: Dict[int, UnitInjectionNote], pred_alive,
) -> List[ControllerPair]:
    """Post-step target identity and auto-attack fire, per NetId.

    Only units that **both** sides kept alive are scored: a unit one side
    killed has no meaningful post-step target, and pass 1 already owns that
    disagreement.  Scoring it here as well would double-count one bug and
    contaminate a target-selection number with a death-timing one.
    """
    internals_n1 = {iv.net_id: iv for iv in snap_n1.ai_internals}
    slot_to_net_id = {slot: net_id for net_id, slot in pre_net_id_to_slot.items()}
    post_target = np.asarray(pred_state.target)
    post_attacking = np.asarray(pred_state.is_attacking)
    post_has_aa = np.asarray(pred_state.has_auto_attacked)
    post_cd = np.asarray(pred_state.aa_cooldown)
    pre_cd = np.asarray(state_n.aa_cooldown)
    pre_has_aa = np.asarray(state_n.has_auto_attacked)

    out: List[ControllerPair] = []
    for net_id, slot in pre_net_id_to_slot.items():
        iv = internals_n1.get(net_id)
        note = note_by_slot.get(slot)
        real = real_by_net_id.get(net_id)
        if iv is None or note is None or real is None or real.dead:
            continue
        if not bool(pred_alive[slot]):
            continue

        sim_slot = int(post_target[slot])
        if sim_slot < 0:
            sim_target_net_id: Optional[int] = 0
            sim_label = "-"
        else:
            sim_target_net_id = slot_to_net_id.get(sim_slot)
            tnote = note_by_slot.get(sim_slot)
            sim_label = tnote.kind if tnote is not None else "spawned-this-tick"

        pre_cd_q = _q_away_from_zero(pre_cd[slot], StatQ)
        sim_cd_q = _q_away_from_zero(post_cd[slot], StatQ)
        out.append(ControllerPair(
            kind=note.kind, team=note.team, slot=slot, net_id=net_id,
            sim_target_net_id=sim_target_net_id,
            server_target_net_id=iv.target_net_id,
            sim_target_label=sim_label,
            server_target_label=iv.target_kind,
            # A re-armed cooldown is the server's own swing-start signature:
            # `_autoAttackCurrentCooldown` only ever decreases (Update) or is
            # zeroed (CancelAutoAttack) except in the `AutoAttackSpell.Cast`
            # branch, which sets it to `1 / GetTotalAttackSpeed()`.
            sim_fire=sim_cd_q > pre_cd_q,
            server_fire=iv.q_aa_cooldown > pre_cd_q,
            sim_hit=bool(post_has_aa[slot]) and not bool(pre_has_aa[slot]),
            server_hit=iv.has_auto_attacked and not bool(pre_has_aa[slot]),
            sim_attacking=bool(post_attacking[slot]),
            server_attacking=bool(iv.is_attacking),
            sim_has_auto_attacked=bool(post_has_aa[slot]),
            server_has_auto_attacked=bool(iv.has_auto_attacked),
            sim_aa_cooldown_q=sim_cd_q,
            server_aa_cooldown_q=iv.q_aa_cooldown,
        ))
    return out


def compare_one_tick(state_n: LaneState, notes: List[UnitInjectionNote],
                     snap_n1: Snapshot, params: dict, lane_path,
                     endpoint_orders: Optional[Orders] = None,
                     pre_net_id_to_slot: Optional[Dict[int, int]] = None, *,
                     route_table=None, terrain=None) -> TickResult:
    """Step injected N once, apply an endpoint action, and diff vs N+1.

    The endpoint placement is not arbitrary: ``LanerlControl`` executes an
    action immediately before the state dump at that control boundary, after
    the preceding object's tick.  See :mod:`lanerl_jax.parity.action_replay`.
    """
    out = TickResult(t_ms=int(state_n.t_ms))
    out.n_units_injected = len(notes)
    out.n_untrustworthy_movement = sum(1 for n in notes if not n.movement_trustworthy)
    note_by_slot = {n.slot: n for n in notes}

    pred_state = _tick_jit(state_n, params, lane_path=lane_path)
    if endpoint_orders is not None:
        pred_state = apply_orders(
            pred_state, endpoint_orders, params,
            route_table=route_table, terrain=terrain)
    pred_snapshot = state_to_snapshot(pred_state, t_ms=float(pred_state.t_ms),
                                      params=params)
    pred_kind = np.asarray(pred_state.kind)
    pred_alive = np.asarray(pred_state.alive)
    pred_slots = np.flatnonzero((pred_kind != Kind.NONE) & pred_alive)
    assert len(pred_slots) == len(pred_snapshot.entities)

    # ---- pass 1: death / spawn ---------------------------------------------
    real_all = _by_group(snap_n1.entities,
                         lambda e: (e.kind, e.team) if e.kind in LANE_KINDS
                         and e.team is not None else None)
    pre_by_group = _by_group(notes, lambda n: (n.kind, n.team))
    from .diagnostic_identity import net_id_to_entity
    real_by_net_id = net_id_to_entity(snap_n1)
    pre_net_id_to_slot = pre_net_id_to_slot or {}
    identity_complete = (
        bool(notes) and len(pre_net_id_to_slot) == len(notes)
        and bool(snap_n1.ai_internals))
    if identity_complete:
        out.identity_mode = "diagnostic NetId"
        pre_ids = set(pre_net_id_to_slot)
        for net_id, slot in pre_net_id_to_slot.items():
            n = note_by_slot[slot]
            b = real_by_net_id.get(net_id)
            out.deaths.append(DeathEvent(
                kind=n.kind, team=n.team, slot=slot,
                sim_alive_after=bool(pred_alive[slot]),
                server_alive_after=b is not None and not b.dead,
                server_row_present_but_dead=b is not None and bool(b.dead),
            ))
        new_real_by_group = _by_group(
            [entity for net_id, entity in real_by_net_id.items()
             if net_id not in pre_ids],
            lambda e: (e.kind, e.team) if e.kind in LANE_KINDS
            and e.team is not None else None)
        for key in set(pre_by_group) | set(real_all):
            out.spawns.append(SpawnEvent(
                kind=key[0], team=key[1],
                n_real_new=len(new_real_by_group.get(key, [])), n_sim_new=0,
            ))
    else:
        # Legacy canonical dumps omit NetId.  Retain the historical bounded
        # proximity recovery, explicitly labelled; collision can legitimately
        # push a unit beyond this radius, so it is never used when IDs exist.
        for key in set(pre_by_group) | set(real_all):
            pre_list = pre_by_group.get(key, [])
            real_list = real_all.get(key, [])
            left = [n.entity for n in pre_list]
            matched, only_l, only_r = _match_group(left, real_list, MATCH_RADIUS_Q)
            note_by_entity_id = {id(n.entity): n for n in pre_list}
            for a, b, _d in matched:
                n = note_by_entity_id[id(a)]
                out.deaths.append(DeathEvent(
                    kind=key[0], team=key[1], slot=n.slot,
                    sim_alive_after=bool(pred_alive[n.slot]),
                    server_alive_after=not b.dead,
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
                kind=key[0], team=key[1], n_real_new=len(only_r), n_sim_new=0,
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
    if identity_complete:
        slot_to_pred = {int(slot): entity
                        for slot, entity in zip(pred_slots, pred_snapshot.entities)}
        pre_ids = set(pre_net_id_to_slot)
        for net_id, slot in pre_net_id_to_slot.items():
            a, b = slot_to_pred.get(slot), real_by_net_id.get(net_id)
            if a is None or b is None or b.dead:
                continue
            key = (a.kind, a.team)
            # Diagnostic identity is allowed to cross the old 8-unit match
            # radius; retain distance only as a diagnostic value.
            d = math.hypot(a.q_x - b.q_x, a.q_y - b.q_y)
            note = note_by_slot.get(slot)
            trust = True if note is None else note.movement_trustworthy
            out.matched.append(MatchedPair(
                kind=key[0], team=key[1], slot=slot, match_distance_q=d,
                pred=a, real=b, movement_trustworthy=trust,
                movement_reason=("freshly spawned this tick" if note is None
                                 else note.movement_reason),
                pre_x=(float("nan") if note is None else note.x),
                pre_y=(float("nan") if note is None else note.y),
            ))
        # Newly spawned entities do not have a predecessor identity.  Match
        # only this new/new remainder by proximity.
        new_pred = [(slot, entity) for slot, entity in zip(pred_slots, pred_snapshot.entities)
                    if int(slot) not in note_by_slot]
        new_real = [entity for net_id, entity in real_by_net_id.items()
                    if net_id not in pre_ids and not entity.dead]
        for key in set(_by_group(new_pred, lambda p: (p[1].kind, p[1].team))) | set(
                _by_group(new_real, lambda e: (e.kind, e.team))):
            plist = [p for p in new_pred if (p[1].kind, p[1].team) == key]
            rlist = [e for e in new_real if (e.kind, e.team) == key]
            matched, _, _ = _match_group([e for _, e in plist], rlist, MATCH_RADIUS_Q)
            slot_by_entity = {id(e): int(slot) for slot, e in plist}
            for a, b, d in matched:
                slot = slot_by_entity[id(a)]
                out.matched.append(MatchedPair(
                    kind=key[0], team=key[1], slot=slot, match_distance_q=d,
                    pred=a, real=b, movement_trustworthy=True,
                    movement_reason="freshly spawned this tick",
                    pre_x=float("nan"), pre_y=float("nan")))
    else:
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

    # ---- pass 3: post-step controller state ---------------------------------
    # Needs NetIds on both sides: a target is an *identity*, and proximity
    # recovery cannot tell "kept the same target" from "acquired the unit
    # standing where the old one was".  Legacy traces are left unscored rather
    # than scored approximately.
    if identity_complete:
        out.controller = _compare_controller(
            state_n, pred_state, snap_n1, real_by_net_id, pre_net_id_to_slot,
            note_by_slot, pred_alive)
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
    #: Injection provenance, accumulated rather than discarded.  The one-step
    #: scores below are only interpretable alongside this ledger: a green score
    #: must not be read as confirmation of a hidden field we defaulted.
    recovery_counts: Dict[str, int] = field(default_factory=dict)
    #: Driven fixtures only: one boundary contains two semantic side orders.
    n_action_boundaries: int = 0
    action_kinds: Dict[str, int] = field(default_factory=dict)
    n_diagnostic_identity_ticks: int = 0
    n_legacy_proximity_identity_ticks: int = 0
    #: Post-step controller scoring (see :class:`ControllerPair`).  Separate
    #: counters because these are scored on a strictly smaller population than
    #: the field-accuracy pairs: diagnostic-identity ticks only, and only for
    #: units both sides kept alive.
    n_controller_ticks: int = 0
    n_controller_pairs: int = 0
    #: the sim targeted a slot with no tick-N NetId (a unit it spawned this
    #: tick), so the comparison is not expressible.  Never scored as a miss.
    n_target_unmappable: int = 0
    #: mismatch breakdowns, keyed so the *shape* of a disagreement is visible
    #: without reading the examples: a percentage cannot distinguish "targets
    #: the wrong minion" from "holds a target where the server has none".
    target_mismatch_kinds: Dict[str, int] = field(default_factory=dict)
    target_mismatch_examples: List[str] = field(default_factory=list)
    aa_fire_mismatch_kinds: Dict[str, int] = field(default_factory=dict)
    aa_fire_mismatch_examples: List[str] = field(default_factory=list)
    #: How often each side's boolean was TRUE at all.  Without this a 100%
    #: agreement rate is unreadable: "never fires, and neither does the
    #: server" and "fires 4,000 times on the same ticks" score identically,
    #: and only one of them is evidence.
    controller_event_counts: Dict[str, int] = field(default_factory=dict)

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
        lines.append("-- position: trustworthy vs untrustworthy injection, "
                     "bit-exact vs Euclidean and componentwise <=1/16 --")
        lines.append("   x/y are quantised independently; componentwise/L-inf "
                     "is the identifiable dump-resolution gate. Euclidean is "
                     "retained as a stricter historical diagnostic.")
        kinds = sorted({k for k, name in self.fields if name == "position"})
        for kind in kinds:
            t_exact = self.fields.get((kind, "position"))
            u_exact = self.fields.get((kind, "position_untrustworthy_injection"))
            t_tol = self.fields.get((kind, "position_le1_16"))
            u_tol = self.fields.get((kind, "position_le1_16_untrustworthy_injection"))
            t_linf = self.fields.get((kind, "position_linf_le1_16"))
            u_linf = self.fields.get(
                (kind, "position_linf_le1_16_untrustworthy_injection"))
            def pct(fs):
                return f"{100*fs.frac_exact:.2f}% ({fs.n_exact}/{fs.n_total})" \
                    if fs and fs.n_total else "n/a"
            lines.append(f"  {kind}:")
            lines.append(f"    bit-exact   trustworthy={pct(t_exact)}  "
                         f"untrustworthy={pct(u_exact)}")
            lines.append(f"    <=1/16      trustworthy={pct(t_tol)}  "
                         f"untrustworthy={pct(u_tol)}")
            lines.append(f"    L-inf<=1/16 trustworthy={pct(t_linf)}  "
                         f"untrustworthy={pct(u_linf)}")
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
        lines.append("-- missile confound (legacy traces without diagnostic "
                     "internals cannot inject a live missile) --")
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
        lines.append("")
        lines.append("-- hidden-state injection provenance (not agreement) --")
        if self.recovery_counts:
            for label, count in sorted(self.recovery_counts.items()):
                lines.append(f"  {count}: {label}")
        else:
            lines.append("  no injections")
        if self.n_action_boundaries:
            lines.append("")
            lines.append("-- recorded endpoint actions replayed --")
            lines.append(f"  boundaries: {self.n_action_boundaries}")
            for kind, count in sorted(self.action_kinds.items()):
                lines.append(f"  {kind}: {count}")
        lines.append("")
        lines.append("-- post-step controller state (diagnostic NetId ticks "
                     "only; units both sides kept alive) --")
        lines.append(f"  scored on {self.n_controller_pairs} unit-ticks over "
                     f"{self.n_controller_ticks} ticks")
        if self.n_controller_pairs:
            for name in ("target", "aa_fire", "aa_hit", "is_attacking",
                         "has_auto_attacked", "aa_cooldown"):
                for kind in sorted({k for k, n in self.fields if n == name}):
                    lines.append("  " + self.fields[(kind, name)].summary())
            lines.append("  base rates (how often each boolean was true at "
                         "all -- a 100% agreement on an event that never "
                         "happens is not evidence):")
            for key in sorted(self.controller_event_counts):
                lines.append(f"    {key}: {self.controller_event_counts[key]}")
            lines.append(f"  sim target unmappable (targeted a slot spawned "
                         f"this tick): {self.n_target_unmappable}")
            if self.target_mismatch_kinds:
                lines.append("  target mismatches by shape:")
                for key, n in sorted(self.target_mismatch_kinds.items(),
                                     key=lambda kv: -kv[1]):
                    lines.append(f"    {n}: {key}")
            for line in self.target_mismatch_examples[:EXAMPLE_LIMIT]:
                lines.append(f"    e.g. {line}")
            if self.aa_fire_mismatch_kinds:
                lines.append("  auto-attack fire-tick mismatches by shape:")
                for key, n in sorted(self.aa_fire_mismatch_kinds.items(),
                                     key=lambda kv: -kv[1]):
                    lines.append(f"    {n}: {key}")
            for line in self.aa_fire_mismatch_examples[:EXAMPLE_LIMIT]:
                lines.append(f"    e.g. {line}")
        lines.append("")
        lines.append("-- entity correspondence provenance --")
        lines.append(f"  diagnostic NetId: {self.n_diagnostic_identity_ticks} ticks")
        lines.append(
            f"  legacy proximity fallback: "
            f"{self.n_legacy_proximity_identity_ticks} ticks")
        return "\n".join(lines)


def merge_one_step_results(parts: Iterable[OneStepResult]) -> OneStepResult:
    """Combine disjoint one-step shards without changing serial semantics.

    Callers must provide shards in increasing tick-range order.  Scalar counts
    are additive, field error samples retain serial tick order, and the final
    worst-position series is sorted defensively by timestamp.  The latter
    makes the result deterministic even when callers collect futures in
    completion order.
    """
    out = OneStepResult()
    for part in parts:
        out.n_ticks += part.n_ticks
        out.n_ticks_skipped += part.n_ticks_skipped
        out.n_untrustworthy_movement_ticks += part.n_untrustworthy_movement_ticks
        out.n_units_seen += part.n_units_seen
        out.gap4_candidates += part.gap4_candidates
        out.n_ticks_with_inflight_missile += part.n_ticks_with_inflight_missile
        out.n_ticks_with_inflight_missile_and_hp_diff += (
            part.n_ticks_with_inflight_missile_and_hp_diff)
        out.n_ticks_without_missile_and_hp_diff += (
            part.n_ticks_without_missile_and_hp_diff)
        out.n_action_boundaries += part.n_action_boundaries
        out.n_diagnostic_identity_ticks += part.n_diagnostic_identity_ticks
        out.n_legacy_proximity_identity_ticks += (
            part.n_legacy_proximity_identity_ticks)
        out.n_controller_ticks += part.n_controller_ticks
        out.n_controller_pairs += part.n_controller_pairs
        out.n_target_unmappable += part.n_target_unmappable
        out.target_mismatch_examples.extend(part.target_mismatch_examples)
        out.aa_fire_mismatch_examples.extend(part.aa_fire_mismatch_examples)

        for key, src in part.fields.items():
            dst = out.fields.get(key)
            if dst is None:
                dst = FieldStats(name=src.name, kind=src.kind)
                out.fields[key] = dst
            dst.n_total += src.n_total
            dst.n_exact += src.n_exact
            dst.errors.extend(src.errors)

        for attr in (
            "death_confusion", "hp_change_confusion",
        ):
            dst_groups = getattr(out, attr)
            for kind, src_counts in getattr(part, attr).items():
                dst_counts = _confusion_key(dst_groups, kind)
                for key, value in src_counts.items():
                    dst_counts[key] += value

        for attr in (
            "spawn_mismatches", "n_spawn_ticks", "recovery_counts", "action_kinds",
            "target_mismatch_kinds", "aa_fire_mismatch_kinds",
            "controller_event_counts",
        ):
            dst_counts = getattr(out, attr)
            for key, value in getattr(part, attr).items():
                dst_counts[key] = dst_counts.get(key, 0) + value

        out.per_tick_worst_pos_error.extend(part.per_tick_worst_pos_error)

    out.per_tick_worst_pos_error.sort(key=lambda row: row[0])
    return out


def run_one_step_differential(
    trace, patch=None, max_pairs: Optional[int] = None, *,
    pair_start: int = 0, pair_stop: Optional[int] = None,
    progress_every: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    action_log=None,
    route_table=None,
    terrain=None,
) -> OneStepResult:
    """Run Tier 1 over a whole trace or a contiguous pair-index shard.

    ``pair_start`` is inclusive and ``pair_stop`` exclusive.  They index tick
    *pairs* in the original trace, so hidden-state recovery can still inspect
    the real preceding snapshot and wave replay still starts at episode zero.
    This property is what makes independently evaluated shards equivalent to
    one serial pass.

    ``progress_callback(done, total)`` runs after each ``progress_every``
    attempted pairs (and once at the end).  It is observational only and does
    not receive mutable result state.
    """
    import jax.numpy as jnp

    from ..data.patch import load_patch
    from ..sim.init import lane_params

    patch = patch or load_patch()
    params = lane_params(patch)
    if max_pairs is not None and pair_start == 0:
        # `max_pairs` bounded the WORK but not the MEMORY: both
        # `replay_wave_states` and `align_action_log` below walk the whole
        # trace, and they ran before the cap was applied to the loop bound.
        # That is survivable on the 115 MB idle fixture gate 1 has always used
        # and is not on a 537 MB driven one -- which OOM-killed a 28 GB job
        # with `--max-pairs 6000` set, i.e. the flag that exists to make a
        # trace affordable could not.
        #
        # Truncating here is exactly equivalent for a shard starting at pair 0,
        # because pairs beyond the cap are never evaluated. It is deliberately
        # NOT done for a non-zero `pair_start`: this function's contract is
        # that a shard still sees the real preceding snapshots, so that
        # hidden-state recovery and wave replay start from episode zero and
        # independently evaluated shards equal one serial pass.
        trace = list(trace)[:max_pairs + 1]
    wave_states = replay_wave_states(trace)
    lane_path = jnp.asarray(np.asarray(TOP_LANE_PATH, np.float32))
    action_at_snapshot = {}
    if action_log is not None:
        from .action_replay import align_action_log
        action_at_snapshot = align_action_log(trace, action_log)

    result = OneStepResult()
    n = len(trace)
    if max_pairs is not None:
        n = min(n, max_pairs + 1)
    available_pairs = max(0, n - 1)
    start = min(max(0, pair_start), available_pairs)
    stop = available_pairs if pair_stop is None else min(
        max(start, pair_stop), available_pairs)
    total_attempted = stop - start
    progress_every = progress_every if progress_every and progress_every > 0 else None

    for attempted, i in enumerate(range(start, stop), 1):
        snap_n, snap_n1 = trace[i], trace[i + 1]
        dt = snap_n1.t_ms - snap_n.t_ms
        if dt <= 0 or dt > 34:      # more than ~2 ticks: a gap in the log
            result.n_ticks_skipped += 1
            if progress_callback is not None and (
                    attempted == total_attempted or
                    (progress_every is not None and attempted % progress_every == 0)):
                progress_callback(attempted, total_attempted)
            continue

        previous = trace[i - 1] if i else None
        if previous is not None and not (0 < snap_n.t_ms - previous.t_ms <= 34):
            previous = None
        state_n, report = inject_snapshot(
            snap_n, wave_states[i], params, PROFILES, previous_snapshot=previous)
        for label, count in report.provenance_counts().items():
            result.recovery_counts[label] = result.recovery_counts.get(label, 0) + count
        endpoint_orders = None
        from .diagnostic_identity import net_id_to_injected_slot
        net_id_slots = net_id_to_injected_slot(snap_n, report.notes)
        decision = action_at_snapshot.get(i + 1)
        if decision is not None:
            from .action_replay import decision_to_orders
            endpoint_orders = decision_to_orders(
                decision, net_id_slots)
            result.n_action_boundaries += 1
            for wire in (decision.blue, decision.red):
                kind = str(wire.get("t", "noop"))
                result.action_kinds[kind] = result.action_kinds.get(kind, 0) + 1
        tr = compare_one_tick(
            state_n, report.notes, snap_n1, params, lane_path,
            endpoint_orders=endpoint_orders, pre_net_id_to_slot=net_id_slots,
            route_table=route_table, terrain=terrain)
        if tr.identity_mode == "diagnostic NetId":
            result.n_diagnostic_identity_ticks += 1
        else:
            result.n_legacy_proximity_identity_ticks += 1
        result.n_ticks += 1
        result.n_units_seen += tr.n_units_injected
        if tr.n_untrustworthy_movement:
            result.n_untrustworthy_movement_ticks += 1

        # Missile confound, computed up front (not just at the bottom of the
        # loop where the HP-change bookkeeping already wants it): a
        # `SpellMissile` row in the INJECTED (pre-tick) snapshot means a
        # missile launched on an earlier tick is still in flight, invisible to
        # the injector -- see OneStepResult's docstring. Used below to keep
        # the one-sided-bias analysis (Sec3 task) restricted to ticks that
        # cannot be contaminated by that blind spot.
        has_missile = any(e.kind == "SpellMissile" for e in snap_n.entities)

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

            # The gate's own criterion (<= 1/16 unit), tracked SEPARATELY from
            # bit-exactness above -- see POS_GATE_TOL's docstring. Same
            # trustworthy/untrustworthy split, so the two criteria can be
            # compared side by side in the report.
            within_gate_tol = euclid <= POS_GATE_TOL + POS_GATE_SLACK
            field_name_tol = "position_le1_16" if m.movement_trustworthy else \
                "position_le1_16_untrustworthy_injection"
            fs_tol = result.get(m.kind, field_name_tol)
            fs_tol.n_total += 1
            fs_tol.n_exact += int(within_gate_tol)
            if not within_gate_tol:
                fs_tol.errors.append(euclid)

            # The wire rounds x and y independently. A one-bin miss on both
            # axes has Euclidean length sqrt(2)/16, but neither component is
            # distinguishable beyond the dump's own 1/16 resolution.
            linf = max(abs(dx), abs(dy))
            within_linf_tol = linf <= POS_GATE_TOL + POS_GATE_SLACK
            linf_name = ("position_linf_le1_16" if m.movement_trustworthy else
                         "position_linf_le1_16_untrustworthy_injection")
            fs_linf = result.get(m.kind, linf_name)
            fs_linf.n_total += 1
            fs_linf.n_exact += int(within_linf_tol)
            if not within_linf_tol:
                fs_linf.errors.append(linf)

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

                    # Same signal, restricted to ticks with NO in-flight
                    # missile at all -- the known blind spot (missiles are
                    # invisible to the injector, docs/TIER1_POST_REORDER.md
                    # task 3) cannot be what is producing this one, if it is
                    # still there.
                    if not has_missile:
                        fslm = result.get(m.kind, "position_along_heading_missile_free")
                        fslm.n_total += 1
                        fslm.n_exact += int(abs(signed_along) < POS_Q_UNIT)
                        if abs(signed_along) >= POS_Q_UNIT:
                            fslm.errors.append(signed_along)

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
        # `has_missile` was already computed above, before the field-accuracy
        # loop, so the bias analysis could use it too.
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

        # Post-step controller state.  The injector restores target identity
        # and the auto-attack clock (RESET-001/RESET-002); these are the
        # comparisons that make restoring them falsifiable.
        if tr.controller:
            result.n_controller_ticks += 1
            result.n_controller_pairs += len(tr.controller)
        for c in tr.controller:
            target_agrees = c.sim_target_net_id == c.server_target_net_id
            if c.sim_target_net_id is None:
                result.n_target_unmappable += 1
            else:
                fs_t = result.get(c.kind, "target")
                fs_t.n_total += 1
                fs_t.n_exact += int(target_agrees)
                if not target_agrees:
                    shape = (f"{c.kind}: sim->{c.sim_target_label} "
                             f"server->{c.server_target_label}")
                    result.target_mismatch_kinds[shape] = (
                        result.target_mismatch_kinds.get(shape, 0) + 1)
                    if len(result.target_mismatch_examples) < EXAMPLE_LIMIT:
                        result.target_mismatch_examples.append(
                            f"t={snap_n.t_ms} {c.kind}(team={c.team}, "
                            f"net={c.net_id}): server target "
                            f"{c.server_target_net_id}({c.server_target_label}) "
                            f"vs sim {c.sim_target_net_id}({c.sim_target_label})")

            for name, sim_v, srv_v in (
                ("target_held", c.sim_target_net_id not in (0, None),
                 c.server_target_net_id != 0),
                ("aa_fire", c.sim_fire, c.server_fire),
                ("aa_hit", c.sim_hit, c.server_hit),
                ("is_attacking", c.sim_attacking, c.server_attacking),
                ("has_auto_attacked", c.sim_has_auto_attacked,
                 c.server_has_auto_attacked),
                ("aa_cooldown", c.sim_aa_cooldown_q, c.server_aa_cooldown_q),
            ):
                if name != "target_held":
                    fs_c = result.get(c.kind, name)
                    fs_c.n_total += 1
                    fs_c.n_exact += int(sim_v == srv_v)
                    if name == "aa_cooldown" and sim_v != srv_v:
                        fs_c.errors.append((sim_v - srv_v) / StatQ)
                if isinstance(sim_v, bool):
                    for side, value in (("sim", sim_v), ("server", srv_v)):
                        if value:
                            k = f"{c.kind}.{name}.{side}"
                            result.controller_event_counts[k] = (
                                result.controller_event_counts.get(k, 0) + 1)

            if c.sim_fire != c.server_fire:
                shape = (f"{c.kind}: sim_fire={int(c.sim_fire)} "
                         f"server_fire={int(c.server_fire)}, "
                         f"target_agrees={int(bool(target_agrees))}, "
                         f"sim_attacking={int(c.sim_attacking)} "
                         f"server_attacking={int(c.server_attacking)}")
                result.aa_fire_mismatch_kinds[shape] = (
                    result.aa_fire_mismatch_kinds.get(shape, 0) + 1)
                if len(result.aa_fire_mismatch_examples) < EXAMPLE_LIMIT:
                    result.aa_fire_mismatch_examples.append(
                        f"t={snap_n.t_ms} {c.kind}(team={c.team}, "
                        f"net={c.net_id}): fire sim={int(c.sim_fire)} "
                        f"server={int(c.server_fire)}; aa_cooldown q "
                        f"sim={c.sim_aa_cooldown_q} "
                        f"server={c.server_aa_cooldown_q}; target sim="
                        f"{c.sim_target_net_id} server={c.server_target_net_id}")

        result.per_tick_worst_pos_error.append((snap_n.t_ms, worst_pos))

        if progress_callback is not None and (
                attempted == total_attempted or
                (progress_every is not None and attempted % progress_every == 0)):
            progress_callback(attempted, total_attempted)

    return result
