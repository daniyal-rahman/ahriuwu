"""Tier 1.5: a bounded, injected, free-running differential.

WHY THIS EXISTS
---------------
Two differential harnesses were already here and neither can answer "given the
*same* situation, do the two engines behave the same?".

* **Tier 1** (`one_step.py`, `tier1_full.py`) injects the server's real state,
  steps ONE tick, diffs, and then throws the sim's result away and re-injects.
  That is the tightest possible isolation of a single tick's logic, and it
  destroys exactly the compounding the open questions are about. It also reads
  an unobservable sub-quantum auto-attack cooldown as 0, which is how `AA-001`
  came to be stated as a 2.19x rate error against a true rollout effect of
  1.009x -- a per-pair injection artifact reported as a dynamics result.
* **Tier 2** (`tier2.py`, `tier2_batch.py`) free-runs both engines 600 s from a
  reset. It compounds, but it cannot attribute: by minute two the two engines
  are in different lane states and every later difference is conditioned on a
  divergence that happened much earlier.

This module is the missing middle. ONE state, injected once, then N decisions
of free running on both sides with an identical order stream, and the
divergence reported as a **curve** rather than a scalar.

THE SERVER CANNOT BE INJECTED. READ THIS BEFORE BELIEVING ANY NUMBER BELOW.
--------------------------------------------------------------------------
The brief for this harness said "inject one identical full state into both
engines". Half of that is not possible, and pretending otherwise would be the
exact failure this repo keeps paying for, so it is stated first.

`LanerlControl.OnTick` (`GameServerLib/Lanerl/LanerlControl.cs:110-166`) reads
one line per decision and `LanerlWire.Parse` recognises exactly two things: the
literal reset line `{"cmd":"reset"}`, and a per-champion action (move / attack
/ cast / recall / noop). There is **no** set-position, set-health, spawn or
load-state command anywhere in `GameServerLib/Lanerl`. The only state-writing
path is `LanerlEpisode.Reset`, and it is a reset in the strict sense: it
rewinds `GameTime` to 0, respawns both champions at their fountains at full
health, revives and full-heals every structure, and rewinds the wave spawner
to `NextSpawnTime = 90 s` with `_minionNumber = 0`. Its only degree of freedom
is `LANERL_FIRST_WAVE_MS`. So the set of server states reachable by command is
{the canonical t=0 state}, and an arbitrary mid-lane state is not in it.

WHAT IS DONE INSTEAD, AND WHY IT IS NOT FAKING IT
-------------------------------------------------
The server is driven **to** the state rather than loaded with it, and the state
it reaches is used as the shared start:

1. Record ONE driven fixture with `record.record_fixture` -- the per-tick state
   dump including the diagnostic internals stream, plus `ActionLog`, the exact
   order stream the server executed. `scripted_action` is deterministic in the
   decision index, so the fixture is a function of the build and nothing else.
2. Pick a start tick `D` by **predicate** (see :func:`select_start`).
3. Inject `trace[D]` into a `LaneState` with `inject.inject_snapshot`.
4. Free-run the sim forward N decisions, replaying the fixture's own recorded
   orders at their real decision boundaries, with **no further injection**.
5. Compare against `trace[D+1 ...]`, which is the server's own free run from
   that state under that same order stream.

Step 5 is the part that needs defending, and it rests on determinism: there is
no RNG on this path. `sweep.py` measured a 4-seed sweep coming back
bit-identical on both engines; `tier2.py`'s docstring records why (the sim's
tick is a pure function of state and `state.key` is never consumed; the
server's `run_server_episode` seed does not reach the lane and `bot_teams=none`
means `bot_seed` cannot matter). `tier15 record --repeat 2` re-records the same
fixture and compares the server's own FNV-1a state-hash stream tick by tick, so
that claim is checked rather than asserted -- run it if you are about to quote
a number from this module in a gate.

Given determinism, "the server, free-running from the state at D under order
stream A" and "the tail of the recorded trace from D under order stream A" are
the same trajectory. So this harness *does* free-run both engines from one
state. What it does not do is inject both of them.

THE ASYMMETRY THAT FOLLOWS, AND THE FLOOR IT CREATES
----------------------------------------------------
Only one side is injected, so only one side pays injection error. The server
arrived at the state; the sim arrives at a *reconstruction* of it. Every field
`inject.py` cannot observe (its module docstring enumerates them: minion target
identity, the auto-attack clock, ATTACK_TO waypoints, the AI's private timers)
starts wrong at decision 0. That is not a defect of this design -- it is
irreducible while the server has no load-state command -- but it means a
divergence curve from this harness has a **non-zero intercept that is the
harness's own**, and the report prints it first, as `t+1` residuals, before any
curve. A curve whose growth never leaves that intercept has measured nothing.

Two injection approximations bite a driven fixture specifically, and both are
named in the report rather than absorbed:

* `inject.reconstruct_waypoints` projects a MOVE_TO unit onto the **lane
  minion corridor**. That is exact for a marching minion and simply wrong for
  a champion, which does not walk the corridor. This module therefore does not
  use it for champions: `--champion-waypoints last-move` (default) re-issues
  the most recent recorded champion MOVE order at the injected position through
  `sim.orders.apply_orders`, which is not a guess -- it is the order the server
  was actually executing on that tick -- and `--champion-waypoints freeze`
  (`n_waypoints=0`) is the control that assumes nothing at all. The two differ
  only in the first tick or two and the report says which was used.
* A minion in ATTACK_TO is injected frozen, because its target is
  unobservable. In an *engaged* start state that is most of the wave. The
  report counts them as `untrustworthy_movement` and a Q2-style question asked
  from a heavily engaged start must be read against that count.

THE ORDER STREAM IS IDENTICAL, AND IT IS OPEN-LOOP ON THE SIM SIDE
------------------------------------------------------------------
Requirement 3 is "an identical order stream on both sides", and that is what is
replayed: the same wire orders, at the same decision boundaries, decoded by
`action_replay.decision_to_orders` exactly as `one_step.py` decodes them.
Identical, not equivalent -- `record.scripted_action` chose those orders from
the *server's* observation, so on the sim side they are open-loop. A `move` is
a fixed world destination, so the sim's champion is neither corrected toward
the server's position nor dragged by a feedback loop; it executes the same
order from wherever it actually is. An `attack` names a NetId, and a NetId that
was not present at injection cannot be represented in the sim's slot space --
that window is **truncated** at that decision with a stated reason rather than
substituted with something plausible.

ROUTING
-------
`PATH-006`: a scripted driver that paths Moves straight instead of through the
route table measures a run that the gate does not perform. This one loads the
production `map1_garen_r35_o50_v2` artifact by default, exactly as
`last_hit_drive`/`hp_band`/`isolation` now do, and `--table-disabled` is the
named two-point control, not the default.

CANONICAL COMMAND
-----------------
See the ledger's "Canonical commands" table. In short::

    sbatch slurm/parity.sbatch python -m lanerl_jax.parity.archive.tier15 record \\
        --out-dir lanerl_jax/runs/tier15_noshop --decisions 12000 \\
        --port-base 45300 --no-autobuy --repeat 2
    sbatch slurm/parity.sbatch python -m lanerl_jax.parity.archive.tier15 run \\
        --fixture lanerl_jax/runs/tier15_noshop --predicate champ_near_minion \\
        --radius 90 --min-hp-frac 0.5 --decisions 600 \\
        --out lanerl_jax/runs/tier15_noshop/q1_champ_into_wave.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...sim.init import TOP_LANE_PATH
from ...sim.profiles import PROFILES
from ...sim.state import CH_SLICE, Kind, MoveOrder, Team
from ...sim.targeting import MinionType
from ..action_replay import (ActionReplayError, RecordedDecision,
                            align_action_log, decision_to_orders)
from ..diagnostic_identity import net_id_to_entity, net_id_to_injected_slot
from ..diff import DEFAULT_TOLERANCE, LANE_KINDS, _compare
from ..inject import (InjectionReport, infer_minion_model, inject_snapshot,
                     replay_wave_states)
from ..record import ActionLog, Fixture
from .sim_vs_server import NOT_MODELLED
from ..trace import HASH_RE, Snapshot, Trace, load_trace, parse_stream

__all__ = [
    "StartSelection", "select_start", "Tick15Row", "Tier15Result",
    "run_tier15", "main",
]

#: `LANERL_STEP_TICKS` in this stack: 2 server ticks per agent decision.
STEP_TICKS = 2
#: the server team ids, as they appear in the dump and on the wire.
BLUE, RED = 100, 200
#: `trace.PosQ` is 1/16 of a world unit, so two *identical* states still read
#: 0.0625 u apart on each axis after the dump quantises them. Any divergence
#: threshold below this measures the dump, not the engines.
QUANT_FLOOR = 1.0 / 16.0
#: champion-position divergence thresholds, world units. Reported as "first
#: decision at which the curve crosses this", which is the shape of the answer
#: a single max error hides.
POS_THRESHOLDS: Tuple[float, ...] = (1.0, 5.0, 25.0, 100.0)
#: champion-HP divergence thresholds, HP.
HP_THRESHOLDS: Tuple[float, ...] = (1.0, 25.0, 100.0)


# ---------------------------------------------------------------------------
# start-state selection: by predicate, never by hand
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StartSelection:
    """Which tick was chosen, and the whole reason it was chosen.

    `provenance` is not decoration. A start state picked by eye is not
    reproducible and cannot be re-derived after the fixture is re-recorded;
    every field here is what lets a later reader confirm the same predicate
    still selects the same tick, or see that it no longer does.
    """

    index: int
    t_ms: int
    predicate: str
    detail: Dict[str, object] = field(default_factory=dict)

    def describe(self) -> str:
        bits = "  ".join(f"{k}={v}" for k, v in sorted(self.detail.items()))
        return (f"start: trace index {self.index}  t={self.t_ms} ms "
                f"({self.t_ms / 1000:.2f} s)  predicate={self.predicate}\n"
                f"       {bits}")


def load_trace_upto(path, max_t_ms: Optional[int] = None) -> Trace:
    """`trace.load_trace`, but able to stop early.

    A 300 s fixture is ~250 MB of dump and every snapshot before the window is
    only needed for two scalars (`replay_wave_states` and `align_action_log`
    read `t_ms` and nothing else). Parsing the whole file is still the default,
    because an early stop silently narrows the predicate search -- which is why
    this is an explicit `--max-t-ms` and not an automatic optimisation.
    """
    if max_t_ms is None:
        return load_trace(path)

    def _lines():
        with Path(path).open(errors="replace") as fh:
            for line in fh:
                m = HASH_RE.search(line)
                if m is not None and int(m.group(1)) > max_t_ms:
                    return
                yield line

    tr = parse_stream(_lines())
    tr.source = Path(path)
    return tr


def _minion_type(entity, params, profiles=PROFILES) -> Optional[int]:
    """Melee/caster/cannon/super for one dumped LaneMinion row, or None.

    Delegates to `inject.infer_minion_model` rather than re-deriving the
    max-HP table: a second copy of "700 means cannon" is precisely the sort of
    constant that goes stale against a patch swap while still looking right.
    """
    if entity.kind != "LaneMinion" or entity.q_max_hp is None:
        return None
    team = Team.BLUE if entity.team == BLUE else Team.RED
    row, _ = infer_minion_model(entity.q_max_hp / 1024.0, team, params, profiles)
    return None if row is None else profiles[row][1]


def _champion_net_id(snap: Snapshot, team: int) -> Optional[int]:
    for iv in snap.ai_internals:
        if iv.kind == "Champion" and iv.team == team:
            return iv.net_id
    return None


def select_start(
    trace: Trace, predicate: str, params, *,
    team: int = BLUE, radius: float = 90.0, minion_type: str = "any",
    min_attackers: int = 2, index: Optional[int] = None,
    t_ms: Optional[int] = None, earliest_t_ms: int = 0,
    min_hp_frac: float = 0.0, tail_ticks: int = 0,
) -> StartSelection:
    """Choose the injected tick by a stated rule, evaluated on the trace.

    Predicates:

    ``champ_near_minion``
        the first tick at which the champion of ``team`` is within ``radius``
        of a live enemy LaneMinion **and is moving** (move order MOVE_TO or
        ATTACK_TO). This is the `COLL-003` question's geometry: the collision
        ratchet only exists for a champion that is trying to walk through a
        wave, so a stationary champion in contact is the wrong start state.
        ``minion_type=cannon`` narrows it to the 55.7 u PathfindingRadius case,
        which is the only one whose escape (91.7 u) exceeds the trigger (70 u)
        by more than a tick of travel.
    ``engaged``
        the first tick at which at least ``min_attackers`` live enemy
        LaneMinions hold the champion of ``team`` as their target, read from
        the diagnostic internals stream. This is the gate-3 question's state.
    ``index`` / ``t_ms``
        explicit, for re-running a previously reported selection.
    """
    n = len(trace)
    usable = n - tail_ticks
    if usable <= 0:
        raise ValueError(f"trace has {n} snapshots, fewer than the {tail_ticks} "
                         "ticks the requested window needs")

    if predicate == "index":
        if index is None:
            raise ValueError("--predicate index needs --start-index")
        if not 0 <= index < usable:
            raise ValueError(f"--start-index {index} outside [0, {usable})")
        return StartSelection(index, trace[index].t_ms, "index",
                              {"explicit": True})
    if predicate == "t_ms":
        if t_ms is None:
            raise ValueError("--predicate t_ms needs --start-t-ms")
        cand = [i for i in range(usable) if trace[i].t_ms >= t_ms]
        if not cand:
            raise ValueError(f"no snapshot at or after t={t_ms} ms within the "
                             "usable part of the trace")
        i = cand[0]
        return StartSelection(i, trace[i].t_ms, "t_ms", {"requested_t_ms": t_ms})

    want_type = {"any": None, "melee": MinionType.MELEE,
                 "caster": MinionType.CASTER, "cannon": MinionType.CANNON,
                 "super": MinionType.SUPER}.get(minion_type)
    if minion_type != "any" and want_type is None:
        raise ValueError(f"unknown --minion-type {minion_type!r}")
    enemy = RED if team == BLUE else BLUE

    for i in range(usable):
        snap = trace[i]
        if snap.t_ms < earliest_t_ms:
            continue
        ch = snap.champion(team)
        if ch is None or ch.dead:
            continue
        # A champion that is about to die is the wrong start state for both
        # questions: the window it opens is dominated by the death and the
        # fountain teleport that follows, not by the mechanic under test.
        # Measured: the first `champ_near_minion` hit in the canonical fixture
        # starts at 9.9% HP and the champion is dead 38 ticks later.
        if min_hp_frac > 0.0 and ch.hp is not None and ch.q_max_hp:
            if ch.q_hp / ch.q_max_hp < min_hp_frac:
                continue

        if predicate == "champ_near_minion":
            if ch.ai is None or ch.ai.move_order not in (
                    MoveOrder.MOVE_TO, MoveOrder.ATTACK_TO):
                continue
            best = None
            for e in snap.entities:
                if e.kind != "LaneMinion" or e.team != enemy or e.dead:
                    continue
                if want_type is not None and _minion_type(e, params) != want_type:
                    continue
                d = math.hypot(e.x - ch.x, e.y - ch.y)
                if best is None or d < best[0]:
                    best = (d, e)
            if best is None or best[0] > radius:
                continue
            d, e = best
            return StartSelection(i, snap.t_ms, "champ_near_minion", {
                "team": team, "radius": radius, "minion_type": minion_type,
                "nearest_enemy_minion_u": round(d, 3),
                "nearest_enemy_minion_type": _minion_type(e, params),
                "champ_move_order": ch.ai.move_order,
                "champ_hp_frac": round(ch.q_hp / ch.q_max_hp, 4) if ch.q_max_hp else None,
                "champ_xy": (round(ch.x, 2), round(ch.y, 2)),
                "enemy_minion_xy": (round(e.x, 2), round(e.y, 2)),
                "live_enemy_minions": sum(
                    1 for u in snap.entities
                    if u.kind == "LaneMinion" and u.team == enemy and not u.dead),
            })

        if predicate == "engaged":
            cid = _champion_net_id(snap, team)
            if cid is None:
                continue
            attackers = [iv for iv in snap.ai_internals
                         if iv.kind == "LaneMinion" and iv.team == enemy
                         and iv.target_net_id == cid]
            if len(attackers) < min_attackers:
                continue
            dists = sorted(round(math.hypot(iv.q_x / 16.0 - ch.x,
                                            iv.q_y / 16.0 - ch.y), 2)
                           for iv in attackers)
            return StartSelection(i, snap.t_ms, "engaged", {
                "team": team, "min_attackers": min_attackers,
                "n_attackers": len(attackers),
                "attacker_distances_u": dists,
                "champ_hp": None if ch.hp is None else round(ch.hp, 1),
                "champ_hp_frac": round(ch.q_hp / ch.q_max_hp, 4) if ch.q_max_hp else None,
                "champ_xy": (round(ch.x, 2), round(ch.y, 2)),
                "live_enemy_minions": sum(
                    1 for u in snap.entities
                    if u.kind == "LaneMinion" and u.team == enemy and not u.dead),
            })

        if predicate not in ("champ_near_minion", "engaged"):
            raise ValueError(f"unknown predicate {predicate!r}")

    raise ValueError(
        f"predicate {predicate!r} never fired in the usable part of this "
        f"fixture ({usable} of {n} snapshots, t<= {trace[usable - 1].t_ms} ms). "
        "Record a longer fixture or relax the predicate -- do NOT hand-pick a "
        "nearby tick instead, the point of a predicate is that it is stated.")


# ---------------------------------------------------------------------------
# per-tick observations
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SideTick:
    """The scalars one engine reports on one tick, in world units."""

    t_ms: int
    champ_x: float
    champ_y: float
    champ_hp: float
    champ_alive: bool
    champ_move_order: int
    champ_cs: int
    champ_deaths: int
    displacement: float          # since the previous tick of the same engine
    n_own_minions: int
    n_foe_minions: int
    nearest_enemy_minion_u: float
    n_attackers: int             # enemy minions holding this champion
    attacker_ids: Tuple[int, ...]
    #: one entry per attacker, NOT a per-tick median: `CFH-001`'s 71.8%/44.7%
    #: is a fraction of attacker-decisions, and a median-of-medians is a
    #: different quantity that would not be comparable to it.
    attacker_dist_to_own_wave_u: Tuple[float, ...]
    attacker_dist_to_champ_u: Tuple[float, ...]


@dataclass(slots=True)
class MinionDetail:
    """One matched minion on one tick, with both sides' controller state.

    WHY THIS EXISTS.  The first revision of this module reported
    ``minion_pos_err_max`` and nothing else, so a run could say "the minions
    were at the dump's quantisation through tick 33 and at 5 u by tick 34" and
    could not say *which* minion or *which* branch.  That is a magnitude
    without a unit, and attributing it meant re-deriving the whole window by
    hand.  This row carries the NetId and the four inputs that decide a
    minion's next step on both engines -- move order, waypoint count, current
    target, and the auto-attack clock -- so the onset tick names a branch.

    ``srv_*`` comes from the diagnostic internals stream (``AIInternal``), not
    from the hashed state rows, so it is the server's *own* controller state
    rather than an inference from position.  ``sim_target_net`` uses the
    server's own spelling -- **0 means no target**, matching
    ``AIInternal.target_net_id`` -- and -2 means "a target slot that no NetId
    maps to" (a unit the server had already removed).  Those last two are
    different findings and must not collapse; using the sim's internal -1 for
    "none" would have made every targetless minion read as a disagreement.
    """

    net_id: int
    slot: int
    pos_err: float
    sim_x: float
    sim_y: float
    srv_x: float
    srv_y: float
    sim_hp: float
    srv_hp: float
    sim_move_order: int
    srv_move_order: int
    sim_n_waypoints: int
    srv_n_waypoints: int
    sim_target_net: int
    srv_target_net: int
    sim_is_attacking: bool
    srv_is_attacking: bool
    sim_aa_cooldown: float
    srv_aa_cooldown: float
    sim_aa_windup: float
    srv_aa_windup: float
    srv_aa_state: int
    #: names of the scoped fields that disagreed on this pair this tick
    fields: Tuple[str, ...]


@dataclass(slots=True)
class Tick15Row:
    """One tick of the differential."""

    k: int                       # ticks since injection
    decision: float              # k / STEP_TICKS
    t_ms: int
    sim: SideTick
    srv: SideTick
    champ_pos_err: float
    champ_hp_err: float
    matched_minions: int
    minion_pos_err_max: float
    minion_pos_err_mean: float
    unmatched_sim: int
    unmatched_srv: int
    field_diffs: Tuple[str, ...]
    #: NetId of the minion carrying ``minion_pos_err_max``; -1 when none matched
    worst_minion_net: int = -1
    #: only the minions that disagreed on something this tick.  Emitting all 38
    #: every tick would be 16k rows of mostly zeros; emitting none is what made
    #: the last revision unattributable.
    minions: Tuple[MinionDetail, ...] = ()


@dataclass(slots=True)
class Tier15Result:
    fixture: str
    start: StartSelection
    decisions_requested: int
    ticks_run: int
    truncated_reason: Optional[str]
    injection: Dict[str, object]
    champion_waypoint_mode: str
    routed: bool
    ablations: Tuple[str, ...] = ()
    #: orders that could not be represented in the sim's slot space and were
    #: replaced by a noop under ``--on-unresolvable noop``. Non-zero means the
    #: two sides did NOT execute an identical order stream for the whole
    #: window, and the report says so in its header rather than in a footnote.
    orders_substituted: int = 0
    rows: List[Tick15Row] = field(default_factory=list)

    # -- curve summaries ---------------------------------------------------
    def first_crossing(self, values: Sequence[float], threshold: float
                       ) -> Optional[int]:
        for row, v in zip(self.rows, values):
            if v > threshold:
                return row.k
        return None

    def champ_pos_curve(self) -> List[float]:
        return [r.champ_pos_err for r in self.rows]

    def champ_hp_curve(self) -> List[float]:
        return [abs(r.champ_hp_err) for r in self.rows]

    def field_onsets(self) -> Dict[str, int]:
        """First tick at which each scoped field disagreed on any matched pair."""
        out: Dict[str, int] = {}
        for r in self.rows:
            for name in r.field_diffs:
                out.setdefault(name, r.k)
        return out

    def minion_onsets(self) -> Dict[str, Dict[str, int]]:
        """First tick at which each *named* minion disagreed on each field.

        Keyed by NetId as a string so the JSON round-trips.  This is the
        instrument Task 1 needed: the whole-corpus residuals are per-field
        counts, and joining them to a compounding run requires the unit.
        """
        out: Dict[str, Dict[str, int]] = {}
        for r in self.rows:
            for m in r.minions:
                per = out.setdefault(str(m.net_id), {})
                for name in m.fields:
                    per.setdefault(name, r.k)
        return out

    def minion_divergence_order(self) -> List[Dict[str, object]]:
        """Minions ordered by the tick they *first* left the quantisation floor.

        Position only, and deliberately so: a target or cooldown disagreement
        is often unobservable at injection (see ``RESET-002``/``RESET-004``),
        while a position disagreement after a shared start is a statement
        about movement and nothing else.
        """
        first: Dict[int, Dict[str, object]] = {}
        for r in self.rows:
            for m in r.minions:
                if m.pos_err <= QUANT_FLOOR * 1.5:
                    continue
                if m.net_id in first:
                    continue
                first[m.net_id] = {
                    "net_id": m.net_id, "slot": m.slot, "tick": r.k,
                    "decision": r.decision, "t_ms": r.t_ms,
                    "pos_err": m.pos_err,
                    "sim_move_order": m.sim_move_order,
                    "srv_move_order": m.srv_move_order,
                    "sim_target_net": m.sim_target_net,
                    "srv_target_net": m.srv_target_net,
                    "sim_is_attacking": m.sim_is_attacking,
                    "srv_is_attacking": m.srv_is_attacking,
                    "fields": list(m.fields),
                }
        return sorted(first.values(), key=lambda d: d["tick"])

    # -- Q1: per-tick champion displacement --------------------------------
    def displacement_stats(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, get in (("sim", lambda r: r.sim), ("server", lambda r: r.srv)):
            disp = np.array([get(r).displacement for r in self.rows], float)
            near = np.array([get(r).nearest_enemy_minion_u for r in self.rows],
                            float)
            vecs = []
            prev = None
            for r in self.rows:
                s = get(r)
                cur = (s.champ_x, s.champ_y)
                if prev is not None:
                    vecs.append((cur[0] - prev[0], cur[1] - prev[1]))
                prev = cur
            reversals = sum(
                1 for a, b in zip(vecs, vecs[1:])
                if (a[0] * b[0] + a[1] * b[1]) < 0
                and math.hypot(*a) > 0.05 and math.hypot(*b) > 0.05)
            n = max(1, len(disp))
            out[name] = {
                "ticks": float(len(disp)),
                "mean_u_per_tick": float(disp.mean()) if len(disp) else 0.0,
                "median_u_per_tick": float(np.median(disp)) if len(disp) else 0.0,
                "mean_u_per_decision": float(disp.mean() * STEP_TICKS) if len(disp) else 0.0,
                "total_path_u": float(disp.sum()),
                "net_displacement_u": float(math.hypot(
                    get(self.rows[-1]).champ_x - get(self.rows[0]).champ_x,
                    get(self.rows[-1]).champ_y - get(self.rows[0]).champ_y))
                if self.rows else 0.0,
                "frac_ticks_under_1u": float((disp < 1.0).sum() / n),
                "direction_reversals": float(reversals),
                "frac_ticks_in_contact_70u": float((near <= 70.0).sum() / n),
                "median_nearest_enemy_minion_u": float(np.median(near)) if len(near) else 0.0,
            }
        return out

    # -- Q2: holds, attacker geometry, champion HP -------------------------
    def hold_stats(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, get in (("sim", lambda r: r.sim), ("server", lambda r: r.srv)):
            runs: Dict[int, int] = {}
            closed: List[int] = []
            per_tick = []
            dist_wave: List[float] = []
            dist_champ: List[float] = []
            for r in self.rows:
                s = get(r)
                per_tick.append(s.n_attackers)
                dist_wave.extend(s.attacker_dist_to_own_wave_u)
                dist_champ.extend(s.attacker_dist_to_champ_u)
                live = set(s.attacker_ids)
                for aid in list(runs):
                    if aid not in live:
                        closed.append(runs.pop(aid))
                for aid in live:
                    runs[aid] = runs.get(aid, 0) + 1
            censored = sorted(runs.values(), reverse=True)
            arr = np.array(closed, float)
            out[name] = {
                "holds_closed": float(len(closed)),
                "holds_open_at_window_end": float(len(censored)),
                "mean_hold_ticks": float(arr.mean()) if len(arr) else 0.0,
                # `gate3_arrivals` reports holds in DECISIONS (51/267 sim vs
                # 45/211 server, the 1.26x). Carried in both units so the two
                # can be read against each other without a mental conversion.
                "mean_hold_decisions": float(arr.mean() / STEP_TICKS) if len(arr) else 0.0,
                "median_hold_ticks": float(np.median(arr)) if len(arr) else 0.0,
                "max_hold_ticks": float(arr.max()) if len(arr) else 0.0,
                "longest_open_hold_ticks": float(censored[0]) if censored else 0.0,
                "mean_attackers_per_tick": float(np.mean(per_tick)) if per_tick else 0.0,
                "frac_ticks_with_any_attacker": float(
                    np.mean([1.0 if a else 0.0 for a in per_tick])) if per_tick else 0.0,
                "attacker_ticks": float(len(dist_champ)),
                "median_attacker_dist_to_own_wave_u": float(np.median(dist_wave)) if dist_wave else float("nan"),
                "median_attacker_dist_to_champ_u": float(np.median(dist_champ)) if dist_champ else float("nan"),
                "frac_attacker_ticks_beyond_475u": float(
                    np.mean([1.0 if d > 475.0 else 0.0 for d in dist_wave])) if dist_wave else float("nan"),
            }
        return out

    def hp_stats(self) -> Dict[str, float]:
        if not self.rows:
            return {}
        a, b = self.rows[0], self.rows[-1]
        return {
            "sim_hp_start": a.sim.champ_hp, "sim_hp_end": b.sim.champ_hp,
            "srv_hp_start": a.srv.champ_hp, "srv_hp_end": b.srv.champ_hp,
            "sim_hp_lost": a.sim.champ_hp - b.sim.champ_hp,
            "srv_hp_lost": a.srv.champ_hp - b.srv.champ_hp,
            "max_abs_hp_err": max(abs(r.champ_hp_err) for r in self.rows),
        }

    def to_json(self) -> Dict[str, object]:
        return {
            "fixture": self.fixture,
            "start": {"index": self.start.index, "t_ms": self.start.t_ms,
                      "predicate": self.start.predicate,
                      "detail": self.start.detail},
            "decisions_requested": self.decisions_requested,
            "ticks_run": self.ticks_run,
            "truncated_reason": self.truncated_reason,
            "injection": self.injection,
            "champion_waypoint_mode": self.champion_waypoint_mode,
            "routed": self.routed,
            "ablations": list(self.ablations),
            "orders_substituted": self.orders_substituted,
            "displacement": self.displacement_stats(),
            "holds": self.hold_stats(),
            "hp": self.hp_stats(),
            "field_onsets": self.field_onsets(),
            "minion_onsets": self.minion_onsets(),
            "minion_divergence_order": self.minion_divergence_order(),
            "rows": [
                {"k": r.k, "decision": r.decision, "t_ms": r.t_ms,
                 "champ_pos_err": r.champ_pos_err,
                 "champ_hp_err": r.champ_hp_err,
                 "minion_pos_err_max": r.minion_pos_err_max,
                 "matched_minions": r.matched_minions,
                 "worst_minion_net": r.worst_minion_net,
                 "minions": [asdict(m) for m in r.minions],
                 "sim": asdict(r.sim), "srv": asdict(r.srv)}
                for r in self.rows
            ],
        }

    def report(self, curve_points: int = 21) -> str:
        L: List[str] = []
        L.append("=" * 74)
        L.append("TIER 1.5  bounded, injected, free-running differential")
        L.append("=" * 74)
        L.append("")
        L.append("ONLY THE SIM IS INJECTED. The server has no load-state command")
        L.append("(LanerlWire.Parse accepts reset + actions and nothing else), so")
        L.append("its side of this comparison is its own recorded free run from the")
        L.append("same tick under the same order stream. See the module docstring.")
        L.append("")
        L.append(f"fixture: {self.fixture}")
        L.append(self.start.describe())
        L.append(f"window: {self.ticks_run} ticks "
                 f"({self.ticks_run / STEP_TICKS:.0f} decisions, "
                 f"{self.ticks_run * 16.667 / 1000:.2f} s) of "
                 f"{self.decisions_requested} requested")
        if self.truncated_reason:
            L.append(f"TRUNCATED: {self.truncated_reason}")
        L.append(f"champion waypoints at injection: {self.champion_waypoint_mode}")
        L.append(f"routing: {'production route table' if self.routed else 'TABLE DISABLED -- control, not gate evidence'}")
        if self.ablations:
            L.append(f"ABLATIONS ACTIVE: {', '.join(self.ablations)} "
                     "-- this is a causal control, NOT gate evidence")
        if self.orders_substituted:
            L.append(f"ORDER STREAM NOT IDENTICAL: {self.orders_substituted} "
                     "order(s) replaced by a noop on the sim side because their "
                     "target did not exist at injection. Every number below is "
                     "conditioned on that substitution.")
        L.append("")
        L.append("-- injection (the harness's own error, before any dynamics) --")
        for k, v in sorted(self.injection.items()):
            L.append(f"   {k}: {v}")
        if not self.rows:
            L.append("no rows: nothing to report")
            return "\n".join(L)

        r0 = self.rows[0]
        L.append("")
        L.append("-- t+1 residual: the floor this curve starts from --")
        L.append(f"   champion position error {r0.champ_pos_err:.3f} u "
                 f"(dump quantisation alone is {QUANT_FLOOR:.4f} u/axis)")
        L.append(f"   champion HP error       {r0.champ_hp_err:+.3f}")
        L.append(f"   matched minions {r0.matched_minions}, "
                 f"max position error {r0.minion_pos_err_max:.3f} u, "
                 f"mean {r0.minion_pos_err_mean:.3f} u")
        L.append(f"   unmatched: sim {r0.unmatched_sim}, server {r0.unmatched_srv}")

        L.append("")
        L.append("-- divergence as a curve (champion position error, world units) --")
        L.append("   decision       t(s)    pos_err     hp_err   minion_max   attackers(sim/srv)")
        step = max(1, len(self.rows) // max(1, curve_points - 1))
        for r in self.rows[::step] + ([self.rows[-1]] if len(self.rows) % step else []):
            L.append(f"   {r.decision:8.1f}  {r.t_ms / 1000:9.3f} "
                     f"{r.champ_pos_err:10.3f} {r.champ_hp_err:10.2f} "
                     f"{r.minion_pos_err_max:12.3f}   "
                     f"{r.sim.n_attackers:6d}/{r.srv.n_attackers:<6d}")

        L.append("")
        L.append("-- first crossing (decision index; None = never in this window) --")
        pos = self.champ_pos_curve()
        for thr in POS_THRESHOLDS:
            k = self.first_crossing(pos, thr)
            L.append(f"   champion position > {thr:7.1f} u : "
                     f"{'never' if k is None else f'decision {k / STEP_TICKS:.1f} (tick {k})'}")
        hp = self.champ_hp_curve()
        for thr in HP_THRESHOLDS:
            k = self.first_crossing(hp, thr)
            L.append(f"   champion HP       > {thr:7.1f}   : "
                     f"{'never' if k is None else f'decision {k / STEP_TICKS:.1f} (tick {k})'}")
        L.append(f"   max champion position error over window: {max(pos):.3f} u")
        L.append(f"   max champion HP error over window:       {max(hp):.3f}")

        L.append("")
        L.append("-- per-field divergence onset (scoped, matched pairs only) --")
        onsets = self.field_onsets()
        if not onsets:
            L.append("   no scoped field ever disagreed")
        for name, k in sorted(onsets.items(), key=lambda kv: (kv[1], kv[0])):
            L.append(f"   {name:<28} first at tick {k} "
                     f"(decision {k / STEP_TICKS:.1f})")
        L.append(f"   fields excluded as not modelled: {', '.join(NOT_MODELLED)}")

        L.append("")
        L.append("-- named minions, in the order they left the quantisation floor --")
        L.append("   (position only; the branch inputs at that exact tick)")
        div = self.minion_divergence_order()
        if not div:
            L.append("   no matched minion ever exceeded "
                     f"{QUANT_FLOOR * 1.5:.4f} u")
        L.append(f"   {'net_id':>10} {'tick':>5} {'dec':>7} {'pos_err':>9}  "
                 f"{'order s/v':>11} {'target s/v':>21} {'atk s/v':>9}  fields")
        for d in div[:12]:
            L.append(
                f"   {d['net_id']:>10} {d['tick']:>5} {d['decision']:>7.1f} "
                f"{d['pos_err']:>9.3f}  "
                f"{d['sim_move_order']:>5}/{d['srv_move_order']:<5} "
                f"{d['sim_target_net']:>10}/{d['srv_target_net']:<10} "
                f"{int(d['sim_is_attacking']):>4}/{int(d['srv_is_attacking']):<4}  "
                f"{','.join(d['fields'])}")
        if len(div) > 12:
            L.append(f"   ... {len(div) - 12} more (full list in the JSON)")

        L.append("")
        L.append("-- Q1  champion displacement (the COLL-003 question) --")
        d = self.displacement_stats()
        keys = list(d["sim"])
        L.append(f"   {'metric':<34} {'sim':>12} {'server':>12}")
        for key in keys:
            L.append(f"   {key:<34} {d['sim'][key]:12.4f} {d['server'][key]:12.4f}")

        L.append("")
        L.append("-- Q2  holds / attacker geometry / champion HP (the gate-3 question) --")
        h = self.hold_stats()
        L.append(f"   {'metric':<34} {'sim':>12} {'server':>12}")
        for key in list(h["sim"]):
            L.append(f"   {key:<34} {h['sim'][key]:12.4f} {h['server'][key]:12.4f}")
        s = self.hp_stats()
        L.append(f"   champion HP  sim {s['sim_hp_start']:.1f} -> {s['sim_hp_end']:.1f} "
                 f"(lost {s['sim_hp_lost']:.1f})   "
                 f"server {s['srv_hp_start']:.1f} -> {s['srv_hp_end']:.1f} "
                 f"(lost {s['srv_hp_lost']:.1f})")
        return "\n".join(L)


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _server_side(snap: Snapshot, team: int, prev: Optional[SideTick]
                 ) -> Optional[SideTick]:
    enemy = RED if team == BLUE else BLUE
    ch = snap.champion(team)
    if ch is None:
        return None
    cid = _champion_net_id(snap, team)
    by_id = net_id_to_entity(snap)
    own_alive = [e for e in snap.entities
                 if e.kind == "LaneMinion" and e.team == team and not e.dead]
    foe_alive = [e for e in snap.entities
                 if e.kind == "LaneMinion" and e.team == enemy and not e.dead]
    near = min((math.hypot(e.x - ch.x, e.y - ch.y) for e in foe_alive),
               default=float("inf"))
    ids, dw, dc = [], [], []
    if cid is not None:
        for iv in snap.ai_internals:
            if iv.kind != "LaneMinion" or iv.team != enemy:
                continue
            if iv.target_net_id != cid:
                continue
            e = by_id.get(iv.net_id)
            if e is None or e.dead:
                continue
            ids.append(iv.net_id)
            ax, ay = iv.q_x / 16.0, iv.q_y / 16.0
            dc.append(math.hypot(ax - ch.x, ay - ch.y))
            if own_alive:
                dw.append(min(math.hypot(b.x - ax, b.y - ay) for b in own_alive))
    x, y = ch.x, ch.y
    return SideTick(
        t_ms=snap.t_ms, champ_x=x, champ_y=y,
        champ_hp=0.0 if ch.hp is None else ch.hp,
        champ_alive=not bool(ch.dead),
        champ_move_order=ch.ai.move_order if ch.ai else -1,
        champ_cs=ch.champ.minions_killed if ch.champ else -1,
        champ_deaths=ch.champ.deaths if ch.champ else -1,
        displacement=0.0 if prev is None else math.hypot(x - prev.champ_x,
                                                         y - prev.champ_y),
        n_own_minions=len(own_alive), n_foe_minions=len(foe_alive),
        nearest_enemy_minion_u=near,
        n_attackers=len(ids), attacker_ids=tuple(sorted(ids)),
        attacker_dist_to_own_wave_u=tuple(dw),
        attacker_dist_to_champ_u=tuple(dc),
    )


def _sim_side(state, team_id: int, ch_slot: int, prev: Optional[SideTick]
              ) -> SideTick:
    sim_team = Team.BLUE if team_id == BLUE else Team.RED
    sim_enemy = Team.RED if sim_team == Team.BLUE else Team.BLUE
    kind = np.asarray(state.kind)
    tm = np.asarray(state.team)
    alive = np.asarray(state.alive)
    x = np.asarray(state.x)
    y = np.asarray(state.y)
    tgt = np.asarray(state.target)
    cx, cy = float(x[ch_slot]), float(y[ch_slot])
    minion = (kind == Kind.LANE_MINION) & alive
    own = np.flatnonzero(minion & (tm == sim_team))
    foe = np.flatnonzero(minion & (tm == sim_enemy))
    near = float(np.min(np.hypot(x[foe] - cx, y[foe] - cy))) if foe.size else float("inf")
    att = foe[tgt[foe] == ch_slot]
    dw, dc = [], []
    for i in att:
        ax, ay = float(x[i]), float(y[i])
        dc.append(math.hypot(ax - cx, ay - cy))
        if own.size:
            dw.append(float(np.min(np.hypot(x[own] - ax, y[own] - ay))))
    # slot identity is stable within a window, but a slot freed by a death can
    # be re-used by a spawn; `spawn_seq` disambiguates so a recycled slot is
    # never counted as one long uninterrupted hold.
    seq = np.asarray(state.spawn_seq)
    ids = tuple(sorted(int(i) * 1_000_003 + int(seq[i]) for i in att))
    return SideTick(
        t_ms=int(round(float(state.t_ms))), champ_x=cx, champ_y=cy,
        champ_hp=float(np.asarray(state.hp)[ch_slot]),
        champ_alive=bool(alive[ch_slot]),
        champ_move_order=int(np.asarray(state.move_order)[ch_slot]),
        champ_cs=int(np.asarray(state.cs)[ch_slot]),
        champ_deaths=int(np.asarray(state.deaths)[ch_slot]),
        displacement=0.0 if prev is None else math.hypot(cx - prev.champ_x,
                                                         cy - prev.champ_y),
        n_own_minions=int(own.size), n_foe_minions=int(foe.size),
        nearest_enemy_minion_u=near,
        n_attackers=int(att.size), attacker_ids=ids,
        attacker_dist_to_own_wave_u=tuple(dw),
        attacker_dist_to_champ_u=tuple(dc),
    )


def _seed_champion_orders(decision: Optional[RecordedDecision],
                          net_id_slots) -> Optional["Orders"]:
    """Re-issue the most recent recorded champion **move**, and only a move.

    A move is idempotent: re-ordering a champion to the destination it is
    already walking to restores the waypoint list the injector cannot see,
    from the order the server was demonstrably executing. A cast is not
    idempotent (it would fire a second time) and an attack re-acquires,
    so both are dropped here rather than replayed.
    """
    if decision is None:
        return None
    blue = decision.blue if decision.blue.get("t") == "move" else {"t": "noop"}
    red = decision.red if decision.red.get("t") == "move" else {"t": "noop"}
    if blue["t"] == "noop" and red["t"] == "noop":
        return None
    return decision_to_orders(
        RecordedDecision(decision.source_t_ms, blue, red), net_id_slots)


def run_tier15(
    fixture: Fixture, selection_kwargs: Dict[str, object], *,
    decisions: int = 600, team: int = BLUE,
    champion_waypoints: str = "last-move",
    route_artifact: Optional[Path] = None, table_disabled: bool = False,
    enable_call_for_help: bool = True, enable_collision: bool = True,
    on_unresolvable: str = "truncate", max_t_ms: Optional[int] = None,
    stop_on_death: bool = True, patch=None,
) -> Tier15Result:
    """Inject one tick and free-run the sim against the recorded server tail."""
    import jax
    import jax.numpy as jnp

    from ...data.patch import load_patch
    from ...sim.config import DEFAULT_ROUTE_ARTIFACT, SimConfig
    from ...sim.orders import apply_orders
    from ...sim.step import tick
    from .one_step import _tick_jit          # the exact tick one_step diffs

    if enable_call_for_help and enable_collision:
        step_one = _tick_jit                 # share one_step's compiled tick
    else:
        # the ablation controls need the flags STATIC (`tick` branches on them
        # in Python), so they cannot go through `_tick_jit`.
        ablated = jax.jit(tick, static_argnames=("enable_call_for_help",
                                                 "enable_collision"))
        step_one = lambda s, p, lane_path: ablated(          # noqa: E731
            s, p, lane_path=lane_path,
            enable_call_for_help=enable_call_for_help,
            enable_collision=enable_collision)

    patch = patch or load_patch()
    # `STRUCT-003`: params, lane path and routing from ONE SimConfig. The
    # tick itself stays per-tick `tick` (injection needs one tick at a time).
    sim = SimConfig.scripted(
        patch, route_artifact=None if table_disabled else (
            Path(route_artifact) if route_artifact else DEFAULT_ROUTE_ARTIFACT))
    params = sim.params
    lane_path = sim.lane_path

    trace = load_trace_upto(fixture.log, max_t_ms)
    actions = ActionLog.load(fixture.actions)
    action_at = align_action_log(trace.snapshots, actions)

    n_ticks = decisions * STEP_TICKS
    sel = select_start(trace, params=params, team=team,
                       tail_ticks=n_ticks + 1, **selection_kwargs)

    route_table, terrain = sim.route_table, sim.terrain

    # Orders go through a JITTED wrapper, not a bare `apply_orders`.
    # Measured on `desktop`: applying a routed order eagerly, once per
    # decision, grows RSS by ~170 MB per call and OOM-killed a 20 GB job at
    # ~300 decisions -- the route artifact is reconstructed op by op every
    # time. Every other driver in this package (`last_hit_drive`, `hp_band`,
    # `isolation`, `trainer`) already calls it inside a `jax.jit`; this one
    # has to as well, and separately from the tick because the order lands
    # *after* the tick, at the control boundary.
    @jax.jit
    def _order_jit(st, orders):
        return apply_orders(st, orders, params, route_table=route_table,
                            terrain=terrain)

    wave_states = replay_wave_states(trace)
    i0 = sel.index
    previous = trace[i0 - 1] if i0 else None
    if previous is not None and not (0 < trace[i0].t_ms - previous.t_ms <= 34):
        previous = None
    state, rep = inject_snapshot(trace[i0], wave_states[i0], params, PROFILES,
                                previous_snapshot=previous)
    net_id_slots = net_id_to_injected_slot(trace[i0], rep.notes)

    ch_slot = CH_SLICE.start + (0 if team == BLUE else 1)
    # champion waypoints: NOT the corridor projection inject.py uses for a
    # marching minion (see module docstring).
    n_wp = np.array(state.n_waypoints, copy=True)
    for s in (CH_SLICE.start, CH_SLICE.start + 1):
        n_wp[s] = 0
    state = state.replace(n_waypoints=jnp.asarray(n_wp))
    if champion_waypoints == "last-move":
        prior = max((j for j in action_at if j <= i0), default=None)
        seed = _seed_champion_orders(
            action_at.get(prior) if prior is not None else None, net_id_slots)
        if seed is not None:
            state = _order_jit(state, seed)
            champ_mode = (f"last-move (re-issued the recorded move from "
                          f"snapshot {prior}, t={trace[prior].t_ms} ms)")
        else:
            champ_mode = "last-move requested, but no prior move order: frozen"
    elif champion_waypoints == "freeze":
        champ_mode = "freeze (n_waypoints=0; assumes nothing)"
    else:
        raise ValueError(f"unknown --champion-waypoints {champion_waypoints!r}")

    identity_complete = (bool(rep.notes)
                         and len(net_id_slots) == len(rep.notes)
                         and bool(trace[i0].ai_internals))
    injection = {
        "units_injected": rep.n_units,
        "untrustworthy_movement_units": rep.n_untrustworthy_movement,
        "dropped_unknown_kind": rep.dropped_unknown_kind,
        "dropped_capacity": dict(rep.dropped_capacity),
        "identity": ("diagnostic NetId (complete)" if identity_complete
                     else "INCOMPLETE -- NetIds missing, minion identity is a guess"),
        "attack_to_minions_frozen": sum(
            1 for n in rep.notes
            if not n.movement_trustworthy and n.kind == "LaneMinion"),
    }
    # Champion move speed is checked explicitly and loudly, because it is the
    # one injected stat whose mismatch silently invalidates every displacement
    # number this module prints, and it DOES mismatch: `LanerlHooks`'
    # auto-shop buys Boots of Speed (item 1001, +25 flat MS) mid-episode, the
    # sim has no item model, and the profile table therefore keeps the
    # champion at base. Not a rounding difference: 370/345 is 7.2% of travel
    # per tick, compounding over a whole approach.
    srv_ch = trace[i0].champion(team)
    if srv_ch is not None and srv_ch.champ is not None:
        sim_ms = float(params["move_speed"][int(np.asarray(state.model)[ch_slot])])
        srv_ms = srv_ch.champ.q_move_speed / 1024.0
        injection["champion_move_speed_sim"] = round(sim_ms, 3)
        injection["champion_move_speed_server"] = round(srv_ms, 3)
        if abs(sim_ms - srv_ms) > 0.5:
            injection["MOVE_SPEED_MISMATCH"] = (
                f"server {srv_ms:.1f} vs sim {sim_ms:.1f} "
                f"({srv_ms / sim_ms:.4f}x) -- displacement comparisons below "
                "are NOT engine-fidelity measurements while this holds. "
                "Re-record with LANERL_AUTOBUY=0 (`record --no-autobuy`).")

    tol = replace(DEFAULT_TOLERANCE, kinds=LANE_KINDS,
                  ignore_fields=NOT_MODELLED)
    slot_of_net = dict(net_id_slots)
    net_of_slot = {slot: net for net, slot in slot_of_net.items()}

    rows: List[Tick15Row] = []
    truncated: Optional[str] = None
    substituted = 0
    sim_prev = _sim_side(state, team, ch_slot, None)
    srv_prev = _server_side(trace[i0], team, None)
    if srv_prev is None:
        raise ValueError(f"the selected start tick has no champion for team {team}")
    rows0_deaths = (sim_prev.champ_deaths, srv_prev.champ_deaths)

    from .sim_vs_server import state_to_snapshot

    for k in range(1, n_ticks + 1):
        j = i0 + k
        if j >= len(trace):
            truncated = f"trace exhausted at index {j}"
            break
        dt = trace[j].t_ms - trace[j - 1].t_ms
        if not (0 < dt <= 34):
            truncated = (f"gap in the server dump at index {j} "
                         f"({trace[j - 1].t_ms} -> {trace[j].t_ms} ms)")
            break

        state = step_one(state, params, lane_path=lane_path)
        dec = action_at.get(j)
        if dec is not None:
            try:
                orders = decision_to_orders(dec, slot_of_net)
            except ActionReplayError as exc:
                if on_unresolvable == "truncate":
                    truncated = (
                        f"order at t={dec.source_t_ms} ms cannot be represented "
                        f"without guessing: {exc}. The window ends here rather "
                        "than substituting a different order on the two sides.")
                    break
                substituted += 1
                orders = decision_to_orders(
                    RecordedDecision(dec.source_t_ms, {"t": "noop"},
                                     {"t": "noop"}), slot_of_net)
            state = _order_jit(state, orders)

        snap = trace[j]
        sim_t = _sim_side(state, team, ch_slot, sim_prev)
        srv_t = _server_side(snap, team, srv_prev)
        if srv_t is None:
            truncated = f"server champion row vanished at index {j}"
            break
        # A champion death ends comparability, on either side. `Respawn()`
        # teleports to the fountain, so one more tick of this window compares
        # a champion in lane against a champion 12,000 units away and every
        # aggregate below becomes a statement about a teleport. Measured: an
        # un-truncated 600-decision window over a death reported the server
        # champion moving 13.77 u/tick, against a walk speed of 6.17.
        if stop_on_death and (
                not sim_t.champ_alive or not srv_t.champ_alive
                or sim_t.champ_deaths != rows0_deaths[0]
                or srv_t.champ_deaths != rows0_deaths[1]):
            truncated = (
                f"champion death at tick {k} (t={snap.t_ms} ms): sim alive="
                f"{sim_t.champ_alive} deaths={sim_t.champ_deaths}, server "
                f"alive={srv_t.champ_alive} deaths={srv_t.champ_deaths}. "
                "A respawn teleports to the fountain, so nothing after this "
                "tick is a comparison of the same situation.")
            break

        # matched-minion comparison by diagnostic NetId: exact identity, so a
        # position error is attributable rather than a matching artifact.
        sim_snap = state_to_snapshot(state, t_ms=float(state.t_ms), params=params)
        sim_by_slot = {}
        kind_a = np.asarray(state.kind); alive_a = np.asarray(state.alive)
        live_slots = list(np.flatnonzero((kind_a != Kind.NONE) & alive_a))
        for slot, ent in zip(live_slots, sim_snap.entities):
            sim_by_slot[int(slot)] = ent
        srv_by_id = net_id_to_entity(snap)
        srv_int_by_id = {ai.net_id: ai for ai in snap.ai_internals}
        # Read the sim's controller arrays once per tick rather than per unit:
        # `state.target` etc. are device arrays and a per-unit `int(...)` on
        # each would transfer 38 times a tick for 428 ticks.
        s_target = np.asarray(state.target)
        s_move_order = np.asarray(state.move_order)
        s_nwp = np.asarray(state.n_waypoints)
        s_attacking = np.asarray(state.is_attacking)
        s_aacd = np.asarray(state.aa_cooldown)
        s_aawindup = np.asarray(state.aa_windup)
        errs: List[float] = []
        names: set = set()
        details: List[MinionDetail] = []
        matched = 0
        unmatched_srv = 0
        worst_err, worst_net = -1.0, -1
        for net_id, slot in slot_of_net.items():
            b = srv_by_id.get(net_id)
            a = sim_by_slot.get(slot)
            if b is None or b.dead:
                if a is not None:
                    unmatched_srv += 1
                continue
            if a is None:
                unmatched_srv += 1
                continue
            if b.kind not in LANE_KINDS or a.kind != b.kind:
                continue
            matched += 1
            err = math.hypot(a.x - b.x, a.y - b.y)
            errs.append(err)
            pair_fields = tuple(fd.name for fd in _compare(a, b, tol))
            names.update(pair_fields)
            if err > worst_err:
                worst_err, worst_net = err, net_id
            if b.kind != "LaneMinion":
                continue
            bi = srv_int_by_id.get(net_id)
            sim_tgt_slot = int(s_target[slot])
            sim_tgt_net = (0 if sim_tgt_slot < 0
                           else net_of_slot.get(sim_tgt_slot, -2))
            srv_move_order = -1 if b.ai is None else b.ai.move_order
            srv_nwp = -1 if b.ai is None else b.ai.waypoints
            # move_order and waypoints are in NOT_MODELLED, so `_compare`
            # never sees them; they are the branch inputs, so they are named
            # here explicitly rather than silently skipped.
            extra = []
            if int(s_move_order[slot]) != srv_move_order:
                extra.append("move_order")
            if bi is not None and sim_tgt_net != bi.target_net_id:
                extra.append("target")
            if bi is not None and bool(s_attacking[slot]) != bi.is_attacking:
                extra.append("is_attacking")
            all_fields = tuple(sorted(set(pair_fields) | set(extra)))
            if not all_fields and err <= QUANT_FLOOR * 1.5:
                continue
            details.append(MinionDetail(
                net_id=net_id, slot=slot, pos_err=err,
                sim_x=a.x, sim_y=a.y, srv_x=b.x, srv_y=b.y,
                sim_hp=float(a.hp or 0.0), srv_hp=float(b.hp or 0.0),
                sim_move_order=int(s_move_order[slot]),
                srv_move_order=srv_move_order,
                sim_n_waypoints=int(s_nwp[slot]), srv_n_waypoints=srv_nwp,
                sim_target_net=sim_tgt_net,
                # -3, not 0: "the diagnostic line is missing" is not
                # "the server held no target".
                srv_target_net=-3 if bi is None else bi.target_net_id,
                sim_is_attacking=bool(s_attacking[slot]),
                srv_is_attacking=False if bi is None else bi.is_attacking,
                sim_aa_cooldown=float(s_aacd[slot]),
                srv_aa_cooldown=(-1.0 if bi is None
                                 else bi.q_aa_cooldown / 1024.0),
                sim_aa_windup=float(s_aawindup[slot]),
                srv_aa_windup=(-1.0 if bi is None
                               else bi.q_aa_windup / 1024.0),
                srv_aa_state=-1 if bi is None else bi.aa_state,
                fields=all_fields,
            ))
        sim_minion_slots = int(np.count_nonzero(
            (kind_a == Kind.LANE_MINION) & alive_a))
        unmatched_sim = max(0, sim_minion_slots - matched)

        rows.append(Tick15Row(
            k=k, decision=k / STEP_TICKS, t_ms=snap.t_ms,
            sim=sim_t, srv=srv_t,
            champ_pos_err=math.hypot(sim_t.champ_x - srv_t.champ_x,
                                     sim_t.champ_y - srv_t.champ_y),
            champ_hp_err=sim_t.champ_hp - srv_t.champ_hp,
            matched_minions=matched,
            minion_pos_err_max=max(errs) if errs else 0.0,
            minion_pos_err_mean=float(np.mean(errs)) if errs else 0.0,
            unmatched_sim=unmatched_sim, unmatched_srv=unmatched_srv,
            field_diffs=tuple(sorted(names)),
            worst_minion_net=worst_net,
            minions=tuple(details),
        ))
        sim_prev, srv_prev = sim_t, srv_t

    return Tier15Result(
        fixture=str(fixture.log), start=sel, decisions_requested=decisions,
        ticks_run=len(rows), truncated_reason=truncated, injection=injection,
        champion_waypoint_mode=champ_mode, routed=not table_disabled,
        ablations=tuple(
            n for n, on in (("call-for-help disabled", not enable_call_for_help),
                            ("collision disabled", not enable_collision)) if on),
        orders_substituted=substituted, rows=rows,
    )


# ---------------------------------------------------------------------------
# fixture recording + the determinism check the whole design rests on
# ---------------------------------------------------------------------------

def record_fixtures(out_dir: Path, decisions: int, port_base: int,
                    repeat: int = 1, autobuy: bool = True) -> List[Fixture]:
    """Record the driven fixture, optionally with the server's shop turned off.

    `LanerlHooks`' auto-shop (`LANERL_AUTOBUY`, `LanerlConfig.BuildPath`) buys
    Doran's Shield, two potions and then **Boots of Speed** out of ambient
    gold. The sim has no item model at all, so from the tick the boots land
    the two champions have different movement speeds (370 against 345) and no
    champion-kinematics differential is measuring the engines any more. The
    lane's first wave is at 90 s and the boots land at ~81 s, so there is no
    champion-in-a-wave tick in a default fixture that predates them:
    `autobuy=False` is the only way to get a clean one.
    """
    from ..record import record_fixture

    extra = None if autobuy else {"LANERL_AUTOBUY": "0"}
    out = []
    for r in range(repeat):
        tag = "drive" if r == 0 else f"drive_rep{r}"
        print(f"=== recording {tag}: {decisions} decisions "
              f"(autobuy={'on' if autobuy else 'OFF'})", flush=True)
        out.append(record_fixture(out_dir, decisions=decisions,
                                  port_base=port_base + 10 * r, tag=tag,
                                  extra_env=extra))
    return out


def compare_determinism(a: Fixture, b: Fixture) -> str:
    """Two recordings of the same fixture, compared on the server's own hash.

    This is the load-bearing assumption of the whole module: if the server is
    deterministic under a fixed order stream, its recorded tail IS its free run
    from any tick of that recording. `LANERL_STATEHASH` is the server's own
    FNV-1a over its canonical rows, so equal hashes mean identical states by
    construction -- no tolerance argument required.
    """
    ta, tb = load_trace(a.log), load_trace(b.log)
    n = min(len(ta), len(tb))
    first_bad = None
    same = 0
    for i in range(n):
        ha, hb = ta[i].state_hash, tb[i].state_hash
        if ha is None or hb is None:
            continue
        if ha == hb and ta[i].t_ms == tb[i].t_ms:
            same += 1
        elif first_bad is None:
            first_bad = (i, ta[i].t_ms, ha, tb[i].t_ms, hb)
    lines = [f"determinism: {len(ta)} vs {len(tb)} snapshots, "
             f"{same}/{n} hash-identical at the same game time"]
    if first_bad is None:
        lines.append("  -> IDENTICAL. The recorded tail is the server's free run.")
    else:
        i, ta_ms, ha, tb_ms, hb = first_bad
        lines.append(f"  -> DIVERGES at index {i}: {ta_ms}ms {ha} vs {tb_ms}ms {hb}")
        lines.append("  -> This module's central assumption does NOT hold for this "
                     "build. Do not quote a tier15 number until it is explained.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m lanerl_jax.parity.archive.tier15", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="record a driven fixture (boots a server)")
    rec.add_argument("--out-dir", default="lanerl_jax/runs/tier15")
    rec.add_argument("--decisions", type=int, default=9000,
                     help="30 Hz decisions; 9000 = 300 s of game time")
    rec.add_argument("--port-base", type=int, default=45100)
    rec.add_argument("--no-autobuy", action="store_true",
                     help="LANERL_AUTOBUY=0: no items, so the champion keeps "
                          "base move speed and champion kinematics are "
                          "comparable against the item-free sim")
    rec.add_argument("--repeat", type=int, default=1,
                     help="record N times and compare the server's state-hash "
                          "streams -- the determinism check this module needs")

    run = sub.add_parser("run", help="inject one tick and free-run the differential")
    run.add_argument("--fixture", default="lanerl_jax/runs/tier15",
                     help="directory produced by `record`")
    run.add_argument("--tag", default="drive")
    run.add_argument("--log", default=None,
                     help="server log path, if not <fixture>/<tag>/instance000.log")
    run.add_argument("--decisions", type=int, default=600,
                     help="window length; 600 decisions = 20 s")
    run.add_argument("--team", type=int, default=BLUE, choices=(BLUE, RED))
    run.add_argument("--predicate", default="champ_near_minion",
                     choices=("champ_near_minion", "engaged", "index", "t_ms"))
    run.add_argument("--radius", type=float, default=90.0)
    run.add_argument("--minion-type", default="any",
                     choices=("any", "melee", "caster", "cannon", "super"))
    run.add_argument("--min-attackers", type=int, default=2)
    run.add_argument("--start-index", type=int, default=None)
    run.add_argument("--start-t-ms", type=int, default=None)
    run.add_argument("--min-hp-frac", type=float, default=0.5,
                     help="refuse a start state whose champion is below this "
                          "fraction of max HP -- a champion about to die opens "
                          "a window about a respawn, not about the mechanic")
    run.add_argument("--no-stop-on-death", action="store_true",
                     help="keep running past a champion death. The window then "
                          "spans a fountain teleport and its aggregates are "
                          "not comparisons of the same situation.")
    run.add_argument("--earliest-t-ms", type=int, default=0,
                     help="ignore ticks before this game time when searching")
    run.add_argument("--champion-waypoints", default="last-move",
                     choices=("last-move", "freeze"))
    run.add_argument("--route-artifact", default=None)
    run.add_argument("--table-disabled", action="store_true",
                     help="two-point control (PATH-006). NOT gate evidence.")
    run.add_argument("--no-call-for-help", action="store_true")
    run.add_argument("--no-collision", action="store_true",
                     help="ablation control for COLL-003. NOT gate evidence.")
    run.add_argument("--on-unresolvable", default="truncate",
                     choices=("truncate", "noop"),
                     help="what to do when a recorded attack names a unit that "
                          "did not exist at injection. `truncate` (default) "
                          "ends the window; `noop` keeps going with a DIFFERENT "
                          "order stream and says so loudly in the report.")
    run.add_argument("--max-t-ms", type=int, default=None,
                     help="stop parsing the fixture at this game time. Bounds "
                          "memory on a long fixture; it also narrows the "
                          "predicate search, so it is never automatic.")
    run.add_argument("--curve-points", type=int, default=21)
    run.add_argument("--out", default=None, help="write the full curve as JSON")

    a = ap.parse_args(argv)

    if a.cmd == "record":
        fixtures = record_fixtures(Path(a.out_dir), a.decisions, a.port_base,
                                   repeat=a.repeat, autobuy=not a.no_autobuy)
        for f in fixtures:
            print(f"recorded: {f.log}")
        if len(fixtures) >= 2:
            print(compare_determinism(fixtures[0], fixtures[1]))
        return 0

    root = Path(a.fixture)
    log = Path(a.log) if a.log else root / a.tag / "instance000.log"
    fixture = Fixture(log=log, actions=root / f"{a.tag}_actions.json",
                      observations=root / f"{a.tag}_obs.jsonl")
    for p in (fixture.log, fixture.actions):
        if not p.exists():
            raise SystemExit(f"missing fixture file: {p}")

    sel_kwargs: Dict[str, object] = {
        "predicate": a.predicate, "radius": a.radius,
        "minion_type": a.minion_type, "min_attackers": a.min_attackers,
        "index": a.start_index, "t_ms": a.start_t_ms,
        "earliest_t_ms": a.earliest_t_ms, "min_hp_frac": a.min_hp_frac,
    }
    res = run_tier15(
        fixture, sel_kwargs, decisions=a.decisions, team=a.team,
        champion_waypoints=a.champion_waypoints,
        route_artifact=a.route_artifact, table_disabled=a.table_disabled,
        enable_call_for_help=not a.no_call_for_help,
        enable_collision=not a.no_collision,
        on_unresolvable=a.on_unresolvable, max_t_ms=a.max_t_ms,
        stop_on_death=not a.no_stop_on_death,
    )
    print(res.report(curve_points=a.curve_points))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res.to_json(), indent=1))
        print(f"\ncurve written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
