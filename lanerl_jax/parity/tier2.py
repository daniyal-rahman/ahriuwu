"""Tier 2: free-running divergence, characterised over the full episode.

Tier 1 (`one_step.py`) answers "is one tick correct given the server's own
state." Tier 2 answers a different question: run both engines from the same
start, give them the same action stream, never inject anything again, and
measure **when and how they part**. That is weaker evidence of a bug (nothing
here is attributable to a single tick) but it is the only question that
matters for training: an agent trained on the free-running sim has to farm on
the free-running server, so the drift between the two free-running
trajectories over a real episode is the thing that ultimately bounds transfer.

Two things this module exists to keep honest, both because they have already
gone wrong once on this project.

1. **A raw trajectory diff at long horizon is not evidence of a missing
   mechanic on its own.** `perturbation.py`'s module docstring and
   `docs/TICK_DIVERGENCE_TRACE.md` both measured that float32 is not
   associative and this lane is close to chaotic: a single, tiny difference
   compounds into a completely different siege outcome by minute five, **even
   for a perfect reimplementation**. So this module reports a **self-divergence
   floor** (:func:`run_sim_raw` with ``ulp_perturb_axis`` set) alongside the
   sim-vs-server numbers: perturb one engine's own initial state by the
   smallest representable float32 step and measure how fast it diverges from
   itself. Anything sim-vs-server that sits at or below that floor is not
   distinguishable from ordinary chaos.

2. **"Several seeds" is not automatically several samples.**
   `sweep.py`'s own docstring records the mistake directly: a 4-seed sweep of
   this exact lane came back bit-identical on both engines at every seed,
   because neither engine has any RNG on this path -- the sim's tick is a pure
   function of state, `init_lane(seed=...)` never reaches anything that
   varies (`state.key` is stored and never consumed by `step.py` or any
   module it calls -- checked directly), and the server's determinism comes
   from `bot_seed` and the fixed config, not from `run_server_episode`'s
   `seed` argument (its own docstring says so, and `bot_teams="none"` means
   `bot_seed` cannot matter either -- there is no bot to seed). A "5-seed
   Tier-2 sweep" on the idle lane would silently be n=1 five times over. What
   actually varies the experiment, following `sweep.py`'s own fix for the
   same problem, is the SCENARIO: when the champion is ordered into the wave
   and for how long. `SCENARIOS` below are genuinely different initial
   dynamics, not re-labelled copies of one run.

WHERE THE METRICS COME FROM
----------------------------
Same channel as `perturbation.py`: `sim_units()` reads `LaneState` directly;
`server_units()` reads `LanerlControl`'s own observation
(`obs["u"]`), already flowing over the TCP channel that drives the champion.
No server-log round trip, no `LANERL_STATE_DUMP` needed.

ENTITY CORRESPONDENCE, AND WHY MINIONS ARE MATCHED BY LANE RANK, NOT POSITION
------------------------------------------------------------------------------
`diff.py`'s nearest-position matching is right for Tier 1 (state is injected,
so nothing can be more than one tick of movement apart) and wrong here:
`docs/ONE_STEP_DIFFERENTIAL.md` section 3 found its own 8-unit match radius
silently right-censoring the worst 40x of "unmatched" entities, precisely
because a fixed spatial radius stops meaning anything once the two runs are
allowed to actually separate -- which is the normal, expected case by minute
two of a Tier-2 run, not a failure mode.

Lane minions of one team walk one shared, unbranching corridor
(`TOP_LANE_PATH`) and do not pass each other (collision keeps them ordered).
So correspondence is recovered by **rank along the lane**
(`lane_fraction`, ascending), not by nearest 2D position: sort each side's
same-team minions by lane fraction and pair up the ith of each. This has no
radius to censor, degrades gracefully under a population mismatch (the
smaller side's count sets how many pairs exist; the rest are counted as a
population gap, not silently dropped), and is exactly the "same queue
position" correspondence that is actually meaningful for "is the wave in the
same place." Champions need no matching (one per team). Turrets are matched
to their own known, static, patch-derived position
(`lanerl_jax.sim.init.TOP_OUTER_TURRET`) -- turrets never move on either engine, so
this is exact identity, not a heuristic.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .perturbation import (
    IdleLane,
    KillMinions,
    Perturbation,
    SceneView,
    StandInWave,
    UnitRecord,
    UnitView,
    lane_fraction,
    server_units,
    sim_units,
)
from ..sim.init import TOP_LANE_PATH, init_lane, lane_params
from ..sim.orders import OrderKind, Orders, apply_orders
from ..sim.state import Kind, Team
from ..sim.step import step_decision

__all__ = [
    "RawCurve", "run_sim_raw", "run_server_raw",
    "SCENARIOS", "compare_curves", "Tier2Report",
]

# ---------------------------------------------------------------------------
# Scenarios: genuinely different dynamics, not re-labelled seeds (see module
# docstring, item 2). Each maps to (Perturbation instance, perturbed flag).
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Tuple[Perturbation, bool]] = {
    # Zero orders, ever. The established comparison scenario
    # (`docs/TICK_DIVERGENCE_TRACE.md`, `test_minion_population_is_close_to_the_server`).
    "idle": (IdleLane(), False),
    # Champion parked in the enemy wave right after the first two waves have
    # spawned and can plausibly have met (90s + wave travel), for 10s -- a
    # very different, early-game dynamic from `idle`.
    "stand_early": (StandInWave(trigger_ms=100_000.0, hold_s=10.0), True),
    # Same perturbation, deep into the episode instead, after several wave
    # cycles have had time to build up whatever imbalance they build up.
    "stand_late": (StandInWave(trigger_ms=400_000.0, hold_s=10.0), True),
    # Champion actively fights, camped in the wave the whole game -- the one
    # scenario here that exercises the attack/damage/death path, at the cost
    # of the known sim/server AD-scaling asymmetry (`last_hit_drive.py`).
    "kill": (KillMinions(trigger_ms=150_000.0, n_targets=3, timeout_ms=60_000.0), True),
}

_SERVER_TEAM = {Team.BLUE: 100, Team.RED: 200}
_TEAM_NAME = {100: "blue", 200: "red"}


# ---------------------------------------------------------------------------
# Raw per-decision unit capture, one engine at a time
# ---------------------------------------------------------------------------

def _record_to_tuple(u: UnitRecord) -> Tuple[str, int, float, float, float]:
    return (u.kind, u.team, u.x, u.y, u.hp)


def _tuple_to_record(t: Sequence) -> UnitRecord:
    return UnitRecord(kind=t[0], team=int(t[1]), x=float(t[2]), y=float(t[3]), hp=float(t[4]))


@dataclass
class RawCurve:
    """One episode's raw unit lists, sampled on a time grid. Unlike
    `perturbation.ResponseCurve`, this keeps every live unit's (kind, team,
    x, y, hp) at each sample instead of collapsing to aggregate metrics --
    Tier 2 needs the raw entities to do its own rank-based matching against
    the other engine."""

    engine: str
    scenario: str
    seed: int
    decisions: int
    sample_every: int
    perturb_note: str = ""
    wall_s: float = 0.0
    t_s: List[float] = field(default_factory=list)
    units: List[List[Tuple[str, int, float, float, float]]] = field(default_factory=list)

    def units_at(self, i: int) -> List[UnitRecord]:
        return [_tuple_to_record(t) for t in self.units[i]]

    def to_json(self) -> dict:
        return {
            "engine": self.engine, "scenario": self.scenario, "seed": self.seed,
            "decisions": self.decisions, "sample_every": self.sample_every,
            "perturb_note": self.perturb_note, "wall_s": self.wall_s,
            "t_s": self.t_s, "units": self.units,
        }

    @staticmethod
    def from_json(d: dict) -> "RawCurve":
        return RawCurve(
            engine=d["engine"], scenario=d["scenario"], seed=int(d["seed"]),
            decisions=int(d["decisions"]), sample_every=int(d["sample_every"]),
            perturb_note=d.get("perturb_note", ""), wall_s=float(d.get("wall_s", 0.0)),
            t_s=list(d["t_s"]),
            units=[[tuple(u) for u in row] for row in d["units"]],
        )


def run_sim_raw(perturbation: Perturbation, perturbed: bool, seed: int,
                 decisions: int, sample_every: int = 60,
                 enable_call_for_help: bool = False,
                 ulp_perturb_axis: Optional[str] = None,
                 ulp_perturb_unit: Optional[int] = None,
                 ulp_perturb_dir: int = 1,
                 ulp_perturb_at_decision: int = 0) -> RawCurve:
    """Drive ``perturbation`` against the JAX sim, recording every live
    unit's (kind, team, x, y, hp) at each sample.

    ``ulp_perturb_axis``, when given (``"x"`` or ``"y"``), nudges
    ``state.<axis>[ulp_perturb_unit]`` by exactly one float32 ULP
    (`numpy.nextafter`) at decision ``ulp_perturb_at_decision`` (default 0,
    i.e. right after `init_lane`, before the first decision). This is the
    smallest representable difference two "identical" states can have -- the
    chaos-floor probe (module docstring, item 1). Everything else about the
    run is byte-identical to an unperturbed one with the same seed and
    scenario.

    ``ulp_perturb_unit=None`` (the default whenever a perturbation is
    requested) means "the first live LaneMinion, team blue, at the moment of
    perturbation" -- picked at runtime rather than hardcoded, because which
    slot a minion occupies depends on spawn order and is not worth
    memorising. Perturbing a champion at t=0 in a scenario where the champion
    never moves (``idle``) is a null test by construction: nothing downstream
    ever reads that coordinate differently, so it cannot show chaos even if
    the underlying dynamics have some. A live minion mid-wave is coupled to
    collision and targeting on every tick, which is what actually gives a
    chaos probe a chance to find something.
    """
    import jax
    import jax.numpy as jnp

    params_tbl = lane_params()
    params_np = {k: np.asarray(v) for k, v in params_tbl.items()}
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))
    state = init_lane(seed=seed)
    note = ""

    def _apply_ulp(state):
        unit = ulp_perturb_unit
        if unit is None:
            kind = np.asarray(state.kind)
            team = np.asarray(state.team)
            alive = np.asarray(state.alive)
            cand = np.flatnonzero((kind == Kind.LANE_MINION) & (team == Team.BLUE) & alive)
            if cand.size == 0:
                raise RuntimeError("ulp_perturb_unit=None but no live blue "
                                   "LaneMinion exists at the perturbation decision")
            unit = int(cand[0])
        arr = getattr(state, ulp_perturb_axis)
        old = float(np.asarray(arr)[unit])
        target = np.inf if ulp_perturb_dir > 0 else -np.inf
        new = float(np.nextafter(np.float32(old), np.float32(target)))
        state = state.replace(**{ulp_perturb_axis: arr.at[unit].set(new)})
        nonlocal note
        note = (f"ulp_perturb {ulp_perturb_axis}[{unit}] @decision={ulp_perturb_at_decision} "
                f"{old!r} -> {new!r} (delta {new - old:.3e})")
        return state

    if ulp_perturb_axis is not None and ulp_perturb_at_decision == 0:
        state = _apply_ulp(state)

    @jax.jit
    def _step(state, kind, x, y, target):
        orders = Orders(
            kind=jnp.array([kind, OrderKind.NOOP], dtype=jnp.int8),
            x=jnp.array([x, 0.0], dtype=state.x.dtype),
            y=jnp.array([y, 0.0], dtype=state.y.dtype),
            target=jnp.array([target, -1], dtype=jnp.int8),
        )
        return step_decision(apply_orders(state, orders), params_tbl, lane_path=path,
                             enable_call_for_help=enable_call_for_help)

    script = perturbation.new_script()
    t_s: List[float] = []
    units: List[List[Tuple]] = []
    t0 = time.perf_counter()
    for i in range(decisions):
        if ulp_perturb_axis is not None and i == ulp_perturb_at_decision and i > 0:
            state = _apply_ulp(state)
        kind = np.asarray(state.kind)
        team = np.asarray(state.team)
        alive = np.asarray(state.alive)
        x = np.asarray(state.x)
        y = np.asarray(state.y)
        hp = np.asarray(state.hp)
        model = np.asarray(state.model)

        enemy_idx = np.flatnonzero((kind == Kind.LANE_MINION) & (team == Team.RED) & alive)
        enemy = tuple(
            UnitView(uid=int(j), x=float(x[j]), y=float(y[j]), hp=float(hp[j]))
            for j in enemy_idx)
        scene = SceneView(
            t_ms=float(state.t_ms), decision=i,
            champ_x=float(x[0]), champ_y=float(y[0]), champ_alive=bool(alive[0]),
            attack_range=float(params_np["attack_range"][model[0]]), enemy=enemy)

        act = perturbation.decide(scene, perturbed, script)
        if act.kind == "noop":
            state = _step(state, OrderKind.NOOP, 0.0, 0.0, -1)
        elif act.kind == "move":
            state = _step(state, OrderKind.MOVE, act.x, act.y, -1)
        elif act.kind == "attack":
            state = _step(state, OrderKind.ATTACK, 0.0, 0.0, int(act.target))
        else:
            raise ValueError(f"unknown ActionSpec.kind {act.kind!r}")

        if i % sample_every == 0:
            t_s.append(float(state.t_ms) / 1000.0)
            units.append([_record_to_tuple(u) for u in sim_units(state)])
    wall_s = time.perf_counter() - t0
    return RawCurve(engine="sim", scenario=perturbation.name, seed=seed,
                     decisions=decisions, sample_every=sample_every,
                     perturb_note=note, wall_s=wall_s, t_s=t_s, units=units)


def run_server_raw(perturbation: Perturbation, perturbed: bool, seed: int,
                    decisions: int, sample_every: int = 60,
                    bot_seed: int = 4242, port_base: Optional[int] = None,
                    log_dir: Optional[Path] = None) -> RawCurve:
    """Drive ``perturbation`` against one real server instance, recording raw
    units. Same boot as `perturbation.run_server_episode`; HARD CONSTRAINT:
    exactly one server instance, never launched concurrently with another."""
    from lanerl_train.ports import PortAllocator
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

    log_dir = Path(log_dir) if log_dir is not None else Path(
        tempfile.mkdtemp(prefix="tier2_"))
    ports = (PortAllocator(base=port_base) if port_base else PortAllocator()).allocate(1)
    env = VecLaneEnv(
        1,
        spec=ServerLaunchSpec(toponly=True, bot_teams="none", bot_seed=bot_seed, step_ticks=2),
        log_dir=log_dir, ports=ports, step_timeout_s=180.0, auto_restart=False,
    )
    env.start()
    script = perturbation.new_script()
    t_s: List[float] = []
    units: List[List[Tuple]] = []
    t0 = time.perf_counter()
    try:
        if not all(env.alive):
            raise RuntimeError(f"server failed to boot: {env.alive}")
        for i in range(decisions):
            obs = env.last_obs[0]
            if obs is None:
                raise RuntimeError(f"no observation at decision {i}")
            raw_units = obs.get("u", [])
            blue = next(u for u in raw_units if u.get("k") == "Champion" and u.get("tm") == 100)
            enemy = tuple(
                UnitView(uid=int(u["id"]), x=float(u["x"]), y=float(u["y"]), hp=float(u["hp"]))
                for u in raw_units if u.get("k") == "LaneMinion" and u.get("tm") == 200)
            scene = SceneView(
                t_ms=float(obs.get("t", 0)), decision=i,
                champ_x=float(blue["x"]), champ_y=float(blue["y"]),
                champ_alive=float(blue.get("hp", 1)) > 0,
                attack_range=float(blue["rng"]), enemy=enemy)

            act = perturbation.decide(scene, perturbed, script)
            if act.kind == "noop":
                a = {"t": "noop"}
            elif act.kind == "move":
                a = {"t": "move", "x": act.x, "y": act.y}
            elif act.kind == "attack":
                a = {"t": "attack", "id": int(act.target)}
            else:
                raise ValueError(f"unknown ActionSpec.kind {act.kind!r}")
            env.step([{"blue": a}])

            if i % sample_every == 0:
                obs2 = env.last_obs[0]
                t_s.append(float(obs2.get("t", 0)) / 1000.0)
                units.append([_record_to_tuple(u) for u in server_units(obs2)])
    finally:
        env.close()
    wall_s = time.perf_counter() - t0
    return RawCurve(engine="server", scenario=perturbation.name, seed=seed,
                     decisions=decisions, sample_every=sample_every,
                     wall_s=wall_s, t_s=t_s, units=units)


# ---------------------------------------------------------------------------
# Cross-engine comparison: matching, per-sample metrics, aggregation
# ---------------------------------------------------------------------------

def _match_champion(a: Sequence[UnitRecord], b: Sequence[UnitRecord],
                     team: int) -> Optional[Tuple[float, float, float, float]]:
    """-> (pos_err, hp_err_signed(a-b), hp_a, hp_b), or None if either side
    has no live champion of this team (a death, which cannot happen to a
    champion in this ruleset, or a data gap)."""
    ua = next((u for u in a if u.kind == "Champion" and u.team == team), None)
    ub = next((u for u in b if u.kind == "Champion" and u.team == team), None)
    if ua is None or ub is None:
        return None
    return (math.hypot(ua.x - ub.x, ua.y - ub.y), ua.hp - ub.hp, ua.hp, ub.hp)


def _match_turret(a: Sequence[UnitRecord], b: Sequence[UnitRecord],
                   team: int, tx: float, ty: float) -> Tuple[float, float, float, bool, bool]:
    """-> (hp_err_signed(a-b), hp_a, hp_b, alive_a, alive_b). A turret absent
    from the live list at its known static position is destroyed (turrets
    never move, so "missing" cannot mean anything else here)."""
    def find(units: Sequence[UnitRecord]) -> Optional[UnitRecord]:
        best, best_d2 = None, None
        for u in units:
            if u.kind != "LaneTurret" or u.team != team:
                continue
            d2 = (u.x - tx) ** 2 + (u.y - ty) ** 2
            if d2 > 2500.0:   # 50 world units; turrets are static, see docstring
                continue
            if best_d2 is None or d2 < best_d2:
                best_d2, best = d2, u
        return best

    ua, ub = find(a), find(b)
    hp_a = ua.hp if ua is not None else 0.0
    hp_b = ub.hp if ub is not None else 0.0
    return (hp_a - hp_b, hp_a, hp_b, ua is not None, ub is not None)


def _match_minions_by_rank(a: Sequence[UnitRecord], b: Sequence[UnitRecord],
                            team: int) -> Tuple[List[Tuple[float, float]], int, int]:
    """Quantile-along-the-lane correspondence -- see module docstring for why
    rank replaces nearest-position matching here. -> (list of (pos_err,
    hp_err), n_a, n_b).

    Matched by PROPORTIONAL rank (quantile), not absolute index: a wave
    spawns every ~30s and the two sides' live counts routinely differ by a
    handful (measured on the idle scenario: population gap p95 9, max 12,
    against typical live counts in the 10-25 range -- not a rare edge case).
    Absolute-index matching would let a mere population difference shift
    every downstream pairing by that many slots and get reported as a
    "position error" it is not. Quantile matching instead asks "is the
    front-most minion near the front-most, the median minion near the
    median" -- the shape-of-the-queue comparison that is actually meaningful
    when the two counts differ. Reduces to plain rank matching when the
    counts are equal.
    """
    ma = [u for u in a if u.kind == "LaneMinion" and u.team == team]
    mb = [u for u in b if u.kind == "LaneMinion" and u.team == team]

    def frac(u: UnitRecord) -> float:
        return float(lane_fraction(u.x, u.y)[0])

    ma.sort(key=frac)
    mb.sort(key=frac)
    na, nb = len(ma), len(mb)
    n = min(na, nb)
    out = []
    for k in range(n):
        ia = min(na - 1, (k * na) // n)
        ib = min(nb - 1, (k * nb) // n)
        out.append((math.hypot(ma[ia].x - mb[ib].x, ma[ia].y - mb[ib].y),
                     ma[ia].hp - mb[ib].hp))
    return out, na, nb


@dataclass
class Tier2Sample:
    t_s: float
    champ_pos_err: Dict[str, Optional[float]]
    champ_hp_err: Dict[str, Optional[float]]
    turret_hp_err: Dict[str, float]
    turret_alive_a: Dict[str, bool]
    turret_alive_b: Dict[str, bool]
    minion_pos_err: List[float]     # both teams pooled
    minion_hp_err: List[float]
    n_minions_a: Dict[str, int]
    n_minions_b: Dict[str, int]


def _compare_sample(a: Sequence[UnitRecord], b: Sequence[UnitRecord], t_s: float) -> Tier2Sample:
    champ_pos, champ_hp = {}, {}
    for team, name in _TEAM_NAME.items():
        r = _match_champion(a, b, team)
        champ_pos[name] = r[0] if r else None
        champ_hp[name] = r[1] if r else None

    # Top-lane outer turrets only, matching every prior report in this tree
    # (PERTURBATION_RESPONSE.md, TICK_DIVERGENCE_TRACE.md) -- the pair either
    # side of a 1v1 top lane can plausibly reach in one episode.
    turret_hp_err, turret_alive_a, turret_alive_b = {}, {}, {}
    from ..sim.init import TOP_OUTER_TURRET
    for team_enum, name in ((Team.BLUE, "blue"), (Team.RED, "red")):
        tx, ty = TOP_OUTER_TURRET[team_enum]
        server_team = _SERVER_TEAM[team_enum]
        e, hp_a, hp_b, alive_a, alive_b = _match_turret(a, b, server_team, tx, ty)
        turret_hp_err[name] = e
        turret_alive_a[name] = alive_a
        turret_alive_b[name] = alive_b

    minion_pos: List[float] = []
    minion_hp: List[float] = []
    n_a: Dict[str, int] = {}
    n_b: Dict[str, int] = {}
    for team, name in _TEAM_NAME.items():
        pairs, na, nb = _match_minions_by_rank(a, b, team)
        n_a[name], n_b[name] = na, nb
        minion_pos.extend(p for p, _ in pairs)
        minion_hp.extend(h for _, h in pairs)

    return Tier2Sample(
        t_s=t_s, champ_pos_err=champ_pos, champ_hp_err=champ_hp,
        turret_hp_err=turret_hp_err, turret_alive_a=turret_alive_a, turret_alive_b=turret_alive_b,
        minion_pos_err=minion_pos, minion_hp_err=minion_hp,
        n_minions_a=n_a, n_minions_b=n_b,
    )


#: First-divergence thresholds. Position in world units, HP in HP. Chosen as
#: "clearly more than one decision's worth of ordinary movement/quantisation",
#: not tuned to pass -- see docs/TIER2_DIVERGENCE.md for the values actually
#: observed and how they compare.
DIVERGENCE_THRESHOLDS = {
    "champion_pos": 5.0,
    "champion_hp": 1.0,
    "turret_hp": 1.0,
    "minion_pos": 25.0,
    "minion_count": 1,
}


@dataclass
class Tier2Report:
    scenario: str
    a_engine: str
    b_engine: str
    n_samples: int
    samples: List[Tier2Sample] = field(default_factory=list)
    first_divergence: Optional[Dict] = None

    def summarize(self) -> Dict:
        def stat(vals: List[float]) -> Dict[str, float]:
            if not vals:
                return {"median": float("nan"), "p95": float("nan"), "max": float("nan"),
                        "mean_signed": float("nan"), "n": 0}
            arr = np.asarray(vals, dtype=np.float64)
            return {"median": float(np.median(np.abs(arr))),
                    "p95": float(np.percentile(np.abs(arr), 95)),
                    "max": float(np.max(np.abs(arr))),
                    "mean_signed": float(np.mean(arr)), "n": int(arr.size)}

        champ_pos = {n: [] for n in _TEAM_NAME.values()}
        champ_hp = {n: [] for n in _TEAM_NAME.values()}
        turret_hp = {n: [] for n in _TEAM_NAME.values()}
        minion_pos_all: List[float] = []
        minion_hp_all: List[float] = []
        pop_gap_blue: List[float] = []
        pop_gap_red: List[float] = []
        for s in self.samples:
            for n in _TEAM_NAME.values():
                if s.champ_pos_err[n] is not None:
                    champ_pos[n].append(s.champ_pos_err[n])
                    champ_hp[n].append(s.champ_hp_err[n])
                turret_hp[n].append(s.turret_hp_err[n])
            minion_pos_all.extend(s.minion_pos_err)
            minion_hp_all.extend(s.minion_hp_err)
            pop_gap_blue.append(s.n_minions_a["blue"] - s.n_minions_b["blue"])
            pop_gap_red.append(s.n_minions_a["red"] - s.n_minions_b["red"])

        return {
            "scenario": self.scenario, "a_engine": self.a_engine, "b_engine": self.b_engine,
            "n_samples": self.n_samples,
            "champion_pos_err": {n: stat(v) for n, v in champ_pos.items()},
            "champion_hp_err": {n: stat(v) for n, v in champ_hp.items()},
            "turret_hp_err": {n: stat(v) for n, v in turret_hp.items()},
            "minion_pos_err": stat(minion_pos_all),
            "minion_hp_err": stat(minion_hp_all),
            "population_gap": {"blue": stat(pop_gap_blue), "red": stat(pop_gap_red)},
            "first_divergence": self.first_divergence,
        }

    def time_series(self, bucket_s: float = 60.0) -> List[Dict]:
        """Bucketed median/p95/max of minion position error and turret HP
        error over time -- the "does it saturate or grow" table."""
        out = []
        if not self.samples:
            return out
        t_max = self.samples[-1].t_s
        n_buckets = int(t_max // bucket_s) + 1
        for k in range(n_buckets):
            lo, hi = k * bucket_s, (k + 1) * bucket_s
            in_bucket = [s for s in self.samples if lo <= s.t_s < hi]
            if not in_bucket:
                continue
            pos = [p for s in in_bucket for p in s.minion_pos_err]
            row = {
                "t_s": lo,
                "minion_pos_median": float(np.median(pos)) if pos else float("nan"),
                "minion_pos_p95": float(np.percentile(pos, 95)) if pos else float("nan"),
                "minion_pos_max": float(np.max(pos)) if pos else float("nan"),
                "turret_hp_err_blue": in_bucket[-1].turret_hp_err["blue"],
                "turret_hp_err_red": in_bucket[-1].turret_hp_err["red"],
                "n_minions_a_blue": in_bucket[-1].n_minions_a["blue"],
                "n_minions_b_blue": in_bucket[-1].n_minions_b["blue"],
                "n_minions_a_red": in_bucket[-1].n_minions_a["red"],
                "n_minions_b_red": in_bucket[-1].n_minions_b["red"],
            }
            out.append(row)
        return out


def compare_curves(a: RawCurve, b: RawCurve) -> Tier2Report:
    """Compare two `RawCurve`s sample-for-sample. ``a`` and ``b`` may be two
    different engines (sim vs server -- the headline comparison) or the same
    engine run twice with a tiny initial perturbation (the chaos-floor probe).
    Aligned by index: both curves are driven decision-for-decision from the
    same scenario at the same ``sample_every``, so their time grids match."""
    n = min(len(a.t_s), len(b.t_s))
    report = Tier2Report(scenario=a.scenario, a_engine=a.engine, b_engine=b.engine,
                          n_samples=n)
    for i in range(n):
        ua = a.units_at(i)
        ub = b.units_at(i)
        sample = _compare_sample(ua, ub, a.t_s[i])
        report.samples.append(sample)
        if report.first_divergence is None:
            found = _check_divergence(sample)
            if found is not None:
                report.first_divergence = found
    return report


def _check_divergence(s: Tier2Sample) -> Optional[Dict]:
    for name in ("blue", "red"):
        if s.champ_pos_err[name] is not None and s.champ_pos_err[name] > DIVERGENCE_THRESHOLDS["champion_pos"]:
            return {"t_s": s.t_s, "class": "champion", "team": name, "field": "position",
                    "value": s.champ_pos_err[name]}
        if s.champ_hp_err[name] is not None and abs(s.champ_hp_err[name]) > DIVERGENCE_THRESHOLDS["champion_hp"]:
            return {"t_s": s.t_s, "class": "champion", "team": name, "field": "hp",
                    "value": s.champ_hp_err[name]}
        if abs(s.turret_hp_err[name]) > DIVERGENCE_THRESHOLDS["turret_hp"]:
            return {"t_s": s.t_s, "class": "turret", "team": name, "field": "hp",
                    "value": s.turret_hp_err[name]}
        if abs(s.n_minions_a[name] - s.n_minions_b[name]) >= DIVERGENCE_THRESHOLDS["minion_count"]:
            return {"t_s": s.t_s, "class": "minion", "team": name, "field": "population",
                    "value": s.n_minions_a[name] - s.n_minions_b[name]}
    if s.minion_pos_err and max(s.minion_pos_err) > DIVERGENCE_THRESHOLDS["minion_pos"]:
        return {"t_s": s.t_s, "class": "minion", "team": "either", "field": "position",
                "value": max(s.minion_pos_err)}
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_raw(args) -> None:
    perturbation, perturbed = SCENARIOS[args.scenario]
    decisions = int(round(args.minutes * 60.0 * 30.0))
    sample_every = max(1, int(round(args.sample_every_s * 30.0)))
    if args.engine == "sim":
        curve = run_sim_raw(perturbation, perturbed, args.seed, decisions, sample_every,
                             enable_call_for_help=args.call_for_help,
                             ulp_perturb_axis=args.ulp_axis,
                             ulp_perturb_unit=args.ulp_unit,
                             ulp_perturb_dir=args.ulp_dir,
                             ulp_perturb_at_decision=args.ulp_decision)
    else:
        curve = run_server_raw(perturbation, perturbed, args.seed, decisions, sample_every,
                                bot_seed=args.bot_seed, port_base=args.port_base)
    Path(args.out).write_text(json.dumps(curve.to_json()))
    print(f"{args.engine} {args.scenario}: {len(curve.t_s)} samples over "
          f"{decisions} decisions, wall={curve.wall_s:.1f}s -> {args.out}"
          + (f"  [{curve.perturb_note}]" if curve.perturb_note else ""))


def _cli_compare(args) -> None:
    a = RawCurve.from_json(json.loads(Path(args.a).read_text()))
    b = RawCurve.from_json(json.loads(Path(args.b).read_text()))
    report = compare_curves(a, b)
    out = {"summary": report.summarize(), "time_series": report.time_series(args.bucket_s)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    print(f"-> {args.out}")


def _run_cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_raw = sub.add_parser("raw", help="run one engine's raw episode")
    p_raw.add_argument("--engine", choices=["sim", "server"], required=True)
    p_raw.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    p_raw.add_argument("--seed", type=int, default=0)
    p_raw.add_argument("--minutes", type=float, default=10.0)
    p_raw.add_argument("--sample-every-s", type=float, default=2.0)
    p_raw.add_argument("--bot-seed", type=int, default=4242)
    p_raw.add_argument("--port-base", type=int, default=None)
    p_raw.add_argument("--call-for-help", action="store_true")
    p_raw.add_argument("--ulp-axis", choices=["x", "y"], default=None,
                        help="sim only: nudge state.<axis>[--ulp-unit] by one "
                             "float32 ULP at --ulp-decision (chaos-floor probe)")
    p_raw.add_argument("--ulp-unit", type=int, default=None,
                        help="slot index to perturb; omit for \"first live "
                             "blue LaneMinion at --ulp-decision\"")
    p_raw.add_argument("--ulp-dir", type=int, default=1, choices=[1, -1])
    p_raw.add_argument("--ulp-decision", type=int, default=0,
                        help="decision index to apply the ULP nudge at "
                             "(0 = at init, before the first decision)")
    p_raw.add_argument("--out", type=str, required=True)
    p_raw.set_defaults(func=_cli_raw)

    p_cmp = sub.add_parser("compare", help="compare two raw curves")
    p_cmp.add_argument("--a", type=str, required=True)
    p_cmp.add_argument("--b", type=str, required=True)
    p_cmp.add_argument("--bucket-s", type=float, default=60.0)
    p_cmp.add_argument("--out", type=str, required=True)
    p_cmp.set_defaults(func=_cli_compare)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    _run_cli()
