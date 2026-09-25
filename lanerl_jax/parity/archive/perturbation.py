"""Perturbation-response parity: does the JAX sim react to a poke the way the
C# server does?

THE PROBLEM A RAW TRAJECTORY DIFF CANNOT SEE
----------------------------------------------
A lane is near-chaotic: float32 is not associative, and small per-tick
differences amplify over a long horizon (`docs/TICK_DIVERGENCE_TRACE.md`
measured this directly -- a single spawn-coordinate bug compounds into a
completely different siege outcome by five minutes). That means comparing a
sim trajectory against a server trajectory at, say, t = 40 min will show large
differences **even for a perfect reimplementation**, because chaotic
amplification alone produces them. A raw trajectory diff at long horizon is
therefore not evidence of a missing mechanic -- it is evidence that float32
addition is not associative, which everyone already knew.

What a raw diff also cannot see: a class of bug where the sim has no
*restoring force* somewhere the server has one (the canonical example here is
`docs/TARGET_ACQUISITION_DIFF.md`'s missing call-for-help -- the server
recycles a stray minion off a champion and back onto the wave every few
seconds; the sim has no such recycling, so an independently-acquired minion
just stays locked on). That kind of gap does not show up as "field X is a bit
off at tick Y" -- it shows up as a large-scale divergence in wave position,
minion population and turret HP that only becomes visible over a long window.

THE FIX: COMPARE RESPONSES, NOT TRAJECTORIES
----------------------------------------------
Each implementation against **its own** baseline::

    R_server = f(server, perturbed) - f(server, baseline)
    R_sim    = f(sim,    perturbed) - f(sim,    baseline)
    report R_sim vs R_server

Chaotic divergence is common-mode between a baseline and a perturbed run of
the *same* implementation with the *same* seed -- both runs accumulate the
same float32 drift right up until the perturbation acts, so it largely cancels
in the difference. A missing mechanic does not cancel: it changes what happens
*after* the perturbation in a way that is specific to the perturbation, not to
the RNG seed. Every number this module reports is a response, or a
distribution over responses -- never a raw sim-vs-server trajectory diff.

THE NULL CONTROL IS NOT OPTIONAL
----------------------------------
"No perturbation, run twice" measures the floor: how much response two
nominally identical runs of *one* implementation already show, from nothing.
Every real perturbation's response has to clear this floor before it means
anything. See `docs/PERTURBATION_RESPONSE.md` for the measured floor and
whether it is small enough at the pilot's 10-minute horizon to trust the rest
of this module's output.

Both champions in `lanerl/cfg/garen1v1.json` are Garen (`"map": 1`, confirmed
at line 132) -- the same config the whole project trains and evals against.
Red never receives an order in any of these runs, matching
`docs/TICK_DIVERGENCE_TRACE.md` and `lanerl_jax.parity.archive.last_hit_drive`'s own
setup: only blue (slot 0 in the sim, `tm == 100` on the wire) is scripted.

WHERE THE METRICS COME FROM
----------------------------
`docs/TICK_DIVERGENCE_TRACE.md` and `lanerl_jax/parity/trace.py` parse the
server's `LANERL_STATE_DUMP` log -- built for a different job (bit-exact
per-tick diffing) and requiring `LANERL_STATE_DUMP_FULL=1` plus a log-file
round-trip. This module does not need that: `LanerlControl`'s own observation
(`lanerl_jax.parity.archive.last_hit_drive.run_oracle_on_server`'s `obs["u"]`) already
carries every live `AttackableUnit` -- champions, lane minions *and* lane
turrets, with `k`/`tm`/`x`/`y`/`hp` -- once per decision, over the same TCP
channel already used to drive the champion. Dead units are simply absent (the
engine removes them from `ObjectManager`), so no `alive` bookkeeping is needed
on the server side either. Reusing the control channel keeps this module to
one code path per side instead of two (drive orders over TCP, then separately
parse a log file), and it costs nothing extra: the observation was already
being read every decision.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ...sim.config import SimConfig
from ...sim.init import TOP_LANE_PATH, TOP_OUTER_TURRET, init_lane
from ...sim.orders import OrderKind, Orders
from ...sim.state import Kind, Team
from ...sim.step import env_step

__all__ = [
    "lane_fraction",
    "UnitRecord", "UnitView", "SceneView", "ActionSpec",
    "sim_units", "server_units", "compute_metrics",
    "CAMP_POINT", "ENGAGE_POINT",
    "Perturbation", "NullControl", "IdleLane", "StandInWave", "KillMinions",
    "ResponseCurve", "run_sim_episode", "run_server_episode",
    "response", "curve_to_dict", "curve_from_dict",
    "summarize_response",
]

# ---------------------------------------------------------------------------
# Lane-fraction projection
# ---------------------------------------------------------------------------

_PATH = np.asarray(TOP_LANE_PATH, dtype=np.float64)          # (11, 2)
_SEG = _PATH[1:] - _PATH[:-1]                                  # (10, 2)
_SEG_LEN = np.linalg.norm(_SEG, axis=1)                        # (10,)
_SEG_LEN2_SAFE = np.where(_SEG_LEN > 1e-9, _SEG_LEN, 1.0) ** 2
_CUM = np.concatenate([[0.0], np.cumsum(_SEG_LEN)])            # (11,)
_TOTAL_LEN = float(_CUM[-1])


def lane_fraction(x, y) -> np.ndarray:
    """Project ``(x, y)`` onto :data:`TOP_LANE_PATH`; arc-length fraction in
    ``[0, 1]``, 0 at blue's barracks end, 1 at red's.

    Nearest-segment projection (the standard point-to-polyline construction),
    vectorized over an arbitrary-shape array of points. Direction-agnostic on
    purpose: a red minion walks the reversed polyline and a blue minion walks
    it forward, but the same physical location must map to the same fraction
    for "mean lane fraction of live minions" to mean "where the battlefront
    is" rather than "which team is closer to done walking its own path".

    No projection helper of this shape exists elsewhere in the tree (checked
    `lanerl_jax/parity/sim_vs_server.py`, `lanerl_jax/obs/frame.py` -- the
    latter's `(s, n)` frame is turret-to-turret, not the minion polyline, and
    is used for the observation encoding, not for a scalar "how far down the
    lane" summary); this is a new, self-contained implementation.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    p = np.stack([x, y], axis=-1)
    best_d2 = np.full(p.shape[:-1], np.inf)
    best_s = np.zeros(p.shape[:-1])
    for i in range(_SEG.shape[0]):
        d = _SEG[i]
        w = p - _PATH[i]
        t = (w[..., 0] * d[0] + w[..., 1] * d[1]) / _SEG_LEN2_SAFE[i]
        t = np.clip(t, 0.0, 1.0)
        proj = _PATH[i] + t[..., None] * d
        d2 = np.sum((p - proj) ** 2, axis=-1)
        better = d2 < best_d2
        best_d2 = np.where(better, d2, best_d2)
        best_s = np.where(better, _CUM[i] + t * _SEG_LEN[i], best_s)
    return best_s / _TOTAL_LEN if _TOTAL_LEN > 0 else best_s


# ---------------------------------------------------------------------------
# A canonical unit record both engines can produce, and the response metrics
# ---------------------------------------------------------------------------

class UnitRecord(NamedTuple):
    """One live unit, in the server's own vocabulary, from either engine."""

    kind: str    # "Champion" | "LaneMinion" | "LaneTurret"
    team: int    # 100 (blue) | 200 (red)
    x: float
    y: float
    hp: float


_KIND_NAME = {Kind.CHAMPION: "Champion", Kind.LANE_MINION: "LaneMinion",
              Kind.TURRET: "LaneTurret"}
_SERVER_TEAM = {Team.BLUE: 100, Team.RED: 200, Team.NEUTRAL: 300}
#: known-good outer-turret positions, keyed by the SERVER's team numbering
#: (100/200), so both adapters below can use one lookup.
_OUTER_TURRET_XY = {100: TOP_OUTER_TURRET[Team.BLUE], 200: TOP_OUTER_TURRET[Team.RED]}


def sim_units(state) -> List[UnitRecord]:
    """:class:`~lanerl_jax.sim.state.LaneState` -> live :class:`UnitRecord`\\ s.

    Mirrors `lanerl_jax.parity.archive.sim_vs_server.state_to_snapshot`'s entity
    filter (``kind != NONE`` and ``alive``) and kind/team naming, without
    building a full `Snapshot`/`Entity` (this module needs four scalar fields,
    not the whole parity-trace row shape).
    """
    kind = np.asarray(state.kind)
    team = np.asarray(state.team)
    alive = np.asarray(state.alive)
    x = np.asarray(state.x)
    y = np.asarray(state.y)
    hp = np.asarray(state.hp)
    out: List[UnitRecord] = []
    for i in np.flatnonzero((kind != Kind.NONE) & alive):
        i = int(i)
        name = _KIND_NAME.get(int(kind[i]))
        if name is None:
            continue
        out.append(UnitRecord(kind=name, team=_SERVER_TEAM[int(team[i])],
                               x=float(x[i]), y=float(y[i]), hp=float(hp[i])))
    return out


def server_units(obs: Mapping) -> List[UnitRecord]:
    """``LanerlControl``'s observation dict -> live :class:`UnitRecord`\\ s.

    No ``alive`` filter needed: a dead unit is removed from the server's
    `ObjectManager` and simply does not appear in `obs["u"]` again.
    """
    out: List[UnitRecord] = []
    for u in obs.get("u", []):
        k = u.get("k")
        if k not in ("Champion", "LaneMinion", "LaneTurret"):
            continue
        out.append(UnitRecord(kind=k, team=int(u["tm"]), x=float(u["x"]),
                               y=float(u["y"]), hp=float(u["hp"])))
    return out


def _nearest_outer_turret_hp(units: Sequence[UnitRecord], team: int) -> float:
    """HP of the outer turret at :data:`_OUTER_TURRET_XY[team]`, or ``0.0`` if
    it has been destroyed.

    Turrets are static -- they never move -- so the only two possibilities for
    "no `LaneTurret` of this team sits at the known position" are (a) it is
    dead and the engine has removed it from the live-unit list, which is a
    real, reportable event (the turret took lethal siege damage), or (b) the
    position/team mapping itself is wrong. Distinguishing them from geometry
    alone is not possible, so this treats every miss as (a) -- destroyed,
    HP 0 -- on the strength of an independent check: `ALL_TURRETS` and
    `TOP_OUTER_TURRET` in `lanerl_jax/sim/init.py` agree to within a rounding
    error (574.625 vs 574.6, 3911.6875 vs 3911.7), and every episode's t=0
    sample (before any wave can possibly reach a turret) matches at native
    float precision -- see `docs/PERTURBATION_RESPONSE.md`. A 10-minute idle
    lane genuinely can lose an outer turret: `docs/TICK_DIVERGENCE_TRACE.md`'s
    300 s idle-lane run already measured one at 60% HP and falling.
    """
    tx, ty = _OUTER_TURRET_XY[team]
    best_hp = None
    best_d2 = None
    for u in units:
        if u.kind != "LaneTurret" or u.team != team:
            continue
        d2 = (u.x - tx) ** 2 + (u.y - ty) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2, best_hp = d2, u.hp
    if best_d2 is None or best_d2 > 2500.0:   # 50 world units
        return 0.0
    return best_hp


def compute_metrics(units: Sequence[UnitRecord], t_s: float) -> Dict[str, float]:
    """One time-grid sample: everything the pilot reports a response for.

    * mean lane fraction of live minions (all, and split by team)
    * per-team live minion counts
    * both outer turrets' HP
    """
    minions = [u for u in units if u.kind == "LaneMinion"]
    blue_m = [u for u in minions if u.team == 100]
    red_m = [u for u in minions if u.team == 200]

    def mean_frac(ms: Sequence[UnitRecord]) -> float:
        if not ms:
            return float("nan")
        xs = np.array([m.x for m in ms])
        ys = np.array([m.y for m in ms])
        return float(np.mean(lane_fraction(xs, ys)))

    return {
        "t_s": float(t_s),
        "n_minions_blue": float(len(blue_m)),
        "n_minions_red": float(len(red_m)),
        "lane_frac_all": mean_frac(minions),
        "lane_frac_blue": mean_frac(blue_m),
        "lane_frac_red": mean_frac(red_m),
        "turret_hp_blue": _nearest_outer_turret_hp(units, 100),
        "turret_hp_red": _nearest_outer_turret_hp(units, 200),
    }


#: Metrics a response is reported for (everything `compute_metrics` produces
#: except the time axis itself).
RESPONSE_KEYS = (
    "n_minions_blue", "n_minions_red",
    "lane_frac_all", "lane_frac_blue", "lane_frac_red",
    "turret_hp_blue", "turret_hp_red",
)


# ---------------------------------------------------------------------------
# The perturbation set
# ---------------------------------------------------------------------------

class UnitView(NamedTuple):
    """One live enemy minion, as a perturbation script can see it.

    ``uid`` is engine-local identity (a sim slot index or a server NetId) --
    never shared or compared across engines, only used to keep "the same
    target" stable across a script's own decisions on ONE engine.
    """

    uid: int
    x: float
    y: float
    hp: float


class SceneView(NamedTuple):
    """What a perturbation script gets to see each decision, from either
    engine, through one shared interface."""

    t_ms: float
    decision: int
    champ_x: float
    champ_y: float
    champ_alive: bool
    attack_range: float
    #: live RED lane minions only (blue is the scripted side; red never
    #: receives an order, matching every other parity driver in this tree).
    enemy: Tuple[UnitView, ...]


class ActionSpec(NamedTuple):
    kind: str              # "noop" | "move" | "attack"
    x: float = 0.0
    y: float = 0.0
    target: int = -1


#: `TOP_LANE_PATH[2]`: lane fraction ~0.218 (`lane_fraction(*CAMP_POINT)`) --
#: nowhere near the ~0.5 fraction where the two waves clash, and reachable from
#: `CHAMPION_SPAWN[BLUE]` in well under the 90 s before the first wave spawns.
#: This is the shared "nothing is happening" position every perturbation's
#: BASELINE run camps at for the whole episode, and every PERTURBED run camps
#: at until its trigger fires -- so the only scripted difference between a
#: baseline and a perturbed run is the perturbation itself, not where the
#: champion happens to be standing.
CAMP_POINT: Tuple[float, float] = TOP_LANE_PATH[2]

#: `TOP_LANE_PATH[5]` -- the exact point `lanerl_jax.parity.archive.last_hit_drive
#: .APPROACH_WAYPOINTS` stops at, and for the same reason cited there: lane
#: fraction ~0.55 (past the ~0.5 point where the two waves meet, so real
#: minion contact is reliable), 1,248 units from the red outer turret's 750
#: acquisition range -- outside it. `last_hit_drive.py`'s own history is why
#: this matters: an earlier version of that harness parked the champion one
#: waypoint further on (`TOP_LANE_PATH[6]`, the wave-clash vertex itself) and
#: it turned out to sit 412 units inside that same turret's range, so what
#: was meant to measure minion behaviour partly measured turret behaviour
#: instead. Reusing the already-fixed point avoids re-discovering that bug.
ENGAGE_POINT: Tuple[float, float] = TOP_LANE_PATH[5]

#: How close counts as "arrived", generous relative to one decision's travel
#: at Garen's move speed (~11-12 units at 345-370 u/s, 30 Hz).
_ARRIVE_RADIUS = 100.0
_ARRIVE_R2 = _ARRIVE_RADIUS * _ARRIVE_RADIUS


def _arrived(x: float, y: float, target: Tuple[float, float]) -> bool:
    tx, ty = target
    return (x - tx) ** 2 + (y - ty) ** 2 <= _ARRIVE_R2


def _camp_step(scene: SceneView, script: dict) -> ActionSpec:
    """Walk to :data:`CAMP_POINT` once, then hold there forever.

    The shared baseline for every perturbation below, and what a perturbed
    run falls back to before its trigger and (for `StandInWave`) after it
    finishes.
    """
    if script.get("phase", "approach") == "approach":
        if _arrived(scene.champ_x, scene.champ_y, CAMP_POINT):
            script["phase"] = "camp"
        else:
            return ActionSpec("move", CAMP_POINT[0], CAMP_POINT[1])
    return ActionSpec("noop")


class Perturbation:
    """One perturbation protocol, appliable identically to both engines.

    A perturbation is a pure function of a :class:`SceneView` plus its own
    small mutable script state (``new_script()``'s dict) -- it never touches
    an engine's native state directly, which is what makes "identical
    protocol on both engines" possible despite the sim and the server having
    completely different internal representations.
    """

    name: str = "base"

    def new_script(self) -> dict:
        return {"phase": "approach"}

    def decide(self, scene: SceneView, perturbed: bool, script: dict) -> ActionSpec:
        raise NotImplementedError


@dataclass
class NullControl(Perturbation):
    """No perturbation at all. Run twice.

    ``perturbed`` is read but never branched on: the "baseline" and
    "perturbed" runs of this perturbation execute the byte-identical script.
    Any measured response is therefore not a response to anything this module
    did -- it is the floor every real perturbation's response has to clear.
    Exists to answer two questions, both required before trusting any other
    number this module produces: is each engine deterministic run-to-run
    under this harness, and how large is "nothing" at this horizon.
    """

    name: str = "null_control"

    def decide(self, scene: SceneView, perturbed: bool, script: dict) -> ActionSpec:
        return _camp_step(scene, script)


@dataclass
class IdleLane(Perturbation):
    """No champion action at all, ever -- zero orders, full stop.

    Not a perturbation in the response-diff sense: baseline and perturbed
    issue the byte-identical script (like :class:`NullControl`), so
    ``response()`` against it is trivially the null floor and is not the
    point of this class. What it buys is sharing THIS module's own
    instrumentation -- :func:`compute_metrics`, :func:`sim_units` /
    :func:`server_units`, :func:`run_sim_episode` / :func:`run_server_episode`
    -- with the fully-idle scenario
    ``lanerl_jax.sim.tests.test_lane.test_minion_population_is_close_to_the_server``
    and ``docs/TICK_DIVERGENCE_TRACE.md`` measure by their own, separate,
    ad-hoc code paths.

    Why that sharing matters: ``docs/CALL_FOR_HELP_SWITCH_RATE.md`` found
    call-for-help closer to the server on a champion-in-lane response
    (:class:`StandInWave`) and further on a fully idle one -- but those two
    findings came from TWO DIFFERENT INSTRUMENTS (this module's sampled
    response curves vs. the population test's own median-of-counts), so
    "closer on one, worse on the other" was never a like-for-like comparison
    until both scenarios run through the same sampling cadence, the same
    metric set and the same two engines. This class is what makes that
    comparison possible without inventing a third instrument.

    Unlike :class:`NullControl`, this does not even camp at
    :data:`CAMP_POINT` -- the champion is left exactly where it spawns, for
    the whole episode, matching ``bot_teams="none"`` plus "champions never
    ordered" precisely (`lanerl_train`'s own idle-lane recordings, and
    `test_minion_population_is_close_to_the_server`'s docstring, both note
    that camping the champion IN the lane -- as ``NullControl`` does --
    already changes the minion population by soaking up lane-clash melee a
    stationary champion would not otherwise contest).
    """

    name: str = "idle_lane"

    def decide(self, scene: SceneView, perturbed: bool, script: dict) -> ActionSpec:
        return ActionSpec("noop")


@dataclass
class StandInWave(Perturbation):
    """Garen stands in the enemy wave for ``hold_s`` seconds, then leaves.

    Chosen for the pilot over :class:`KillMinions` for three reasons:

    1. **Pure positioning.** Every action here is a ``move`` order (or a
       ``noop`` hold); nothing depends on either engine's attack-damage
       model, windup timing, or how many hits a minion needs to die -- all
       places `docs/TARGET_ACQUISITION_DIFF.md` and `last_hit_drive.py`
       already documented the sim and server disagreeing for reasons
       unrelated to what this test wants to measure.
    2. **Directly targets the one confirmed real gap.**
       `docs/TARGET_ACQUISITION_DIFF.md` found the sim has no call-for-help
       rescue channel, so a minion that locks onto a standing champion never
       gets pulled back onto the wave the way the server's does. Standing in
       the wave is exactly the scenario that provokes that channel on the
       server and cannot on the sim -- if the response test has power to
       detect anything at this horizon, this is the perturbation most likely
       to show it.
    3. **Deterministic given move-order semantics alone**, which
       `lanerl_jax/parity/movement_parity.py` already checks in isolation --
       this perturbation adds no new "does an order do the same thing on both
       sides" surface beyond what that module already covers.

    Both drivers issue the same three-order sequence: move to
    :data:`ENGAGE_POINT`, hold once arrived, move back to
    :data:`CAMP_POINT` once ``hold_s`` seconds have elapsed since arrival.
    """

    name: str = "stand_in_wave"
    #: game time the perturbed run's excursion begins. Comfortably after the
    #: first wave (90 s) and the second (~126 s) have spawned, so live
    #: minions are actually near the lane-clash zone by the time Garen gets
    #: there; comfortably before the 10-minute pilot horizon ends, so there
    #: is a long post-perturbation window to look for a response in.
    trigger_ms: float = 180_000.0
    hold_s: float = 5.0

    def decide(self, scene: SceneView, perturbed: bool, script: dict) -> ActionSpec:
        if not perturbed or scene.t_ms < self.trigger_ms:
            return _camp_step(scene, script)
        phase = script.get("phase", "camp")
        if phase in ("approach", "camp"):
            script["phase"] = "advancing"
            return ActionSpec("move", ENGAGE_POINT[0], ENGAGE_POINT[1])
        if phase == "advancing":
            if _arrived(scene.champ_x, scene.champ_y, ENGAGE_POINT):
                script["phase"] = "holding"
                script["hold_until_ms"] = scene.t_ms + self.hold_s * 1000.0
                return ActionSpec("noop")
            return ActionSpec("move", ENGAGE_POINT[0], ENGAGE_POINT[1])
        if phase == "holding":
            if scene.t_ms >= script.get("hold_until_ms", 0.0):
                script["phase"] = "retreating"
                return ActionSpec("move", CAMP_POINT[0], CAMP_POINT[1])
            return ActionSpec("noop")
        if phase == "retreating":
            if _arrived(scene.champ_x, scene.champ_y, CAMP_POINT):
                script["phase"] = "camp"
                return ActionSpec("noop")
            return ActionSpec("move", CAMP_POINT[0], CAMP_POINT[1])
        return ActionSpec("noop")


@dataclass
class KillMinions(Perturbation):
    """Garen kills up to ``n_targets`` minions in one wave at a fixed time.

    NOT exercised in the 10-minute pilot (see `docs/PERTURBATION_RESPONSE.md`
    for why) -- implemented and left in the tree for the next pass once the
    methodology itself is validated. Included in the reusable set because the
    task calls for 2-3 candidate perturbations; this is the second.

    Unlike :class:`StandInWave`, this one's outcome depends on each engine's
    own attack-damage/windup model, which `last_hit_drive.py`'s "KNOWN
    ASYMMETRY" notes are NOT matched between sim and server (the sim's
    champion AD does not scale with level; the server's does). That is fine
    for the response comparison itself (each engine only ever races against
    its OWN baseline), but it means the two engines may take a different
    number of ticks to actually land ``n_targets`` kills for the same script,
    which is worth knowing before spending compute on this one.

    Both baseline and perturbed runs camp at :data:`ENGAGE_POINT` (not
    :data:`CAMP_POINT`) from the start, so the champion is already in
    contact with the wave well before the trigger -- this isolates "N
    minions removed" from "the champion is standing somewhere new", which
    :class:`StandInWave` deliberately does NOT do (there, presence IS the
    perturbation). At the trigger, the perturbed run snapshots the
    ``n_targets`` nearest live enemy minions and focus-fires them in
    nearest-first order (attacking when in range, closing the distance
    otherwise) until all are dead or ``timeout_ms`` elapses, then resumes
    holding at `ENGAGE_POINT` for the rest of the episode. The baseline never
    attacks.
    """

    name: str = "kill_minions"
    trigger_ms: float = 180_000.0
    n_targets: int = 3
    timeout_ms: float = 30_000.0

    def new_script(self) -> dict:
        return {"phase": "approach", "kill_phase": "select"}

    def _camp_at_engage(self, scene: SceneView, script: dict) -> ActionSpec:
        if script.get("phase", "approach") == "approach":
            if _arrived(scene.champ_x, scene.champ_y, ENGAGE_POINT):
                script["phase"] = "camp"
            else:
                return ActionSpec("move", ENGAGE_POINT[0], ENGAGE_POINT[1])
        return ActionSpec("noop")

    def decide(self, scene: SceneView, perturbed: bool, script: dict) -> ActionSpec:
        if not perturbed or scene.t_ms < self.trigger_ms:
            return self._camp_at_engage(scene, script)

        phase = script.get("kill_phase", "select")
        if phase == "select":
            targets = sorted(
                scene.enemy,
                key=lambda u: (u.x - scene.champ_x) ** 2 + (u.y - scene.champ_y) ** 2,
            )[: self.n_targets]
            script["targets"] = [t.uid for t in targets]
            script["kill_deadline_ms"] = scene.t_ms + self.timeout_ms
            script["kill_phase"] = "killing"
            phase = "killing"

        if phase == "killing":
            if scene.t_ms >= script.get("kill_deadline_ms", 0.0):
                script["kill_phase"] = "done"
                return self._camp_at_engage(scene, script)
            live = {u.uid: u for u in scene.enemy}
            targets = [t for t in script.get("targets", []) if t in live]
            script["targets"] = targets
            if not targets:
                script["kill_phase"] = "done"
                return self._camp_at_engage(scene, script)
            tgt = live[targets[0]]
            dist2 = (tgt.x - scene.champ_x) ** 2 + (tgt.y - scene.champ_y) ** 2
            if dist2 <= scene.attack_range * scene.attack_range:
                return ActionSpec("attack", target=targets[0])
            return ActionSpec("move", tgt.x, tgt.y)

        return self._camp_at_engage(scene, script)


# ---------------------------------------------------------------------------
# Drivers: one decision loop per engine, both calling the same Perturbation
# ---------------------------------------------------------------------------

@dataclass
class ResponseCurve:
    """One (engine, perturbation, baseline-or-perturbed) episode's response
    metrics, sampled on a time grid."""

    engine: str            # "sim" | "server"
    perturbation: str
    perturbed: bool
    seed: int
    samples: List[Dict[str, float]] = field(default_factory=list)
    wall_s: float = 0.0
    decisions: int = 0
    log_path: Optional[str] = None

    def series(self, key: str) -> np.ndarray:
        return np.array([s[key] for s in self.samples], dtype=np.float64)

    @property
    def t_s(self) -> np.ndarray:
        return self.series("t_s")


def run_sim_episode(perturbation: Perturbation, perturbed: bool, seed: int,
                     decisions: int, sample_every: int = 60,
                     enable_call_for_help: bool = False, *,
                     sim_config: Optional[SimConfig] = None) -> ResponseCurve:
    """Drive ``perturbation`` against the JAX sim for ``decisions`` decisions
    (30 Hz, i.e. ``step_ticks=2``, matching every other driver in this tree).

    Only blue (slot 0) is ever ordered; red (slot 1) always gets a NOOP,
    matching `docs/TICK_DIVERGENCE_TRACE.md`'s and `last_hit_drive.py`'s
    setup.

    ``enable_call_for_help`` defaults to ``False``, matching every other sim
    driver in this tree and `lanerl_jax.sim.step.tick`'s own default -- see
    `docs/CALL_FOR_HELP_SWITCH_RATE.md` for why this stays a toggle a caller
    opts into rather than the sim's permanent default.

    ``sim_config`` overrides the step configuration (and then
    ``enable_call_for_help`` is ignored); default
    ``SimConfig.scripted(enable_call_for_help=...)``.
    """
    import jax
    import jax.numpy as jnp

    # `STRUCT-003`: one step configuration. The default is exactly what this
    # driver always ran -- TOP lane waves, inline terrain repair, UNROUTED
    # Moves (`PATH-006`: pass `SimConfig.scripted(route_artifact=...)` to
    # route them; that changes results, so it is not the default here).
    sim = (sim_config if sim_config is not None else
           SimConfig.scripted(enable_call_for_help=enable_call_for_help))
    params_tbl = sim.params
    params_np = {k: np.asarray(v) for k, v in params_tbl.items()}
    state = init_lane(seed=seed)

    @jax.jit
    def _step(state, kind, x, y, target):
        orders = Orders(
            kind=jnp.array([kind, OrderKind.NOOP], dtype=jnp.int8),
            x=jnp.array([x, 0.0], dtype=state.x.dtype),
            y=jnp.array([y, 0.0], dtype=state.y.dtype),
            target=jnp.array([target, -1], dtype=jnp.int8),
        )
        return env_step(state, orders, sim)

    script = perturbation.new_script()
    samples: List[Dict[str, float]] = []
    t0 = time.perf_counter()
    for i in range(decisions):
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
            samples.append(compute_metrics(sim_units(state), float(state.t_ms) / 1000.0))
    wall_s = time.perf_counter() - t0
    return ResponseCurve(engine="sim", perturbation=perturbation.name, perturbed=perturbed,
                          seed=seed, samples=samples, wall_s=wall_s, decisions=decisions)


def run_server_episode(perturbation: Perturbation, perturbed: bool, seed: int,
                        decisions: int, sample_every: int = 60,
                        bot_seed: int = 4242, port_base: Optional[int] = None,
                        log_dir: Optional[Path] = None) -> ResponseCurve:
    """Drive ``perturbation`` against one real server instance.

    Booted exactly like `last_hit_drive.run_oracle_on_server`:
    ``toponly=True`` (no other lanes/jungle), ``bot_teams="none"`` (the
    in-server scripted bot never touches either champion), ``step_ticks=2``
    (30 Hz). ``seed`` names this episode for bookkeeping only -- the server
    side has no equivalent of the sim's ``init_lane(seed=...)`` RNG key;
    determinism here comes from ``bot_seed`` and the fixed config, not from
    this argument. HARD CONSTRAINT: exactly one server instance (``n=1``),
    never more, and the caller must not launch a second one concurrently --
    this machine has 6 cores and a prior run at n=5 concurrent instances
    drove load average to 19.75 and made every run slower than running them
    serially.
    """
    from lanerl_train.ports import PortAllocator
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

    log_dir = Path(log_dir) if log_dir is not None else Path(
        tempfile.mkdtemp(prefix="perturbation_"))
    ports = (PortAllocator(base=port_base) if port_base else PortAllocator()).allocate(1)
    env = VecLaneEnv(
        1,
        spec=ServerLaunchSpec(toponly=True, bot_teams="none", bot_seed=bot_seed, step_ticks=2),
        log_dir=log_dir, ports=ports, step_timeout_s=180.0, auto_restart=False,
    )
    env.start()
    script = perturbation.new_script()
    samples: List[Dict[str, float]] = []
    log_path: Optional[str] = None
    t0 = time.perf_counter()
    try:
        if not all(env.alive):
            raise RuntimeError(f"server failed to boot: {env.alive}")
        log_path = str(env.handles[0].log_path)
        for i in range(decisions):
            obs = env.last_obs[0]
            if obs is None:
                raise RuntimeError(f"no observation at decision {i}")
            units = obs.get("u", [])
            blue = next(u for u in units if u.get("k") == "Champion" and u.get("tm") == 100)
            enemy = tuple(
                UnitView(uid=int(u["id"]), x=float(u["x"]), y=float(u["y"]), hp=float(u["hp"]))
                for u in units if u.get("k") == "LaneMinion" and u.get("tm") == 200)
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
                samples.append(compute_metrics(server_units(obs2), float(obs2.get("t", 0)) / 1000.0))
    finally:
        env.close()
    wall_s = time.perf_counter() - t0
    return ResponseCurve(engine="server", perturbation=perturbation.name, perturbed=perturbed,
                          seed=seed, samples=samples, wall_s=wall_s, decisions=decisions,
                          log_path=log_path)


# ---------------------------------------------------------------------------
# Response computation and (de)serialisation
# ---------------------------------------------------------------------------

def response(baseline: ResponseCurve, perturbed: ResponseCurve,
             keys: Sequence[str] = RESPONSE_KEYS) -> Dict[str, np.ndarray]:
    """``R = f(perturbed) - f(baseline)``, same engine, aligned by sample
    index (both curves are driven decision-for-decision from the same seed
    at the same ``sample_every``, so their ``t_s`` grids match)."""
    if baseline.engine != perturbed.engine:
        raise ValueError(f"response() compares one engine to itself, got "
                          f"{baseline.engine!r} vs {perturbed.engine!r}")
    n = min(len(baseline.samples), len(perturbed.samples))
    if n == 0:
        raise ValueError("no overlapping samples to compare")
    out: Dict[str, np.ndarray] = {"t_s": baseline.series("t_s")[:n]}
    for k in keys:
        out[k] = perturbed.series(k)[:n] - baseline.series(k)[:n]
    return out


def summarize_response(r: Mapping[str, np.ndarray],
                        keys: Sequence[str] = RESPONSE_KEYS) -> Dict[str, Dict[str, float]]:
    """Per-metric ``{max_abs, rms}`` of a response dict from :func:`response`.

    NaNs (e.g. ``lane_frac_*`` when a team has zero live minions at that
    sample) are excluded rather than propagated, so one empty-wave sample
    cannot blank out an entire metric's summary.
    """
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        v = np.asarray(r[k], dtype=np.float64)
        v = v[~np.isnan(v)]
        if v.size == 0:
            out[k] = {"max_abs": float("nan"), "rms": float("nan"), "n": 0}
            continue
        out[k] = {"max_abs": float(np.max(np.abs(v))), "rms": float(np.sqrt(np.mean(v * v))),
                   "n": int(v.size)}
    return out


def curve_to_dict(c: ResponseCurve) -> dict:
    return {
        "engine": c.engine, "perturbation": c.perturbation, "perturbed": c.perturbed,
        "seed": c.seed, "wall_s": c.wall_s, "decisions": c.decisions,
        "log_path": c.log_path, "samples": c.samples,
    }


def curve_from_dict(d: Mapping) -> ResponseCurve:
    return ResponseCurve(
        engine=d["engine"], perturbation=d["perturbation"], perturbed=bool(d["perturbed"]),
        seed=int(d["seed"]), samples=list(d["samples"]), wall_s=float(d["wall_s"]),
        decisions=int(d.get("decisions", 0)), log_path=d.get("log_path"))


# ---------------------------------------------------------------------------
# CLI: run exactly one (engine, perturbation, baseline-or-perturbed) episode
# and write its ResponseCurve to a JSON file. The pilot driver calls this
# once per episode rather than importing the module, so a server crash in
# one episode cannot take a whole batch of episodes down with it, and so
# server episodes are trivially run one at a time (the hard hardware
# constraint this project is under -- see run_server_episode's docstring).
# ---------------------------------------------------------------------------

def _run_cli(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run one perturbation-response episode.")
    parser.add_argument("--engine", choices=["sim", "server"], required=True)
    parser.add_argument("--perturbation",
                        choices=["null_control", "idle_lane", "stand_in_wave", "kill_minions"],
                        required=True)
    parser.add_argument("--perturbed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--sample-every-s", type=float, default=2.0)
    parser.add_argument("--bot-seed", type=int, default=4242)
    parser.add_argument("--port-base", type=int, default=None)
    parser.add_argument("--call-for-help", action="store_true",
                        help="sim only: enable_call_for_help=True (default off, "
                             "matching lanerl_jax.sim.step.tick's own default)")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args(argv)

    perturbation: Perturbation = {
        "null_control": NullControl(),
        "idle_lane": IdleLane(),
        "stand_in_wave": StandInWave(),
        "kill_minions": KillMinions(),
    }[args.perturbation]

    decisions = int(round(args.minutes * 60.0 * 30.0))     # 30 Hz decision rate
    sample_every = max(1, int(round(args.sample_every_s * 30.0)))

    if args.engine == "sim":
        curve = run_sim_episode(perturbation, args.perturbed, args.seed, decisions, sample_every,
                                enable_call_for_help=args.call_for_help)
    else:
        curve = run_server_episode(perturbation, args.perturbed, args.seed, decisions,
                                    sample_every, bot_seed=args.bot_seed,
                                    port_base=args.port_base)

    Path(args.out).write_text(json.dumps(curve_to_dict(curve)))
    print(f"{args.engine} {args.perturbation} perturbed={args.perturbed}: "
          f"{len(curve.samples)} samples over {decisions} decisions, "
          f"wall={curve.wall_s:.1f}s -> {args.out}")


if __name__ == "__main__":
    _run_cli()
