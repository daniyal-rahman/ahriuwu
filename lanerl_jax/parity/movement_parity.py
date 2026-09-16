"""Tier-1 parity for movement: pathfinding + waypoint following, vs the server.

The experiment
--------------
Walk the champion out to the top lane, then repeatedly: read its exact position
from the state dump, issue one ``move`` order to a random nearby point, hold for
a fixed number of ticks, and compare the server's per-tick trajectory against
:func:`lanerl_jax.sim.movement.follow` run over
:meth:`lanerl_jax.data.navgrid.NavGrid.get_path`.

This exercises, in one number, everything underneath a champion's position:
the navgrid parse, ``TranslateToNavGrid``, ``IsWalkable``, ``CastCircle``,
``GetClosestTerrainExit``, the A* (including .NET's heap order),
``SmoothPath``, and ``AttackableUnit.Move``.

Two traps this harness exists to avoid
--------------------------------------
**Take the start position from the state dump, never from the observation.**
``LanerlControl`` writes positions as ``((int)au.Position.X)`` -- truncated to
whole units -- while ``LanerlStateDump`` quantises to 1/16.  Seeding the
reconstruction from the observation injects up to a unit of error before the
first tick and makes a correct port look wrong by ~1 unit.  Measured: median
worst-tick error fell from 0.9 to 0.07 units on the same runs when the start
came from the dump instead.

**Hold for less than the re-path interval, and check.** ``PathingHandler.Update``
re-paths every 3000 ms, which would replace the waypoints mid-comparison.  The
default hold is under that.  (Tested and *falsified* as the cause of the
divergences that led to the ``GetClosestTerrainExit`` fix: the divergence ticks
were scattered across the 3000 ms cycle, and the server's waypoint count never
changed at the divergence point.  Recorded here so nobody re-runs that check.)

Reading the result
------------------
``waypoint_count_agrees`` isolates *pathing* from *following*: it compares the
number of waypoints the server holds (``Waypoints.Count``, in the dump) against
the path this port computed.  When counts agree and the trajectory still
diverges, the bug is in the follower or in which cells the path went through;
when counts differ, it is the pathfinder.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..data.navgrid import GAREN_PATHFINDING_RADIUS, NavGrid
from ..sim.movement import follow
from .recover_waypoints import compare_paths, recover_waypoints
from .trace import load_trace

__all__ = ["MoveTrial", "MovementParityResult", "run_movement_parity"]

#: A vertex of ``LanerlLane.TopLaneDefault``, in open lane.
LANE_POINT = (1268.0, 11675.0)


@dataclass(slots=True)
class MoveTrial:
    order_t_ms: int
    start: Tuple[float, float]
    goal: Tuple[float, float]
    goal_walkable: bool
    server_waypoints: int
    my_waypoints: int
    path_was_none: bool
    max_error: float
    mean_error: float
    ticks: int
    #: max distance from the server's recovered waypoints to the computed path.
    #: Near zero means the server walked this route whatever the counts say.
    route_deviation: float = 0.0
    recovered_vertices: int = 0
    #: per-tick position error. A CONSTANT offset is a tick-alignment lag (the
    #: order lands a tick or two after the observation was built); a GROWING one
    #: is a real divergence. Distinguishing the two without this curve is
    #: guesswork, and a systematic lag would bias every mechanic, not just this.
    error_curve: Optional[np.ndarray] = None
    #: best whole-tick shift of the reconstruction, and the error it leaves.
    best_shift: int = 0
    best_shift_error: float = 0.0

    @property
    def waypoint_count_agrees(self) -> bool:
        return self.server_waypoints == self.my_waypoints


@dataclass(slots=True)
class MovementParityResult:
    trials: List[MoveTrial]

    @property
    def max_errors(self) -> np.ndarray:
        return np.array([t.max_error for t in self.trials])

    def report(self) -> str:
        e = self.max_errors
        agree = sum(t.waypoint_count_agrees for t in self.trials)
        none_n = sum(t.path_was_none for t in self.trials)
        clean = [t.max_error for t in self.trials if not t.path_was_none]
        lines = [
            f"{len(self.trials)} move orders",
            f"  worst-tick error : max {e.max():.3f}  median {np.median(e):.3f} units",
            f"  within 1/16 (the dump's own quantisation): "
            f"{int((e <= 0.0625).sum())}/{len(e)}",
            f"  within 1 unit    : {int((e <= 1.0).sum())}/{len(e)}",
            f"  waypoint count agrees: {agree}/{len(self.trials)}",
            f"  A* returned no path  : {none_n}/{len(self.trials)}",
        ]
        if clean:
            c = np.array(clean)
            lines.append(f"  excluding no-path fallbacks: max {c.max():.3f} "
                         f"median {np.median(c):.3f}")
        rd = np.array([t.route_deviation for t in self.trials])
        lines.append(f"  route deviation (server's walked route vs computed): "
                     f"max {rd.max():.3f} median {np.median(rd):.3f}")
        lines.append(f"  routes matching within 1 unit: "
                     f"{int((rd <= 1.0).sum())}/{len(rd)}")
        return "\n".join(lines)


def run_movement_parity(
    out_dir: Path,
    trials: int = 20,
    hold_decisions: int = 85,
    port_base: int = 47000,
    seed: int = 11,
    min_dist: float = 400.0,
    max_dist: float = 1200.0,
    require_reachable_goal: bool = True,
    tag: str = "movement_parity",
) -> MovementParityResult:
    from lanerl_train.ports import PortAllocator
    from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    _grid = NavGrid.load()
    env = VecLaneEnv(
        1,
        spec=ServerLaunchSpec(
            toponly=True, bot_teams="none", bot_seed=4242, step_ticks=2,
            extra_env={"LANERL_STATE_DUMP": "1", "LANERL_STATE_DUMP_FULL": "1"},
        ),
        log_dir=out_dir / tag,
        ports=PortAllocator(base=port_base).allocate(1),
        step_timeout_s=180.0,
        auto_restart=False,
    )
    noop = {"blue": {"t": "noop"}, "red": {"t": "noop"}}
    env.start()
    issued: List[Tuple[int, Tuple[float, float]]] = []
    try:
        if not all(env.alive):
            raise RuntimeError(f"server failed to boot: {env.alive}")
        for _ in range(30):
            env.step([noop])
        # walk out of the fountain: base terrain is tight and unrepresentative
        for _ in range(14):
            env.step([{"blue": {"t": "move", "x": LANE_POINT[0], "y": LANE_POINT[1]},
                       "red": {"t": "noop"}}])
            for _ in range(60):
                env.step([noop])
        for _ in range(trials):
            obs = env.last_obs[0]
            ch = [u for u in obs["u"] if u["k"] == "Champion" and u["tm"] == 100][0]
            # Reject goals off the navgrid. Not cosmetic: `IsWalkable(p, r)` with
            # r > 0 iterates `GetAllCellsInRange`, which yields NOTHING for an
            # off-grid point, and the server's loop then falls through to
            # `return true` -- so an off-map point reports *walkable* at radius 35
            # and unwalkable at radius 0. `GetClosestTerrainExit` therefore returns
            # it unchanged, `GetPath` gets a null `cellTo` and returns null, and
            # `LanerlControl` issues a raw straight line at the edge of the world.
            # This port reproduces that chain exactly (it is the server's quirk,
            # not ours) -- but the champion then walks into terrain, and terrain
            # collision during UN-pathed movement is not modelled here, so those
            # orders measure a different thing. Excluded by default and counted.
            for _ in range(64):
                ang = rng.uniform(0, 2 * math.pi)
                r = rng.uniform(min_dist, max_dist)
                goal = (float(ch["x"]) + r * math.cos(ang),
                        float(ch["y"]) + r * math.sin(ang))
                if not require_reachable_goal:
                    break
                gx, gy = _grid.to_nav(*goal)
                if 0 <= gx < _grid.cell_count_x and 0 <= gy < _grid.cell_count_y:
                    break
            t0 = int(obs["t"])
            env.step([{"blue": {"t": "move", "x": goal[0], "y": goal[1]},
                       "red": {"t": "noop"}}])
            for _ in range(hold_decisions):
                env.step([noop])
            issued.append((t0, goal))
        log = Path(env.handles[0].log_path)
    finally:
        env.close()

    trace = load_trace(log)
    by_t = {}
    for s in trace:
        c = s.champion(100)
        if c is not None:
            by_t[s.t_ms] = (c.x, c.y, c.champ.q_move_speed / 1024.0, c.ai.waypoints)
    times = sorted(by_t)
    grid = NavGrid.load()

    out: List[MoveTrial] = []
    for i, (t0, goal) in enumerate(issued):
        t1 = issued[i + 1][0] if i + 1 < len(issued) else times[-1] + 1
        seg = [t for t in times if t0 <= t < t1]
        if len(seg) < 10:
            continue
        sx, sy, ms, _ = by_t[seg[0]]
        start = (sx, sy)
        raw = grid.get_path(start, goal, radius=GAREN_PATHFINDING_RADIUS)
        was_none = raw is None or len(raw) < 2
        # LanerlControl: null/short path -> a raw two-point order; else path[0]=us
        path = [start, goal] if was_none else [start] + raw[1:]
        mine = follow(start, path, ms, ticks=len(seg) - 1)
        err = np.array([math.dist(by_t[t][:2], tuple(mine[k]))
                        for k, t in enumerate(seg)])
        # How much of the error is a pure whole-tick lag? Re-run the follower
        # from a start advanced by k ticks and see which k fits best.
        best_shift, best_err = 0, float(err.max())
        for k in range(1, 13):
            if len(seg) - 1 - k < 10:
                break
            shifted = follow(start, path, ms, ticks=len(seg) - 1)
            e = np.array([math.dist(by_t[t][:2], tuple(shifted[max(0, i - k)]))
                          for i, t in enumerate(seg)])
            if e.max() < best_err:
                best_shift, best_err = k, float(e.max())
        rec = recover_waypoints([by_t[t][:2] for t in seg])
        cmp = compare_paths(path, rec)
        out.append(MoveTrial(
            error_curve=err, best_shift=best_shift, best_shift_error=best_err,
            route_deviation=cmp["route_deviation"],
            recovered_vertices=cmp["recovered_vertices"],
            order_t_ms=t0, start=start, goal=goal,
            goal_walkable=grid.is_walkable_world(*goal, GAREN_PATHFINDING_RADIUS),
            server_waypoints=max(by_t[t][3] for t in seg[:6]),
            my_waypoints=len(path), path_was_none=was_none,
            max_error=float(err.max()), mean_error=float(err.mean()), ticks=len(seg),
        ))
    return MovementParityResult(trials=out)
