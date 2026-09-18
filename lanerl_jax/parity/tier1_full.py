"""Tier 1 over the minion-bearing part of a fresh idle-lane trace.

Committed version of the script that produced ``docs/TIER1_POST_REORDER.md``
(that run was driven from an uncommitted scratch script -- see this file's
git history / the commit that added it for the note). Queue it on `desktop`
via slurm, never on the login node (a full-trace one-step differential was
OOM-killed there before, and it is CPU-bound work this repo already has a
node for):

    sbatch slurm/parity_g1.sbatch python -m lanerl_jax.parity.tier1_full

``python -m`` is required (not a bare path) so the ``lanerl_jax`` package
resolves the same way it does under pytest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .one_step import (
    GAME_SECONDS,
    record_idle_trace,
    run_one_step_differential,
)
from .parallel_one_step import (
    ParallelProgress,
    run_parallel_one_step_differential,
)
from .trace import load_trace

#: The recorded fixture is idle from t=0, but nothing with a minion in it
#: exists before the first wave (`sim.waves.FIRST_WAVE_MS`) -- comparing
#: those ticks would silently pad every LaneMinion field's denominator with
#: zero-information pairs.
FIRST_WAVE_MS = 90_000


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="lanerl_jax/runs/tier1_full",
                    help="directory for the recorded server log")
    ap.add_argument("--game-seconds", type=float, default=420.0,
                    help="matches the 420s idle-lane recording "
                        "docs/TIER1_POST_REORDER.md reports numbers for")
    ap.add_argument("--port-base", type=int, default=51100)
    ap.add_argument("--max-pairs", type=int, default=None,
                    help="cap the number of tick-pairs compared, for a quick "
                        "smoke run rather than the full corpus")
    ap.add_argument("--workers", type=int, default=1,
                    help="independent tick-pair shards sharing one JAX runtime; "
                         "benchmark before increasing because a saturated "
                         "accelerator may be faster with one worker")
    ap.add_argument("--progress-every", type=int, default=250,
                    help="report aggregate pair throughput this often")
    ap.add_argument("--existing-log", default=None,
                    help="reuse an already-recorded server log instead of "
                        "booting a fresh server (for iterating on the "
                        "differ/report without paying to re-record)")
    ap.add_argument("--action-log", default=None,
                    help="ActionLog JSON recorded beside a driven fixture; "
                         "orders are replayed at their endpoint dump boundary")
    ap.add_argument("--route-artifact", default=None,
                    help="local Map1 route artifact for driven Move replay; "
                         "defaults to the production training artifact when "
                         "--action-log is present")
    a = ap.parse_args(argv)

    if a.existing_log:
        log = Path(a.existing_log)
        print(f"reusing existing log: {log}", flush=True)
    else:
        out = Path(a.out)
        log = record_idle_trace(out, game_seconds=a.game_seconds,
                                port_base=a.port_base)
        print(f"recorded: {log}", flush=True)

    action_log = None
    route_table = terrain = None
    if a.action_log:
        from .record import ActionLog
        from ..data.local_route_artifact import load_local_route_artifact
        from ..sim.terrain_jax import map1_terrain
        from ..train.run_train import DEFAULT_ROUTE_ARTIFACT

        action_log = ActionLog.load(Path(a.action_log))
        print(f"loaded recorded actions: {a.action_log} "
              f"({len(action_log.t_ms)} decisions)", flush=True)
        route_path = (Path(a.route_artifact) if a.route_artifact
                      else DEFAULT_ROUTE_ARTIFACT)
        artifact = load_local_route_artifact(
            route_path, pathfinding_radius=35.0)
        route_table = artifact.as_jax()
        terrain = map1_terrain()
        print(f"loaded production local routes: {route_path}", flush=True)

    trace = load_trace(log)
    # Idle fixtures contain no lane units before the first wave, so those
    # pairs only dilute minion metrics.  Driven fixtures are different: their
    # pre-wave champion actions are exactly what the action replay validates.
    selected = (trace.snapshots if action_log is not None else
                [s for s in trace.snapshots if s.t_ms >= FIRST_WAVE_MS])
    selection = "full driven trace" if action_log is not None else "at/after first wave"
    print(f"{len(trace)} snapshots, {len(selected)} {selection}", flush=True)

    if a.workers == 1:
        res = run_one_step_differential(
            selected, max_pairs=a.max_pairs, action_log=action_log,
            route_table=route_table, terrain=terrain)
    else:
        def progress(p: ParallelProgress) -> None:
            print(f"progress: {p.attempted_pairs}/{p.total_pairs} pairs "
                  f"({p.pairs_per_s:.2f} pairs/s, {p.elapsed_s:.1f}s)",
                  flush=True)

        res, stats = run_parallel_one_step_differential(
            selected, max_pairs=a.max_pairs, workers=a.workers,
            progress_every=a.progress_every, progress_callback=progress,
            action_log=action_log, route_table=route_table, terrain=terrain)
        print(f"parallel pass: {stats.compared_pairs} compared, "
              f"{stats.skipped_pairs} skipped in {stats.elapsed_s:.2f}s "
              f"({stats.compared_pairs_per_s:.2f} compared pairs/s)",
              flush=True)
    print(res.report())


if __name__ == "__main__":
    main()
