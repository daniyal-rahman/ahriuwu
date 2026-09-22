"""Tier 1 over the minion-bearing part of a fresh idle-lane trace.

Committed version of the script that produced ``docs/TIER1_POST_REORDER.md``
(that run was driven from an uncommitted scratch script -- see this file's
git history / the commit that added it for the note). Queue it on `desktop`
via slurm.  It is CPU-bound work this repo has a node for, and an
unchunked full-trace run holds the entire 575 MB dump as parsed `Entity`
objects (~26 GB), which is why it was OOM-killed on the login node before:

    sbatch slurm/parity.sbatch python -m lanerl_jax.parity.tier1_full

``--chunk-pairs`` removes that constraint by parsing one window at a time, so
the same corpus, pair for pair, fits in a few GB and can run anywhere -- which
matters whenever `desktop` is down.  Use it with a hard cap that has swap
disabled (`ops/login_capped.sh`) rather than trusting the estimate.

``python -m`` is required (not a bare path) so the ``lanerl_jax`` package
resolves the same way it does under pytest.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .one_step import (
    GAME_SECONDS,
    merge_one_step_results,
    record_idle_trace,
    run_one_step_differential,
)
from .parallel_one_step import (
    ParallelProgress,
    run_parallel_one_step_differential,
)
from .trace import load_trace, load_trace_window

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
    ap.add_argument(
        "--instrumented", action="store_true",
        help="Record against the observability build, so the corpus carries "
             "aacdbits/aagate/status/aibuffs and the branch stream. Every "
             "emission is behaviour-neutral, so this reproduces the stock "
             "corpus byte for byte and merely carries more -- which is what "
             "lets the canonical numbers be ATTRIBUTED instead of counted.")
    ap.add_argument("--action-log", default=None,
                    help="ActionLog JSON recorded beside a driven fixture; "
                         "orders are replayed at their endpoint dump boundary")
    ap.add_argument("--chunk-pairs", type=int, default=None,
                    help="stream the corpus this many tick-pairs at a time, "
                        "parsing only the snapshots each chunk needs. The "
                        "shards are the SAME disjoint pair ranges the "
                        "--workers path already merges, so the report is "
                        "identical -- this bounds memory, not work.")
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
                                port_base=a.port_base,
                                instrumented=a.instrumented)
        print(f"recorded: {log}", flush=True)

    action_log = None
    route_table = terrain = None
    if not a.action_log and a.existing_log:
        # Same refusal as the residual drill (`METH-006`). A DRIVEN fixture read
        # without its action stream does not fail: it steps the simulator with
        # no orders while the server's champion follows the script, and scores
        # the difference as a port defect. On the drill that produced 186
        # phantom champion target disagreements and 142 phantom move-order rows
        # -- numbers with exactly the shape of a large real residual.
        from .record import ActionLog as _AL
        _lp = Path(a.existing_log)
        for _g in sorted(_lp.parent.parent.glob("*_actions.json")):
            try:
                _c = _AL.load(_g)
            except Exception:
                continue
            _n = sum(1 for w in list(_c.blue) + list(_c.red)
                     if str(w.get("t", "noop")) != "noop")
            if _n:
                raise SystemExit(
                    f"REFUSING TO RUN: {_g} carries {_n} non-noop orders, so "
                    "this is a DRIVEN fixture, but no --action-log was given. "
                    "Every champion row would measure the missing orders "
                    f"rather than the port.\n  pass:  --action-log {_g}")
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

    # Idle fixtures contain no lane units before the first wave, so those
    # pairs only dilute minion metrics.  Driven fixtures are different: their
    # pre-wave champion actions are exactly what the action replay validates.
    def _select(snapshots):
        return (snapshots if action_log is not None else
                [s for s in snapshots if s.t_ms >= FIRST_WAVE_MS])

    selection = "full driven trace" if action_log is not None else "at/after first wave"

    if a.chunk_pairs:
        if a.workers != 1:
            ap.error("--chunk-pairs streams shards serially; it is an "
                     "alternative to --workers, not a companion")
        # One cheap pass for the tick times alone: `max_snapshots=0` makes
        # every snapshot a placeholder, so this reads the file without
        # building a single Entity.  The pair indices computed here are the
        # same indices the unchunked path uses, because `load_trace_window`
        # preserves snapshot count and order exactly.
        times = [s.t_ms for s in _select(
            load_trace_window(log, max_snapshots=0).snapshots)]
        n_pairs = max(0, len(times) - 1)
        if a.max_pairs is not None:
            n_pairs = min(n_pairs, a.max_pairs)
        print(f"{len(times)} snapshots {selection}; streaming {n_pairs} pairs "
              f"in chunks of {a.chunk_pairs}", flush=True)
        parts = []
        for start in range(0, n_pairs, a.chunk_pairs):
            stop = min(start + a.chunk_pairs, n_pairs)
            # `times[stop]` is snapshot `stop`, which pair `stop - 1` needs as
            # its endpoint -- so the window is inclusive of it.
            chunk = _select(load_trace_window(
                log, from_ms=times[start], to_ms=times[stop]).snapshots)
            parts.append(run_one_step_differential(
                chunk, pair_start=start, pair_stop=stop, action_log=action_log,
                route_table=route_table, terrain=terrain))
            print(f"progress: {stop}/{n_pairs} pairs", flush=True)
        res = merge_one_step_results(parts)
        print(res.report())
        return

    trace = load_trace(log)
    selected = _select(trace.snapshots)
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
