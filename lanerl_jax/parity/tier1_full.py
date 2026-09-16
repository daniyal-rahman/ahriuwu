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
    ap.add_argument("--existing-log", default=None,
                    help="reuse an already-recorded server log instead of "
                        "booting a fresh server (for iterating on the "
                        "differ/report without paying to re-record)")
    a = ap.parse_args(argv)

    if a.existing_log:
        log = Path(a.existing_log)
        print(f"reusing existing log: {log}", flush=True)
    else:
        out = Path(a.out)
        log = record_idle_trace(out, game_seconds=a.game_seconds,
                                port_base=a.port_base)
        print(f"recorded: {log}", flush=True)

    trace = load_trace(log)
    after = [s for s in trace.snapshots if s.t_ms >= FIRST_WAVE_MS]
    print(f"{len(trace)} snapshots, {len(after)} at/after the first wave",
          flush=True)

    res = run_one_step_differential(after, max_pairs=a.max_pairs)
    print(res.report())


if __name__ == "__main__":
    main()
