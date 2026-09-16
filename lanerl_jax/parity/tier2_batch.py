"""Run several Tier-2 scenarios back to back, one server instance at a time.

`tier2.py`'s CLI runs exactly one (engine, scenario) episode per process --
deliberately, matching `perturbation.py`'s own pilot driver, so a crash in one
scenario cannot take the rest of the batch down with it and so server
instances are trivially serialised (never more than one process alive at
once). This module is the loop around that CLI: it shells out once per
scenario via `subprocess`, in-process for `--engine sim` (cheap, no shared
resource) or as a fresh subprocess for `--engine server` (so a server
crash mid-scenario is isolated and does not corrupt the JAX process for the
next one).

Meant to be the single command handed to `sbatch slurm/parity_g2.sbatch`, so
one job runs the whole scenario list under slurm's serialisation rather than
one job per scenario (cheaper for both this job and the sibling agents also
queued on `desktop`).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["sim", "server"], required=True)
    ap.add_argument("--scenarios", default="idle,stand_early,stand_late,kill")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--sample-every-s", type=float, default=2.0)
    ap.add_argument("--out-dir", default="lanerl_jax/runs/tier2")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = args.scenarios.split(",")
    failures = []
    for name in scenarios:
        out = out_dir / f"{args.engine}_{name}.json"
        cmd = [sys.executable, "-m", "lanerl_jax.parity.tier2", "raw",
               "--engine", args.engine, "--scenario", name,
               "--seed", str(args.seed), "--minutes", str(args.minutes),
               "--sample-every-s", str(args.sample_every_s), "--out", str(out)]
        print(f"=== {args.engine} {name}: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!!! {args.engine} {name} FAILED (exit {r.returncode})", flush=True)
            failures.append(name)
        else:
            print(f"=== {args.engine} {name}: done -> {out}", flush=True)
    if failures:
        print(f"FAILURES: {failures}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
