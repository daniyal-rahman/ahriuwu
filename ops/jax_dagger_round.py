#!/usr/bin/env python3
"""One DAgger round on the JAX sim, three srun steps on the desktop:

  1. roll the current clone out (sampled, mirror) and record its observations,
  2. relabel every recorded observation with the scripted last-hitter and merge
     with the previous dataset (train/bc_dagger.py),
  3. retrain the GRU clone on the merged set (train/bc_diag.py --core gru).

    ops/jax_dagger_round.py <round> <clone_ckpt> <prev_dataset.npz> [--envs 16] [--epochs 30]

Paths are validated before anything is submitted (four launches were lost to
paths composed in the shell). Prints the new clone's checkpoint path last.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

REPO_SRV = Path("/srv/nfs/projects/ahriuwu-lanerl-jax")
REPO_MNT = "/mnt/nfs/projects/ahriuwu-lanerl-jax"
PY = "env XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 ./.venv-gpu/bin/python"


def srun(name, cmd, cpus=4, mem="12G", minutes=60):
    full = ["srun", "-p", "cpu", "-w", "desktop", f"--cpus-per-task={cpus}", f"--mem={mem}", f"--time={minutes}",
            f"--chdir={REPO_MNT}", f"--job-name={name}", "bash", "-c", f"{PY} {cmd}"]
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {cmd[:200]}", flush=True)
    r = subprocess.run(full, capture_output=True, text=True, cwd=REPO_SRV)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if not any(x in l for x in ("absl", "cudart", "srun:", '"kernel"'))]
    print("\n".join(lines[-12:]), flush=True)
    if r.returncode != 0:
        sys.exit(f"{name} FAILED rc {r.returncode}")
    return r.stdout


def mnt(p):
    """Desktop-side spelling of a login-side path (/srv/nfs -> /mnt/nfs)."""
    return str(p).replace("/srv/nfs/", "/mnt/nfs/")


def must(p, what):
    p = Path(p)
    if not p.exists():
        sys.exit(f"REFUSED: {what} missing: {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round", type=int); ap.add_argument("clone"); ap.add_argument("prev")
    ap.add_argument("--envs", type=int, default=16); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    clone = must(a.clone, "clone checkpoint"); must(clone.parent / "manifest.json", "clone manifest")
    prev = must(a.prev, "previous dataset")
    seed = a.round + 10 if a.seed is None else a.seed
    out_roll = REPO_SRV / f"lanerl_jax/runs/JAX_ORACLE/dagger{a.round}_rollout"
    srun(f"dagger{a.round}-rollout",
         f"-m lanerl_jax.train.jax_train --envs {a.envs} --opponent mirror --episode-s 600 --start-near-wave "
         f"--step-ticks 6 --eval-episodes 1 --record-npz rollout.npz --seed {seed} --resume {mnt(clone.resolve())} --out {mnt(out_roll)}",
         mem="12G")
    roll = sorted(out_roll.glob("*/rollout.npz"), key=lambda p: p.stat().st_mtime)[-1]
    merged = REPO_SRV / f"lanerl_jax/runs/JAX_ORACLE/jax_dagger{a.round}.npz"
    srun(f"dagger{a.round}-relabel", f"-m lanerl_jax.train.bc_dagger {mnt(merged)} {mnt(prev.resolve())} {mnt(roll)}", cpus=2, mem="16G")
    must(merged, "merged dataset")
    before = {p for p in (REPO_SRV / "lanerl_jax/runs/BC").glob("bc-gru*")}
    srun(f"dagger{a.round}-bc", f"-m lanerl_jax.train.bc_diag {mnt(merged)} --out lanerl_jax/runs/BC --core gru --core-norm --core-residual --epochs {a.epochs}",
         mem="16G", minutes=120)
    new = sorted({p for p in (REPO_SRV / "lanerl_jax/runs/BC").glob("bc-gru*")} - before)
    if not new:
        sys.exit("no new clone directory")
    ck = new[-1] / "ckpt_latest.msgpack"; must(ck, "new clone checkpoint")
    (REPO_SRV / f"lanerl_jax/runs/JAX_ORACLE/dagger{a.round}.json").write_text(json.dumps(
        {"round": a.round, "from_clone": str(clone), "rollout": str(roll), "dataset": str(merged), "clone": str(ck)}, indent=1))
    print("NEW_CLONE", ck)


if __name__ == "__main__":
    main()
