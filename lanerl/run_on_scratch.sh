#!/usr/bin/env bash
# Launch a training run with its WRITES on desktop's local NVMe, and mirror the
# durable parts back to NFS.
#
# Why: the repo lives on NFS (192.168.1.104:/srv/nfs) and everything a run
# writes went there too -- metrics.jsonl appended every update, a ~70 MB
# checkpoint every --checkpoint-every, and ONE CONTINUOUSLY APPENDED LOG PER
# SERVER INSTANCE. At 32+ instances inside a lockstep loop those writes sit on
# the critical path over the network. /scratch is local ext4 on the NVMe.
#
# /scratch is VOLATILE -- it gets overwritten. So: run there, and copy back
# what must survive. Nothing durable is ever left only on scratch.
#
#   lanerl/run_on_scratch.sh <run-name> [extra args to python -m lanerl_train...]
set -euo pipefail

RUN="${1:?usage: run_on_scratch.sh <run-name> [args...]}"; shift
REPO=/mnt/nfs/projects/ahriuwu-lanerl
SCRATCH=/scratch/lanerl
PY=/home/dani/miniconda3/envs/ml/bin/python

mkdir -p "$SCRATCH/runs"
# Inputs are COPIED, not linked: a symlink into NFS would put the checkpoint
# read back on the network, and a symlink out of scratch would dangle when
# scratch is cleared.
mkdir -p "$SCRATCH/demos"
# Every checkpoint and dataset, not two hardcoded names. The screen-space
# action change meant a second BC policy (bc_policy_screen.pt) and a second
# demo set, and a name-by-name list silently leaves the new one on NFS --
# where it still works, so nothing complains, and the reason to copy at all
# is lost without a symptom.
cp -f "$REPO"/demos/*.pt "$SCRATCH/demos/" 2>/dev/null || true
cp -f "$REPO"/demos/*.npz "$SCRATCH/demos/" 2>/dev/null || true

sync_back() {
  local dest="$REPO/runs/$RUN"
  mkdir -p "$dest"
  # metrics and configs always; checkpoints are large, so only the newest few
  cp -f "$SCRATCH/runs/$RUN"/*.json* "$dest/" 2>/dev/null || true
  mkdir -p "$dest/checkpoints"
  ls -t "$SCRATCH/runs/$RUN"/checkpoints/*.pt 2>/dev/null | head -3 \
    | xargs -r -I{} cp -f {} "$dest/checkpoints/" 2>/dev/null || true
}
trap 'echo "[scratch] syncing back on exit"; sync_back' EXIT

# mirror every 2 minutes while the run proceeds, so a lost node costs 2 minutes
( while sleep 120; do sync_back; done ) &
MIRROR=$!
trap 'kill $MIRROR 2>/dev/null || true; sync_back' EXIT

cd "$REPO"
LANERL_RUNS_DIR="$SCRATCH/runs" PYTHONUNBUFFERED=1 \
  "$PY" -m lanerl_train --run-name "$RUN" "$@"
