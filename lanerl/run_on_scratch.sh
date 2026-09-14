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

# torch's default sharing strategy passes every shared tensor as a FILE
# DESCRIPTOR over a unix socket, so the parent holds descriptors roughly in
# proportion to (actors x envs x tensors in flight). The login shell's soft
# limit here is 1024 against a 1048576 hard limit, and at 4 actors x 6 envs
# that ran out 16 minutes in: rl-screen-bc-0914 died at update 206 with
# "RuntimeError: received 0 items of ancdata", which is recvfds() finding no
# descriptor in the message.
#
# Raising the soft limit is the fix. The file_system strategy is the other
# way out and was tried: it produced c10::Error from MapAllocator::close and
# SIGABRTed the actors five minutes in, which is worse than what it fixed.
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
echo "[scratch] fd soft limit: $(ulimit -n)"

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
