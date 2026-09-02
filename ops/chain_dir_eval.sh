#!/bin/bash
# Run the direction eval as soon as the 5080 frees up. Chained rather than
# hand-launched: a GPU that finishes a job and then sits idle waiting for someone
# to notice cost ~11h on 2026-08-29.
set -uo pipefail
REPO=/mnt/nfs/projects/ahriuwu
LOG=$REPO/ops/chain_dir_eval.log
echo "$(date -u '+%m-%d %H:%M UTC') waiting for the GPU" >> "$LOG"
# free = no python holding CUDA for 3 consecutive checks
idle=0
while [ "$idle" -lt 3 ]; do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then idle=$((idle+1)); else idle=0; fi
  sleep 60
done
echo "$(date -u '+%m-%d %H:%M UTC') GPU free -> dir_eval" >> "$LOG"
exec "$REPO/ops/dir_eval.sh"
