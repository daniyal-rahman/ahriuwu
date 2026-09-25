#!/usr/bin/env bash
# Where does the PROCESS-actor design top out on this box?
#
# Threads capped at 1.8 of 16 cores by the GIL. Processes reached 6.28 cores /
# 3,409 dec/s at 4 actors x 12 envs, so the box still had 60% spare. This walks
# it up until something else binds -- most likely the servers (0.07 cores each)
# or the learner, which is still one process consuming every rollout.
#
# RESULT (2026-09-13, desktop: 16 cores / 30 GB / RTX 5080):
#
#     4 actors x 12 =  48 instances    6.43 cores    3,634 dec/s
#     8 actors x 12 =  96 instances   10.13 cores    4,829 dec/s   <- use this
#    12 actors x 12 = 144 instances   11.62 cores    OOM KILLED
#
# The ceiling is RAM, not CPU. 144 instances took 10 oom_kill events while
# still BOOTING, at 11.6 of 16 cores -- so there are cores left that this box
# cannot feed. ~200 MB per game server against 30 GB total.
#
# At the working ceiling the CPU finally splits sensibly: python 5.83 cores,
# .NET game servers 4.19. Under threads it was 1.8 cores total, all of it
# contending.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/dani/miniconda3/envs/ml/bin/python
OUT=${OUT:-lanerl/logs/proc_scale.txt}
SECS=${SECS:-200}
: > "$OUT"

run_one() {
  local actors=$1 envs=$2 port=$3 aport=$4
  local name="proc-a${actors}x${envs}"
  rm -rf "/scratch/lanerl/runs/${name}"
  LANERL_RUNS_DIR=/scratch/lanerl/runs PYTHONUNBUFFERED=1 \
    $PY lanerl/gil_probe.py --seconds "$SECS" --run-dir "/scratch/lanerl/runs/${name}" \
      --label "${actors} actors x ${envs} envs = $((actors * envs)) instances" -- \
      $PY -m lanerl_train --run-name "$name" --actor-mode process \
        --num-actors "$actors" --envs-per-actor "$envs" \
        --rollout-steps 128 --total-updates 100000 \
        --checkpoint-every 999999 --eval-every 0 --anchors '' --device cuda \
        --port-base "$port" --anchor-port-base "$aport" \
        --init-from demos/bc_policy.pt --lr 1e-5 --alpha 0.5 --no-end-on-death \
    2>&1 | tee -a "$OUT"
  sleep 20
}

run_one  4 12 27000 60000
run_one  8 12 29000 61000
run_one 12 12 31000 62000

echo; echo "=== SUMMARY ==="; grep -E "^=== |CORES USED|throughput" "$OUT"
