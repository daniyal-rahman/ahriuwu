#!/usr/bin/env bash
# Threads vs processes, same code, same total instances.
#
# Threaded actors cap at ~1.8 of 16 cores because obs_build is pure Python and
# holds the GIL (1/0.55 by Amdahl; 1.81 measured). One process per actor gives
# each its own interpreter, so the serial fraction stops being shared.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/dani/miniconda3/envs/ml/bin/python
OUT=${OUT:-lanerl/logs/mode_compare.txt}
SECS=${SECS:-200}
: > "$OUT"

run_one() {
  local mode=$1 actors=$2 envs=$3 port=$4 aport=$5
  local name="mode-${mode}-a${actors}x${envs}"
  rm -rf "/scratch/lanerl/runs/${name}"
  LANERL_RUNS_DIR=/scratch/lanerl/runs PYTHONUNBUFFERED=1 \
    $PY lanerl/gil_probe.py --seconds "$SECS" --run-dir "/scratch/lanerl/runs/${name}" \
      --label "${mode}: ${actors} actors x ${envs} envs = $((actors * envs)) instances" -- \
      $PY -m lanerl_train --run-name "$name" --actor-mode "$mode" \
        --num-actors "$actors" --envs-per-actor "$envs" \
        --rollout-steps 128 --total-updates 100000 \
        --checkpoint-every 999999 --eval-every 0 --anchors '' --device cuda \
        --port-base "$port" --anchor-port-base "$aport" \
        --init-from demos/bc_policy.pt --lr 1e-5 --alpha 0.5 --no-end-on-death \
    2>&1 | tee -a "$OUT"
  sleep 20
}

run_one thread  1 12 21000 54000
run_one process 1 12 22000 55000
run_one thread  4  3 23000 56000
run_one process 4  3 24000 57000
run_one process 4 12 25000 58000

echo; echo "=== SUMMARY ==="; grep -E "^=== |CORES USED|throughput" "$OUT"
