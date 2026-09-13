#!/usr/bin/env bash
# Does --num-actors buy parallelism?
#
# The controlled comparison: hold TOTAL INSTANCES fixed and vary only the
# number of Python threads. Same servers, same decisions, same work -- the only
# difference is how many actor threads are contending for one GIL.
#
#   12 instances as 1x12, 2x6, 4x3
#   24 instances as 1x24, 4x6
#
# If actors were real parallelism, throughput rises with actor count. If the
# rollout loop is GIL-bound it is flat or FALLS, and `cores used` pins near 1.0
# regardless.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/home/dani/miniconda3/envs/ml/bin/python
SECS=${SECS:-180}
OUT=${OUT:-lanerl/logs/gil_sweep.txt}
: > "$OUT"

run_one() {
  local actors=$1 envs=$2 port=$3 aport=$4
  local name="gil-a${actors}x${envs}"
  local rundir="/scratch/lanerl/runs/${name}"
  rm -rf "$rundir"
  echo "--- ${actors} actors x ${envs} envs = $((actors * envs)) instances ---" | tee -a "$OUT"
  LANERL_RUNS_DIR=/scratch/lanerl/runs PYTHONUNBUFFERED=1 \
    $PY lanerl/gil_probe.py --seconds "$SECS" --run-dir "$rundir" \
      --label "${actors} actors x ${envs} envs = $((actors * envs)) instances" -- \
      $PY -m lanerl_train --run-name "$name" \
        --num-actors "$actors" --envs-per-actor "$envs" \
        --rollout-steps 128 --total-updates 100000 \
        --checkpoint-every 999999 --eval-every 0 --anchors '' --device cuda \
        --port-base "$port" --anchor-port-base "$aport" \
        --init-from demos/bc_policy.pt --lr 1e-5 --alpha 0.5 --no-end-on-death \
    2>&1 | tee -a "$OUT"
  # Servers are torn down by the probe's process-group kill, but a stray one
  # holds its port and the next config then fails to bind -- silently, as a
  # short run rather than an error. Give the kernel a moment to release them.
  sleep 15
}

run_one 1 12 15000 48000
run_one 2  6 16000 49000
run_one 4  3 17000 50000
run_one 1 24 18000 51000
run_one 4  6 19000 52000

echo; echo "=== SUMMARY ==="; grep -E "^===|CORES USED|throughput" "$OUT"
