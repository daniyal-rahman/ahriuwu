#!/usr/bin/env bash
# Where does the decline begin, and WHICH phase grows?
#
# The phase timer has only ever been run at 1/4/12 instances, where it says the
# bottleneck is obs_build and the servers are 5%. The scaling probe's decline
# (498 -> 209 dec/s) was measured from 8 to 72. Those are different regimes and
# the conclusion from one was being applied to the other -- including by me.
# This walks the phase split up into the regime where the decline actually is.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/dani/miniconda3/envs/ml/bin/python
for n in 12 24 48; do
  echo "=== ${n} instances, 1 actor ==="
  LANERL_TIME_PHASES=1 LANERL_TIME_PHASES_EVERY=400 \
  LANERL_RUNS_DIR=/scratch/lanerl/runs PYTHONUNBUFFERED=1 timeout 420 \
    $PY -m lanerl_train --run-name "phase-n${n}" \
      --num-actors 1 --envs-per-actor "$n" --rollout-steps 128 --total-updates 100000 \
      --checkpoint-every 999999 --eval-every 0 --anchors '' --device cuda \
      --port-base $((20000 + n * 200)) --anchor-port-base $((53000 + n * 200)) \
      --init-from demos/bc_policy.pt --lr 1e-5 --alpha 0.5 --no-end-on-death 2>&1 \
    | grep -E "PHASE|decisions_per_s" | tail -3
  sleep 15
done
