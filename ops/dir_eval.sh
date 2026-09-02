#!/bin/bash
# Direction eval on the retrain vs the deployed model, both with the crutch CUT.
# This is the number the bronze A/B was supposed to produce and never did.
set -uo pipefail
REPO=/mnt/nfs/projects/ahriuwu
cd "$REPO" || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
LOG=$REPO/ops/dir_eval.log
for pair in "retrainA:$REPO/data/bronze_retrain_A/agent_finetune_latest.pt" \
            "deployed:$REPO/data/phase2_bc_clicks/agent_finetune_latest.pt"; do
  name=${pair%%:*}; ck=${pair#*:}
  [ -f "$ck" ] || { echo "  $name: missing" >> "$LOG"; continue; }
  echo "$(date -u '+%H:%M UTC') $name" >> "$LOG"
  PYTHONPATH=src /home/dani/miniconda3/envs/ml/bin/python -u scratchpad/lane/probe_single.py \
    --ckpt "$ck" --movement-action-mode none --tag "$name" \
    >> "$REPO/ops/dir_${name}.log" 2>&1
  echo "  rc=$?" >> "$LOG"
done
echo "$(date -u '+%m-%d %H:%M UTC') dir_eval done" >> "$LOG"
