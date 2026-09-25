#!/bin/bash
# Continue the GATED action-model BC run on the desktop 5080 (moved off the slow
# 1060). Resumes from data/phase2_bc_gate1060 (gs~80930, ~0.8 epoch) at batch 16.
# Same recipe: act8775 backbone + movement-gate + action-dropout 0.15 + aux-state
# 0.5 + calibrated BCE. WSD holds LR (decay-steps=0), so batch change is safe.
# Heartbeat + auto-restart. Stop: touch scratchpad/bc5080_gate.stop
set -uo pipefail
cd /mnt/nfs/projects/ahriuwu
STOP=scratchpad/bc5080_gate.stop
LOG=scratchpad/bc_gate_5080.log
STATUS=scratchpad/bc5080_gate_status.txt
CKPT=data/phase2_bc_gate1060
rm -f "$STOP"

launch() {
  setsid nohup env PYTHONPATH=/mnt/nfs/projects/ahriuwu/src CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/dani/miniconda3/envs/ml/bin/python -u scripts/train_agent_finetune.py \
    --dynamics-checkpoint rollout_stage/desktop_resume_8775_stripped.pt \
    --model-size medium --num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0 \
    --latents-dir /mnt/nfs/datasets/replay_latents_v7_bc \
    --labels-root /mnt/nfs/datasets/lol_replays_16_9_772 \
    --seq-len 16 --stride 8 --batch-size 16 --epochs 10 \
    --lr 3e-4 --warmup-steps 2000 --num-workers 8 \
    --checkpoint-dir "$CKPT" --log-interval 50 \
    --resume auto --checkpoint-minutes 20 \
    --ability-pos-weight 1.0 --aux-state-weight 0.5 \
    --movement-gate --action-dropout 0.15 \
    --no-use-8bit-adam \
    --dataset-cache "$CKPT/dataset_cache.pt" >> "$LOG" 2>&1 &
}

while [ ! -f "$STOP" ]; do
  if ! pgrep -f "train_agent[_]finetune.*gate1060" >/dev/null; then
    echo "[wd $(date '+%m-%d %H:%M')] launch/resume" >> "$LOG"
    launch; sleep 40
  fi
  last=$(grep -aE "Epoch [0-9]+ \[" "$LOG" 2>/dev/null | tail -1)
  ck="$CKPT/agent_finetune_latest.pt"
  age=$([ -f "$ck" ] && echo "$((($(date +%s)-$(stat -c %Y "$ck"))/60))min" || echo NONE)
  up=$(pgrep -f "train_agent[_]finetune.*gate1060" >/dev/null && echo UP || echo DOWN)
  printf '%s | proc:%s | ckpt:%s | %s\n' "$(date '+%m-%d %H:%M')" "$up" "$age" "${last:-init}" > "$STATUS"
  sleep 120
done
echo "[wd] stop" >> "$LOG"
