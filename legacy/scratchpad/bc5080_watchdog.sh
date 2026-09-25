#!/bin/bash
# Watchdog for the act8775 BC run on the desktop 5080 (mirrors bc_watchdog.sh's
# lesson: heartbeat + auto-restart with --resume auto). Stop cleanly:
#   touch scratchpad/bc5080.stop   (then kill the trainer if you want it down NOW)
set -uo pipefail
cd /mnt/nfs/projects/ahriuwu
STOP=scratchpad/bc5080.stop
LOG=scratchpad/bc_5080_act8775.log
STATUS=scratchpad/bc5080_status.txt
CKPT=data/phase2_bc_garen_act8775
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
    --resume auto --checkpoint-minutes 20 --ability-pos-weight 1.0 \
    --aux-state-weight 0.5 \
    --dataset-cache "$CKPT/dataset_cache.pt" >> "$LOG" 2>&1 &
}

while [ ! -f "$STOP" ]; do
  if ! pgrep -f "train_agent[_]finetune.*act8775" >/dev/null; then
    echo "[watchdog $(date '+%m-%d %H:%M')] trainer down -> relaunch (resume auto)" >> "$LOG"
    launch
    sleep 30
  fi
  last=$(grep -aE "Epoch [0-9]+ \[" "$LOG" 2>/dev/null | tail -1)
  ck="$CKPT/agent_finetune_latest.pt"
  if [ -f "$ck" ]; then age="$((($(date +%s) - $(stat -c %Y "$ck")) / 60))min"; else age="NONE-YET"; fi
  alive=$(pgrep -f "train_agent[_]finetune.*act8775" >/dev/null && echo UP || echo DOWN)
  printf '%s | proc:%s | ckpt_age:%s | %s\n' \
    "$(date '+%m-%d %H:%M')" "$alive" "$age" "${last:-<initializing>}" > "$STATUS"
  sleep 120
done
echo "[watchdog] stop file -> exit" >> "$LOG"
