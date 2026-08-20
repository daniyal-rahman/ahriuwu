#!/bin/bash
# PAPER-PARITY Phase 2 on the 5080.
#
# Differences from ops/bc5080_clicks_watchdog.sh, all deliberate:
#   --unfreeze-backbone   Algorithm 1 Phase 2 finetunes the WORLD MODEL, not just
#                         the heads. Required for the video loss to reach anything.
#   --video-loss-weight 1 Eq (7), which the paper runs alongside Eq (9) throughout
#                         Phase 2. Dropping it is the leading suspect for BC
#                         overfitting 119 games (train fell while val flattened).
#   --movement-mode joint_noop
#                         ONE categorical over the 21x21 grid + NO_OP, replacing
#                         two per-axis categoricals + the sticky gate. Fixes the
#                         x-y independence assumption, keeps same-bin commands
#                         (18.6% of them), and needs no previous action -> Phase 3
#                         works with the plain log_prob.
#   --batch-size 2 --grad-accum 8
#                         Unfreezing fits only at batch 2 on a 16GB card (measured
#                         12.84 GiB; batch 4 OOMs). Accumulation restores the
#                         effective batch of 16 so this stays comparable to the
#                         frozen baseline. ~17.8 h/epoch at 6.3 samp/s.
#
# Stop: touch ops/bc5080_parity.stop
set -uo pipefail
cd /mnt/nfs/projects/ahriuwu
STOP=ops/bc5080_parity.stop
LOG=ops/bc_parity_5080.log
STATUS=ops/bc5080_parity_status.txt
CKPT=data/phase2_parity
rm -f "$STOP"; mkdir -p "$CKPT"

launch() {
  # WANDB_MODE=offline as well as --no-wandb: on 2026-08-19 an EXPIRED wandb TLS
  # cert (valid to 2025-05-29, clock reads 2026) raised inside a wandb network
  # call and killed the trainer. The watchdog restarted it, it resumed from the
  # same checkpoint, died again before the 20-min save -- 2.5 days pinned at
  # step 55216 producing nothing. Telemetry must never be able to stop training.
  setsid nohup env PYTHONPATH=/mnt/nfs/projects/ahriuwu/src CUDA_VISIBLE_DEVICES=0 \
    WANDB_MODE=offline WANDB_DISABLED=true \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/dani/miniconda3/envs/ml/bin/python -u scripts/train_agent_finetune.py \
    --dynamics-checkpoint rollout_stage/desktop_resume_8775_stripped.pt \
    --model-size medium --num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0 \
    --latents-dir /mnt/nfs/datasets/replay_latents_v7_bc \
    --labels-root /mnt/nfs/datasets/lol_replays_16_9_772 \
    --seq-len 16 --stride 8 --batch-size 2 --grad-accum 8 --epochs 10 \
    --lr 1e-4 --warmup-steps 2000 --num-workers 4 \
    --checkpoint-dir "$CKPT" --log-interval 25 \
    --resume auto --checkpoint-minutes 20 \
    --ability-pos-weight 1.0 --aux-state-weight 0.5 \
    --movement-mode joint_noop --action-dropout 0.15 \
    --unfreeze-backbone --video-loss-weight 1.0 \
    --movement-source clicks --val-games 6 --val-interval 500 \
    --no-use-8bit-adam --no-wandb \
    --dataset-cache "$CKPT/dataset_cache.pt" >> "$LOG" 2>&1 &
}

# STALL DETECTION. "Is the process alive" is the wrong question: on 2026-08-19 a
# wandb TLS failure killed the trainer every few minutes, this watchdog restarted
# it every time, and the run sat pinned at step 55216 for 2.5 days while looking
# perfectly healthy from outside (proc UP, GPU busy, watchdog UP). Track the last
# reported step and shout if it does not advance.
last_step=""; stall=0
while [ ! -f "$STOP" ]; do
  if ! pgrep -f "train_agent[_]finetune.*phase2_parity" >/dev/null; then
    echo "[wd $(date -u '+%m-%d %H:%M UTC')] launch/resume" >> "$LOG"
    launch; sleep 60
  fi
  # step counter from the newest VAL line (monotonic, written every --val-interval)
  cur=$(grep -a "VAL @" "$LOG" 2>/dev/null | tail -1 | sed -E 's/.*step ([0-9]+)\].*/\1/')
  if [ -n "$cur" ] && [ "$cur" = "$last_step" ]; then
    stall=$((stall+1))
    # 15 polls x 120s = 30 min with zero step progress
    if [ "$stall" -ge 15 ]; then
      echo "[wd $(date -u '+%m-%d %H:%M UTC')] *** STALLED at step $cur for ~$((stall*2))min ***" >> "$LOG"
      echo "STALLED at step $cur since $(date -u '+%m-%d %H:%M UTC')" > "$STATUS.stall"
      stall=0
    fi
  else
    stall=0; rm -f "$STATUS.stall"
  fi
  last_step="$cur"
  last=$(grep -aE "Epoch [0-9]+ \[" "$LOG" 2>/dev/null | tail -1)
  val=$(grep -a "VAL @" "$LOG" 2>/dev/null | tail -1)
  ck="$CKPT/agent_finetune_latest.pt"
  age=$([ -f "$ck" ] && echo "$((($(date +%s)-$(stat -c %Y "$ck"))/60))min" || echo NONE)
  up=$(pgrep -f "train_agent[_]finetune.*phase2_parity" >/dev/null && echo UP || echo DOWN)
  stallmsg=""; [ -f "$STATUS.stall" ] && stallmsg="  *** $(cat "$STATUS.stall") ***"
  printf '%s | proc:%s | ckpt:%s |%s %s\n  val: %s\n' \
    "$(date -u '+%m-%d %H:%M UTC')" "$up" "$age" "$stallmsg" "${last:-init}" "${val:-none yet}" > "$STATUS"
  sleep 120
done
echo "[wd] stop" >> "$LOG"
