#!/bin/bash
# GATED action-model BC on the desktop 5080, retrained on the FIXED labels.
#
# Why a fresh run and not a resume from data/phase2_bc_gate1060: three of that
# run's supervision signals changed meaning on 2026-08-12, so its heads are
# fitted to targets that no longer exist.
#   * movement target: 47.5% of its "transitions" were camera drift, not commands
#   * movement_event:  the gate used to be supervised by BIN CHANGES, which
#                      swallowed 38% of real commands and injected drift noise.
#                      That is the leading suspect for why the trained gate sat
#                      at ~0.2 commands/s against a human ~2/s -- drift made
#                      "when does a command happen" look unpredictable, so the
#                      gate hedged toward never firing. This run is the test.
#   * enemy_visible:   29.6% of the corpus flipped 1->0, and enemy_hp_frac was
#                      being supervised on every one of those frames.
# The old checkpoint dir is left untouched (never overwrite checkpoints).
#
# --resume auto resolves to <checkpoint-dir>/agent_finetune_latest.pt, so with a
# NEW dir this starts fresh and still self-resumes after a crash/window close.
# Stop: touch ops/bc5080_clicks.stop
set -uo pipefail
cd /mnt/nfs/projects/ahriuwu
STOP=ops/bc5080_clicks.stop
LOG=ops/bc_clicks_5080.log
STATUS=ops/bc5080_clicks_status.txt
CKPT=data/phase2_bc_clicks
rm -f "$STOP"
mkdir -p "$CKPT"

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
    --movement-source clicks --val-games 6 \
    --no-use-8bit-adam \
    --dataset-cache "$CKPT/dataset_cache.pt" >> "$LOG" 2>&1 &
}

while [ ! -f "$STOP" ]; do
  if ! pgrep -f "train_agent[_]finetune.*phase2_bc_clicks" >/dev/null; then
    echo "[wd $(date '+%m-%d %H:%M')] launch/resume" >> "$LOG"
    launch; sleep 40
  fi
  last=$(grep -aE "Epoch [0-9]+ \[" "$LOG" 2>/dev/null | tail -1)
  val=$(grep -aE "val/" "$LOG" 2>/dev/null | tail -1)
  ck="$CKPT/agent_finetune_latest.pt"
  age=$([ -f "$ck" ] && echo "$((($(date +%s)-$(stat -c %Y "$ck"))/60))min" || echo NONE)
  up=$(pgrep -f "train_agent[_]finetune.*phase2_bc_clicks" >/dev/null && echo UP || echo DOWN)
  printf '%s | proc:%s | ckpt:%s | %s\n  val: %s\n' \
    "$(date '+%m-%d %H:%M')" "$up" "$age" "${last:-init}" "${val:-none yet}" > "$STATUS"
  sleep 120
done
echo "[wd] stop" >> "$LOG"
