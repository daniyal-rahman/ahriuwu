#!/bin/bash
# Free desktop smoke: validate the action-conditioned MIXED pipeline end-to-end
# (placeholder YT labels -> mixed ReplayLatentSequenceDataset -> action-conditioned
# training + eval) on the 5080, before any Vast spend. Also calibrates 1-GPU step/s.
# Bypasses `conda activate` (its activate-binutils breaks under set -u) by using the
# env python directly.
set -o pipefail
cd /mnt/nfs/projects/ahriuwu
export PATH=/home/dani/miniconda3/envs/ml/bin:$PATH
export LD_LIBRARY_PATH=/home/dani/miniconda3/envs/ml/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/nfs/projects/ahriuwu/src
PY=/home/dani/miniconda3/envs/ml/bin/python
R=~/bin/rclone
SM=/mnt/storage/data/ahriuwu/smoke_mixed
LAT=$SM/latents; LAB=$SM/labels
rm -rf "$SM"; mkdir -p "$LAT" "$LAB" "$SM/ckpt"

echo "[smoke] stage 3 replay latents + their real labels"
for m in $(ls /mnt/nfs/datasets/replay_latents_v7_bc/NA1_*.pt | head -3); do
  id=$(basename "$m" .pt)
  ln -sf "$m" "$LAT/$id.pt"
  ln -sf "/mnt/nfs/datasets/lol_replays_16_9_772/$id" "$LAB/$id"
done
echo "[smoke] pull 2 YT latents from R2"
for y in -05oyD6OXE8 -4VGr4S0tPU; do
  $R copyto "r2:ahriuwu-yt-pretrain/dynamics_yt_subset80/$y.pt" "$LAT/$y.pt" --s3-no-check-bucket 2>/dev/null
done
echo "[smoke] latents staged: $(ls "$LAT"/*.pt | wc -l)"
echo "[smoke] generate placeholder YT labels"
$PY scripts/gen_yt_placeholder_labels.py --latents-dir "$LAT" --out "$LAB" --fps 30

echo "[smoke] launch action-conditioned MIXED training (plain latent loss, no pixel-HUD)"
echo "[smoke] (timeout-bounded; validates load+train+eval + gives 1-GPU step/s)"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
timeout 600 $PY -u scripts/train_dynamics.py \
  --use-actions --labels-root "$LAB" --latents-dir "$LAT" --checkpoint-dir "$SM/ckpt" \
  --model-size medium --latent-dim 32 \
  --num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0 --shortcut-k-max 64 \
  --alternating-lengths --seq-len-short 128 --seq-len-long 256 \
  --batch-size-short 1 --batch-size-long 1 --long-ratio 0.1 --stride 64 \
  --gradient-accumulation 8 --gradient-accumulation-short 8 --gradient-accumulation-long 16 \
  --gradient-checkpointing --no-compile \
  --lr 3e-4 --lr-schedule wsd --warmup-steps 50 --adam-betas 0.9 0.999 \
  --independent-frame-ratio 0.3 --seed 0 --holdout-videos 1 \
  --epochs 1 --eval-interval 40 --checkpoint-minutes 999 --num-workers 3 \
  2>&1 | grep -vE "UserWarning|warnings.warn"
echo "[smoke] DONE (timeout=stopped-by-design)"
