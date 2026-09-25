#!/bin/bash
#SBATCH --job-name=tok-v7-yt
#SBATCH --output=/mnt/nfs/projects/ahriuwu/logs/slurm_tok_train_v7yt_%j.out
#SBATCH --open-mode=append
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
set +u
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate ml
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/mnt/nfs/projects/ahriuwu/src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
cd /mnt/nfs/projects/ahriuwu
mkdir -p logs

# PLATEAU-BREAK TEST (2026-08-05): v7 is hard-plateaued at ~29 dB after replays-only
# training (v7c: 9 days -> +0.28 dB). This run tests the DATA lever: mixed corpus =
# 142 replay matches + 250 YT games (352 jpgs renamed .png), with the blacked-HUD
# fix (--exclude-blacked-regions: constant-black pixels per clip excluded from the
# loss; no-op on replays) and stride 16 (1Hz phase-lock fix). Same paper-faithful
# arch/objective as v7 (512x16, te=4, full-frame MSE, tube masking).
# Seeded from v7c latest (~gs8.8k, the furthest checkpoint) with a fresh LR clock.
# CAPPED at --max-steps 5000 = the plateau verdict (~1 week on the 5080; external
# fixed-eval watcher on the login node tracks held-out PSNR per checkpoint; the
# null slope to beat is v7c's +0.03 dB/day). Extend max-steps on a positive verdict.
# Checkpoints go to REAL NFS (data/, not the desktop-local checkpoints symlink) so
# the login-side watcher can read them.
CHECKPOINT_DIR=/mnt/nfs/projects/ahriuwu/data/tokenizer_v7_yt
SAMPLE_DIR=/mnt/nfs/projects/ahriuwu/data/tokenizer_v7_yt/samples
FRAMES_DIR=/scratch/ahriuwu/tok_mixed_flat
SEED=/mnt/nfs/projects/ahriuwu/checkpoints/tokenizer_v7_cont/transformer_tokenizer_latest.pt
mkdir -p "$CHECKPOINT_DIR" "$SAMPLE_DIR"

CMD=(
  python -u scripts/train_transformer_tokenizer.py
  --frames-dir "$FRAMES_DIR"
  --file-ext png
  --skip-resize
  --augment
  --checkpoint-dir "$CHECKPOINT_DIR"
  --sample-dir "$SAMPLE_DIR"
  --model-size large
  --num-encoder-layers 8
  --num-decoder-layers 8
  --latent-dim 16
  --num-latents 512
  --temporal-every 4
  --mse-on-full-frame
  # Paper masking regime (paper 3.1): PER-IMAGE masking, flat p~U(0,0.9). --tube-masking
  # and --mask-warmup-steps are ablation opt-ins now — see PAPER_DEVIATIONS 1.1/1.2.
  --use-rope
  --sequence-length 20
  --stride 16
  --exclude-blacked-regions
  --batch-size 1
  --gradient-accumulation 64
  --gradient-checkpointing
  --lpips-frame-subsample 16
  --no-use-8bit-adam
  --adam-betas 0.9 0.999
  --lr 1e-4
  --weight-decay 0.1
  --lr-schedule wsd
  --warmup-steps 500
  --decay-steps 0
  --epochs 999
  --max-steps 5000
  --mask-ratio-min 0.0
  --mask-ratio-max 0.9
  --mask-warmup-steps 0
  --step-save-interval 30
  --checkpoint-warn-minutes 90
  --seed 0
  --num-workers 6
  --log-interval 50
  --wandb-project ahriuwu-tokenizer
  --wandb-tags v7-yt masked-mix plateau-test
)
if [ -f "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt" ]; then
  CMD+=(--resume "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt")
  echo "Resuming v7-yt from its own checkpoint (requeue)"
else
  CMD+=(--resume "$SEED" --reset-schedule)
  echo "First launch: seeding from v7c latest with --reset-schedule (fresh LR clock)"
fi

PY_PID=""
trap 'if [ -n "$PY_PID" ]; then kill -TERM "$PY_PID" 2>/dev/null; wait "$PY_PID" 2>/dev/null; fi' TERM

"${CMD[@]}" &
PY_PID=$!
wait "$PY_PID"
