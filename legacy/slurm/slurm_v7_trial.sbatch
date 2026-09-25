#!/bin/bash
#SBATCH --job-name=v7-trial
#SBATCH --output=/mnt/nfs/projects/ahriuwu/logs/slurm_v7trial_%j.out
#SBATCH --open-mode=append
#SBATCH --partition=gpuhog
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --requeue

set +u
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate ml
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/mnt/nfs/projects/ahriuwu/src:${PYTHONPATH:-}
export WANDB_MODE=online

cd /mnt/nfs/projects/ahriuwu
mkdir -p logs

# v7 trial: paper-faithful 256x32 bottleneck on D=1024 (8+8 layers, temporal-every-4)
# Full production recipe (mask, LPIPS, aug, wd 0.1), short 3000-optstep trial run.
# Goal: confirm holdout PSNR trajectory is healthy before committing to multi-day full run.
CHECKPOINT_DIR=/mnt/nfs/projects/ahriuwu/checkpoints/tokenizer_v7_trial
SAMPLE_DIR=/mnt/nfs/projects/ahriuwu/samples/tokenizer_v7_trial
FRAMES_DIR=/scratch/ahriuwu/action_labeled_352png_train_flat

CMD=(
  python -u scripts/train_transformer_tokenizer.py
  --frames-dir "$FRAMES_DIR"
  --file-ext png
  --skip-resize
  --augment
  --checkpoint-dir "$CHECKPOINT_DIR"
  --sample-dir "$SAMPLE_DIR"
  # --- ARCHITECTURE: paper-faithful 256x32 on D=1024 8+8, temporal every 4 ---
  --model-size large
  --num-encoder-layers 8
  --num-decoder-layers 8
  --latent-dim 32
  --num-latents 256
  --temporal-every 4
  # --- training ---
  --use-rope
  --sequence-length 16
  --batch-size 2
  --gradient-accumulation 16
  --gradient-checkpointing
  --lpips-frame-subsample 16
  --no-use-8bit-adam
  --adam-betas 0.9 0.999
  --lr 1e-4
  --weight-decay 0.1
  --lr-schedule wsd
  --decay-steps 0
  --warmup-steps 500
  --epochs 999            # epoch count irrelevant; --max-steps stops us first
  --max-steps 3000
  --mask-ratio-min 0.0
  --mask-warmup-steps 2000
  --step-save-interval 300
  --checkpoint-warn-minutes 60
  --num-workers 6
  --log-interval 50
  --wandb-project ahriuwu-tokenizer
  --wandb-tags v7-trial paper-bottleneck temporal-every-4 d1024-8x8
)
# Fresh run — no resume (v7_trial checkpoint_dir starts empty).
if [ -f "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt" ]; then
  CMD+=(--resume "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt")
  echo "Resuming v7 trial from $CHECKPOINT_DIR/transformer_tokenizer_latest.pt"
else
  echo "Fresh v7 trial (no prior checkpoint)"
fi

PY_PID=""
trap 'if [ -n "$PY_PID" ]; then kill -TERM "$PY_PID" 2>/dev/null; wait "$PY_PID" 2>/dev/null; fi' TERM
trap 'if [ -n "$PY_PID" ]; then kill -USR1 "$PY_PID" 2>/dev/null; fi' USR1

"${CMD[@]}" &
PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?
exit "$EXIT_CODE"
