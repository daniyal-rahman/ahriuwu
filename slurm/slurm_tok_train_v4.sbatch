#!/bin/bash
#SBATCH --job-name=tok-train-v4
#SBATCH --output=/mnt/nfs/projects/ahriuwu/logs/slurm_tok_train_%j.out
#SBATCH --partition=gpuhog
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --mail-user=danizainr@gmail.com
#SBATCH --mail-type=FAIL,END

set +u
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate ml
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/nfs/projects/ahriuwu/src:${PYTHONPATH:-}
export WANDB_MODE=online

cd /mnt/nfs/projects/ahriuwu
mkdir -p logs

CHECKPOINT_DIR=/mnt/nfs/projects/ahriuwu/checkpoints/tokenizer_v6
SAMPLE_DIR=/mnt/nfs/projects/ahriuwu/samples/tokenizer_v6
FRAMES_DIR=/scratch/ahriuwu/action_labeled_352png_train_flat

# Paper-faithful config (DreamerV4 §3.4):
#   AdamW (NOT 8-bit), betas=(0.9, 0.999), weight_decay=0.1
#   LR 1e-4 with linear warmup + cosine decay to 0
#   medium tokenizer (~130M params) per the prior successfully-converging run
#   B=2 T=16 grad_accum=16  →  effective batch 32  (gradient noise tractable)
#   mask ratio U(0, 0.9) — true paper recipe (no min=0.1 floor)
#   LPIPS-lib full-K=16 (half of B*T=32) — paper uses full but K=16 fits 5.2GB headroom

CMD=(
  python -u scripts/train_transformer_tokenizer.py
  --frames-dir "$FRAMES_DIR"
  --file-ext png
  --skip-resize
  --augment
  --checkpoint-dir "$CHECKPOINT_DIR"
  --sample-dir "$SAMPLE_DIR"
  --model-size medium
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
  --lr-schedule cosine
  --warmup-steps 500
  --epochs 2
  --mask-ratio-min 0.0
  --mask-warmup-steps 2000
  --step-save-interval 0
  --checkpoint-minutes 30
  --num-workers 6
  --log-interval 50
  --wandb-project ahriuwu-tokenizer
)
# Resume-if-exists. On the INITIAL submission this file doesn't exist yet,
# so the run starts from scratch (intended). On slurm REQUEUE (preemption,
# node failure, SIGTERM during shutdown), the file exists from a prior
# step-save-interval=1000 checkpoint and we resume — no progress lost.
if [ -f "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt" ]; then
  CMD+=(--resume "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt")
  echo "Resuming from $CHECKPOINT_DIR/transformer_tokenizer_latest.pt"
fi

PY_PID=""
trap 'if [ -n "$PY_PID" ]; then kill -TERM "$PY_PID" 2>/dev/null; wait "$PY_PID" 2>/dev/null; fi' TERM
trap 'if [ -n "$PY_PID" ]; then kill -USR1 "$PY_PID" 2>/dev/null; fi' USR1

"${CMD[@]}" &
PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?

exit "$EXIT_CODE"
