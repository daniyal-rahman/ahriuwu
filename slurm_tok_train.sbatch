#!/bin/bash
#SBATCH --job-name=tok-train
#SBATCH --output=/mnt/storage/ahriuwu/repo/logs/slurm_tok_train_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --qos=training
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --mail-user=danizainr@gmail.com
#SBATCH --mail-type=FAIL,END

set +u
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate ml
set -euo pipefail

export PYTHONUNBUFFERED=1

cd /mnt/storage/ahriuwu/repo

CHECKPOINT_DIR=/mnt/storage/ahriuwu/repo/checkpoints
FRAMES_DIR=/mnt/storage/ahriuwu/frames/frames

CMD=(
  python scripts/train_transformer_tokenizer.py
  --frames-dir "$FRAMES_DIR"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --model-size medium
  --use-rope
  --batch-size 2
  --gradient-accumulation 16
  --gradient-checkpointing
  --sequence-length 16
  --lr 3e-4
  --warmup-steps 5000
  --step-save-interval 0
  --checkpoint-minutes 60
  --epochs 50
  --wandb --wandb-project ahriuwu
)
if [ -f "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt" ]; then
  CMD+=(--resume "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt")
fi

PY_PID=""
trap 'if [ -n "$PY_PID" ]; then kill -TERM "$PY_PID" 2>/dev/null; wait "$PY_PID" 2>/dev/null; fi' TERM
trap 'if [ -n "$PY_PID" ]; then kill -USR1 "$PY_PID" 2>/dev/null; fi' USR1

"${CMD[@]}" &
PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?

exit "$EXIT_CODE"
