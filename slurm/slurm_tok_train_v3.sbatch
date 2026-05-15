#!/bin/bash
#SBATCH --job-name=tok-train-v3
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

CHECKPOINT_DIR=/mnt/nfs/projects/ahriuwu/checkpoints/tokenizer_v3
SAMPLE_DIR=/mnt/nfs/projects/ahriuwu/samples/tokenizer_v3
FRAMES_DIR=/scratch/ahriuwu/action_labeled_flat

CMD=(
  python -u scripts/train_transformer_tokenizer.py
  --frames-dir "$FRAMES_DIR"
  --file-ext png
  --checkpoint-dir "$CHECKPOINT_DIR"
  --sample-dir "$SAMPLE_DIR"
  --model-size small
  --use-rope
  --sequence-length 16
  --batch-size 2
  --gradient-accumulation 16
  --gradient-checkpointing
  --lpips-frame-subsample 16
  --lr 3e-4
  --weight-decay 0.01
  --warmup-steps 1500
  --epochs 1
  --mask-warmup-steps 2000
  --step-save-interval 1000
  --num-workers 6
  --log-interval 100
  --wandb-project ahriuwu-tokenizer
)
# NOTE: deliberately no --resume here. Each launch starts from scratch.
# To continue an existing run, add `--resume "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt"`
# to CMD after confirming that's desired.

PY_PID=""
trap 'if [ -n "$PY_PID" ]; then kill -TERM "$PY_PID" 2>/dev/null; wait "$PY_PID" 2>/dev/null; fi' TERM
trap 'if [ -n "$PY_PID" ]; then kill -USR1 "$PY_PID" 2>/dev/null; fi' USR1

"${CMD[@]}" &
PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?

exit "$EXIT_CODE"
