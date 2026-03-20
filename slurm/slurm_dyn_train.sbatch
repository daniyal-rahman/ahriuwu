#!/bin/bash
#SBATCH --job-name=dyn-train
#SBATCH --output=/mnt/storage/ahriuwu-data/logs/slurm_dyn_train_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --qos=gpu-long
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --mail-user=danizainr@gmail.com
#SBATCH --mail-type=FAIL,END

set +u
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate ml
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /mnt/storage/ahriuwu

CHECKPOINT_DIR=/mnt/storage/ahriuwu-data/checkpoints
LATENTS_DIR=/opt/ahriuwu/latents_pt

CMD=(
  python scripts/train_dynamics.py
  --latents-dir "$LATENTS_DIR"
  --packed
  --checkpoint-dir "$CHECKPOINT_DIR"
  --model-size small
  --latent-dim 48
  --tokenizer-type transformer
  --alternating-lengths
  --seq-len-short 32
  --seq-len-long 64
  --batch-size-short 2
  --batch-size-long 1
  --long-ratio 0.1
  --gradient-accumulation 16
  --gradient-accumulation-short 16
  --gradient-accumulation-long 32
  --gradient-checkpointing
  --shortcut-forcing
  --shortcut-k-max 64
  --num-kv-heads 4
  --num-register-tokens 8
  --soft-cap 50.0
  --lr 3e-4
  --warmup-steps 5000
  --checkpoint-minutes 60
  --epochs 50
  --wandb --wandb-project ahriuwu
)
if [ -f "$CHECKPOINT_DIR/dynamics_latest.pt" ]; then
  CMD+=(--resume "$CHECKPOINT_DIR/dynamics_latest.pt")
fi

PY_PID=""
trap 'if [ -n "$PY_PID" ]; then kill -TERM "$PY_PID" 2>/dev/null; wait "$PY_PID" 2>/dev/null; fi' TERM
trap 'if [ -n "$PY_PID" ]; then kill -USR1 "$PY_PID" 2>/dev/null; fi' USR1

"${CMD[@]}" &
PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?

exit "$EXIT_CODE"
