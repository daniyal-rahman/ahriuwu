#!/bin/bash
#SBATCH --job-name=dyn-train
#SBATCH --output=/mnt/storage/ahriuwu-data/logs/slurm_dyn_train_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=14-00:00:00
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

# Dataloaders use max batch size (B=8 short, B=4 long).
# Shortcut steps slice to B=2/B=1 (3 forward passes need less VRAM).
# Standard steps use full batch. ~2x faster than old dual-dataloader approach.
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
  --batch-size-short 8
  --batch-size-long 4
  --long-ratio 0.1
  --gradient-accumulation 4
  --gradient-accumulation-short 4
  --gradient-accumulation-long 8
  --gradient-checkpointing
  --no-compile
  --shortcut-forcing
  --shortcut-k-max 64
  --bootstrap-weight 3.0
  --num-kv-heads 4
  --num-register-tokens 8
  --soft-cap 50.0
  --lr 3e-4
  --warmup-steps 3000
  --eval-interval 200
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
