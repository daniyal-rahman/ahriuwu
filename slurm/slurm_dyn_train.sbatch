#!/bin/bash
#SBATCH --job-name=dyn-train
#SBATCH --partition=gpup
#SBATCH --output=/mnt/storage/data/ahriuwu/logs/slurm_dyn_train_%j.out
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
export PYTHONPATH=/mnt/nfs/projects/ahriuwu/src

cd /mnt/nfs/projects/ahriuwu

CHECKPOINT_DIR=/mnt/storage/data/ahriuwu/checkpoints/dynamics_v7_replay
LATENTS_DIR=/scratch/ahriuwu/dynamics_replay_latents_v7_dim32

# Base flow-matching pretrain (standard x-prediction diffusion forcing) at
# 128/256 context, medium model, gradient checkpointing. Shortcut forcing is
# OMITTED here: its GC-disabled 3-forward-pass steps OOM medium@128 on the 5080.
# Do shortcut distillation as a separate Phase B (finetune_shortcut.py) after.
CMD=(
  python scripts/train_dynamics.py
  --latents-dir "$LATENTS_DIR"
  --packed
  --checkpoint-dir "$CHECKPOINT_DIR"
  --model-size medium
  --latent-dim 32
  --tokenizer-type transformer
  --alternating-lengths
  --seq-len-short 128
  --seq-len-long 256
  --batch-size-short 2
  --batch-size-long 1
  --long-ratio 0.1
  --gradient-accumulation 8
  --gradient-accumulation-short 8
  --gradient-accumulation-long 16
  --gradient-checkpointing
  --no-compile
  --num-kv-heads 4
  --num-register-tokens 8
  --soft-cap 50.0
  --use-actions
  --labels-root /mnt/nfs/datasets/lol_replays_16_9_772
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
