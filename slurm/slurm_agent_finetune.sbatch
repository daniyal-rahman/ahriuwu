#!/bin/bash
#SBATCH --job-name=agent-ft
#SBATCH --output=/mnt/storage/ahriuwu-data/logs/slurm_agent_ft_%j.out
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
DYNAMICS_CKPT=$CHECKPOINT_DIR/dynamics_small_step34k_phase1_final.pt
TOKENIZER_CKPT=$CHECKPOINT_DIR/transformer_tokenizer_latest.pt
LATENTS_DIR=/opt/ahriuwu/latents_pt
FEATURES_DIR=/mnt/storage/ahriuwu-data/processed

CMD=(
  python scripts/train_agent_finetune.py
  --dynamics-checkpoint "$DYNAMICS_CKPT"
  --tokenizer-checkpoint "$TOKENIZER_CKPT"
  --latents-dir "$LATENTS_DIR"
  --features-dir "$FEATURES_DIR"
  --model-size small
  --latent-dim 48
  --seq-len 32
  --batch-size 2
  --mtp-length 9
  --num-kv-heads 4
  --num-register-tokens 8
  --soft-cap 50.0
  --no-compile
  --lr 3e-4
  --warmup-steps 2000
  --epochs 20
  --wandb --wandb-project ahriuwu
)

# Resume from Phase 2 checkpoint if it exists
if [ -f "$CHECKPOINT_DIR/agent_finetune_latest.pt" ]; then
  CMD+=(--resume "$CHECKPOINT_DIR/agent_finetune_latest.pt")
fi

PY_PID=""
trap 'if [ -n "$PY_PID" ]; then kill -TERM "$PY_PID" 2>/dev/null; wait "$PY_PID" 2>/dev/null; fi' TERM
trap 'if [ -n "$PY_PID" ]; then kill -USR1 "$PY_PID" 2>/dev/null; fi' USR1

"${CMD[@]}" &
PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?

exit "$EXIT_CODE"
