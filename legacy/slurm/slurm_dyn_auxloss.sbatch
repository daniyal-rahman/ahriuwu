#!/bin/bash
#SBATCH --job-name=dyn-auxloss
#SBATCH --partition=gpup
#SBATCH --output=/mnt/storage/data/ahriuwu/logs/slurm_dyn_yt578_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=14-00:00:00
#SBATCH --qos=gpu-long
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --mail-user=danizainr@gmail.com
#SBATCH --mail-type=FAIL,END

# Broad-video world-model pretrain (DreamerV4 recipe): 578 games UNLABELED
# (453 YT + 125 replay, all via PackedLatentSequenceDataset -> no_action_embed),
# on the fast NVMe scratch. WARM-STARTED from the furthest action-conditioned
# checkpoint (135, ~step 9k): the shared dynamics weights transfer via strict=False;
# the action layers drop (relearned in a later action-conditioned fine-tune phase).
# The seed at $CHECKPOINT_DIR/dynamics_latest.pt has batch_idx/global_step zeroed
# and no optimizer state -> fresh optimizer + fresh warmup (gentle re-entry).
set +u
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate ml
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/mnt/nfs/projects/ahriuwu/src
cd /mnt/nfs/projects/ahriuwu

CHECKPOINT_DIR=/mnt/storage/data/ahriuwu/checkpoints/dyn179_auxloss
LATENTS_DIR=/scratch/ahriuwu/dynamics_all_v7   # 578 combined latents on NVMe

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
  --stride 64

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
  --lr 3e-4
  --warmup-steps 3000
  --eval-interval 50
  --checkpoint-minutes 60
  --epochs 50
  --pixel-hud-loss
  --pixel-hud-aux
  --pixel-loss-frames 4
  --tokenizer-ckpt /mnt/storage/data/ahriuwu-checkpoints/tokenizer_v7/transformer_tokenizer_latest.pt
  --hud-mask /mnt/nfs/projects/ahriuwu/scratchpad/hud_valid_mask_352.pt
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
exit $?
