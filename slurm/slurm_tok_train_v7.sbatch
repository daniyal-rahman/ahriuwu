#!/bin/bash
#SBATCH --job-name=tok-train-v7
#SBATCH --output=/mnt/nfs/projects/ahriuwu/logs/slurm_tok_train_v7_%j.out
#SBATCH --open-mode=append
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --mail-user=danizainr@gmail.com
#SBATCH --mail-type=FAIL,END

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

# v7 production: paper-faithful 512x16 ENCODER bottleneck on D=1024 / 8+8 layers /
# temporal_every=4. B=1 T=20 grad_accum=64 -> effective batch 64 (2x v6's 32).
# T=20 at 20 fps = 1 s of game footage per sample.
# Per VRAM probe (job 73): T=16 used 12.88GB, T=24 used 13.65GB -> T=20 ~13.3GB
# (~2.4GB headroom on 5080), step ~1.9s/micro-batch -> ~122 s/optstep.
# 6000 optsteps -> ~8.5 days wallclock.
# NB: user chose 512x16 despite Step-5 overfit showing 256x32 was ~7 dB better
# at matched params. Paper's pre-reshape encoder structure with more spatial readers.
CHECKPOINT_DIR=/mnt/nfs/projects/ahriuwu/checkpoints/tokenizer_v7
SAMPLE_DIR=/mnt/nfs/projects/ahriuwu/samples/tokenizer_v7
FRAMES_DIR=/scratch/ahriuwu/action_labeled_352png_train_flat

CMD=(
  python -u scripts/train_transformer_tokenizer.py
  --frames-dir "$FRAMES_DIR"
  --file-ext png
  --skip-resize
  --augment
  --checkpoint-dir "$CHECKPOINT_DIR"
  --sample-dir "$SAMPLE_DIR"
  # --- ARCHITECTURE: paper encoder 512x16 on D=1024 8+8, temporal-every-4 ---
  --model-size large
  --num-encoder-layers 8
  --num-decoder-layers 8
  --latent-dim 16
  --num-latents 512
  --temporal-every 4
  # --- training (B=1 T=20 grad_accum=64 = eff batch 64; T=20 = 1s of footage @ 20fps) ---
  --use-rope
  --sequence-length 20
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
  --decay-steps 1500       # last 25% of 6k-optstep run; v6 had decay=0
  --epochs 999
  --max-steps 6000         # ~8.5 days at ~122 s/optstep with T=20
  --mask-ratio-min 0.0
  --mask-warmup-steps 2000
  --step-save-interval 30  # ~61 min between saves at 122s/optstep
  --checkpoint-warn-minutes 90
  --num-workers 6
  --log-interval 50
  --wandb-project ahriuwu-tokenizer
  --wandb-tags v7 d1024-8x8 paper-encoder-512x16 temporal-every-4 wsd-with-decay eff-batch-64
)
# Resume strategy:
# 1. If v7 has its own latest.pt, resume from there (normal requeue cycle).
# 2. Otherwise COLD START - no warm-start from v6 (architecture differs: bottleneck
#    shape AND temporal pattern would shape-mismatch).
if [ -f "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt" ]; then
  CMD+=(--resume "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt")
  echo "Resuming v7 from $CHECKPOINT_DIR/transformer_tokenizer_latest.pt"
else
  echo "Fresh v7 cold-start (no warm-start from v6 - architecture differs)"
fi

PY_PID=""
trap 'if [ -n "$PY_PID" ]; then kill -TERM "$PY_PID" 2>/dev/null; wait "$PY_PID" 2>/dev/null; fi' TERM
trap 'if [ -n "$PY_PID" ]; then kill -USR1 "$PY_PID" 2>/dev/null; fi' USR1

"${CMD[@]}" &
PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?
exit "$EXIT_CODE"
