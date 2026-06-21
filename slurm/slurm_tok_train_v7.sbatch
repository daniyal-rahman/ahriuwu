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

# v7 args come from the SINGLE SOURCE OF TRUTH (scripts/v7_train_args.sh), shared
# with the multi-GPU DDP launcher. Only run-specific bits are set here. The Slurm
# autoresume requeue must CONTINUE the original LR schedule, so RESET_SCHEDULE is
# left unset (the DDP one-shot continuation sets it).
RESUME="$CHECKPOINT_DIR/transformer_tokenizer_latest.pt"
WANDB_TAGS="v7 d1024-8x8 paper-encoder-512x16 temporal-every-4 wsd-with-decay eff-batch-64"
source scripts/v7_train_args.sh    # -> V7_ARGS  (reads FRAMES_DIR/CHECKPOINT_DIR/SAMPLE_DIR set above)
if [ -f "$RESUME" ]; then
  echo "Resuming v7 from $RESUME"
else
  echo "Fresh v7 cold-start (no warm-start from v6 - architecture differs)"
fi
CMD=(python -u scripts/train_transformer_tokenizer.py "${V7_ARGS[@]}")

PY_PID=""
trap 'if [ -n "$PY_PID" ]; then kill -TERM "$PY_PID" 2>/dev/null; wait "$PY_PID" 2>/dev/null; fi' TERM
trap 'if [ -n "$PY_PID" ]; then kill -USR1 "$PY_PID" 2>/dev/null; fi' USR1

"${CMD[@]}" &
PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?
exit "$EXIT_CODE"
