#!/bin/bash
# Launch the v7 tokenizer training across N GPUs on ONE box via DDP (torchrun).
# Matches slurm_tok_train_v7.sbatch hyperparameters; resumes the v7 checkpoint
# with a reset LR schedule. Designed for a single-box 8x RTX 5090 rental.
#
# Consumer Blackwell/Ada cards have NO working GPU P2P -> NCCL must bounce via
# host RAM, so NCCL_P2P_DISABLE/SHM_DISABLE are mandatory or NCCL hangs at init.
#
# Override via env: NGPU, FRAMES_DIR, RESUME, CHECKPOINT_DIR, SAMPLE_DIR,
#                   MAX_STEPS, NUM_WORKERS, WANDB_MODE
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1                     # consumer cards: no P2P -> host-RAM bounce
export NCCL_SHM_DISABLE=1                     # more robust on multi-4090/5090 boxes
export WANDB_MODE="${WANDB_MODE:-online}"

NGPU="${NGPU:-8}"
# v7 run-specific defaults for the DDP box (read by the shared args file)
FRAMES_DIR="${FRAMES_DIR:-/scratch/ahriuwu/frames_train_flat}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints/tokenizer_v7_ddp}"
SAMPLE_DIR="${SAMPLE_DIR:-/workspace/samples/tokenizer_v7_ddp}"
RESUME="${RESUME:-$CHECKPOINT_DIR/transformer_tokenizer_latest.pt}"
MAX_STEPS="${MAX_STEPS:-6000}"
NUM_WORKERS="${NUM_WORKERS:-6}"              # PER RANK (NGPU*NUM_WORKERS total dataloader procs)
WANDB_TAGS="${WANDB_TAGS:-v7 ddp 8x5090 d1024-8x8 paper-encoder-512x16}"
RESET_SCHEDULE="${RESET_SCHEDULE:-1}"        # continuation past decay -> fresh LR schedule

source scripts/v7_train_args.sh              # -> V7_ARGS (single source of truth)

[ -f "$RESUME" ] && echo "Resuming from $RESUME (--reset-schedule)" \
                 || echo "No checkpoint at $RESUME — COLD START (set RESUME to the v7 latest.pt)"
echo "DDP tokenizer: NGPU=$NGPU  frames=$FRAMES_DIR  max_steps=$MAX_STEPS  workers/rank=$NUM_WORKERS"

exec torchrun --standalone --nproc_per_node="$NGPU" scripts/train_transformer_tokenizer.py "${V7_ARGS[@]}"
