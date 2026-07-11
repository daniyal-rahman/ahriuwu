#!/bin/bash
# Launch dynamics training across N GPUs on ONE box via DDP (torchrun).
# Ports scripts/run_ddp_tok.sh to the dynamics trainer: identical consumer-GPU
# NCCL env (no P2P/IB; SHM host-specific). The trainer splits grad-accum across
# ranks so the effective batch stays as configured in dyn_train_args.sh.
#
# Consumer Blackwell/Ada cards have NO working GPU P2P -> NCCL host-stages via
# host RAM. Init on a no-NVLink box can take MINUTES; the trainer uses a 20-min
# init timeout — never gate readiness on a short external clock (docs/VAST.md §4).
#
# Override via env: NGPU, LATENTS_DIR, CHECKPOINT_DIR, LABELS_ROOT, RESUME,
#                   NUM_WORKERS, NCCL_SHM_DISABLE, COMPILE, WANDB_MODE
set -euo pipefail
cd "$(dirname "$0")/.."                      # repo root

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --- NCCL for consumer (GeForce, no NVLink/P2P, no IB) single-node multi-GPU ---
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"   # HOST-SPECIFIC: flip to 1 per-box if init IMAs
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_MODE="${WANDB_MODE:-online}"

NGPU="${NGPU:-4}"
# run-specific defaults for the DDP box (read by dyn_train_args.sh)
LATENTS_DIR="${LATENTS_DIR:-/workspace/latents/dynamics_replay_v7_dim32}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints/dynamics_v7_ddp}"
LABELS_ROOT="${LABELS_ROOT:-/workspace/labels/lol_replays_16_9_772}"
RESUME="${RESUME:-$CHECKPOINT_DIR/dynamics_latest.pt}"
NUM_WORKERS="${NUM_WORKERS:-6}"              # PER RANK
export LATENTS_DIR CHECKPOINT_DIR LABELS_ROOT RESUME NUM_WORKERS

# Box deps beyond the base pytorch image (verified on the 2026-07-06 Vast smoke):
# dataset.py imports cv2 unconditionally; utils/logging touches wandb. Idempotent.
python -c "import cv2"   2>/dev/null || pip -q install opencv-python-headless
python -c "import wandb" 2>/dev/null || pip -q install wandb

source scripts/dyn_train_args.sh             # -> DYN_ARGS (single source of truth)

[ -f "$RESUME" ] && echo "Resuming from $RESUME" \
                 || echo "No checkpoint at $RESUME — COLD START"
echo "DDP dynamics: NGPU=$NGPU  latents=$LATENTS_DIR  labels=$LABELS_ROOT  workers/rank=$NUM_WORKERS"

exec torchrun --standalone --nproc_per_node="$NGPU" scripts/train_dynamics.py "${DYN_ARGS[@]}"
