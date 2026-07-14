#!/bin/bash
# Phase-2 BC (behavior cloning + reward) on the login-node GTX 1060.
# Frozen 179 world-model backbone + TRAINABLE agent blocks + heads (see the
# agent-block-freeze fix in train_agent_finetune.py). Data: the 125 action-labeled
# replays staged to NFS. Backgrounded + logged so it survives the ssh session.
set -uo pipefail
cd /srv/nfs/projects/ahriuwu
export PYTHONPATH=/srv/nfs/projects/ahriuwu/src
export CUDA_VISIBLE_DEVICES=0            # the 1060 (login node)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BACKBONE=${BACKBONE:-scratchpad/dyn_s1prime_gs7550_backbone.pt}   # RECOVERED encoder (Step 1', gs7603, tau0.9 ~26)
LAT=${LAT:-/srv/nfs/datasets/replay_latents_v7_bc}
LABELS=${LABELS:-/srv/nfs/datasets/lol_replays_16_9_772}
# NB: repo 'checkpoints' is a symlink to desktop-local /mnt/storage (dangling on
# the login node), so BC (1060, login) writes to a real NFS path instead.
CKPT=${CKPT:-/srv/nfs/projects/ahriuwu/data/phase2_bc_garen}
LOG=${LOG:-/srv/nfs/projects/ahriuwu/scratchpad/bc_1060.log}
mkdir -p "$CKPT"

exec /home/dani/miniconda3/envs/ml/bin/python -u scripts/train_agent_finetune.py \
  --dynamics-checkpoint "$BACKBONE" \
  --model-size medium --num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0 \
  --latents-dir "$LAT" --labels-root "$LABELS" \
  --seq-len 32 --stride 8 --batch-size 6 --epochs 10 \
  --lr 3e-4 --warmup-steps 2000 --num-workers 4 \
  --checkpoint-dir "$CKPT" --log-interval 50 \
  --resume auto --checkpoint-minutes 20 --ability-pos-weight 5.0 \
  --dataset-cache "$CKPT/dataset_cache.pt" >> "$LOG" 2>&1
