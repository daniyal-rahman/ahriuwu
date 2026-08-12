#!/bin/bash
# Gate-BC validation run on the login 1060 (sticky-categorical movement +
# action-dropout + aux state head + calibrated BCE, act8775 backbone).
# Relaunched 2026-08-06 on schema-3 dead-banded movement targets.
set -uo pipefail
cd /srv/nfs/projects/ahriuwu
export PYTHONPATH=/srv/nfs/projects/ahriuwu/src
export CUDA_VISIBLE_DEVICES=0
exec /home/dani/miniconda3/envs/ml/bin/python -u scripts/train_agent_finetune.py \
  --dynamics-checkpoint rollout_stage/desktop_resume_8775_stripped.pt \
  --model-size medium --num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0 \
  --latents-dir /srv/nfs/datasets/replay_latents_v7_bc \
  --labels-root /srv/nfs/datasets/lol_replays_16_9_772 \
  --seq-len 16 --stride 8 --batch-size 4 --epochs 10 \
  --lr 3e-4 --warmup-steps 2000 --num-workers 4 \
  --checkpoint-dir data/phase2_bc_gate1060 --log-interval 50 \
  --resume auto --checkpoint-minutes 20 \
  --ability-pos-weight 1.0 --aux-state-weight 0.5 \
  --movement-gate --action-dropout 0.15 \
  --dataset-cache data/phase2_bc_gate1060/dataset_cache.pt \
  >> scratchpad/bc_gate_1060.log 2>&1
