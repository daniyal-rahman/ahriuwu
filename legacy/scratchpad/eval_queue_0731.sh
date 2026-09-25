#!/bin/bash
# Sequential GPU eval queue on the 1060 while BC is paused (2026-07-31).
# 1) encode held-out laning slice  2) reward-head evals (in-train + held-out)
# 3) BC sim eval on the held-out match  4) Phase-3 imagination real-data smoke.
set -uo pipefail
cd /srv/nfs/projects/ahriuwu
PY=/home/dani/miniconda3/envs/ml/bin/python

echo "=== [1/4] encode held-out slice ($(date '+%H:%M')) ==="
PYTHONPATH=src:scripts $PY scratchpad/encode_heldout_slice.py

echo "=== [2/4] reward-head eval: IN-TRAINING matches ($(date '+%H:%M')) ==="
PYTHONPATH=src $PY scripts/eval_reward_head.py --matches NA1_5549995114 NA1_5550013959
echo "--- reward-head eval: HELD-OUT match ---"
PYTHONPATH=src $PY scripts/eval_reward_head.py --matches NA1_5549981347 \
  --latents-dir /srv/nfs/datasets/replay_latents_v7_heldout

echo "=== [3/4] BC sim eval on HELD-OUT match ($(date '+%H:%M')) ==="
PYTHONPATH=src $PY scripts/eval_bc_sim.py \
  --phase2-ckpt data/phase2_bc_garen/agent_finetune_latest.pt \
  --match NA1_5549981347 --latents-dir /srv/nfs/datasets/replay_latents_v7_heldout \
  --frames 800 --ability-thresh -4.0

echo "=== [4/4] imagination smoke, real data ($(date '+%H:%M')) ==="
mkdir -p scratchpad/imag_smoke scratchpad/imag_latents
ln -sf /srv/nfs/datasets/replay_latents_v7_bc/NA1_5549995114.pt scratchpad/imag_latents/
PYTHONPATH=src $PY scripts/train_imagination.py \
  --agent-checkpoint data/phase2_bc_garen/agent_finetune_latest.pt \
  --latents-dir scratchpad/imag_latents \
  --labels-root /srv/nfs/datasets/lol_replays_16_9_772 \
  --model-size medium --num-kv-heads 4 \
  --batch-size 2 --stride 512 --horizon 4 --gen-steps 8 \
  --epochs 1 --checkpoint-dir scratchpad/imag_smoke --log-interval 1

echo "=== QUEUE DONE ($(date '+%H:%M')) ==="
