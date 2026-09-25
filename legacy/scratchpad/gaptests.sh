#!/bin/bash
cd /mnt/nfs/projects/ahriuwu
PY=/home/dani/miniconda3/envs/ml/bin/python
LAB=/mnt/nfs/datasets/lol_replays_16_9_772
echo "===== WORLD-MODEL ROLLOUT (135 action-conditioned, num_steps=1, decode) ====="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY scripts/rollout_check.py \
  --checkpoint /mnt/storage/data/ahriuwu/checkpoints/dynamics_v7_accel_resume/dynamics_latest.pt \
  --model-size medium --device cuda --decode --ctx 6 --horizon 32 --num-steps 1 \
  --match NA1_5549995114 --seed 0 --latents-dir /scratch/ahriuwu/dynamics_replay_latents_v7_dim32 \
  --labels-root $LAB --out-plot /mnt/nfs/projects/ahriuwu/wm_rollout_135.png --out-png /tmp/m.png --out-mp4 /tmp/m.mp4 2>&1 \
  | grep -aE "checkpoint:|TEACHER-FORCED|ROLLOUT|plot:"
echo "===== BC CONTEXT ABLATION (does more history help? ctx 8 / 16 / 32) ====="
for C in 8 16 32; do
  echo "--- ctx=$C ---"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY scripts/eval_bc_sim.py --latents-dir rollout_stage \
    --labels-root $LAB --frames 1500 --context $C --temperature 0.0 --ability-thresh -3.6 --device cuda 2>&1 \
    | grep -aE "MOVEMENT|SUMMARY"
done
echo GAPTESTS_DONE
