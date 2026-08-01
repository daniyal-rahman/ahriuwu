#!/bin/bash
cd /mnt/nfs/projects/ahriuwu
PY=/home/dani/miniconda3/envs/ml/bin/python
LAB=/mnt/nfs/datasets/lol_replays_16_9_772
echo "=== CALIBRATED EVAL (ability-thresh -4.0, 4000 frames) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY scripts/eval_bc_sim.py --latents-dir rollout_stage \
  --labels-root $LAB --frames 4000 --context 16 --temperature 0.0 --ability-thresh -4.0 --device cuda 2>&1 \
  | grep -aE "MOVEMENT|ABILITY|P=|SUMMARY|sim e2e"
echo "=== OVERLAY VIDEO (start 3000, 400 frames) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src $PY scripts/overlay_e2e.py --latents-dir rollout_stage \
  --labels-root $LAB --start 3000 --frames 400 --context 16 --fps 15 \
  --out /mnt/nfs/projects/ahriuwu/e2e_overlay.mp4 --device cuda 2>&1 | grep -aE "wrote"
echo FIXVID_DONE
