#!/bin/bash
set -uo pipefail
cd /mnt/nfs/projects/ahriuwu
PY=/home/dani/miniconda3/envs/ml/bin/python
C="--model-size medium --device cuda --decode --ctx 6 --horizon 16 --match NA1_5549995114 --seed 0 --num-steps 1 --latents-dir /scratch/ahriuwu/dynamics_replay_latents_v7_dim32 --labels-root /mnt/nfs/datasets/lol_replays_16_9_772 --out-mp4 /tmp/x.mp4 --out-png /tmp/x.png"
echo "===135@N1==="; PYTHONPATH=/mnt/nfs/projects/ahriuwu/src $PY -u scripts/rollout_check.py --checkpoint /mnt/storage/data/ahriuwu/checkpoints/dynamics_v7_accel_resume/dynamics_latest.pt $C --out-plot /tmp/p135.png 2>&1 | grep -aE "plot:|ROLLOUT"
echo "===179@N1==="; PYTHONPATH=/mnt/nfs/projects/ahriuwu/src $PY -u scripts/rollout_check.py --checkpoint /mnt/storage/data/ahriuwu/checkpoints/dynamics_v7_yt578_hudfix/dynamics_latest.pt $C --no-actions --out-plot /tmp/p179.png 2>&1 | grep -aE "plot:|ROLLOUT"
echo DECODE_DONE
