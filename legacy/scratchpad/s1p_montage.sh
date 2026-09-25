#!/bin/bash
set -uo pipefail
cd /mnt/nfs/projects/ahriuwu
PY=/home/dani/miniconda3/envs/ml/bin/python
CK=/mnt/storage/data/ahriuwu/checkpoints/dyn179_s1prime_latentonly/dynamics_latest.pt
PYTHONPATH=/mnt/nfs/projects/ahriuwu/src $PY -u scripts/rollout_check.py --checkpoint "$CK" \
  --model-size medium --no-actions --device cuda --decode --ctx 6 --horizon 16 \
  --match NA1_5549995114 --seed 0 --num-steps 1 \
  --latents-dir /scratch/ahriuwu/dynamics_replay_latents_v7_dim32 --labels-root /mnt/nfs/datasets/lol_replays_16_9_772 \
  --out-plot /mnt/nfs/projects/ahriuwu/s1p_psnr.png \
  --out-png /mnt/nfs/projects/ahriuwu/s1p_montage.png \
  --out-mp4 /tmp/s1p.mp4 2>&1 | grep -aE "plot:|ROLLOUT"
echo MONTAGE_DONE
