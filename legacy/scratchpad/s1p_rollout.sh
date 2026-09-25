#!/bin/bash
set -uo pipefail
cd /mnt/nfs/projects/ahriuwu
PY=/home/dani/miniconda3/envs/ml/bin/python
CK=/mnt/storage/data/ahriuwu/checkpoints/dyn179_s1prime_latentonly/dynamics_latest.pt
C="--checkpoint $CK --model-size medium --no-actions --device cuda --decode --ctx 6 --horizon 16 --match NA1_5549995114 --seed 0 --latents-dir /scratch/ahriuwu/dynamics_replay_latents_v7_dim32 --labels-root /mnt/nfs/datasets/lol_replays_16_9_772 --out-mp4 /tmp/s.mp4 --out-png /tmp/s.png"
for N in 1 4; do
  echo "===== S1prime gs7550 num_steps=$N ====="
  PYTHONPATH=/mnt/nfs/projects/ahriuwu/src $PY -u scripts/rollout_check.py $C --num-steps $N --out-plot /tmp/s1p_$N.png 2>&1 | grep -aE "checkpoint:|TEACHER-FORCED|ROLLOUT|plot:|ARCH|Error"
done
echo S1P_ROLLOUT_DONE
