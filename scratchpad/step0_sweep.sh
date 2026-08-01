#!/bin/bash
set -uo pipefail
export PATH=/opt/conda/bin:$PATH
cd /mnt/nfs/projects/ahriuwu
PY=/home/dani/miniconda3/envs/ml/bin/python
LAT=/scratch/ahriuwu/dynamics_replay_latents_v7_dim32
LABELS=/mnt/nfs/datasets/lol_replays_16_9_772
CK135=/mnt/storage/data/ahriuwu/checkpoints/dynamics_v7_accel_resume/dynamics_latest.pt
CK179=/mnt/storage/data/ahriuwu/checkpoints/dynamics_v7_yt578_hudfix/dynamics_latest.pt
COMMON="--model-size medium --device cuda --ctx 6 --horizon 16 --match NA1_5549995114 --seed 0 --latents-dir $LAT --labels-root $LABELS"
run(){ # $1=label $2=ckpt $3=extra
  echo "===== $1 ====="
  PYTHONPATH=/mnt/nfs/projects/ahriuwu/src $PY -u scripts/rollout_check.py --checkpoint "$2" $COMMON $3 2>&1 \
    | grep -aE "checkpoint:|TEACHER-FORCED|ROLLOUT|ARCH MISMATCH|Error" | sed "s/^/[$1] /"
}
for N in 1 4 16; do
  run "135_act_N$N"   "$CK135" "--num-steps $N"
  run "179_noact_N$N" "$CK179" "--num-steps $N --no-actions"
done
# control: 135 WITHOUT actions at N=1 (isolate model-quality from action-info-at-eval)
run "135_noact_N1" "$CK135" "--num-steps 1 --no-actions"
echo "SWEEP_DONE"
