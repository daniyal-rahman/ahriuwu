#!/bin/bash
# Run a dynamics rollout / dream check on danilogin's GPU (the login GTX 1060),
# off the training GPU. Stages the latest job-124 checkpoint (+ first-run the
# tokenizer and one match's latents) from the desktop-local disks to NFS via a
# brief overlap on the running job, then runs rollout_check.py on the login GPU.
#
# Pass-through args to rollout_check.py, e.g.:
#     bash scripts/rollout_login.sh --decode --horizon 20
#
# Assumes job 124 is running (the stage copies desktop-local files via --overlap).
set -euo pipefail
STAGE=/srv/nfs/projects/ahriuwu/rollout_stage
JOB=${ROLLOUT_JOBID:-124}
mkdir -p "$STAGE"

echo "[stage] latest checkpoint (+ first-run tokenizer/latents) desktop -> NFS via job $JOB ..."
srun --jobid="$JOB" --overlap --time=8:00 bash -c '
  S=/mnt/nfs/projects/ahriuwu/rollout_stage; mkdir -p "$S"
  cp -f /mnt/storage/data/ahriuwu/checkpoints/dynamics_v7_replay/dynamics_latest.pt "$S"/
  [ -f "$S/transformer_tokenizer_latest.pt" ] || cp /mnt/storage/data/ahriuwu-checkpoints/tokenizer_v7/transformer_tokenizer_latest.pt "$S"/
  [ -f "$S/NA1_5549995114.pt" ] || cp /scratch/ahriuwu/dynamics_replay_latents_v7_dim32/NA1_5549995114.pt "$S"/'

echo "[run] rollout_check on the login GPU (cuda) ..."
PYTHONPATH=/srv/nfs/projects/ahriuwu/src /home/dani/miniconda3/envs/ml/bin/python \
  /srv/nfs/projects/ahriuwu/scripts/rollout_check.py \
  --checkpoint "$STAGE/dynamics_latest.pt" \
  --latents-dir "$STAGE" \
  --tokenizer "$STAGE/transformer_tokenizer_latest.pt" \
  --labels-root /srv/nfs/datasets/lol_replays_16_9_772 \
  --device cuda "$@"
