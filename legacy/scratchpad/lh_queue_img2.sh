#!/bin/bash
# Image probes, round 2: best-inner-val model selection + the gold anchoring.
cd /srv/nfs/projects/ahriuwu || exit 1
P=/home/dani/miniconda3/envs/ml/bin/python
export PYTHONPATH=src PYTHONUNBUFFERED=1
set -x
$P scratchpad/lh_run.py --probe Acrop --epochs 30 --shuffle-epochs 12 --bs 48 \
    >> scratchpad/lh_img2.log 2>&1
$P scratchpad/lh_run.py --probe Acrop --anchor gold --frames scratchpad/lh_frames_gold \
    --epochs 30 --shuffle-epochs 12 --bs 48 >> scratchpad/lh_img2.log 2>&1
$P scratchpad/lh_run.py --probe Acrop --target level --epochs 20 --bs 48 \
    --no-shuffle-control >> scratchpad/lh_img2.log 2>&1
echo IMG2_QUEUE_DONE >> scratchpad/lh_img2.log
