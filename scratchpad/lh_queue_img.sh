#!/bin/bash
# Image probes (share the GTX 1060 with the sparse v7 encode; run strictly serially).
cd /srv/nfs/projects/ahriuwu || exit 1
P=/home/dani/miniconda3/envs/ml/bin/python
export PYTHONPATH=src PYTHONUNBUFFERED=1
set -x
$P scratchpad/lh_run.py --probe cheat --epochs 15 --shuffle-epochs 8  --bs 32 \
    >> scratchpad/lh_img.log 2>&1
$P scratchpad/lh_run.py --probe Acrop --epochs 25 --shuffle-epochs 12 --bs 48 \
    >> scratchpad/lh_img.log 2>&1
$P scratchpad/lh_run.py --probe A     --epochs 20 --shuffle-epochs 10 --bs 32 \
    >> scratchpad/lh_img.log 2>&1
echo IMG_QUEUE_DONE >> scratchpad/lh_img.log
