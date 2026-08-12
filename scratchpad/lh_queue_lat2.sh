#!/bin/bash
# Gold-anchored latent probes. Waits for the commit-anchored latent queue so the
# 6 GB GTX 1060 only ever has one training job on it.
cd /srv/nfs/projects/ahriuwu || exit 1
while ! grep -q LAT_QUEUE_DONE scratchpad/lh_lat.log 2>/dev/null; do sleep 60; done
P=/home/dani/miniconda3/envs/ml/bin/python
export PYTHONPATH=src PYTHONUNBUFFERED=1
set -x
$P scratchpad/lh_run.py --probe B --anchor gold --cap-per-game 130 --epochs 20 \
    --shuffle-epochs 10 --bs 128 --model gridcnn >> scratchpad/lh_lat2.log 2>&1
$P scratchpad/lh_run.py --probe C --anchor gold --cap-per-game 130 --window 16 \
    --epochs 20 --shuffle-epochs 10 --bs 64 --model gridcnn >> scratchpad/lh_lat2.log 2>&1
$P scratchpad/lh_run.py --probe D --anchor gold --cap-per-game 130 --window 16 \
    --epochs 40 --shuffle-epochs 20 --bs 256 --model mlp --agent-cap 60 \
    >> scratchpad/lh_lat2.log 2>&1
echo LAT2_QUEUE_DONE >> scratchpad/lh_lat2.log
