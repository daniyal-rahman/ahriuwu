#!/bin/bash
# Latent / agent-token probes. Model selection is on the inner-val GAMES (never on
# held-out), so these re-runs use log_every=1 and best-val restore.
cd /srv/nfs/projects/ahriuwu || exit 1
P=/home/dani/miniconda3/envs/ml/bin/python
export PYTHONPATH=src PYTHONUNBUFFERED=1
set -x
$P scratchpad/lh_run.py --probe B --epochs 20 --shuffle-epochs 10 --bs 128 \
    --model gridcnn >> scratchpad/lh_lat.log 2>&1
$P scratchpad/lh_run.py --probe B --epochs 20 --shuffle-epochs 0 --bs 128 \
    --model linear --no-shuffle-control >> scratchpad/lh_lat.log 2>&1
$P scratchpad/lh_run.py --probe C --window 16 --epochs 20 --shuffle-epochs 10 \
    --bs 64 --model gridcnn >> scratchpad/lh_lat.log 2>&1
$P scratchpad/lh_run.py --probe D --window 16 --epochs 40 --shuffle-epochs 20 \
    --bs 256 --model mlp --agent-cap 70 >> scratchpad/lh_lat.log 2>&1
$P scratchpad/lh_run.py --probe D --window 16 --epochs 40 --shuffle-epochs 0 \
    --bs 256 --model linear --no-shuffle-control >> scratchpad/lh_lat.log 2>&1
echo LAT_QUEUE_DONE >> scratchpad/lh_lat.log
