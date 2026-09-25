#!/bin/bash
# Gold-anchored latent probes (D-gold dropped: the frozen-dynamics forward runs at
# ~1.5 rows/s on the 1060 and the commit anchoring already answers the D question).
cd /srv/nfs/projects/ahriuwu || exit 1
while pgrep -f "lh_run.py --probe D" >/dev/null; do sleep 60; done
P=/home/dani/miniconda3/envs/ml/bin/python
export PYTHONPATH=src PYTHONUNBUFFERED=1
set -x
$P scratchpad/lh_run.py --probe B --anchor gold --cap-per-game 130 --epochs 20 \
    --shuffle-epochs 10 --bs 128 --model gridcnn >> scratchpad/lh_lat2.log 2>&1
$P scratchpad/lh_run.py --probe C --anchor gold --cap-per-game 130 --window 16 \
    --epochs 20 --shuffle-epochs 10 --bs 64 --model gridcnn >> scratchpad/lh_lat2.log 2>&1
echo LAT2_QUEUE_DONE >> scratchpad/lh_lat2.log
