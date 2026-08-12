#!/bin/bash
# Re-run ONLY the B/C shuffle controls under the corrected null (train labels AND
# the inner-val labels used for epoch selection are both permuted). The original
# B/C controls were run before that fix and sit at 0.52-0.55 purely because epoch
# selection was reading real val labels.
cd /srv/nfs/projects/ahriuwu || exit 1
while pgrep -f "lh_run.py --probe [BCD]" >/dev/null; do sleep 30; done
P=/home/dani/miniconda3/envs/ml/bin/python
export PYTHONPATH=src PYTHONUNBUFFERED=1
$P scratchpad/lh_run.py --probe B --epochs 2 --shuffle-epochs 20 --bs 128 \
    --model gridcnn --out scratchpad/lh_res_Bshufnull.json >> scratchpad/lh_shufnull.log 2>&1
$P scratchpad/lh_run.py --probe C --window 16 --epochs 2 --shuffle-epochs 20 --bs 64 \
    --model gridcnn --out scratchpad/lh_res_Cshufnull.json >> scratchpad/lh_shufnull.log 2>&1
echo SHUFNULL_DONE >> scratchpad/lh_shufnull.log
