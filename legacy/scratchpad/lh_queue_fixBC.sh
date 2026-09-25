#!/bin/bash
# Regenerate the B/C commit-anchor score dumps. The shuffle-null re-run reused the
# same score-dump filename (the dump name is derived from probe/anchor/model, not
# from --out) and clobbered them with its 2-epoch stand-in model.
cd /srv/nfs/projects/ahriuwu || exit 1
P=/home/dani/miniconda3/envs/ml/bin/python
export PYTHONPATH=src PYTHONUNBUFFERED=1
$P scratchpad/lh_run.py --probe B --epochs 20 --bs 128 --model gridcnn \
    --no-shuffle-control >> scratchpad/lh_fixBC.log 2>&1
$P scratchpad/lh_run.py --probe C --window 16 --epochs 20 --bs 64 --model gridcnn \
    --no-shuffle-control >> scratchpad/lh_fixBC.log 2>&1
$P scratchpad/lh_summary.py > scratchpad/lh_final_table.txt 2>&1
echo FIXBC_DONE >> scratchpad/lh_fixBC.log
