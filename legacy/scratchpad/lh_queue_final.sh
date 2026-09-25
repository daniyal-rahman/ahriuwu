#!/bin/bash
# Last two runs: the gold-anchored latent probe (does the second anchoring agree?)
# and a clean champion-level positive control on the latents (do v7 latents carry
# ANY generalising signal on games the tokenizer never saw?).
cd /srv/nfs/projects/ahriuwu || exit 1
while pgrep -f "lh_run.py --probe D" >/dev/null; do sleep 30; done
P=/home/dani/miniconda3/envs/ml/bin/python
export PYTHONPATH=src PYTHONUNBUFFERED=1
$P scratchpad/lh_run.py --probe B --target level --epochs 20 --bs 64 \
    --model gridcnn --no-shuffle-control >> scratchpad/lh_final.log 2>&1
$P scratchpad/lh_run.py --probe B --anchor gold --cap-per-game 130 --epochs 20 \
    --shuffle-epochs 10 --bs 128 --model gridcnn >> scratchpad/lh_final.log 2>&1
$P scratchpad/lh_run.py --probe C --anchor gold --cap-per-game 130 --window 16 \
    --epochs 20 --shuffle-epochs 10 --bs 64 --model gridcnn >> scratchpad/lh_final.log 2>&1
echo FINAL_QUEUE_DONE >> scratchpad/lh_final.log
