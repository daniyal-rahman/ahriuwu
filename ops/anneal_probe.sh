#!/bin/bash
# HYPOTHESIS TEST: is the world model's blur/plateau caused by a never-annealed LR?
#
# The dynamics trained with --lr-schedule wsd and DECAY_STEPS=0. In
# src/ahriuwu/utils/training.py:294 the decay branch is
#   if decay_steps > 0 and step >= total_steps - decay_steps:
# so with 0 it NEVER fires: warmup 3000 then a flat 3e-4 forever. The checkpoint
# stopped at global_step 8775, i.e. 5,775 steps at constant high LR, never annealed.
#
# That single fact is consistent with four separate measurements:
#   - "plateaued" despite being under one epoch
#   - mean-seeking / blurry predictions (sharpness 26 vs 43 for real frames)
#   - more denoising steps RAISE MSE (off-manifold blur, not honest stochasticity)
#   - a single 1-step prediction (NMSE 0.0275) LOSES to copying the previous
#     frame (0.0224) -- a blurry predictor loses to a sharp stale one
#
# If the plateau is the schedule, this is a cheap fix to something the dream study
# concluded was structural, and Phase 3 may not be as dead as it looks.
#
# TEST: resume from 8775, run a short REAL decay phase, re-measure 1-step NMSE
# against the copy baseline. Falsifies in ~1-2h.
set -uo pipefail
REPO=/mnt/nfs/projects/ahriuwu
cd "$REPO" || exit 1
LOG=$REPO/ops/anneal_probe.log
export PYTORCH_ALLOC_CONF=expandable_segments:True
# wait for the node
until timeout 10 nvidia-smi >/dev/null 2>&1; do sleep 120; done
idle=0
while [ "$idle" -lt 3 ]; do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && idle=$((idle+1)) || idle=0
  sleep 60
done
echo "$(date -u '+%m-%d %H:%M UTC') anneal_probe: BASELINE (pre-anneal)" >> "$LOG"
PYTHONPATH=src /home/dani/miniconda3/envs/ml/bin/python -u scripts/eval_dream_fidelity.py \
  --ckpt rollout_stage/desktop_resume_8775_stripped.pt --tag pre_anneal \
  >> "$REPO/ops/anneal_pre.log" 2>&1 || echo "  baseline eval rc=$?" >> "$LOG"
echo "$(date -u '+%m-%d %H:%M UTC') anneal_probe: annealing 800 steps" >> "$LOG"
export LATENTS_DIR=/mnt/nfs/datasets/replay_latents_v7_bc
export LABELS_ROOT=/mnt/nfs/datasets/lol_replays_16_9_772
export CHECKPOINT_DIR=$REPO/data/dyn_annealed
export WARMUP_STEPS=0 EPOCHS=1
source $REPO/scripts/dyn_train_args.sh
PYTHONPATH=src /home/dani/miniconda3/envs/ml/bin/python -u scripts/train_dynamics.py \
  "${DYN_ARGS[@]}" --resume rollout_stage/desktop_resume_8775_stripped.pt \
  --decay-steps 800 --lr 3e-4 \
  >> "$REPO/ops/anneal_train.log" 2>&1
echo "$(date -u '+%m-%d %H:%M UTC') anneal_probe: train rc=$?" >> "$LOG"
echo "$(date -u '+%m-%d %H:%M UTC') anneal_probe: POST eval" >> "$LOG"
PYTHONPATH=src /home/dani/miniconda3/envs/ml/bin/python -u scripts/eval_dream_fidelity.py \
  --ckpt $CHECKPOINT_DIR/latest.pt --tag post_anneal \
  >> "$REPO/ops/anneal_post.log" 2>&1 || echo "  post eval rc=$?" >> "$LOG"
echo "$(date -u '+%m-%d %H:%M UTC') anneal_probe: done" >> "$LOG"
