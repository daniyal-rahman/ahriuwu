#!/bin/bash
# Four short BC runs to decide whether the bronze retrain is worth a full epoch.
#
# Judged on move_event_ce (the movement CATEGORICAL CE on click frames, gate
# excluded) against a blind p(next bin | prev bin) table fitted on the SAME split
# -- bc_movement is the gated loss and fell 0.871 -> 0.727 while the head learned
# nothing from pixels, so it cannot answer this.
#
# Design. --movement-interp is REJECTED with --movement-action-mode held (the
# interpolated target depends on the NEXT click, so feeding it as action
# conditioning leaks the future), which is why there is no labels-only-with-held
# arm:
#   R0 control   baseline labels + held        (what we ship today)
#   R1 channel   baseline labels + none        (input fix alone)
#   R2 both-mild fixed labels    + event_only  (labels + sparse action)
#   R3 both-full fixed labels    + none        (labels + no action at all)
#
# Sequential: one GPU. Same seed, same steps, same everything else.
#
# Launch:  ssh desktop 'nohup /mnt/nfs/projects/ahriuwu/ops/bronze_ab.sh >/dev/null 2>&1 &'
# Stop:    touch /mnt/nfs/projects/ahriuwu/ops/bronze_ab.stop
set -uo pipefail
REPO=/mnt/nfs/projects/ahriuwu
cd "$REPO" || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY=/home/dani/miniconda3/envs/ml/bin/python
LOG=$REPO/ops/bronze_ab.log
STEPS=${STEPS:-5000}
rm -f "$REPO/ops/bronze_ab.stop"
echo "$(date -u '+%m-%d %H:%M UTC') bronze_ab: start, STEPS=$STEPS" >> "$LOG"

COMMON="--unfreeze-backbone --dynamics-checkpoint rollout_stage/desktop_resume_8775_stripped.pt \
--model-size medium --num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0 \
--latents-dir /mnt/nfs/datasets/replay_latents_v7_bc \
--labels-root /mnt/nfs/datasets/lol_replays_16_9_772 \
--seq-len 16 --stride 8 --batch-size 1 --grad-accum 16 --lr 1e-4 --no-wandb \
--warmup-steps 200 --num-workers 4 --epochs 1 --seed 1234 \
--movement-mode axis --movement-gate --movement-source clicks"

run () {  # name  cache  prefirst  interp_flag  action_mode
  local name=$1 cache=$2 pf=$3 mi=$4 am=$5
  [ -f "$REPO/ops/bronze_ab.stop" ] && { echo "stopped before $name" >> "$LOG"; exit 0; }
  local out=$REPO/data/bronze_ab_$name
  echo "$(date -u '+%m-%d %H:%M UTC')   $name: start (prefirst=$pf interp=$mi action=$am)" >> "$LOG"
  # shellcheck disable=SC2086
  PYTHONPATH=src $PY scripts/train_agent_finetune.py $COMMON \
      --dataset-cache "$cache" --checkpoint-dir "$out" \
      --prefirst-mode "$pf" $mi --movement-action-mode "$am" \
      --max-steps "$STEPS" >> "$REPO/ops/bronze_ab_$name.log" 2>&1
  echo "$(date -u '+%m-%d %H:%M UTC')   $name: rc=$?" >> "$LOG"
  grep -E "move_event_ce" "$REPO/ops/bronze_ab_$name.log" | tail -2 >> "$LOG"
}

CB=$REPO/data/cache_baseline.pt
CF=$REPO/data/cache_fixed.pt
run R0_control   "$CB" sentinel ""                  held
run R1_channel   "$CB" sentinel ""                  none
run R2_bothmild  "$CF" heading  "--movement-interp" event_only
run R3_bothfull  "$CF" heading  "--movement-interp" none
echo "$(date -u '+%m-%d %H:%M UTC') bronze_ab: all done" >> "$LOG"
