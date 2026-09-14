#!/usr/bin/env bash
# Continuously evaluate the newest checkpoint against the FIXED scripted bot.
#
# Why this is a daemon on the second node and not a bigger --eval-every.
#
# Self-play CS@10 cannot measure progress. It is opponent-relative, and in
# self-play the opponent improves at exactly your rate, so the number goes flat
# while absolute skill rises. Measured on run rl-0913d: self-play sat at 37.5
# while the SAME checkpoint scored 46.6 against the fixed bot. Under a league
# it is worse than flat -- the opponent distribution itself changes as the pool
# fills, so the series is not even comparable with itself over time.
#
# So the only honest progress curve is against a frozen opponent. Getting it
# from --eval-every costs training throughput on the GPU node and, at
# eval_every=400 with anchor_episodes=2, produced two games per 400 updates:
# far too sparse and far too noisy to steer on (n=8 on this metric has sd 7.4).
#
# This runs on the idle node instead, so the curve is dense and costs the
# training run nothing. Each row appends to one JSONL keyed by update, which is
# the series to plot against updates -- not the self-play number.
#
#   RUN=rl-league-0913c EPISODES=6 bash lanerl/eval_daemon.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/home/dani/miniconda3/envs/ml/bin/python
RUN=${RUN:-rl-league-0913c}
EPISODES=${EPISODES:-6}
EVERY=${EVERY:-200}          # evaluate a checkpoint every this many updates
OUT=${OUT:-lanerl/logs/eval_curve_${RUN}.jsonl}
REMOTE=${REMOTE:-desktop}
CKDIR=/scratch/lanerl/runs/${RUN}/checkpoints

mkdir -p ckpt_eval
: > "${OUT}.lock" 2>/dev/null || true
echo "eval daemon: ${RUN}, every ${EVERY} updates, n=${EPISODES} per point -> ${OUT}"

done_updates=""
while true; do
  # Newest checkpoint on the training node, as an update number.
  latest=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE" \
      "ls ${CKDIR}/update_*.pt 2>/dev/null | tail -1" 2>/dev/null || true)
  if [ -z "$latest" ]; then
    # The run may not have started, or may have finished. Either way this
    # daemon should keep waiting rather than exit -- a dead evaluator looks
    # exactly like a flat curve.
    sleep 60; continue
  fi
  u=$(basename "$latest" .pt); u=${u#update_}; u=$((10#$u))
  bucket=$(( u / EVERY * EVERY ))
  case " $done_updates " in *" $bucket "*) sleep 60; continue;; esac
  [ "$bucket" -eq 0 ] && { sleep 60; continue; }

  ck=$(printf "${CKDIR}/update_%08d.pt" "$bucket")
  ssh -o ConnectTimeout=10 "$REMOTE" "test -f $ck" 2>/dev/null || { sleep 60; continue; }

  local_ck="ckpt_eval/${RUN}_u${bucket}.pt"
  scp -o ConnectTimeout=20 "$REMOTE":"$ck" "$local_ck" >/dev/null 2>&1 || { sleep 60; continue; }

  echo "--- evaluating ${RUN} @ update ${bucket} (n=${EPISODES})"
  res=$($PY -m lanerl_train.eval_vs_bot --checkpoint "$local_ck" \
          --episodes "$EPISODES" --out "lanerl/logs/eval_${RUN}_u${bucket}.json" 2>&1 | tail -3)
  echo "$res"
  # One line per point. The summary line looks like:
  #   <name>: n=8  CS=46.6 vs bot 42.9  gold=... lvl=... hp_lost=... dist=...
  cs=$(echo "$res" | grep -oE "CS=[0-9.]+" | head -1 | cut -d= -f2)
  bot=$(echo "$res" | grep -oE "vs bot [0-9.]+" | head -1 | awk '{print $3}')
  if [ -n "${cs:-}" ]; then
    echo "{\"update\": $bucket, \"cs_at_10\": $cs, \"bot_cs_at_10\": ${bot:-null}, \"n\": $EPISODES, \"run\": \"$RUN\"}" >> "$OUT"
    echo "curve: update=$bucket cs=$cs bot=${bot:-?}"
  else
    echo "WARNING: no CS parsed at update $bucket -- eval_vs_bot output changed or the run failed"
  fi
  rm -f "$local_ck"
  done_updates="$done_updates $bucket"
done
