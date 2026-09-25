#!/usr/bin/env bash
# Evaluate a league checkpoint against the fixed bot at the SAME update count
# the mirror run was evaluated at, so the two numbers are comparable.
#
# The mirror baseline (rl-0913d @ update 950, n=8): CS 46.6 +/- 7.4, bot 42.9.
# Self-play CS cannot answer "is the league better" -- it is opponent-relative
# and the two runs face different opponents by construction. Only a FIXED
# opponent can.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/dani/miniconda3/envs/ml/bin/python
RUN=${RUN:-rl-league-0913c}
WANT=${WANT:-950}
CK=$(printf "/scratch/lanerl/runs/%s/checkpoints/update_%08d.pt" "$RUN" "$WANT")

for i in $(seq 1 240); do
  ssh -o ConnectTimeout=10 -o BatchMode=yes desktop "test -f $CK" 2>/dev/null && break
  sleep 30
done
ssh -o ConnectTimeout=10 desktop "test -f $CK" 2>/dev/null || { echo "checkpoint $CK never appeared"; exit 1; }

mkdir -p ckpt_eval
scp -o ConnectTimeout=15 desktop:"$CK" "ckpt_eval/${RUN}_u${WANT}.pt" || exit 1
echo "evaluating ${RUN} @ update ${WANT} against the fixed bot (n=8)"
$PY -m lanerl_train.eval_vs_bot --checkpoint "ckpt_eval/${RUN}_u${WANT}.pt" \
    --episodes 8 --out "lanerl/logs/eval_${RUN}_u${WANT}.json"
