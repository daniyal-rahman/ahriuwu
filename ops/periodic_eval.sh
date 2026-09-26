#!/bin/bash
# Every time a training run advances >= STEP updates past the last evaluated
# checkpoint, copy its ckpt_latest to eval_u<N>.msgpack and run EVAL_frozen.
#   ops/periodic_eval.sh <run root under lanerl_jax/runs> [STEP=300]
cd /srv/nfs/projects/ahriuwu-lanerl-jax
ROOT=$1; STEP=${2:-300}; LAST=${LAST:-760}
while true; do
  RUN=$(ls -d $ROOT/*-s[0-9]*/ | tail -1)
  U=$(python3 -c "import json;m=json.load(open('$RUN/manifest.json'));print(m['checkpoints'][-1]['update'] if m.get('checkpoints') else 0)" 2>/dev/null || echo 0)
  if [ "$U" -ge $((LAST + STEP)) ]; then
    cp $RUN/ckpt_latest.msgpack $RUN/eval_u$U.msgpack
    # Through the validating launcher (no env-var plumbing): OPPONENT / OPPONENT_CKPT are optional.
    python3 ops/launch.py eval --ckpt /mnt/nfs/projects/ahriuwu-lanerl-jax/$RUN/eval_u$U.msgpack \
      ${OPPONENT:+--opponent $OPPONENT} ${OPPONENT_CKPT:+--opponent-ckpt $OPPONENT_CKPT} > lanerl_jax/runs/EVAL/eval_u$U.launch 2>&1
    # the eval job writes its own log; wait for it to leave the queue
    sleep 20; while squeue -h -n EVAL | grep -q .; do sleep 30; done
    LOG=$(ls -t lanerl_jax/runs/EVAL/EVAL.out 2>/dev/null | head -1); cp "$LOG" lanerl_jax/runs/EVAL/eval_u$U.out 2>/dev/null; : > "$LOG"
    SUMMARY=$(grep -o '{"0": .*}' lanerl_jax/runs/EVAL/eval_u$U.out | tail -1)
    [ -z "$SUMMARY" ] && SUMMARY='null'
    echo "{\"run\": \"$ROOT\", \"update\": $U, \"checkpoint\": \"$RUN/eval_u$U.msgpack\", \"time\": \"$(date -u +%FT%TZ)\", \"summary\": $SUMMARY}" >> lanerl_jax/runs/EVAL/summary.jsonl
    LAST=$U
  fi
  sleep 300
done
