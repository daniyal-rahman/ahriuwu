#!/bin/bash
# Periodic frozen evaluation for a JAX run: every STEP updates, copy ckpt_latest
# and play 16 mirror games (seeded start jitter) on the desktop GPU (cpu partition,
# GPU visible). Rows -> lanerl_jax/runs/EVAL/jax_summary.jsonl.
#   ops/jax_periodic_eval.sh <run root under lanerl_jax/runs> [STEP=500]
cd /srv/nfs/projects/ahriuwu-lanerl-jax
ROOT=$1; STEP=${2:-500}; LAST=${LAST:-0}; ID=$(basename $(dirname $ROOT))
while true; do
  RUN=$(ls -d $ROOT/jax-farm-*/ 2>/dev/null | tail -1)
  U=$(python3 -c "import json;m=json.load(open('$RUN/manifest.json'));print(m['checkpoints'][-1]['update'] if m.get('checkpoints') else 0)" 2>/dev/null || echo 0)
  if [ "$U" -ge $((LAST + STEP)) ]; then
    for try in 1 2 3; do cp $RUN/ckpt_latest.msgpack $RUN/eval_u$U.msgpack; sleep 3; cmp -s $RUN/ckpt_latest.msgpack $RUN/eval_u$U.msgpack && break; sleep 10; done
    OUT=lanerl_jax/runs/EVAL/jax_${ID}_u$U
    srun -p cpu -w desktop --cpus-per-task=2 --mem=6G --time=40 --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax --job-name=JEVAL_${ID}_u$U bash -c "env -u XLA_FLAGS XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 ./.venv-gpu/bin/python -m lanerl_jax.train.jax_train --envs 16 --opponent mirror --episode-s 600 --start-near-wave --step-ticks 6 --eval-episodes 1 --seed 7 --start-jitter-s 20 --resume /mnt/nfs/projects/ahriuwu-lanerl-jax/$RUN/eval_u$U.msgpack --out $OUT" > $OUT.out 2>&1
    SUMMARY=$(grep -o '{"0": .*}' $OUT.out | tail -1); [ -z "$SUMMARY" ] && SUMMARY='null'
    echo "{\"run\": \"$ROOT\", \"update\": $U, \"checkpoint\": \"$RUN/eval_u$U.msgpack\", \"time\": \"$(date -u +%FT%TZ)\", \"summary\": $SUMMARY}" >> lanerl_jax/runs/EVAL/jax_summary.jsonl
    LAST=$U
  fi
  sleep 300
done
