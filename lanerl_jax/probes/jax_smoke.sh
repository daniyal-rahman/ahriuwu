#!/bin/bash
set -x
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/mnt/nfs/projects/ahriuwu-lanerl-jax
OUT=lanerl_jax/runs/throughput_server_20260925/jax-smoke-mirror; rm -rf $OUT $OUT-eval
.venv-gpu/bin/python -m lanerl_jax.train.jax_train --envs 4 --rollout 16 --updates 2 --opponent mirror --start-near-wave --step-ticks 6 --lr 3e-4 --minibatches 2 --episode-s 600 --out $OUT || exit 1
CK=$(ls -d $OUT/jax-farm-s0-*/ | head -1)ckpt_latest.msgpack
.venv-gpu/bin/python -m lanerl_jax.train.jax_train --envs 2 --opponent mirror --step-ticks 6 --episode-s 20 --out $OUT-eval --resume $CK --eval-episodes 1
