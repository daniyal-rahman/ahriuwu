#!/bin/bash
set -x
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
OUT=lanerl_jax/runs/throughput_server_20260925/smoke-mirror
rm -rf $OUT
.venv-jax/bin/python -m lanerl_jax.train.server_train --envs 2 --rollout 8 --updates 2 --opponent mirror --start-near-wave --step-ticks 6 --lr 3e-4 --minibatches 2 --episode-s 600 --port-base 50100 --server-dir /mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0 --out $OUT || exit 1
CK=$(ls -d $OUT/server-farm-s0-*/ | head -1)ckpt_latest.msgpack
.venv-jax/bin/python -m lanerl_jax.train.server_train --envs 2 --rollout 8 --updates 3 --opponent mirror --start-near-wave --step-ticks 6 --lr 3e-4 --minibatches 2 --episode-s 600 --port-base 50100 --server-dir /mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0 --out $OUT-resume --resume $CK
# frozen evaluation through the collector, 2 envs mirror, short episodes so it finishes
.venv-jax/bin/python -m lanerl_jax.train.server_train --envs 2 --opponent mirror --step-ticks 6 --episode-s 20 --port-base 50100 --server-dir /mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0 --out $OUT-eval --resume $CK --eval-episodes 2
