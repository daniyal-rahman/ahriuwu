#!/bin/bash
# Interface oracle on the C# server: the scripted last-hitter (train/scripted_policy.py)
# through the exact observation -> click path the trained policy uses.
#   oracle_eval.sh <mirror|idle> <lasthit|any> <port>
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 LANERL_VENDOR_ROOT=/mnt/nfs/projects/lanerl-vendor JAX_PLATFORMS=cpu
OUT=lanerl_jax/runs/ORACLE/$1-$2; mkdir -p $OUT
./.venv-jax/bin/python -m lanerl_jax.train.server_train --envs 4 --opponent $1 --start-near-wave --step-ticks 6 --episode-s 600 --seed 0 --port-base $3 --server-dir /mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/ClickV3/net6.0 --eval-episodes 1 --scripted $2 --out $OUT
