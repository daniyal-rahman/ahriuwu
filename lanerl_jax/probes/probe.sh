#!/bin/bash
# usage: probe.sh ENVS STEP_TICKS ROLLOUT UPDATES PORTBASE
E=$1; T=$2; R=$3; U=$4; P=$5
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 LANERL_TIME_PHASES=1 LANERL_TIME_PHASES_EVERY=50
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4"
exec .venv-jax/bin/python -m lanerl_jax.train.server_train --envs $E --rollout $R --updates $U \
  --start-near-wave --step-ticks $T --episode-s 600 --port-base $P \
  --server-dir /mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0 \
  --out lanerl_jax/runs/throughput_server_20260925/e$E-t$T
