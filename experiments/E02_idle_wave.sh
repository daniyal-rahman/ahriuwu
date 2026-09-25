#!/bin/bash
# E02: C# blue vs idle red, near-wave start, 4 servers; same learner as E01.
# Runs on the cpu partition (GPU is visible there) so it can share the node with E01.
SEED=${SEED:-0}; RESUME=${RESUME:-}
SD=/mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0
OUT=lanerl_jax/runs/E02_idle_wave/seed$SEED; mkdir -p $OUT
ARGS=(--envs 4 --opponent idle --start-near-wave --rollout 128 --updates 20000
      --step-ticks 6 --episode-s 600 --lr 3e-4 --critic-lr 3e-4 --entropy-coef 0.001
      --epochs 4 --minibatches 2 --ckpt-every 20 --save-updates 500 1000 2000
      --seed "$SEED" --port-base $((21900 + 40 * SEED)) --server-dir $SD --out $OUT)
[ -n "$RESUME" ] && ARGS+=(--resume "$RESUME")
exec srun -p cpu -w desktop --cpus-per-task=5 --mem=8G --time=24:00:00 --job-name=E02-s$SEED \
  --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax \
  env XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 LANERL_VENDOR_ROOT=/mnt/nfs/projects/lanerl-vendor \
  ./.venv-gpu/bin/python -m lanerl_jax.train.server_train "${ARGS[@]}"
