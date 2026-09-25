#!/bin/bash
# E03: E01's policy/optimizer state at update 1120, continued at lr 1e-4 (actor and
# critic) instead of 3e-4. Everything else as E01 except 6 servers x 256-step
# rollouts (same 3072 decisions per update) so it fits beside E01 on the desktop.
# Question: is E01's plateau at 16-18 train CS a step-size problem?
SEED=${SEED:-0}
RESUME=${RESUME:-/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/server_train/mirror-wave-s0/server-farm-s0-20260925-195150-1716b293/branch_u1120.msgpack}
SD=/mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0
OUT=lanerl_jax/runs/E03_mirror_wave_lr1e-4/seed$SEED; mkdir -p $OUT
ARGS=(--envs 6 --opponent mirror --start-near-wave --rollout 256 --updates 20000
      --step-ticks 6 --episode-s 600 --lr 1e-4 --critic-lr 1e-4 --entropy-coef 0.001
      --epochs 4 --minibatches 4 --ckpt-every 20 --seed "$SEED" --port-base $((22100 + 40 * SEED))
      --server-dir $SD --resume "$RESUME" --out $OUT)
exec srun -p cpu -w desktop --cpus-per-task=6 --mem=8G --time=24:00:00 --job-name=E03-s$SEED \
  --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax \
  env XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 LANERL_VENDOR_ROOT=/mnt/nfs/projects/lanerl-vendor \
  ./.venv-gpu/bin/python -m lanerl_jax.train.server_train "${ARGS[@]}"
