#!/bin/bash
# Frozen-policy evaluation through the training collector (no learning).
#   CKPT=<ckpt_latest.msgpack> [ENVS=5] [EPISODES=1] [OPP=mirror] [WAVE=1] [SEED=0] experiments/EVAL_frozen.sh
# Writes lanerl_jax/runs/EVAL/<run-name>/ with per-episode CS/deaths per agent.
set -e
: "${CKPT:?set CKPT}"; ENVS=${ENVS:-4}; EPISODES=${EPISODES:-1}; OPP=${OPP:-mirror}; WAVE=${WAVE:-1}; SEED=${SEED:-0}
SD=/mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/ClickV3/net6.0
ARGS=(${EXTRA:-} --envs $ENVS --opponent $OPP --step-ticks 6 --episode-s 600 --seed $SEED --port-base ${PORT:-22500}
      --server-dir $SD --resume "$CKPT" --eval-episodes $EPISODES --out lanerl_jax/runs/EVAL)
[ "$WAVE" = 1 ] && ARGS+=(--start-near-wave)
exec srun -p cpu -w desktop --cpus-per-task=2 --mem=6G --time=4:00:00 --job-name=EVAL \
  --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax \
  env XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 LANERL_VENDOR_ROOT=/mnt/nfs/projects/lanerl-vendor \
  ./.venv-gpu/bin/python -m lanerl_jax.train.server_train "${ARGS[@]}"
