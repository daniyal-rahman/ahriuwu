#!/bin/bash
# E04: E01's state at update 1520, continued on the ClickV3 server build (unwalkable clicks resolve server-side, PATH-011) with XP weight 0.002 (was 0.005; lr 3e-4 as
# E01). Everything else as E01 except 6 servers x 256-step
# rollouts (same 3072 decisions per update) so it fits beside E01 on the desktop.
# Question: do wall-clicks (46-54% of movement clicks) and the XP camping reward explain the plateau?
SEED=${SEED:-0}
RESUME=${RESUME:-/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/server_train/mirror-wave-s0/server-farm-s0-20260925-195150-1716b293/branch_u1520.msgpack}
SD=/mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/ClickV3/net6.0
OUT=lanerl_jax/runs/E04_mirror_wave_snap_noxp/seed$SEED; mkdir -p $OUT
ARGS=(--envs 6 --opponent mirror --start-near-wave --rollout 256 --updates 20000
      --step-ticks 6 --episode-s 600 --lr 3e-4 --critic-lr 3e-4 --xp-weight 0.002 --no-snap-clicks --entropy-coef 0.001
      --epochs 4 --minibatches 4 --ckpt-every 20 --seed "$SEED" --port-base $((22700 + 40 * SEED))
      --server-dir $SD --resume "$RESUME" --out $OUT)
exec srun -p cpu -w desktop --cpus-per-task=6 --mem=8G --time=24:00:00 --job-name=E04-s$SEED \
  --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax \
  env XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 LANERL_VENDOR_ROOT=/mnt/nfs/projects/lanerl-vendor \
  ./.venv-gpu/bin/python -m lanerl_jax.train.server_train "${ARGS[@]}"
