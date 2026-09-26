#!/bin/bash
# E07: E06's learner (GRU, standard defaults) continued from E06 update 3140,
# but the opponent is FROZEN at E06's best checkpoint (update 2200) instead of
# the live learner; the learner's side alternates blue/red across the 6 servers.
# Question: is E06's 12-20 CS plateau self-play drift into duelling?
SEED=${SEED:-0}
RESUME=${RESUME:-/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/E06_mirror_wave_gru_ent3/seed0/server-farm-s0-20260926-151710-1a89b8e4/eval_u3140.msgpack}
OPP=${OPP:-/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/E06_mirror_wave_gru_ent3/seed0/server-farm-s0-20260926-142410-38b42e5a/eval_u2200.msgpack}
ARGS=(--envs 6 --opponent frozen --opponent-ckpt "$OPP" --start-near-wave --rollout 256 --updates 6000
      --step-ticks 6 --episode-s 600 --preset standard --core gru --lr-anneal
      --xp-weight 0.002 --no-snap-clicks --minibatches 3 --ckpt-every 20
      --save-updates 3500 4000 5000 --seed "$SEED" --port-base $((23300 + 40 * SEED))
      --out lanerl_jax/runs/E07_frozen_opponent/seed$SEED)
if [ -n "$RESUME" ]; then [ -f "${RESUME/\/mnt\/nfs/\/srv\/nfs}" ] || { echo "RESUME must exist: $RESUME" >&2; exit 2; }; ARGS+=(--resume "$RESUME"); fi
exec srun -p cpu -w desktop --cpus-per-task=6 --mem=10G --time=24:00:00 --job-name=E07-s$SEED \
  --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax \
  env XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 LANERL_VENDOR_ROOT=/mnt/nfs/projects/lanerl-vendor \
  ./.venv-gpu/bin/python -m lanerl_jax.train.server_train "${ARGS[@]}"
