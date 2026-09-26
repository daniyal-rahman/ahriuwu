#!/bin/bash
# E01: C# mirror self-play, both champions start behind their first wave at
# 120 s, 10 servers, 10 Hz, lr 3e-4, entropy 0.001, +xp reward, no adv-norm.
SEED=${SEED:-0}; RESUME=${RESUME:-}
ARGS=(--envs 10 --opponent mirror --start-near-wave --rollout 128 --updates 20000
      --step-ticks 6 --episode-s 600 --lr 3e-4 --critic-lr 3e-4 --entropy-coef 0.001
      --epochs 4 --minibatches 4 --ckpt-every 20 --save-updates 500 1000 2000 4000
      --seed "$SEED" --port-base $((21700 + 40 * SEED)) --out lanerl_jax/runs/E01_mirror_wave/seed$SEED)
if [ -n "$RESUME" ]; then [ -f "${RESUME/\/mnt\/nfs/\/srv\/nfs}" ] || { echo "RESUME must exist: $RESUME" >&2; exit 2; }; ARGS+=(--resume "$RESUME"); fi
exec sbatch --parsable --job-name=E01-s$SEED slurm/server_train.sbatch "${ARGS[@]}"
