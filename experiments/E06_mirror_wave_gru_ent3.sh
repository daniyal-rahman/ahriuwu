#!/bin/bash
# E06: E05 with the entropy coefficient scaled per head (0.01/3 on the 3-head sum).
# (docs/HYPERPARAMS.md: lr 2.5e-4 annealed, lambda 0.95, entropy 0.01, grad clip 0.5,
# adv-norm on, no KL stop), ClickV3 server, mirror near-wave, 10 servers x 128 steps.
# Budget 6000 updates = 15.4M agent decisions (~10 h on the desktop).
SEED=${SEED:-0}; RESUME=${RESUME:-}
ARGS=(--envs 10 --opponent mirror --start-near-wave --rollout 128 --updates 6000
      --step-ticks 6 --episode-s 600 --preset standard --core gru --lr-anneal
      --xp-weight 0.002 --no-snap-clicks --minibatches 4 --ckpt-every 20
      --save-updates 500 1000 2000 4000 --seed "$SEED" --port-base $((23100 + 40 * SEED))
      --out lanerl_jax/runs/E06_mirror_wave_gru_ent3/seed$SEED)
[ -n "$RESUME" ] && ARGS+=(--resume "$RESUME")
exec sbatch --parsable --job-name=E06-s$SEED slurm/server_train.sbatch "${ARGS[@]}"
