#!/bin/bash
# E05: from scratch, GRU core (memory is learned), standard PPO defaults
# (docs/HYPERPARAMS.md: lr 2.5e-4 annealed, lambda 0.95, entropy 0.01, grad clip 0.5,
# adv-norm on, no KL stop), ClickV3 server, mirror near-wave, 10 servers x 128 steps.
# Budget 6000 updates = 15.4M agent decisions (~10 h on the desktop).
# STOPPED 2026-09-26 03:20 at update 912: entropy 0.01 on the 3-head SUM pinned
# entropy at 9.0 (frozen 6.5/10.0 CS at u860). Pinned here for reproducibility;
# E06 is the same with the per-head-scaled coefficient (the preset's new default).
SEED=${SEED:-0}; RESUME=${RESUME:-}
ARGS=(--envs 10 --opponent mirror --start-near-wave --rollout 128 --updates 6000
      --step-ticks 6 --episode-s 600 --preset standard --entropy-coef 0.01 --core gru --lr-anneal
      --xp-weight 0.002 --no-snap-clicks --minibatches 4 --ckpt-every 20
      --save-updates 500 1000 2000 4000 --seed "$SEED" --port-base $((22900 + 40 * SEED))
      --out lanerl_jax/runs/E05_mirror_wave_gru_standard/seed$SEED)
[ -n "$RESUME" ] && ARGS+=(--resume "$RESUME")
exec sbatch --parsable --job-name=E05-s$SEED slurm/server_train.sbatch "${ARGS[@]}"
