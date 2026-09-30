#!/bin/bash
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
unset XLA_FLAGS
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
./.venv-jax/bin/python -m pytest -q lanerl_jax/tests/test_replay.py lanerl_jax/train/tests/test_wave_scenario.py
 echo 'CANARY PASSED: replay schema and scenario geometry'
./.venv-jax/bin/python -m lanerl_jax.probes.wave_replay "$1"
for SIDE in 0 1; do
 ./.venv-jax/bin/python -m lanerl_jax.replay_render "/mnt/nfs/shared/$1/low_team_${SIDE}/trace.npz" --out-dir "/mnt/nfs/shared/$1/low_team_${SIDE}" --video --speed 1 --fps 10
 done
