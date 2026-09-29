#!/bin/bash
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
unset XLA_FLAGS
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "host $(hostname) job $SLURM_JOB_ID $(date -u)"
./.venv-gpu/bin/python -m pytest -q lanerl_jax/train/tests/test_wave_scenario.py lanerl_jax/train/tests/test_vec_train.py -k 'scenario or relative_reward or gru-mirror or gru-frozen'
echo 'CANARY PASSED: scenario geometry, health roles, GRU likelihood and reward'
exec ./.venv-gpu/bin/python -m lanerl_jax.train.wave_scenario_train "$1"
