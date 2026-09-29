#!/bin/bash
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
unset XLA_FLAGS
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
# exec lets Slurm TERM/USR1 reach the Python checkpoint handler.
echo "host $(hostname) job $SLURM_JOB_ID $(date -u)"
./.venv-gpu/bin/python -m pytest -q lanerl_jax/train/tests/test_paired_initialization.py lanerl_jax/train/tests/test_vec_train.py -k 'shift or relative_reward or gru-mirror'
echo 'CANARY PASSED: vectorized GRU actor/learner and reward invariants'
exec ./.venv-gpu/bin/python -m lanerl_jax.train.paired_vec_train "$1"
