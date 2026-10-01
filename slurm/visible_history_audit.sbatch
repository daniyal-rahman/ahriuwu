#!/bin/bash
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
unset XLA_FLAGS
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
./.venv-gpu/bin/python -m pytest -q lanerl_jax/obs/tests/test_visible_history.py lanerl_jax/train/tests/test_visible_history.py
./.venv-gpu/bin/python -m pytest -q lanerl_jax/train/tests/test_vec_train.py -k 'relative_reward or gru-afk'
echo 'CANARY PASSED: history visibility, migration, nonzero actor/learner likelihood, reset/resume, original v3 AFK collector'
exec ./.venv-gpu/bin/python -m lanerl_jax.probes.visible_history_audit "$1"
