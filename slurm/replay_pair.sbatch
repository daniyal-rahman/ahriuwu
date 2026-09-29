#!/bin/bash
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
unset XLA_FLAGS
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
./.venv-gpu/bin/python -m pytest -q lanerl_jax/tests/test_replay.py lanerl_jax/train/tests/test_jax_eval.py
echo 'CANARY PASSED: replay snapshot and summary tests'
exec ./.venv-gpu/bin/python -m ops.replay_pair "$1"
