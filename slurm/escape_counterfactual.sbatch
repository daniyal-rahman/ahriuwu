#!/bin/bash
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
unset XLA_FLAGS
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
./.venv-jax/bin/python -m pytest -q lanerl_jax/train/tests/test_jax_eval.py lanerl_jax/tests/test_replay.py
echo 'CANARY PASSED: replay and snapshot checks'
exec ./.venv-jax/bin/python -m lanerl_jax.probes.escape_counterfactual "$1"
