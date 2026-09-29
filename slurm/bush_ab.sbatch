#!/bin/bash
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
unset XLA_FLAGS
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
trap 'exit 143' TERM
profile_id="$1"
echo "host $(hostname) job $SLURM_JOB_ID $(date -u)"
./.venv-gpu/bin/python -m lanerl_jax.probes.perf005_vision_ab --canary
./.venv-gpu/bin/python -m pytest -q lanerl_jax/obs/tests/test_vision.py
echo 'CANARY PASSED: fused ray equivalence and lane lookup semantics'
./.venv-gpu/bin/python -m lanerl_jax.probes.perf005_vision_ab --bush-ab --out "lanerl_jax/runs/$profile_id/main"
