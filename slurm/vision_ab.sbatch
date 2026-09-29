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
./.venv-gpu/bin/python - <<'PY'
import pytest
from lanerl_jax.obs import vision
from lanerl_jax.probes.perf005_ray_kernel import clear_ray_fused
vision.clear_ray = clear_ray_fused
raise SystemExit(pytest.main(['-q', 'lanerl_jax/obs/tests/test_vision.py']))
PY
echo 'CANARY PASSED: GPU ray equivalence and existing vision integration tests'
./.venv-gpu/bin/python -m lanerl_jax.probes.perf005_vision_ab --out "lanerl_jax/runs/$profile_id/main"
