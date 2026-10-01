#!/bin/bash
set -euo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export LANERL_VENDOR_ROOT=/mnt/nfs/projects/lanerl-vendor
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
WORLD_PYTHON=/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python
"$WORLD_PYTHON" -m pytest -q lanerl_jax/data/tests/test_modern_map.py lanerl_jax/data/tests/test_modern_routes.py
printf 'CANARY PASSED\n'
if [[ "$1" == routes ]]; then
    "$WORLD_PYTHON" -m lanerl_jax.data.modern_routes --map /mnt/nfs/datasets/league/26.19/map11-base --out "$2/routes"
else
    "$WORLD_PYTHON" -m pytest -q lanerl_jax/sim/tests/test_modern_*.py tests/test_modern_champions.py tests/test_modern_collector.py
fi
printf 'MODERN WORLD VALIDATION COMPLETE\n'
