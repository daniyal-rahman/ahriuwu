#!/bin/bash
set -euo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export LANERL_VENDOR_ROOT=/mnt/nfs/projects/lanerl-vendor
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python -m pytest tests/test_modern_champions.py lanerl_jax/parity/tests/test_autoattack.py lanerl_jax/sim/tests/test_orders.py lanerl_jax/train/tests/test_actions.py -q
/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python -m lanerl_jax.probes.modern_champions "$1" "$2"
