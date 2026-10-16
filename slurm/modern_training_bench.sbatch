#!/bin/bash
#SBATCH --job-name=modern-ppo-compare
#SBATCH --partition=gpup
#SBATCH --nodelist=desktop
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --chdir=/mnt/nfs/shared/champ-expansion
#SBATCH --output=/mnt/nfs/shared/champ-expansion/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
: "${BENCH:?Set BENCH to a prepared immutable benchmark directory}"
PY=/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python
unset XLA_FLAGS
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# TERM aborts the active cell; previously completed JSON rows survive pre-emption.
trap 'exit 143' TERM INT
scontrol show job "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
srun "$PY" "$BENCH/training_bench.py" matrix --root "$BENCH" \
    --output "$BENCH/results-$SLURM_JOB_ID" "$@"
