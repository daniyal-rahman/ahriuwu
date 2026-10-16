#!/bin/bash
#SBATCH --job-name=modern045-ppo-memory
#SBATCH --partition=gpup
#SBATCH --nodelist=desktop
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:45:00
#SBATCH --chdir=/mnt/nfs/shared/champ-expansion
#SBATCH --output=/mnt/nfs/shared/champ-expansion/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
: "${BENCH:?Set BENCH to the immutable MODERN-045 diagnostic snapshot}"
PY=/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python
unset XLA_FLAGS
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
trap 'exit 143' TERM INT
scontrol show job "$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cd "$BENCH/modern"
srun "$PY" -m ops.modern.ppo_memory_bench --root "$BENCH" --output "$BENCH/results-$SLURM_JOB_ID"
