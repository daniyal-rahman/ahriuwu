#!/bin/bash
# Build a modern-world BC dataset (lanerl_jax/modern/bc/dataset.py) on the desktop CPU. From the login node:
#   sbatch -p cpu --nodelist=desktop -c 2 --mem=6G slurm/modern_bc_dataset.sbatch plan OUT
#   sbatch -p cpu --nodelist=desktop -c 2 --mem=6G --array=0-23 slurm/modern_bc_dataset.sbatch convert OUT 24
#   sbatch -p cpu --nodelist=desktop -c 2 --mem=6G --array=0-23 slurm/modern_bc_dataset.sbatch relabel OUT 24
#   sbatch -p cpu --nodelist=desktop -c 2 --mem=6G slurm/modern_bc_dataset.sbatch finalize OUT
set -euo pipefail
CODE=${CODE:-/mnt/nfs/projects/ahriuwu-modern-bc}
cd "$CODE"
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python
command=$1; out=$2
case $command in
  convert|relabel) exec $PY -m lanerl_jax.modern.bc.dataset "$command" "$out" "$SLURM_ARRAY_TASK_ID" "$3" ;;
  *) exec $PY -m lanerl_jax.modern.bc.dataset "$command" "$out" "${@:3}" ;;
esac
