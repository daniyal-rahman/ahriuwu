#!/bin/bash
#SBATCH --job-name=modern-bc-train
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/nfs/shared/modern-bc/%x-%j.out
# Behaviour cloning on a converted dataset (lanerl_jax/modern/bc/train.py). From the login node:
#   sbatch slurm/modern_bc_train.sbatch /mnt/nfs/shared/bc_modern_v1 [trainer flags...]
set -euo pipefail
CODE=${CODE:-/mnt/nfs/projects/ahriuwu-modern-bc}
cd "$CODE"
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
PY=/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python
exec $PY -m lanerl_jax.modern.bc.train "$@"
