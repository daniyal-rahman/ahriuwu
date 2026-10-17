#!/bin/bash
#SBATCH --job-name=modern-bc-gate
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/nfs/shared/modern-bc/%x-%j.out
# Frozen in-sim rollouts of a BC checkpoint (lanerl_jax/modern/bc/gate.py). From the login node:
#   sbatch slurm/modern_bc_gate.sbatch /mnt/nfs/checkpoints/lanerl-jax/modern-bc/<run>/ckpt_latest.msgpack [gate flags...]
set -euo pipefail
CODE=${CODE:-/mnt/nfs/projects/ahriuwu-modern-bc}
cd "$CODE"
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
PY=/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python
exec $PY -m lanerl_jax.modern.bc.gate "$@"
