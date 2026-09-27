#!/bin/bash
#SBATCH --job-name=jax-train
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/jax_train/%x-%j.out
#
# PPO on the JAX lane sim (lanerl_jax.train.jax_train) on the desktop GPU.
# gpup holds the GPU against llm-serve (which OOMed the 32-env probe).
#
#   sbatch --job-name=E13_jax_gru_scratch-s0 slurm/jax_train.sbatch --envs 16 --opponent mirror ...
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
mkdir -p lanerl_jax/runs/jax_train
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
echo "host $(hostname) job ${SLURM_JOB_ID:-none} $(date -u)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
echo "args: $*"
exec ./.venv-gpu/bin/python -m lanerl_jax.train.jax_train "$@"
