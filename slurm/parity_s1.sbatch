#!/bin/bash
#SBATCH --job-name=parity-s1
#SBATCH --partition=cpu
#SBATCH --nodelist=desktop
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax-s1
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax-s1/lanerl_jax/runs/%x-%j.out
set -euo pipefail
REPO=/mnt/nfs/projects/ahriuwu-lanerl-jax-s1
test -d "$REPO/lanerl_jax" || { echo "REPO $REPO not visible from $(hostname)"; exit 1; }
cd "$REPO"
mkdir -p lanerl_jax/runs

export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  cpus ${SLURM_CPUS_PER_TASK:-?}  $(date -u)"
nproc; free -g | sed -n 2p
./.venv-jax/bin/"$@"
echo "done $(date -u)"
