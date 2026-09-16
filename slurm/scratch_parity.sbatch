#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --partition=cpu
#SBATCH --nodelist=desktop
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --chdir=/scratch/parity/lanerl-jax
#SBATCH --output=/scratch/parity/lanerl-jax/runs-%x-%j.out
#
# Parity work against the SCRATCH clone, not the shared NFS worktree.
#
# Two reasons. Local disk: a single idle-lane trace is 86 MB and the one-step
# differential reads all of it, which is wasteful over NFS. And isolation: the
# NFS worktree is where subagents edit, so measuring there means measuring a
# moving target -- that is what made an earlier run report CS 1 then CS 0 for
# the same seed and nearly get filed as a nondeterminism bug.
#
# The clone is pinned to a commit and refreshed deliberately:
#   ssh desktop 'cd /scratch/parity/lanerl-jax && git fetch -q /mnt/nfs/projects/ahriuwu lane-rl/jax && git checkout -q <sha>'
#
# `lanerl-vendor` is symlinked beside the clone so `paths.vendor_root()`
# (repo_root().parent / "lanerl-vendor") resolves. `.venv-jax` points at the
# NFS venv -- desktop's bare conda env has no jax.
set -euo pipefail
cd /scratch/parity/lanerl-jax
export JAX_PLATFORMS=cpu
export PYTHONPATH=/scratch/parity/lanerl-jax
export PYTHONUNBUFFERED=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  HEAD $(git rev-parse --short HEAD)  $(date -u)"
nproc; free -g | sed -n 2p
./.venv-jax/bin/"$@"
echo "done $(date -u)"
