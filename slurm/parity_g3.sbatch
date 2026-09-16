#!/bin/bash
#SBATCH --job-name=parity-g3
#SBATCH --partition=cpu
#SBATCH --nodelist=desktop
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax-g3
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax-g3/lanerl_jax/runs/%x-%j.out
#
# Worktree-local copy of slurm/parity.sbatch, pointed at the g3 worktree
# instead of the main one -- see that file for why this exists at all
# (CPU-only parity work belongs on `desktop`, not the shared login node: one
# run was already OOM-killed there while two subagents were running, and
# separately, running an oracle-gate server boot on the login node
# concurrently with OTHER worktrees' server boots produced visibly corrupted
# timing -- LANERL_TPS swinging between 0.13x and 6x real time in the same
# 600 s episode -- which is exactly the kind of run that once got filed as a
# false nondeterminism bug (see slurm/scratch_parity.sbatch's docstring).
#
#   sbatch slurm/parity_g3.sbatch python -m lanerl_jax.parity.hp_band ...
#
# REPO is the one portable literal: danilogin has /mnt/nfs symlinked to
# /srv/nfs and the compute node has only /mnt/nfs, so this spelling resolves on
# both. Do NOT derive it from BASH_SOURCE -- sbatch copies the script to the
# node's spool.
set -euo pipefail
REPO=/mnt/nfs/projects/ahriuwu-lanerl-jax-g3
test -d "$REPO/lanerl_jax" || { echo "REPO $REPO not visible from $(hostname)"; exit 1; }
cd "$REPO"
mkdir -p lanerl_jax/runs

export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
# The vendored server needs its own dotnet; lanerl_train.paths resolves it.
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  cpus ${SLURM_CPUS_PER_TASK:-?}  $(date -u)"
nproc; free -g | sed -n 2p
./.venv-jax/bin/"$@"
echo "done $(date -u)"
