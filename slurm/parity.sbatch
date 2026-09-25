#!/bin/bash
#SBATCH --job-name=parity
#SBATCH --partition=cpu
#SBATCH --nodelist=desktop
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/%x-%j.out
#
# CPU-only parity work: one-step differentials, trace recording, perturbation
# sweeps. None of it needs the GPU -- the plan is explicit that J1's Tier-1 work
# is CPU-only by design and only the throughput gates touch the 5080.
#
# Queue it here rather than running on the login node. `danilogin` has 6 cores
# and 23 GB and is shared with editors, agents and whatever else; `desktop` has
# 16 and 29 GB. A one-step differential over a full trace was OOM-killed on the
# login node while two subagents were running, which is a scheduling problem,
# not a memory-footprint problem.
#
#   sbatch slurm/parity.sbatch python -m lanerl_jax.parity.archive.one_step ...
#
# REPO is the one portable literal: danilogin has /mnt/nfs symlinked to
# /srv/nfs and the compute node has only /mnt/nfs, so this spelling resolves on
# both. Do NOT derive it from BASH_SOURCE -- sbatch copies the script to the
# node's spool.
set -euo pipefail
REPO=/mnt/nfs/projects/ahriuwu-lanerl-jax
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
