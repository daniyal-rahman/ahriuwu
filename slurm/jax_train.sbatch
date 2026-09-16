#!/bin/bash
#SBATCH --job-name=jax-lane
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/%x-%j.out
#
# Detached so it does not depend on a live shell.
#
# An `srun` foreground job keeps a client process alive on the LOGIN node for
# the whole run, and a long one gets reaped there -- which is how a 400-update
# job died with nothing to show for it. sbatch writes to the file above and
# survives whatever happens to the session that submitted it.
#
#   sbatch slurm/jax_train.sbatch --envs 256 --rollout 128 --updates 400
#
# REPO is the one portable literal: danilogin has /mnt/nfs as a symlink to
# /srv/nfs and the compute node has only /mnt/nfs, so this is the spelling that
# resolves on both. (Do NOT derive it from BASH_SOURCE: sbatch copies the
# script to the node's spool, and lanerl's own launcher once resolved REPO to
# /var/spool/slurmd that way.)
set -euo pipefail
REPO=/mnt/nfs/projects/ahriuwu-lanerl-jax
test -d "$REPO/lanerl_jax" || { echo "REPO $REPO not visible from $(hostname)"; exit 1; }
cd "$REPO"
mkdir -p lanerl_jax/runs

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -u)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
./.venv-gpu/bin/python -m lanerl_jax.train.run_train "$@"
echo "done $(date -u)"
