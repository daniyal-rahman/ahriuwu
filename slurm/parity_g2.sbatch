#!/bin/bash
#SBATCH --job-name=parity-g2
#SBATCH --partition=cpu
#SBATCH --nodelist=desktop
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax-g2
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax-g2/lanerl_jax/runs/%x-%j.out
#
# Copy of slurm/parity.sbatch (main worktree), retargeted at the J1 gate 2
# worktree (branch lane-rl/jax-g2) so its own heavy parity runs -- Tier 2
# server episodes for docs/TIER2_DIVERGENCE.md -- are queued on desktop
# rather than run on the shared login node, per this gate's own instructions.
# Two sibling agents queue to the same partition concurrently; slurm
# serialises all three, which is intended, not a bug to route around.
#
#   sbatch slurm/parity_g2.sbatch python -m lanerl_jax.parity.tier2 ...
#
# REPO is the one portable literal: danilogin has /mnt/nfs symlinked to
# /srv/nfs and the compute node has only /mnt/nfs, so this spelling resolves
# on both. Do NOT derive it from BASH_SOURCE -- sbatch copies the script to
# the node's spool.
set -euo pipefail
REPO=/mnt/nfs/projects/ahriuwu-lanerl-jax-g2
test -d "$REPO/lanerl_jax" || { echo "REPO $REPO not visible from $(hostname)"; exit 1; }
cd "$REPO"
mkdir -p lanerl_jax/runs lanerl_jax/runs/tier2

# This worktree's `.venv-jax` is a symlink to the MAIN worktree's real venv,
# written with an absolute /srv/nfs/... target (baked in on danilogin, where
# it resolves). desktop only has /mnt/nfs, so `./.venv-jax/bin/python` (the
# pattern slurm/parity.sbatch uses, which works there because the main
# worktree's .venv-jax is a real directory, not a symlink) fails here with
# "No such file or directory" -- found by running this once. Resolve the
# symlink's literal target ourselves and translate the mount prefix, rather
# than editing the symlink (outside this worktree's job to fix, and it is
# correct as-is on danilogin).
VENV_TARGET="$(readlink "$REPO/.venv-jax")"
VENV="$(echo "$VENV_TARGET" | sed 's#^/srv/nfs#/mnt/nfs#')"
test -x "$VENV/bin/python" || { echo "resolved venv $VENV has no bin/python (from target $VENV_TARGET)"; exit 1; }

export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  cpus ${SLURM_CPUS_PER_TASK:-?}  venv $VENV  $(date -u)"
nproc; free -g | sed -n 2p
"$VENV"/bin/"$@"
echo "done $(date -u)"
