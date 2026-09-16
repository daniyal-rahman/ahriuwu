#!/bin/bash
#SBATCH --job-name=parity-g1
#SBATCH --partition=cpu
#SBATCH --nodelist=desktop
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax-g1
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax-g1/lanerl_jax/runs/%x-%j.out
#
# g1's own copy of slurm/parity.sbatch, pointed at the g1 worktree instead of
# the main one -- the main copy hardcodes `/mnt/nfs/projects/ahriuwu-lanerl-jax`
# in --chdir/--output/REPO, and sbatch directives cannot be parameterised from
# the command line, so a separate file is the only way to run this worktree's
# code on `desktop` without touching the main worktree (out of bounds, see
# CLAUDE.md / the gate-1 task brief).
#
# `--chdir`/`--output` MUST be spelled /mnt/nfs, not /srv/nfs: slurm opens
# --output on the COMPUTE node, which only has /mnt/nfs mounted (the login
# node has both /srv/nfs and /mnt/nfs -> same export). A /srv/nfs spelling
# here produces a job that dies before writing anything, silently.
#
# g1's .venv-jax is a RELATIVE symlink (../ahriuwu-lanerl-jax/.venv-jax) so it
# resolves under either mount point -- an earlier absolute-under-/srv/nfs
# version of that symlink dangled on desktop and killed a job at start (exit
# 127, no output). Fixed upstream; noted here so nobody re-breaks it by hand.
#
#   sbatch slurm/parity_g1.sbatch python -m lanerl_jax.runs.tier1_full
#
set -euo pipefail
REPO=/mnt/nfs/projects/ahriuwu-lanerl-jax-g1
test -d "$REPO/lanerl_jax" || { echo "REPO $REPO not visible from $(hostname)"; exit 1; }
cd "$REPO"
mkdir -p lanerl_jax/runs

export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
export PYTHONUNBUFFERED=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  cpus ${SLURM_CPUS_PER_TASK:-?}  $(date -u)"
nproc; free -g | sed -n 2p
ls -la .venv-jax/bin/python || { echo ".venv-jax/bin/python not found or symlink dangling"; exit 1; }
./.venv-jax/bin/"$@"
echo "done $(date -u)"
