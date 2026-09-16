#!/bin/bash
#SBATCH --job-name=parity-s2
#SBATCH --partition=cpu
#SBATCH --nodelist=desktop
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax-s2
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax-s2/lanerl_jax/runs/%x-%j.out
#
# CPU-only parity work for THIS worktree (lane-rl/jax-s2). Copied from
# slurm/parity.sbatch (the ahriuwu-lanerl-jax sibling worktree's script)
# rather than written from scratch, per the same lesson that script already
# paid for: Slurm opens --chdir/--output ON THE COMPUTE NODE, and `desktop`
# mounts the shared export at /mnt/nfs while `danilogin` (and this worktree's
# own path as seen from the login node) uses /srv/nfs. A job submitted with
# /srv/nfs paths in --chdir/--output fails before it can print why (job 842,
# 2026-09-16: exit 1 after 31s, no log anywhere, because the ad hoc script
# used to submit it spelled both paths from the login node's mount).
#
#   sbatch slurm/parity_s2.sbatch python -m pytest -q -s \
#       lanerl_jax/parity/tests/test_last_hit_gate.py \
#       -k test_oracle_scores_the_same_cs_in_sim_and_server
#
# REPO is the one portable literal: danilogin has /mnt/nfs symlinked to
# /srv/nfs and the compute node has only /mnt/nfs, so this spelling resolves
# on both. Do NOT derive it from BASH_SOURCE -- sbatch copies the script to
# the node's spool.
set -euo pipefail
REPO=/mnt/nfs/projects/ahriuwu-lanerl-jax-s2
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
