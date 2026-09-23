#!/bin/bash
#SBATCH --job-name=rl-lane
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/train/%x-%j.out
#
# THE ONE CANONICAL RL TRAINING ENTRYPOINT.
#
# Why exactly one: `slurm/` already holds 24 sbatch files, two of which pointed
# at worktrees that had been deleted, and the gate-4 throughput figure came from
# a script that was never committed and went stale without anyone noticing
# (docs/JAX_FIDELITY_LEDGER.md says so). A trial campaign whose launcher drifts
# mid-campaign produces results that cannot be compared to each other.
#
# So: everything that defines a run is either baked in here or passed on the
# command line and recorded in the run's own manifest.json. Do not edit the
# defaults below while a campaign is in flight -- start a new tag instead.
#
#   sbatch slurm/rl_train.sbatch --tag base --updates 600
#   sbatch slurm/rl_train.sbatch --tag lr3e4 --updates 600 --lr 3e-4
#
# Every run writes lanerl_jax/runs/train/<tag>-<stamp>-<sha>/ containing
# manifest.json, README.md, metrics.jsonl and rotated checkpoints. Read the
# README first; it is written for someone with no context.
#
# 1060 vs 5080: the gate hardware is `desktop`'s 5080 and gate-4 numbers are
# only valid there. `danilogin`'s 1060 will run a smoke test and must never be
# quoted for throughput.
set -euo pipefail
REPO=/mnt/nfs/projects/ahriuwu-lanerl-jax
test -d "$REPO/lanerl_jax" || { echo "REPO $REPO not visible from $(hostname)"; exit 1; }
cd "$REPO"
mkdir -p lanerl_jax/runs/train

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
# 1024 envs OOMs a 16 GB 5080 (25.84 GiB requested); 256 is the tested default.
export WANDB_DIR="$REPO/lanerl_jax/runs/train"

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -u)"
echo "git  $(git rev-parse --short HEAD)$(git status --porcelain | grep -q . && echo ' DIRTY' || echo ' clean')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

./.venv-gpu/bin/python -m lanerl_jax.train.run_train \
    --envs 256 --rollout 128 --minibatches 4 \
    --chunk 20 --ckpt-every 5 \
    "$@"
echo "done $(date -u)"
