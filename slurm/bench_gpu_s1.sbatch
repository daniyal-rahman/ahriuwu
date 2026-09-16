#!/bin/bash
#SBATCH --job-name=bench-gpu-s1
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax-s1
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax-s1/lanerl_jax/runs/%x-%j.out
#
# J1 gate 4/5 re-measurement (throughput with the real policy in the loop,
# and compile time) for the collision-parity pass, on the worktree's own
# code. This worktree has no `.venv-gpu` of its own (a worktree only shares
# git history, not an untracked venv directory) -- reusing the sibling
# `ahriuwu-lanerl-jax` worktree's `.venv-gpu` interpreter is safe because
# that venv does not have `lanerl_jax` pip-installed at all (verified: no
# entry in site-packages, no .pth pointing at it); everything resolves
# through PYTHONPATH, exactly like `.venv-jax` already does in this repo.
set -euo pipefail
REPO=/mnt/nfs/projects/ahriuwu-lanerl-jax-s1
VENV=/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu
test -d "$REPO/lanerl_jax" || { echo "REPO $REPO not visible from $(hostname)"; exit 1; }
test -x "$VENV/bin/python" || { echo "VENV $VENV not visible from $(hostname)"; exit 1; }
cd "$REPO"
mkdir -p lanerl_jax/runs

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -u)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Fail loudly rather than silently reporting a CPU number as a GPU gate-4
# figure: gate 4's published 164x was measured on the 5080, and a CPU number
# is not comparable to it.
"$VENV/bin/python" -c "
import jax
d = jax.devices()[0]
print('jax device:', d, 'platform:', d.platform)
assert d.platform == 'gpu', f'expected a CUDA device, got platform={d.platform!r} ({d})'
"

"$VENV/bin/python" -m lanerl_jax.train.benchmark "$@"
echo "done $(date -u)"
