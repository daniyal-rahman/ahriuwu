#!/bin/bash
#SBATCH --job-name=modern-perf-guard
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/nfs/shared/modern-perf-guard/%x-%j.out
#
# Throughput / memory regression guard for the modern world tick (ops/modern/perf_guard.py).
# Exclusive node so nothing else skews the timing. Runs from CODE (an NFS checkout or snapshot;
# the desktop cannot see login-node worktrees). Exit 0 ok, 1 regression, 3 no baseline.
#
#   mkdir -p /mnt/nfs/shared/modern-perf-guard
#   sbatch slurm/modern_perf_guard.sbatch                       # GPU, 256 envs, vs baseline
#   sbatch slurm/modern_perf_guard.sbatch --cpu                 # CPU backend, 16 envs
#   CODE=/mnt/nfs/shared/<snapshot> sbatch --export=ALL slurm/modern_perf_guard.sbatch --update-baseline
#
# --update-baseline rewrites $CODE/ops/modern/perf_baseline.json; copy it back and commit it.
set -euo pipefail
CODE=${CODE:-/mnt/nfs/projects/ahriuwu-lanerl-jax}
PY=/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python
cd "$CODE"
unset XLA_FLAGS
export PYTHONPATH="$CODE" PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
if [[ "${1:-}" == --cpu ]]; then
    export JAX_PLATFORMS=cpu
    shift
fi
trap 'exit 143' TERM          # gpup pre-emption: no partial result is recorded
echo "host $(hostname) job ${SLURM_JOB_ID:-none} $(date -u) code $CODE platforms $JAX_PLATFORMS args: $*"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
"$PY" -m ops.modern.perf_guard "$@"
