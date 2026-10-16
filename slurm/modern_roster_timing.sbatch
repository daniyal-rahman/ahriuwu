#!/bin/bash
#SBATCH --job-name=thr-bench
#SBATCH --partition=gpup
#SBATCH --nodelist=desktop
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH --chdir=/mnt/nfs/shared/champ-expansion
#SBATCH --output=/mnt/nfs/shared/champ-expansion/%x-%j.out
#SBATCH --no-requeue
# Replaces 6088; profile mode replaces 6092, submitted with --dependency=afterany:BENCH_ID.
set -euo pipefail
PY=/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python
BASE=/mnt/nfs/shared/champ-expansion
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
export LANERL_JAX_CACHE=/scratch/modern-roster-cache-TEMP-$SLURM_JOB_ID
unset XLA_FLAGS
trap 'exit 143' TERM INT
TOP=(--lanes 2 --no-jungle --no-objectives)
if [[ ${1:-bench} == profile ]]; then
    cd "$BASE/throughput-roster-TEMP"
    "$PY" -m ops.modern.profile_tick --envs 1024 --ticks 100 --top 60 "${TOP[@]}" \
        --trace-dir "$BASE/roster-profile-TEMP-$SLURM_JOB_ID"
    "$PY" -m ops.modern.profile_tick --envs 1024 --ticks 100 --top 60 "${TOP[@]}" --roster all \
        --trace-dir "$BASE/all-kits-profile-TEMP-$SLURM_JOB_ID"
else
    for snapshot in throughput-head-TEMP throughput-roster-TEMP; do
        cd "$BASE/$snapshot"
        "$PY" -m ops.modern.bench --envs 1024 4096 --ticks 300 "${TOP[@]}"
        "$PY" -m ops.modern.bench --envs 1024 --ticks 300
    done
    cd "$BASE/throughput-roster-TEMP"
    "$PY" -m ops.modern.bench --envs 1024 --ticks 300 "${TOP[@]}" --roster all
    for pair in 'Jayce Garen' 'KSante Garen' 'Garen Garen'; do
        read -r blue red <<< "$pair"
        "$PY" -m ops.modern.bench --envs 1024 --ticks 300 "${TOP[@]}" --champions "$blue" "$red"
    done
fi
