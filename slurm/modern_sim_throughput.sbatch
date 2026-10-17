#!/bin/bash
#SBATCH --job-name=modern-sim-throughput
#SBATCH --partition=gpup
#SBATCH --nodelist=desktop
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/nfs/shared/champ-expansion
#SBATCH --output=/mnt/nfs/shared/champ-expansion/%x-%j.out
#SBATCH --no-requeue
set -euo pipefail
: "${CANDIDATE:?Set CANDIDATE to the pinned sim snapshot}"
PY=/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python
unset XLA_FLAGS
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
trap 'exit 143' TERM INT
TOP=(--lanes 2 --no-jungle --no-objectives)
if [[ ${1:-bench} == profile ]]; then
    cd "$CANDIDATE"
    export LANERL_JAX_CACHE=/scratch/modern-sim-profile-TEMP-$SLURM_JOB_ID
    "$PY" -m ops.modern.profile_tick --envs 1024 --ticks 100 --warm-ticks 5400 --orders abilities \
        "${TOP[@]}" --top 40 --trace-dir "/mnt/nfs/shared/champ-expansion/sim-top-profile-TEMP-$SLURM_JOB_ID"
else
    : "${BASELINE:?Set BASELINE to the comparison snapshot with the same benchmark driver}"
    for snapshot in "$BASELINE" "$CANDIDATE"; do
        cd "$snapshot"
        export LANERL_JAX_CACHE=/scratch/modern-sim-bench-TEMP-$SLURM_JOB_ID-$(basename "$snapshot")
        echo "=== snapshot $snapshot"
        "$PY" -m ops.modern.bench --envs 1024 --warm-ticks 1200 --samples 5 "${TOP[@]}"
        for envs in 1024 4096; do
            "$PY" -m ops.modern.bench --envs "$envs" --warm-ticks 5400 --samples 5 --orders abilities "${TOP[@]}"
        done
        "$PY" -m ops.modern.bench --envs 1024 --warm-ticks 5400 --samples 5 --orders abilities
        "$PY" -m ops.modern.bench --envs 1024 --warm-ticks 5400 --samples 5 --orders abilities \
            --champions Jayce Garen "${TOP[@]}"
    done
fi
