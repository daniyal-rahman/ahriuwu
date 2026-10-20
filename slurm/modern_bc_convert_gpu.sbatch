#!/bin/bash
#SBATCH --job-name=modern-bc-convert-gpu
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=27G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/nfs/shared/modern-bc/%x-%A_%a.out
# Converts a planned dataset on the GPU (pools take 60% of the card: loaded executables live outside them): one Slurm task runs WORKERS conversion processes that share the card
# (the observation and fog are batched JAX; reading and labelling stay on the CPU, one core each). Array task j
# takes dataset tasks j*WORKERS .. j*WORKERS+WORKERS-1 of WORKERS*CHUNKS; run the array %1 (one GPU). Converted
# games are skipped, so a pre-empted task is resubmitted as is. From the login node:
#   sbatch --array=0-$((CHUNKS-1))%1 --export=ALL,CHUNKS=$CHUNKS slurm/modern_bc_convert_gpu.sbatch OUT
set -uo pipefail
CODE=${CODE:-/mnt/nfs/projects/ahriuwu-modern-bc}
cd "$CODE"
W=${WORKERS:-8}; N=$((W * CHUNKS)); j=${SLURM_ARRAY_TASK_ID:-0}
export JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(python3 -c "print(round(0.6 / $W, 3))")
PY=/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python
pids=()
for w in $(seq 0 $((W - 1))); do
  $PY -m lanerl_jax.modern.bc.dataset convert "$1" $((j * W + w)) $N > "${LOGDIR:-/mnt/nfs/shared/modern-bc}/convert-$SLURM_JOB_ID-w$w.log" 2>&1 &
  pids+=($!)
done
trap 'kill -TERM "${pids[@]}" 2>/dev/null' TERM
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done; exit $rc
