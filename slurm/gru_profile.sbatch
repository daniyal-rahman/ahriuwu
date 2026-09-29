#!/bin/bash
# PERF-003 diagnostic: no checkpoint writes, repeated updates discarded.
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
unset XLA_FLAGS
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
profile_id="$1"
out="lanerl_jax/runs/$profile_id"
echo "host $(hostname) job $SLURM_JOB_ID $(date -u)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
./.venv-gpu/bin/python -m lanerl_jax.probes.gru_throughput --envs 2 --rollout 4 --reps 1 --canary --out "$out/canary"
for count in 16 128; do
    ./.venv-gpu/bin/python -m lanerl_jax.probes.gru_throughput --envs "$count" --out "$out/env$count"
done
echo 'PROFILE COMPLETE'
