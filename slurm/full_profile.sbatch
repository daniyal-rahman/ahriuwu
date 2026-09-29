#!/bin/bash
# PERF-004: frozen-input diagnostics only, bounded and preemptable.
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
unset XLA_FLAGS
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
profile_id="$1"
out="lanerl_jax/runs/$profile_id"
scratch="/scratch/${profile_id}-${SLURM_JOB_ID}"
mkdir -p "$scratch"
finish() {
    if test -d "$scratch/xla"; then
        tar -czf "$out/xla-dumps.tar.gz" -C "$scratch" xla
    fi
}
trap finish EXIT
trap 'exit 143' TERM
echo "host $(hostname) job $SLURM_JOB_ID $(date -u)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
./.venv-gpu/bin/python -m lanerl_jax.probes.perf004_profile --canary --out "$out/canary"
export XLA_FLAGS="--xla_dump_to=$scratch/xla --xla_dump_hlo_as_text --xla_dump_hlo_module_re=jit_(collect|learn|lambda)"
profile_args=()
if [[ "$profile_id" == PERF004c_full_profile ]]; then
    profile_args+=(--single-session)
fi
./.venv-gpu/bin/python -m lanerl_jax.probes.perf004_profile --out "$out/main" "${profile_args[@]}"
