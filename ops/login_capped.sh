#!/usr/bin/env bash
# Run a command on the LOGIN node under a hard memory cap with swap disabled.
#
# `desktop` is the only node meant to carry parity work, but it goes down, and
# when it does the alternative is not "run it unbounded on danilogin" -- a bare
# `pytest` once held this 6-core shared login node for 5h19m. `MemorySwapMax=0`
# is the load-bearing setting: without it an over-budget job does not fail, it
# quietly pushes every other user's pages to swap and takes hours.  With it the
# job is OOM-killed in seconds and only the job dies.
#
#   ops/login_capped.sh 8G 3 .venv-jax/bin/python -m ...
#            memory cap ^  ^ cpu cores
set -euo pipefail
mem="$1"; cpus="$2"; shift 2
exec systemd-run --user --scope --quiet \
    -p MemoryMax="$mem" -p MemorySwapMax=0 -p CPUQuota="$((cpus * 100))%" \
    nice -n 10 env OMP_NUM_THREADS="$cpus" MKL_NUM_THREADS="$cpus" \
        OPENBLAS_NUM_THREADS="$cpus" JAX_PLATFORMS=cpu \
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$cpus" \
        "$@"
