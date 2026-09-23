#!/usr/bin/env bash
# Run lanerl_jax/sim/tests on the DESKTOP, several files at once, each in its
# own memory-capped scope (MemorySwapMax=0: an over-budget file is OOM-killed
# alone instead of thrashing). Goes over ssh, not slurm, on purpose: a GPU
# training job reserves 24 GB / 8 cores through slurm but uses ~4 GB / 2
# cores, and that reservation would otherwise lock the idle cores out.
# test_combat.py (the largest) runs last, alone, with a bigger cap.
#
#   ops/desktop_suite.sh [parallel=5] [per-file mem=5G]  > log
set -uo pipefail
par="${1:-5}"; mem="${2:-5G}"
src="$(cd "$(dirname "$0")/.." && pwd)"
dir="$(echo "$src" | sed 's|^/srv/nfs/|/mnt/nfs/|')"
ssh -o BatchMode=yes desktop bash -s -- "$dir" "$par" "$mem" <<'REMOTE'
set -uo pipefail
dir="$1"; par="$2"; mem="$3"; cd "$dir"
run() {  # file mem cores
    f="$1"; m="$2"; c="$3"
    if out=$(systemd-run --user --scope --quiet -p MemoryMax="$m" -p MemorySwapMax=0 \
             nice -n 5 env JAX_PLATFORMS=cpu OMP_NUM_THREADS="$c" \
             XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$c" \
             .venv-jax/bin/python -m pytest -q -p no:cacheprovider "$f" 2>&1); then
        echo "PASS $f: $(echo "$out" | tail -1)"
    else
        echo "FAIL $f"; echo "$out" | grep -v "cudart\|absl" | tail -30
    fi
}
export -f run
ls lanerl_jax/sim/tests/test_*.py | grep -v test_combat.py \
  | xargs -P "$par" -I{} bash -c 'run {} '"$mem"' 2'
run lanerl_jax/sim/tests/test_combat.py 16G 4
REMOTE
# (summary: grep -c ^PASS / ^FAIL on the log)
