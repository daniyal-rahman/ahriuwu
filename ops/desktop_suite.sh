#!/usr/bin/env bash
# Run lanerl_jax/sim/tests on the DESKTOP, several files at once, each in its
# own memory-capped scope (MemorySwapMax=0: an over-budget file is OOM-killed
# alone instead of thrashing). Goes over ssh, not slurm, on purpose: a GPU
# training job reserves 24 GB / 8 cores through slurm but uses ~4 GB / 2
# cores, and that reservation would otherwise lock the idle cores out.
# Total budget while a training job shares the desktop: par x mem must stay
# well under the node's free memory. On 2026-09-23 a 16 GB test_combat.py
# beside a training job starved the desktop, slurmd missed its heartbeats,
# slurmctld marked the node FAILED and REQUEUED the training job from zero.
# So the suite refuses to start if par x mem exceeds FREE memory minus 8 GB,
# and any file whose tests need more than the per-file cap is killed alone.
#
#   ops/desktop_suite.sh [parallel=5] [per-file mem=5G]  > log
set -uo pipefail
par="${1:-5}"; mem="${2:-5G}"
if [[ ! "$par" =~ ^[1-9][0-9]*$ || ! "$mem" =~ ^[1-9][0-9]*G$ ]]; then
    echo "usage: $0 [positive parallel count] [positive memory in G]" >&2
    exit 2
fi
src="$(cd "$(dirname "$0")/.." && pwd)"
dir="$(echo "$src" | sed 's|^/srv/nfs/|/mnt/nfs/|')"
ssh -o BatchMode=yes desktop bash -s -- "$dir" "$par" "$mem" <<'REMOTE'
set -uo pipefail
dir="$1"; par="$2"; mem="$3"; cd "$dir"
avail=$(free -g | awk '/^Mem:/{print $7}')
need=$(( par * ${mem%G} ))
# The sequential route-table test uses 10G even with a smaller batch cap.
(( need >= 10 )) || need=10
if (( need > avail - 8 )); then echo "REFUSED: peak budget ${need}G > available ${avail}G - 8G headroom"; exit 2; fi
run() {  # file mem cores
    f="$1"; m="$2"; c="$3"
    if out=$(systemd-run --user --scope --quiet -p MemoryMax="$m" -p MemorySwapMax=0 \
             nice -n 5 env JAX_PLATFORMS=cpu LANERL_VENDOR_ROOT=/mnt/nfs/projects/lanerl-vendor OMP_NUM_THREADS="$c" \
             XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$c" \
             .venv-jax/bin/python -m pytest -q -p no:cacheprovider "$f" 2>&1); then
        echo "PASS $f: $(echo "$out" | tail -1)"
    else
        echo "FAIL $f"; echo "$out" | grep -v "cudart\|absl" | tail -30
        return 1
    fi
}
export -f run
# Files that load the full route table need more than a small per-file cap;
# they run after the parallel batch, one at a time, at 10G.
HEAVY='test_local_pathing.py'
status=0
ls lanerl_jax/sim/tests/test_*.py | grep -v -E "$HEAVY" \
  | xargs -P "$par" -I{} bash -c 'run {} '"$mem"' 2'
[[ $? -eq 0 ]] || status=1
for h in $(echo "$HEAVY" | tr '|' ' '); do
    run "lanerl_jax/sim/tests/$h" 10G 4 || status=1
done
exit "$status"
REMOTE
# (summary: grep -c ^PASS / ^FAIL on the log)
