#!/usr/bin/env bash
# Run a CPU command on the DESKTOP node's cores through slurm, falling back to
# the capped login node only when the desktop is down.
#
# The desktop has 16 cores / 29 GB; a GPU job there (llm-serve, training)
# typically holds the GPU plus 2 cores, leaving the rest idle. The login node
# has 6 cores shared with everyone. Test suites and scratch probes belong on
# the desktop's idle cores (Dani's standing rule: use the desktop whenever it
# is up). The desktop mounts /srv/nfs as /mnt/nfs, so the working directory
# is translated.
#
#   ops/desktop_cpu.sh 8 14G .venv-jax/bin/python -m pytest -q file.py
#            cores ^  ^ memory
set -euo pipefail
cores="$1"; mem="$2"; shift 2
here="$(cd "$(dirname "$0")/.." && pwd)"
# Desktop only when it is up AND its unallocated cores/memory fit the request;
# otherwise srun would sit PENDING behind a training job instead of falling back.
to_mb() { case "$1" in *G) echo $(( ${1%G} * 1024 ));; *M) echo "${1%M}";; *) echo "$1";; esac; }
fits=0
if sinfo -h -n desktop -p cpu -o "%t" 2>/dev/null | grep -qE "^(idle|mix)"; then
    node=$(scontrol show node desktop 2>/dev/null)
    tot=$(grep -oP "CPUTot=\K[0-9]+" <<<"$node"); alc=$(grep -oP "CPUAlloc=\K[0-9]+" <<<"$node")
    rmem=$(grep -oP "RealMemory=\K[0-9]+" <<<"$node"); amem=$(grep -oP "AllocMem=\K[0-9]+" <<<"$node")
    if (( tot - alc >= cores && rmem - amem >= $(to_mb "$mem") )); then fits=1; fi
fi
if (( fits )); then
    dir="$(pwd | sed 's|^/srv/nfs/|/mnt/nfs/|')"
    exec srun -p cpu -w desktop --cpus-per-task="$cores" --mem="$mem" \
        --time=180 --chdir="$dir" --quiet \
        env JAX_PLATFORMS=cpu OMP_NUM_THREADS="$cores" \
        XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$cores" \
        "$@"
fi
echo "desktop down or full ($cores cores / $mem not free): capped login node" >&2
exec "$here/ops/login_capped.sh" "$mem" "$(( cores < 4 ? cores : 4 ))" "$@"
