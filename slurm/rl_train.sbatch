#!/bin/bash
#SBATCH --job-name=rl-lane
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/train/%x-%j.out
#
# THE ONE CANONICAL RL TRAINING ENTRYPOINT.
#
# Why exactly one: `slurm/` already holds 24 sbatch files, two of which pointed
# at worktrees that had been deleted, and the gate-4 throughput figure came from
# a script that was never committed and went stale without anyone noticing
# (docs/JAX_FIDELITY_LEDGER.md says so). A trial campaign whose launcher drifts
# mid-campaign produces results that cannot be compared to each other.
#
# So: everything that defines a run is either baked in here or passed on the
# command line and recorded in the run's own manifest.json. Do not edit the
# defaults below while a campaign is in flight -- start a new tag instead.
#
#   sbatch slurm/rl_train.sbatch --tag base --updates 600
#   sbatch slurm/rl_train.sbatch --tag lr3e4 --updates 600 --lr 3e-4
#
# Every run writes lanerl_jax/runs/train/<tag>-<stamp>-<sha>/ containing
# manifest.json, README.md, metrics.jsonl and rotated checkpoints. Read the
# README first; it is written for someone with no context.
#
# 1060 vs 5080: the gate hardware is `desktop`'s 5080 and gate-4 numbers are
# only valid there. `danilogin`'s 1060 will run a smoke test and must never be
# quoted for throughput.
set -euo pipefail
REPO=${REPO:-/mnt/nfs/projects/ahriuwu-lanerl-jax}
test -d "$REPO/lanerl_jax" || { echo "REPO $REPO not visible from $(hostname)"; exit 1; }
cd "$REPO"
mkdir -p lanerl_jax/runs/train

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
# 1024 envs OOMs a 16 GB 5080 (25.84 GiB requested); 256 is the tested default.
export WANDB_DIR="$REPO/lanerl_jax/runs/train"

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -u)"
# This is a WORKTREE: .git is a file pointing at /srv/nfs/..., which this node
# cannot see (it mounts the same filesystem at /mnt/nfs and has no /srv). Left
# untranslated, every git command here failed and the log printed "clean" for a
# tree it had never read. run_manifest._worktree_git_env does the same thing for
# the manifest.
if ! git rev-parse --git-dir >/dev/null 2>&1 && [ -f .git ]; then
    gd=$(sed -n 's/^gitdir: *//p' .git)
    [ -d "$gd" ] || gd=${gd/\/srv\/nfs\//\/mnt\/nfs\/}
    # `cmd && export` as the last statement of the block would return 1 when
    # the test fails, and `set -e` would kill the whole job over a missing
    # git dir. `if` instead.
    if [ -d "$gd" ]; then export GIT_DIR="$gd" GIT_WORK_TREE="$REPO"; fi
fi
echo "git  $(git rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)$(git status --porcelain 2>/dev/null | grep -q . && echo ' DIRTY' || echo ' clean')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- pinned GEMM autotuning (THROUGHPUT_AUDIT.md section 7, RL-007) --------
# Runs were NOT reproducible across processes: the same command and seed gave
# 1.86 and 18.69 CS. The cause is XLA autotuning picking different GEMM
# kernels per process (the first rollout matches; update 1 diverges). Pinning
# the autotune results makes runs bit-identical across processes for +0.8%
# (noise) and ~100 s less compile; the deterministic-ops flags cost +56%.
# The file is keyed on everything that changes the HLO: the lanerl_jax source,
# the shape/optimiser arguments, jax/jaxlib, and the GPU + driver. A new key
# DUMPS a fresh file (that run autotunes normally); an existing key LOADS it.
set -- --envs 256 --rollout 128 --minibatches 4 --chunk 20 --ckpt-every 5 "$@"
AT_DIR=/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/autotune
mkdir -p "$AT_DIR"
AT_KEY=$(./.venv-gpu/bin/python - "$@" <<'PY'
import hashlib, pathlib, subprocess, sys
import jax, jaxlib
h = hashlib.sha256()
for f in sorted(pathlib.Path("lanerl_jax").rglob("*.py")):
    if "/tests/" in str(f) or "/archive/" in str(f):
        continue
    h.update(str(f).encode()); h.update(f.read_bytes())
# arguments that do not change the compiled program are left out of the key
skip = {"--tag", "--notes", "--out-root", "--resume", "--seed", "--updates",
        "--wandb-tags", "--wandb-project", "--wandb-entity", "--ckpt-every"}
args, it = [], iter(sys.argv[1:])
for a in it:
    if a in skip:
        next(it, None)
        continue
    args.append(a)
h.update(" ".join(args).encode())
h.update(f"{jax.__version__} {jaxlib.__version__}".encode())
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
h.update(gpu.encode())
print(h.hexdigest()[:16])
PY
)
AT_FILE="$AT_DIR/autotune_${AT_KEY}.textproto"
if [ -s "$AT_FILE" ]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_load_autotune_results_from=$AT_FILE"
    echo "autotune: LOAD $AT_FILE (sha256 $(sha256sum "$AT_FILE" | cut -c1-16))"
else
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_dump_autotune_results_to=$AT_FILE"
    echo "autotune: DUMP $AT_FILE (first run for key $AT_KEY; it autotunes normally)"
fi

./.venv-gpu/bin/python -m lanerl_jax.train.run_train "$@"
echo "done $(date -u)"
