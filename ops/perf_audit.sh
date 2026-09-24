#!/bin/bash
# Throughput audit jobs (docs/THROUGHPUT_AUDIT.md). One job per argument, each
# < 30 min, results appended to lanerl_jax/runs/perf/audit.jsonl.
#   sbatch --partition=gpup --gres=gpu:1 --cpus-per-task=4 --mem=12G --time=0:30:00 \
#     --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax -J pa-<job> \
#     -o /mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/perf/%x-%j.out ops/perf_audit.sh <job>
set -uo pipefail
ROOT=/mnt/nfs/projects
LIVE=$ROOT/ahriuwu-lanerl-jax
cd "$LIVE"
mkdir -p lanerl_jax/runs/perf
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export PYTHONUNBUFFERED=1
PY=./.venv-gpu/bin/python
OUT=lanerl_jax/runs/perf/audit.jsonl
POOL=lanerl_jax/runs/perf/pool256.npz
CK=lanerl_jax/runs/train/diag1b-20260923-231936-54371e99/ckpt_latest.msgpack
A="$PY -m lanerl_jax.train.throughput_audit"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} $(date -u) job=$1"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
$PY -c "import lanerl_jax; print('lanerl_jax from', lanerl_jax.__file__)"

case "$1" in
acting)   # gate-4 benchmark, HEAD vs the gate-4 commit, same GPU, same job
  for t in ahriuwu-lanerl-jax ahriuwu-perf-d47ab44; do
    cd $ROOT/$t
    echo "=== tree $t"
    $PY -c "import lanerl_jax; print('lanerl_jax from', lanerl_jax.__file__)"
    $PY -m lanerl_jax.train.benchmark --envs 4096 256
    $PY -m lanerl_jax.train.benchmark --envs 4096 --no-smooth
  done ;;
trend)    # coarse bisect: canonical gate-4 number at intermediate commits
  for t in ahriuwu-perf-6bf6968 ahriuwu-perf-2813e91 ahriuwu-bisect-54371e9 ahriuwu-bisect-2d0fdb2 ahriuwu-bisect-a365a52; do
    cd $ROOT/$t
    echo "=== tree $t"
    $PY -m lanerl_jax.train.benchmark --envs 4096
  done ;;
split)    # pool + baseline split + profile + 5-update chunk check
  $A pool --ckpt $CK --out $POOL --jsonl $OUT
  $A split --envs 256 --rollout 128 --ckpt $CK --pool $POOL --reps 5 \
     --chunk-check 5 --profile lanerl_jax/runs/perf/prof_base --jsonl $OUT ;;
phases)
  $A phases --envs 256 --ckpt $CK --pool $POOL --jsonl $OUT
  $A phases --envs 4096 --ckpt $CK --pool $POOL --jsonl $OUT
  $A phases --envs 4096 --ckpt $CK --gate-state --jsonl $OUT ;;
envs)
  for er in "512 64" "1024 32" "512 128" "1024 64" "1024 128"; do
    set -- $er
    $A split --envs $1 --rollout $2 --ckpt $CK --pool $POOL --reps 3 --jsonl $OUT
  done ;;
knobs)
  $A split --envs 256 --rollout 128 --minibatches 8 --ckpt $CK --pool $POOL --reps 3 --jsonl $OUT
  $A split --envs 256 --rollout 128 --epochs 2 --ckpt $CK --pool $POOL --reps 3 --jsonl $OUT
  $A split --envs 256 --rollout 128 --precision bfloat16 --ckpt $CK --pool $POOL --reps 3 --jsonl $OUT
  $A split --envs 256 --rollout 128 --precision highest --ckpt $CK --pool $POOL --reps 3 --jsonl $OUT
  XLA_PYTHON_CLIENT_PREALLOCATE=true $A split --envs 256 --rollout 128 --ckpt $CK --pool $POOL --reps 3 --jsonl $OUT ;;
followup) # combos enabled by mb8's lower VRAM, and the rollout floor
  for c in "2048 16 4 4" "1024 32 4 2" "512 128 8 4" "1024 64 8 4" "1024 32 8 4"; do
    set -- $c
    $A split --envs $1 --rollout $2 --minibatches $3 --epochs $4 --ckpt $CK --pool $POOL --reps 3 --jsonl $OUT
  done ;;
determ_pin) # autotune once, pin the result, reuse it: deterministic at tuned speed?
  AT=$LIVE/lanerl_jax/runs/perf/autotune_base.textproto; rm -f $AT
  XLA_FLAGS="--xla_gpu_dump_autotune_results_to=$AT" $A determ --envs 256 --rollout 128 --updates 20 --jsonl $OUT
  ls -la $AT
  for p in 1 2; do XLA_FLAGS="--xla_gpu_load_autotune_results_from=$AT" $A determ --envs 256 --rollout 128 --updates 20 --jsonl $OUT; done ;;
determ)   # two processes per XLA_FLAGS setting (cross-process autotune differences)
  for p in 1 2; do $A determ --envs 256 --rollout 128 --updates 20 --jsonl $OUT; done ;;
determ_flag)
  for p in 1 2; do XLA_FLAGS="$2" $A determ --envs 256 --rollout 128 --updates 20 --jsonl $OUT; done ;;
*) echo "unknown job $1"; exit 2 ;;
esac
echo "done $(date -u)"
