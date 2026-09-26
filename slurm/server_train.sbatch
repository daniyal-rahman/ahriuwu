#!/bin/bash
#SBATCH --job-name=srv-train
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax
#SBATCH --output=/mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/server_train/%x-%j.out
#
# Source-server PPO training (lanerl_jax.train.server_train) on the desktop:
# N C# server processes on the cores, the policy/learner on the 5080.
#
#   sbatch slurm/server_train.sbatch --envs 12 --opponent mirror --updates 2000 ...
#
# Everything after the script name is passed to server_train; --server-dir
# and --out default here to the DeadProbe build (all five vendor patches)
# and lanerl_jax/runs/server_train/. Pass them explicitly to override.
set -euo pipefail
REPO=${REPO:-/mnt/nfs/projects/ahriuwu-lanerl-jax}
cd "$REPO"
mkdir -p lanerl_jax/runs/server_train
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export LANERL_VENDOR_ROOT=${LANERL_VENDOR_ROOT:-/mnt/nfs/projects/lanerl-vendor}
SERVER_DIR=${SERVER_DIR:-$LANERL_VENDOR_ROOT/LoLServer/GameServerConsole/bin/ClickV3/net6.0}
ARGS=("$@")
case " $* " in *" --server-dir "*) ;; *) ARGS+=(--server-dir "$SERVER_DIR");; esac
case " $* " in *" --out "*) ;; *) ARGS+=(--out lanerl_jax/runs/server_train);; esac
echo "host $(hostname) job ${SLURM_JOB_ID:-none} $(date -u)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "args: ${ARGS[*]}"
exec ./.venv-gpu/bin/python -m lanerl_jax.train.server_train "${ARGS[@]}"
