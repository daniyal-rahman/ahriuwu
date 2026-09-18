#!/bin/bash
#SBATCH --job-name=lanerl-train
#SBATCH --partition=gpup
#SBATCH --chdir=/mnt/nfs/projects/_deprecated/ahriuwu-lanerl
#SBATCH --output=/mnt/nfs/projects/_deprecated/ahriuwu-lanerl/lanerl/logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --mail-user=danizainr@gmail.com
#SBATCH --mail-type=FAIL,END
#
# The 1v1 Garen lane-rl PPO run. Submit with:
#   sbatch slurm/slurm_lanerl_train.sbatch <run-name> [extra python -m lanerl_train args...]
#
# Two independent SIGTERM paths land on the same handler in
# lanerl_train/__main__.py, and both matter:
#   - gpup pre-emption sends SIGTERM ~60s before the node is reclaimed
#     (danilogin's ~/CLAUDE.md) -- outside this script's control.
#   - --signal=B:TERM@120 above asks Slurm itself to SIGTERM the job 120s
#     before ITS OWN --time budget runs out, so a run that simply goes the
#     distance also gets a clean checkpoint instead of a SIGKILL.
# Either way lanerl_train/__main__.py forces an immediate checkpoint and
# exits well under 60s; re-submit the same run-name with --resume (this
# script does not do that for you -- see the reminder printed at the end).

set +u
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate ml
set -euo pipefail

# Resolve the repo from the ONE portable literal, /mnt/nfs.
#
# This used to derive REPO from BASH_SOURCE, to avoid hardcoding a mount point
# (lanerl_audit/FINDINGS.md #7, #18: danilogin has /mnt/nfs as a symlink to
# /srv/nfs, desktop has only /mnt/nfs). But sbatch COPIES the batch script to
# the node's spool, so on a compute node BASH_SOURCE is
# /var/spool/slurmd/jobNNNNN/slurm_script and REPO resolved to
# /var/spool/slurmd. Measured on job 692. The avoidance was worse than the
# thing it avoided, and this launcher had never worked -- every real run went
# through lanerl/rl_watchdog.sh, which uses --wrap with an explicit cd.
REPO=/mnt/nfs/projects/_deprecated/ahriuwu-lanerl
test -d "$REPO/lanerl_train" || {
  echo "REPO $REPO is not visible from $(hostname) -- refusing to run from the wrong tree"
  exit 1
}
cd "$REPO"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

RUN_NAME="${1:?usage: sbatch slurm_lanerl_train.sbatch <run-name> [extra args...]}"
shift || true

echo "node=$(hostname) job=${SLURM_JOB_ID:-none} run=$RUN_NAME started=$(date -Is)"

python -m lanerl_train \
  --run-name "$RUN_NAME" \
  --device cuda \
  --num-actors 4 \
  --envs-per-actor 4 \
  --rollout-steps 129 \
  --max-staleness 2 \
  --queue-capacity 2 \
  --checkpoint-every 100 \
  --eval-every 200 \
  --total-updates 200000 \
  "$@" && rc=0 || rc=$?

# $? after the echo's own $(date) is DATE's status, not python's -- the old
# line reported rc=0 for every outcome it was capable of reaching.
echo "finished=$(date -Is) rc=$rc"
echo "REMINDER: a --time-limit or pre-emption exit checkpoints but does NOT resubmit."
echo "Re-run: sbatch $0 $RUN_NAME --resume [same extra args]"
