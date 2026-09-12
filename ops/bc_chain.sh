#!/usr/bin/env bash
# Collect demos -> behaviour-clone -> evaluate against the frozen scripted bot.
#
# Chained with --dependency=afterok so a failed stage stops the chain instead of
# training on a stale artifact. Every previous run of this chain was submitted
# as an ad-hoc `sbatch --wrap`, which is why none of it was reproducible and the
# exact flags had to be recovered from log headers.
#
# Baselines to beat, all measured against the same frozen bot:
#   random policy  CS  5.0
#   scripted bot   CS 49.0
#   BC (2026-09-12, before the target-head fix)  CS 0/0/1
#
# Usage: ops/bc_chain.sh [games] [epochs]
set -euo pipefail

# /mnt/nfs is the ONE portable literal. danilogin has /mnt/nfs as a symlink to
# /srv/nfs; desktop has ONLY /mnt/nfs. A job submitted with a /srv/nfs path is
# scheduled on desktop, cannot chdir, and cannot even create its own --output
# file -- so it fails in 1 second with an empty log and nothing to read. That is
# exactly how job 683 died.
REPO=/mnt/nfs/projects/ahriuwu-lanerl
test -d "$REPO" || { echo "REPO $REPO not visible from $(hostname)"; exit 1; }
GAMES=${1:-6}
EPOCHS=${2:-8}
# PYTHONUNBUFFERED: without it a stage writes NOTHING to its log until it
# exits, so a 20-minute job is indistinguishable from a hung one and the
# only way to check progress is to wait for it to finish.
PY="PYTHONUNBUFFERED=1 /home/dani/miniconda3/envs/ml/bin/python"
LOGS=$REPO/lanerl/logs
mkdir -p "$LOGS" "$REPO/demos"

# 4 cores, not 8: danilogin has only 6, so an 8-core request can ONLY ever be
# satisfied by desktop -- and desktop is powered off outside 00:00-09:00, so
# the whole chain sat in PENDING (ReqNodeNotAvail) instead of running on the
# idle node. These stages run one server at a time; 4 is ample.
#
# cpu partition: these stages are server-bound, not GPU-bound. The server runs
# one game per core and the 5080 sits idle through all three, so holding a gpu
# partition here is what left the eval jobs PENDING for hours last time.
# --parsed is not in this Slurm build; take the id off the normal output.
sub() { sbatch --chdir="$REPO" "$@" | awk '{print $NF}'; }

DEMOS=$(sub --partition=cpu --job-name=demos --cpus-per-task=4 \
  --output="$LOGS/demos-%j.out" \
  --wrap "cd $REPO && $PY -m lanerl_train.collect_demos \
          --games $GAMES --out demos/bot_demos.npz")

BC=$(sub --partition=cpu --job-name=bc --cpus-per-task=4 \
  --dependency=afterok:$DEMOS --output="$LOGS/bc-%j.out" \
  --wrap "cd $REPO && $PY -m lanerl_train.bc \
          --demos demos/bot_demos.npz --out demos/bc_policy.pt --epochs $EPOCHS")

EVAL=$(sub --partition=cpu --job-name=bceval --cpus-per-task=4 \
  --dependency=afterok:$BC --output="$LOGS/bceval-%j.out" \
  --wrap "cd $REPO && $PY -m lanerl_train.eval_vs_bot \
          --checkpoint demos/bc_policy.pt --episodes 3 \
          --out lanerl/figs/eval_bc.json")

echo "demos=$DEMOS -> bc=$BC -> eval=$EVAL"
echo "$DEMOS $BC $EVAL" > "$LOGS/.bc_chain_ids"
