#!/usr/bin/env bash
# Submit a training run to desktop's Slurm. Wraps the three things that have
# each silently killed a run here at least once.
#
#   lanerl/launch_rl.sh <run-name> [extra args to python -m lanerl_train ...]
#
# 1. --chdir MUST be /mnt/nfs/... . The repo is reachable as both
#    /srv/nfs/projects/ahriuwu-lanerl (on the login box) and
#    /mnt/nfs/projects/ahriuwu-lanerl (on the desktop node), and only the
#    second exists on the node. A job submitted with the /srv path dies in
#    about a second having written no output at all -- there is no error to
#    read, because the shell it would have come from never started.
#
# 2. Writes go to /scratch (local NVMe), not NFS, via run_on_scratch.sh. At
#    32+ server instances in a lockstep loop the per-instance logs and the
#    ~70 MB checkpoints put network writes on the critical path.
#
# 3. Cores. A request over ~6 CPUs can never be satisfied on the login
#    partition and the job sits PENDING forever without complaining, so this
#    always targets the desktop node explicitly.
#
# Everything after <run-name> is forwarded verbatim, so the per-run knobs
# (--opponent, --init-from, --total-updates, ...) stay at the call site and
# out of this file.
set -euo pipefail

RUN="${1:?usage: launch_rl.sh <run-name> [args...]}"; shift

REPO_NODE=/mnt/nfs/projects/ahriuwu-lanerl
LOGS="$REPO_NODE/lanerl/logs"
CPUS="${LANERL_CPUS:-14}"
TIME="${LANERL_TIME:-48:00:00}"

script=$(mktemp /tmp/launch_rl.XXXXXX.sbatch)
cat > "$script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$RUN
#SBATCH --chdir=$REPO_NODE
#SBATCH -o $LOGS/$RUN.%j.out
#SBATCH -c $CPUS
#SBATCH --gres=gpu:1
#SBATCH -t $TIME
srun lanerl/run_on_scratch.sh "$RUN" $*
EOF

mkdir -p "$LOGS"
sbatch "$script"
