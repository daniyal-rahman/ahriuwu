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

# 4. WAIT FOR THE PREVIOUS RUN'S PORTS TO ACTUALLY GO AWAY.
#
# A server whose process is still exiting keeps its control port bound, and
# SO_REUSEADDR does not help against a LIVE listener -- it only covers
# TIME_WAIT. Job 787 died two minutes in because it was launched seconds
# after 786's servers began shutting down: instance 2 got
# "SocketException: Address already in use" on 38388, exited 97, and the run
# failed with "produced no first observation". The port was free by the time
# anyone looked, which is what made it confusing twice.
#
# Bounded, and it says what it is waiting for rather than sleeping blindly.
"$(dirname "$0")/kill_orphan_servers.sh" || true
for _ in $(seq 1 60); do
  busy=$(ss -tan 2>/dev/null     | awk '$4 ~ /^127\.0\.0\.1:(3[89][0-9][0-9][0-9]|40[0-4][0-9][0-9]|62[0-9][0-9][0-9])$/'     | wc -l)
  [ "$busy" -eq 0 ] && break
  echo "[ports] $busy socket(s) still held in the lanerl range; waiting"
  sleep 2
done

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
