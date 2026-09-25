#!/usr/bin/env bash
# Keep the overnight PPO run alive without a human.
#
# The failure this exists for: a training job dies at 2am, nothing notices, and
# six hours of GPU time are spent idle. A watcher that merely *reports* has the
# same failure mode as no watcher, so this one RESUBMITS.
#
# Everything here is bounded. It stops at HARD_DEADLINE no matter what, it will
# not resubmit more than MAX_RESUBMITS times, and every wait is a sleep with a
# fixed bound rather than a blocking wait on a condition that may never come.
#
#   usage: rl_watchdog.sh <run-name> <hard-deadline-epoch-seconds> [extra args...]
#
# Extra args are passed through to every resubmit VERBATIM. They have to be:
# the resubmit line used to hardcode its own flag list, so any flag the first
# launch used and this list omitted was silently dropped the moment the job
# bounced. With --kl-ref-coef that is invisible and fatal -- the KL term to the
# BC prior would just stop applying partway through the night, and the loss
# curve would not obviously say so.
set -uo pipefail

RUN="${1:?run name required}"
HARD_DEADLINE="${2:?deadline epoch seconds required}"
shift 2
EXTRA=("$@")
REPO=/mnt/nfs/projects/ahriuwu-lanerl          # portable on BOTH nodes
RUNDIR="$REPO/runs/$RUN"
LOGDIR="$REPO/lanerl/logs"
STATUS="$LOGDIR/watchdog-$RUN.log"
# Anchor evaluation now starts real game servers, so a resubmit must pin its
# own port base or it collides with the actors. max_staleness dropped 4 -> 2:
# 4 was chosen to silence rejections, and queue_capacity+num_actors-1 = 4 meant
# the run sat AT the bound (mean 3.31), which is most of the clip fraction.
MAX_RESUBMITS=8
POLL_S=300
STALL_S=1800          # no new checkpoint for this long => treat as hung

say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }

resubmits=0
last_ckpt_count=-1
last_progress=$(date +%s)

say "watchdog up: run=$RUN deadline=$(date -d "@$HARD_DEADLINE" '+%F %T') max_resubmits=$MAX_RESUBMITS extra=${EXTRA[*]-none}"

while :; do
    now=$(date +%s)
    if [ "$now" -ge "$HARD_DEADLINE" ]; then
        say "hard deadline reached; watchdog exiting (training job, if any, is left alone)"
        exit 0
    fi

    running=$(squeue -h -u dani -n lrl-train 2>/dev/null | wc -l)
    ckpts=$(ls "$RUNDIR"/checkpoints/update_*.pt 2>/dev/null | wc -l)
    latest=$(ls -1 "$RUNDIR"/checkpoints/update_*.pt 2>/dev/null | tail -1)
    updates=$(grep -c '"kind":"update"' "$RUNDIR/metrics.jsonl" 2>/dev/null || echo 0)

    # Progress is the UPDATE COUNT, not the number of checkpoint FILES.
    # --keep-last-checkpoints 8 caps that file count at 8, so once the run had
    # written 8 it never changed again, every poll looked like a stall, and this
    # watchdog cancelled four perfectly healthy jobs before spending its budget.
    # A watchdog that kills the thing it is guarding is worse than none.
    progress=$updates
    if [ "$progress" -ne "$last_ckpt_count" ]; then
        last_ckpt_count=$progress
        last_progress=$now
    fi
    stalled=$(( now - last_progress ))

    say "poll: running=$running checkpoints=$ckpts updates=$updates stalled_for=${stalled}s latest=$(basename "${latest:-none}")"

    if [ "$running" -eq 0 ]; then
        if [ "$resubmits" -ge "$MAX_RESUBMITS" ]; then
            say "job is down and resubmit budget ($MAX_RESUBMITS) is spent; giving up"
            exit 1
        fi
        resubmits=$((resubmits+1))
        say "job is DOWN -- resubmitting with --resume ($resubmits/$MAX_RESUBMITS)"
        # -t is clamped to what is left before the deadline, so a resubmit can
        # never outlive the window the operator asked for.
        left_min=$(( (HARD_DEADLINE - now) / 60 ))
        [ "$left_min" -lt 10 ] && { say "under 10 min left; not resubmitting"; exit 0; }
        sbatch -p cpu -w desktop -J lrl-train -c 15 --gres=gpu:1 -t "$left_min" \
            --chdir="$REPO" -o "$LOGDIR/train-%j.out" --wrap \
            "cd $REPO && /home/dani/miniconda3/envs/ml/bin/python -m lanerl_train \
               --run-name $RUN --num-actors 3 --envs-per-actor 4 \
               --rollout-steps 256 --total-updates 20000 \
               --checkpoint-every 25 --snapshot-every 200 --eval-every 400 \
               --keep-last-checkpoints 12 --max-staleness 2 --queue-capacity 2 \
               --device cuda --port-base 5700 --anchor-port-base 31000 --resume \
               ${EXTRA[*]+${EXTRA[*]}}" >> "$STATUS" 2>&1
    elif [ "$stalled" -ge "$STALL_S" ]; then
        # Running but producing nothing: a wedged server or a deadlocked
        # lockstep read looks exactly like this, and would burn the whole night.
        if [ "$resubmits" -ge "$MAX_RESUBMITS" ]; then
            say "stalled ${stalled}s and resubmit budget spent; leaving it alone"
            exit 1
        fi
        resubmits=$((resubmits+1))
        say "STALLED ${stalled}s with no new checkpoint -- cancelling and resuming ($resubmits/$MAX_RESUBMITS)"
        scancel -n lrl-train 2>/dev/null
        sleep 20
        last_progress=$(date +%s)
    fi

    sleep "$POLL_S"
done
