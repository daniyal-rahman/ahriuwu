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
#   usage: rl_watchdog.sh <run-name> <hard-deadline-epoch-seconds>
set -uo pipefail

RUN="${1:?run name required}"
HARD_DEADLINE="${2:?deadline epoch seconds required}"
REPO=/mnt/nfs/projects/ahriuwu-lanerl          # portable on BOTH nodes
RUNDIR="$REPO/runs/$RUN"
LOGDIR="$REPO/lanerl/logs"
STATUS="$LOGDIR/watchdog-$RUN.log"
MAX_RESUBMITS=8
POLL_S=300
STALL_S=1800          # no new checkpoint for this long => treat as hung

say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }

resubmits=0
last_ckpt_count=-1
last_progress=$(date +%s)

say "watchdog up: run=$RUN deadline=$(date -d "@$HARD_DEADLINE" '+%F %T') max_resubmits=$MAX_RESUBMITS"

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

    if [ "$ckpts" -ne "$last_ckpt_count" ]; then
        last_ckpt_count=$ckpts
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
               --keep-last-checkpoints 8 --max-staleness 4 --queue-capacity 2 \
               --device cuda --port-base 5700 --resume" >> "$STATUS" 2>&1
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
