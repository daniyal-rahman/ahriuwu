#!/bin/bash
# Night-shift operator: every 2 h, if no training job is on the desktop, an
# Opus agent (claude -p) launches the next NIGHT QUEUE item (STATUS.md).
# Registered in /mnt/nfs/shared/jobs/REGISTRY.tsv. Stop: remove the crontab line.
LOG=/mnt/nfs/shared/jobs/night-shift.out
exec 9>/tmp/night-shift.lock; flock -n 9 || { echo "$(date -u +%FT%TZ) already running" >> $LOG; exit 0; }
cd /srv/nfs/projects/ahriuwu-lanerl-jax || exit 1
export PATH=/home/dani/.local/bin:/usr/local/bin:/usr/bin:/bin
echo "=== $(date -u +%FT%TZ) night shift" >> $LOG
# busy = any experiment run, canary or evaluation in the queue (other agents run sequential pipelines with short gaps)
busy=$(squeue -h -o '%t %j' 2>/dev/null | awk '($2 ~ /^E[0-9]+_.*-s[0-9]+$/ || $2 ~ /^canary-/ || $2 ~ /^EVAL/) && ($1=="R" || $1=="PD") {print $2}' | head -3 | tr '\n' ' ')
state=$(sinfo -h -N -p cpu -o '%N %t' | awk '$1=="desktop"{print $2}')
if [ -n "$busy" ]; then echo "busy: $busy (desktop $state); no agent invoked" >> $LOG; exit 0; fi
case "$state" in idle|mix|alloc) ;; *) echo "desktop state '$state'; no agent invoked" >> $LOG; exit 0;; esac
if ! grep -q "^- E[0-9]" <(sed -n '/^## NIGHT QUEUE/,/^## /p' STATUS.md); then echo "night queue empty; no agent invoked" >> $LOG; exit 0; fi
claude -p "$(cat ops/night_shift_prompt.md)" --model opus --dangerously-skip-permissions --output-format text --max-turns 80 < /dev/null >> $LOG 2>&1
echo "=== $(date -u +%FT%TZ) agent exit $?" >> $LOG
