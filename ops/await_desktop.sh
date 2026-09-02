#!/bin/bash
# Runs on LOGIN. Polls for the desktop and launches the anneal probe the moment
# it returns. Written because the machine is dual-boot and unreachable most of the
# day, and the alternative -- me noticing it came back -- has already cost ~11h of
# idle GPU once this week.
set -uo pipefail
REPO=/srv/nfs/projects/ahriuwu
LOG=$REPO/ops/await_desktop.log
echo "$(date -u '+%m-%d %H:%M UTC') waiting for desktop" >> "$LOG"
until timeout 12 ssh -o ConnectTimeout=8 -o BatchMode=yes desktop 'true' 2>/dev/null; do sleep 180; done
echo "$(date -u '+%m-%d %H:%M UTC') desktop UP -> launching anneal probe" >> "$LOG"
timeout 30 ssh -o ConnectTimeout=10 -o BatchMode=yes desktop \
  'nohup /mnt/nfs/projects/ahriuwu/ops/anneal_probe.sh >/dev/null 2>&1 &' 2>/dev/null
echo "$(date -u '+%m-%d %H:%M UTC') launched" >> "$LOG"
