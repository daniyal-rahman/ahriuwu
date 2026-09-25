#!/bin/bash
# Crash-safe supervisor for the Phase-2 BC run on the 1060. Runs the trainer,
# and on any non-zero exit restarts it (--resume auto picks up the latest
# time-based checkpoint, so <=checkpoint-minutes is lost). A background poller
# writes a heartbeat/status file every 2 min = the telemetry that was missing.
# Stop cleanly: `touch scratchpad/bc.stop`.
set -uo pipefail
cd /srv/nfs/projects/ahriuwu
STOP=scratchpad/bc.stop
LOG=scratchpad/bc_1060.log
STATUS=scratchpad/bc_status.txt
CKPT=data/phase2_bc_garen
rm -f "$STOP"
[ -f "$LOG" ] && mv "$LOG" "$LOG.prev"
: > "$LOG"

# --- heartbeat/telemetry poller ---
(
  while [ ! -f "$STOP" ]; do
    last=$(grep -aE "Epoch [0-9]+ \[" "$LOG" 2>/dev/null | tail -1)
    ck="$CKPT/agent_finetune_latest.pt"
    if [ -f "$ck" ]; then age="$((($(date +%s) - $(stat -c %Y "$ck")) / 60))min"; else age="NONE-YET"; fi
    alive=$(pgrep -f train_agent_finetune >/dev/null && echo UP || echo DOWN)
    printf '%s | proc:%s | ckpt_age:%s | %s\n' \
      "$(date '+%m-%d %H:%M')" "$alive" "$age" "${last:-<initializing>}" > "$STATUS"
    sleep 120
  done
) &
POLLER=$!
trap 'kill $POLLER 2>/dev/null' EXIT

# --- run + auto-resume loop ---
runs=0
while [ ! -f "$STOP" ]; do
  runs=$((runs + 1))
  echo "[watchdog $(date '+%T')] launch attempt #$runs (resume auto)" >> "$LOG"
  bash scripts/launch_bc_1060.sh
  rc=$?
  echo "[watchdog $(date '+%T')] trainer exited rc=$rc" >> "$LOG"
  [ -f "$STOP" ] && { echo "[watchdog] stop file -> exit" >> "$LOG"; break; }
  if [ "$rc" -eq 0 ]; then echo "[watchdog] clean finish -> exit" >> "$LOG"; break; fi
  echo "[watchdog] crash rc=$rc -> resuming in 15s" >> "$LOG"; sleep 15
done
