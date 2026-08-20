#!/bin/bash
# Nightly BC training window: 11pm-11am PT, so the 5080 is free for probes/tests
# during the day. Desktop clock is UTC and PT is UTC-7 (PDT) => 06:00-18:00 UTC.
# NB: at the Nov DST change PT becomes UTC-8; shift these to 07:00-19:00 then.
#
#   bc_night.sh supervise  - THE ONE TO RUN. Owns the schedule: loops forever,
#                            starts the trainer when the window opens, stops it
#                            when it closes. Launch detached:
#         setsid nohup bash scratchpad/bc_night.sh supervise >/dev/null 2>&1 &
#   bc_night.sh start|stop - manual override
#   bc_night.sh status     - where are we
#
# NB: this supervisor does NOT survive a desktop reboot — relaunch it with the
# command above (it self-corrects immediately, starting or staying idle
# according to the clock).
set -uo pipefail
REPO=/mnt/nfs/projects/ahriuwu
WD="${BC_WATCHDOG:-$REPO/ops/bc5080_clicks_watchdog.sh}"
STOP="${BC_STOP:-$REPO/ops/bc5080_clicks.stop}"
# pgrep pattern for THIS watchdog, derived from $WD. The bracket trick stops
# the pattern from matching the pgrep/pkill command line itself.
WDBASE=$(basename "$WD" .sh)
WDPAT="${WDBASE:0:8}[${WDBASE:8:1}]${WDBASE:9}"
LOG="$REPO/ops/bc_night.log"
START_H=6      # UTC
END_H=18       # UTC

log() { echo "$(date -u '+%m-%d %H:%M UTC') $*" >> "$LOG"; }

in_window() {
  local h; h=$(date -u +%-H)
  [ "$h" -ge "$START_H" ] && [ "$h" -lt "$END_H" ]
}

start() {
  if pgrep -f "$WDPAT" >/dev/null; then
    log "start: already running"; return 0
  fi
  # NFS must be up (checkpoints + code live there)
  if [ ! -f "$WD" ]; then log "start: ABORT, $WD not visible (NFS down?)"; return 1; fi
  rm -f "$STOP"
  setsid nohup bash "$WD" </dev/null >/dev/null 2>&1 &
  sleep 20
  if pgrep -f "train_agent[_]finetune" >/dev/null; then
    log "start: trainer up"
  else
    log "start: watchdog launched but trainer not up yet (it retries)"
  fi
}

stop() {
  touch "$STOP"
  sleep 2
  pkill -f "$WDPAT" 2>/dev/null
  # SIGTERM the trainer; it checkpoints every --checkpoint-minutes, so we lose
  # at most that much progress. Resume is automatic next window.
  pkill -TERM -f "train_agent[_]finetune" 2>/dev/null
  sleep 5
  pkill -KILL -f "train_agent[_]finetune" 2>/dev/null
  log "stop: window closed; GPU released"
}

supervise() {
  log "supervise: up (window ${START_H}-${END_H} UTC = 11pm-11am PT)"
  local was=""
  while true; do
    if in_window; then
      if [ "$was" != "in" ]; then log "window OPEN -> starting"; was=in; fi
      pgrep -f "$WDPAT" >/dev/null || start
    else
      if [ "$was" != "out" ]; then
        log "window CLOSED -> stopping"; was=out; stop
      elif pgrep -f "train_agent[_]finetune" >/dev/null; then
        log "trainer running outside window -> stopping"; stop   # manual leftovers
      fi
    fi
    sleep 300
  done
}

case "${1:-}" in
  supervise) supervise ;;
  start) in_window || log "start: called outside window, running anyway (manual)"; start ;;
  stop)  stop ;;
  status)
    pgrep -f "train_agent[_]finetune" >/dev/null && echo "trainer UP" || echo "trainer down"
    pgrep -f "$WDPAT" >/dev/null && echo "watchdog UP" || echo "watchdog down"
    in_window && echo "inside window (06-18 UTC / 23-11 PT)" || echo "outside window"
    ;;
  *) echo "usage: $0 {start|stop|maybe|status}"; exit 2 ;;
esac
