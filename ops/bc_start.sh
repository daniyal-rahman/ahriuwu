#!/bin/bash
# Bulletproof detached start for the clicks BC run.
#
# Launching the watchdog straight from an ssh command line does NOT reliably
# survive the ssh session ending: `setsid nohup ... &` inside a `cd && ...`
# chain still got reaped on 2026-08-14, killing a live run with no traceback
# (the tell: no Python error in the log, just silence). Invoking a FILE with
# nohup, and double-forking via setsid inside it, does survive -- same pattern
# the action-conditioning A/B used successfully.
#
# Usage (from login):  ssh desktop 'nohup /mnt/nfs/projects/ahriuwu/ops/bc_start.sh >/dev/null 2>&1 &'
# Stop:                touch /mnt/nfs/projects/ahriuwu/ops/bc5080_parity.stop
set -uo pipefail
REPO=/mnt/nfs/projects/ahriuwu
cd "$REPO" || exit 1
rm -f ops/bc5080_parity.stop

if pgrep -f "bc5080_p[a]rity_watchdog" >/dev/null; then
  echo "$(date -u '+%m-%d %H:%M UTC') bc_start: watchdog already up" >> ops/bc_night.log
  exit 0
fi

setsid bash ops/bc5080_parity_watchdog.sh </dev/null >/dev/null 2>&1 &
disown
echo "$(date -u '+%m-%d %H:%M UTC') bc_start: watchdog launched" >> ops/bc_night.log
