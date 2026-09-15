#!/usr/bin/env bash
# Kill game servers left behind by a dead or cancelled run.
#
# An orphaned server keeps its control port bound, and the next run's instance
# then cannot bind: the server exits 97 (LanerlControl's fatal bind guard), the
# harness retries, and the run dies with "produced no first observation".
# That killed job 785 within two minutes of launching it.
#
# WHY A SCRIPT FILE AND NOT AN INLINE COMMAND. `pgrep -f GameServerConsole`
# matches any process whose FULL COMMAND LINE contains the pattern -- including
# an `ssh desktop 'pkill -f GameServerConsole'`, which is how the calling shell
# got killed three times in this project. Run from a file, this process's own
# cmdline is the script path, which does not contain the pattern. The own-pid
# filter below is belt and braces.
#
# `pgrep -x` is not an option: these run under `dotnet`, and a process NAME
# longer than 15 characters never matches.
set -uo pipefail

me=$$
pids=$(pgrep -f GameServerConsole 2>/dev/null | grep -v "^${me}$" || true)

if [ -z "$pids" ]; then
  echo "[orphans] none"
  exit 0
fi

echo "[orphans] killing: $(echo "$pids" | tr '\n' ' ')"
kill $pids 2>/dev/null || true
sleep 3

still=$(pgrep -f GameServerConsole 2>/dev/null | grep -v "^${me}$" || true)
if [ -n "$still" ]; then
  echo "[orphans] SIGKILL: $(echo "$still" | tr '\n' ' ')"
  kill -9 $still 2>/dev/null || true
  sleep 1
fi

left=$(pgrep -f GameServerConsole 2>/dev/null | grep -v "^${me}$" || true)
echo "[orphans] remaining: $(echo "$left" | grep -c . || true)"
