#!/bin/bash
# Fallback watcher for slurm batches.
#
# WHY: launching a batch and checking back by hand has failed twice in one
# session -- five jobs died on a NameError within seconds, and a recording job
# hung for 70 minutes after finishing its work, blocking the queue. Both were
# found by chance. Silence is not success, so this reports every terminal
# state, not just the happy one.
#
#   usage: ops/watch_jobs.sh <name-prefix> <outfile-glob> [stall-seconds]
#
# Emits one line per event. Exits when no job with the prefix remains.
PREFIX="${1:?job name prefix}"
GLOB="${2:?output file glob}"
STALL="${3:-600}"
declare -A SEEN SIZE STAMP
SEEN_ANY=
EMPTY_ONCE=
while true; do
  RUNNING=$(squeue -h -o '%j' | grep -c "^${PREFIX}"); RUNNING=${RUNNING:-0}
  # STALL must only consider jobs actually RUNNING. Counting queued ones made a
  # job pending longer than the stall window fire a false STALL -- the
  # cries-wolf problem the chdir filter comment warns about.
  ACTIVE=$(squeue -h -t R -o '%j' | grep -c "^${PREFIX}"); ACTIVE=${ACTIVE:-0}
  for f in $GLOB; do
    [ -e "$f" ] || continue
    # failure signatures -- report once each
    # NOTE the exclusion. Every desktop job prints
    #   slurmstepd: error: couldn't chdir to `/srv/nfs/...': going to /tmp
    # because /srv/nfs is not mounted there and the payload cds to /mnt/nfs
    # itself. Matching it made the watcher fire FAIL on all five healthy jobs
    # within seconds of arming -- a filter that cries wolf on every run is
    # worse than none, because it trains you to ignore it.
    if grep -vE "couldn't chdir|cudart_stub|absl::|RuntimeWarning" "$f" 2>/dev/null \
         | grep -qE 'Traceback|Error:|error:|FAILED|Killed|OOM|exit code [1-9]'; then
      key="err:$f"
      if [ -z "${SEEN[$key]}" ]; then
        SEEN[$key]=1
        echo "FAIL $f :: $(grep -vE "couldn't chdir|cudart_stub|absl::" "$f" \
              | grep -m1 -E 'Traceback|Error:|error:|FAILED|Killed|OOM' \
              | cut -c1-140)"
      fi
    fi
    # progress
    n=$(grep -c 'seed ' "$f" 2>/dev/null); n=${n:-0}
    key="prog:$f"
    if [ "${SEEN[$key]:-0}" != "$n" ]; then
      SEEN[$key]=$n
      [ "$n" -gt 0 ] && echo "PROG $f :: $n seeds done"
    fi
    # stall detection: file has not grown in STALL seconds while a job runs
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ "${SIZE[$f]:-0}" != "$sz" ]; then SIZE[$f]=$sz; STAMP[$f]=$now; fi
    age=$(( now - ${STAMP[$f]:-$now} ))
    if [ "$ACTIVE" -gt 0 ] && [ "$age" -gt "$STALL" ] && [ -z "${SEEN[stall:$f]}" ]; then
      SEEN[stall:$f]=1
      echo "STALL $f :: no output for ${age}s while jobs still queued/running"
    fi
  done
  # GRACE BEFORE DECLARING DONE. `RUNNING` is read at the top of the loop, so
  # arming the watcher in the same breath as the submit -- or hitting any squeue
  # gap before the jobs register -- saw 0 on the first pass and printed
  # "DONE ... no jobs remain", which is the precise failure the header warns
  # against: silence that reads as success. Require the queue to be empty on
  # SEEN_ANY, or twice in a row, before believing it.
  if [ "$RUNNING" -eq 0 ]; then
    if [ -n "${SEEN_ANY}" ] || [ -n "${EMPTY_ONCE}" ]; then
      echo "DONE no '${PREFIX}*' jobs remain in the queue"
      exit 0
    fi
    EMPTY_ONCE=1
    echo "WAIT queue shows no '${PREFIX}*' job yet -- not declaring done on the first pass"
  else
    SEEN_ANY=1
    EMPTY_ONCE=
  fi
  sleep 20
done
