#!/usr/bin/env bash
# EXPERIMENT #0 (fixed): every instance needs its OWN PORT -- they all defaulted
# to 5119, so only the first bound successfully and the rest died silently.
# Also: no bc on this node, so all arithmetic is done in python afterwards.
#
# -e/pipefail: the "N=$N done" line used to print the reported-instance count and
# move on, so the case this script exists to detect -- instances dying silently --
# was printed as a number nobody compared to N.  Every `pkill` below is `|| true`
# because "no process to kill" is the normal case, not a failure.
set -euo pipefail
# derived, not literal: the export is /srv/nfs on danilogin and /mnt/nfs on desktop
NFS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
OUT=$NFS/ahriuwu-lanerl/lanerl/logs/scaling
mkdir -p "$OUT"; rm -f "$OUT"/*.log
export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
export LANERL_HEADLESS=1 LANERL_FREERUN=1 LANERL_TOPONLY=1
BIN=$NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
CFG=$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json
cd "$BIN"
[ -x ./GameServerConsole ] || { echo "FATAL: $BIN/GameServerConsole missing -- nothing to scale"; exit 1; }
[ -f "$CFG" ] || { echo "FATAL: config $CFG does not resolve on $(hostname)"; exit 1; }

FAILED=0
for N in 1 2 4 8 16; do
    pkill -f GameServerConsole 2>/dev/null || true   # nothing running is the normal case
    sleep 3
    for i in $(seq 1 $N); do
        PORT=$((5200 + N*20 + i))
        ( timeout 100 ./GameServerConsole --config "$CFG" --port $PORT > "$OUT/n${N}_i${i}.log" 2>&1 ) &
    done
    sleep 95
    pkill -f GameServerConsole 2>/dev/null || true
    sleep 2
    # an instance that never bound its port writes a log with no LANERL_TPS line;
    # that is the silent death this experiment measures, so compare it to N
    # `|| true`: grep exits 1 when NO log reported, and zero is the answer we
    # want to print, not a reason to abort under set -e/pipefail
    REPORTED=$(grep -l LANERL_TPS "$OUT"/n${N}_i*.log 2>/dev/null | wc -l || true)
    if [ "$REPORTED" -eq "$N" ]; then
        echo "N=$N ok ($REPORTED/$N instances reported)"
    else
        echo "N=$N INSTANCES DIED: only $REPORTED/$N reported LANERL_TPS -- see $OUT/n${N}_i*.log"
        FAILED=1
    fi
done
pkill -f GameServerConsole 2>/dev/null || true
[ "$FAILED" -eq 0 ] || { echo "=== scaling_test FAILED: at least one N lost instances ==="; exit 1; }
echo "=== scaling_test: every instance reported at every N ==="
