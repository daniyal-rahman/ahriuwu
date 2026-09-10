#!/usr/bin/env bash
# Headless verification: run a real game with NO client and record full state at
# 10Hz. This is the evidence that matters -- wave timings, minion pathing, turret
# behaviour, gold/XP curves, Garen's abilities -- none of which need a renderer.
set -uo pipefail
DUR=${1:-660}
# derived, not literal: the export is /srv/nfs on danilogin and /mnt/nfs on
# desktop, so either literal resolves on exactly one node.
NFS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
export LANERL_BOT=blue   # was the server-side default until 2026-09-10;
                         # DriveTeams now defaults to "none", so stating it
                         # here keeps this script doing what it always did
export LANERL_HEADLESS=1
export LANERL_RECORD=$OUT/state.jsonl
rm -f "$LANERL_RECORD"

cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
"$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json" \
    >"$OUT/server_headless.log" 2>&1 &
SRV=$!
echo "server pid $SRV; forcedStart=120s then it runs with zero clients"

for t in $(seq 1 $((DUR/60))); do
    sleep 60
    lines=$(wc -l < "$LANERL_RECORD" 2>/dev/null || echo 0)
    gt=$(tail -1 "$LANERL_RECORD" 2>/dev/null | grep -oE '"t":[0-9]+' | head -1 | cut -d: -f2)
    units=$(tail -1 "$LANERL_RECORD" 2>/dev/null | grep -o '"id"' | wc -l)
    echo "  ${t}m ticks=$lines gametime=${gt:-0}ms units=$units alive=$(kill -0 $SRV 2>/dev/null && echo yes || echo NO)"
done
kill $SRV 2>/dev/null
echo "=== recorded $(wc -l < "$LANERL_RECORD" 2>/dev/null) ticks, $(du -h "$LANERL_RECORD" 2>/dev/null | cut -f1) ==="
grep -vE "WARN|Could not find script" "$OUT/server_headless.log" | tail -5
