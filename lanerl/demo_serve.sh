#!/usr/bin/env bash
# Demo server for the Windows validation gate. Run on a Linux node; the real
# Windows client connects over the LAN.
#
# LANERL_HEADLESS=1 is REQUIRED here even though a client does attach. Without
# it the stock server counts "2/2 players disconnected" and quits 120 s after
# "Game is ready" (forcedStart in the config), which killed a capture attempt
# while the client was still on its connect-retry dialog. The headless patch
# makes CheckIfAllPlayersLeft() return false so the server waits indefinitely.
#
# NOT set, deliberately:
#   LANERL_FREERUN  -- must stay off; free-run runs ~150x and no client can
#                      render that. Real time is required for a watchable video.
#   LANERL_TOPONLY  -- off, so the demo shows the full map.
set -uo pipefail
NFS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
OUT="$NFS/ahriuwu-lanerl/lanerl/logs"
LOG="$OUT/demo_server.log"

export DOTNET_ROOT="$NFS/lanerl-vendor/dotnet"
export LANERL_BOT=both            # both Garens scripted -> a real 1v1 lane
export LANERL_HEADLESS=1          # see above: do not remove
export LANERL_RECORD="$OUT/state_demo.jsonl"

CFG="$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json"
BIN="$NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0"
rm -f "$LOG" "$LANERL_RECORD"
cd "$BIN"
nohup "$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$CFG" > "$LOG" 2>&1 &
echo "pid $!  log $LOG"
for _ in $(seq 1 40); do grep -q "Game is ready" "$LOG" 2>/dev/null && break; sleep 1; done
grep -qi "Game is ready" "$LOG" && echo "READY" || { echo "FAILED to become ready"; tail -5 "$LOG"; exit 1; }
