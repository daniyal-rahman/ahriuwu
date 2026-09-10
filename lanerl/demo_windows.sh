#!/usr/bin/env bash
# Heuristic-Garen demo for the Windows validation gate.
#
# Purpose: 5 minutes of follow-locked-cam footage from t=0 proving the sim plays
# real League -- both Garens driven by the scripted bot, watched through the
# actual game client.
#
# RUNBOOK
#   1. Boot Windows (needs sudo from Linux; Dani has to do this).
#   2. On the Windows box, set the camera lock ONCE:
#        C:\lanerl\...\DATA\cfg\defaults\Game.cfg   ->   LockCamera=1
#      (line ~137. It is already set on the Linux copy; the Windows copy is a
#      separate file and was never edited.)
#   3. Run THIS script on whichever Linux node is up.  It binds Address.Any so
#      the Windows client can reach it over the LAN.
#   4. Connect the client as player 1 and start recording BEFORE the game clock
#      moves -- the gate is "from t=0", and the first minute (fountain -> lane
#      -> first wave at 90 s) is the part worth seeing.
#
# THINGS THAT WILL RUIN THE CAPTURE, all learned the hard way:
#   * LANERL_FREERUN must stay OFF. Free-run decouples the tick loop from the
#     wall clock and runs ~150x; the client cannot render that, and an earlier
#     "gameplay" capture came out as 39 byte-identical PNGs because of a related
#     mistake. Real-time is REQUIRED for a watchable video.
#   * LANERL_BOT must be "both". It defaults to "blue", which leaves the enemy
#     Garen standing in his fountain for five minutes -- not a 1v1 demo.
#   * LANERL_TOPONLY must stay OFF here. It is a training optimisation that
#     suppresses mid/bot waves and all neutral camps; the gate is "everything
#     else working properly", so the demo runs the full map.
#   * Watch the champion, not the map. A previous capture was called fake
#     because Garen HAD moved but the camera never left the fountain. That is
#     what LockCamera=1 in step 2 is for.
set -uo pipefail

# derived, not literal: the export is /srv/nfs on danilogin and /mnt/nfs on
# desktop, so either literal resolves on exactly one node
NFS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
OUT="$NFS/ahriuwu-lanerl/lanerl/logs"
export DOTNET_ROOT="$NFS/lanerl-vendor/dotnet"

export LANERL_BOT=both          # BOTH Garens scripted -> a real 1v1 lane
export LANERL_RECORD="$OUT/state_demo.jsonl"   # state track alongside the video
unset LANERL_FREERUN            # real time, or the client renders nothing
unset LANERL_HEADLESS           # a client is going to attach
unset LANERL_TOPONLY            # full map for the demo

CFG="$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json"
BIN="$NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0"
[ -f "$BIN/GameServerConsole.dll" ] || { echo "FATAL: no build at $BIN" >&2; exit 1; }
[ -f "$CFG" ] || { echo "FATAL: no config at $CFG" >&2; exit 1; }
rm -f "$LANERL_RECORD"

echo "serving $(hostname) : connect the Windows client, then record from t=0"
echo "  bot=both  freerun=off(real time)  toponly=off(full map)"
cd "$BIN"
exec "$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$CFG"
