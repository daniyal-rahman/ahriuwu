#!/usr/bin/env bash
# Server on danilogin (binds Address.Any); the real Windows client connects over LAN.
# Recording on too, so we get state alongside whatever the client renders.
set -uo pipefail
# derived, not literal: the export is /srv/nfs on danilogin and /mnt/nfs on
# desktop, so either literal resolves on exactly one node.
NFS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"; OUT=$NFS/ahriuwu-lanerl/lanerl/logs
export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
export LANERL_BOT=blue   # was the server-side default until 2026-09-10;
                         # DriveTeams now defaults to "none", so stating it
                         # here keeps this script doing what it always did
export LANERL_RECORD=$OUT/state_win.jsonl
rm -f "$LANERL_RECORD"
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
exec "$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json"
