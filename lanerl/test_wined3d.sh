#!/usr/bin/env bash
# wined3d (OpenGL) on the REAL X server WITH GPU access -- never tested together.
# Every prior wined3d attempt was on Xvfb with no GPU, where it had no chance.
set -uo pipefail
ROOT=/mnt/storage/lanerl; NFS=/mnt/nfs/projects
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
D=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy
export WINEPREFIX=$ROOT/prefix DISPLAY=:1
export WINEDLLOVERRIDES="mscoree,mshtml="     # NO d3d9=n -> builtin wined3d
W=$ROOT/wine/bin/wine
$W reg add "HKCU\\Software\\Wine\\Drivers" /v Audio /t REG_SZ /d "" /f >/dev/null 2>&1
$W reg add "HKCU\\Software\\Wine\\Explorer\\Desktops" /v Default /t REG_SZ /d 1920x1080 /f >/dev/null 2>&1
$W reg add "HKCU\\Software\\Wine\\AppDefaults\\League of Legends.exe\\Explorer" /v Desktop /t REG_SZ /d Default /f >/dev/null 2>&1
$ROOT/wine/bin/wineserver -k 2>/dev/null; sleep 3

export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
export LANERL_BOT=blue   # was the server-side default until 2026-09-10;
                         # DriveTeams now defaults to "none", so stating it
                         # here keeps this script doing what it always did
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
$DOTNET_ROOT/dotnet ./GameServerConsole.dll --config $NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json >"$OUT/server_wd.log" 2>&1 &
SRV=$!
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_wd.log" 2>/dev/null && break; sleep 2; done
echo "server ready"

cd "$D"
$W "League of Legends.exe" "8394" "LoLLauncher.exe" "" "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/client_wd.log" 2>&1 &
for i in $(seq 1 45); do grep -q "Accepted client version" "$OUT/server_wd.log" 2>/dev/null && { echo "CONNECTED at $((i*4))s"; break; }; sleep 4; done

for t in $(seq 1 8); do
    sleep 60
    ffmpeg -loglevel error -y -f x11grab -video_size 2560x1440 -i :1 -frames:v 1 "$OUT/shots/wd_$t.png" >/dev/null 2>&1
    echo "  ${t}m procs=$(pgrep -fc 'League of Legends' || echo 0) png=$(stat -c%s "$OUT/shots/wd_$t.png" 2>/dev/null)B crashes=$(grep -c c0000005 "$OUT/client_wd.log")"
done
echo "--- trace tail:"; grep "League of Legends.exe" "$OUT/client_wd.log" | tail -8
kill $SRV 2>/dev/null; $ROOT/wine/bin/wineserver -k 2>/dev/null
