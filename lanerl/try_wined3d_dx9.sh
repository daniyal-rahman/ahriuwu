#!/usr/bin/env bash
# d3dx9_39 is now native (d3dx fixmes: many -> 0) but the stall is unchanged under
# DXVK. wined3d has never been tried WITH native d3dx9 on the real X server + GPU.
# Also capture the exit code and any BugSplat dump, which we keep losing.
set -uo pipefail
ROOT=/mnt/storage/lanerl; NFS=/mnt/nfs/projects; OUT=$NFS/ahriuwu-lanerl/lanerl/logs
D=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy
export WINEPREFIX=$ROOT/prefix DISPLAY=:1 DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
W=$ROOT/wine/bin/wine
$ROOT/wine/bin/wineserver -k 2>/dev/null; sleep 2

cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
$DOTNET_ROOT/dotnet ./GameServerConsole.dll --config $NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json >"$OUT/server_w3.log" 2>&1 &
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_w3.log" 2>/dev/null && break; sleep 2; done
echo "server ready"

cd "$D"
# wined3d this time: no d3d9=n. Keep d3dx9_39 native.
export WINEDLLOVERRIDES="mscoree,mshtml=;d3dx9_39=n"
( $W "League of Legends.exe" "8394" "LoLLauncher.exe" "" "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" \
    >"$OUT/client_w3.log" 2>&1; echo "EXIT=$?" >>"$OUT/client_w3.log" ) &
for i in $(seq 1 45); do grep -q "Accepted client version" "$OUT/server_w3.log" 2>/dev/null && { echo "CONNECTED"; break; }; sleep 4; done
for t in 1 2 3; do
    sleep 60
    ffmpeg -loglevel error -y -f x11grab -video_size 2560x1440 -i :1 -frames:v 1 "$OUT/shots/w3_$t.png" >/dev/null 2>&1
    echo "  ${t}m procs=$(pgrep -fc 'League of Legends' || echo 0) png=$(stat -c%s "$OUT/shots/w3_$t.png" 2>/dev/null)B"
done
echo "exit: $(grep -o 'EXIT=.*' "$OUT/client_w3.log" | tail -1)"
echo "--- trace:"; grep "League of Legends.exe" "$OUT/client_w3.log" | tail -6
echo "--- crash dump:"; python3 -c "
import json,glob,os
fs=sorted(glob.glob('$D/*crash*.json'),key=os.path.getmtime)
if fs:
    d=json.load(open(fs[-1])); print('  ',{k:d.get(k) for k in ('exception_type','module_name','access_addr')})
" 2>/dev/null
