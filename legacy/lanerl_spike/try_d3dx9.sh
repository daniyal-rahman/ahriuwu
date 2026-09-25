#!/usr/bin/env bash
# DEAD LINE OF WORK (marked 2026-09-12). Every hypothesis in this file is about
# getting the real 4.20 rendering client to draw. That approach was abandoned:
# the headless LeagueSandbox sim is the verified RL environment. None of the
# hypotheses below was ever confirmed or refuted -- they are orphaned, not open.
# Do not spend time chasing them; read them as a record of what was tried.
# The client is probably NOT incomplete: a private-server 4.20 install launches from
# solutions/.../deploy and does not ship RADS/projects/lol_game_client. The documented
# cause of "connects but never renders" is a missing d3dx9_39.dll -- consistent with
# our stall at r3dRenderLayer::SetMode: Initializing and with wine's builtin d3dx9
# stub logging "d3dx_load_pixels_from_pixels Unhandled filter 0x3".
set -uo pipefail
ROOT=/mnt/storage/lanerl; NFS=/mnt/nfs/projects; OUT=$NFS/ahriuwu-lanerl/lanerl/logs
D=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy
export WINEPREFIX=$ROOT/prefix DISPLAY=:1
export WINEDLLOVERRIDES="mscoree,mshtml=;d3d9=n;d3dx9_39=n"
export DXVK_STATE_CACHE_PATH=$ROOT/dxvk_cache DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
export LANERL_BOT=blue   # was the server-side default until 2026-09-10;
                         # DriveTeams now defaults to "none", so stating it
                         # here keeps this script doing what it always did
export PATH="$ROOT/wine/bin:$ROOT/env/bin:$PATH" WINE=$ROOT/wine/bin/wine
W=$ROOT/wine/bin/wine

echo "=== install d3dx9 via winetricks (no root) ==="
mkdir -p $ROOT/tools && cd $ROOT/tools
[ -x winetricks ] || { curl -sSL -o winetricks https://raw.githubusercontent.com/Winetricks/winetricks/master/src/winetricks; chmod +x winetricks; }
# winetricks needs cabextract to unpack the DirectX redist
command -v cabextract >/dev/null || /home/dani/miniconda3/bin/conda install -y -p $ROOT/env -c conda-forge cabextract 2>&1 | tail -2
WINE=$W ./winetricks -q d3dx9 2>&1 | tail -8
echo "--- d3dx9_39 present?"; ls -la $WINEPREFIX/drive_c/windows/syswow64/d3dx9_39.dll 2>/dev/null || echo "  MISSING"

echo "=== relaunch client ==="
$ROOT/wine/bin/wineserver -k 2>/dev/null; sleep 2
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
$DOTNET_ROOT/dotnet ./GameServerConsole.dll --config $NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json >"$OUT/server_dx9.log" 2>&1 &
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_dx9.log" 2>/dev/null && break; sleep 2; done
cd "$D"
$W "League of Legends.exe" "8394" "LoLLauncher.exe" "" "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/client_dx9.log" 2>&1 &
for i in $(seq 1 45); do grep -q "Accepted client version" "$OUT/server_dx9.log" 2>/dev/null && { echo "CONNECTED"; break; }; sleep 4; done
for t in 1 2 3 4 5; do
    sleep 60
    ffmpeg -loglevel error -y -f x11grab -video_size 2560x1440 -i :1 -frames:v 1 "$OUT/shots/dx9_$t.png" >/dev/null 2>&1
    echo "  ${t}m procs=$(pgrep -fc 'League of Legends' || echo 0) png=$(stat -c%s "$OUT/shots/dx9_$t.png" 2>/dev/null)B"
done
echo "--- past SetMode?"; grep "League of Legends.exe" "$OUT/client_dx9.log" | tail -5
