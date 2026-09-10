#!/usr/bin/env bash
# Retry the whole pipeline WITH GPU device access (nested sg for video+render).
# Hypothesis: the vkGetPhysicalDeviceSurfaceCapabilitiesKHR segfault was caused by
# the /dev/dri permission denial that appeared alongside it, not by Xvfb per se.
# If so, DXVK can now create a real presentation surface and the 3D scene renders.
set -uo pipefail
DUR=${1:-600}
ROOT=/mnt/storage/lanerl; NFS=/mnt/nfs/projects
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
CLIENT=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy
export WINEPREFIX="$ROOT/prefix" DISPLAY=:99
export WINEDLLOVERRIDES="mscoree,mshtml=;d3d9=n"
export DXVK_STATE_CACHE_PATH="$ROOT/dxvk_cache"; mkdir -p "$ROOT/dxvk_cache"
WINE="$ROOT/wine/bin/wine"; mkdir -p "$OUT/shots"
cleanup(){ pkill -f "League of Legends.exe" 2>/dev/null; pkill -f explorer 2>/dev/null
           "$ROOT/wine/bin/wineserver" -k 2>/dev/null
           [ -n "${SRVPID:-}" ] && kill $SRVPID 2>/dev/null; [ -n "${XP:-}" ] && kill $XP 2>/dev/null; }
trap cleanup EXIT

echo "gpu access: $(test -r /dev/dri/renderD128 && echo YES || echo no)"
printf '@echo off\r\n"League of Legends.exe" "8394" "LoLLauncher.exe" "" "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1"\r\n' > "$CLIENT/lanerl_launch.bat"

"$ROOT/env/bin/Xvfb" :99 -screen 0 1920x1080x24 +extension GLX +extension RANDR +render >/dev/null 2>&1 & XP=$!; sleep 4
export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
export LANERL_BOT=blue   # was the server-side default until 2026-09-10;
                         # DriveTeams now defaults to "none", so stating it
                         # here keeps this script doing what it always did
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
"$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json" \
    >"$OUT/server_g.log" 2>&1 & SRVPID=$!
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_g.log" 2>/dev/null && break; sleep 2; done
echo "server ready"

cd "$CLIENT"
"$WINE" explorer /desktop=lol,1920x1080 cmd /c lanerl_launch.bat >"$OUT/client_g.log" 2>&1 &

for i in $(seq 1 45); do grep -q "Accepted client version" "$OUT/server_g.log" 2>/dev/null && { echo "CONNECTED at $((i*4))s"; break; }; sleep 4; done
for i in $(seq 1 60); do grep -q "Patience is over" "$OUT/server_g.log" 2>/dev/null && { echo "GAME STARTED"; break; }; sleep 3; done

ffmpeg -loglevel error -y -f x11grab -framerate 10 -video_size 1920x1080 -i :99 -t "$DUR" \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p "$OUT/gameplay_final.mp4" >/dev/null 2>&1 &
FF=$!
for t in $(seq 1 $((DUR/60))); do
    sleep 60
    ffmpeg -loglevel error -y -f x11grab -video_size 1920x1080 -i :99 -frames:v 1 "$OUT/shots/gpu_$t.png" >/dev/null 2>&1
    echo "  ${t}m alive=$(pgrep -f 'League of Legends.exe' >/dev/null && echo yes || echo NO) png=$(stat -c%s "$OUT/shots/gpu_$t.png" 2>/dev/null)B"
done
wait $FF 2>/dev/null
echo "mp4: $(ls -lh "$OUT/gameplay_final.mp4" 2>/dev/null | awk '{print $5}')"
echo "surface segfaults: $(grep -c c0000005 "$OUT/client_g.log")"
