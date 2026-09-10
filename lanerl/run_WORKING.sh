#!/usr/bin/env bash
# WORKING CONFIG (crash-free, verified alive 70s+ with 0 access violations):
#   real Xorg :1              -- Xvfb has no DRI3; NVIDIA Vulkan segfaulted on it
#   per-app virtual desktop   -- display is 2560x1440 but the client demands a
#                                1920x1080 mode; fullscreen mode-set returns NULL
#                                and the game deref's it. A virtual desktop gives
#                                it a 1920x1080 window instead of mode-setting.
#   wineserver -k after reg   -- registry is cached; without this the setting is ignored
#   direct launch with argv   -- `explorer ... cmd /c x.bat` swallowed args (no connect)
#   DXVK                      -- reaches the 5080 via /dev/nvidia*
set -uo pipefail
DUR=${1:-600}
ROOT=/mnt/storage/lanerl; NFS=/mnt/nfs/projects
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
CLIENT=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy
export WINEPREFIX="$ROOT/prefix" DISPLAY=:1
export WINEDLLOVERRIDES="mscoree,mshtml=;d3d9=n"
export DXVK_STATE_CACHE_PATH="$ROOT/dxvk_cache"
WINE="$ROOT/wine/bin/wine"; mkdir -p "$OUT/shots"
cleanup(){ pkill -f "League of Legends" 2>/dev/null; "$ROOT/wine/bin/wineserver" -k 2>/dev/null
           [ -n "${SRVPID:-}" ] && kill $SRVPID 2>/dev/null; }
trap cleanup EXIT

"$WINE" reg add "HKCU\\Software\\Wine\\Explorer\\Desktops" /v Default /t REG_SZ /d 1920x1080 /f >/dev/null 2>&1
"$WINE" reg add "HKCU\\Software\\Wine\\AppDefaults\\League of Legends.exe\\Explorer" /v Desktop /t REG_SZ /d Default /f >/dev/null 2>&1
"$ROOT/wine/bin/wineserver" -k 2>/dev/null; sleep 3

export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
"$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json" \
    >"$OUT/server_w.log" 2>&1 & SRVPID=$!
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_w.log" 2>/dev/null && break; sleep 2; done
echo "server ready"

cd "$CLIENT"
"$WINE" "League of Legends.exe" "8394" "LoLLauncher.exe" "" \
  "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/client_w.log" 2>&1 &

for i in $(seq 1 45); do grep -q "Accepted client version" "$OUT/server_w.log" 2>/dev/null && { echo "CONNECTED at $((i*4))s"; break; }; sleep 4; done
for i in $(seq 1 60); do grep -q "Patience is over" "$OUT/server_w.log" 2>/dev/null && { echo "GAME STARTED"; break; }; sleep 3; done
sleep 20

echo "=== recording ${DUR}s at 2560x1440 (the real display size) ==="
ffmpeg -loglevel error -y -f x11grab -framerate 10 -video_size 2560x1440 -i :1 -t "$DUR" \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p "$OUT/gameplay_final.mp4" >/dev/null 2>&1 &
FF=$!
for t in $(seq 1 $((DUR/60))); do
    sleep 60
    ffmpeg -loglevel error -y -f x11grab -video_size 2560x1440 -i :1 -frames:v 1 "$OUT/shots/w_$t.png" >/dev/null 2>&1
    echo "  ${t}m procs=$(pgrep -fc 'League of Legends' || echo 0) png=$(stat -c%s "$OUT/shots/w_$t.png" 2>/dev/null)B"
done
wait $FF 2>/dev/null
echo "mp4: $(ls -lh "$OUT/gameplay_final.mp4" 2>/dev/null | awk '{print $5}')  crashes: $(grep -c c0000005 "$OUT/client_w.log")"
grep -vE "WARN|Could not find" "$OUT/server_w.log" | tail -4
