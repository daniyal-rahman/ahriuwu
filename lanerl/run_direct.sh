#!/usr/bin/env bash
# Real Xorg (:1) killed the Vulkan surface segfault (0 occurrences). The wine
# virtual desktop was only a workaround for Xvfb having no WM/valid monitor info,
# and routing through `cmd /c a.bat` swallowed the game's stdout so the crash was
# invisible. On a real X server, launch the exe directly: proper args (connects),
# full logging (so we can see where the null-deref happens).
set -uo pipefail
DUR=${1:-600}
ROOT=/mnt/storage/lanerl; NFS=/mnt/nfs/projects
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
CLIENT=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy
export WINEPREFIX="$ROOT/prefix" DISPLAY=:1
export WINEDLLOVERRIDES="mscoree,mshtml=;d3d9=n"
export DXVK_STATE_CACHE_PATH="$ROOT/dxvk_cache"
WINE="$ROOT/wine/bin/wine"; mkdir -p "$OUT/shots"
cleanup(){ pkill -f "League of Legends.exe" 2>/dev/null; "$ROOT/wine/bin/wineserver" -k 2>/dev/null
           [ -n "${SRVPID:-}" ] && kill $SRVPID 2>/dev/null; }
trap cleanup EXIT

# turn OFF the registry virtual desktop left over from an earlier attempt
"$WINE" reg delete 'HKCU\Software\Wine\Explorer' /v Desktop /f >/dev/null 2>&1
# no audio device here; a 2014 title initialising DirectSound is a crash candidate
"$WINE" reg add 'HKCU\Software\Wine\Drivers' /v Audio /t REG_SZ /d "" /f >/dev/null 2>&1

export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
"$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json" \
    >"$OUT/server_dx.log" 2>&1 & SRVPID=$!
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_dx.log" 2>/dev/null && break; sleep 2; done
echo "server ready"

cd "$CLIENT"
( "$WINE" "League of Legends.exe" "8394" "LoLLauncher.exe" "" \
  "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/client_dx.log" 2>&1; echo "EXIT=$?" >>"$OUT/client_dx.log" ) &

for i in $(seq 1 45); do grep -q "Accepted client version" "$OUT/server_dx.log" 2>/dev/null && { echo "CONNECTED at $((i*4))s"; break; }; sleep 4; done
for i in $(seq 1 60); do grep -q "Patience is over" "$OUT/server_dx.log" 2>/dev/null && { echo "GAME STARTED"; break; }; sleep 3; done

ffmpeg -loglevel error -y -f x11grab -framerate 10 -video_size 1920x1080 -i :1 -t "$DUR" \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p "$OUT/gameplay_final.mp4" >/dev/null 2>&1 &
FF=$!
for t in $(seq 1 $((DUR/60))); do
    sleep 60
    ffmpeg -loglevel error -y -f x11grab -video_size 1920x1080 -i :1 -frames:v 1 "$OUT/shots/dx_$t.png" >/dev/null 2>&1
    echo "  ${t}m alive=$(pgrep -f 'League of Legends.exe' >/dev/null && echo yes || echo NO) png=$(stat -c%s "$OUT/shots/dx_$t.png" 2>/dev/null)B"
done
wait $FF 2>/dev/null
echo "mp4: $(ls -lh "$OUT/gameplay_final.mp4" 2>/dev/null | awk '{print $5}')"
echo "--- last engine lines:"; grep "League of Legends.exe" "$OUT/client_dx.log" | tail -12
