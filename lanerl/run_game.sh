#!/usr/bin/env bash
# Start the server, connect the REAL 4.20 client to it under a virtual display,
# and record what actually happens. Duration via $1 (seconds, default 180).
#
# Note on rendering: Xvfb gives GLX via llvmpipe (software), so wined3d may be slow.
# If fps is unusable we add DXVK so d3d9 goes over NVIDIA Vulkan instead -- Vulkan
# needs X only to present, not to render, so it can use the 5080 under Xvfb.
set -uo pipefail
DUR=${1:-180}
ROOT=/mnt/storage/lanerl
NFS=/mnt/nfs/projects
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
SRV=$NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
CLIENT="$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy"
mkdir -p "$OUT" "$OUT/shots"

export WINEPREFIX="$ROOT/prefix"
export WINEDEBUG=-all
# d3d9=n -> use DXVK's native d3d9.dll instead of wine's builtin wined3d.
# Why: wined3d goes d3d9->OpenGL, and under Xvfb that resolves to Mesa llvmpipe
# (software) because /dev/dri is permission-denied inside the Slurm cgroup --
# the 4.20 client logged 'gpu: llvmpipe' then BugSplat'd creating the device.
# DXVK goes d3d9->Vulkan->nvidia_icd, which uses /dev/nvidia* and can reach the 5080.
export WINEDLLOVERRIDES="mscoree,mshtml=;d3d9=n"
export DISPLAY=:99
export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
export LANERL_BOT=blue   # was the server-side default until 2026-09-10;
                         # DriveTeams now defaults to "none", so stating it
                         # here keeps this script doing what it always did
WINE="$ROOT/wine/bin/wine"

cleanup() {
    pkill -f "League of Legends.exe" 2>/dev/null
    "$ROOT/wine/bin/wineserver" -k 2>/dev/null
    [ -n "${SRV_PID:-}" ] && kill "$SRV_PID" 2>/dev/null
    [ -n "${XVFB_PID:-}" ] && kill "$XVFB_PID" 2>/dev/null
}
trap cleanup EXIT

echo "=== [0/5] DXVK d3d9 (32-bit) into the prefix ==="
DXVK=$NFS/lanerl-vendor/dxvk-2.7/x32/d3d9.dll
if [ -f "$DXVK" ]; then
    cp -f "$DXVK" "$WINEPREFIX/drive_c/windows/syswow64/d3d9.dll" && echo "dxvk d3d9 installed"
else
    echo "DXVK MISSING at $DXVK -- will fall back to wined3d/llvmpipe"
fi

# The client asks for 1920x1080 (seen in its own log); giving it a smaller
# framebuffer than it resizes to is a likely cause of the device-creation crash.
echo "=== [1/5] Xvfb 1920x1080 ==="
"$ROOT/env/bin/Xvfb" :99 -screen 0 1920x1080x24 >/dev/null 2>&1 &
XVFB_PID=$!; sleep 4
kill -0 $XVFB_PID 2>/dev/null && echo "Xvfb up" || { echo "Xvfb DIED"; exit 1; }

echo "=== [2/5] game server ==="
cd "$SRV"
"$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json" \
    >"$OUT/server.log" 2>&1 &
SRV_PID=$!
for i in $(seq 1 40); do
    grep -q "Game is ready" "$OUT/server.log" 2>/dev/null && break
    sleep 2
done
grep -q "Game is ready" "$OUT/server.log" && echo "server ready ($(grep -c . "$OUT/server.log") log lines)" || { echo "SERVER NOT READY"; tail -5 "$OUT/server.log"; exit 1; }

echo "=== [3/5] launch client ==="
cd "$CLIENT"
timeout $((DUR + 120)) "$WINE" "League of Legends.exe" "8394" "LoLLauncher.exe" "" \
    "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/client.log" 2>&1 &
sleep 45
pgrep -f "League of Legends.exe" >/dev/null && echo "client process alive" || echo "CLIENT DIED (see client.log)"

# Wait for the game to ACTUALLY start before recording, so the video is gameplay
# and not 2 minutes of loading screen. forcedStart is 120s (the client needs ~60s
# to connect, and DXVK shader compilation makes that slower, not faster -- at
# forcedStart=5 the server force-started, saw no client, and exited before the
# client finished loading: 'Unable to connect to the server').
echo "--- waiting for game start ..."
for i in $(seq 1 90); do
    grep -q "Patience is over" "$OUT/server.log" 2>/dev/null && { echo "GAME STARTED at ~$((i*2))s"; break; }
    grep -q "All players have left" "$OUT/server.log" 2>/dev/null && { echo "SERVER EXITED EARLY"; break; }
    sleep 2
done

echo "=== [4/5] record ${DUR}s ==="
ffmpeg -loglevel error -y -f x11grab -framerate 10 -video_size 1920x1080 -i :99 \
    -t "$DUR" -c:v libx264 -preset veryfast -pix_fmt yuv420p "$OUT/gameplay.mp4" \
    >"$OUT/ffmpeg.log" 2>&1 &
FF_PID=$!
# stills every 30s so we still have evidence if the video is unusable
for i in $(seq 1 $((DUR / 30))); do
    sleep 30
    ffmpeg -loglevel error -y -f x11grab -video_size 1920x1080 -i :99 -frames:v 1 \
        "$OUT/shots/t$(printf %03d $((i * 30))).png" >/dev/null 2>&1
done
wait $FF_PID 2>/dev/null

echo "=== [5/5] results ==="
ls -lh "$OUT/gameplay.mp4" 2>/dev/null
echo "--- stills: $(ls "$OUT/shots"/*.png 2>/dev/null | wc -l)"
echo "--- server log tail:"; grep -vE "WARN|Could not find" "$OUT/server.log" 2>/dev/null | tail -12
echo "--- minion/gold/damage evidence:"
grep -ciE "minion" "$OUT/server.log" 2>/dev/null
echo "--- client log tail:"; tail -5 "$OUT/client.log" 2>/dev/null
