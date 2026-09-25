#!/usr/bin/env bash
# NVIDIA Vulkan cannot make a presentation surface on Xvfb (no DRI3): it segfaults
# in vkGetPhysicalDeviceSurfaceCapabilitiesKHR (0xc0000005). Both options here avoid
# that path. Slower, but the goal right now is to SEE the game, not to be fast.
#   A: DXVK but with ONLY lavapipe visible (software Vulkan, valid surfaces)
#   B: no DXVK at all -- wine's builtin wined3d over OpenGL/llvmpipe.
#      (B crashed before, but that was at 1024x768 while the client asked for
#       1920x1080; the display now matches, so it deserves a retry.)
set -uo pipefail
ROOT=/mnt/storage/lanerl; NFS=/mnt/nfs/projects
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
CLIENT=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy
export WINEPREFIX="$ROOT/prefix" DISPLAY=:99
WINE="$ROOT/wine/bin/wine"
cleanup(){ pkill -f "League of Legends.exe" 2>/dev/null; "$ROOT/wine/bin/wineserver" -k 2>/dev/null
           [ -n "${SRVPID:-}" ] && kill $SRVPID 2>/dev/null; [ -n "${XP:-}" ] && kill $XP 2>/dev/null; }
trap cleanup EXIT

"$ROOT/env/bin/Xvfb" :99 -screen 0 1920x1080x24 >/dev/null 2>&1 & XP=$!; sleep 4
export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
export LANERL_BOT=blue   # was the server-side default until 2026-09-10;
                         # DriveTeams now defaults to "none", so stating it
                         # here keeps this script doing what it always did
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
"$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json" \
    >"$OUT/server_ab.log" 2>&1 & SRVPID=$!
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_ab.log" 2>/dev/null && break; sleep 2; done
echo "server ready"

run() {  # $1=label  $2=extra env assignments
    echo "=== $1 ==="
    pkill -f "League of Legends.exe" 2>/dev/null; sleep 3
    cd "$CLIENT"
    ( env $2 WINEPREFIX="$ROOT/prefix" DISPLAY=:99 WINEDLLOVERRIDES="mscoree,mshtml=;$3" \
      "$WINE" "League of Legends.exe" "8394" "LoLLauncher.exe" "" \
      "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/ab_$1.log" 2>&1; \
      echo "EXIT=$?" >>"$OUT/ab_$1.log" ) &
    for t in $(seq 1 12); do
        sleep 15
        a=$(pgrep -f "League of Legends.exe" >/dev/null && echo yes || echo NO)
        ffmpeg -loglevel error -y -f x11grab -video_size 1920x1080 -i :99 -frames:v 1 "$OUT/shots/$1_$t.png" >/dev/null 2>&1
        sz=$(stat -c%s "$OUT/shots/$1_$t.png" 2>/dev/null)
        echo "  $((t*15))s alive=$a png=${sz}B"
        [ "$a" = "NO" ] && break
    done
    echo "  surface-crash? $(grep -c '0xc0000005' "$OUT/ab_$1.log")  exit: $(grep -o 'EXIT=.*' "$OUT/ab_$1.log" | tail -1)"
}

run A "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json DXVK_FILTER_DEVICE_NAME=llvmpipe" "d3d9=n"
run B "" ""
