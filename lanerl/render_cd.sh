#!/usr/bin/env bash
# All 3 render paths died with a NULL-pointer read right after d3d device creation.
# Xvfb has no window manager and a minimal RandR/mode list, and this 2014 client
# queries monitor info then dereferences the result unchecked. wine's built-in
# virtual desktop (explorer /desktop=) supplies a proper managed desktop window
# and a sane mode -- the standard fix for games that misbehave with no WM.
set -uo pipefail
ROOT=/mnt/storage/lanerl; NFS=/mnt/nfs/projects
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
CLIENT=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy
export WINEPREFIX="$ROOT/prefix" DISPLAY=:99
WINE="$ROOT/wine/bin/wine"
cleanup(){ pkill -f "League of Legends.exe" 2>/dev/null; pkill -f explorer 2>/dev/null
           "$ROOT/wine/bin/wineserver" -k 2>/dev/null
           [ -n "${SRVPID:-}" ] && kill $SRVPID 2>/dev/null; [ -n "${XP:-}" ] && kill $XP 2>/dev/null; }
trap cleanup EXIT

"$ROOT/env/bin/Xvfb" :99 -screen 0 1920x1080x24 +extension GLX +extension RANDR +render >/dev/null 2>&1 & XP=$!; sleep 4
export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
export LANERL_BOT=blue   # was the server-side default until 2026-09-10;
                         # DriveTeams now defaults to "none", so stating it
                         # here keeps this script doing what it always did
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
"$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json" \
    >"$OUT/server_cd.log" 2>&1 & SRVPID=$!
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_cd.log" 2>/dev/null && break; sleep 2; done
echo "server ready"

run() {  # $1 label  $2 dll overrides
    echo "=== $1 (virtual desktop, overrides='$2') ==="
    pkill -f "League of Legends.exe" 2>/dev/null; pkill -f explorer 2>/dev/null; sleep 3
    cd "$CLIENT"
    ( WINEDLLOVERRIDES="mscoree,mshtml=;$2" "$WINE" explorer /desktop=lol,1920x1080 \
        "League of Legends.exe" "8394" "LoLLauncher.exe" "" \
        "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/cd_$1.log" 2>&1; \
      echo "EXIT=$?" >>"$OUT/cd_$1.log" ) &
    for t in $(seq 1 10); do
        sleep 20
        a=$(pgrep -f "League of Legends.exe" >/dev/null && echo yes || echo NO)
        ffmpeg -loglevel error -y -f x11grab -video_size 1920x1080 -i :99 -frames:v 1 "$OUT/shots/$1_$t.png" >/dev/null 2>&1
        echo "  $((t*20))s alive=$a png=$(stat -c%s "$OUT/shots/$1_$t.png" 2>/dev/null)B"
        [ "$a" = "NO" ] && break
    done
    echo "  exit: $(grep -o 'EXIT=.*' "$OUT/cd_$1.log" | tail -1)  nullderef: $(grep -c 'c0000005' "$OUT/cd_$1.log")"
}
run C "d3d9=n"   # DXVK
run D ""         # wined3d
