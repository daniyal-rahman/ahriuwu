#!/usr/bin/env bash
# Variant C: a STRUCTURALLY CORRECT minimal solutionmanifest.
# Format (decoded from the original 172-line file):
#   line1  "RADS Solution Manifest"
#   line2  format version
#   line3  solution name
#   line4  solution version
#   line5  project count N   -> then N x 4 lines: name, version, flagA, flagB
#   then   locale count M    -> then M x 5 lines: locale, 0, 2, proj, localeproj
# Variant B failed with "ReadDWORD: Could not convert string to number" because it
# omitted the whole locale block, so the parser ran off the end of the file.
set -uo pipefail
ROOT=/mnt/storage/lanerl
NFS=/mnt/nfs/projects
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
REL=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68
CLIENT=$REL/deploy
PROJ=$ROOT/client/extracted/RADS/projects
SRC_MANIFEST=$PROJ/lol_patcher/releases/0.0.0.11/releasemanifest

export WINEPREFIX="$ROOT/prefix" WINEDEBUG=-all
export WINEDLLOVERRIDES="mscoree,mshtml=;d3d9=n" DISPLAY=:99
WINE="$ROOT/wine/bin/wine"

cleanup(){ pkill -f "League of Legends.exe" 2>/dev/null; "$ROOT/wine/bin/wineserver" -k 2>/dev/null; [ -n "${XP:-}" ] && kill $XP 2>/dev/null; }
trap cleanup EXIT

# both projects the manifest will declare, each pointing at the real assets
for p in "lol_game_client/releases/0.0.1.7" "lol_game_client_en_gb/releases/0.0.0.235"; do
    mkdir -p "$PROJ/$p"
    ln -sfn "$CLIENT" "$PROJ/$p/deploy"
    cp -f "$SRC_MANIFEST" "$PROJ/$p/releasemanifest" 2>/dev/null
done
echo "projects now: $(ls $PROJ)"

cat > "$REL/solutionmanifest" <<'EOF'
RADS Solution Manifest
1.0.0.0
lol_game_client_sln
0.0.1.68
2
lol_game_client
0.0.1.7
0
0
lol_game_client_en_gb
0.0.0.235
10
0
1
en_gb
0
2
lol_game_client
lol_game_client_en_gb
EOF

"$ROOT/env/bin/Xvfb" :99 -screen 0 1920x1080x24 >/dev/null 2>&1 & XP=$!
sleep 4

echo "=== start server (so the client has something to join) ==="
export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
export LANERL_BOT=blue   # was the server-side default until 2026-09-10;
                         # DriveTeams now defaults to "none", so stating it
                         # here keeps this script doing what it always did
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
"$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json" \
    >"$OUT/server_c.log" 2>&1 &
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_c.log" 2>/dev/null && break; sleep 2; done
echo "server: $(grep -c 'Game is ready' "$OUT/server_c.log") ready"

echo "=== variant C ==="
cd "$CLIENT"
timeout 200 "$WINE" "League of Legends.exe" "8394" "LoLLauncher.exe" "" \
    "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/rads_C.log" 2>&1 &
sleep 150
ffmpeg -loglevel error -y -f x11grab -video_size 1920x1080 -i :99 -frames:v 1 "$OUT/shots/rads_C.png" >/dev/null 2>&1
echo "RADS errors: $(grep -c 'RADS' "$OUT/rads_C.log" 2>/dev/null)"
grep -m3 'ERROR' "$OUT/rads_C.log" 2>/dev/null | cut -c1-150
echo "past RADS? $(grep -qiE 'level|loadlevel|gamestate|spawn|hud' "$OUT/rads_C.log" && echo YES || echo no)"
echo "--- last lines:"; tail -5 "$OUT/rads_C.log"
