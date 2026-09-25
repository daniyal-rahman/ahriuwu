#!/usr/bin/env bash
# The repack ships 5GB of assets under solutions/.../deploy but NO RADS project
# tree, so RADS init fails and the client renders black. solutionmanifest is
# plain text declaring 19 projects that don't exist -- so try, in order:
#   A) declare 0 projects
#   B) declare only lol_game_client, with a project dir symlinked to the assets
# Each variant runs the client for 60s and we diff the RADS errors.
set -uo pipefail
ROOT=/mnt/storage/lanerl
NFS=/mnt/nfs/projects
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
REL=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68
CLIENT=$REL/deploy
PROJ=$ROOT/client/extracted/RADS/projects

export WINEPREFIX="$ROOT/prefix" WINEDEBUG=-all
export WINEDLLOVERRIDES="mscoree,mshtml=;d3d9=n" DISPLAY=:99
WINE="$ROOT/wine/bin/wine"

[ -f "$REL/solutionmanifest.orig" ] || cp "$REL/solutionmanifest" "$REL/solutionmanifest.orig"

cleanup(){ pkill -f "League of Legends.exe" 2>/dev/null; "$ROOT/wine/bin/wineserver" -k 2>/dev/null; [ -n "${XP:-}" ] && kill $XP 2>/dev/null; }
trap cleanup EXIT

"$ROOT/env/bin/Xvfb" :99 -screen 0 1920x1080x24 >/dev/null 2>&1 & XP=$!
sleep 4

try_variant() {
    local name=$1 log="$OUT/rads_$1.log"
    echo "--- variant $name"
    pkill -f "League of Legends.exe" 2>/dev/null; sleep 2
    cd "$CLIENT"
    timeout 75 "$WINE" "League of Legends.exe" "8394" "LoLLauncher.exe" "" \
        "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$log" 2>&1 &
    sleep 55
    ffmpeg -loglevel error -y -f x11grab -video_size 1920x1080 -i :99 -frames:v 1 \
        "$OUT/shots/rads_$name.png" >/dev/null 2>&1
    echo "    RADS errors: $(grep -c 'RADS' "$log" 2>/dev/null)"
    echo "    first RADS err: $(grep -m1 'RADS.*ERROR\|ERROR.*RADS' "$log" 2>/dev/null | cut -c1-140)"
    echo "    got past RADS? $(grep -qE 'Loading Level|LoadLevel|GameState|Spawn' "$log" && echo YES || echo no)"
    pkill -f "League of Legends.exe" 2>/dev/null
}

echo "=== A: declare 0 projects ==="
{ echo "RADS Solution Manifest"; echo "1.0.0.0"; echo "lol_game_client_sln"; echo "0.0.1.68"; echo "0"; } > "$REL/solutionmanifest"
try_variant A

echo "=== B: declare lol_game_client only, project dir -> assets ==="
mkdir -p "$PROJ/lol_game_client/releases/0.0.1.7"
ln -sfn "$CLIENT" "$PROJ/lol_game_client/releases/0.0.1.7/deploy"
# no authentic releasemanifest exists for this project; reuse one so the file parses
cp -f "$PROJ/lol_patcher/releases/0.0.0.11/releasemanifest" "$PROJ/lol_game_client/releases/0.0.1.7/releasemanifest" 2>/dev/null
{ echo "RADS Solution Manifest"; echo "1.0.0.0"; echo "lol_game_client_sln"; echo "0.0.1.68"; echo "1";
  echo "lol_game_client"; echo "0.0.1.7"; echo "0"; echo "0"; } > "$REL/solutionmanifest"
try_variant B

cp -f "$REL/solutionmanifest.orig" "$REL/solutionmanifest"
echo "=== restored original manifest ==="
