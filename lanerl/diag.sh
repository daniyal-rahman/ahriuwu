#!/usr/bin/env bash
# Why does the client exit ~60s in, right after "Creating d3d device"?
# Also fix a real bug: RADS looks up "Data/Menu/IMEConfig.xml" but the on-disk
# dir is "DATA", and manifest lookup is case-sensitive even though wine's file
# layer is not -- so add case-variant symlinks and re-generate the manifests.
set -uo pipefail
ROOT=/mnt/storage/lanerl; NFS=/mnt/nfs/projects
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
REL=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68
CLIENT=$REL/deploy; PROJ=$ROOT/client/extracted/RADS/projects
RLSM=$NFS/ahriuwu-lanerl/lanerl/rlsm.py

export WINEPREFIX="$ROOT/prefix" WINEDLLOVERRIDES="mscoree,mshtml=;d3d9=n" DISPLAY=:99
export DXVK_LOG_LEVEL=info DXVK_STATE_CACHE_PATH="$ROOT/dxvk_cache"
mkdir -p "$ROOT/dxvk_cache"
WINE="$ROOT/wine/bin/wine"

cleanup(){ pkill -f "League of Legends.exe" 2>/dev/null; "$ROOT/wine/bin/wineserver" -k 2>/dev/null
           [ -n "${SRVPID:-}" ] && kill $SRVPID 2>/dev/null; [ -n "${XP:-}" ] && kill $XP 2>/dev/null; }
trap cleanup EXIT

echo "=== case-variant symlinks so RADS's case-sensitive lookup resolves ==="
cd "$CLIENT"
for d in DATA LEVELS; do
    lower=$(echo "$d" | tr 'A-Z' 'a-z'); title=$(echo "$d" | sed 's/^\(.\)\(.*\)/\1\L\2/')
    [ -e "$title" ] || ln -sfn "$d" "$title"
    [ -e "$lower" ] || ln -sfn "$d" "$lower"
done
ls -d DATA Data data LEVELS Levels levels 2>/dev/null | tr '\n' ' '; echo

echo "=== regenerate manifests (now include the case variants) ==="
python3 "$RLSM" build "$CLIENT" 1 263 "$PROJ/lol_game_client/releases/0.0.1.7/releasemanifest"
for q in lol_game_client_en_gb/releases/0.0.0.235 LEVELS/releases/0.0.0.235 \
         LEVELS/releases/0.0.1.7 DATA/releases/0.0.0.235 DATA/releases/0.0.1.7; do
    cp -f "$PROJ/lol_game_client/releases/0.0.1.7/releasemanifest" "$PROJ/$q/releasemanifest"
done

"$ROOT/env/bin/Xvfb" :99 -screen 0 1920x1080x24 >/dev/null 2>&1 & XP=$!; sleep 4
export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
"$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json" \
    >"$OUT/server_g.log" 2>&1 & SRVPID=$!
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_g.log" 2>/dev/null && break; sleep 2; done

echo "=== client (capturing exit code) ==="
cd "$CLIENT"
( "$WINE" "League of Legends.exe" "8394" "LoLLauncher.exe" "" \
    "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/diag_client.log" 2>&1; \
  echo "CLIENT EXIT CODE: $?" >>"$OUT/diag_client.log" ) &
# sample every 10s to catch a crash dialog before it disappears
for t in $(seq 1 18); do
    sleep 10
    a=$(pgrep -f "League of Legends.exe" >/dev/null && echo yes || echo NO)
    ffmpeg -loglevel error -y -f x11grab -video_size 1920x1080 -i :99 -frames:v 1 "$OUT/shots/g_$t.png" >/dev/null 2>&1
    echo "  ${t}0s alive=$a png=$(stat -c%s "$OUT/shots/g_$t.png" 2>/dev/null)B"
    [ "$a" = "NO" ] && break
done
echo "=== exit / last lines ==="
grep -E "CLIENT EXIT CODE" "$OUT/diag_client.log" 2>/dev/null
echo "not-in-manifest now: $(grep -c 'not in the manifest' "$OUT/diag_client.log" 2>/dev/null)"
tail -12 "$OUT/diag_client.log"
