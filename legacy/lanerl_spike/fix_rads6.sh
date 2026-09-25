#!/usr/bin/env bash
# Variant E: same as D, plus LEVELS/DATA pseudo-projects (RADS resolves the first
# path component as a project name, so LEVELS/Map1/... needs projects/LEVELS/).
# enumerate the 5GB asset tree. Variant C proved the structure works (RADS init
# succeeded, "past RADS? YES") but every file was refused as "not in the manifest"
# because we reused the patcher's manifest. rlsm.py now round-trips both real
# manifests byte-exactly, so we can emit correct ones.
# Version encoding: a.b.c.d -> (c<<8)|d  (0.0.0.227 -> 227, 0.0.0.11 -> 11).
set -uo pipefail
ROOT=/mnt/storage/lanerl
NFS=/mnt/nfs/projects
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
REL=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68
CLIENT=$REL/deploy
PROJ=$ROOT/client/extracted/RADS/projects
RLSM=$NFS/ahriuwu-lanerl/lanerl/rlsm.py

export WINEPREFIX="$ROOT/prefix" WINEDEBUG=-all
export WINEDLLOVERRIDES="mscoree,mshtml=;d3d9=n" DISPLAY=:99
WINE="$ROOT/wine/bin/wine"

cleanup(){ pkill -f "League of Legends.exe" 2>/dev/null; "$ROOT/wine/bin/wineserver" -k 2>/dev/null
           [ -n "${SRVPID:-}" ] && kill $SRVPID 2>/dev/null; [ -n "${XP:-}" ] && kill $XP 2>/dev/null; }
trap cleanup EXIT

echo "=== [1/4] projects + generated manifests over the real asset tree ==="
# RADS resolves "LEVELS/Map1/Mission.inibin" against a project literally named
# LEVELS, so it needs its own release dir (also writable, for its SOFT_REPAIR file).
# The client requests DATA at 0.0.1.7 (it mirrors lol_game_client's version) and
# LEVELS at 0.0.0.235 -- the assets were on disk all along, we just had the wrong
# release dir. Create both versions for each so whichever it asks for resolves.
for p in "lol_game_client/releases/0.0.1.7" "lol_game_client_en_gb/releases/0.0.0.235" \
         "LEVELS/releases/0.0.0.235" "LEVELS/releases/0.0.1.7" \
         "DATA/releases/0.0.0.235"   "DATA/releases/0.0.1.7"; do
    mkdir -p "$PROJ/$p"
    ln -sfn "$CLIENT" "$PROJ/$p/deploy"
done
python3 "$RLSM" build "$CLIENT" 1 263   "$PROJ/lol_game_client/releases/0.0.1.7/releasemanifest"
python3 "$RLSM" build "$CLIENT" 2 235   "$PROJ/lol_game_client_en_gb/releases/0.0.0.235/releasemanifest"
for q in LEVELS/releases/0.0.0.235 LEVELS/releases/0.0.1.7 DATA/releases/0.0.0.235 DATA/releases/0.0.1.7; do
    cp -f "$PROJ/lol_game_client/releases/0.0.1.7/releasemanifest" "$PROJ/$q/releasemanifest"
done
python3 "$RLSM" parse "$PROJ/lol_game_client/releases/0.0.1.7/releasemanifest" | head -2

echo "=== [2/4] Xvfb + server ==="
"$ROOT/env/bin/Xvfb" :99 -screen 0 1920x1080x24 >/dev/null 2>&1 & XP=$!
sleep 4
export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
export LANERL_BOT=blue   # was the server-side default until 2026-09-10;
                         # DriveTeams now defaults to "none", so stating it
                         # here keeps this script doing what it always did
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
"$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json" \
    >"$OUT/server_f.log" 2>&1 & SRVPID=$!
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_f.log" 2>/dev/null && break; sleep 2; done
echo "server ready"

echo "=== [3/4] client ==="
cd "$CLIENT"
timeout 320 "$WINE" "League of Legends.exe" "8394" "LoLLauncher.exe" "" \
    "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/rads_F.log" 2>&1 &
# DXVK is compiling shaders cold (no cache) and the client sits in
# r3dRenderLayer::SetMode for a long time on first run. Be patient and watch,
# rather than screenshotting at 60s and calling it black.
for t in $(seq 1 14); do
    sleep 60
    alive=$(pgrep -f "League of Legends.exe" >/dev/null && echo yes || echo NO)
    ffmpeg -loglevel error -y -f x11grab -video_size 1920x1080 -i :99 -frames:v 1 "$OUT/shots/F_$t.png" >/dev/null 2>&1
    # a black 1920x1080 png compresses to ~6KB; anything materially bigger has content
    sz=$(stat -c%s "$OUT/shots/F_$t.png" 2>/dev/null)
    echo "t=${t}m alive=$alive png=${sz}B srvlines=$(wc -l < "$OUT/server_f.log")"
    [ "$alive" = "NO" ] && { echo "client exited at ${t}m"; break; }
done

echo "=== [4/4] verdict ==="
echo "not-in-manifest: $(grep -c 'not in the manifest' "$OUT/rads_F.log")"
echo "loaded level?  $(grep -icE 'level|loadscreen|hud|champion' "$OUT/rads_F.log")"
grep -iE "level|spawn|hud|champion|connected" "$OUT/rads_F.log" 2>/dev/null | tail -8
echo "--- server:"; grep -vE "WARN|Could not find" "$OUT/server_f.log" | tail -6
