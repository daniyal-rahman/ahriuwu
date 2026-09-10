#!/usr/bin/env bash
# Variant D: fabricated RADS projects + GENERATED releasemanifests that actually
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

echo "=== [1/4] generate manifests over the real asset tree ==="
python3 "$RLSM" build "$CLIENT" 1 263   "$PROJ/lol_game_client/releases/0.0.1.7/releasemanifest"
python3 "$RLSM" build "$CLIENT" 2 235   "$PROJ/lol_game_client_en_gb/releases/0.0.0.235/releasemanifest"
python3 "$RLSM" parse "$PROJ/lol_game_client/releases/0.0.1.7/releasemanifest" | head -2

echo "=== [2/4] Xvfb + server ==="
"$ROOT/env/bin/Xvfb" :99 -screen 0 1920x1080x24 >/dev/null 2>&1 & XP=$!
sleep 4
export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
"$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json" \
    >"$OUT/server_d.log" 2>&1 & SRVPID=$!
for i in $(seq 1 40); do grep -q "Game is ready" "$OUT/server_d.log" 2>/dev/null && break; sleep 2; done
echo "server ready"

echo "=== [3/4] client ==="
cd "$CLIENT"
timeout 320 "$WINE" "League of Legends.exe" "8394" "LoLLauncher.exe" "" \
    "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/rads_D.log" 2>&1 &
sleep 60
echo "not-in-manifest errors: $(grep -c 'not in the manifest' "$OUT/rads_D.log" 2>/dev/null)"
echo "waiting for game start ..."
for i in $(seq 1 60); do grep -q "Patience is over" "$OUT/server_d.log" 2>/dev/null && { echo "GAME STARTED"; break; }; sleep 3; done
sleep 20
for t in 1 2 3; do
    ffmpeg -loglevel error -y -f x11grab -video_size 1920x1080 -i :99 -frames:v 1 "$OUT/shots/D_$t.png" >/dev/null 2>&1
    sleep 20
done

echo "=== [4/4] verdict ==="
echo "not-in-manifest: $(grep -c 'not in the manifest' "$OUT/rads_D.log")"
echo "loaded level?  $(grep -icE 'level|loadscreen|hud|champion' "$OUT/rads_D.log")"
grep -iE "level|spawn|hud|champion|connected" "$OUT/rads_D.log" 2>/dev/null | tail -8
echo "--- server:"; grep -vE "WARN|Could not find" "$OUT/server_d.log" | tail -6
