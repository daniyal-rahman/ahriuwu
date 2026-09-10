#!/usr/bin/env bash
set -euo pipefail          # before anything runs; see the note above the banner
# Virtual desktop STOPS the null-deref crash (proven: alive 200s vs dying at 10-60s).
# But passing the game's args through `explorer /desktop=` swallowed them, so the
# client never connected (server saw 2/2 disconnected). Set the virtual desktop in
# the registry instead, then launch the game with its own command line untouched.
# Renderer: DXVK -- it reaches the 5080 via /dev/nvidia*, while Mesa/EGL is blocked
# by the Slurm cgroup (/dev/dri/renderD128: Permission denied).
#
# The "=== done: <size> ===" banner at the bottom used to print whatever `ls -lh`
# said and exit 0.  logs/shots/ then accumulated 39 BYTE-IDENTICAL PNGs and a
# gameplay.mp4 that is that one still frame, while game-498.out reported
# `CLIENT DIED` earlier in the same run.  Every wait-for-marker loop below now
# records whether it actually saw its marker, and the banner is gated on the
# capture not being degenerate.
DUR=${1:-600}
ROOT=/mnt/storage/lanerl
# derived, not literal: the export is /srv/nfs on danilogin and /mnt/nfs on desktop
NFS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
OUT=$NFS/ahriuwu-lanerl/lanerl/logs
CLIENT=$ROOT/client/extracted/RADS/solutions/lol_game_client_sln/releases/0.0.1.68/deploy
export WINEPREFIX="$ROOT/prefix" DISPLAY=:99
export WINEDLLOVERRIDES="mscoree,mshtml=;d3d9=n"
export DXVK_STATE_CACHE_PATH="$ROOT/dxvk_cache"; mkdir -p "$ROOT/dxvk_cache"
WINE="$ROOT/wine/bin/wine"
mkdir -p "$OUT/shots"
# teardown: "nothing to kill" is the normal case, never a script failure
cleanup(){ pkill -f "League of Legends.exe" 2>/dev/null || true
           "$ROOT/wine/bin/wineserver" -k 2>/dev/null || true
           [ -n "${SRVPID:-}" ] && kill $SRVPID 2>/dev/null || true
           [ -n "${XP:-}" ] && kill $XP 2>/dev/null || true; }
trap cleanup EXIT

RUN_FAIL=0
fail(){ echo "  FAIL: $*"; RUN_FAIL=$((RUN_FAIL+1)); }

"$ROOT/env/bin/Xvfb" :99 -screen 0 1920x1080x24 +extension GLX +extension RANDR +render >/dev/null 2>&1 & XP=$!; sleep 4

echo "=== enable wine virtual desktop via registry ==="
# if this write does not land, the null-deref crash comes back and the whole run
# is invalid -- record it rather than assuming it worked
REG_RC=0
"$WINE" reg add 'HKCU\Software\Wine\Explorer\Desktops' /v Default /t REG_SZ /d 1920x1080 /f >/dev/null 2>&1 || REG_RC=$?
"$WINE" reg add 'HKCU\Software\Wine\Explorer' /v Desktop /t REG_SZ /d Default /f >/dev/null 2>&1 || REG_RC=$?
[ "$REG_RC" -eq 0 ] && echo "  set" || fail "wine reg add exited $REG_RC -- virtual desktop may be off"

echo "=== server ==="
export DOTNET_ROOT=$NFS/lanerl-vendor/dotnet
CFG="$NFS/ahriuwu-lanerl/lanerl/cfg/garen1v1.json"
[ -f "$CFG" ] || { echo "FATAL: $CFG does not resolve on $(hostname)"; exit 1; }
cd $NFS/lanerl-vendor/LoLServer/GameServerConsole/bin/Release/net6.0
"$DOTNET_ROOT/dotnet" ./GameServerConsole.dll --config "$CFG" \
    >"$OUT/server_fin.log" 2>&1 & SRVPID=$!
READY=0
for i in $(seq 1 40); do
    if grep -q "Game is ready" "$OUT/server_fin.log" 2>/dev/null; then READY=1; break; fi
    kill -0 $SRVPID 2>/dev/null || break        # server exited; stop waiting on a corpse
    sleep 2
done
# hard stop: with no server there is nothing to record, and 10 minutes of Xvfb
# would still have produced an mp4 that looks like a result
[ "$READY" -eq 1 ] || { echo "FATAL: server never reached 'Game is ready' in 80s"; tail -20 "$OUT/server_fin.log" || true; exit 1; }
echo "  ready"

echo "=== client (own command line, virtual desktop from registry) ==="
cd "$CLIENT"
( "$WINE" "League of Legends.exe" "8394" "LoLLauncher.exe" "" \
  "127.0.0.1 5119 17BLOhi6KZsTtldTsizvHg== 1" >"$OUT/client_fin.log" 2>&1; echo "EXIT=$?" >>"$OUT/client_fin.log" ) &

# confirm the connection actually happens before we bother recording
CONNECTED=0
for i in $(seq 1 45); do
    if grep -q "Accepted client version" "$OUT/server_fin.log" 2>/dev/null; then
        CONNECTED=1; echo "  CLIENT CONNECTED at $((i*4))s"; break
    fi
    sleep 4
done
[ "$CONNECTED" -eq 1 ] || fail "client never connected (no 'Accepted client version' in 180s)"
STARTED=0
for i in $(seq 1 60); do
    if grep -q "Patience is over" "$OUT/server_fin.log" 2>/dev/null; then
        STARTED=1; echo "  GAME STARTED"; break
    fi
    sleep 3
done
[ "$STARTED" -eq 1 ] || fail "game never started (no 'Patience is over' in 180s)"

echo "=== recording ${DUR}s ==="
ffmpeg -loglevel error -y -f x11grab -framerate 10 -video_size 1920x1080 -i :99 -t "$DUR" \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p "$OUT/gameplay_final.mp4" >/dev/null 2>&1 &
FF=$!
rm -f "$OUT"/shots/fin_*.png          # stale shots must not count toward this run
for t in $(seq 1 $((DUR/60))); do
    sleep 60
    ffmpeg -loglevel error -y -f x11grab -video_size 1920x1080 -i :99 -frames:v 1 "$OUT/shots/fin_$t.png" >/dev/null 2>&1 || true
    ALIVE=$(pgrep -f 'League of Legends.exe' >/dev/null && echo yes || echo NO)
    echo "  ${t}m alive=$ALIVE png=$(stat -c%s "$OUT/shots/fin_$t.png" 2>/dev/null || echo 0)B"
    [ "$ALIVE" = yes ] || fail "client was dead at ${t}m -- everything recorded after this is a frozen frame"
done
FF_RC=0
wait $FF 2>/dev/null || FF_RC=$?

# The artefact must not be degenerate. logs/shots/ once held 39 byte-identical
# PNGs and a gameplay.mp4 that was that single still frame, reported as a
# successful ten-minute capture.
MP4=$OUT/gameplay_final.mp4
MP4_BYTES=$(stat -c%s "$MP4" 2>/dev/null || echo 0)
NSHOTS=$(ls "$OUT"/shots/fin_*.png 2>/dev/null | wc -l || true)
NDISTINCT=$(md5sum "$OUT"/shots/fin_*.png 2>/dev/null | awk '{print $1}' | sort -u | wc -l || true)
echo "  mp4=${MP4_BYTES}B ffmpeg_rc=$FF_RC shots=$NSHOTS distinct=$NDISTINCT"
[ "$FF_RC" -eq 0 ]        || fail "recording ffmpeg exited $FF_RC"
[ "$MP4_BYTES" -gt 1000000 ] || fail "$MP4 is ${MP4_BYTES}B -- too small to be ${DUR}s of gameplay"
[ "$NSHOTS" -ge 2 ] && [ "$NDISTINCT" -lt 2 ] \
    && fail "all $NSHOTS sample PNGs are byte-identical -- the capture is one frozen frame" || true

grep -vE "WARN|Could not find" "$OUT/server_fin.log" | tail -6 || true
if [ "$RUN_FAIL" -eq 0 ]; then
    echo "=== done: $MP4 ${MP4_BYTES}B, $NDISTINCT/$NSHOTS distinct sample frames ==="
else
    echo "=== RUN FAILED: $RUN_FAIL check(s) failed -- the recording is NOT a valid result ==="
    exit 1
fi
