#!/usr/bin/env bash
set -euo pipefail          # before anything runs; see the note above STAGE_FAIL
# Stage the display stack and prove it works BEFORE attempting the real client,
# so a failure tells us which layer broke.
#
# Two bugs killed the first attempt, both fixed here:
#  - piping wine into `tail` hangs forever: wineserver is a persistent daemon that
#    inherits stdout, so the pipe never closes even after wine itself exits.
#    => redirect to a file, never pipe.
#  - wineboot tries to fetch wine-mono/wine-gecko and blocks on a dialog nobody
#    can click. => WINEDLLOVERRIDES=mscoree,mshtml= disables both.
# The whole point of this script is that "a failure tells us which layer broke",
# but it used to print "=== staged ===" and exit 0 whatever the four stages said:
# disp-491.out:6 reports `prefix OK` next to a 0-byte wineboot.log.  Each stage
# now records into STAGE_FAIL and the banner is gated on it.  set -e with an
# explicit `|| true` wherever a non-zero status is genuinely expected.
ROOT=/mnt/storage/lanerl
export WINEPREFIX="$ROOT/prefix"
export WINEDEBUG=-all
export WINEDLLOVERRIDES="mscoree,mshtml="
export DISPLAY=:99
WINE="$ROOT/wine/bin/wine"
WINESERVER="$ROOT/wine/bin/wineserver"
# derived, not literal: the export is /srv/nfs on danilogin and /mnt/nfs on desktop
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/logs"
mkdir -p "$OUT"

STAGE_FAIL=0
fail() { echo "  FAIL: $*"; STAGE_FAIL=$((STAGE_FAIL+1)); }

cleanup() {
    # teardown: "nothing to kill" is the normal case, never a script failure
    "$WINESERVER" -k 2>/dev/null || true
    [ -n "${XVFB_PID:-}" ] && kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== [1/4] Xvfb ==="
"$ROOT/env/bin/Xvfb" :99 -screen 0 1280x720x24 >/dev/null 2>&1 &
XVFB_PID=$!
sleep 4
kill -0 $XVFB_PID 2>/dev/null && echo "Xvfb up (pid $XVFB_PID)" || { echo "Xvfb DIED"; exit 1; }

echo "=== [2/4] wine prefix ==="
# a timeout here is a real failure, but it must be recorded rather than abort the
# run, so the remaining stages still tell us which layer broke
WB_RC=0
timeout 240 "$WINE" wineboot -i >"$OUT/wineboot.log" 2>&1 || WB_RC=$?
echo "wineboot rc=$WB_RC (124=timeout)"
[ "$WB_RC" -eq 0 ] || fail "wineboot exited $WB_RC"
"$WINESERVER" -w 2>/dev/null &
sleep 2
echo "prefix: $(du -sh "$WINEPREFIX" 2>/dev/null | cut -f1 || true)"
# an empty wineboot.log next to "prefix OK" is what disp-491.out reported; the
# prefix directory existing does not mean wineboot did anything
[ -d "$WINEPREFIX/drive_c/windows" ] && echo "prefix OK" || fail "prefix MISSING ($WINEPREFIX/drive_c/windows)"
[ -s "$OUT/wineboot.log" ] || fail "wineboot.log is empty -- wineboot produced no output at all"

echo "=== [3/4] wine renders a window ==="
timeout 60 "$WINE" notepad >"$OUT/notepad.log" 2>&1 &
sleep 15
pgrep -f "notepad" >/dev/null && echo "notepad running" || fail "notepad NOT running"

echo "=== [4/4] ffmpeg grabs the display ==="
# ffmpeg's own status is not the check; the artefact is. Record both.
FF_RC=0
timeout 40 ffmpeg -loglevel error -y -f x11grab -video_size 1280x720 -i :99 -frames:v 1 "$OUT/display_test.png" >"$OUT/ffmpeg.log" 2>&1 || FF_RC=$?
if [ -s "$OUT/display_test.png" ]; then
    echo "CAPTURE OK: $(ls -lh "$OUT/display_test.png" | awk '{print $5}')"
else
    fail "CAPTURE FAILED (ffmpeg rc=$FF_RC)"; tail -3 "$OUT/ffmpeg.log" 2>/dev/null || true
fi

pkill -f notepad 2>/dev/null || true
if [ "$STAGE_FAIL" -eq 0 ]; then
    echo "=== staged: all 4 stages verified ==="
else
    echo "=== NOT staged: $STAGE_FAIL stage(s) failed -- do NOT launch the real client ==="
    exit 1
fi
