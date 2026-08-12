#!/bin/bash
# Receive a short Practice Tool clip and save EXACTLY what the model would see
# (limited->full range expansion + 352x352 squish), for validating the
# color-range fix WITHOUT loading the model (leaves the GPU to BC training).
#
#   bash capture_validate.sh [seconds] [outdir]
#
# Then: brightness is reported here; run the freeze metric separately.
set -uo pipefail
SECS=${1:-30}
OUT=${2:-/mnt/storage/ahriuwu-live/validate}
mkdir -p "$OUT"
STAMP=$(date +%H%M%S)

echo "listening udp :5000 for ${SECS}s — START THE WINDOWS FFMPEG NOW..."
# fixed: expand TV(16-235) -> PC(0-255), then squish to the model's 352x352
ffmpeg -hide_banner -loglevel error \
  -fflags nobuffer -flags low_delay -probesize 32 -analyzeduration 0 \
  -i "udp://@:5000?fifo_size=1000000&overrun_nonfatal=1&timeout=30000000" \
  -t "$SECS" -vf "scale=in_range=tv:out_range=pc,scale=352:352" \
  -c:v libx264 -preset ultrafast -pix_fmt yuv420p "$OUT/fixed_$STAMP.mp4" -y 2>&1 | tail -3
# same clip WITHOUT the fix, for A/B
ffmpeg -hide_banner -loglevel error -i "$OUT/fixed_$STAMP.mp4" -vf "scale=352:352" \
  -f null - 2>/dev/null

if [ ! -s "$OUT/fixed_$STAMP.mp4" ]; then
  echo "NO STREAM RECEIVED — is the Windows ffmpeg running and pointed at 192.168.1.100:5000?"
  exit 1
fi
echo "saved $OUT/fixed_$STAMP.mp4"
/home/dani/miniconda3/envs/ml/bin/python - "$OUT/fixed_$STAMP.mp4" <<'PYEOF'
import sys, cv2, numpy as np
cap=cv2.VideoCapture(sys.argv[1]); vals=[]
while True:
    ok,fr=cap.read()
    if not ok: break
    vals.append(fr.mean()/255)
cap.release()
m=float(np.mean(vals))
print(f"frames={len(vals)}  MEAN BRIGHTNESS={m:.3f}")
print(f"  training reference = 0.203 | old (broken, dark) live = 0.136")
verdict = "LOOKS FIXED" if m>0.18 else ("STILL DARK" if m<0.16 else "borderline")
print(f"  VERDICT: {verdict}")
PYEOF
echo "CAPTURE-DONE"
