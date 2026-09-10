#!/usr/bin/env bash
# slurmd cached our old credentials, so the new video/render membership isn't in
# the job's group set -- but `sg` adopts them fine (verified VIDEO-OK/RENDER-OK).
# Nest sg to hold both, then start a REAL Xorg on the 5080 instead of Xvfb.
# Xvfb was the blocker: no DRI3, so NVIDIA Vulkan segfaulted creating a
# presentation surface and the client null-deref'd entering the 3D scene.
set -uo pipefail
echo "=== both groups at once? ==="
sg video -c "sg render -c 'test -r /dev/dri/card1 && test -r /dev/dri/renderD128 && echo BOTH-OK || echo partial'"

echo "=== start Xorg :1 on the GPU ==="
sg video -c "sg render -c '
  /usr/lib/xorg/Xorg :1 -novtswitch -sharevts -nolisten tcp -noreset > /tmp/xorg.log 2>&1 &
  sleep 8
  if [ -S /tmp/.X11-unix/X1 ]; then echo \"XORG UP (socket present)\"; else echo \"XORG FAILED\"; tail -25 /tmp/xorg.log; fi
'"
echo "=== glx/vulkan on :1 ==="
sg video -c "sg render -c 'DISPLAY=:1 /mnt/storage/lanerl/env/bin/xdpyinfo 2>/dev/null | head -5 || echo \"(no xdpyinfo)\"'"
grep -iE "NVIDIA|GLX|screen|abort|fatal|(EE)" /tmp/xorg.log 2>/dev/null | head -12
