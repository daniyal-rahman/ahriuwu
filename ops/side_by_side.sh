#!/bin/bash
# Side-by-side of two replay videos, optional trims (seconds of VIDEO time).
# Labels via drawtext when this ffmpeg has it; plain hstack otherwise.
#   ops/side_by_side.sh left.mp4 "left label" right.mp4 "right label" out.mp4 [left_trim] [right_trim]
set -euo pipefail
L=$1; LL=$2; R=$3; RL=$4; OUT=$5; LT=${6:-0}; RT=${7:-0}
if ffmpeg -hide_banner -filters 2>/dev/null | grep -q " drawtext "; then
  FC="[0:v]drawtext=text='$LL':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[l];[1:v]drawtext=text='$RL':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[r];[l][r]hstack=inputs=2:shortest=1"
else
  FC="[0:v][1:v]hstack=inputs=2:shortest=1"
fi
ffmpeg -y -loglevel error -ss "$LT" -i "$L" -ss "$RT" -i "$R" -filter_complex "$FC" -c:v libx264 -pix_fmt yuv420p -crf 23 "$OUT"
echo "$OUT"
