#!/bin/bash
# Stage a DISJOINT slice of the YT corpus for a training run.
#
# THE BUG THIS FIXES: the old ad-hoc staging did `lsf ... | head -156` every run,
# so consecutive runs trained on the SAME first ~150 games -> no new data -> pure
# step-scaling -> diminishing returns. This stages a per-run *window* of the sorted
# corpus so each consecutive run sees games it has NEVER trained on.
#
# Holdout is a FIXED set (first HOLDOUT_N games) held out from EVERY run, so
# before/after eval is comparable across runs and never contaminated by a train slice.
#
# Usage (on the GPU box), 1-based SLICE:
#   SLICE=1 SLICE_SIZE=150 DEST=/root bash stage_corpus_slice.sh   # games 7..156   (== old run3 set)
#   SLICE=2 SLICE_SIZE=150 DEST=/root bash stage_corpus_slice.sh   # games 157..306 (UNSEEN -> the next run)
# Env: R2_BUCKET(=r2:ahriuwu-yt-pretrain) DEST(=/root) HOLDOUT_N(=6) SLICE(=1) SLICE_SIZE(=150) RCLONE(=rclone)
set -euo pipefail
R="${RCLONE:-rclone}"
BUCKET="${R2_BUCKET:-r2:ahriuwu-yt-pretrain}"
DEST="${DEST:-/root}"
HOLDOUT_N="${HOLDOUT_N:-6}"
SLICE="${SLICE:-1}"
SLICE_SIZE="${SLICE_SIZE:-150}"

# full sorted game list (exclude _run/_ckpt/_meta prefixes); SORTED so slices are stable across runs
$R lsf "$BUCKET" --include '*.tar' 2>/dev/null | grep -v '^_' | sort > /tmp/allgames.txt
TOTAL=$(wc -l < /tmp/allgames.txt)

# holdout = first HOLDOUT_N games (FIXED, never trained on)
sed -n "1,${HOLDOUT_N}p" /tmp/allgames.txt > /tmp/holdout.txt
# train window = disjoint slice AFTER the holdout, offset by (SLICE-1)*SLICE_SIZE
START=$(( HOLDOUT_N + (SLICE - 1) * SLICE_SIZE + 1 ))
END=$(( HOLDOUT_N + SLICE * SLICE_SIZE ))
sed -n "${START},${END}p" /tmp/allgames.txt > /tmp/train.txt
# optional cap: stage only the first NUM_TRAIN games of the slice (e.g. a cheap probe that
# still trains on UNSEEN games from the correct disjoint block). 0 = whole slice.
NUM_TRAIN="${NUM_TRAIN:-0}"
if [ "$NUM_TRAIN" -gt 0 ]; then head -n "$NUM_TRAIN" /tmp/train.txt > /tmp/train.cap && mv /tmp/train.cap /tmp/train.txt; fi
echo "corpus=$TOTAL  holdout=$(wc -l </tmp/holdout.txt) (games 1..$HOLDOUT_N)  train SLICE=$SLICE (cap=$NUM_TRAIN) = games from $START ($(wc -l </tmp/train.txt) staged)"
[ -s /tmp/train.txt ] || { echo "ERROR: empty train slice — SLICE=$SLICE too high for corpus of $TOTAL games"; exit 1; }

mkdir -p "$DEST/dl" "$DEST/frames_train" "$DEST/frames_holdout"
pull_untar(){  # $1=listfile  $2=destdir
  while read -r t; do
    [ -z "$t" ] && continue
    $R copy "$BUCKET/$t" "$DEST/dl" --transfers 12 --multi-thread-streams 4 --s3-no-check-bucket 2>/dev/null
    tar -xf "$DEST/dl/$t" -C "$2" && rm -f "$DEST/dl/$t"
  done < "$1"
}
pull_untar /tmp/holdout.txt "$DEST/frames_holdout"
pull_untar /tmp/train.txt   "$DEST/frames_train"
echo "staged: train games=$(ls "$DEST/frames_train" | wc -l)  holdout games=$(ls "$DEST/frames_holdout" | wc -l)"
echo "train frames=$(find "$DEST/frames_train" -name '*.jpg' | wc -l)"
