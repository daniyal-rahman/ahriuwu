#!/bin/bash
# Stage the mixed tokenizer corpus on the desktop NVMe (2026-08-05 plateau test):
#   /scratch/ahriuwu/tok_mixed_flat/<video>/*.png
#     = symlinks to the 142 replay match dirs (already-352 png)
#     + 250 extracted YT games (352 jpgs, renamed *.png in-flight — loaders sniff
#       content, extension is cosmetic; keeps the single --file-ext glob happy)
# Also extracts 4 held-out YT games to NFS for the login-side fixed-eval watcher.
# Resumable: per-game .done markers. Run on the DESKTOP.
set -uo pipefail
MIX=/scratch/ahriuwu/tok_mixed_flat
TARS=/mnt/nfs/datasets/yt_pretrain_garen
EVAL_DST=/mnt/nfs/datasets/yt_eval_frames_352
N_TRAIN=250
mkdir -p "$MIX" "$EVAL_DST"

echo "[stage] symlinking replay match dirs (NFS source — the old /scratch flat corpus is dangling symlinks)..."
for m in /mnt/nfs/datasets/lol_replays_16_9_772/NA1_*/; do
  mid=$(basename "$m")
  grep -qx "$mid" /scratch/ahriuwu/holdout_matches.txt && continue
  [ -d "${m}frames" ] && ln -sfn "${m}frames" "$MIX/$mid"
done

mapfile -t ALL < <(ls "$TARS"/*.tar | sort)
EVAL_TARS=("${ALL[@]: -4}")                      # last 4 (sorted) = eval, never trained
TRAIN_TARS=("${ALL[@]:0:$N_TRAIN}")

extract() {                                       # $1=tar $2=destroot $3=rename(0/1)
  local vid dest
  vid=$(basename "$1" .tar)
  dest="$2/yt_${vid}"
  [ -f "$dest/.done" ] && return 0
  mkdir -p "$dest"
  if [ "$3" = 1 ]; then
    tar --transform='s/\.jpg$/.png/' -xf "$1" -C "$dest" && touch "$dest/.done"
  else
    tar -xf "$1" -C "$dest" && touch "$dest/.done"
  fi
}
export -f extract

echo "[stage] extracting ${#TRAIN_TARS[@]} YT train tars -> $MIX (parallel)..."
printf '%s\n' "${TRAIN_TARS[@]}" | xargs -P 6 -I{} bash -c 'extract "$@"' _ {} "$MIX" 1

echo "[stage] extracting ${#EVAL_TARS[@]} YT eval tars -> $EVAL_DST ..."
for t in "${EVAL_TARS[@]}"; do extract "$t" "$EVAL_DST" 0; done

n_dirs=$(ls "$MIX" | wc -l)
n_done=$(ls "$MIX"/yt_*/.done 2>/dev/null | wc -l)
df -h /scratch | tail -1
echo "[stage] DONE: $n_dirs video dirs in $MIX ($n_done YT extracted)"
echo "STAGING-COMPLETE"
