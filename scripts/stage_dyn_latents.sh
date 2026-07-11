#!/bin/bash
# Stage dynamics packed latents + action labels from R2 to local disk (on the GPU box).
#
# The pretokenized v7 dim-32 latents (one <match>.pt each: {latents:(N,32,16,16),
# frame_indices:(N,)}) and the per-match action labels (<match>/{labels.json,
# clicks.json}) must already be uploaded to R2. This mirrors the tokenizer's
# scripts/stage_corpus_slice.sh but for the dynamics inputs (no slice/holdout —
# the whole 125-game replay set is small).
#
# VERIFIED (2026-07-06): the dim-32 v7 latents are at the path below — 125 matches,
# 51.6 GiB, one <match>.pt each (see its INFO.md). rclone must have the `r2:` remote
# (configured on the Vast box; on danilogin the binary is /home/dani/bin/rclone).
#
# NOTE: the ACTION LABELS are NOT on R2 (only latents). For the action-conditioned
# run (--use-actions) upload the replay labels first (labels.json/clicks.json per
# match, from /srv/nfs/datasets/lol_replays_16_9_772) and set R2_LABELS; otherwise
# omit --use-actions for a latents-only run.
#
# Env: R2_LATENTS(=r2:ahriuwu-yt-pretrain/dynamics_replay_latents_v7_tok6000_clean)
#      R2_LABELS(=<unset>) DEST(=/workspace) RCLONE(=rclone)
set -euo pipefail
R="${RCLONE:-rclone}"
R2_LATENTS="${R2_LATENTS:-r2:ahriuwu-yt-pretrain/dynamics_replay_latents_v7_tok6000_clean}"
R2_LABELS="${R2_LABELS:-}"
DEST="${DEST:-/workspace}"

mkdir -p "$DEST/latents"
echo "[stage] latents $R2_LATENTS -> $DEST/latents"
$R copy "$R2_LATENTS" "$DEST/latents" --include '*.pt' --transfers 12 --multi-thread-streams 4 --s3-no-check-bucket
NLAT=$(find "$DEST/latents" -name '*.pt' | wc -l)
echo "staged $NLAT latent matches -> LATENTS_DIR=$DEST/latents"
[ "$NLAT" -gt 0 ] || { echo "ERROR: no latents staged — check R2_LATENTS"; exit 1; }

if [ -n "$R2_LABELS" ]; then
  mkdir -p "$DEST/labels"
  echo "[stage] labels $R2_LABELS -> $DEST/labels"
  $R copy "$R2_LABELS" "$DEST/labels" --transfers 12 --multi-thread-streams 4 --s3-no-check-bucket
  echo "staged $(find "$DEST/labels" -mindepth 1 -maxdepth 1 -type d | wc -l) label dirs -> LABELS_ROOT=$DEST/labels"
else
  echo "R2_LABELS unset -> latents-only (run without --use-actions, or upload labels + set R2_LABELS)"
fi
