#!/bin/bash
# Stage the ACTION-CONDITIONED MIXED inputs on the Vast box: replay latents (real
# actions via replay labels) + a YT subset (no actions -> no_action_embed via
# placeholder labels). Plain latent loss, so NO tokenizer/HUD-mask needed. Mirrors
# stage_dyn_hudfix.sh. Seed = the best action-conditioned checkpoint (135 lineage).
set -euo pipefail
R="${RCLONE:-rclone}"
DEST="${DEST:-/workspace}"
SRC="${SRC:-/workspace}"                       # shipped working tree (src/ + scripts/)
R2_REPLAY="${R2_REPLAY:-r2:ahriuwu-yt-pretrain/dynamics_replay_latents_v7_tok6000_clean}"
R2_YT="${R2_YT:-r2:ahriuwu-yt-pretrain/dynamics_yt_subset80}"
R2_LABELS="${R2_LABELS:-r2:ahriuwu-yt-pretrain/dynamics_replay_labels}"
R2_SEED="${R2_SEED:-r2:ahriuwu-yt-pretrain/dynamics_accel/dynamics_latest.pt}"
YT_LIMIT="${YT_LIMIT:-80}"

LAT="$DEST/latents_action"; LAB="$DEST/labels_action"
mkdir -p "$LAT" "$LAB" "$DEST/seed"

echo "[stage] replay latents -> $LAT"
$R copy "$R2_REPLAY" "$LAT" --include '*.pt' --transfers 12 --multi-thread-streams 4 --s3-no-check-bucket
echo "[stage] replay labels -> $LAB (NA1_*/labels.json)"
$R copy "$R2_LABELS" "$LAB" --transfers 16 --s3-no-check-bucket

echo "[stage] YT latents (limit $YT_LIMIT) -> $LAT"
$R lsf "$R2_YT" --include '*.pt' --s3-no-check-bucket | grep '\.pt$' | head -n "$YT_LIMIT" > /tmp/yt_list.txt
$R copy "$R2_YT" "$LAT" --files-from /tmp/yt_list.txt --transfers 12 --multi-thread-streams 4 --s3-no-check-bucket

echo "[stage] generate placeholder YT labels -> $LAB"
PYTHONPATH="$SRC/src" python "$SRC/scripts/gen_yt_placeholder_labels.py" \
  --latents-dir "$LAT" --out "$LAB" --fps 30

echo "[stage] seed (best action-conditioned ckpt)"
$R copyto "$R2_SEED" "$DEST/seed/seed.pt" --s3-no-check-bucket || echo "[stage] WARN: no seed (train from scratch)"

NLAT=$(find "$LAT" -name '*.pt' | wc -l)
NREP=$(find "$LAT" -name 'NA1_*.pt' | wc -l)
NLAB=$(find "$LAB" -name 'labels.json' | wc -l)
echo "[stage] latents=$NLAT (replay=$NREP, yt=$((NLAT-NREP)))  labels=$NLAB  LATENTS_DIR=$LAT LABELS_ROOT=$LAB"
[ "$NLAT" -gt 0 ] && [ "$NLAB" -ge "$NLAT" ] || { echo "ERROR: latents/labels mismatch"; exit 1; }
