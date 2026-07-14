#!/bin/bash
# Stage the hud-fix inputs from R2 onto the Vast box: combined UNLABELED latents
# (replay set + YT subset, one <id>.pt each) + the frozen v7 tokenizer + the HUD
# valid-mask. Mirrors stage_dyn_latents.sh but for the packed pixel-HUD run.
set -euo pipefail
R="${RCLONE:-rclone}"
DEST="${DEST:-/workspace}"
R2_REPLAY="${R2_REPLAY:-r2:ahriuwu-yt-pretrain/dynamics_replay_latents_v7_tok6000_clean}"
R2_YT="${R2_YT:-r2:ahriuwu-yt-pretrain/dynamics_yt_subset80}"
R2_TOK="${R2_TOK:-r2:ahriuwu-yt-pretrain/tokenizer_v7}"
R2_MASK="${R2_MASK:-r2:ahriuwu-yt-pretrain/hud_mask}"

mkdir -p "$DEST/latents_hudfix" "$DEST/tok" "$DEST/mask"
echo "[stage] replays -> latents_hudfix"
$R copy "$R2_REPLAY" "$DEST/latents_hudfix" --include '*.pt' --transfers 12 --multi-thread-streams 4 --s3-no-check-bucket
echo "[stage] YT subset -> latents_hudfix"
$R copy "$R2_YT" "$DEST/latents_hudfix" --include '*.pt' --transfers 12 --multi-thread-streams 4 --s3-no-check-bucket
NLAT=$(find "$DEST/latents_hudfix" -name '*.pt' | wc -l)
echo "combined latents: $NLAT (replay + YT subset) -> LATENTS_DIR=$DEST/latents_hudfix"
[ "$NLAT" -gt 0 ] || { echo "ERROR: no latents staged"; exit 1; }
echo "[stage] tokenizer + HUD mask"
$R copy "$R2_TOK"  "$DEST/tok"  --transfers 4 --s3-no-check-bucket
$R copy "$R2_MASK" "$DEST/mask" --transfers 4 --s3-no-check-bucket
echo "tokenizer=$(ls "$DEST"/tok/*.pt 2>/dev/null)  mask=$(ls "$DEST"/mask/*.pt 2>/dev/null)"
[ -n "$(ls "$DEST"/tok/*.pt 2>/dev/null)" ] && [ -n "$(ls "$DEST"/mask/*.pt 2>/dev/null)" ] \
  || { echo "ERROR: tokenizer or mask missing"; exit 1; }
