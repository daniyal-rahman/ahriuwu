#!/bin/bash
# Stage everything Phase-2 parity needs onto R2 so a Vast box can pull it.
# Runs detached; safe to re-run (rclone copy is incremental).
set -uo pipefail
R=~/bin/rclone
DST=r2:ahriuwu-yt-pretrain/_phase2
REPO=/srv/nfs/projects/ahriuwu
LOG=$REPO/scratchpad/stage_r2.log
: > "$LOG"
say(){ echo "$(date -u '+%H:%M:%S') $*" >> "$LOG"; }

say "=== staging Phase-2 payload to $DST ==="

# 1) small, high-value first so a Vast box can start smoke-testing early
say "[1/5] checkpoints"
$R copy "$REPO/rollout_stage/desktop_resume_8775_stripped.pt" "$DST/ckpt/" --stats-one-line --stats 30s >>"$LOG" 2>&1
$R copy "$REPO/data/phase2_parity/agent_finetune_latest.pt"    "$DST/ckpt/" --stats-one-line --stats 30s >>"$LOG" 2>&1
$R copy "$REPO/data/phase2_parity/dataset_cache.pt"            "$DST/ckpt/" --stats-one-line --stats 30s >>"$LOG" 2>&1
say "[1/5] done"

# 2) clicks.json (tiny, and the whole point of the movement fix)
say "[2/5] clicks.json"
$R copy /srv/nfs/datasets/lol_replays_16_9_772 "$DST/labels/" \
  --include "*/clicks.json" --stats-one-line --stats 30s >>"$LOG" 2>&1
say "[2/5] done"

# 3) labels.json
say "[3/5] labels.json (7.9G)"
$R copy /srv/nfs/datasets/lol_replays_16_9_772 "$DST/labels/" \
  --include "*/labels.json" --transfers 8 --stats-one-line --stats 60s >>"$LOG" 2>&1
say "[3/5] done"

# 4) the big one
say "[4/5] latents (52G)"
$R copy /srv/nfs/datasets/replay_latents_v7_bc "$DST/latents/" \
  --transfers 8 --stats-one-line --stats 60s >>"$LOG" 2>&1
say "[4/5] done"

# 5) verify
say "[5/5] verify sizes"
$R size "$DST" >>"$LOG" 2>&1
say "STAGE-DONE"
