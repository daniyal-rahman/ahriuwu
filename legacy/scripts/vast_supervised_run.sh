#!/bin/bash
# Supervised, self-healing DDP training for Vast consumer-GPU boxes.
# - Survives SSH disconnect (run via onstart.sh or `setsid bash vast_supervised_run.sh`).
# - FIRST launch: resume WEIGHTS from $INIT_RESUME with a FRESH LR schedule (new horizon).
# - After the first local checkpoint exists: auto-resume CONTINUING (step+optimizer+scheduler),
#   so a crash/preemption doesn't re-warmup or rehash.
# - Streams the latest checkpoint to R2 (durable source of truth). keep-last-N is done in the
#   train script. NCCL/init robustness is in run_ddp_tok.sh (P2P off, SHM on, 20-min init).
# Stop cleanly: touch $STOP_FILE.
#
# Required env: INIT_RESUME (path to the checkpoint to start from), CHECKPOINT_DIR, FRAMES_DIR,
#               MAX_STEPS, R2_CKPT (e.g. r2:ahriuwu-yt-pretrain/_run3). Others have defaults.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin
: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}"; : "${INIT_RESUME:?set INIT_RESUME}"
RUN_LOG="${RUN_LOG:-/root/train.log}"
STOP_FILE="${STOP_FILE:-/root/.run_stop}"
R2_CKPT="${R2_CKPT:-}"
LAT="$CHECKPOINT_DIR/transformer_tokenizer_latest.pt"
log(){ echo "[$(date '+%F %T')] supervisor: $*" >> "$RUN_LOG"; }

log "start. INIT_RESUME=$INIT_RESUME CHECKPOINT_DIR=$CHECKPOINT_DIR MAX_STEPS=${MAX_STEPS:-?}"
while true; do
  [ -f "$STOP_FILE" ] && { log "STOP flag -> exit"; break; }
  if grep -q "Training complete\|max_steps=.*reached" "$RUN_LOG" 2>/dev/null; then log "COMPLETE -> exit"; break; fi

  if ! pgrep -f "[t]rain_transformer_tokenizer.py" >/dev/null; then
    if [ -f "$LAT" ]; then
      export RESUME="$LAT"; export RESET_SCHEDULE=0          # crash/preempt resume -> CONTINUE
      log "resume CONTINUE from $LAT"
    else
      export RESUME="$INIT_RESUME"; export RESET_SCHEDULE="${INIT_RESET:-1}"  # new phase -> fresh schedule
      log "initial launch from $INIT_RESUME (reset_schedule=$RESET_SCHEDULE)"
    fi
    setsid bash scripts/run_ddp_tok.sh </dev/null >>"$RUN_LOG" 2>&1 &
    sleep 900   # allow multi-minute NCCL init + a run window before re-checking
  fi

  # durable: stream latest checkpoint to R2 every cycle
  if [ -n "$R2_CKPT" ] && [ -f "$LAT" ]; then
    rclone copyto "$LAT" "$R2_CKPT/tokenizer_latest.pt" --s3-no-check-bucket --s3-disable-checksum 2>/dev/null && log "streamed latest -> $R2_CKPT"
  fi
  sleep 120
done
