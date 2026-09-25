#!/bin/bash
# Supervised, self-healing DDP dynamics training for Vast consumer-GPU boxes.
# Ports scripts/vast_supervised_run.sh (tokenizer) to the dynamics trainer.
# - Survives SSH disconnect (run via onstart.sh or `setsid bash vast_supervised_dyn.sh`).
# - After the first local checkpoint exists: auto-resume CONTINUING (step+optimizer+
#   scheduler), so a crash/preemption doesn't re-warmup.
# - Optional INIT_RESUME seeds from an existing checkpoint on the very first launch
#   (unset = cold start from scratch — the usual case for a fresh dynamics run).
# - Streams the latest checkpoint to R2 (durable source of truth) every cycle.
# Stop cleanly: touch $STOP_FILE.
#
# Required env: CHECKPOINT_DIR, LATENTS_DIR, LABELS_ROOT. Optional: R2_CKPT
#               (e.g. r2:ahriuwu-dyn-ckpt/_run1), INIT_RESUME, NGPU, NUM_WORKERS.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin
: "${CHECKPOINT_DIR:?set CHECKPOINT_DIR}"
RUN_LOG="${RUN_LOG:-/root/dyn_train.log}"
STOP_FILE="${STOP_FILE:-/root/.dyn_stop}"
R2_CKPT="${R2_CKPT:-}"
INIT_RESUME="${INIT_RESUME:-}"
LAT="$CHECKPOINT_DIR/dynamics_latest.pt"
log(){ echo "[$(date '+%F %T')] supervisor: $*" >> "$RUN_LOG"; }

log "start. CHECKPOINT_DIR=$CHECKPOINT_DIR INIT_RESUME=${INIT_RESUME:-<cold>} R2_CKPT=${R2_CKPT:-<none>}"
while true; do
  [ -f "$STOP_FILE" ] && { log "STOP flag -> exit"; break; }
  if grep -q "Training complete" "$RUN_LOG" 2>/dev/null; then log "COMPLETE -> exit"; break; fi

  if ! pgrep -f "[t]rain_dynamics.py" >/dev/null; then
    if [ -f "$LAT" ]; then
      export RESUME="$LAT"                       # crash/preempt -> CONTINUE
      log "resume CONTINUE from $LAT"
    elif [ -n "$INIT_RESUME" ] && [ -f "$INIT_RESUME" ]; then
      export RESUME="$INIT_RESUME"               # first launch from a seed checkpoint
      log "initial launch from seed $INIT_RESUME"
    else
      export RESUME=""                           # cold start
      log "COLD start (no local checkpoint, no INIT_RESUME)"
    fi
    setsid bash scripts/run_ddp_dyn.sh </dev/null >>"$RUN_LOG" 2>&1 &
    sleep 900   # allow multi-minute NCCL init + a run window before re-checking
  fi

  # durable: stream latest checkpoint to R2 every cycle
  if [ -n "$R2_CKPT" ] && [ -f "$LAT" ]; then
    rclone copyto "$LAT" "$R2_CKPT/dynamics_latest.pt" --s3-no-check-bucket --s3-disable-checksum 2>/dev/null && log "streamed latest -> $R2_CKPT"
  fi
  sleep 120
done
