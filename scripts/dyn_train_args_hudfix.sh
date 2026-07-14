# shellcheck shell=bash
# === HUD-FIX dynamics args (UNLABELED + pixel-HUD masked loss) ===
# Variant of dyn_train_args.sh for the 578-game world-model pretrain: --packed
# (no actions -> no_action_embed), plus the pixel-space masked loss so blacked YT
# HUD regions never train the model to reproduce black. Sourced by run_ddp_dyn.sh
# when DYN_ARGS_FILE points here. Model-SHAPE args are RESUME-CRITICAL (must match
# the 168/135 checkpoint being resumed).
: "${LATENTS_DIR:?dyn_train_args_hudfix.sh: LATENTS_DIR must be set}"
: "${CHECKPOINT_DIR:?dyn_train_args_hudfix.sh: CHECKPOINT_DIR must be set}"
: "${TOK_CKPT:?dyn_train_args_hudfix.sh: TOK_CKPT (frozen v7 tokenizer) must be set}"
: "${HUD_MASK:?dyn_train_args_hudfix.sh: HUD_MASK (352x352 valid-mask .pt) must be set}"
: "${NUM_WORKERS:=6}"
: "${EPOCHS:=50}"
: "${EVAL_INTERVAL:=200}"
: "${GRAD_ACCUM_SHORT:=8}"
: "${GRAD_ACCUM_LONG:=16}"
: "${PIXEL_FRAMES:=4}"
: "${LR:=3e-4}"
: "${WARMUP_STEPS:=3000}"
: "${SEED:=0}"

DYN_ARGS=(
  --latents-dir "$LATENTS_DIR" --packed --checkpoint-dir "$CHECKPOINT_DIR"
  # --- architecture (FIXED, resume-critical: must match 168/135) ---
  --model-size medium --latent-dim 32 --tokenizer-type transformer
  --num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0 --shortcut-k-max 64
  # --- alternating context lengths ---
  --alternating-lengths --seq-len-short 128 --seq-len-long 256
  --batch-size-short 2 --batch-size-long 1 --long-ratio 0.1
  --gradient-accumulation 8
  --gradient-accumulation-short "$GRAD_ACCUM_SHORT" --gradient-accumulation-long "$GRAD_ACCUM_LONG"
  --gradient-checkpointing --no-compile
  # --- FIX #1: pixel-space masked loss for the blacked-HUD YT frames ---
  --pixel-hud-loss --pixel-loss-frames "$PIXEL_FRAMES"
  --tokenizer-ckpt "$TOK_CKPT" --hud-mask "$HUD_MASK"
  # --- optimization ---
  --lr "$LR" --lr-schedule wsd --warmup-steps "$WARMUP_STEPS" --adam-betas 0.9 0.999
  --independent-frame-ratio 0.3 --seed "$SEED"
  --epochs "$EPOCHS" --eval-interval "$EVAL_INTERVAL" --checkpoint-minutes 30
  --num-workers "$NUM_WORKERS"
)

if [ -n "${RESUME:-}" ] && [ -f "$RESUME" ]; then
  DYN_ARGS+=(--resume "$RESUME")
fi
