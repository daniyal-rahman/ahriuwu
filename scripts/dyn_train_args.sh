# shellcheck shell=bash
# === Canonical dynamics training args — SINGLE SOURCE OF TRUTH ===
# Ports the tokenizer's v7_train_args.sh pattern to the dynamics trainer. Sourced
# by scripts/run_ddp_dyn.sh (torchrun, multi-GPU DDP). Exports the array DYN_ARGS
# (the args AFTER the script path; the launcher prepends `torchrun ... scripts/...`).
#
# Model-SHAPE args (model-size / latent-dim / num-kv-heads / register tokens /
# shortcut-k-max) are RESUME-CRITICAL — changing any breaks loading a checkpoint.
#
# Run-specific bits are env-overridable; the launcher sets them before sourcing:
: "${LATENTS_DIR:?dyn_train_args.sh: LATENTS_DIR must be set}"
: "${CHECKPOINT_DIR:?dyn_train_args.sh: CHECKPOINT_DIR must be set}"
: "${LABELS_ROOT:?dyn_train_args.sh: LABELS_ROOT must be set (dir of <match>/labels.json)}"
: "${NUM_WORKERS:=6}"
: "${EPOCHS:=50}"
: "${EVAL_INTERVAL:=200}"
# Effective grad-accum PER LENGTH (the trainer splits these across ranks under DDP
# so the effective batch = world_size * batch * accum stays as configured).
: "${GRAD_ACCUM_SHORT:=8}"
: "${GRAD_ACCUM_LONG:=16}"
: "${LR:=3e-4}"
: "${WARMUP_STEPS:=3000}"
: "${SEED:=0}"

DYN_ARGS=(
  --latents-dir "$LATENTS_DIR" --packed --checkpoint-dir "$CHECKPOINT_DIR"
  # --- architecture (FIXED, resume-critical) ---
  --model-size medium --latent-dim 32 --tokenizer-type transformer
  --num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0 --shortcut-k-max 64
  # --- alternating context lengths (DreamerV4 §3.4) ---
  --alternating-lengths --seq-len-short 128 --seq-len-long 256
  --batch-size-short 2 --batch-size-long 1 --long-ratio 0.1
  --gradient-accumulation 8
  --gradient-accumulation-short "$GRAD_ACCUM_SHORT" --gradient-accumulation-long "$GRAD_ACCUM_LONG"
  --gradient-checkpointing
  # --- action conditioning ---
  --use-actions --labels-root "$LABELS_ROOT"
  # --- optimization (paper: AdamW betas 0.9/0.999, i.i.d. per-frame tau, 30% images) ---
  --lr "$LR" --lr-schedule wsd --warmup-steps "$WARMUP_STEPS" --adam-betas 0.9 0.999
  --independent-frame-ratio 0.3 --seed "$SEED"
  --epochs "$EPOCHS" --eval-interval "$EVAL_INTERVAL" --checkpoint-minutes 60
  --num-workers "$NUM_WORKERS"
  --wandb --wandb-project ahriuwu --wandb-tags dynamics ddp
)

# torch.compile: default OFF — safe on Blackwell sm_120 + torch 2.7 (compiled
# kernels hit a CUDA illegal-memory-access there; see docs/VAST.md §4). COMPILE=1
# to try it once you've verified the box is stable.
[ "${COMPILE:-0}" = "1" ] || DYN_ARGS+=(--no-compile)

# Resume iff RESUME is a real file — CONTINUE step/optimizer/scheduler state.
if [ -n "${RESUME:-}" ] && [ -f "$RESUME" ]; then
  DYN_ARGS+=(--resume "$RESUME")
fi
