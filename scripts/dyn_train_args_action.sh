# Action-conditioned MIXED dynamics args (the paper recipe: replays w/ real actions
# + YT w/ no_action_embed). Plain latent x-prediction loss (the Step-1' fix) — NO
# --packed, NO --pixel-hud-loss, NO tokenizer. Micro-epoch eval (interval 150) so the
# H=8 rollout gate is visible early. Sourced by run_ddp_dyn.sh into DYN_ARGS.
: "${LATENTS_DIR:?dyn_train_args_action.sh: LATENTS_DIR must be set}"
: "${LABELS_ROOT:?dyn_train_args_action.sh: LABELS_ROOT must be set}"
: "${CHECKPOINT_DIR:?dyn_train_args_action.sh: CHECKPOINT_DIR must be set}"
: "${NUM_WORKERS:=6}"
: "${EPOCHS:=50}"
: "${EVAL_INTERVAL:=150}"
: "${GRAD_ACCUM_SHORT:=8}"
: "${GRAD_ACCUM_LONG:=16}"
: "${LR:=1e-4}"
: "${WARMUP_STEPS:=500}"
: "${STRIDE:=64}"
: "${SEED:=0}"
DYN_ARGS=(
  --use-actions --labels-root "$LABELS_ROOT" --packed
  --latents-dir "$LATENTS_DIR" --checkpoint-dir "$CHECKPOINT_DIR"
  --model-size medium --latent-dim 32
  --num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0 --shortcut-k-max 64
  --alternating-lengths --seq-len-short 128 --seq-len-long 256
  --batch-size-short 2 --batch-size-long 1 --long-ratio 0.1 --stride "$STRIDE"
  --gradient-accumulation 8
  --gradient-accumulation-short "$GRAD_ACCUM_SHORT" --gradient-accumulation-long "$GRAD_ACCUM_LONG"
  --gradient-checkpointing --no-compile
  --lr "$LR" --lr-schedule wsd --warmup-steps "$WARMUP_STEPS" --adam-betas 0.9 0.999
  --independent-frame-ratio 0.3 --seed "$SEED"
  --epochs "$EPOCHS" --eval-interval "$EVAL_INTERVAL" --checkpoint-minutes 20
  --num-workers "$NUM_WORKERS" --holdout-videos 2
)
if [ -n "${RESUME:-}" ] && [ -f "$RESUME" ]; then
  DYN_ARGS+=(--resume "$RESUME")
fi
