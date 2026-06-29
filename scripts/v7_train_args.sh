# shellcheck shell=bash
# === Canonical v7 tokenizer training args — SINGLE SOURCE OF TRUTH ===
# Sourced by BOTH launchers so the model/optimization config is defined ONCE:
#   - slurm/slurm_tok_train_v7.sbatch   (Slurm, single-GPU)
#   - scripts/run_ddp_tok.sh            (torchrun, multi-GPU DDP)
# Exports the array V7_ARGS (the args AFTER the script path; the launcher prepends
# `python -u scripts/...` or `torchrun ... scripts/...`).
#
# The model-SHAPE args (model-size / layers / latent-dim / num-latents /
# temporal-every) are RESUME-CRITICAL — changing any of them breaks loading the
# v7 checkpoint. Touch with care.
#
# Run-specific bits are env-overridable; the launcher sets them before sourcing:
: "${FRAMES_DIR:?v7_train_args.sh: FRAMES_DIR must be set}"
: "${CHECKPOINT_DIR:?v7_train_args.sh: CHECKPOINT_DIR must be set}"
: "${SAMPLE_DIR:=${CHECKPOINT_DIR%/}/samples}"
: "${FILE_EXT:=png}"
: "${NUM_WORKERS:=6}"
: "${MAX_STEPS:=6000}"
# Effective batch = BATCH_SIZE * GRAD_ACCUM (the DDP script then splits accum across ranks).
# Default 1*64=64 is the known-safe config. Sweep hint: batch 2 / accum 32 is ~7% faster but
# was only tested with gradient-checkpointing OFF — verify it fits (checkpointing ON) on the
# box for the first few steps before trusting an unattended run. batch>=4 OOMs.
: "${BATCH_SIZE:=1}"
: "${GRAD_ACCUM:=64}"
# WSD schedule knobs (overridable). Defaults suit a multi-thousand-step run; for a SHORT
# probe (e.g. 1000 steps) set WARMUP_STEPS small (~50) + DECAY_STEPS small (~300) so the run
# spends its steps at flat LR actually learning, not warming up/decaying.
: "${WARMUP_STEPS:=500}"
: "${DECAY_STEPS:=1500}"
: "${WANDB_TAGS:=v7 d1024-8x8 paper-encoder-512x16 temporal-every-4}"

V7_ARGS=(
  --frames-dir "$FRAMES_DIR" --file-ext "$FILE_EXT" --skip-resize --augment
  --checkpoint-dir "$CHECKPOINT_DIR" --sample-dir "$SAMPLE_DIR"
  # --- architecture (FIXED, resume-critical): paper-encoder 512x16, D=1024, 8+8, temporal-every-4 ---
  --model-size large --num-encoder-layers 8 --num-decoder-layers 8
  --latent-dim 16 --num-latents 512 --temporal-every 4
  # --- objective / masking ---
  --mse-on-full-frame --tube-masking --mask-ratio-min 0.0 --mask-warmup-steps 2000
  # --- training (eff batch 64; under DDP the script splits grad-accum across ranks) ---
  --use-rope --sequence-length 20 --batch-size "$BATCH_SIZE" --gradient-accumulation "$GRAD_ACCUM"
  --gradient-checkpointing --lpips-frame-subsample 16
  --no-use-8bit-adam --adam-betas 0.9 0.999
  --lr 1e-4 --weight-decay 0.1
  --lr-schedule wsd --warmup-steps "$WARMUP_STEPS" --decay-steps "$DECAY_STEPS"
  --epochs 999 --max-steps "$MAX_STEPS"
  --step-save-interval 200 --checkpoint-warn-minutes 90
  --num-workers "$NUM_WORKERS" --log-interval 50
  --wandb-project ahriuwu-tokenizer --wandb-tags $WANDB_TAGS
)

# Optional: disable torch.compile (Inductor/Triton). On some new-GPU + torch combos (e.g.
# Blackwell sm_120 + torch 2.7.1) the compiled kernels hit a CUDA illegal-memory-access;
# NO_COMPILE=1 falls back to eager. Default keeps compile on.
[ "${NO_COMPILE:-0}" = "1" ] && V7_ARGS+=(--no-compile)

# Resume from RESUME iff it's a real file. --reset-schedule (fresh LR) is OPT-IN via
# RESET_SCHEDULE=1 — the DDP continuation wants it (v7 ended decayed at LR~0); the
# Slurm autoresume requeue cycle does NOT (it must continue the original schedule).
if [ -n "${RESUME:-}" ] && [ -f "$RESUME" ]; then
  V7_ARGS+=(--resume "$RESUME")
  # reset-schedule is a STRICT opt-in (=1). Previously `[ -n "$RESET_SCHEDULE" ]` fired even
  # for "0" (non-empty string), so resumes wrongly reset global_step + LR schedule. Now a
  # plain resume CONTINUES step/optimizer/scheduler state; pass RESET_SCHEDULE=1 only to
  # deliberately start a fresh LR cycle on the loaded weights.
  [ "${RESET_SCHEDULE:-0}" = "1" ] && V7_ARGS+=(--reset-schedule)
fi
