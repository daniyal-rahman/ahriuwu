#!/bin/bash
#SBATCH --job-name=tok-train-v7c
#SBATCH --output=/mnt/nfs/projects/ahriuwu/logs/slurm_tok_train_v7c_%j.out
#SBATCH --open-mode=append
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --requeue
#SBATCH --mail-user=danizainr@gmail.com
#SBATCH --mail-type=FAIL,END

set +u
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate ml
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/mnt/nfs/projects/ahriuwu/src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online

cd /mnt/nfs/projects/ahriuwu
mkdir -p logs

# v7 CONTINUATION to 2 epochs. v7 finished its 6k-step WSD run at LR~0 (m=0=28.2,
# ~v6, data-starved/hazy). This extends training to ~2 epochs (15132 optsteps @
# eff-batch 64 over 484217 seqs) with a fresh warmup->flat->decay schedule.
# Same arch/objective as v7 (512x16, te=4, full-frame MSE, tube masking) — only
# more data exposure. Decay is PLANNED at the 2-epoch tail but can be cut early
# (re-render rollouts; if m=0 stops climbing, lower --max-steps on a relaunch).
#
# NEW checkpoint dir so v7's finished run stays intact. First launch seeds from
# v7 step-5970 with --reset-schedule (keep weights+optimizer, fresh LR clock);
# requeues then resume THIS run's own latest.pt normally (no reset).
CHECKPOINT_DIR=/mnt/nfs/projects/ahriuwu/checkpoints/tokenizer_v7_cont
SAMPLE_DIR=/mnt/nfs/projects/ahriuwu/samples/tokenizer_v7_cont
FRAMES_DIR=/scratch/ahriuwu/action_labeled_352png_train_flat
V7_SEED=/mnt/nfs/projects/ahriuwu/checkpoints/tokenizer_v7/transformer_tokenizer_latest.pt

CMD=(
  python -u scripts/train_transformer_tokenizer.py
  --frames-dir "$FRAMES_DIR"
  --file-ext png
  --skip-resize
  --augment
  --checkpoint-dir "$CHECKPOINT_DIR"
  --sample-dir "$SAMPLE_DIR"
  # --- ARCHITECTURE (identical to v7) ---
  --model-size large
  --num-encoder-layers 8
  --num-decoder-layers 8
  --latent-dim 16
  --num-latents 512
  --temporal-every 4
  --mse-on-full-frame
  # Paper masking regime (paper 3.1): PER-IMAGE masking, flat p~U(0,0.9). --tube-masking
  # and --mask-warmup-steps are ablation opt-ins now — see PAPER_DEVIATIONS 1.1/1.2.
  # --- training: warmup -> flat -> planned decay at 2-epoch tail ---
  --use-rope
  --sequence-length 20
  --batch-size 1
  --gradient-accumulation 64
  --gradient-checkpointing
  --lpips-frame-subsample 16
  --no-use-8bit-adam
  --adam-betas 0.9 0.999
  --lr 1e-4
  --weight-decay 0.1
  --lr-schedule wsd
  --warmup-steps 500
  --decay-steps 1500       # decay over last ~1500 (13632->15132); cut early by lowering --max-steps on a relaunch
  --epochs 999
  --max-steps 15132        # ~2 epochs total (484217/64 = 7566 optsteps/epoch x 2)
  --mask-ratio-min 0.0
  --mask-ratio-max 0.9
  --mask-warmup-steps 0
  --step-save-interval 30
  --checkpoint-warn-minutes 90
  --num-workers 6
  --log-interval 50
  --wandb-project ahriuwu-tokenizer
  --wandb-tags v7-cont 2epoch warmup-flat-decay
)
# Resume strategy:
#  - This continuation's OWN latest.pt exists (a requeue happened): resume it
#    normally (keeps its schedule clock — do NOT reset).
#  - First launch: seed from v7 step-5970 weights with --reset-schedule so the
#    fresh warmup->flat->decay runs from step 0.
if [ -f "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt" ]; then
  CMD+=(--resume "$CHECKPOINT_DIR/transformer_tokenizer_latest.pt")
  echo "Resuming v7-cont from its own checkpoint (requeue)"
else
  CMD+=(--resume "$V7_SEED" --reset-schedule)
  echo "First launch: seeding from v7 step-5970 with --reset-schedule (fresh LR clock)"
fi

PY_PID=""
trap 'if [ -n "$PY_PID" ]; then kill -TERM "$PY_PID" 2>/dev/null; wait "$PY_PID" 2>/dev/null; fi' TERM
trap 'if [ -n "$PY_PID" ]; then kill -USR1 "$PY_PID" 2>/dev/null; fi' USR1

"${CMD[@]}" &
PY_PID=$!
wait "$PY_PID"
EXIT_CODE=$?
exit "$EXIT_CODE"
