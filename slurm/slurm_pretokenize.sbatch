#!/bin/bash
#SBATCH --job-name=pretokenize
#SBATCH --output=/mnt/storage/ahriuwu-data/logs/slurm_pretokenize_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --qos=training
#SBATCH --signal=B:TERM@60
#SBATCH --mail-user=danizainr@gmail.com
#SBATCH --mail-type=FAIL,END

set +u
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate ml
set -euo pipefail

export PYTHONUNBUFFERED=1

cd /mnt/storage/ahriuwu

CHECKPOINT=/mnt/storage/ahriuwu-data/checkpoints/transformer_tokenizer_latest.pt
FRAMES_DIR=/mnt/storage/ahriuwu/frames/frames
OUTPUT_DIR=/mnt/storage/ahriuwu/latents

python scripts/pretokenize_frames.py \
  --checkpoint "$CHECKPOINT" \
  --frames-dir "$FRAMES_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --num-workers 4 \
  --resume
