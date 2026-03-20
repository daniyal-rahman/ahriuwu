#!/bin/bash
#SBATCH --job-name=pack-latents
#SBATCH --output=/mnt/storage/ahriuwu-data/logs/slurm_pack_latents_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --qos=short
#SBATCH --mail-user=danizainr@gmail.com
#SBATCH --mail-type=FAIL,END

set +u
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate ml
set -euo pipefail

export PYTHONUNBUFFERED=1

cd /mnt/storage/ahriuwu

INPUT_DIR=/mnt/storage/ahriuwu/latents
OUTPUT_DIR=/mnt/storage/ahriuwu/latents_packed

python scripts/pack_latents.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --workers 4 \
  --skip-existing \
  --delete-source
