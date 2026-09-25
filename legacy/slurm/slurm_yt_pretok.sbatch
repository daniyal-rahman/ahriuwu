#!/bin/bash
#SBATCH --job-name=yt-pretok
#SBATCH --partition=gpup
#SBATCH --output=/mnt/storage/data/ahriuwu/logs/yt_pretok_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=28G
#SBATCH --time=2-12:00:00
#SBATCH --qos=gpu-long
#SBATCH --requeue
#SBATCH --mail-user=danizainr@gmail.com
#SBATCH --mail-type=FAIL,END

# Pre-tokenize the 906-game YT corpus -> v7 dim-32 latents for UNLABELED
# world-model pretraining. Streams frames from the NFS tars with overlapped
# extraction/decode/encode (see pretokenize_yt_v7.py). --resume makes it
# idempotent, so a requeue/preemption picks up where it left off (per-tar .pt).
set +u
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate ml
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/nfs/projects/ahriuwu/src
cd /mnt/nfs/projects/ahriuwu

# Storage architecture: PERMA latents -> 8TB danilogin (/mnt/nfs/datasets, 5.8TB
# free); tar-extraction workdir -> fast NVMe /scratch (transient, ~2GB). The slow
# spinning HDD /mnt/storage is avoided. All 906 fit the 8TB (disk is not a cap).
python scripts/pretokenize_yt_v7.py \
  --tars-dir /mnt/nfs/datasets/yt_pretrain_garen \
  --checkpoint /mnt/storage/data/ahriuwu-checkpoints/tokenizer_v7/transformer_tokenizer_latest.pt \
  --out /mnt/nfs/datasets/ahriuwu/dynamics_yt_latents_v7_dim32 \
  --workdir /scratch/ahriuwu/_yt_extract \
  --num-workers 12 --batch-size 48 --compile --prefetch-tars 3 --resume
