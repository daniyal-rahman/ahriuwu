#!/bin/bash
#SBATCH --job-name=pretok-replay-v7
#SBATCH --partition=gpup
#SBATCH --output=/mnt/storage/data/ahriuwu/logs/pretok_replay_v7_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --qos=gpu-long
#SBATCH --requeue
#SBATCH --open-mode=append

# Encode the action-labeled replay frames through the frozen v7 tokenizer into
# dim-32 packed latents for dynamics training. Resumable (--resume skips matches
# already written), so a requeue after a desktop reboot just continues.

set +u
source /home/dani/miniconda3/etc/profile.d/conda.sh
conda activate ml
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH=/mnt/nfs/projects/ahriuwu/src
cd /mnt/nfs/projects/ahriuwu

# --resume skips matches already written (atomic per-match .pt), so a restart
# just continues. Desktop is stable now (no manual reboots), so no requeue trap
# — that made scancel requeue instead of die. Re-add if reboots return.
python -u scripts/pretokenize_replay_v7.py --resume
