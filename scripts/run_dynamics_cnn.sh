#!/bin/bash
# Run dynamics training with CNN tokenizer latents
# Use: nohup ./scripts/run_dynamics_cnn.sh > logs/dynamics_cnn256_tiny.log 2>&1 &

cd /home/dani/Repos/ahriuwu
source .venv/bin/activate

python -u scripts/train_dynamics.py \
    --latents-dir data/processed/latents_cnn \
    --latent-dim 256 \
    --tokenizer-type cnn \
    --model-size tiny \
    --use-actions \
    --epochs 10 \
    --batch-size 4 \
    --sequence-length 32 \
    --num-workers 2
