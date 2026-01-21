#!/bin/bash
# Run dynamics training with CNN tokenizer latents
# Use: nohup ./scripts/run_dynamics_cnn.sh > logs/dynamics_cnn256_tiny.log 2>&1 &
# Use: nohup ./scripts/run_dynamics_cnn.sh --packed > logs/dynamics_cnn256_tiny.log 2>&1 &

cd /home/dani/Repos/ahriuwu
source .venv/bin/activate

# Check if --packed flag is passed
PACKED_FLAG=""
LATENTS_DIR="data/processed/latents_cnn"

for arg in "$@"; do
    if [ "$arg" == "--packed" ]; then
        PACKED_FLAG="--packed"
        LATENTS_DIR="data/processed/latents_cnn_packed"
    fi
done

python -u scripts/train_dynamics.py \
    --latents-dir "$LATENTS_DIR" \
    --latent-dim 256 \
    --tokenizer-type cnn \
    --model-size tiny \
    --use-actions \
    --epochs 10 \
    --batch-size 8 \
    --sequence-length 32 \
    --num-workers 4 \
    $PACKED_FLAG
