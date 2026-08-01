#!/bin/bash
cd /srv/nfs/projects/ahriuwu
for T in 0.0 1.0; do
  echo "=== temperature=$T ==="
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/dani/miniconda3/envs/ml/bin/python scripts/agent_infer.py --test-latents \
    --phase2-ckpt data/phase2_bc_garen/agent_finetune_latest.pt \
    --latents rollout_stage/NA1_5549995114.pt --frames 120 --context 16 --temperature $T 2>&1 \
    | grep -aE "replay|ms/frame|ability presses|movement x|PLUMBING"
done
echo "E2E_DONE"
