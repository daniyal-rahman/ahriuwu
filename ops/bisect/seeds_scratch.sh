#!/bin/bash
# FROM-SCRATCH CS at each commit, baseline config, seed 0, 280 updates (the
# diag run read ~11 CS there on 54371e9; base1 read ~0.4 on 214bc3a).
#SBATCH --partition=gpup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=3:00:00
#SBATCH --job-name=cs_seeds
set -uo pipefail
for c in $COMMITS; do for s in $SEEDS; do
  cd /mnt/nfs/projects/ahriuwu-bisect-$c
  echo "=== $c seed $s $(git log --oneline -1 | cut -c1-80)"
  ./.venv-gpu/bin/python -m lanerl_jax.train.run_train --envs 256 --rollout 128 --minibatches 4 --chunk 20 --ckpt-every 0 \
     --updates 280 --target-kl 1e9 --seed $s --tag "scratch-$c-s$s" \
     --out-root /mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/bisect 2>&1 | grep -E "chunk|Traceback|Error" | tail -5
done; done
