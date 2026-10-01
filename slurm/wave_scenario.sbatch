#!/bin/bash
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
unset XLA_FLAGS
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "host $(hostname) job $SLURM_JOB_ID $(date -u)"
./.venv-gpu/bin/python -m pytest -q lanerl_jax/train/tests/test_wave_scenario.py lanerl_jax/train/tests/test_vec_train.py lanerl_jax/train/tests/test_own_action.py -k 'scenario or relative_reward or gru-mirror or gru-frozen or gru-afk or own_action'
if ./.venv-gpu/bin/python -c 'import json,sys; sys.exit(not json.load(open("experiments/"+sys.argv[1]+".json")).get("click_proposals",False))' "$1"; then
    ./.venv-gpu/bin/python -m pytest -q lanerl_jax/train/tests/test_click_proposals.py
fi
if ./.venv-gpu/bin/python -c 'import json,sys; sys.exit(not json.load(open("experiments/"+sys.argv[1]+".json")).get("visible_history",False))' "$1"; then
    ./.venv-gpu/bin/python -m pytest -q lanerl_jax/obs/tests/test_visible_history.py lanerl_jax/train/tests/test_visible_history.py
fi
if ./.venv-gpu/bin/python -c 'import json,sys; sys.exit(not json.load(open("experiments/"+sys.argv[1]+".json")).get("stagger_initial",False))' "$1"; then
    ./.venv-gpu/bin/python -m pytest -q lanerl_jax/train/tests/test_vec_train.py -k stagger_warmup
fi
if ./.venv-gpu/bin/python -c 'import json,sys; sys.exit(not json.load(open("experiments/"+sys.argv[1]+".json")).get("combat_features",False))' "$1"; then
    ./.venv-gpu/bin/python -m pytest -q lanerl_jax/obs/tests/test_combat_features.py lanerl_jax/train/tests/test_combat_features.py
fi
if ./.venv-gpu/bin/python -c 'import json,sys; sys.exit(not json.load(open("experiments/"+sys.argv[1]+".json")).get("click_mask",False))' "$1"; then
    ./.venv-gpu/bin/python -m pytest -q lanerl_jax/train/tests/test_click_mask.py
fi
echo 'CANARY PASSED: scenario geometry, health roles, GRU likelihood and reward'
exec ./.venv-gpu/bin/python -m lanerl_jax.train.wave_scenario_train "$@"
