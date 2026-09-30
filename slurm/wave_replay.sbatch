#!/bin/bash
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
unset XLA_FLAGS
export JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
./.venv-jax/bin/python -m pytest -q lanerl_jax/tests/test_replay.py lanerl_jax/train/tests/test_wave_scenario.py
./.venv-jax/bin/python -m pytest -q lanerl_jax/sim/tests/test_combat.py lanerl_jax/sim/tests/test_spells.py -k 'a_swing_that_leaves_range_mid_windup or a_swing_that_completes_this_tick or autoattacks_are_suppressed_while_spinning or e_recast_is_ignored_early'
echo 'CANARY PASSED: replay schema, scenario, attack windup and E cancellation'
./.venv-jax/bin/python -m lanerl_jax.probes.wave_replay "$1"
for SIDE in 0 1; do
 if [ ! -f "/mnt/nfs/shared/$1/low_team_${SIDE}/trace.npz" ]; then continue; fi
 ./.venv-jax/bin/python -m lanerl_jax.replay_render "/mnt/nfs/shared/$1/low_team_${SIDE}/trace.npz" --out-dir "/mnt/nfs/shared/$1/low_team_${SIDE}" --video --speed 1 --fps 10
 ./.venv-jax/bin/python -m lanerl_jax.replay_render "/mnt/nfs/shared/$1/low_team_${SIDE}/trace.npz" --out-dir "/mnt/nfs/shared/$1/low_team_${SIDE}/combat" --video --view combat --speed 1 --fps 10
done
