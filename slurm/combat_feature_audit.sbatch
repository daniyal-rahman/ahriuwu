#!/bin/bash
set -euo pipefail
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
unset XLA_FLAGS
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
./.venv-gpu/bin/python -m pytest -q lanerl_jax/obs/tests/test_combat_features.py lanerl_jax/train/tests/test_combat_features.py
./.venv-gpu/bin/python -m pytest -q lanerl_jax/train/tests/test_vec_train.py -k 'relative_reward or gru-afk'
echo 'CANARY PASSED: combat visibility, bar quantization, migration, nonzero actor/learner, reset/update and original v3 AFK collector'
./.venv-gpu/bin/python - "$1" <<'PY'
import json, os, sys
from pathlib import Path
spec = json.loads(Path('experiments',sys.argv[1]+'.json').read_text())
out = Path('/mnt/nfs/shared')/spec['id']
out.mkdir(parents=True,exist_ok=True)
launch = json.loads(Path('lanerl_jax/runs',spec['id'],'launch.json').read_text())
assert launch['job'] == os.environ['SLURM_JOB_ID'] and launch['spec'] == spec
result = dict(passed=True, spec=spec, job=os.environ['SLURM_JOB_ID'],
              source=launch['source'],
              gate='GPU feature invariants, migration/gradient, actor/learner/reset/update and v3 regression tests')
(out/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
PY
