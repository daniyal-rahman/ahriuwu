#!/usr/bin/env bash
# Run lanerl_jax/sim/tests one FILE per process under the login cap.
#
# One process for the whole suite OOM-kills at a 14G cap around test 58, and
# test_combat.py alone needs ~16G (its eager 200-tick tests; 37 passed at 18G):
# XLA's compile caches for full-`tick` tests accumulate across the process
# and never shrink. A per-file process holds the peak to a single file's
# worth, and a file that fails is named in the summary instead of killing
# the others. Output: one line per file, then `pass`/`fail` totals.
#
#   ops/sim_tests_per_file.sh [pytest args...]
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0; pass=0
for f in lanerl_jax/sim/tests/test_*.py; do
    if out=$(ops/desktop_cpu.sh 8 16G .venv-jax/bin/python -m pytest -q -p no:cacheprovider "$f" "$@" 2>&1); then
        pass=$((pass+1)); echo "PASS $f: $(echo "$out" | tail -1)"
    else
        fail=$((fail+1)); echo "FAIL $f"; echo "$out" | tail -40
    fi
done
echo "files passed=$pass failed=$fail"
[ "$fail" -eq 0 ]
