#!/usr/bin/env bash
# set -e + pipefail: `... | tail -8` used to hide replicate.py's exit status, so
# the banner printed over a sweep that never produced a report.
set -euo pipefail
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"   # never the mount literal
cd "$BENCH"
/home/dani/miniconda3/envs/ml/bin/python replicate.py --configs out/cfg_curriculum.json --seeds 6 --port0 6300 --out rep_curriculum.json 2>&1 | tail -8
[ -s "$BENCH/out/rep_curriculum.json" ] || { echo "FAILED: out/rep_curriculum.json was not written"; exit 1; }
echo "=== CURRICULUM DONE ==="
