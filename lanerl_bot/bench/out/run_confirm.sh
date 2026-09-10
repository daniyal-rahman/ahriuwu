#!/usr/bin/env bash
# set -e + pipefail: `... | tail -8` used to hide replicate.py's exit status, so
# the banner printed over a sweep that never produced a report.
set -euo pipefail
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"   # never the mount literal
cd "$BENCH"
/home/dani/miniconda3/envs/ml/bin/python replicate.py --configs out/cfg_confirm.json --seeds 6 --port0 6400 --out rep_confirm.json 2>&1 | tail -8
[ -s "$BENCH/out/rep_confirm.json" ] || { echo "FAILED: out/rep_confirm.json was not written"; exit 1; }
echo "=== CONFIRM DONE ==="
