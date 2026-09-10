#!/usr/bin/env bash
# -e: a python step that dies must stop the script.  pipefail: `$PY ... | tail`
# used to report *tail's* status, so a server that never booted read as a clean
# run and the next step ran anyway.  The banner at the bottom is gated on the
# report files actually appearing, not on reaching the last line.
set -euo pipefail
PY=/home/dani/miniconda3/envs/ml/bin/python
# never write the mount point: /srv/nfs exists only on danilogin, /mnt/nfs only
# on desktop.  Derive the bench dir from this script instead.
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$BENCH"

echo "=== 1/2 reset benchmark ==="
$PY measure_all.py --only reset 2>&1 | tail -20
echo "=== 2/2 CS@10, 8 seeds ==="
$PY replicate.py --configs out/cfg_final.json --seeds 8 --port0 6100 --out rep_final.json 2>&1 | tail -8

for f in out/measure_all.json out/rep_final.json; do
    [ -s "$BENCH/$f" ] || { echo "FAILED: $f was not written"; exit 1; }
done
echo "=== ALL DONE ==="
