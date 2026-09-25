#!/usr/bin/env bash
# WHAT is eating the 30 GB? Measured, not back-calculated from an OOM.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/dani/miniconda3/envs/ml/bin/python
LANERL_RUNS_DIR=/scratch/lanerl/runs PYTHONUNBUFFERED=1 \
  $PY -m lanerl_train --run-name memprobe2 --actor-mode process \
    --num-actors 8 --envs-per-actor 12 --rollout-steps 128 --total-updates 100000 \
    --checkpoint-every 999999 --eval-every 0 --anchors '' --device cuda \
    --port-base 34000 --anchor-port-base 64000 --init-from demos/bc_policy.pt \
    --lr 1e-5 --alpha 0.5 --no-end-on-death >/dev/null 2>&1 &
TRAIN=$!
sleep 540
echo "=== RSS by process class (MB), 8 actors x 12 envs = 96 instances ==="
ps -eo rss,comm --no-headers | awk '
  {cls=$2; rss[cls]+=$1; n[cls]++}
  END {for (c in rss) printf "%-22s %8.0f MB total  %5d procs  %7.1f MB each\n", c, rss[c]/1024, n[c], rss[c]/1024/n[c]}
' | sort -k2 -rn | head -8
echo "=== machine ==="; free -m | head -2
echo "=== learner the new constraint? ==="
$PY - <<'PYEOF'
import json, pathlib
rows=[json.loads(l) for l in pathlib.Path("/scratch/lanerl/runs/memprobe2/metrics.jsonl").read_text().splitlines() if l.strip()]
u=[r for r in rows if r.get("kind")=="update"]
rej=[r for r in rows if r.get("kind")=="rollout_rejected"]
if u:
    t=u[len(u)//2:]
    m=lambda k: sum(r[k] for r in t if r.get(k) is not None)/max(1,len([r for r in t if r.get(k) is not None]))
    print(f"updates={len(u)} rejected={len(rej)} ({100*len(rej)/max(1,len(u)+len(rej)):.1f}%)")
    print(f"learner_s/update={m('throughput/learner_s'):.3f}  wait_s={m('throughput/wait_s'):.3f}  "
          f"learner_frac={m('throughput/learner_frac'):.3f}  dec/s={m('throughput/decisions_per_s'):,.0f}")
PYEOF
kill -TERM $TRAIN 2>/dev/null; wait $TRAIN 2>/dev/null
