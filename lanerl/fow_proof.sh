#!/usr/bin/env bash
# Run N short bot-vs-bot games with the fog audit on and collect the evidence.
cd /mnt/nfs/projects/ahriuwu-lanerl
PY=/home/dani/miniconda3/envs/ml/bin/python
for i in $(seq 1 "${1:-8}"); do
  LANERL_FOW_AUDIT=1 timeout 420 $PY \
    /mnt/nfs/projects/ahriuwu-lanerl/lanerl/fow_probe.py "/tmp/fow_$i.log" 200000 \
    >/dev/null 2>&1 || echo "  run $i: probe exited nonzero"
done
echo "=== evidence across runs ==="
grep -h "LANERL_FOW_NULL_PROVIDER" /tmp/fow_*.log 2>/dev/null | head -5
echo "null-provider reads : $(grep -hc 'LANERL_FOW_NULL_PROVIDER' /tmp/fow_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)"
echo "off-thread mutations: $(grep -hc 'LANERL_FOW_OFFTHREAD' /tmp/fow_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)"
echo "mutations during scan: $(grep -hc 'MUTATION_DURING_SCAN' /tmp/fow_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)"
echo "server crashes      : $(grep -hc 'NullReferenceException' /tmp/fow_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)"
echo "--- off-thread stacks, if any ---"
grep -hA 6 "LANERL_FOW_OFFTHREAD" /tmp/fow_*.log 2>/dev/null | head -12
