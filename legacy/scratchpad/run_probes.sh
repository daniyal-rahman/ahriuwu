#!/bin/bash
# Run both perception probes on the desktop, sequentially, detached.
# ssh to this box drops often; this script owns the whole job so a dropped
# connection can't half-launch or half-kill anything.
set -uo pipefail
cd /mnt/nfs/projects/ahriuwu
export PYTHONPATH=/mnt/nfs/projects/ahriuwu/src
PY=/home/dani/miniconda3/envs/ml/bin/python
S=scratchpad

pkill -9 -f "bar_absolute_probe|lasthit_lead_probe" 2>/dev/null
sleep 3
: > $S/probes.log
echo "=== $(date -u '+%H:%M UTC') absolute-readout probe ===" >> $S/probes.log
$PY $S/bar_absolute_probe.py >> $S/probes.log 2>&1
echo "ABS-DONE rc=$?" >> $S/probes.log
echo "=== $(date -u '+%H:%M UTC') temporal-lead probe ===" >> $S/probes.log
$PY $S/lasthit_lead_probe.py >> $S/probes.log 2>&1
echo "LEAD-DONE rc=$?" >> $S/probes.log
echo "ALL-PROBES-DONE" >> $S/probes.log
