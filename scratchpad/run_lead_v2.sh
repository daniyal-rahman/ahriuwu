#!/bin/bash
set -uo pipefail
cd /mnt/nfs/projects/ahriuwu
export PYTHONPATH=/mnt/nfs/projects/ahriuwu/src
pkill -9 -f lead_probe_v2 2>/dev/null; sleep 2
: > scratchpad/lead_v2.log
/home/dani/miniconda3/envs/ml/bin/python scratchpad/lead_probe_v2.py >> scratchpad/lead_v2.log 2>&1
echo "LEADV2-DONE rc=$?" >> scratchpad/lead_v2.log
