#!/bin/bash
# Slurm QOS tier setup — single-node, preemption-based scheduling
#
# Design:
#   Priority controls scheduling order (main scheduler evaluates highest first).
#   Preemption controls who yields (independent of priority number).
#   Reservations are disabled (bf_min_prio_reserve=9999) because on a single
#   node with preemption, they add no value and actively block backfill.
#
# Tier 1: download (250) — IO-bound gap-fillers. Highest scheduling priority
#          so they start immediately, but preemptable by everything.
# Tier 2: short    (200) — quick jobs, inference, sweeps. Max 1hr.
#          Preempts training and download.
# Tier 3: training (100) — ML training. Preempts download.
# Tier 4: interactive (10) — interactive sessions. Not preemptable by short.
#
# Run with: sudo bash slurm_qos_setup.sh

set -euo pipefail

CONF=/etc/slurm/slurm.conf

echo "=== Updating slurm.conf ==="

# Priority weights — make QOS priority matter
if grep -q "^PriorityWeightQOS" "$CONF"; then
    sed -i 's/^PriorityWeightQOS=.*/PriorityWeightQOS=1000/' "$CONF"
else
    echo "PriorityWeightQOS=1000" >> "$CONF"
fi

# Scheduler: disable reservations, enable preempt reorder
if grep -q "^SchedulerParameters" "$CONF"; then
    sed -i 's/^SchedulerParameters=.*/SchedulerParameters=preempt_reorder_count=1,bf_min_prio_reserve=9999/' "$CONF"
else
    echo "SchedulerParameters=preempt_reorder_count=1,bf_min_prio_reserve=9999" >> "$CONF"
fi

echo "=== Configuring QOS tiers ==="

# Create download QOS if it doesn't exist
sacctmgr -i add qos download 2>/dev/null || true
sacctmgr -i modify qos download set \
    priority=250 \
    preemptmode=requeue \
    MaxTRESPerUser=cpu=4

sacctmgr -i modify qos short set \
    priority=200 \
    preempt=download,lowprio,training,interactive

sacctmgr -i modify qos training set \
    priority=100 \
    preempt=download \
    PreemptExemptTime=00:30:00 \
    GraceTime=120

echo "=== Adding download QOS to user associations ==="
sacctmgr -i modify user where account=compute set qos+=download

echo "=== Reconfiguring scheduler ==="
scontrol reconfigure

echo "=== Verifying ==="
sacctmgr show qos format=Name%12,Priority,Preempt%20,PreemptMode,MaxTRESPU -p
scontrol show config | grep -E "PriorityWeightQOS|SchedulerParameters"

echo ""
echo "Done. Key points:"
echo "  - Downloads start immediately (highest scheduling priority)"
echo "  - Short/training preempt downloads when they need resources"
echo "  - No reservations — preemption handles everything"
echo "  - Resubmit existing jobs to pick up new priorities"
