#!/bin/bash
set -uo pipefail
DEST=/srv/nfs/datasets/replay_latents_v7_bc
LOG=/srv/nfs/projects/ahriuwu/scratchpad/bc_1060.log
: > "$LOG"
# wait up to 20 min for the stage copy to finish
for i in $(seq 1 120); do
  n=$(ls "$DEST"/NA1_*.pt 2>/dev/null | wc -l)
  { [ -f "$DEST/.copy_done" ] || [ "$n" -ge 125 ]; } && break
  sleep 10
done
n=$(ls "$DEST"/NA1_*.pt 2>/dev/null | wc -l)
echo "[watcher] launching BC on $n staged games" >> "$LOG"
exec bash /srv/nfs/projects/ahriuwu/scripts/launch_bc_1060.sh
