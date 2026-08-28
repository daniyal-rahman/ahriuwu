#!/bin/bash
# Build the dataset caches for the bronze retrain configs.
#
# Runs on the LOGIN node on purpose: parsing is CPU/JSON/NFS work with no GPU in
# it, and login has no availability window, whereas the 5080 desktop is only
# reliably up 00:00-09:00. Spending that window on a reparse would waste it.
#
# Needed because the cache key now covers prefirst_mode / movement_interp /
# reward_config (schema 5, commit 35d8b74). Before that, changing any of those
# with a cache on disk was a SILENT NO-OP -- the run trained on the old labels
# and looked like the flag did nothing.
#
# Launch:  nohup /srv/nfs/projects/ahriuwu/ops/build_caches.sh >/dev/null 2>&1 &
set -uo pipefail
REPO=/srv/nfs/projects/ahriuwu
cd "$REPO" || exit 1
PY=/home/dani/miniconda3/envs/ml/bin/python
LOG=$REPO/ops/build_caches.log
echo "$(date -u '+%m-%d %H:%M UTC') build_caches: start" >> "$LOG"

build () {  # name  prefirst  interp
  local name=$1 pf=$2 mi=$3
  local out=$REPO/data/cache_${name}.pt
  if [ -f "$out" ]; then
    echo "$(date -u '+%m-%d %H:%M UTC')   $name: exists, skip" >> "$LOG"; return
  fi
  echo "$(date -u '+%m-%d %H:%M UTC')   $name: building (prefirst=$pf interp=$mi)" >> "$LOG"
  PYTHONPATH=src $PY - "$out" "$pf" "$mi" >> "$LOG" 2>&1 <<'PY'
import sys, time, json, glob, os
sys.path.insert(0, "/srv/nfs/projects/ahriuwu/src")
from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset
out, pf, mi = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
LAT = "/srv/nfs/datasets/replay_latents_v7_bc"
ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
mids = [os.path.basename(p)[:-3] for p in sorted(glob.glob(f"{LAT}/NA1_*.pt"))]
t0 = time.time()
ds = ReplayLatentSequenceDataset(
    LAT, ROOT, outcomes={m: True for m in mids}, sequence_length=16, stride=8,
    movement_source="clicks", cache_path=out, prefirst_mode=pf, movement_interp=mi)
print(f"BUILT {out}  matches={len(getattr(ds,'match_data',{}))} "
      f"sequences={len(ds)} in {(time.time()-t0)/60:.1f} min", flush=True)
PY
  echo "$(date -u '+%m-%d %H:%M UTC')   $name: done rc=$?" >> "$LOG"
}

build baseline sentinel 0
build fixed    heading  1
echo "$(date -u '+%m-%d %H:%M UTC') build_caches: all done" >> "$LOG"
