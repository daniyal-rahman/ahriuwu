#!/bin/bash
#SBATCH --job-name=yt-download
#SBATCH --output=/mnt/storage/ahriuwu/frames/download_%j.log
#SBATCH --qos=download
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@120

source ~/Repos/ahriuwu/.venv/bin/activate
cd ~/Repos/ahriuwu

# Trap SIGTERM: forward to python, wait for it to exit, then continue to resubmit
PY_PID=""
trap 'if [ -n "$PY_PID" ]; then kill -TERM "$PY_PID" 2>/dev/null; wait "$PY_PID" 2>/dev/null; fi' TERM

python3 -u scripts/download_yt_frames.py \
    --num-videos 2000 \
    --output-dir /mnt/storage/ahriuwu/frames \
    --fps 4 \
    --workers 2 &
PY_PID=$!
wait "$PY_PID"

# Auto-resubmit if there's still work to do
PROGRESS=/mnt/storage/ahriuwu/frames/progress.json
if [ -f "$PROGRESS" ]; then
    DONE=$(python3 -c "import json; p=json.load(open('$PROGRESS')); print(len(p['completed']) + len(p['failed']))")
    TOTAL=2000
    if [ "$DONE" -lt "$TOTAL" ]; then
        echo "Progress: $DONE/$TOTAL — resubmitting..."
        sbatch ~/Repos/ahriuwu/download_frames.sbatch
    else
        echo "All $TOTAL videos processed — done!"
    fi
fi
