#!/bin/bash
# Extract frames from all videos using parallel workers with GPU acceleration

cd ~/Repos/ahriuwu

WORKERS=2
VIDEOS_DIR="data/raw/videos"
FRAMES_DIR="data/processed/frames"

extract_video() {
    local vid="$1"
    echo "[$(date +%H:%M:%S)] Starting: $vid"
    mkdir -p "$FRAMES_DIR/$vid"
    # Extract at 256x256 directly (tokenizer input size) - saves 16x disk space
    ffmpeg -hwaccel cuda -i "$VIDEOS_DIR/$vid.mp4" -vf "fps=20,scale=256:256" -start_number 0 -y -qscale:v 2 \
        "$FRAMES_DIR/$vid/frame_%06d.jpg" -loglevel error 2>&1
    echo "[$(date +%H:%M:%S)] Done: $vid"
}

# Get list of unprocessed videos
mapfile -t remaining < <(comm -23 \
    <(ls "$VIDEOS_DIR"/*.mp4 2>/dev/null | xargs -n1 basename | sed 's/.mp4//' | sort) \
    <(ls -d "$FRAMES_DIR"/*/ 2>/dev/null | xargs -n1 basename | sort 2>/dev/null))

total=${#remaining[@]}

echo "=== Frame Extraction ==="
echo "Workers: $WORKERS"
echo "Remaining: $total videos"
echo ""

# Process with xargs for proper parallelism (starts new job as soon as one finishes)
printf '%s\n' "${remaining[@]}" | xargs -P $WORKERS -I {} bash -c "
    vid={}
    echo \"[\$(date +%H:%M:%S)] Starting: \$vid\"
    mkdir -p \"$FRAMES_DIR/\$vid\"
    ffmpeg -hwaccel cuda -i \"$VIDEOS_DIR/\$vid.mp4\" -vf \"fps=20,scale=256:256\" -start_number 0 -y -qscale:v 2 \
        \"$FRAMES_DIR/\$vid/frame_%06d.jpg\" -loglevel error 2>&1
    echo \"[\$(date +%H:%M:%S)] Done: \$vid\"
"

echo ""
echo "=== Complete ==="
echo "Total dirs: $(ls -d "$FRAMES_DIR"/*/ | wc -l)"
