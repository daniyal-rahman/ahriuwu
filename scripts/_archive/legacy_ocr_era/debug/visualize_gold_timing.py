#!/usr/bin/env python3
"""Visualize frames around gold events to check timing.

Creates a composite image showing frames t-3 to t+3 around gold events.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_gold_event_strip(video_id: str, frame_idx: int, features_dir: str, output_path: str, window: int = 3):
    """Create a horizontal strip of frames around a gold event."""
    features_dir = Path(features_dir)
    video_dir = features_dir / video_id
    frames_dir = video_dir / "frames"

    # Load features
    with open(video_dir / "features.json") as f:
        features = json.load(f)

    frames_data = features.get("frames", [])

    # Collect frames
    images = []
    labels = []

    for t in range(frame_idx - window, frame_idx + window + 1):
        if t < 0 or t >= len(frames_data):
            continue

        frame_path = frames_dir / f"frame_{t:06d}.jpg"
        if not frame_path.exists():
            continue

        img = Image.open(frame_path)

        # Get gold value for this frame
        gold = frames_data[t].get("gold_gained", 0)
        rel_t = t - frame_idx
        label = f"t={rel_t:+d}"
        if gold > 0:
            label += f"\n+{gold}g"

        images.append(img)
        labels.append(label)

    if not images:
        print(f"No images found for {video_id} frame {frame_idx}")
        return

    # Create composite
    # Resize images for display
    target_width = 320
    target_height = int(images[0].height * target_width / images[0].width)

    resized = [img.resize((target_width, target_height), Image.LANCZOS) for img in images]

    # Create output image
    padding = 5
    label_height = 50
    total_width = len(resized) * (target_width + padding) - padding
    total_height = target_height + label_height

    composite = Image.new('RGB', (total_width, total_height), (30, 30, 30))
    draw = ImageDraw.Draw(composite)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    x = 0
    for img, label in zip(resized, labels):
        composite.paste(img, (x, 0))

        # Draw label
        text_x = x + target_width // 2
        text_y = target_height + 5

        # Highlight the gold frame
        if "+0" in label:
            draw.rectangle([x, 0, x + target_width, target_height], outline='yellow', width=3)

        # Draw label (handle multiline)
        lines = label.split('\n')
        color = 'yellow' if 'g' in label else 'white'
        for j, line in enumerate(lines):
            draw.text((text_x, text_y + j * 18), line, fill=color, font=font, anchor='mt')

        x += target_width + padding

    composite.save(output_path)
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--output-dir", default="docs/images/gold_timing")
    parser.add_argument("--n-events", type=int, default=5)
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find gold events
    features_dir = Path(args.features_dir)
    events = []

    for video_dir in sorted(features_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        features_path = video_dir / "features.json"
        if not features_path.exists():
            continue

        with open(features_path) as f:
            features = json.load(f)

        frames = features.get("frames", [])
        for i, frame in enumerate(frames):
            gold = frame.get("gold_gained", 0)
            # Look for significant gold (minion kills, not passive gold)
            if 15 <= gold <= 100:  # Typical minion kill range
                events.append({
                    "video_id": video_dir.name,
                    "frame_idx": i,
                    "gold": gold,
                })

    print(f"Found {len(events)} minion kill gold events")

    # Sample diverse events
    import random
    random.seed(42)

    # Group by gold value to get variety
    by_gold = {}
    for e in events:
        g = e["gold"]
        if g not in by_gold:
            by_gold[g] = []
        by_gold[g].append(e)

    selected = []
    for gold_val in sorted(by_gold.keys()):
        if len(selected) >= args.n_events:
            break
        selected.append(random.choice(by_gold[gold_val]))

    # Create visualizations
    for i, event in enumerate(selected):
        output_path = output_dir / f"gold_event_{i+1}_{event['video_id'][:8]}_f{event['frame_idx']}.png"
        print(f"\nEvent {i+1}: {event['video_id']} frame {event['frame_idx']} (+{event['gold']}g)")
        create_gold_event_strip(
            event["video_id"],
            event["frame_idx"],
            args.features_dir,
            str(output_path),
            window=args.window
        )


if __name__ == "__main__":
    main()
