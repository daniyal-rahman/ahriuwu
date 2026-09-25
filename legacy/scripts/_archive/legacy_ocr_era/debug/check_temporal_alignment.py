#!/usr/bin/env python3
"""Check temporal alignment of gold events.

Investigates whether gold_gained labels are aligned with the actual
minion death frames or delayed (e.g., OCR detecting gold popup after death).
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


def find_gold_events(features_dir: str, n_events: int = 10):
    """Find frames where gold_gained > 0."""
    features_dir = Path(features_dir)

    all_events = []

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
            if gold > 0:
                all_events.append({
                    "video_id": video_dir.name,
                    "frame_idx": i,
                    "gold_gained": gold,
                    "total_frames": len(frames),
                })

    print(f"Found {len(all_events)} gold events across all videos")

    # Sample events with enough context (not near start/end)
    valid_events = [e for e in all_events if e["frame_idx"] >= 10 and e["frame_idx"] < e["total_frames"] - 10]

    random.seed(42)
    sampled = random.sample(valid_events, min(n_events, len(valid_events)))

    return sampled


def analyze_event_context(event: dict, features_dir: str, window: int = 5):
    """Analyze frames around a gold event."""
    features_dir = Path(features_dir)
    video_dir = features_dir / event["video_id"]

    # Load features
    with open(video_dir / "features.json") as f:
        features = json.load(f)

    frames = features.get("frames", [])
    center = event["frame_idx"]

    print(f"\n{'='*70}")
    print(f"Video: {event['video_id']}, Frame {center}, Gold: +{event['gold_gained']}")
    print(f"{'='*70}")

    # Show context window
    print(f"\n{'t':>4} | {'gold':>6} | {'cs':>4} | {'hp%':>5} | {'dead':>5} | {'minions':>8} | notes")
    print(f"{'-'*4}-+-{'-'*6}-+-{'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*8}-+{'-'*20}")

    for t in range(center - window, center + window + 1):
        if t < 0 or t >= len(frames):
            continue

        frame = frames[t]
        gold = frame.get("gold_gained", 0)
        cs = frame.get("cs", 0)
        hp_pct = frame.get("hp_percent", 0)
        is_dead = frame.get("is_dead", False)

        # Check for nearby minions (if available)
        minions = frame.get("nearby_minions", "?")

        # Mark the gold event frame
        marker = " <-- GOLD" if t == center else ""
        if gold > 0 and t != center:
            marker = f" <-- +{gold}g"

        rel_t = t - center
        print(f"{rel_t:+4} | {gold:>6} | {cs:>4} | {hp_pct:>5.1f} | {str(is_dead):>5} | {str(minions):>8} | {marker}")

    # Check for CS changes
    cs_values = []
    for t in range(max(0, center - window), min(len(frames), center + window + 1)):
        cs_values.append((t - center, frames[t].get("cs", 0)))

    # Find when CS increased
    cs_changes = []
    for i in range(1, len(cs_values)):
        if cs_values[i][1] > cs_values[i-1][1]:
            cs_changes.append(cs_values[i][0])

    if cs_changes:
        print(f"\nCS increased at t={cs_changes} (relative to gold frame)")

    return {
        "video_id": event["video_id"],
        "gold_frame": center,
        "cs_change_frames": cs_changes,
    }


def check_frame_images(event: dict, features_dir: str, frames_subdir: str = "frames", window: int = 5):
    """Check if frame images exist and report their availability."""
    features_dir = Path(features_dir)
    video_dir = features_dir / event["video_id"]
    frames_dir = video_dir / frames_subdir

    if not frames_dir.exists():
        print(f"\n[WARN] Frames directory not found: {frames_dir}")
        return

    center = event["frame_idx"]

    print(f"\nFrame images around t={center}:")
    for t in range(center - window, center + window + 1):
        # Try common naming patterns
        for pattern in [f"frame_{t:06d}.png", f"frame_{t:05d}.png", f"{t:06d}.png", f"{t}.png"]:
            frame_path = frames_dir / pattern
            if frame_path.exists():
                img = Image.open(frame_path)
                print(f"  t={t-center:+3}: {pattern} ({img.size[0]}x{img.size[1]})")
                break
        else:
            # Check what files exist
            existing = list(frames_dir.glob(f"*{t}*"))
            if existing:
                print(f"  t={t-center:+3}: Found {existing[0].name}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--n-events", type=int, default=10)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("TEMPORAL ALIGNMENT CHECK")
    print("=" * 70)
    print(f"Looking for gold events and analyzing context window of ±{args.window} frames")

    # Find gold events
    events = find_gold_events(args.features_dir, args.n_events)

    # Analyze each event
    results = []
    for event in events:
        result = analyze_event_context(event, args.features_dir, args.window)
        results.append(result)

        if args.check_images:
            check_frame_images(event, args.features_dir, window=args.window)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check if CS changes align with gold
    aligned = 0
    delayed = 0
    early = 0

    for r in results:
        if not r["cs_change_frames"]:
            continue
        # Check if CS changed at same frame as gold (t=0) or before/after
        if 0 in r["cs_change_frames"]:
            aligned += 1
        elif any(t < 0 for t in r["cs_change_frames"]):
            delayed += 1  # CS changed before gold label
        else:
            early += 1  # CS changed after gold label

    print(f"\nCS change alignment with gold_gained label:")
    print(f"  Aligned (CS changes at gold frame): {aligned}")
    print(f"  Delayed (CS changes before gold):   {delayed}")
    print(f"  Early (CS changes after gold):      {early}")

    if delayed > aligned:
        print("\n[WARN] Gold labels appear to be DELAYED relative to CS changes!")
        print("       This means the reward signal comes AFTER the action.")
        print("       The model may be learning to predict rewards from post-action states.")


if __name__ == "__main__":
    main()
