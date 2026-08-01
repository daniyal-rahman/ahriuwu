#!/usr/bin/env python3
"""Generate placeholder labels.json for unlabeled YT latents so the
action-conditioned ReplayLatentSequenceDataset can ingest them alongside replays.

YT games have no action labels. We emit a minimal, invariant-satisfying labels.json
(frames[i].frame == i, label=null) so:
  * movement parses to the (0.5, 0.5) fallback  -> cursor_valid=False in the dataset
    (match_id has no NA1_ prefix) -> the model substitutes no_action_embed. Good:
    unlabeled video is modeled without action conditioning (the paper recipe).
  * abilities parse to all-zero (no clicks.json)  -> the no-press class, ~99% correct.
  * rewards are computed but UNUSED by the dynamics x-prediction loss.

    python scripts/gen_yt_placeholder_labels.py --latents-dir <yt_pt_dir> --out <labels_root> [--fps 30]
"""
import argparse
import glob
import json
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", required=True, help="dir of YT <id>.pt latents")
    ap.add_argument("--out", required=True, help="labels_root: writes <id>/labels.json")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--res", type=int, nargs=2, default=[1280, 720])
    args = ap.parse_args()

    pts = [p for p in sorted(glob.glob(f"{args.latents_dir}/*.pt"))
           if os.path.basename(p) != "index.pt"]
    n = 0
    for p in pts:
        mid = os.path.basename(p)[:-3]
        if mid.startswith("NA1_"):
            continue  # a real replay — has genuine labels elsewhere, skip
        fi = torch.load(p, weights_only=True)["frame_indices"]
        maxfi = int(fi.max())
        labels = {
            "match_id": mid, "champion": "Garen", "team": "BLUE", "slot": 0,
            "lane_opponent": "", "fps": args.fps,
            "screen_resolution": list(args.res), "frame_resolution": list(args.res),
            "total_frames": maxfi + 1, "projection": None, "action_distribution": {},
            "frames": [{"frame": i, "gt": i / float(args.fps), "label": None}
                       for i in range(maxfi + 1)],
        }
        d = os.path.join(args.out, mid)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "labels.json"), "w") as f:
            json.dump(labels, f)
        n += 1
    print(f"wrote placeholder labels for {n} YT games -> {args.out}")


if __name__ == "__main__":
    main()
