#!/usr/bin/env python3
"""Build + cache the shared event table for the last-hit triage (all matches)."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasthit_events import build_events, load_match  # noqa: E402

OUT = "scratchpad/lh_events.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--frames-root", default="/srv/nfs/datasets/lol_replays_16_9_772")
    args = ap.parse_args()

    mids = sorted(d for d in os.listdir(args.frames_root)
                  if os.path.isdir(os.path.join(args.frames_root, d)))
    cols = {k: [] for k in ["mid", "anchor", "frame", "y", "csx", "csy", "level", "hp"]}
    for mid in mids:
        lp = os.path.join(args.frames_root, mid, "labels.json")
        fp = os.path.join(args.frames_root, mid, "frames")
        if not (os.path.exists(lp) and os.path.isdir(fp)):
            continue
        try:
            m = load_match(mid, args.frames_root)
        except Exception as e:  # noqa: BLE001
            print(f"!! {mid}: {e}", flush=True)
            continue
        nf = len(os.listdir(fp))
        for anchor in ["commit", "gold"]:
            t, y, _ = build_events(m, anchor)
            keep = t + 20 < nf                       # frames on disk must exist
            t, y = t[keep], y[keep]
            if len(t) == 0:
                continue
            cols["mid"] += [mid] * len(t)
            cols["anchor"] += [anchor] * len(t)
            cols["frame"].append(t)
            cols["y"].append(y)
            cols["csx"].append(m["cs_x"][t])
            cols["csy"].append(m["cs_y"][t])
            cols["level"].append(m["level"][t])
            cols["hp"].append(m["hp"][t])
        print(f"{mid}: T={m['T']} pngs={nf}", flush=True)
    np.savez_compressed(
        args.out,
        mid=np.array(cols["mid"]), anchor=np.array(cols["anchor"]),
        frame=np.concatenate(cols["frame"]), y=np.concatenate(cols["y"]),
        csx=np.concatenate(cols["csx"]), csy=np.concatenate(cols["csy"]),
        level=np.concatenate(cols["level"]), hp=np.concatenate(cols["hp"]))
    print("wrote", args.out, len(cols["mid"]))


if __name__ == "__main__":
    main()
