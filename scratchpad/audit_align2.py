#!/usr/bin/env python3
"""Sharp frame<->label alignment test using TELEPORT events.

When champion_world jumps >1500 units in one frame (recall completes, death
respawn, Flash), the champion-locked camera cuts and the PNG changes almost
completely. Locate the PNG-side cut and compare to the label-side jump index.
An offset != 0, or an offset that GROWS through the game, proves the synthetic
`gt = rec_start + i/20` timebase drifts from the recorder.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/srv/nfs/datasets/lol_replays_16_9_772")
CACHE = Path("/srv/nfs/projects/ahriuwu/scratchpad/audit_cache")
W = 12  # search window in frames


def png_diff_window(mid, center):
    d = ROOT / mid / "frames"
    idxs = list(range(center - W - 1, center + W + 1))
    ims = {}
    for i in idxs:
        p = d / f"{i:06d}.png"
        if p.exists():
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                ims[i] = im.astype(np.float32)
    diffs = {}
    for i in idxs[1:]:
        if i in ims and i - 1 in ims:
            diffs[i] = float(np.abs(ims[i] - ims[i - 1]).mean())
    return diffs


def main():
    mids = sys.argv[1:] or [p.stem for p in sorted(CACHE.glob("*.npz"))]
    offsets = []
    rows = []
    for mid in mids:
        z = np.load(CACHE / f"{mid}.npz", allow_pickle=True)
        cw = np.stack([z["cw_x"], z["cw_y"]], 1)
        T = len(cw)
        d = np.full(T, 0.0)
        d[1:] = np.linalg.norm(np.diff(cw, axis=0), axis=1)
        d = np.nan_to_num(d)
        cand = np.where(d > 1500)[0]
        # keep well-separated events, away from the edges
        keep = []
        for c in cand:
            if c < W + 5 or c > T - W - 5:
                continue
            if keep and c - keep[-1] < 40:
                continue
            keep.append(int(c))
        keep = keep[:14]
        for c in keep:
            diffs = png_diff_window(mid, c)
            if len(diffs) < 2 * W - 2:
                continue
            k = max(diffs, key=lambda i: diffs[i])
            base = np.median(list(diffs.values()))
            if diffs[k] < 3 * base:  # no clear cut in the PNGs
                continue
            off = k - c
            offsets.append(off)
            rows.append((mid, c, off, diffs[k] / max(base, 1e-6), c / T))
        if rows:
            mo = [r[2] for r in rows if r[0] == mid]
            if mo:
                print(f"{mid}: {len(mo)} teleports, offsets {mo}", flush=True)
    if offsets:
        o = np.array(offsets)
        print("\n" + "=" * 60)
        print(f"N events {len(o)}  offset: mean {o.mean():+.2f} median {np.median(o):+.1f} "
              f"std {o.std():.2f} min {o.min():+d} max {o.max():+d}")
        u, c = np.unique(o, return_counts=True)
        print("offset histogram:", dict(zip(u.tolist(), c.tolist())))
        # drift with game progress?
        fr = np.array([r[4] for r in rows])
        if len(o) > 5:
            print(f"corr(offset, position-in-game) = {np.corrcoef(fr, o)[0,1]:+.3f} "
                  f"(nonzero => accumulating timebase drift)")
        np.save("/srv/nfs/projects/ahriuwu/scratchpad/audit_align_offsets.npy", o)


if __name__ == "__main__":
    main()
