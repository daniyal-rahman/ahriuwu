#!/usr/bin/env python3
"""Frame <-> label temporal alignment test.

Camera is champion-locked, so when champion_world moves the whole frame scrolls.
Cross-correlate |d champion_world/dt| (from labels) against mean-abs-pixel-diff
(from the PNGs) over integer lags. Peak lag != 0 => frame/label off-by-N.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/srv/nfs/datasets/lol_replays_16_9_772")
CACHE = Path("/srv/nfs/projects/ahriuwu/scratchpad/audit_cache")


def frame_motion(mid, i0, n):
    d = ROOT / mid / "frames"
    prev = None
    out = np.full(n, np.nan)
    for k in range(n):
        p = d / f"{i0 + k:06d}.png"
        if not p.exists():
            prev = None
            continue
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if prev is not None:
            out[k] = np.abs(im - prev).mean()
        prev = im
    return out


def main():
    mids = sys.argv[1:] or [p.stem for p in sorted(CACHE.glob("*.npz"))[:6]]
    for mid in mids:
        z = np.load(CACHE / f"{mid}.npz", allow_pickle=True)
        cw = np.stack([z["cw_x"], z["cw_y"]], 1)
        T = len(cw)
        # pick a mid-game window with continuous labels
        i0 = min(4000, max(0, T - 2600))
        n = min(2400, T - i0)
        vel = np.full(n, np.nan)
        seg = cw[i0:i0 + n]
        d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        vel[1:] = d
        fm = frame_motion(mid, i0, n)
        ok = np.isfinite(vel) & np.isfinite(fm)
        v = vel[ok]
        f = fm[ok]
        # clip world-teleports (recall/death) which break the linear relation
        v = np.clip(v, 0, np.percentile(v, 99.5))
        v = (v - v.mean()) / (v.std() + 1e-9)
        f = (f - f.mean()) / (f.std() + 1e-9)
        lags = range(-8, 9)
        cors = {}
        for L in lags:
            if L < 0:
                a, b = v[-L:], f[:len(f) + L]
            elif L > 0:
                a, b = v[:len(v) - L], f[L:]
            else:
                a, b = v, f
            cors[L] = float((a * b).mean())
        best = max(cors, key=lambda k: cors[k])
        print(f"{mid}: n={ok.sum()} best_lag={best:+d} r={cors[best]:.3f} | "
              f"r(0)={cors[0]:.3f} " +
              " ".join(f"{L:+d}:{cors[L]:.2f}" for L in range(-3, 4)), flush=True)


if __name__ == "__main__":
    main()
