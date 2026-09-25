#!/usr/bin/env python3
"""Direct scale check of pipeline.project(), over 1-second windows.

Predicted: the label's own champion_screen for the SAME world point, i.e. take a
held cursor_world point and ask how far project() says it moves over 20 frames.
Measured: phase correlation of a CENTRE CROP of the PNGs (crop matches the depth
of the predicted point, so ground-plane parallax cannot explain a mismatch).

Also reports px-per-world-unit both ways, cross-checked against champion move
speed (League: 315-450 units/s) as a physical anchor.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/srv/nfs/datasets/lol_replays_16_9_772")
CACHE = Path("/srv/nfs/projects/ahriuwu/scratchpad/audit_cache")
SZ = 352
K = 20          # window length in frames (1 s of game time)
CROP = 176      # centre crop side


def main():
    mids = sys.argv[1:] or [p.stem for p in sorted(CACHE.glob("*.npz"))[:4]]
    P, M, W = [], [], []
    for mid in mids:
        z = np.load(CACHE / f"{mid}.npz", allow_pickle=True)
        m = json.loads(str(z["meta"][0]))
        sw, sh = m["screen_resolution"]
        T = m["T"]
        sx, sy = z["cur_sx"] / sw, z["cur_sy"] / sh
        wx, wy = z["cur_wx"], z["cur_wy"]
        cwx, cwy = z["cw_x"], z["cw_y"]
        # windows where cursor_world is constant across the whole window AND the
        # champion is moving steadily
        ok = []
        for i in range(200, T - K - 200, 53):
            a, b = i, i + K
            seg_w = wx[a:b + 1]
            if not np.isfinite(seg_w).all() or seg_w.min() != seg_w.max():
                continue
            if not (np.isfinite(sx[a]) and np.isfinite(sx[b])):
                continue
            if not (0.30 < sx[a] < 0.70 and 0.30 < sy[a] < 0.70):
                continue
            d = np.hypot(cwx[b] - cwx[a], cwy[b] - cwy[a])
            path = np.nansum(np.hypot(np.diff(cwx[a:b + 1]), np.diff(cwy[a:b + 1])))
            if not (np.isfinite(d) and d > 150 and d > 0.9 * path):   # straight, fast
                continue
            ok.append((a, b, d))
            if len(ok) >= 60:
                break
        d0 = ROOT / mid / "frames"
        c0 = (SZ - CROP) // 2
        win = np.outer(np.hanning(CROP), np.hanning(CROP))
        n = 0
        for a, b, d in ok:
            ia = cv2.imread(str(d0 / f"{a:06d}.png"), cv2.IMREAD_GRAYSCALE)
            ib = cv2.imread(str(d0 / f"{b:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if ia is None or ib is None:
                continue
            A = ia[c0:c0 + CROP, c0:c0 + CROP].astype(np.float32) * win
            B = ib[c0:c0 + CROP, c0:c0 + CROP].astype(np.float32) * win
            (mx, my), resp = cv2.phaseCorrelate(A, B)
            if resp < 0.15 or max(abs(mx), abs(my)) > CROP * 0.4:
                continue
            px = (sx[b] - sx[a]) * SZ
            py = (sy[b] - sy[a]) * SZ
            P.append((px, py))
            M.append((mx, my))
            W.append(d)
            n += 1
        print(f"{mid}: {n} windows (of {len(ok)} candidates)", flush=True)

    P, M, W = np.array(P), np.array(M), np.array(W)
    print("\n" + "=" * 70)
    print(f"N = {len(P)} one-second windows")
    for k, ax in ((0, "x"), (1, "y")):
        p, mm = P[:, k], M[:, k]
        sl = float((p * mm).sum() / (p * p).sum())
        print(f"  {ax}: measured_scroll = {sl:.4f} * project()_prediction "
              f"(r = {np.corrcoef(p, mm)[0,1]:.3f})")
    pm = np.hypot(P[:, 0], P[:, 1])
    mm = np.hypot(M[:, 0], M[:, 1])
    print(f"  magnitude: measured = {float((pm*mm).sum()/(pm*pm).sum()):.4f} * predicted")
    print(f"\n  px(352) per world unit:")
    print(f"    from project()   : {np.median(pm/W):.4f}")
    print(f"    from the pixels  : {np.median(mm/W):.4f}")
    print(f"  champion speed over these windows: median {np.median(W)/(K/20):.0f} world units/s "
          f"(League champions run 315-450)")


if __name__ == "__main__":
    main()
