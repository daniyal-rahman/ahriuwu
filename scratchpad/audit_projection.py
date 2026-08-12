#!/usr/bin/env python3
"""Is pipeline.project() calibrated against the actual rendered pixels?

Every screen coordinate in labels.json (champion_screen, visible_heroes.screen,
action.screen and — critically — cursor.screen, the ONLY BC movement target) is
produced by project(world, camera) with HARD-CODED, "empirically fit" constants
(FOV_V=40 deg, TILT=56 deg, FLOOR_Y=52, CAM_Y=1912, CAM_Z_OFFSET=-1292).

Test without needing raw_cam.json: on frames where cursor.world is UNCHANGED,
cursor.screen moves purely because the camera moved. That predicted screen
displacement of a fixed world point must equal the measured optical scroll of
the ground between the two PNGs. Measure the latter with phase correlation and
regress measured-on-predicted. Slope 1.0 => calibrated; slope != 1 => every
projected coordinate is scaled wrong by that factor.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/srv/nfs/datasets/lol_replays_16_9_772")
CACHE = Path("/srv/nfs/projects/ahriuwu/scratchpad/audit_cache")
SZ = 352


def main():
    mids = sys.argv[1:] or [p.stem for p in sorted(CACHE.glob("*.npz"))[:5]]
    P, M = [], []
    lag_scores = {L: [] for L in (-2, -1, 0, 1, 2)}
    for mid in mids:
        z = np.load(CACHE / f"{mid}.npz", allow_pickle=True)
        m = json.loads(str(z["meta"][0]))
        sw, sh = m["screen_resolution"]
        T = m["T"]
        sx, sy = z["cur_sx"] / sw, z["cur_sy"] / sh
        wx, wy = z["cur_wx"], z["cur_wy"]
        hold = np.zeros(T, bool)
        hold[1:] = (wx[1:] == wx[:-1]) & (wy[1:] == wy[:-1]) & np.isfinite(wx[1:])
        near = (np.abs(sx - 0.5) < 0.18) & (np.abs(sy - 0.5) < 0.18)
        dsx = np.full(T, np.nan)
        dsy = np.full(T, np.nan)
        dsx[1:] = (sx[1:] - sx[:-1]) * SZ
        dsy[1:] = (sy[1:] - sy[:-1]) * SZ
        mag = np.hypot(dsx, dsy)
        cand = np.where(hold & near & np.isfinite(mag) & (mag > 0.8) & (mag < 12))[0]
        rng = np.random.default_rng(0)
        if len(cand) > 400:
            cand = rng.choice(cand, 400, replace=False)
        d = ROOT / mid / "frames"
        win = np.outer(np.hanning(SZ), np.hanning(SZ))
        cache = {}

        def g(i):
            if i not in cache:
                p = d / f"{i:06d}.png"
                im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                cache[i] = None if im is None else im.astype(np.float32)
            return cache[i]

        n_ok = 0
        for i in sorted(cand.tolist()):
            for L in lag_scores:
                a, b = g(i - 1 + L), g(i + L)
                if a is None or b is None:
                    continue
                (mx, my), resp = cv2.phaseCorrelate(a * win, b * win)
                if resp < 0.20:
                    continue
                # phaseCorrelate(a,b) = shift taking a -> b, same sense as ds
                if L == 0:
                    P.append((dsx[i], dsy[i]))
                    M.append((mx, my))
                    n_ok += 1
                lag_scores[L].append(
                    (mx - dsx[i]) ** 2 + (my - dsy[i]) ** 2)
            if len(cache) > 60:
                cache.clear()
        print(f"{mid}: {n_ok} usable pairs", flush=True)

    P = np.array(P)
    M = np.array(M)
    print("\n" + "=" * 70)
    print(f"N pairs = {len(P)}")
    for k, ax in ((0, "x"), (1, "y")):
        p, mm = P[:, k], M[:, k]
        sl = float((p * mm).sum() / (p * p).sum())
        r = float(np.corrcoef(p, mm)[0, 1])
        print(f"  {ax}: measured = {sl:.4f} x predicted   (r = {r:.3f})   "
              f"=> project() {'OK' if abs(sl-1) < 0.06 else 'MISCALIBRATED'} "
              f"by {100*(sl-1):+.1f}% on this axis")
    # residual as a fraction of screen
    res = np.hypot(M[:, 0] - P[:, 0], M[:, 1] - P[:, 1])
    print(f"  residual |measured-predicted|: p50 {np.median(res):.2f}px "
          f"p90 {np.percentile(res,90):.2f}px (of {SZ}px)")
    print("\n  lag sweep (mean squared px error; lowest = best frame<->label offset):")
    for L, v in sorted(lag_scores.items()):
        if v:
            print(f"    lag {L:+d}: {np.mean(v):8.3f}  (n={len(v)})")


if __name__ == "__main__":
    main()
