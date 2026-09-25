#!/usr/bin/env python3
"""Exact replication of ReplayLatentSequenceDataset._parse_movement (schema 3)
+ PolicyHead.discretize_movement, then attribute every resulting BC movement
target transition to a REAL player command or to camera-projection drift.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

CACHE = Path("/srv/nfs/projects/ahriuwu/scratchpad/audit_cache")
DEADBAND = 0.01


def run(BINS):
    t = Counter()
    hist = np.zeros((BINS, BINS), dtype=np.int64)
    dwell = []
    for p in sorted(CACHE.glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        m = json.loads(str(z["meta"][0]))
        sw, sh = m["screen_resolution"]
        T = m["T"]
        sx, sy = z["cur_sx"] / sw, z["cur_sy"] / sh
        wx, wy = z["cur_wx"], z["cur_wy"]
        have = np.isfinite(sx) & np.isfinite(sy)

        mv = np.full((T, 2), 0.5)
        src_world = np.full((T, 2), np.nan)   # cursor_world behind the accepted target
        last = None
        lastw = None
        for i in range(T):
            if have[i]:
                nx, ny = sx[i], sy[i]
                if last is None or abs(nx - last[0]) > DEADBAND or abs(ny - last[1]) > DEADBAND:
                    last = (nx, ny)
                    lastw = (wx[i], wy[i])
            if last is not None:
                mv[i] = last
                src_world[i] = lastw
        idx = np.clip(np.round(mv * (BINS - 1)).astype(int), 0, BINS - 1)
        np.add.at(hist, (idx[:, 0], idx[:, 1]), 1)

        trans = np.zeros(T, dtype=bool)
        trans[1:] = (idx[1:] != idx[:-1]).any(1)
        # world behind the target changed => a genuinely new command
        wchg = np.zeros(T, dtype=bool)
        wchg[1:] = ((src_world[1:, 0] != src_world[:-1, 0])
                    | (src_world[1:, 1] != src_world[:-1, 1]))
        t["frames"] += T
        t["trans"] += int(trans.sum())
        t["trans_real"] += int((trans & wchg).sum())
        t["trans_drift"] += int((trans & ~wchg).sum())
        t["cmd_missed"] += int((wchg & ~trans).sum())
        t["wchg"] += int(wchg.sum())
        t["nocursor"] += int((~have).sum())
        t["nocursor_world_ok"] += int(((~have) & np.isfinite(wx)).sum())
        # dwell = frames between bin transitions
        tp = np.where(trans)[0]
        if len(tp) > 1:
            dwell.append(np.diff(tp))
    d = np.concatenate(dwell) if dwell else np.array([1])
    print(f"--- BINS={BINS} (bin width {100/(BINS-1):.2f}% screen = "
          f"{1280/(BINS-1):.0f}px x, {720/(BINS-1):.0f}px y) ---")
    print(f"BC movement-target bin transitions : {t['trans']:,} "
          f"({100*t['trans']/t['frames']:.2f}% of frames)")
    print(f"  from a genuinely new command     : {t['trans_real']:,} "
          f"({100*t['trans_real']/max(t['trans'],1):.1f}%)")
    print(f"  from camera-projection DRIFT     : {t['trans_drift']:,} "
          f"({100*t['trans_drift']/max(t['trans'],1):.1f}%)  <-- pure label noise")
    print(f"new commands that produced NO transition (lost): {t['cmd_missed']:,} "
          f"({100*t['cmd_missed']/max(t['wchg'],1):.1f}% of the "
          f"{t['wchg']:,} accepted command updates)")
    print(f"dwell between transitions: p50 {np.median(d):.0f} p90 {np.percentile(d,90):.0f} frames")
    nz = (hist > 0).sum()
    pr = hist / hist.sum()
    ent = -(pr[pr > 0] * np.log(pr[pr > 0])).sum()
    print(f"target distribution: {nz}/{BINS*BINS} cells used, top cell "
          f"{100*hist.max()/hist.sum():.1f}%, entropy {ent:.3f} nats = "
          f"{np.exp(ent):.1f} effective cells (uniform={np.log(BINS*BINS):.3f})")
    # marginal per-axis usage
    mx = hist.sum(1) / hist.sum()
    my = hist.sum(0) / hist.sum()
    print(f"  x-marginal: {np.array2string(mx, precision=3, suppress_small=True)}")
    print(f"  y-marginal: {np.array2string(my, precision=3, suppress_small=True)}")
    return hist


if __name__ == "__main__":
    h = run(21)
    print()
    run(41)
    print()
    run(11)
    np.save("/srv/nfs/projects/ahriuwu/scratchpad/audit_bin_hist21.npy", h)
