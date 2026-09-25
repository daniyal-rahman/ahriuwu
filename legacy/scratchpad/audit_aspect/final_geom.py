"""Definitive bar-geometry stats: width histogram split by bar height.

h==1  -> minion bars (verified by hand against NA1_5549995114/005000.png wave)
h>=3  -> champion / turret bars
"""
import glob, json, os
import numpy as np, cv2

ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
OUT = "/srv/nfs/projects/ahriuwu/scratchpad/audit_aspect"


def bar_masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    blue = (H >= 95) & (H <= 125) & (S >= 120) & (V >= 120)
    red = (((H <= 8) | (H >= 172)) & (S >= 140) & (V >= 110))
    return blue.astype(np.uint8), red.astype(np.uint8)


rows = []
matches = sorted(d for d in os.listdir(ROOT) if d.startswith("NA1_"))[:12]
for m in matches:
    fl = sorted(glob.glob(f"{ROOT}/{m}/frames/*.png"))
    if len(fl) < 8000:
        continue
    for idx in range(4000, 8001, 100):
        im = cv2.imread(fl[idx])
        if im is None:
            continue
        b, r = bar_masks(im)
        for team, mk in (("blue", b), ("red", r)):
            n, _, st, _ = cv2.connectedComponentsWithStats(mk, 8)
            for i in range(1, n):
                x, y, w, h, a = st[i]
                if w < 2 or w / max(h, 1) < 2.5 or a / float(w * h) < 0.6 or h > 6:
                    continue
                rows.append((int(w), int(h), team))
json.dump(rows, open(f"{OUT}/final_geom.json", "w"))

W = np.array([r[0] for r in rows]); Hh = np.array([r[1] for r in rows])
print(f"n_components={len(rows)}  matches={len(matches)}  frames_per_match=41")
for lo, hi, name in ((1, 1, "h==1  MINION bars"), (2, 2, "h==2"), (3, 6, "h>=3  CHAMPION/turret bars")):
    sel = (Hh >= lo) & (Hh <= hi)
    ws = W[sel]
    if not len(ws):
        continue
    hist = np.bincount(ws, minlength=32)[:32]
    mx = max(hist.max(), 1)
    print(f"\n--- {name}   n={len(ws)}  mode={int(hist.argmax())}  "
          f"p50={np.percentile(ws,50):.0f} p90={np.percentile(ws,90):.0f} p99={np.percentile(ws,99):.0f}")
    for i, c in enumerate(hist):
        if c:
            print(f"   w={i:3d} n={c:5d} {'#' * (c * 55 // mx)}")

# minion-only: widths <= 12 and h==1  => the fill fraction distribution
sel = (Hh == 1) & (W <= 12)
ws = W[sel]
print(f"\nMINION fill widths (h==1, w<=12): n={len(ws)}  unique={sorted(set(ws.tolist()))}")
print("=> full-bar width (mode of the upper edge):", int(np.bincount(ws).argmax()))
