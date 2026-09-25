"""Measure minion/champion HP-bar geometry in the 352x352 squished frames.

Empirical: color-threshold the characteristic spectator HP-bar colours (blue =
ally-team, red = enemy-team), connected-component, filter to bar-like shapes,
report width/height distributions.
"""
import glob, json, os, sys
import numpy as np
import cv2

ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
OUT = "/srv/nfs/projects/ahriuwu/scratchpad/audit_aspect"

# HSV ranges (OpenCV: H 0-179, S 0-255, V 0-255)
def masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    blue = (H >= 95) & (H <= 125) & (S >= 120) & (V >= 120)
    red = (((H <= 8) | (H >= 172)) & (S >= 140) & (V >= 110))
    return blue.astype(np.uint8), red.astype(np.uint8)


def components(mask, upscale=1):
    """Return list of (x,y,w,h,area) for bar-like components."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if h < 1 * upscale or h > 6 * upscale:
            continue
        if w < 3 * upscale:
            continue
        if w / max(h, 1) < 2.0:
            continue
        fill = a / float(w * h)
        if fill < 0.55:
            continue
        out.append((x, y, w, h, a, fill))
    return out


def main():
    matches = sorted(d for d in os.listdir(ROOT) if d.startswith("NA1_"))[:6]
    rows = []
    for m in matches:
        fr = sorted(glob.glob(f"{ROOT}/{m}/frames/*.png"))
        if len(fr) < 8000:
            continue
        for idx in range(4000, 8001, 250):
            im = cv2.imread(fr[idx])
            if im is None:
                continue
            b, r = masks(im)
            for team, mk in (("blue", b), ("red", r)):
                for (x, y, w, h, a, fill) in components(mk):
                    rows.append(dict(match=m, frame=idx, team=team, x=int(x), y=int(y),
                                     w=int(w), h=int(h), area=int(a), fill=round(float(fill), 3)))
    with open(f"{OUT}/bars_raw.json", "w") as f:
        json.dump(rows, f)
    W = np.array([r["w"] for r in rows])
    H = np.array([r["h"] for r in rows])
    print(f"n_components={len(rows)}  n_matches={len(set(r['match'] for r in rows))}")
    print("WIDTH  percentiles:", {p: float(np.percentile(W, p)) for p in (5, 25, 50, 75, 90, 95, 99)})
    print("HEIGHT percentiles:", {p: float(np.percentile(H, p)) for p in (5, 25, 50, 75, 90, 95, 99)})
    print("width histogram (px in 352 frame):")
    hist = np.bincount(W, minlength=40)
    for i, c in enumerate(hist[:40]):
        if c:
            print(f"  w={i:3d}  n={c:6d}  {'#' * min(60, c // max(1, len(rows) // 400))}")
    print("height histogram:")
    hh = np.bincount(H, minlength=10)
    for i, c in enumerate(hh):
        if c:
            print(f"  h={i:3d}  n={c:6d}")


if __name__ == "__main__":
    main()
