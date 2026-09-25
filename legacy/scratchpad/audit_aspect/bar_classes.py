"""Separate MINION vs CHAMPION hp bars and report full-bar widths.

Minion bars: 1-2 px tall in the 352 squish, isolated, far from any hero sprite.
Champion bars: 3-4 px tall, sit directly above a `visible_heroes[].screen` /
`champion_screen` position, and carry a mana bar underneath.
"""
import glob, json, os
import numpy as np, cv2

ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
OUT = "/srv/nfs/projects/ahriuwu/scratchpad/audit_aspect"
SW, SH, FS = 1280, 720, 352
SX, SY = FS / SW, FS / SH


def bar_masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    blue = (H >= 95) & (H <= 125) & (S >= 120) & (V >= 120)
    red = (((H <= 8) | (H >= 172)) & (S >= 140) & (V >= 110))
    return blue.astype(np.uint8), red.astype(np.uint8)


def main():
    matches = sorted(d for d in os.listdir(ROOT) if d.startswith("NA1_"))[:12]
    minion, champ = [], []
    for m in matches:
        lp = f"{ROOT}/{m}/labels.json"
        if not os.path.exists(lp):
            continue
        lab = json.load(open(lp))
        frames = {f["frame"]: f for f in lab["frames"] if f and f.get("label")}
        fl = sorted(glob.glob(f"{ROOT}/{m}/frames/*.png"))
        if len(fl) < 8000:
            continue
        for idx in range(4000, 8001, 150):
            if idx not in frames:
                continue
            L = frames[idx]["label"]
            excl = []
            if L.get("champion_screen"):
                excl.append((L["champion_screen"][0] * SX, L["champion_screen"][1] * SY))
            for hh in L.get("visible_heroes", []) or []:
                if hh.get("screen"):
                    excl.append((hh["screen"][0] * SX, hh["screen"][1] * SY))
            im = cv2.imread(fl[idx])
            if im is None:
                continue
            b, r = bar_masks(im)
            for team, mk in (("blue", b), ("red", r)):
                n, _, stats, _ = cv2.connectedComponentsWithStats(mk, 8)
                for i in range(1, n):
                    x, y, w, h, a = stats[i]
                    if w < 3 or w / max(h, 1) < 2.5 or a / float(w * h) < 0.6:
                        continue
                    cx, cy = x + w / 2, y + h / 2
                    near = any(abs(cx - ex) < 24 and -50 < (ey - cy) < 14 for ex, ey in excl)
                    rec = dict(match=m, frame=idx, team=team, w=int(w), h=int(h))
                    if near and 2 <= h <= 5:
                        champ.append(rec)
                    elif (not near) and 1 <= h <= 2:
                        minion.append(rec)
    json.dump(dict(minion=minion, champ=champ), open(f"{OUT}/bar_classes.json", "w"))

    for name, arr in (("MINION (h<=2, no hero nearby)", minion), ("CHAMPION (above hero sprite)", champ)):
        W = np.array([r["w"] for r in arr])
        print(f"\n=== {name}  n={len(arr)} ===")
        if not len(W):
            continue
        print("  width percentiles:", {p: float(np.percentile(W, p)) for p in (25, 50, 75, 90, 95, 98, 99)})
        print("  mode:", int(np.bincount(W).argmax()), " max:", int(W.max()))
        hist = np.bincount(W, minlength=30)
        mx = hist[:30].max()
        for i, c in enumerate(hist[:30]):
            if c:
                print(f"    w={i:3d} n={c:5d} {'#' * (c * 60 // mx)}")


if __name__ == "__main__":
    main()
