"""Isolate MINION hp bars (exclude champion + turret bars) and measure the FULL
bar box (coloured fill + depleted dark remainder).

Champion bars are excluded using labels.json champion_screen / visible_heroes
screen coords (720p -> 352 squish mapping). Turret / epic-monster bars are
excluded by their gold outline + large width.
"""
import glob, json, os, sys
import numpy as np, cv2

ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
OUT = "/srv/nfs/projects/ahriuwu/scratchpad/audit_aspect"
SW, SH, FS = 1280, 720, 352
SX, SY = FS / SW, FS / SH   # 0.275, 0.48889


def bar_masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    blue = (H >= 95) & (H <= 125) & (S >= 120) & (V >= 120)
    red = (((H <= 8) | (H >= 172)) & (S >= 140) & (V >= 110))
    return blue.astype(np.uint8), red.astype(np.uint8)


def full_box(bgr, x, y, w, h):
    """Extend the coloured run right (and left) across the dark depleted region.
    Returns (x0, x1) of the full bar box."""
    yc = y + h // 2
    row = bgr[yc]
    hsv_row = cv2.cvtColor(row.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0]
    Vr = hsv_row[:, 2].astype(int)
    Sr = hsv_row[:, 1].astype(int)
    dark = (Vr < 85)
    x1 = x + w
    while x1 < bgr.shape[1] and dark[x1]:
        x1 += 1
    x0 = x
    while x0 - 1 >= 0 and dark[x0 - 1]:
        x0 -= 1
    return x0, x1


def main():
    matches = sorted(d for d in os.listdir(ROOT) if d.startswith("NA1_"))[:8]
    rows = []
    for m in matches:
        lp = f"{ROOT}/{m}/labels.json"
        if not os.path.exists(lp):
            continue
        lab = json.load(open(lp))
        frames = {f["frame"]: f for f in lab["frames"] if f and f.get("label")}
        fl = sorted(glob.glob(f"{ROOT}/{m}/frames/*.png"))
        if len(fl) < 8000:
            continue
        for idx in range(4000, 8001, 200):
            if idx not in frames:
                continue
            L = frames[idx]["label"]
            # champion bar exclusion zones (352-space), bars sit ABOVE the model
            excl = []
            cs = L.get("champion_screen")
            if cs:
                excl.append((cs[0] * SX, cs[1] * SY))
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
                    if not (1 <= h <= 4 and w >= 3 and w / max(h, 1) >= 2.0):
                        continue
                    if a / float(w * h) < 0.55:
                        continue
                    cx, cy = x + w / 2, y + h / 2
                    # champion bars float ~10-40px (352 space) above the champ sprite
                    if any(abs(cx - ex) < 22 and -45 < (ey - cy) < 12 for ex, ey in excl):
                        continue
                    x0, x1 = full_box(im, x, y, w, h)
                    fw = x1 - x0
                    # gold-outlined turret/monster bars: check for gold pixels flanking
                    band = im[max(0, y - 3):y + h + 3, x0:x1]
                    hb = cv2.cvtColor(band, cv2.COLOR_BGR2HSV) if band.size else None
                    gold = 0
                    if hb is not None and hb.size:
                        gold = int((((hb[..., 0] >= 15) & (hb[..., 0] <= 35) &
                                     (hb[..., 1] >= 110) & (hb[..., 2] >= 110))).sum())
                    rows.append(dict(match=m, frame=idx, team=team, fill_w=int(w), h=int(h),
                                     full_w=int(fw), gold=gold, x=int(x0), y=int(y)))
    json.dump(rows, open(f"{OUT}/minion_bars.json", "w"))
    R = [r for r in rows if r["gold"] < 6]           # drop turret/epic (gold frame)
    FW = np.array([r["full_w"] for r in R])
    print(f"total comps={len(rows)}  after gold-filter={len(R)}")
    print("FULL-BOX width percentiles:", {p: float(np.percentile(FW, p)) for p in (10, 25, 50, 75, 90, 95, 99)})
    print("\nfull-box width histogram:")
    hist = np.bincount(FW, minlength=60)
    for i, c in enumerate(hist[:60]):
        if c:
            print(f"  full_w={i:3d}  n={c:5d}  {'#' * min(70, c * 70 // max(hist[:60]))}")
    # cluster: minion bars vs champion-ish leftovers
    print("\nheight distribution of full-box components:")
    HH = np.bincount(np.array([r["h"] for r in R]), minlength=6)
    for i, c in enumerate(HH):
        if c:
            print(f"  h={i} n={c}")


if __name__ == "__main__":
    main()
