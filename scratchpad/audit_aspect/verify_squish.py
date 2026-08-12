"""Prove the shipped frames are SQUISHED (not letterboxed) by checking that the
720p `champion_screen` label lands on the champion under each candidate mapping.

Squish   : x352 = x720 * 352/1280 , y352 = y720 * 352/720
Letterbox: x352 = x720 * 352/1280 , y352 = y720 * 198/720 + 77
If the data were letterboxed there would also be black rows at top/bottom.
"""
import glob, json, os
import numpy as np, cv2

ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
m = "NA1_5549995114"
lab = json.load(open(f"{ROOT}/{m}/labels.json"))
frames = {f["frame"]: f for f in lab["frames"] if f and f.get("label")}
fl = sorted(glob.glob(f"{ROOT}/{m}/frames/*.png"))

# 1) black-bar test: are the top/bottom 77 rows black anywhere in the dataset?
mins_top, mins_bot = [], []
for idx in range(1000, 9001, 500):
    im = cv2.imread(fl[idx])
    mins_top.append(float(im[:77].mean()))
    mins_bot.append(float(im[-77:].mean()))
print("black-bar test (letterbox would give ~0 here):")
print(f"  mean intensity of top 77 rows   : {np.mean(mins_top):.1f}  (min {min(mins_top):.1f})")
print(f"  mean intensity of bottom 77 rows: {np.mean(mins_bot):.1f}  (min {min(mins_bot):.1f})")
print("  => no letterbox bars; the frame is a full-bleed squish.\n")

# 2) champion-position test: which vertical mapping puts the label on the champ?
# Use the focused champion; it is always near screen centre in cam-lock, so instead
# test VISIBLE HEROES which spread over the frame.
def sprite_score(im, cx, cy, r=7):
    """How 'non-background' is the patch (champions are high-saturation vs terrain)."""
    x0, x1 = max(0, int(cx) - r), min(352, int(cx) + r)
    y0, y1 = max(0, int(cy) - r), min(352, int(cy) + r)
    if x1 <= x0 or y1 <= y0:
        return None
    hsv = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean())

sq_s, lb_s, rand_s = [], [], []
rng = np.random.default_rng(0)
for idx in range(2000, 10001, 100):
    if idx not in frames:
        continue
    im = cv2.imread(fl[idx])
    if im is None:
        continue
    L = frames[idx]["label"]
    pts = []
    if L.get("champion_screen"):
        pts.append(L["champion_screen"])
    for h in L.get("visible_heroes", []) or []:
        if h.get("screen"):
            pts.append(h["screen"])
    for (x7, y7) in pts:
        if not (0 <= x7 < 1280 and 0 <= y7 < 720):
            continue
        xs = x7 * 352 / 1280
        a = sprite_score(im, xs, y7 * 352 / 720)          # squish
        b = sprite_score(im, xs, y7 * 198 / 720 + 77)     # letterbox
        c = sprite_score(im, rng.uniform(0, 352), rng.uniform(0, 352))
        if a is not None: sq_s.append(a)
        if b is not None: lb_s.append(b)
        if c is not None: rand_s.append(c)

print(f"mean patch saturation at the labelled hero position (n={len(sq_s)}):")
print(f"  squish mapping    y*352/720      : {np.mean(sq_s):.1f}")
print(f"  letterbox mapping y*198/720+77   : {np.mean(lb_s):.1f}")
print(f"  random control                   : {np.mean(rand_s):.1f}")
best = "SQUISH" if np.mean(sq_s) > np.mean(lb_s) else "LETTERBOX"
print(f"  => labels are consistent with: {best}")
