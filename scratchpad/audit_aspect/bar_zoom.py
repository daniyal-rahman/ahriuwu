"""Visual verification: crop detected bars at high zoom + measure FULL bar box
(coloured fill + depleted dark remainder inside the black outline)."""
import glob, json, os
import numpy as np, cv2

ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
OUT = "/srv/nfs/projects/ahriuwu/scratchpad/audit_aspect"
sys_rows = json.load(open(f"{OUT}/bars_raw.json"))

# group detections by (match, frame)
by_fr = {}
for r in sys_rows:
    by_fr.setdefault((r["match"], r["frame"]), []).append(r)

Z = 10  # zoom
tiles = []
keys = sorted(by_fr)[:400]
picked = 0
for k in keys:
    if picked >= 40:
        break
    m, idx = k
    fr = sorted(glob.glob(f"{ROOT}/{m}/frames/*.png"))
    im = cv2.imread(fr[idx])
    dets = sorted(by_fr[k], key=lambda r: -r["w"])
    for r in dets[:3]:
        if picked >= 40:
            break
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        # generous context box around the bar
        x0, x1 = max(0, x - 12), min(352, x + w + 12)
        y0, y1 = max(0, y - 6), min(352, y + h + 6)
        crop = im[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        big = cv2.resize(crop, ((x1 - x0) * Z, (y1 - y0) * Z), interpolation=cv2.INTER_NEAREST)
        # pad to uniform tile
        tile = np.zeros((26 * Z, 40 * Z, 3), np.uint8)
        hh, ww = big.shape[:2]
        tile[:min(hh, 26 * Z), :min(ww, 40 * Z)] = big[:26 * Z, :40 * Z]
        cv2.putText(tile, f"{r['team'][0]} w{w} h{h}", (4, 26 * Z - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
        picked += 1

rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles) - 4, 5)]
grid = np.vstack(rows)
cv2.imwrite(f"{OUT}/bar_zoom_montage.png", grid)
print("wrote montage", grid.shape, "n_tiles", len(tiles))
