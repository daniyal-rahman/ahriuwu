#!/usr/bin/env python3
"""Batch-sample frames, apply the programmatic bar edits, and render ONE
verification grid so a human can confirm the edits look like real HP changes
(and not artifacts) BEFORE we spend any GPU time encoding.

Rows = 8 sampled frames. Columns = zoomed crop around the edited bar:
  ORIGINAL | -1px (lower HP) | -2px | -4px | +2px (higher HP) | CONTROL (terrain)
Each crop is the same window, so only the bar should differ between columns.
"""
import glob
import sys

import cv2
import numpy as np

sys.path.insert(0, "scratchpad")
from bar_edit import find_minion_bars, pick_bar, edit_bar, control_edit

ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
GAMES = ["NA1_5549995114", "NA1_5550013959", "NA1_5550028932", "NA1_5550045094"]
N_SCAN, N_SHOW, ZOOM, PAD = 900, 8, 9, 13


def main():
    picks = []
    for g in GAMES:
        fs = sorted(glob.glob(f"{ROOT}/{g}/frames/*.png"))
        if not fs:
            continue
        idx = np.linspace(2500, len(fs) - 1, N_SCAN // len(GAMES)).astype(int)
        for i in idx:
            im = cv2.imread(fs[int(i)])
            if im is None:
                continue
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            bars = find_minion_bars(rgb)
            b = pick_bar(bars, rgb.shape)
            if b is not None and b["w"] >= 8:          # room for a -4px edit
                picks.append((fs[int(i)], rgb, b, len(bars)))
                break_outer = len(picks) >= N_SHOW * 3
                if break_outer:
                    break
        if len(picks) >= N_SHOW * 3:
            break

    print(f"frames with a usable minion bar: {len(picks)} (scanned across {len(GAMES)} games)")
    if not picks:
        print("NO BARS FOUND — detector needs retuning; not proceeding.")
        return 1

    sel = [picks[i] for i in np.linspace(0, len(picks) - 1, min(N_SHOW, len(picks))).astype(int)]
    variants = [("ORIGINAL", 0, False), ("-1px", 1, False), ("-2px", 2, False),
                ("-4px", 4, False), ("+2px", -2, False), ("CONTROL", 2, True)]
    rows = []
    for path, rgb, bar, nbars in sel:
        x, y, w, h = bar["x"], bar["y"], bar["w"], bar["h"]
        x0, x1 = max(x - PAD, 0), min(x + w + PAD, rgb.shape[1])
        y0, y1 = max(y - PAD, 0), min(y + h + PAD, rgb.shape[0])
        cells = []
        for name, k, is_ctrl in variants:
            img = (control_edit(rgb, bar, k) if is_ctrl else
                   (rgb if k == 0 else edit_bar(rgb, bar, k)))
            crop = img[y0:y1, x0:x1]
            crop = cv2.resize(crop, (crop.shape[1] * ZOOM, crop.shape[0] * ZOOM),
                              interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 0), 1, cv2.LINE_AA)
            cells.append(crop)
        hgt = min(c.shape[0] for c in cells)
        row = np.concatenate([c[:hgt] for c in cells], axis=1)
        cv2.putText(row, f"{path.split('/')[-2][-6:]}  bar {w}x{h}px  ({nbars} bars in frame)",
                    (4, row.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
        print(f"  {path.split('/')[-2]}/{path.split('/')[-1]}: bar {w}x{h} at ({x},{y}), {nbars} bars in frame")

    wid = min(r.shape[1] for r in rows)
    grid = np.concatenate([r[:, :wid] for r in rows], axis=0)
    out = "scratchpad/bar_edit_grid.png"
    cv2.imwrite(out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"wrote {out}  {grid.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
