#!/usr/bin/env python3
"""Validate the CV HP-bar reader against replay-label ground truth.

Gates (from the expert-review plan):
  own-champ fill vs labels hp_frac  R^2 > 0.9  (exact ground truth exists)
  coverage: matched a champ bar on >= 80%% of labeled frames
  minion visibility: >= 90%% of laning frames detect >= 1 minion bar

    PYTHONPATH=src python scripts/validate_hp_reader.py
"""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "src")
from ahriuwu.vision import detect_bars, read_own_champ_hp

ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
MATCHES = ["NA1_5549981347", "NA1_5550450386", "NA1_5551132630"]   # held-out
N_FRAMES = 300
LABEL_SPACE = (1280.0, 720.0)


def main():
    reads, gts, cover, minion_ok, n_lane = [], [], 0, 0, 0
    per_match = {}
    for m in MATCHES:
        lab = json.load(open(f"{ROOT}/{m}/labels.json"))
        rows = []
        for f in lab["frames"]:
            l = f.get("label") or {}
            cs = l.get("champion_stats") or {}
            scr = l.get("champion_screen")
            if cs.get("hp") is not None and cs.get("hp_max") and isinstance(scr, list):
                rows.append((f["frame"], cs["hp"] / cs["hp_max"], scr[0], scr[1]))
        idx = np.linspace(0, len(rows) - 1, N_FRAMES).astype(int)
        mr, mg = [], []
        for i in idx:
            fr, hp, sx, sy = rows[int(i)]
            im = cv2.imread(f"{ROOT}/{m}/frames/{int(fr):06d}.png")
            if im is None:
                continue
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            cxy = (sx / LABEL_SPACE[0] * w, sy / LABEL_SPACE[1] * h)
            r = read_own_champ_hp(rgb, cxy)
            if r is not None:
                cover += 1
                mr.append(r)
                mg.append(hp)
            # minion visibility on laning-phase frames (frame 1500+)
            if fr >= 1500:
                n_lane += 1
                if any(b["kind"] == "minion" for b in detect_bars(rgb)):
                    minion_ok += 1
        reads += mr
        gts += mg
        if mr:
            a, g = np.array(mr), np.array(mg)
            r2 = 1 - ((a - g) ** 2).sum() / ((g - g.mean()) ** 2).sum()
            per_match[m] = (len(mr), float(r2), float(np.abs(a - g).mean()))

    a, g = np.array(reads), np.array(gts)
    n_total = len(MATCHES) * N_FRAMES
    r2 = 1 - ((a - g) ** 2).sum() / ((g - g.mean()) ** 2).sum()
    print(f"own-champ HP read: n={len(a)}/{n_total} (coverage {cover / n_total:.0%})")
    for m, (n, r, mae) in per_match.items():
        print(f"  {m}: n={n}  R2={r:+.3f}  MAE={mae:.3f}")
    print(f"OVERALL: R2={r2:+.3f}  MAE={np.abs(a - g).mean():.3f}   (gate: R2>0.9, coverage>80%)")
    print(f"minion bars visible: {minion_ok}/{n_lane} laning frames ({minion_ok / max(n_lane, 1):.0%})  (gate: >90%)")


if __name__ == "__main__":
    main()
