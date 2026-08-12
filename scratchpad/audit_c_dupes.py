"""Audit C: exact + near-duplicate consecutive frames, black/frozen runs."""
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
OUT = "/srv/nfs/projects/ahriuwu/scratchpad/audit_c_results.json"

MATCHES = [
    "NA1_5549995114", "NA1_5550028932", "NA1_5550110638", "NA1_5551763045",
    "NA1_5552670437", "NA1_5553559186", "NA1_5553807865", "NA1_5551474361",
]
N_SAMPLE = 3000  # consecutive-frame window per match (contiguous, mid-game)


def do_match(m):
    fdir = os.path.join(ROOT, m, "frames")
    names = sorted(x for x in os.listdir(fdir) if x.endswith(".png"))
    n_total = len(names)
    start = max(0, n_total // 3)
    sel = names[start:start + N_SAMPLE]
    prev_h = None
    prev_img = None
    n_exact = n_near = n_black = n_static_run = 0
    diffs = []
    means = []
    for nm in sel:
        p = os.path.join(fdir, nm)
        raw = open(p, "rb").read()
        h = hashlib.md5(raw).hexdigest()
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        g = img.astype(np.float32)
        mu = float(g.mean())
        means.append(mu)
        if mu < 2.0:
            n_black += 1
        if prev_h is not None:
            if h == prev_h:
                n_exact += 1
            d = float(np.abs(g - prev_img).mean())
            diffs.append(d)
            if d < 1.0:  # < 1/255 in [0,1] units == < 1.0 in 0-255 units
                n_near += 1
        prev_h = h
        prev_img = g
    diffs = np.array(diffs) if diffs else np.array([0.0])
    return dict(
        match=m, n_total_png=n_total, n_sampled=len(sel), start_frame=start,
        n_exact_dup=n_exact, frac_exact_dup=n_exact / max(len(diffs), 1),
        n_near_dup=n_near, frac_near_dup=n_near / max(len(diffs), 1),
        n_black=n_black, frac_black=n_black / max(len(sel), 1),
        mad_mean=float(diffs.mean()), mad_median=float(np.median(diffs)),
        mad_p01=float(np.percentile(diffs, 1)), mad_p05=float(np.percentile(diffs, 5)),
        mad_min=float(diffs.min()), mad_max=float(diffs.max()),
        pix_mean_min=float(np.min(means)), pix_mean_mean=float(np.mean(means)),
    )


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(do_match, MATCHES))
    for r in rows:
        print(json.dumps(r), flush=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    print("WROTE", OUT)
