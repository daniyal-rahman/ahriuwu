"""Audit C follow-up: run-length structure of near-static consecutive frames."""
import json
import os

import cv2
import numpy as np

ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
M = "NA1_5549995114"
fdir = os.path.join(ROOT, M, "frames")
names = sorted(x for x in os.listdir(fdir) if x.endswith(".png"))
start = len(names) // 3
sel = names[start:start + 3000]

prev = None
d = []
for nm in sel:
    g = cv2.imread(os.path.join(fdir, nm), cv2.IMREAD_COLOR).astype(np.float32)
    if prev is not None:
        d.append(float(np.abs(g - prev).mean()))
    prev = g
d = np.array(d)
still = d < 1.0

# run lengths of consecutive "still" pairs
runs = []
cur = 0
for s in still:
    if s:
        cur += 1
    elif cur:
        runs.append(cur)
        cur = 0
if cur:
    runs.append(cur)
runs = np.array(runs) if len(runs) else np.array([0])

# also: exact-zero pixel diffs
n_zero = int((d == 0).sum())

out = dict(
    match=M, n_pairs=len(d), frac_still=float(still.mean()),
    n_pixel_identical=n_zero,
    n_runs=int(len(runs)), max_run=int(runs.max()), mean_run=float(runs.mean()),
    median_run=float(np.median(runs)),
    run_hist={f"len_{k}": int((runs == k).sum()) for k in [1, 2, 3, 4, 5]},
    n_runs_ge10=int((runs >= 10).sum()), n_runs_ge40=int((runs >= 40).sum()),
    frames_in_runs_ge10=int(runs[runs >= 10].sum()),
    mad_percentiles={str(p): float(np.percentile(d, p)) for p in [0.1, 1, 5, 10, 25, 50, 75, 95]},
)
print(json.dumps(out, indent=1))
json.dump(out, open("/srv/nfs/projects/ahriuwu/scratchpad/audit_c_runs.json", "w"), indent=1)
