"""Audit B helper: per-match PNG count vs labels.json total_frames."""
import json
import os

import numpy as np

LAT = "/srv/nfs/datasets/replay_latents_v7_bc"
FRAMES = "/srv/nfs/datasets/lol_replays_16_9_772"
CACHE = "/srv/nfs/projects/ahriuwu/scratchpad/audit_cache"
OUT = "/srv/nfs/projects/ahriuwu/scratchpad/audit_b_counts.json"

matches = sorted(f[:-3] for f in os.listdir(LAT) if f.endswith(".pt"))
rows = {}
for m in matches:
    r = {}
    fdir = os.path.join(FRAMES, m, "frames")
    r["n_png"] = sum(1 for x in os.listdir(fdir) if x.endswith(".png")) if os.path.isdir(fdir) else -1
    if os.path.isdir(fdir):
        names = sorted(x for x in os.listdir(fdir) if x.endswith(".png"))
        if names:
            r["png_min"] = int(names[0][:-4])
            r["png_max"] = int(names[-1][:-4])
    # cache meta
    cp = os.path.join(CACHE, m + ".npz")
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        meta = json.loads(str(z["meta"][0]))
        r["cache_total_frames"] = meta.get("total_frames")
        r["cache_T"] = meta.get("T")
        r["cache_n_png"] = meta.get("n_png")
        r["champion"] = meta.get("champion")
        r["fps"] = meta.get("fps")
    # labels.json streaming: read head for total_frames, and count "frames" entries
    lj = os.path.join(FRAMES, m, "labels.json")
    if os.path.exists(lj):
        r["labels_bytes"] = os.path.getsize(lj)
        with open(lj) as fh:
            head = fh.read(4096)
        for key in ("total_frames", "fps"):
            k = '"%s":' % key
            i = head.find(k)
            if i >= 0:
                j = i + len(k)
                buf = ""
                while j < len(head) and head[j] not in ",}":
                    buf += head[j]
                    j += 1
                try:
                    r["labels_" + key] = json.loads(buf.strip())
                except Exception:
                    pass
    rows[m] = r
    print(m, r, flush=True)

with open(OUT, "w") as fh:
    json.dump(rows, fh, indent=1)
print("WROTE", OUT)
