"""Audit B: verify frame_indices invariants across all replay_latents_v7_bc packs."""
import json
import os
import sys

import numpy as np
import torch

LAT = "/srv/nfs/datasets/replay_latents_v7_bc"
FRAMES = "/srv/nfs/datasets/lol_replays_16_9_772"
CACHE = "/srv/nfs/projects/ahriuwu/scratchpad/audit_cache"
OUT = "/srv/nfs/projects/ahriuwu/scratchpad/audit_b_results.json"

files = sorted(f for f in os.listdir(LAT) if f.endswith(".pt"))
rows = []
for i, f in enumerate(files):
    m = f[:-3]
    d = torch.load(os.path.join(LAT, f), map_location="cpu", mmap=True, weights_only=True)
    fi = d["frame_indices"].to(torch.int64).numpy()
    lat_shape = list(d["latents"].shape)
    N = int(fi.shape[0])
    diff = np.diff(fi)
    strictly_asc = bool((diff > 0).all()) if N > 1 else True
    n_dup = int((diff == 0).sum()) if N > 1 else 0
    n_gaps = int((diff > 1).sum()) if N > 1 else 0
    max_gap = int(diff.max()) if N > 1 else 0
    is_arange = bool(np.array_equal(fi, np.arange(N)))
    # magnitude of misalignment: fi[i] - i
    off = fi - np.arange(N)
    max_off = int(np.abs(off).max())
    mean_off = float(np.abs(off).mean())

    # png count
    fdir = os.path.join(FRAMES, m, "frames")
    n_png = -1
    if os.path.isdir(fdir):
        n_png = sum(1 for x in os.listdir(fdir) if x.endswith(".png"))

    # labels total_frames: prefer npz cache
    total_frames = -1
    cache_T = -1
    cache_npng = -1
    cp = os.path.join(CACHE, m + ".npz")
    if os.path.exists(cp):
        try:
            z = np.load(cp, allow_pickle=True)
            if "meta" in z:
                meta = z["meta"].item() if z["meta"].shape == () else z["meta"]
                if isinstance(meta, dict):
                    cache_T = int(meta.get("T", -1))
                    cache_npng = int(meta.get("n_png", -1))
                    total_frames = int(meta.get("total_frames", -1))
        except Exception as e:
            print("cache err", m, e, file=sys.stderr)

    rows.append(dict(
        match=m, N=N, lat_shape=lat_shape, min=int(fi.min()), max=int(fi.max()),
        strictly_ascending=strictly_asc, n_dup=n_dup, n_gaps=n_gaps, max_gap=max_gap,
        is_arange=is_arange, max_offset=max_off, mean_abs_offset=mean_off,
        n_png=n_png, cache_T=cache_T, cache_n_png=cache_npng, cache_total_frames=total_frames,
    ))
    print(f"[{i+1}/{len(files)}] {m} N={N} max={int(fi.max())} arange={is_arange} "
          f"gaps={n_gaps} maxgap={max_gap} dup={n_dup} npng={n_png} maxoff={max_off}", flush=True)
    del d

with open(OUT, "w") as fh:
    json.dump(rows, fh, indent=1)
print("WROTE", OUT)
