"""Audit B final: replay ReplayLatentSequenceDataset._index_match exactly and
check every emitted window for latent<->label alignment and bounds."""
import json
import os

import numpy as np
import torch

LAT = "/srv/nfs/datasets/replay_latents_v7_bc"
SEQ_LEN, STRIDE = 16, 8   # launch_bc_gate_1060.sh values

counts = json.load(open("/srv/nfs/projects/ahriuwu/scratchpad/audit_b_counts.json"))
rows = {r["match"]: r for r in json.load(open("/srv/nfs/projects/ahriuwu/scratchpad/audit_b_results.json"))}

tot_win = 0
bad_align = bad_label_oob = bad_lat_oob = 0
offenders = []
lost = []
for m in sorted(rows):
    fi = torch.load(os.path.join(LAT, m + ".pt"), map_location="cpu",
                    mmap=True, weights_only=True)["frame_indices"].to(torch.int64).numpy()
    N = len(fi)
    n_label = counts[m]["labels_total_frames"]
    usable = min(N, n_label)
    frame_to_idx = {int(fi[i]): i for i in range(N)}
    frame_nums = sorted(int(f) for f in fi[:usable])

    wins = []
    run_start, run_len = frame_nums[0], 1
    for i in range(1, len(frame_nums)):
        if frame_nums[i] == frame_nums[i - 1] + 1:
            run_len += 1
        else:
            if run_len >= SEQ_LEN:
                wins += [(run_start + o, frame_to_idx[run_start + o])
                         for o in range(0, run_len - SEQ_LEN + 1, STRIDE)]
            run_start, run_len = frame_nums[i], 1
    if run_len >= SEQ_LEN:
        wins += [(run_start + o, frame_to_idx[run_start + o])
                 for o in range(0, run_len - SEQ_LEN + 1, STRIDE)]

    tot_win += len(wins)
    for sf, si in wins:
        if si + SEQ_LEN > N:
            bad_lat_oob += 1
            offenders.append((m, "latent_oob", sf, si))
        elif not np.array_equal(fi[si:si + SEQ_LEN], np.arange(sf, sf + SEQ_LEN)):
            bad_align += 1
            offenders.append((m, "misalign", sf, si))
        if sf + SEQ_LEN > n_label:
            bad_label_oob += 1
            offenders.append((m, "label_oob", sf, si))
    # coverage loss vs full match
    if N < n_label:
        lost.append((m, N, n_label, n_label - N, round(1 - N / n_label, 3)))

print(f"seq_len={SEQ_LEN} stride={STRIDE}")
print(f"total windows across 125 matches: {tot_win}")
print(f"MISALIGNED windows (fi[si:si+T] != arange(sf,sf+T)): {bad_align}")
print(f"latent out-of-bounds windows: {bad_lat_oob}")
print(f"label out-of-bounds windows (start_frame+T > n_label): {bad_label_oob}")
print("first offenders:", offenders[:10])
print(f"\nmatches with latents covering FEWER frames than labels: {len(lost)}")
print(f"{'match':>18} {'N_lat':>8} {'n_label':>8} {'missing':>8} {'frac_lost':>9}")
for x in sorted(lost, key=lambda r: -r[4]):
    print(f"{x[0]:>18} {x[1]:>8} {x[2]:>8} {x[3]:>8} {x[4]:>9}")
tot_lat = sum(r["N"] for r in rows.values())
tot_lab = sum(counts[m]["labels_total_frames"] for m in rows)
print(f"\nTOTAL: latent frames {tot_lat:,} vs label frames {tot_lab:,} "
      f"-> {tot_lab - tot_lat:,} label frames ({100*(1-tot_lat/tot_lab):.1f}%) have NO latent")
