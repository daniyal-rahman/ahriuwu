#!/usr/bin/env python3
"""CPU validation of the action-conditioned MIXED data pipeline: does
ReplayLatentSequenceDataset ingest YT placeholder labels alongside real replays
without crashing, build YT sequences, and set cursor_valid correctly (False for YT
-> no_action_embed, True for replays)?"""
import glob
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from ahriuwu.constants import ABILITY_KEYS
from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset

LAT, LAB = sys.argv[1], sys.argv[2]
mids = [Path(p).stem for p in glob.glob(f"{LAT}/*.pt") if Path(p).stem != "index"]
outc = {m: False for m in mids}
ds = ReplayLatentSequenceDataset(latents_dir=LAT, labels_root=LAB, outcomes=outc,
                                 sequence_length=128, stride=64)
print(f"\nTOTAL sequences: {len(ds)}")
c = Counter(s["video_id"] for s in ds.sequences)
for vid, n in c.items():
    print(f"  {vid}: {n} seqs  ({'YT' if not vid.startswith('NA1_') else 'replay'})")

for want_yt in (False, True):
    hits = [i for i, s in enumerate(ds.sequences)
            if (not s["video_id"].startswith("NA1_")) == want_yt]
    if not hits:
        print(f"\n{'YT' if want_yt else 'REPLAY'}: NO SEQUENCES"); continue
    s = ds[hits[0]]
    a = s["actions"]
    cv = a["cursor_valid"]
    print(f"\n{'YT' if want_yt else 'REPLAY'} sample {s['video_id']}:")
    print(f"  latents {tuple(s['latents'].shape)}  movement[0]={a['movement'][0].tolist()}")
    print(f"  cursor_valid all={bool(cv.all())} any={bool(cv.any())} (expect "
          f"{'all-False' if want_yt else 'all-True'})")
    print(f"  ability presses in window: " + ", ".join(
        f"{k}={int(a[k].sum())}" for k in ABILITY_KEYS if int(a[k].sum()) > 0) or "  (none)")

# assert the contract
yt_ok = all(not ds[i]["actions"]["cursor_valid"].any()
            for i in range(len(ds)) if not ds.sequences[i]["video_id"].startswith("NA1_"))
rep_ok = all(ds[i]["actions"]["cursor_valid"].all()
             for i in range(min(len(ds), 50)) if ds.sequences[i]["video_id"].startswith("NA1_"))
print(f"\nCONTRACT: YT cursor_valid all-False={yt_ok} | replay cursor_valid all-True={rep_ok}")
print("PASS" if yt_ok and rep_ok and any(not v.startswith("NA1_") for v in c) else "FAIL")
