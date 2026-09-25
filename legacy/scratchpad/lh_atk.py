#!/usr/bin/env python3
"""Per-event flag: was Garen in the 'attack' action state at that frame?

Needed because the gold-anchored event set is confounded by construction: its
negatives are attack-STATE frames, but only ~26% of its positives (gold_frame-6)
are, since ~60% of income events carry no attack marking at all. Comparing them
raw would mostly measure "is he swinging", not "will this swing kill". Filtering
both classes to attack-state frames fixes it and still covers the CHAINED autos
that the commit anchoring (a state transition) misses.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasthit_events import load_match  # noqa: E402

OUT = "scratchpad/lh_atk.npz"


def main():
    ev = np.load("scratchpad/lh_events.npz", allow_pickle=True)
    mid, frame = ev["mid"], ev["frame"]
    atk = np.zeros(len(mid), bool)
    for g in sorted(set(mid.tolist())):
        m = load_match(g)
        a = np.array([x == "attack" for x in m["atype"]])
        ii = np.where(mid == g)[0]
        atk[ii] = a[frame[ii].astype(int)]
        print(f"  {g} {atk[ii].mean():.3f}", flush=True)
    np.savez_compressed(OUT, atk=atk)
    print("wrote", OUT, atk.mean())


if __name__ == "__main__":
    main()
