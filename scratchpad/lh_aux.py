#!/usr/bin/env python3
"""Non-visual side features per event: off-screen state + the gold-autocorrelation
baseline. Used by probe S, which is the reference every visual probe must beat.

  level, hp_frac        -> NOT on a HUD-off screen (level badge aside), but they are
                           what decides whether a Garen auto actually kills a minion
  minutes, minutes^2    -> game time (base-rate drift)
  dt_last_gold          -> frames since the previous gold jump  (minions die in waves)
  n_gold_40 / n_gold_100-> gold jumps in the last 40 / 100 frames
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasthit_events import gold_jumps, load_match  # noqa: E402

OUT = "scratchpad/lh_aux.npz"


def main():
    ev = np.load("scratchpad/lh_events.npz", allow_pickle=True)
    mid, frame = ev["mid"], ev["frame"]
    N = len(mid)
    cols = np.zeros((N, 5), np.float32)
    for g in sorted(set(mid.tolist())):
        m = load_match(g)
        j = gold_jumps(m["gold"], 10.0)
        c = np.concatenate([[0], np.cumsum(j)])
        last = np.full(m["T"], -1000)
        cur = -1000
        for i in range(m["T"]):
            last[i] = cur
            if j[i]:
                cur = i
        ii = np.where(mid == g)[0]
        t = frame[ii].astype(int)
        cols[ii, 0] = np.clip(t - last[t], 0, 400)
        cols[ii, 1] = c[t + 1] - c[np.maximum(t + 1 - 40, 0)]
        cols[ii, 2] = c[t + 1] - c[np.maximum(t + 1 - 100, 0)]
        cols[ii, 3] = t / 20.0 / 60.0
        cols[ii, 4] = m["T"] / 20.0 / 60.0
        print(f"  {g}", flush=True)
    np.savez_compressed(OUT, aux=cols,
                        names=np.array(["dt_last_gold", "n_gold_40", "n_gold_100",
                                        "minutes", "game_minutes"]))
    print("wrote", OUT, cols.shape)


if __name__ == "__main__":
    main()
