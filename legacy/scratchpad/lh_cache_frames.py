#!/usr/bin/env python3
"""Cache the raw 352x352 PNGs needed by Probe A into a uint8 memmap.

The frames live on spinning disk (~25 random img/s single-threaded), so training a
CNN straight off the PNGs is hopeless. One parallel pass writes exactly the frames
the probe needs; the crop variant is sliced out of the same rows at train time, so
it costs no extra storage.
"""
import argparse
import os
import sys

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasthit_events import FRAMES_ROOT, splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="scratchpad/lh_events.npz")
    ap.add_argument("--out", default="scratchpad/lh_frames")
    ap.add_argument("--anchor", default="commit")
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 8])
    ap.add_argument("--n-train-games", type=int, default=60)
    ap.add_argument("--cap-per-game", type=int, default=0, help="0 = no cap")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    ev = np.load(args.events, allow_pickle=True)
    sp = splits(args.events)
    sel_games = {}
    for g in sp["train"][: args.n_train_games]:
        sel_games[g] = "train"
    for g in sp["heldout_seen"]:
        sel_games[g] = "heldout_seen"
    for g in sp["unseen_all"]:      # probe A needs no latents -> use every unseen game
        sel_games[g] = "heldout_unseen"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lh_probe import anchor_mask  # noqa: E402
    m = anchor_mask(ev, args.anchor) & np.isin(ev["mid"], list(sel_games))
    idx = np.where(m)[0]
    if args.cap_per_game:
        rng = np.random.default_rng(0)
        keep = []
        for g in sel_games:
            gi = idx[ev["mid"][idx] == g]
            keep.append(gi if len(gi) <= args.cap_per_game
                        else rng.choice(gi, args.cap_per_game, replace=False))
        idx = np.sort(np.concatenate(keep))

    mid = ev["mid"][idx]
    frame = ev["frame"][idx]
    N = len(idx)
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out, "meta.npz"),
        mid=mid, frame=frame, y=ev["y"][idx], csx=ev["csx"][idx], csy=ev["csy"][idx],
        level=ev["level"][idx], hp=ev["hp"][idx],
        split=np.array([sel_games[g] for g in mid]), anchor=np.array([args.anchor] * N),
        offsets=np.array(args.offsets))
    print(f"{N} events over {len(set(mid.tolist()))} games; "
          f"{N*len(args.offsets)} frames -> {N*len(args.offsets)*352*352*3/1e9:.1f} GB", flush=True)

    for off in args.offsets:
        path = os.path.join(args.out, f"frames_off{off}.u8")
        if os.path.exists(path) and os.path.getsize(path) == N * 352 * 352 * 3:
            print(f"off{off}: exists, skip", flush=True)
            continue
        mm = np.memmap(path + ".tmp", dtype=np.uint8, mode="w+", shape=(N, 352, 352, 3))
        bad = 0

        def work(i):
            p = f"{FRAMES_ROOT}/{mid[i]}/frames/{int(frame[i]) + off:06d}.png"
            im = cv2.imread(p)
            if im is None:
                return i
            if im.shape[:2] != (352, 352):
                im = cv2.resize(im, (352, 352), interpolation=cv2.INTER_AREA)
            mm[i] = im[:, :, ::-1]        # BGR -> RGB
            return None

        with ThreadPoolExecutor(args.workers) as ex:
            for k, r in enumerate(ex.map(work, range(N))):
                if r is not None:
                    bad += 1
                if (k + 1) % 2000 == 0:
                    print(f"  off{off}: {k+1}/{N} bad={bad}", flush=True)
        mm.flush()
        del mm
        os.replace(path + ".tmp", path)
        print(f"off{off}: wrote {path} bad={bad}", flush=True)


if __name__ == "__main__":
    main()
