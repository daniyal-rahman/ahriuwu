#!/usr/bin/env python3
"""A/B every Phase-2 checkpoint on HELD-OUT games, with metrics that do not
depend on the movement representation.

The two families encode movement differently -- 'axis' is two 21-way
categoricals plus a sticky gate, 'joint_noop' is one 442-way categorical whose
last class means "no new order" -- so their training losses are not comparable.
These metrics are:

  cmd/s        how often the agent issues a NEW movement order (human: ~2.0/s)
  cell_acc     top-1 accuracy of the commanded 21x21 grid cell, scored ONLY on
               frames where the human actually issued a command
  cells_used   distinct grid cells over the run; detects collapse to one spot
  cast/s       ability presses per second (human ~2-3/s incl. attacks)

Ground truth comes from the same dataset the trainer used, so "what the human
did" is exactly the supervised target.

    PYTHONPATH=src python scripts/ab_checkpoints.py --frames 1500
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ahriuwu.constants import ABILITY_KEYS, MOVEMENT_DIM  # noqa: E402

VAL_GAMES = ["NA1_5549995114", "NA1_5550417257", "NA1_5551063460",
             "NA1_5551782551", "NA1_5552261591", "NA1_5552945604"]


def to_cell(xy, bins=21):
    """(x,y) in [0,1] -> flat 21x21 cell, matching PolicyHead.joint_encode."""
    d = bins - 1
    xi = int(np.clip(round(float(xy[0]) * d), 0, d))
    yi = int(np.clip(round(float(xy[1]) * d), 0, d))
    return yi * bins + xi


def load_games(args):
    """Parse each held-out game ONCE; every checkpoint reuses this."""
    import pathlib
    from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset
    ds = ReplayLatentSequenceDataset(args.latents_dir, args.labels_root,
                                     outcomes={m: True for m in VAL_GAMES},
                                     sequence_length=args.context, stride=args.context,
                                     movement_source='clicks', cache_path=None)
    games = []
    for g in VAL_GAMES[:args.games]:
        md = ds._parse_match(g, pathlib.Path(f'{args.labels_root}/{g}/labels.json'))
        lp = f'{args.latents_dir}/{g}.pt'
        if md is None or not os.path.exists(lp):
            continue
        pack = torch.load(lp, weights_only=True)
        games.append(dict(name=g, lat=pack['latents'], fi=pack['frame_indices'].numpy(),
                          mv=np.asarray(md['movement'], dtype=np.float64),
                          ev=np.asarray(md['movement_event'], dtype=bool)))
        print(f'  parsed {g}: {len(games[-1]["fi"])} latent frames', flush=True)
    return games


def run_one(ckpt, args, games):
    from agent_infer import GarenAgent
    agent = GarenAgent(ckpt, tokenizer_ckpt=args.tokenizer_ckpt,
                       context=args.context, device=args.device)
    n_frames = n_cmd = n_cast = 0
    hit = tot_scored = 0
    cells = Counter()
    fps = 20.0
    for g in games:
        lat, fi, mv, ev = g["lat"], g["fi"], g["mv"], g["ev"]
        agent.reset()
        lo = min(args.start, max(len(fi) - args.frames, 0))
        upto = min(lo + args.frames, len(fi))
        for i in range(lo, upto):
            f = int(fi[i])
            if f >= len(mv):
                break
            out = agent.act_from_latent(lat[i:i+1].float(), temperature=args.temperature,
                                        gate_bias=args.gate_bias)
            n_frames += 1
            if out.get("gate", True):
                n_cmd += 1
                cells[to_cell(out["movement"])] += 1
                # score the cell ONLY where the human also issued a command
                if ev[f]:
                    tot_scored += 1
                    hit += int(to_cell(out["movement"]) == to_cell(mv[f]))
            n_cast += sum(1 for k, v in out["abilities"].items() if v and k != "AA")
    secs = max(n_frames / fps, 1e-9)
    return dict(frames=n_frames, cmd_s=n_cmd / secs, cast_s=n_cast / secs,
                cell_acc=(hit / tot_scored if tot_scored else float("nan")),
                scored=tot_scored, cells_used=len(cells),
                top_cell_share=(cells.most_common(1)[0][1] / max(n_cmd, 1)) if cells else 0.0)


def human_baseline(args, games):
    n = ncmd = 0
    cells = Counter()
    for g in games:
        ev, mv = g["ev"], g["mv"]
        lo = min(args.start, max(len(ev) - args.frames, 0))
        hi = min(lo + args.frames, len(ev))
        n += (hi - lo)
        ncmd += int(ev[lo:hi].sum())
        for f in np.where(ev[lo:hi])[0] + lo:
            cells[to_cell(mv[f])] += 1
    return dict(frames=n, cmd_s=ncmd / max(n / 20.0, 1e-9), cells_used=len(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-ckpt", default="rollout_stage/transformer_tokenizer_latest.pt")
    ap.add_argument("--latents-dir", default="/mnt/nfs/datasets/replay_latents_v7_bc")
    ap.add_argument("--labels-root", default="/mnt/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--frames", type=int, default=1500)
    ap.add_argument("--start", type=int, default=4000,
                    help="Skip the pre-game window. The first ~60s of every "
                         "recording is loading + walk-to-lane with no clicks "
                         "logged, so scoring there compares to a null baseline.")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gate-bias", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpts", nargs="+", required=True)
    a = ap.parse_args()

    print('parsing held-out games once...', flush=True)
    games = load_games(a)
    hb = human_baseline(a, games)
    print(f"\nHUMAN (same held-out games): {hb['cmd_s']:.2f} cmd/s, "
          f"{hb['cells_used']} distinct cells, {hb['frames']} frames\n")
    print(f"{'checkpoint':44s} {'cmd/s':>7s} {'cell_acc':>9s} {'scored':>7s} "
          f"{'cells':>6s} {'top%':>6s} {'cast/s':>7s}")
    for c in a.ckpts:
        try:
            r = run_one(c, a, games)
            print(f"{os.path.basename(c)[:44]:44s} {r['cmd_s']:7.2f} {r['cell_acc']:9.3f} "
                  f"{r['scored']:7d} {r['cells_used']:6d} {100*r['top_cell_share']:5.1f}% {r['cast_s']:7.2f}")
        except Exception as e:
            print(f"{os.path.basename(c)[:44]:44s} FAILED {type(e).__name__}: {str(e)[:60]}")
    print(f"\nread: cmd/s near {hb['cmd_s']:.2f} = human-like pacing; cell_acc >> "
          f"{1/441:.4f} (chance) = real spatial skill; top% near 100 = collapsed.")


if __name__ == "__main__":
    main()
