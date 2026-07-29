#!/usr/bin/env python3
"""Simulated end-to-end BC eval (no live game).

Runs the trained GarenAgent frame-by-frame over a replay's REAL latents, exactly
as it would live, and scores its predicted NEXT action against the LOGGED human
action (BC predicts offset n=1). This is the imitation-accuracy metric — the real
"does the policy work" number — plus plumbing coverage of the full e2e path.

Metrics:
  movement  MAE / bin-accuracy vs logged, and vs an always-center baseline; also
            restricted to frames where the human actually issued a move.
  abilities per-key precision / recall / F1 (casts are sparse -> F1 is the signal),
            vs a never-cast baseline (F1=0).

    PYTHONPATH=src python scripts/eval_bc_sim.py \
        --phase2-ckpt data/phase2_bc_garen/agent_finetune_latest.pt \
        --match NA1_5549995114 --frames 800
"""
import argparse
import glob
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, "scripts")
from agent_infer import GarenAgent
from ahriuwu.constants import ABILITY_KEYS
from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-ckpt", default="data/phase2_bc_garen/agent_finetune_latest.pt")
    ap.add_argument("--match", default="NA1_5549995114")
    ap.add_argument("--latents-dir", default="rollout_stage")
    ap.add_argument("--labels-root", default="/srv/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ability-thresh", type=float, default=0.0,
                    help="Greedy cast logit threshold (default 0=never casts; try -4.0 calibrated)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    # clean latents dir with just this match (avoid globbing sibling .pt files)
    tmp = tempfile.mkdtemp()
    src = glob.glob(f"{args.latents_dir}/{args.match}.pt")[0]
    os.symlink(os.path.abspath(src), f"{tmp}/{args.match}.pt")
    N = args.frames
    ds = ReplayLatentSequenceDataset(latents_dir=tmp, labels_root=args.labels_root,
                                     outcomes={args.match: False}, sequence_length=N + 1, stride=N + 1)
    s = ds[0]
    lat = s["latents"].float()                                   # (N+1, C, 16, 16)
    move_gt = s["actions"]["movement"].float().numpy()           # (N+1, 2)
    abil_gt = torch.stack([s["actions"][k].float() for k in ABILITY_KEYS], -1).numpy().astype(bool)  # (N+1,9)

    ag = GarenAgent(args.phase2_ckpt, tokenizer_ckpt=None, context=args.context, device=dev,
                    ability_thresh=args.ability_thresh)
    print(f"sim e2e: {args.match}, {N} frames, ctx={args.context}, temp={args.temperature}, bf16={ag.amp}")
    ag.reset()
    pm = np.zeros((N, 2), np.float32)
    pa = np.zeros((N, len(ABILITY_KEYS)), bool)
    import time
    t0 = time.perf_counter()
    for t in range(N):
        a = ag.act_from_latent(lat[t:t + 1].to(dev), temperature=args.temperature)
        pm[t] = a["movement"]
        pa[t] = [a["abilities"][k] for k in ABILITY_KEYS]
    if dev == "cuda":
        torch.cuda.synchronize()
    fps = N / (time.perf_counter() - t0)

    # ground truth = the NEXT logged action (BC offset 1)
    gm = move_gt[1:N + 1]                                        # (N, 2)
    ga = abil_gt[1:N + 1]                                        # (N, 9)

    # --- movement ---
    mae = np.abs(pm - gm).mean()
    base_mae = np.abs(0.5 - gm).mean()
    binp, bing = np.round(pm * 20).astype(int), np.round(gm * 20).astype(int)
    bin_acc = (binp == bing).all(1).mean()
    moved = (np.abs(gm - 0.5).sum(1) > 0.02)                     # frames the human actually moved
    mae_moved = np.abs(pm - gm)[moved].mean() if moved.any() else float("nan")
    uniq = len({tuple(r) for r in np.round(pm, 2)})

    print(f"\nspeed: {fps:.1f} fps")
    print(f"MOVEMENT  MAE={mae:.3f}  (center-baseline {base_mae:.3f})  bin-acc={bin_acc:.1%}  "
          f"unique-cells={uniq}  MAE|moved={mae_moved:.3f} (moved {moved.mean():.0%} of frames)")
    print("ABILITY   per-key precision / recall / F1  (support = human presses in window):")
    f1s = []
    for i, k in enumerate(ABILITY_KEYS):
        tp = int((pa[:, i] & ga[:, i]).sum()); fp = int((pa[:, i] & ~ga[:, i]).sum()); fn = int((~pa[:, i] & ga[:, i]).sum())
        sup = int(ga[:, i].sum()); apr = int(pa[:, i].sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if sup or apr:
            f1s.append(f1)
            print(f"  {k:7s} P={prec:.2f} R={rec:.2f} F1={f1:.2f}   human={sup:4d}  agent={apr:4d}")
    print(f"\nSUMMARY: movement bin-acc {bin_acc:.1%} (baseline ~{1/441:.1%} random) | "
          f"mean ability F1 {np.mean(f1s) if f1s else 0:.2f} | "
          f"{'NON-DEGENERATE' if uniq > 5 else 'DEGENERATE'} policy")


if __name__ == "__main__":
    main()
