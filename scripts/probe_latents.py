#!/usr/bin/env python3
"""Latent probe — how much game-state does the tokenizer actually preserve?

Trains a linear (ridge) probe from the frozen tokenizer latents -> each game-state
variable in the labels (champion HP, gold, level, on-screen & world position). Test
R2 near 1 = that info is linearly present in the latent; near 0 = the tokenizer
compressed it away (the "shit data in" concern). A linear probe is conservative:
the transformer world model can read nonlinear features too, so R2 here is a floor.

    /home/dani/miniconda3/envs/ml/bin/python scripts/probe_latents.py
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")

LAT = "rollout_stage/NA1_5549995114.pt"
LAB = "/srv/nfs/datasets/lol_replays_16_9_772/NA1_5549995114/labels.json"


def ridge(Xtr, ytr, Xte, alpha=1.0):
    """Ridge on PCA-reduced features (fast + standard for linear probing)."""
    ym = ytr.mean()
    A = Xtr.T @ Xtr + alpha * torch.eye(Xtr.shape[1])
    w = torch.linalg.solve(A, Xtr.T @ (ytr - ym))
    return Xte @ w + ym


def pca_reduce(X, k=256):
    """Center, then project onto top-k principal components (randomized SVD)."""
    Xc = X - X.mean(0)
    _, _, V = torch.pca_lowrank(Xc, q=k, niter=4)
    return Xc @ V


def main():
    d = torch.load(LAT, weights_only=True)
    z = d["latents"].float()
    fi = d["frame_indices"].numpy()
    N = z.shape[0]
    X = pca_reduce(z.reshape(N, -1), k=256)              # (N, 256) top PCs of the 8192-d latent
    labf = {f["frame"]: f["label"] for f in json.load(open(LAB))["frames"]}

    def g(fr, *path):
        v = labf.get(int(fr))
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
            if v is None:
                return np.nan
        return v

    tg = {"champ_hp_frac": [], "gold_total": [], "level": [],
          "screen_x": [], "screen_y": [], "world_x": [], "world_y": []}
    for fr in fi:
        hp, hpm = g(fr, "champion_stats", "hp"), g(fr, "champion_stats", "hp_max")
        tg["champ_hp_frac"].append(hp / hpm if hpm else np.nan)
        tg["gold_total"].append(g(fr, "champion_stats", "gold_total"))
        tg["level"].append(g(fr, "champion_stats", "level"))
        cs = g(fr, "champion_screen"); cs = cs if isinstance(cs, list) else [np.nan, np.nan]
        cw = g(fr, "champion_world"); cw = cw if isinstance(cw, list) else [np.nan, np.nan]
        tg["screen_x"].append(cs[0]); tg["screen_y"].append(cs[1])
        tg["world_x"].append(cw[0]); tg["world_y"].append(cw[1])

    rng = np.random.default_rng(0)
    rp = torch.from_numpy(rng.permutation(N))
    rnd = (rp[:int(N * 0.8)], rp[int(N * 0.8):])              # random split (neighbor leakage)
    # temporal-block split: 5 held-out contiguous chunks -> no adjacent-frame leakage
    idx = torch.arange(N)
    held = torch.zeros(N, dtype=torch.bool)
    for c in range(5):
        s = int(N * (0.15 + 0.17 * c))
        held[s:s + N // 15] = True
    blk = (idx[~held], idx[held])

    def r2_of(y, split):
        tr, te = split
        p = ridge(X[tr], y[tr], X[te])
        yte = y[te]
        return (1 - ((yte - p) ** 2).sum() / ((yte - yte.mean()) ** 2).sum().clamp_min(1e-9)).item()

    print(f"latent probe: {N} frames, 8192-d latents (top-256 PCs), ridge")
    print(f"{'target':16s}  random_R2  block_R2   readout (block = strict, no neighbor leak)")
    for name, y in tg.items():
        y = np.array(y, float)
        ok = np.isfinite(y)
        if ok.sum() < N * 0.5:
            print(f"  {name:16s}  n/a (sparse labels)")
            continue
        y = torch.from_numpy(np.where(ok, y, np.nanmean(y[ok]))).float()
        rr, br = r2_of(y, rnd), r2_of(y, blk)
        tag = ("STRONG (preserved)" if br > 0.7 else "partial" if br > 0.4
               else "weak" if br > 0.15 else "LOST (not encoded)")
        print(f"  {name:16s}   {rr:+.2f}     {br:+.2f}    {tag}")
    print("\nR2 = variance of that variable linearly readable from ONE latent frame.")
    print("block_R2 is the honest number; random_R2 inflated by adjacent-frame similarity.")


if __name__ == "__main__":
    main()
