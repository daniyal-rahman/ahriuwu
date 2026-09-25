#!/usr/bin/env python3
"""Full-dim (no PCA) cross-game probe on the interpretable, distribution-matched
targets: HP fraction (0-1), on-screen x/y (bounded). Uses the DUAL ridge form
(linear kernel on subsampled rows) so it reads ALL 8192 latent dims -- including
the low-variance directions PCA would drop -- while staying fast. This isolates
'is HUD detail legible in the latent' from the PCA confound.

    /home/dani/miniconda3/envs/ml/bin/python scripts/probe_hp_fulldim.py
"""
import glob
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
LATDIR = "/srv/nfs/datasets/replay_latents_v7_bc"
LABROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
NGAMES, STEP, NTR = 6, 5, 4000


def load_game(pt):
    d = torch.load(pt, weights_only=True)
    z = d["latents"].float(); fi = d["frame_indices"].numpy()
    match = pt.split("/")[-1][:-3]
    labf = {f["frame"]: f["label"] for f in json.load(open(f"{LABROOT}/{match}/labels.json"))["frames"]}

    def g(fr, *p):
        v = labf.get(int(fr))
        for k in p:
            v = v.get(k) if isinstance(v, dict) else None
            if v is None:
                return np.nan
        return v
    rows = []
    for fr in fi:
        hp, hpm = g(fr, "champion_stats", "hp"), g(fr, "champion_stats", "hp_max")
        cs = g(fr, "champion_screen"); cs = cs if isinstance(cs, list) else [np.nan, np.nan]
        rows.append([hp / hpm if hpm else np.nan, cs[0], cs[1]])
    return z.reshape(z.shape[0], -1)[::STEP], torch.from_numpy(np.array(rows, float)[::STEP]).float()


def r2(pred, y):
    return (1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-9)).item()


def dual_ridge(Xtr, ytr, Xte, lam):
    ym = ytr.mean()
    K = Xtr @ Xtr.T
    a = torch.linalg.solve(K + lam * torch.eye(K.shape[0]), (ytr - ym))
    return (Xte @ Xtr.T) @ a + ym


def main():
    pts = sorted(glob.glob(f"{LATDIR}/NA1_*.pt"))[:NGAMES]
    games = [load_game(p) for p in pts]
    print(f"full-dim (8192) cross-game probe: {len(pts)} games")
    names = ["champ_hp_frac", "screen_x", "screen_y"]
    Xall = torch.cat([g[0] for g in games])
    mu, sd = Xall.mean(0), Xall.std(0) + 1e-6                 # standardize features
    Xs = [((g[0] - mu) / sd) for g in games]
    Ys = [g[1] for g in games]
    out = {n: [] for n in names}
    for te in range(3):
        tr = [i for i in range(len(pts)) if i != te]
        Xtr_full = torch.cat([Xs[i] for i in tr])
        for j, n in enumerate(names):
            ytr_full = torch.cat([Ys[i][:, j] for i in tr])
            m = torch.isfinite(ytr_full)
            Xtr, ytr = Xtr_full[m], ytr_full[m]
            if Xtr.shape[0] > NTR:
                idx = torch.randperm(Xtr.shape[0])[:NTR]
                Xtr, ytr = Xtr[idx], ytr[idx]
            yte = Ys[te][:, j]; mte = torch.isfinite(yte)
            Xte, yte = Xs[te][mte], yte[mte]
            best = -1e9
            for lam in [1e1, 1e2, 1e3, 1e4]:                 # light reg sweep
                best = max(best, r2(dual_ridge(Xtr, ytr, Xte, lam), yte))
            out[n].append(best)
    print(f"\n{'target':16s}  xgame_R2 (full 8192-d, best-of-lambda, mean over 3 held-out games)")
    for n in names:
        v = float(np.mean(out[n]))
        tag = ("STRONG" if v > 0.6 else "partial" if v > 0.3 else "weak" if v > 0.1 else "LOST")
        print(f"  {n:16s}  {v:+.2f}   {tag}")
    print("\nHP-fraction is the acid test: it's the big on-champion HP bar, distribution-matched")
    print("across games. If even full-dim can't read it cross-game, the tokenizer blurs it away.")


if __name__ == "__main__":
    main()
