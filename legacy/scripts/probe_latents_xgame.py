#!/usr/bin/env python3
"""Cross-game latent probe — the definitive tokenizer-fidelity test.

TRAIN on a set of games, TEST on held-out games (no neighbor leakage, matched
distributions). Two probes per target:
  linear : ridge on top-256 PCs, alpha picked on a validation game (properly reg.)
  mlp    : 256->128->1 (the WM is nonlinear, so this is the real "is it in there")
R2 on the held-out game = honest recoverability. Linear<mlp means "present but
nonlinear"; both low means genuinely degraded by the tokenizer.

    /home/dani/miniconda3/envs/ml/bin/python scripts/probe_latents_xgame.py
"""
import glob
import json
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")

LATDIR = "/srv/nfs/datasets/replay_latents_v7_bc"
LABROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
NGAMES = 6
STEP = 5
NAMES = ["champ_hp_frac", "gold_total", "level", "screen_x", "screen_y", "world_x", "world_y"]


def load_game(pt):
    d = torch.load(pt, weights_only=True)
    z = d["latents"].float()
    fi = d["frame_indices"].numpy()
    match = pt.split("/")[-1][:-3]
    labf = {f["frame"]: f["label"] for f in json.load(open(f"{LABROOT}/{match}/labels.json"))["frames"]}

    def g(fr, *path):
        v = labf.get(int(fr))
        for p in path:
            v = v.get(p) if isinstance(v, dict) else None
            if v is None:
                return np.nan
        return v

    rows = []
    for fr in fi:
        hp, hpm = g(fr, "champion_stats", "hp"), g(fr, "champion_stats", "hp_max")
        cs = g(fr, "champion_screen"); cs = cs if isinstance(cs, list) else [np.nan, np.nan]
        cw = g(fr, "champion_world"); cw = cw if isinstance(cw, list) else [np.nan, np.nan]
        rows.append([hp / hpm if hpm else np.nan, g(fr, "champion_stats", "gold_total"),
                     g(fr, "champion_stats", "level"), cs[0], cs[1], cw[0], cw[1]])
    return z.reshape(z.shape[0], -1)[::STEP], torch.from_numpy(np.array(rows, float)[::STEP]).float()


def r2(pred, y):
    return (1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-9)).item()


def ridge_sweep(Xtr, ytr, Xva, yva, Xte):
    best_a, best = None, -1e9
    for a in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]:
        w = torch.linalg.solve(Xtr.T @ Xtr + a * torch.eye(Xtr.shape[1]), Xtr.T @ (ytr - ytr.mean()))
        if r2(Xva @ w + ytr.mean(), yva) > best:
            best, best_a = r2(Xva @ w + ytr.mean(), yva), a
    w = torch.linalg.solve(Xtr.T @ Xtr + best_a * torch.eye(Xtr.shape[1]), Xtr.T @ (ytr - ytr.mean()))
    return Xte @ w + ytr.mean()


def mlp_probe(Xtr, ytr, Xva, yva, Xte):
    if Xtr.shape[0] > 6000:                              # cap rows for speed
        idx = torch.randperm(Xtr.shape[0])[:6000]
        Xtr, ytr = Xtr[idx], ytr[idx]
    ym, ys = ytr.mean(), ytr.std() + 1e-6
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.GELU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    ytn = ((ytr - ym) / ys).unsqueeze(1)
    best_state, best = None, -1e9
    for ep in range(250):
        opt.zero_grad()
        loss = ((net(Xtr) - ytn) ** 2).mean()
        loss.backward(); opt.step()
        if ep % 25 == 0:
            with torch.no_grad():
                v = r2(net(Xva).squeeze(1) * ys + ym, yva)
                if v > best:
                    best = v
                    best_state = {k: val.clone() for k, val in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        return net(Xte).squeeze(1) * ys + ym


def main():
    pts = sorted(glob.glob(f"{LATDIR}/NA1_*.pt"))[:NGAMES]
    games = [load_game(p) for p in pts]
    print(f"cross-game probe: {len(pts)} games, every {STEP}th frame, sizes {[g[0].shape[0] for g in games]}")
    Xall = torch.cat([g[0] for g in games]); mu = Xall.mean(0)
    _, _, V = torch.pca_lowrank(Xall - mu, q=256, niter=4)
    sd = ((Xall - mu) @ V).std(0) + 1e-6
    Xp = [(((g[0] - mu) @ V) / sd) for g in games]       # standardized PCs
    Yp = [g[1] for g in games]

    lin, mlp = {n: [] for n in NAMES}, {n: [] for n in NAMES}
    for te in range(3):                                   # hold out each of first 3 games
        va = (te + 1) % len(pts)
        tr = [i for i in range(len(pts)) if i not in (te, va)]
        for j, n in enumerate(NAMES):
            def clean(ids):
                X = torch.cat([Xp[i] for i in ids]); y = torch.cat([Yp[i][:, j] for i in ids])
                m = torch.isfinite(y); return X[m], y[m]
            Xtr, ytr = clean(tr); Xva, yva = clean([va]); Xte, yte = clean([te])
            if len(yte) < 10 or len(ytr) < 100:
                continue
            lin[n].append(r2(ridge_sweep(Xtr, ytr, Xva, yva, Xte), yte))
            mlp[n].append(r2(mlp_probe(Xtr, ytr, Xva, yva, Xte), yte))

    print(f"\n{'target':16s}  linear_R2  mlp_R2   readout (mean over 3 held-out games)")
    for n in NAMES:
        lv = float(np.mean(lin[n])) if lin[n] else float("nan")
        mv = float(np.mean(mlp[n])) if mlp[n] else float("nan")
        best = max(lv, mv)
        tag = ("STRONG (preserved)" if best > 0.6 else "partial" if best > 0.3
               else "weak" if best > 0.1 else "LOST (degraded by tokenizer)")
        print(f"  {n:16s}  {lv:+.2f}      {mv:+.2f}    {tag}")
    print("\nlinear<<mlp => info present but nonlinear (WM can use it). both low => tokenizer lost it.")


if __name__ == "__main__":
    main()
