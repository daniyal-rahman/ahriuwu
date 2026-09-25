#!/usr/bin/env python3
"""Nonlinear (MLP) cross-game probe: is HUD detail legible in the v7 latents?

The linear full-dim probe read champ HP fraction at cross-game R2=0.09 ("weak").
But the policy reads latents NONLINEARLY, so a linear probe is a lower bound.
This trains a small MLP (and a linear head under the identical protocol, as the
baseline/sanity) on latents -> [hp_frac, level/18, screen_x, screen_y], holding
out whole games. If the MLP recovers HP well the tokenizer preserves the info
and the "tokenizer blurs the HUD away" worry is retired; if not, tokenizer
capacity/res becomes a real workstream.

    /home/dani/miniconda3/envs/ml/bin/python scripts/probe_hp_mlp.py
"""
import glob
import json
import sys

import numpy as np
import torch
import torch.nn as nn

LATDIR = "/srv/nfs/datasets/replay_latents_v7_bc"
LABROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
NGAMES, STEP, FOLDS = 8, 3, 3
TARGETS = ["hp_frac", "level", "screen_x", "screen_y"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_game(pt):
    d = torch.load(pt, weights_only=True)
    z = d["latents"].float()
    fi = d["frame_indices"].numpy()
    match = pt.split("/")[-1][:-3]
    labf = {f["frame"]: f["label"]
            for f in json.load(open(f"{LABROOT}/{match}/labels.json"))["frames"]}

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
        lvl = g(fr, "champion_stats", "level")
        cs = g(fr, "champion_screen")
        cs = cs if isinstance(cs, list) else [np.nan, np.nan]
        rows.append([hp / hpm if hpm else np.nan,
                     lvl / 18.0 if lvl == lvl else np.nan, cs[0], cs[1]])
    return (z.reshape(z.shape[0], -1)[::STEP],
            torch.from_numpy(np.array(rows, float)[::STEP]).float())


def r2(pred, y):
    return (1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-9)).item()


class MLP(nn.Module):
    def __init__(self, din, dout, hidden=(1024, 256)):
        super().__init__()
        layers, d = [], din
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_eval(model, Xtr, Ytr, Mtr, Xte, Yte, Mte, epochs=30, bs=512, lr=5e-4):
    """Train on masked targets (per-target NaN mask), return per-target held-out R2."""
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = Xtr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            x = Xtr[idx].to(DEV, non_blocking=True)
            y = Ytr[idx].to(DEV, non_blocking=True)
            m = Mtr[idx].to(DEV, non_blocking=True)
            loss = (((model(x) - y) ** 2) * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, Xte.shape[0], 2048):
            preds.append(model(Xte[i:i + 2048].to(DEV)).cpu())
    pred = torch.cat(preds)
    out = []
    for j in range(Yte.shape[1]):
        m = Mte[:, j]
        out.append(r2(pred[m, j], Yte[m, j]) if m.sum() > 10 else float("nan"))
    return out


def main():
    torch.manual_seed(0)
    pts = sorted(glob.glob(f"{LATDIR}/NA1_*.pt"))[:NGAMES]
    games = [load_game(p) for p in pts]
    print(f"MLP cross-game probe: {len(pts)} games on {DEV}, stride {STEP}, "
          f"{sum(g[0].shape[0] for g in games)} frames total", flush=True)
    Xall = torch.cat([g[0] for g in games])
    mu, sd = Xall.mean(0), Xall.std(0) + 1e-6
    Xs = [((g[0] - mu) / sd) for g in games]
    Ys = [torch.nan_to_num(g[1], nan=0.0) for g in games]
    Ms = [torch.isfinite(g[1]) for g in games]
    del Xall

    res = {"mlp": {t: [] for t in TARGETS}, "linear": {t: [] for t in TARGETS}}
    for te in range(FOLDS):
        tr = [i for i in range(len(pts)) if i != te]
        Xtr = torch.cat([Xs[i] for i in tr])
        Ytr = torch.cat([Ys[i] for i in tr])
        Mtr = torch.cat([Ms[i] for i in tr]).float()
        # standardize each target on TRAIN (masked) — raw screen coords are in
        # pixels (var ~1e4) and would otherwise drown hp_frac in the shared loss.
        # R2 is affine-invariant, so reported numbers are unchanged in meaning.
        ym = (Ytr * Mtr).sum(0) / Mtr.sum(0).clamp_min(1.0)
        ysd = (((Ytr - ym) ** 2 * Mtr).sum(0) / Mtr.sum(0).clamp_min(1.0)).sqrt() + 1e-6
        Ytr = (Ytr - ym) / ysd
        Yte_n = (Ys[te] - ym) / ysd
        print(f"fold {te}: train {Xtr.shape[0]} rows, test {Xs[te].shape[0]} "
              f"({pts[te].split('/')[-1]})", flush=True)
        for name, model in [("mlp", MLP(8192, len(TARGETS))),
                            ("linear", nn.Linear(8192, len(TARGETS)))]:
            r = fit_eval(model, Xtr, Ytr, Mtr, Xs[te], Yte_n, Ms[te])
            for t, v in zip(TARGETS, r):
                res[name][t].append(v)
            print(f"  {name:6s}: " + "  ".join(f"{t}={v:+.2f}" for t, v in zip(TARGETS, r)),
                  flush=True)

    print(f"\n{'target':10s}  {'MLP_R2':>7s}  {'linear_R2':>9s}   (mean over {FOLDS} held-out games)")
    for t in TARGETS:
        m, l = float(np.mean(res["mlp"][t])), float(np.mean(res["linear"][t]))
        tag = ("STRONG" if m > 0.6 else "partial" if m > 0.3 else "weak" if m > 0.1 else "LOST")
        print(f"  {t:10s}  {m:+7.2f}  {l:+9.2f}   {tag}")
    print("\nread: MLP >> linear and STRONG => info present, tokenizer fine (probe was the issue);")
    print("      MLP ~ linear and weak     => the tokenizer really does blur the HUD away.")


if __name__ == "__main__":
    main()
