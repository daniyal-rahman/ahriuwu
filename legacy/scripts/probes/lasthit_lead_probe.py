#!/usr/bin/env python3
"""Is a last-hit OPPORTUNITY visible before the swing — and for how long?

Dani's point: humans don't compute damage, they watch the wave and act when a
minion is lowest relative to what's about to finish it. That is a TEMPORAL,
VISUAL cue, and it may form well before the swing. Every previous test here was
either single-frame or capped at the reward head's 400 ms MTP horizon, so this
hypothesis has never actually been tested.

Design: anchor on the GOLD event (which needs no AA label, so chained autos are
not silently dropped). For lead L in frames, take the window ENDING at
gold - L and ask: is a last-hit coming at +L?
  positives: gold lands at t+L
  negatives: frames with no gold in [t, t+40], drawn from the SAME game and the
             SAME 2-minute slice as a positive (kills the clock/level confound
             that made naive AUC ~0.70 last time)

Three input variants at each lead, so we separate representation from horizon:
  pix1  - single raw 352 frame           (single-frame visual ceiling)
  pixT  - T raw frames, stacked          (TEMPORAL visual ceiling <- the new one)
  latT  - T v7 latents                   (does the tokenizer keep it)

Read: pixT >> pix1 => the cue really is temporal.  pixT >> latT => tokenizer
loses it.  all ~chance at L>=10 => the opportunity is not visible that early.
"""
import argparse
import glob
import json
import sys

import cv2
import os as _os
_NFS = "/mnt/nfs" if _os.path.isdir("/mnt/nfs/datasets") else "/srv/nfs"
import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

ROOT = f"{_NFS}/datasets/lol_replays_16_9_772"
LATD = f"{_NFS}/datasets/replay_latents_v7_bc"
import json as _json
_V = _json.load(open("scratchpad/valid_games.json"))["both"]
TRAIN_GAMES = _V[:8]
HELD_GAMES = _V[100:103]
LEADS = (2, 6, 10, 20, 40)          # frames before the gold: 0.1 .. 2.0 s
T = 8                                # temporal window length
GOLD_MIN = 10                        # a last-hit is >=10 gold


def gold_events(match):
    lab = json.load(open(f"{ROOT}/{match}/labels.json"))
    fr = lab["frames"]
    g = np.full(len(fr), np.nan)
    for i, f in enumerate(fr):
        cs = (f.get("label") or {}).get("champion_stats") or {}
        v = cs.get("gold_total")
        if v is not None:
            g[i] = float(v)
    ok = ~np.isnan(g)
    d = np.zeros(len(g))
    idx = np.where(ok)[0]
    d[idx[1:]] = np.diff(g[idx])
    ev = np.where(d >= GOLD_MIN)[0]
    return ev, d, len(fr)


def build_events(games, lead, per_game=120, seed=0):
    """(match, t_end, label) with negatives matched by game and 2-min slice."""
    rng = np.random.RandomState(seed)
    out = []
    for m in games:
        try:
            ev, d, n = gold_events(m)
        except Exception:
            continue
        has_gold = np.zeros(n, bool)
        for e in ev:
            has_gold[max(e - 40, 0):e + 1] = True     # no-gold-in-next-40 test
        pos = [e - lead for e in ev if e - lead - T >= 0]
        rng.shuffle(pos)
        pos = pos[:per_game]
        for t in pos:
            slice_id = int(t / (120 * 20))            # 2-min bucket
            cand = np.where(~has_gold)[0]
            cand = cand[(cand > T) & (cand < n - 45)]
            cand = cand[np.abs(cand / (120 * 20) - slice_id) < 1]   # same slice
            if len(cand) == 0:
                continue
            neg = int(rng.choice(cand))
            out.append((m, int(t), 1))
            out.append((m, neg, 0))
    return out


def load_pixels(ev, T_len):
    X, Y = [], []
    cache = {}
    for m, t, y in ev:
        fs = cache.get(m)
        if fs is None:
            fs = sorted(glob.glob(f"{ROOT}/{m}/frames/*.png")); cache[m] = fs
        if t >= len(fs) or t - T_len + 1 < 0:
            continue
        ims = []
        for k in range(t - T_len + 1, t + 1):
            im = cv2.imread(fs[k], cv2.IMREAD_GRAYSCALE)
            if im is None:
                ims = None; break
            ims.append(cv2.resize(im, (96, 96)).astype(np.float32) / 255)
        if ims is None:
            continue
        X.append(np.stack(ims)); Y.append(y)
    return np.stack(X), np.array(Y)


def load_latents(ev, T_len):
    X, Y = [], []
    cache = {}
    for m, t, y in ev:
        if m not in cache:
            d = torch.load(f"{LATD}/{m}.pt", weights_only=True)
            cache[m] = (d["latents"].float().numpy(), d["frame_indices"].numpy())
        lat, fi = cache[m]
        pos = np.searchsorted(fi, t)
        if pos - T_len + 1 < 0 or pos >= len(fi) or fi[pos] != t:
            continue
        X.append(lat[pos - T_len + 1:pos + 1].reshape(T_len, -1)); Y.append(y)
    if not X:
        return np.zeros((0, T_len, 8192), np.float32), np.array([])
    return np.stack(X), np.array(Y)


class Small(torch.nn.Module):
    def __init__(self, cin, flat=False, dim=8192):
        super().__init__()
        self.flat = flat
        if flat:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(cin * dim, 256), torch.nn.ReLU(),
                torch.nn.Dropout(0.2), torch.nn.Linear(256, 1))
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(cin, 32, 5, 2, 2), torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, 2, 1), torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 3, 2, 1), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                torch.nn.Dropout(0.2), torch.nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x.flatten(1) if self.flat else x).squeeze(-1)


def fit_auc(Xtr, ytr, Xte, yte, flat, dev, seeds=(0, 1, 2), epochs=25):
    aucs = []
    for sd in seeds:
        torch.manual_seed(sd)
        net = Small(Xtr.shape[1], flat=flat, dim=Xtr.shape[-1] if flat else 0).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        A = torch.tensor(Xtr, dtype=torch.float32)
        yt = torch.tensor(ytr, dtype=torch.float32)
        n, bs = len(A), 64
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    net(A[idx].to(dev)), yt[idx].to(dev))
                loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            s = np.concatenate([net(torch.tensor(Xte[i:i+128], dtype=torch.float32).to(dev)).cpu().numpy()
                                for i in range(0, len(Xte), 128)])
        pos, neg = s[yte == 1], s[yte == 0]
        if not len(pos) or not len(neg):
            continue
        r = np.concatenate([pos, neg]).argsort().argsort().astype(float)
        aucs.append((r[:len(pos)].sum() - len(pos)*(len(pos)-1)/2) / (len(pos)*len(neg)))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-game", type=int, default=110)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    print(f"leads (frames before gold): {LEADS}   T={T}   3 seeds each", flush=True)
    print(f"{'lead':>5s} {'ms':>5s} | {'pix1':>13s} {'pixT':>13s} {'latT':>13s}   n_tr/n_te")
    for L in LEADS:
        tr_ev = build_events(TRAIN_GAMES, L, args.per_game, seed=0)
        te_ev = build_events(HELD_GAMES, L, args.per_game, seed=1)
        row = {}
        # pixels
        Xtr, ytr = load_pixels(tr_ev, T); Xte, yte = load_pixels(te_ev, T)
        if len(Xtr) < 40 or len(Xte) < 20:
            print(f"{L:>5d} {L*50:>5d} |  too few events"); continue
        a1, s1 = fit_auc(Xtr[:, -1:], ytr, Xte[:, -1:], yte, False, dev)
        aT, sT = fit_auc(Xtr, ytr, Xte, yte, False, dev)
        # latents
        Ltr, lytr = load_latents(tr_ev, T); Lte, lyte = load_latents(te_ev, T)
        if len(Ltr) > 40 and len(Lte) > 20:
            aL, sL = fit_auc(Ltr, lytr, Lte, lyte, True, dev)
        else:
            aL, sL = float("nan"), float("nan")
        print(f"{L:>5d} {L*50:>5d} | {a1:6.3f}+-{s1:.3f} {aT:6.3f}+-{sT:.3f} "
              f"{aL:6.3f}+-{sL:.3f}   {len(ytr)}/{len(yte)}", flush=True)
    print("\nread: pixT>>pix1 => cue is temporal; pixT>>latT => tokenizer loses it;")
    print("      all ~0.5 at large L => no early opportunity signal exists.")


if __name__ == "__main__":
    main()
