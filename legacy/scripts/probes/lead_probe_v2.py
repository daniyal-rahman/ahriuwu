#!/usr/bin/env python3
"""Lead probe v2 — two corrections that make the previous run uninterpretable.

FIX 1 (pixel arm was blinded): v1 fed the CNN 96x96 GRAYSCALE frames. A minion
bar is 9-18px wide and 1-3px tall in COLOUR at 352 — downsampling to 96 grey
destroys precisely the feature under test, so "latents beat pixels" was an
artifact of my own preprocessing. v2 feeds a 176x176 COLOUR crop centred on the
champion (where the contested wave is), preserving bar-scale detail.

FIX 2 (negatives were too easy): v1 negatives were "no gold within 2s", so the
probe could win by detecting "am I in a wave fight at all" — context, not
timing. That is why AUC stayed 0.72 even 2s out. v2 draws negatives from INSIDE
the same engagement: frames within +-3s of a gold event but NOT within the lead
window, same game, same 2-min slice. Now both classes are "in a wave, trading";
only the imminence of THIS last-hit differs.

If the fine cue Dani describes exists, v2 shows a STEEP decay with lead. If v2
is flat, the earlier signal was context and there is no early timing cue.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

_NFS = "/mnt/nfs" if os.path.isdir("/mnt/nfs/datasets") else "/srv/nfs"
ROOT = f"{_NFS}/datasets/lol_replays_16_9_772"
LATD = f"{_NFS}/datasets/replay_latents_v7_bc"
_V = json.load(open("scratchpad/valid_games.json"))["both"]
TRAIN_GAMES, HELD_GAMES = _V[:10], _V[100:104]
LEADS = (2, 6, 10, 20, 40)
T = 8
GOLD_MIN = 10
CROP = 176            # colour crop, native resolution (no downsample)
LABEL_SPACE = (1280.0, 720.0)


def game_tables(match):
    lab = json.load(open(f"{ROOT}/{match}/labels.json"))
    fr = lab["frames"]
    n = len(fr)
    gold = np.full(n, np.nan)
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    for i, f in enumerate(fr):
        l = f.get("label") or {}
        cs = l.get("champion_stats") or {}
        if cs.get("gold_total") is not None:
            gold[i] = float(cs["gold_total"])
        scr = l.get("champion_screen")
        if isinstance(scr, list) and len(scr) == 2:
            cx[i], cy[i] = scr
    ok = np.where(~np.isnan(gold))[0]
    d = np.zeros(n)
    if len(ok) > 1:
        d[ok[1:]] = np.diff(gold[ok])
    ev = np.where(d >= GOLD_MIN)[0]
    return ev, n, cx, cy


def build_events(games, lead, per_game, seed):
    """positives: window ends at gold-lead.
    negatives: IN-ENGAGEMENT — within 3s of some gold event, same 2-min slice,
    but no gold within [t, t+lead+8] (so this window is genuinely not a
    lead-up to a last-hit)."""
    rng = np.random.RandomState(seed)
    out = []
    for m in games:
        try:
            ev, n, cx, cy = game_tables(m)
        except Exception:
            continue
        if len(ev) < 20:
            continue
        near = np.zeros(n, bool)          # inside an engagement
        soon = np.zeros(n, bool)          # a gold lands shortly after
        for e in ev:
            near[max(e - 60, 0):min(e + 60, n)] = True
            soon[max(e - lead - 8, 0):min(e + 2, n)] = True
        pos = [e - lead for e in ev if e - lead - T >= 0 and not np.isnan(cx[e - lead])]
        rng.shuffle(pos)
        pos = pos[:per_game]
        neg_pool = np.where(near & ~soon & ~np.isnan(cx))[0]
        neg_pool = neg_pool[(neg_pool > T) & (neg_pool < n - 2)]
        if len(neg_pool) == 0:
            continue
        for t in pos:
            sl = int(t / (120 * 20))
            cand = neg_pool[np.abs(neg_pool / (120 * 20) - sl) < 1]
            if len(cand) == 0:
                cand = neg_pool
            out.append((m, int(t), 1))
            out.append((m, int(rng.choice(cand)), 0))
    return out


def load_pixels(ev, T_len):
    """T colour crops centred on the champion, NATIVE resolution."""
    X, Y = [], []
    fcache, tcache = {}, {}
    for m, t, y in ev:
        if m not in fcache:
            fcache[m] = sorted(glob.glob(f"{ROOT}/{m}/frames/*.png"))
            tcache[m] = game_tables(m)
        fs = fcache[m]
        _, n, cx, cy = tcache[m]
        if t >= len(fs) or t - T_len + 1 < 0 or np.isnan(cx[t]):
            continue
        px = int(cx[t] / LABEL_SPACE[0] * 352)
        py = int(cy[t] / LABEL_SPACE[1] * 352)
        x0 = int(np.clip(px - CROP // 2, 0, 352 - CROP))
        y0 = int(np.clip(py - CROP // 2, 0, 352 - CROP))
        ims = []
        for k in range(t - T_len + 1, t + 1):
            im = cv2.imread(fs[k])
            if im is None:
                ims = None
                break
            c = im[y0:y0 + CROP, x0:x0 + CROP]
            if c.shape[:2] != (CROP, CROP):
                ims = None
                break
            ims.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
        if ims is None:
            continue
        X.append(np.concatenate(ims, axis=2).transpose(2, 0, 1))   # (T*3,H,W)
        Y.append(y)
    if not X:
        return np.zeros((0, T_len * 3, CROP, CROP), np.float32), np.array([])
    return np.stack(X), np.array(Y)


def load_latents(ev, T_len):
    X, Y, cache = [], [], {}
    for m, t, y in ev:
        if m not in cache:
            d = torch.load(f"{LATD}/{m}.pt", weights_only=True)
            cache[m] = (d["latents"].float().numpy(), d["frame_indices"].numpy())
        lat, fi = cache[m]
        p = np.searchsorted(fi, t)
        if p - T_len + 1 < 0 or p >= len(fi) or fi[p] != t:
            continue
        X.append(lat[p - T_len + 1:p + 1].reshape(T_len, -1))
        Y.append(y)
    if not X:
        return np.zeros((0, T_len, 8192), np.float32), np.array([])
    return np.stack(X), np.array(Y)


class CNN(torch.nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Conv2d(cin, 32, 5, 2, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Dropout(0.2), torch.nn.Linear(128, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


class MLP(torch.nn.Module):
    def __init__(self, din):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(din, 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(256, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def auc_of(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def fit(Xtr, ytr, Xte, yte, kind, dev, seeds=(0, 1, 2), epochs=20):
    out = []
    for sd in seeds:
        torch.manual_seed(sd)
        net = (CNN(Xtr.shape[1]) if kind == "cnn" else
               MLP(int(np.prod(Xtr.shape[1:])))).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        n, bs = len(Xtr), 48
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs].numpy()
                xb = torch.tensor(Xtr[idx], dtype=torch.float32).to(dev)
                yb = torch.tensor(ytr[idx], dtype=torch.float32).to(dev)
                opt.zero_grad()
                torch.nn.functional.binary_cross_entropy_with_logits(net(xb), yb).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            s = np.concatenate([net(torch.tensor(Xte[i:i + 64], dtype=torch.float32).to(dev)).cpu().numpy()
                                for i in range(0, len(Xte), 64)])
        out.append(auc_of(s, yte))
    return float(np.mean(out)), float(np.std(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-game", type=int, default=90)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    print("LEAD PROBE v2 — colour native crops + IN-ENGAGEMENT negatives", flush=True)
    print(f"train={len(TRAIN_GAMES)} games  held={len(HELD_GAMES)} games  T={T} crop={CROP}px colour",
          flush=True)
    print(f"{'lead':>5s} {'ms':>5s} | {'pixT(colour)':>14s} {'latT':>14s}   n_tr/n_te  base", flush=True)
    for L in LEADS:
        tr = build_events(TRAIN_GAMES, L, args.per_game, 0)
        te = build_events(HELD_GAMES, L, args.per_game, 1)
        Xtr, ytr = load_pixels(tr, T)
        Xte, yte = load_pixels(te, T)
        if len(Xtr) < 60 or len(Xte) < 30:
            print(f"{L:>5d} {L*50:>5d} | too few events ({len(Xtr)}/{len(Xte)})", flush=True)
            continue
        ap_, as_ = fit(Xtr, ytr, Xte, yte, "cnn", dev)
        Ltr, lytr = load_latents(tr, T)
        Lte, lyte = load_latents(te, T)
        if len(Ltr) > 60 and len(Lte) > 30:
            al, sl = fit(Ltr, lytr, Lte, lyte, "mlp", dev)
        else:
            al = sl = float("nan")
        print(f"{L:>5d} {L*50:>5d} | {ap_:7.3f}+-{as_:.3f} {al:7.3f}+-{sl:.3f}   "
              f"{len(ytr)}/{len(yte)}  {yte.mean():.2f}", flush=True)
    print("\nread: STEEP decay with lead => a real timing cue exists.", flush=True)
    print("      FLAT/high at 2s => still context, no early last-hit cue in this data.", flush=True)


if __name__ == "__main__":
    main()
