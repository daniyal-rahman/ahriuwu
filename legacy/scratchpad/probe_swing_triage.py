#!/usr/bin/env python3
"""Is 'this swing will land a last-hit' recoverable from the v7 latents AT ALL?

The frozen reward head cannot do it (AUC 0.43 at the swing frame, n=241). Two
diagnoses, very different consequences for v1.5:

  (A) HEAD-limited: the info is in the latents, the head was never trained to
      surface it at that horizon -> cheap fix, train a head/probe.
  (B) LATENT-limited: 352x352 -> 16x16 (~22 px cells) destroyed the minion HP
      bars -> no head can recover it, v1.5 needs a pixel-level HP reader.

Test: take every frame where Garen's action state ENTERS "attack" (the commit
moment -- he was not attacking the frame before, so no windup is on screen yet),
and train a probe on the raw latent at that frame to predict whether >=10 gold
arrives in the next 12 frames (600 ms). Train on N matches, evaluate on
held-out. Base rate ~45%, so this is a balanced, well-posed problem.

If a freshly trained probe also lands at ~0.5, the information is not in the
latents and (B) is the answer.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn


def parse_match(labels_root, mid, thresh=0.010, horizon=12):
    """Return (commit_frames, labels) — frames where action.type enters 'attack',
    and whether >=thresh reward lands within `horizon` frames after."""
    with open(os.path.join(labels_root, mid, "labels.json")) as f:
        d = json.load(f)
    frames = d.get("frames") or []
    T = len(frames)
    r = np.zeros(T)
    commit = np.zeros(T, dtype=bool)
    prev_gold, prev_type = None, None
    for i, fr in enumerate(frames):
        lab = fr.get("label")
        if not lab:
            prev_gold, prev_type = None, None
            continue
        cs = lab.get("champion_stats")
        if cs is not None:
            g = cs.get("gold_total")
            if g is None:
                prev_gold = None
            else:
                if prev_gold is not None:
                    r[i] = 1e-3 * (float(g) - prev_gold)
                prev_gold = float(g)
        atype = (lab.get("action") or {}).get("type")
        if atype == "attack" and prev_type != "attack":
            commit[i] = True
        prev_type = atype
    csum = np.concatenate([[0.0], np.cumsum(r)])
    idx = np.where(commit)[0]
    idx = idx[(idx >= 3) & (idx + horizon + 1 < T)]
    y = (csum[idx + horizon + 1] - csum[idx + 1]) >= thresh
    return idx, y.astype(np.float32)


def build(mids, latents_dir, labels_root, horizon, thresh):
    Xs, Ys = [], []
    for mid in mids:
        p = os.path.join(latents_dir, f"{mid}.pt")
        if not os.path.exists(p):
            continue
        idx, y = parse_match(labels_root, mid, thresh, horizon)
        d = torch.load(p, weights_only=True)
        lat, fi = d["latents"], d["frame_indices"].numpy()
        f2i = {int(f): i for i, f in enumerate(fi)}
        keep, ky = [], []
        for f, yy in zip(idx, y):
            i = f2i.get(int(f))
            if i is not None and i - 3 >= 0:
                keep.append(i); ky.append(yy)
        if not keep:
            del d, lat
            continue
        keep = np.array(keep)
        a = lat[keep].float()
        b = lat[keep - 3].float()      # 150 ms earlier -> motion / incoming damage
        Xs.append(torch.cat([a.flatten(1), (a - b).flatten(1)], 1))
        Ys.append(torch.tensor(ky))
        print(f"    {mid}: {len(keep)} commits, pos={float(np.mean(ky)):.2f}", flush=True)
        del d, lat, a, b
    return torch.cat(Xs), torch.cat(Ys)


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    o = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[o] = np.arange(len(allv), dtype=float)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-bc", default="/srv/nfs/datasets/replay_latents_v7_bc")
    ap.add_argument("--latents-ho", default="/srv/nfs/datasets/replay_latents_v7_heldout")
    ap.add_argument("--labels-root", default="/srv/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--heldout", default="NA1_5549981347")
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--thresh", type=float, default=0.010)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open("scratchpad/_usable_matches.txt") as f:
        train = [l.strip() for l in f if l.strip()][: args.n_train]
    print("building train...")
    Xtr, Ytr = build(train, args.latents_bc, args.labels_root, args.horizon, args.thresh)
    print("building heldout...")
    Xte, Yte = build([args.heldout], args.latents_ho, args.labels_root, args.horizon, args.thresh)
    print(f"\ntrain {tuple(Xtr.shape)} pos={Ytr.mean():.3f}   "
          f"heldout {tuple(Xte.shape)} pos={Yte.mean():.3f}")

    dev = args.device
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-5
    Xte_n = ((Xte - mu) / sd)
    Yte_np = Yte.numpy()
    # internal val split from train (last 15%) to see if it even fits in-domain
    ntr = int(len(Xtr) * 0.85)
    perm = torch.randperm(len(Xtr), generator=torch.Generator().manual_seed(0))
    tr, va = perm[:ntr], perm[ntr:]

    for hidden, wd in [(0, 1.0), (0, 0.1), (256, 0.1)]:
        torch.manual_seed(0)
        D = Xtr.shape[1]
        model = (nn.Linear(D, 1) if hidden == 0 else
                 nn.Sequential(nn.Linear(D, hidden), nn.SiLU(), nn.Linear(hidden, 1))).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
        lossf = nn.BCEWithLogitsLoss()
        print(f"\n-- probe hidden={hidden} wd={wd}")
        for ep in range(30):
            model.train()
            p = tr[torch.randperm(len(tr))]
            for i in range(0, len(p), 256):
                idx = p[i:i + 256]
                xb = ((Xtr[idx] - mu) / sd).to(dev); yb = Ytr[idx].to(dev)
                opt.zero_grad(); lossf(model(xb).squeeze(-1), yb).backward(); opt.step()
            if (ep + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    sv = torch.cat([model(((Xtr[va[i:i+2048]] - mu) / sd).to(dev)).squeeze(-1).cpu()
                                    for i in range(0, len(va), 2048)]).numpy()
                    st = torch.cat([model(Xte_n[i:i+2048].to(dev)).squeeze(-1).cpu()
                                    for i in range(0, len(Xte_n), 2048)]).numpy()
                yv = Ytr[va].numpy()
                print(f"   ep{ep+1:3d}  train-val AUC={auc(sv[yv==1], sv[yv==0]):.3f}   "
                      f"HELDOUT AUC={auc(st[Yte_np==1], st[Yte_np==0]):.3f}", flush=True)


if __name__ == "__main__":
    main()
