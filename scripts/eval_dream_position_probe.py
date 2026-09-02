#!/usr/bin/env python3
"""Does DECODABLE STATE survive an imagined rollout?

Fits a probe for champion world position on REAL v7 latents (train games only),
then reads that probe on the IMAGINED latents at each horizon step. The metric is
the one docs/DECISION0_SIGNAL_PRESENT.md uses so the numbers are comparable:
two independent 21-way softmax heads over (x, z) world position mapped
[-120, 15120] -> [0,1] -> 21 bins; cross-entropy summed over both heads, in nats.
Uniform = 2*ln(21) = 6.089.

Reference points printed alongside every horizon:
  marginal   per-bin marginal of the fit games (the blind baseline)
  real       the probe on the REAL latent at that frame (the ceiling)
  copy       the probe on the LAST REAL CONTEXT frame, held (what you already
             know without a world model at all -- the bar a dream must clear)
  shuffle    a random REAL frame from the same game: on-manifold but wrong, so it
             separates "the probe broke on off-distribution input" from "the state
             is genuinely gone"

The headline number is MEDIAN POSITION ERROR IN WORLD UNITS, read off the probe's
per-axis softmax expectation. Cross-entropy is reported too but is secondary: a
linear probe on 8192-d latents is overconfident off-distribution, so its CE can
exceed uniform (6.089 nats) without that meaning anything beyond "confidently
wrong". The distance metric is calibration-free. For scale, Summoner's Rift is
~14700 units across and one 21-way bin is 762 units.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")

BINS = 21
LO, SPAN = -120.0, 15240.0


def to_bin(v):
    return np.clip(np.rint(np.clip((np.asarray(v) - LO) / SPAN, 0, 1) * (BINS - 1)),
                   0, BINS - 1).astype(np.int64)


def ce_nats(logits, y):
    """logits (n, 2*BINS), y (n,2) -> mean CE in nats summed over the two heads."""
    lx, ly = logits[:, :BINS], logits[:, BINS:]
    f = nn.functional.cross_entropy
    return (f(lx, y[:, 0], reduction="none") + f(ly, y[:, 1], reduction="none"))


class Probe(nn.Module):
    def __init__(self, d, hidden=0):
        super().__init__()
        self.net = nn.Linear(d, 2 * BINS) if hidden == 0 else nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, 2 * BINS))

    def forward(self, x):
        return self.net(x)


def load_cw(cache_dir, game):
    cw = np.load(f"{cache_dir}/{game}.npz")["cw"].astype(np.float64)
    for a in range(2):                       # the labels have a few gaps
        c = cw[:, a]
        ok = np.isfinite(c)
        cw[:, a] = np.interp(np.arange(len(c)), np.flatnonzero(ok), c[ok])
    return cw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents-dir", default="/mnt/nfs/datasets/replay_latents_v7_bc")
    ap.add_argument("--cache-dir", default="scratchpad/decision0/labels_cache")
    ap.add_argument("--run", required=True, help="npz written by eval_dream_fidelity.py")
    ap.add_argument("--arms", nargs="+", required=True, help="arms whose .lat_<arm>.npy to score")
    ap.add_argument("--fit-games", type=int, default=40)
    ap.add_argument("--rows-per-game", type=int, default=1200)
    ap.add_argument("--hidden", type=int, default=0, help="0 = linear probe")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--wds", type=float, nargs="+", default=[1.0, 10.0, 100.0],
                    help="weight decays to select between on an inner whole-GAME split")
    ap.add_argument("--cache", default="", help="npz to cache/reuse the fit matrix")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="scratchpad/dreamfid/probe.json")
    args = ap.parse_args()
    dev = args.device
    rng = np.random.default_rng(0)

    run = np.load(args.run, allow_pickle=True)
    eval_games = sorted(set(run["_games"].tolist()))
    print(f"[probe] held-out (excluded from fit): {eval_games}")

    # ---------------- fit set: real latents from TRAIN games ----------------
    all_games = sorted(os.path.basename(p)[:-3] for p in glob.glob(f"{args.latents_dir}/*.pt")
                       if os.path.basename(p) != "index.pt")
    train = [g for g in all_games if g not in eval_games
             and os.path.exists(f"{args.cache_dir}/{g}.npz")]
    rng.shuffle(train)
    train = train[:args.fit_games]
    X, Y, G = [], [], []
    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        X, Y, G = torch.from_numpy(z["X"]), torch.from_numpy(z["Y"]), z["G"]
        train = sorted(set(G.tolist()))
        print(f"[probe] reusing cached fit matrix {tuple(X.shape)} from {args.cache}")
        train_cached = True
    else:
        train_cached = False
    for g in ([] if train_cached else train):
        pack = torch.load(f"{args.latents_dir}/{g}.pt", weights_only=True)
        z, fi = pack["latents"], pack["frame_indices"].numpy()
        cw = load_cw(args.cache_dir, g)
        n = min(z.shape[0], len(cw))
        idx = rng.choice(n, size=min(args.rows_per_game, n), replace=False)
        X.append(z[idx].reshape(len(idx), -1))
        Y.append(to_bin(cw[fi[idx]]))
        G.append(np.array([g] * len(idx)))
        del pack, z
    if not train_cached:
        X = torch.cat(X).float()
        Y = torch.from_numpy(np.concatenate(Y))
        G = np.concatenate(G)
        if args.cache:
            np.savez_compressed(args.cache, X=X.numpy().astype(np.float16),
                                Y=Y.numpy(), G=G)
    X = X.float()
    print(f"[probe] fit rows {tuple(X.shape)} from {len(train)} games")

    mu, sd = X.mean(0), X.std(0).clamp_min(1e-4)
    Xs = ((X - mu) / sd).to(dev)
    Yd = Y.to(dev)
    # marginal baseline from the fit rows (Laplace 0.5), reported in nats
    marg = []
    for a in range(2):
        c = np.bincount(Y[:, a].numpy(), minlength=BINS) + 0.5
        marg.append(np.log(c / c.sum()))
    marg_lp = np.concatenate(marg)
    marg_t = torch.tensor(marg_lp, dtype=torch.float32, device=dev)

    # inner whole-GAME split so weight decay and the stopping epoch are chosen
    # without ever looking at a fit-game row the probe also trained on.
    inner_val = set(train[:max(2, len(train) // 6)])
    vm = torch.from_numpy(np.isin(G, list(inner_val))).to(dev)
    tr_i, va_i = (~vm).nonzero(as_tuple=True)[0], vm.nonzero(as_tuple=True)[0]
    print(f"[probe] inner split: {len(tr_i)} fit rows / {len(va_i)} rows from "
          f"{len(inner_val)} held-back games")

    def fit_one(wd):
        pr = Probe(X.shape[1], args.hidden).to(dev)
        opt = torch.optim.AdamW(pr.parameters(), lr=3e-3, weight_decay=wd)
        nsteps = args.epochs * (len(tr_i) // 1024 + 1)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nsteps)
        best, best_sd = 1e9, None
        for ep in range(args.epochs):
            perm = tr_i[torch.randperm(len(tr_i), device=dev)]
            pr.train()
            for st in range(0, len(perm), 1024):
                b = perm[st:st + 1024]
                loss = ce_nats(pr(Xs[b]), Yd[b]).mean()
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            pr.eval()
            with torch.no_grad():
                v = float(ce_nats(pr(Xs[va_i]), Yd[va_i]).mean())
            if v < best:
                best, best_sd = v, {k: t.clone() for k, t in pr.state_dict().items()}
            print(f"    wd={wd:<6g} ep{ep:<2d} inner-val CE {v:.4f} nats"
                  f"{'  *' if v == best else ''}")
        pr.load_state_dict(best_sd)
        return pr, best

    cands = [fit_one(w) for w in args.wds]
    probe, bestv = min(cands, key=lambda t: t[1])
    wd_star = args.wds[[c[1] for c in cands].index(bestv)]
    print(f"[probe] selected wd={wd_star} (inner-val CE {bestv:.4f} nats)")
    probe.eval()

    centers = torch.arange(BINS, device=dev).float() / (BINS - 1) * SPAN + LO

    def score(zflat, ybin, cw_true=None, bs=4096):
        """-> (CE nats, euclidean world-unit error of the softmax expectation)."""
        ces, errs = [], []
        with torch.no_grad():
            for s0 in range(0, len(zflat), bs):
                x = ((zflat[s0:s0 + bs].float() - mu) / sd).to(dev)
                lg = probe(x)
                yb = ybin[s0:s0 + bs].to(dev)
                ces.append(ce_nats(lg, yb).cpu().numpy())
                ex = torch.stack([(lg[:, :BINS].softmax(-1) * centers).sum(-1),
                                  (lg[:, BINS:].softmax(-1) * centers).sum(-1)], -1)
                tgt = (cw_true[s0:s0 + bs].to(dev) if cw_true is not None
                       else centers[yb])
                errs.append((ex - tgt).pow(2).sum(-1).sqrt().cpu().numpy())
        return np.concatenate(ces), np.concatenate(errs)

    def score_marginal(ybin, cw_true):
        lp = marg_t.unsqueeze(0).expand(len(ybin), -1).to(dev)
        ce = ce_nats(lp, ybin.to(dev)).cpu().numpy()
        cen = torch.arange(BINS, device=dev).float() / (BINS - 1) * SPAN + LO
        ex = torch.stack([(lp[:, :BINS].softmax(-1) * cen).sum(-1),
                          (lp[:, BINS:].softmax(-1) * cen).sum(-1)], -1)
        err = (ex - cw_true.to(dev)).pow(2).sum(-1).sqrt().cpu().numpy()
        return ce, err

    # ---------------- evaluate on the run's rollouts ----------------
    games, starts = run["_games"], run["_starts"]
    lat0 = np.load(args.run.replace(".npz", f".lat_{args.arms[0]}.npy"), mmap_mode="r")
    n, H = lat0.shape[0], lat0.shape[1]
    print(f"[probe] {n} rollouts x H={H}")

    cw_by_game = {g: load_cw(args.cache_dir, g) for g in eval_games}
    zpack = {g: torch.load(f"{args.latents_dir}/{g}.pt", weights_only=True)["latents"]
             for g in eval_games}

    cwt = np.stack([cw_by_game[g][s:s + H] for g, s in zip(games, starts)])   # (n,H,2)
    ytrue = to_bin(cwt)
    yt = torch.from_numpy(ytrue.reshape(n * H, 2))
    cwt_t = torch.from_numpy(cwt.reshape(n * H, 2)).float()
    zreal = torch.stack([zpack[g][s:s + H] for g, s in zip(games, starts)]).reshape(n * H, -1)
    zcopy = torch.stack([zpack[g][s - 1].unsqueeze(0).expand(H, -1, -1, -1)
                         for g, s in zip(games, starts)]).reshape(n * H, -1)
    rs = np.random.default_rng(1)
    zshuf = torch.stack([zpack[g][rs.integers(0, zpack[g].shape[0] - H) + np.arange(H)]
                         for g, s in zip(games, starts)]).reshape(n * H, -1)

    res = {"H": H, "n": n, "games": eval_games, "uniform": float(2 * np.log(BINS)),
           "bin_width_units": SPAN / (BINS - 1)}
    mce, merr = score_marginal(yt, cwt_t)
    res["marginal"] = mce.reshape(n, H).mean(0).tolist()
    res["marginal_err"] = np.median(merr.reshape(n, H), 0).tolist()
    for name, z in [("real", zreal), ("copy", zcopy), ("shuffle", zshuf)]:
        ce, er = score(z, yt, cwt_t)
        res[name] = ce.reshape(n, H).mean(0).tolist()
        res[name + "_err"] = np.median(er.reshape(n, H), 0).tolist()
    for arm in args.arms:
        z = np.load(args.run.replace(".npz", f".lat_{arm}.npy"))
        ce, er = score(torch.from_numpy(z.reshape(n * H, -1)), yt, cwt_t)
        res[arm] = ce.reshape(n, H).mean(0).tolist()
        res[arm + "_err"] = np.median(er.reshape(n, H), 0).tolist()
        pg = er.reshape(n, H)
        res[f"{arm}__per_game_err"] = {g: np.median(
            pg[[i for i, gg in enumerate(games) if gg == g]], 0).tolist() for g in eval_games}

    hs = [h for h in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64] if h <= H]
    keys = ["marginal", "shuffle", "copy", "real"] + list(args.arms)
    print(f"\nMEDIAN champion world-position error, in world units "
          f"(map is ~14700 across; one bin = {SPAN/(BINS-1):.0f} units)")
    print(f"{'source':<14} " + " ".join(f"h={h:<6d}" for h in hs))
    for k in keys:
        print(f"{k:<14} " + " ".join(f"{res[k+'_err'][h-1]:<8.0f}" for h in hs))
    print(f"\ncross-entropy, nats (secondary -- the probe is overconfident "
          f"off-distribution; uniform = {2*np.log(BINS):.3f})")
    print(f"{'source':<14} " + " ".join(f"h={h:<6d}" for h in hs))
    for k in keys:
        print(f"{k:<14} " + " ".join(f"{res[k][h-1]:<8.2f}" for h in hs))
    import json
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
