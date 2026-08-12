#!/usr/bin/env python3
"""Driver: run one probe (A / Acrop / cheat / B / C / D / S) and dump results JSON.

Every probe shares the SAME event rows, the SAME whole-game splits and the SAME
train/eval/control code, so the AUCs are directly comparable.

Controls run for every probe:
  * SHUFFLE — retrain on permuted TRAIN labels, score real held-out labels -> ~0.5
  * base rates + n for train / inner-val / each held-out split
  * cluster (per-GAME) bootstrap 95% CIs
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "src")
from lasthit_events import splits  # noqa: E402
from lh_probe import (GridCNN, ImgCNN, MLP, cat_rows, feat_agent, feat_latents,  # noqa: E402
                      feat_latents_ok, feat_state, fit, load_rows, predict, report)

CROP = 128


# ───────────────────────────── data assembly ─────────────────────────────

def frame_cache(path):
    meta = np.load(os.path.join(path, "meta.npz"), allow_pickle=True)
    offs = meta["offsets"].tolist()
    arrs = {o: np.memmap(os.path.join(path, f"frames_off{o}.u8"), dtype=np.uint8,
                         mode="r", shape=(len(meta["mid"]), 352, 352, 3)) for o in offs}
    return meta, arrs


def img_getter(arr, rows_idx, csx, csy, crop, train_aug, rng):
    """Returns f(batch_idx, training) -> float tensor (B,3,H,W)."""
    def get(b, training):
        gi = rows_idx[b]
        x = np.asarray(arr[gi])                      # (B,352,352,3) uint8
        if crop:
            out = np.zeros((len(b), crop, crop, 3), np.uint8)
            for k, j in enumerate(b):
                cx = int(round(csx[j] * 352 / 1280))
                cy = int(round(csy[j] * 352 / 720))
                if training and train_aug:
                    cx += rng.integers(-6, 7)
                    cy += rng.integers(-6, 7)
                x0 = int(np.clip(cx - crop // 2, 0, 352 - crop))
                y0 = int(np.clip(cy - crop // 2, 0, 352 - crop))
                out[k] = x[k, y0:y0 + crop, x0:x0 + crop]
            x = out
        t = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().div_(255.)
        if training and train_aug:
            t = t * (0.9 + 0.2 * torch.rand(len(b), 1, 1, 1))
        return t
    return get


def vec_getter(X):
    Xt = torch.from_numpy(np.asarray(X)).float()

    def get(b, training):
        return Xt[b]
    return get


def lat_getter(X, mu, sd, aug=False):
    def get(b, training):
        x = torch.from_numpy(np.ascontiguousarray(X[b])).float()
        x = (x - mu) / sd
        if training and aug:
            x = x + 0.05 * torch.randn_like(x)
        return x
    return get


# ───────────────────────────── main ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True,
                    choices=["A", "Acrop", "cheat", "B", "C", "D", "S"])
    ap.add_argument("--anchor", default="commit")
    ap.add_argument("--n-train-games", type=int, default=60)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase2-ckpt", default="data/phase2_bc_garen/agent_finetune_latest.pt")
    ap.add_argument("--frames", default="scratchpad/lh_frames_commit")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-shuffle-control", action="store_true")
    ap.add_argument("--shuffle-epochs", type=int, default=0)
    ap.add_argument("--cap-per-game", type=int, default=0,
                    help="subsample events per game (keeps the w=16 latent cache "
                         "small enough for 23 GB of RAM)")
    ap.add_argument("--state-subset", default="",
                    help="probe S only: comma-separated substrings of feature names")
    ap.add_argument("--agent-cap", type=int, default=70,
                    help="probe D only: max events per game pushed through the dynamics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default="y", choices=["y", "level", "hp"],
                    help="'level'/'hp' are POSITIVE CONTROLS: quantities that really "
                         "are on screen (level badge / champion HP bar). If the same "
                         "CNN reads those but not the last-hit label, 'the CNN cannot "
                         "learn' is not the explanation.")
    args = ap.parse_args()

    sp = splits()
    tr_games = sp["train"][: args.n_train_games]
    # inner validation = whole games held out of TRAIN (never frames)
    n_val = max(6, len(tr_games) // 6)
    val_games, tr_games = tr_games[:n_val], tr_games[n_val:]
    ho_seen, ho_unseen = sp["heldout_seen"], sp["heldout_unseen"]
    print(f"probe={args.probe} anchor={args.anchor} | train={len(tr_games)} "
          f"val={len(val_games)} ho_seen={len(ho_seen)} ho_unseen={len(ho_unseen)} games")
    print(f"  ho_unseen = {ho_unseen}")

    is_img = args.probe in ("A", "Acrop", "cheat")
    if is_img:
        path = args.frames if args.probe != "cheat" else "scratchpad/lh_frames_cheat"
        meta, arrs = frame_cache(path)
        off = 8 if args.probe == "cheat" else 0
        arr = arrs[off]
        mid, frame, y = meta["mid"], meta["frame"], meta["y"]
        csx, csy = meta["csx"], meta["csy"]
        # drop rows whose PNG was missing (all-zero)
        gp = os.path.join(path, f"good_off{off}.npy")
        if os.path.exists(gp):
            good = np.load(gp)
        else:
            good = np.zeros(len(mid), bool)
            for i0 in range(0, len(mid), 512):
                sl = np.asarray(arr[i0:i0 + 512, ::16, ::16, :])
                good[i0:i0 + 512] = sl.reshape(len(sl), -1).max(1) > 0
            np.save(gp, good)
        rows = dict(mid=mid, frame=frame, y=y, csx=csx, csy=csy,
                    level=meta["level"], hp=meta["hp"])
        print(f"  frame cache {path} off{off}: {len(mid)} rows, {int((~good).sum())} dropped")
    else:
        rows = cat_rows([load_rows(args.anchor, tr_games), load_rows(args.anchor, val_games),
                         load_rows(args.anchor, ho_seen), load_rows(args.anchor, ho_unseen)])
        good = np.ones(len(rows["mid"]), bool)
        if args.cap_per_game:
            rg = np.random.default_rng(0)
            keep = np.zeros(len(rows["mid"]), bool)
            for g in sorted(set(rows["mid"].tolist())):
                gi = np.where(rows["mid"] == g)[0]
                keep[gi if len(gi) <= args.cap_per_game
                     else rg.choice(gi, args.cap_per_game, replace=False)] = True
            rows = {k: v[keep] for k, v in rows.items()}
            good = np.ones(len(rows["mid"]), bool)

    if args.target == "level":
        rows["y"] = (np.nan_to_num(rows["level"], nan=1.0) >= 7).astype(np.float32)
    elif args.target == "hp":
        rows["y"] = (np.nan_to_num(rows["hp"], nan=1.0) < 0.85).astype(np.float32)

    grp = {"train": tr_games, "val": val_games, "ho_seen": ho_seen, "ho_unseen": ho_unseen}
    where = {k: np.where(good & np.isin(rows["mid"], v))[0] for k, v in grp.items()}
    for k, v in where.items():
        print(f"  {k:9s} n={len(v):6d} pos={rows['y'][v].mean() if len(v) else float('nan'):.3f} "
              f"games={len(set(rows['mid'][v].tolist()))}")

    # ── features ──
    model_kind = args.model
    if is_img:
        rng = np.random.default_rng(args.seed)
        crop = CROP if args.probe == "Acrop" else 0
        getter = img_getter(arr, np.arange(len(rows["mid"])), rows["csx"], rows["csy"],
                            crop, True, rng)
        make = lambda: ImgCNN(in_ch=3, width=1.0, drop=0.2)  # noqa: E731
        lr, wd, bs, ep = args.lr or 3e-4, args.wd or 0.05, args.bs, args.epochs
    elif args.probe in ("B", "C"):
        w = 1 if args.probe == "B" else args.window
        ltag = f"{args.anchor}_g{args.n_train_games}" + (
            f"_c{args.cap_per_game}" if args.cap_per_game else "")
        X = feat_latents(rows, sp, w, ltag)
        ok = feat_latents_ok(ltag, w)
        for k in where:
            where[k] = where[k][ok[where[k]]]
        tr = where["train"]
        sub = tr[np.random.default_rng(0).permutation(len(tr))[:4000]]
        s = np.asarray(X[sub]).astype(np.float32)
        mu = torch.tensor(s.mean((0, 2, 3), keepdims=True).squeeze(0)[None])
        sd = torch.tensor(s.std((0, 2, 3), keepdims=True).squeeze(0)[None] + 1e-4)
        del s
        getter = lat_getter(X, mu, sd, aug=True)
        model_kind = model_kind or "gridcnn"
        if model_kind == "gridcnn":
            make = lambda: GridCNN(in_ch=w * 32)  # noqa: E731
        else:
            class Flat(torch.nn.Module):
                def __init__(self, h):
                    super().__init__()
                    self.m = MLP(w * 32 * 256, h)

                def forward(self, x):
                    return self.m(x.flatten(1))
            h = 0 if model_kind == "linear" else 256
            make = lambda: Flat(h)  # noqa: E731
        lr, wd, bs, ep = args.lr or 1e-3, args.wd or 0.05, args.bs, args.epochs
        print(f"  latents X={tuple(X.shape)} model={model_kind}")
    elif args.probe == "D":
        sel = np.zeros(len(rows["mid"]), bool)          # cap per game: the frozen
        rg = np.random.default_rng(0)                   # dynamics forward is the cost
        for g in sorted(set(rows["mid"].tolist())):
            gi = np.where(rows["mid"] == g)[0]
            cap = args.agent_cap if g not in ho_unseen else 10 ** 9
            sel[gi if len(gi) <= cap else rg.choice(gi, cap, replace=False)] = True
        Xa, ok = feat_agent(rows, sp, f"{args.anchor}_g{args.n_train_games}" + (
            f"_c{args.cap_per_game}" if args.cap_per_game else ""),
            args.phase2_ckpt, args.window, args.device, sel=sel)
        for k in where:
            where[k] = where[k][ok[where[k]]]
        tr = where["train"]
        mu = Xa[tr].mean(0, keepdims=True)
        sd = Xa[tr].std(0, keepdims=True) + 1e-5
        getter = vec_getter((Xa - mu) / sd)
        model_kind = model_kind or "mlp"
        make = lambda: MLP(Xa.shape[1], 0 if model_kind == "linear" else 256)  # noqa: E731
        lr, wd, bs, ep = args.lr or 1e-3, args.wd or 0.1, 256, args.epochs
        print(f"  agent tokens X={Xa.shape} model={model_kind}")
    else:  # S
        Xs, snames = feat_state(rows)
        if args.state_subset:
            keep = [i for i, n in enumerate(snames)
                    if any(k in n for k in args.state_subset.split(","))]
            Xs, snames = Xs[:, keep], [snames[i] for i in keep]
        print(f"  state features: {snames}")
        tr = where["train"]
        mu = Xs[tr].mean(0, keepdims=True)
        sd = Xs[tr].std(0, keepdims=True) + 1e-5
        getter = vec_getter((Xs - mu) / sd)
        model_kind = model_kind or "mlp"
        make = lambda: MLP(Xs.shape[1], 0 if model_kind == "linear" else 64)  # noqa: E731
        lr, wd, bs, ep = args.lr or 1e-2, args.wd or 0.01, 256, args.epochs

    # ── remap to contiguous per-split getters ──
    def split_getter(idx):
        def g(b, training):
            return getter(idx[b], training)
        return g

    tr_idx = where["train"]
    y_tr = rows["y"][tr_idx].astype(np.float32)
    evals = {k: (split_getter(where[k]), len(where[k]), rows["y"][where[k]])
             for k in ["val", "ho_seen", "ho_unseen"] if len(where[k]) > 20}

    def run(y_train, seed, tag, n_ep=None, shuffle_eval=False):
        # The inner-val epoch selection reads REAL val labels. Under the shuffle
        # control that would leak: with a small model class, picking the epoch whose
        # scores best match the real val labels can reach AUC 0.66 having learned
        # nothing (measured). So the null run selects on SHUFFLED val labels too, and
        # only the final report scores against real labels.
        ev_use = evals
        if shuffle_eval:
            rgs = np.random.default_rng(777)
            ev_use = {k: (v[0], v[1], rgs.permutation(v[2])) for k, v in evals.items()}
        model, hist = fit(make(), split_getter(tr_idx), len(tr_idx), y_train,
                          ev_use, n_ep or ep, lr, wd, bs, args.device, seed=seed)
        res, dump = {}, {}
        for k in ["train", "val", "ho_seen", "ho_unseen"]:
            if len(where[k]) < 20:
                continue
            s = predict(model, split_getter(where[k]), len(where[k]), args.device)
            res[k] = report(f"{tag}/{k}", s, rows["y"][where[k]], rows["mid"][where[k]])
            dump[k] = (s, where[k])
        res["hist"] = hist
        return res, dump

    out = {"probe": args.probe, "anchor": args.anchor, "model": model_kind,
           "n_train_games": args.n_train_games, "window": args.window,
           "epochs": ep, "lr": lr, "wd": wd, "bs": bs,
           "splits": {k: {"n": int(len(v)), "games": len(set(rows["mid"][v].tolist())),
                          "pos": float(rows["y"][v].mean()) if len(v) else None}
                      for k, v in where.items()},
           "ho_unseen_games": ho_unseen}
    print("\n== REAL LABELS ==")
    out["real"], dump = run(y_tr, args.seed, args.probe)
    if not args.no_shuffle_control:
        print("\n== SHUFFLED TRAIN LABELS (control: held-out AUC must be ~0.5) ==")
        y_sh = np.random.default_rng(123).permutation(y_tr)
        out["shuffled"], _ = run(y_sh, args.seed + 1, args.probe + "-shuf",
                                 args.shuffle_epochs or ep, shuffle_eval=True)

    out["target"] = args.target
    tagp = f"{args.probe}_{args.anchor}_{model_kind}"
    if args.state_subset:
        tagp += "_" + args.state_subset.replace(",", "-")
    if args.target != "y":
        tagp += f"_{args.target}"
    np.savez_compressed(
        f"scratchpad/lh_scores_{tagp}.npz",
        **{f"{k}_s": v[0] for k, v in dump.items()},
        **{f"{k}_mid": rows["mid"][v[1]] for k, v in dump.items()},
        **{f"{k}_frame": rows["frame"][v[1]] for k, v in dump.items()},
        **{f"{k}_y": rows["y"][v[1]] for k, v in dump.items()})
    p = args.out or f"scratchpad/lh_res_{tagp}.json"
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()
