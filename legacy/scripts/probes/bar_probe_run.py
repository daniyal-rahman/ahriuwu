#!/usr/bin/env python3
"""Does minion-HP information survive the v7 tokenizer? (no ground truth needed)

PAIRED counterfactual design: for each frame we encode the ORIGINAL and an
edited copy whose only difference is k pixels of HP-bar fill. Because the
tokenizer is deterministic, dz = z_edit - z_orig IS the encoder's response to
that edit. We compare it against a CONTROL edit of identical pixel count and
colour placed on nearby terrain.

Three questions, three stages (pixels -> v7 latents -> agent tokens):
  1. magnitude : is ||dz|| for a bar edit smaller than for an equal-pixel
                 terrain edit?  (encoder specifically discarding bars)
  2. decodable : can a linear probe trained on TRAIN games tell a bar edit from
                 a control edit on HELD-OUT games?  (information survives)
  3. dose      : does ||dz|| grow with k=1,2,4?  (graded response, not a fluke)

Results stratified by bar size: minion-scale (<=12px wide) vs larger/champion.
"""
import argparse
import glob
import json
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "scratchpad")
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from bar_edit import find_minion_bars, pick_bar, edit_bar, control_edit

ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
TOK = "rollout_stage/transformer_tokenizer_latest.pt"
TRAIN_GAMES = ["NA1_5549995114", "NA1_5550013959", "NA1_5550028932", "NA1_5550045094",
               "NA1_5550067582", "NA1_5550073400"]
HELD_GAMES = ["NA1_5549981347", "NA1_5550450386", "NA1_5551132630"]
KS = (1, 2, 4)


def collect(games, per_game, seed=0):
    """Frames with an editable bar -> (rgb, bar) samples."""
    rng = np.random.RandomState(seed)
    out = []
    for g in games:
        fs = sorted(glob.glob(f"{ROOT}/{g}/frames/*.png"))
        if not fs:
            continue
        idx = rng.choice(np.arange(2000, len(fs)), size=min(per_game * 6, len(fs) - 2000),
                         replace=False)
        got = 0
        for i in sorted(idx):
            im = cv2.imread(fs[int(i)])
            if im is None:
                continue
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            b = pick_bar(find_minion_bars(rgb), rgb.shape)
            if b is None or b["w"] < 6:
                continue
            out.append((g, int(i), rgb, b))
            got += 1
            if got >= per_game:
                break
        print(f"  {g}: {got} usable frames", flush=True)
    return out


@torch.no_grad()
def encode_batch(tok, imgs, dev, bs=8):
    zs = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[i:i + bs])).float().div_(255)
        x = x.permute(0, 3, 1, 2).to(dev)
        lat = tok.encode(x)["latent"]                      # (B,512,16)
        zs.append(lat.reshape(lat.shape[0], -1).float().cpu().numpy())
    return np.concatenate(zs)


def probe_auc(Xtr, ytr, Xte, yte, seed=0):
    """Linear probe (logistic, closed-formish via torch) -> held-out AUC."""
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    mu, sd = Xtr_t.mean(0), Xtr_t.std(0) + 1e-6
    Xtr_t = (Xtr_t - mu) / sd
    Xte_t = (torch.tensor(Xte, dtype=torch.float32) - mu) / sd
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    w = torch.zeros(Xtr_t.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-3, weight_decay=1e-3)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xtr_t @ w + b, ytr_t)
        loss.backward()
        opt.step()
    s = (Xte_t @ w + b).detach().numpy()
    pos, neg = s[yte == 1], s[yte == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = np.concatenate([pos, neg]).argsort().argsort().astype(float)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-game", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    from pretokenize_replay_v7 import load_v7
    tok, cfg, step = load_v7(TOK, dev)
    tok = tok.float()
    print(f"v7 loaded (step {step})", flush=True)

    print("collecting TRAIN frames...", flush=True)
    tr = collect(TRAIN_GAMES, args.per_game, seed=0)
    print("collecting HELD-OUT frames...", flush=True)
    te = collect(HELD_GAMES, args.per_game, seed=1)
    print(f"train={len(tr)} held-out={len(te)}", flush=True)

    rows = []
    for split, data in (("train", tr), ("held", te)):
        for gi, (g, fi, rgb, bar) in enumerate(data):
            imgs = [rgb]
            tags = [("orig", 0)]
            for k in KS:
                imgs.append(edit_bar(rgb, bar, k)); tags.append(("bar", k))
            imgs.append(control_edit(rgb, bar, 2)); tags.append(("ctrl", 2))
            Z = encode_batch(tok, imgs, dev)
            z0 = Z[0]
            for (kind, k), z in zip(tags[1:], Z[1:]):
                dz = z - z0
                # pixel-space delta for the same edit, for the stage-1 baseline
                px = np.abs(imgs[tags.index((kind, k))].astype(np.float32) - rgb.astype(np.float32)).sum()
                rows.append({"split": split, "game": g, "frame": fi, "kind": kind, "k": k,
                             "bar_w": bar["w"], "bar_h": bar["h"],
                             "dz_norm": float(np.linalg.norm(dz)),
                             "dz_rel": float(np.linalg.norm(dz) / (np.linalg.norm(z0) + 1e-9)),
                             "px_delta": float(px),
                             "dz": dz.astype(np.float32)})
            if gi % 50 == 0:
                print(f"  [{split}] {gi}/{len(data)}", flush=True)

    # ---------------- analysis ----------------
    def sel(split, kind, k=None, wmax=None):
        return [r for r in rows if r["split"] == split and r["kind"] == kind
                and (k is None or r["k"] == k) and (wmax is None or r["bar_w"] <= wmax)]

    print("\n=== 1. MAGNITUDE: latent response per pixel changed ===")
    print(f"{'split':6s} {'edit':6s} {'k':>2s} {'n':>5s} {'|dz|':>9s} {'|dz|/px':>9s} {'px':>8s}")
    for split in ("train", "held"):
        for kind, k in [("bar", 1), ("bar", 2), ("bar", 4), ("ctrl", 2)]:
            s = sel(split, kind, k)
            if not s:
                continue
            dz = np.array([r["dz_norm"] for r in s])
            px = np.array([r["px_delta"] for r in s])
            print(f"{split:6s} {kind:6s} {k:>2d} {len(s):>5d} {dz.mean():>9.3f} "
                  f"{(dz/np.maximum(px,1)).mean()*1e3:>9.4f} {px.mean():>8.0f}")

    print("\n=== 2. DECODABILITY: bar-edit vs control-edit, linear probe on dz ===")
    for wmax, label in [(12, "minion-scale (<=12px wide)"), (None, "all bars")]:
        tb = sel("train", "bar", 2, wmax); tc = sel("train", "ctrl", 2, wmax)
        hb = sel("held", "bar", 2, wmax);  hc = sel("held", "ctrl", 2, wmax)
        if min(len(tb), len(tc), len(hb), len(hc)) < 15:
            print(f"  {label}: too few samples ({len(tb)}/{len(tc)}/{len(hb)}/{len(hc)}) — skipped")
            continue
        Xtr = np.stack([r["dz"] for r in tb + tc]); ytr = np.array([1]*len(tb) + [0]*len(tc))
        Xte = np.stack([r["dz"] for r in hb + hc]); yte = np.array([1]*len(hb) + [0]*len(hc))
        aucs = [probe_auc(Xtr, ytr, Xte, yte, seed=s) for s in (0, 1, 2)]
        # shuffled-label control
        rs = np.random.RandomState(0); ysh = ytr.copy(); rs.shuffle(ysh)
        auc_sh = probe_auc(Xtr, ysh, Xte, yte, seed=0)
        print(f"  {label}: n_tr={len(ytr)} n_te={len(yte)}  "
              f"AUC={np.mean(aucs):.3f} (seeds {', '.join(f'{a:.3f}' for a in aucs)})  "
              f"shuffled={auc_sh:.3f}")

    print("\n=== 3. DOSE-RESPONSE (held-out, minion-scale) ===")
    for k in KS:
        s = sel("held", "bar", k, 12)
        if s:
            dz = np.array([r["dz_norm"] for r in s])
            print(f"  k={k}px: n={len(s):4d}  |dz|={dz.mean():.3f} +- {dz.std():.3f}")
    c = sel("held", "ctrl", 2, 12)
    if c:
        dz = np.array([r["dz_norm"] for r in c])
        print(f"  control 2px: n={len(c):4d}  |dz|={dz.mean():.3f} +- {dz.std():.3f}")

    out = {"n_rows": len(rows),
           "bar_sizes": sorted({(r["bar_w"], r["bar_h"]) for r in rows})[:40]}
    json.dump(out, open("scratchpad/bar_probe_meta.json", "w"), indent=2)
    np.savez_compressed("scratchpad/bar_probe_rows.npz",
                        **{k: np.array([r[k] for r in rows]) for k in
                           ("split", "game", "kind", "k", "bar_w", "bar_h", "dz_norm", "dz_rel", "px_delta")})
    print("\nsaved scratchpad/bar_probe_rows.npz")


if __name__ == "__main__":
    main()
