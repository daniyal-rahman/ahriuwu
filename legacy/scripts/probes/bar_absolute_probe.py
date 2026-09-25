#!/usr/bin/env python3
"""Can the ABSOLUTE bar fill be read from a single latent (no reference frame)?

The paired test showed z RESPONDS to bar edits (graded, above-control). But a
live policy sees one latent with nothing to difference against. So: generate
frames whose bar is shortened by a known amount s in {0..6} px, encode each
INDEPENDENTLY, and ask a probe to recover s from z alone, on HELD-OUT games.

  readable  -> vision is fine; the last-hit failure is downstream
  chance    -> information is present but not usable in this form (entangled /
               below the SNR a downstream head can exploit)

Control: the same probe on the CONTROL edit (terrain), which carries no HP
meaning — it should be less recoverable if we are truly reading the bar and not
just "how many dark pixels are somewhere in this frame".
"""
import glob
import sys

import cv2
import os as _os
_NFS = "/mnt/nfs" if _os.path.isdir("/mnt/nfs/datasets") else "/srv/nfs"
import numpy as np
import torch

sys.path.insert(0, "scratchpad")
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from bar_edit import find_minion_bars, pick_bar, edit_bar, control_edit

ROOT = f"{_NFS}/datasets/lol_replays_16_9_772"
TOK = "rollout_stage/transformer_tokenizer_latest.pt"
import json as _json
_V = _json.load(open("scratchpad/valid_games.json"))["both"]
TRAIN_GAMES = _V[:8]
HELD_GAMES = _V[100:103]
SHIFTS = (0, 1, 2, 3, 4, 5, 6)
PER_GAME = 140


def collect(games, per_game, seed):
    rng = np.random.RandomState(seed)
    out = []
    for g in games:
        fs = sorted(glob.glob(f"{ROOT}/{g}/frames/*.png"))
        idx = rng.choice(np.arange(2000, len(fs)), size=min(per_game * 8, len(fs) - 2000),
                         replace=False)
        got = 0
        for i in sorted(idx):
            im = cv2.imread(fs[int(i)])
            if im is None:
                continue
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            b = pick_bar(find_minion_bars(rgb), rgb.shape)
            if b is None or b["w"] < 8 or b["w"] > 12:      # MINION-SCALE ONLY
                continue
            out.append((g, rgb, b))
            got += 1
            if got >= per_game:
                break
        print(f"  {g}: {got}", flush=True)
    return out


@torch.no_grad()
def enc(tok, imgs, dev, bs=8):
    zs = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[i:i + bs])).float().div_(255).permute(0, 3, 1, 2).to(dev)
        z = tok.encode(x)["latent"]
        zs.append(z.reshape(z.shape[0], -1).float().cpu().numpy())
    return np.concatenate(zs)


def ridge_r2(Xtr, ytr, Xte, yte, lam=1e3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    B = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32) - float(np.mean(ytr))
    n, d = A.shape
    if d > n:                                  # dual form
        K = A @ A.T
        al = torch.linalg.solve(K + lam * torch.eye(n), yt)
        pred = (B @ A.T) @ al + float(np.mean(ytr))
    else:
        W = torch.linalg.solve(A.T @ A + lam * torch.eye(d), A.T @ yt)
        pred = B @ W + float(np.mean(ytr))
    pred = pred.numpy()
    ss = ((yte - pred) ** 2).sum() / max(((yte - yte.mean()) ** 2).sum(), 1e-9)
    return float(1 - ss), float(np.corrcoef(pred, yte)[0, 1])


def mlp_r2(Xtr, ytr, Xte, yte, hidden=512, epochs=60, seed=0):
    """Nonlinear readout. Guards against a linear-only false negative."""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    B = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    ym, ys = float(np.mean(ytr)), float(np.std(ytr) + 1e-6)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], hidden), torch.nn.ReLU(),
                              torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    n, bs = A.shape[0], 256
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(A[idx]).squeeze(-1), yt[idx])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(B).squeeze(-1).numpy() * ys + ym
    ss = ((yte - pred) ** 2).sum() / max(((yte - yte.mean()) ** 2).sum(), 1e-9)
    return float(1 - ss), float(np.corrcoef(pred, yte)[0, 1])


def main():
    dev = "cuda"
    from pretokenize_replay_v7 import load_v7
    tok, _, step = load_v7(TOK, dev)
    tok = tok.float()
    print(f"v7 step {step}; MINION-SCALE bars only (8-12px wide)", flush=True)
    print("train frames:", flush=True); tr = collect(TRAIN_GAMES, PER_GAME, 0)
    print("held frames:", flush=True);  te = collect(HELD_GAMES, PER_GAME, 1)
    print(f"train={len(tr)} held={len(te)}", flush=True)

    def build(data, kind):
        """Label = ABSOLUTE fill width in px after the edit (w - s), NOT the shift.

        The shift alone is not a property of the image: a naturally short bar at
        s=0 is pixel-indistinguishable from a long bar at s=4, so no encoder
        could recover it from one frame. Fill WIDTH is observable, and is the
        quantity a policy would actually need.
        """
        Z, Y = [], []
        for j, (g, rgb, b) in enumerate(data):
            imgs, ys = [], []
            for s in SHIFTS:
                if s > b["w"] - 2:
                    continue
                img = rgb if s == 0 else (edit_bar(rgb, b, s) if kind == "bar"
                                          else control_edit(rgb, b, s))
                imgs.append(img); ys.append(b["w"] - s)
            if imgs:
                Z.append(enc(tok, imgs, dev)); Y.append(np.array(ys, np.float32))
            if j % 60 == 0:
                print(f"    {kind} {j}/{len(data)}", flush=True)
        return np.concatenate(Z), np.concatenate(Y)

    print("encoding BAR variants...", flush=True)
    Ztr, Ytr = build(tr, "bar"); Zte, Yte = build(te, "bar")
    print("encoding CONTROL variants...", flush=True)
    Ctr, CYtr = build(tr, "ctrl"); Cte, CYte = build(te, "ctrl")

    print("\n=== ABSOLUTE READOUT from a single latent (held-out games) ===")
    print("  label = fill WIDTH in px after edit; CONTROL = same edit on terrain")
    for lam in (1e2, 1e3, 1e4):
        r2, r = ridge_r2(Ztr, Ytr, Zte, Yte, lam)
        c2, cr = ridge_r2(Ctr, CYtr, Cte, CYte, lam)
        print(f"  ridge lam={lam:>6.0f}  BAR: R2={r2:+.3f} corr={r:+.3f}   "
              f"CONTROL: R2={c2:+.3f} corr={cr:+.3f}")
    # MLP — a linear null could be a false negative if the code is entangled
    m_r2, m_r = mlp_r2(Ztr, Ytr, Zte, Yte)
    mc_r2, mc_r = mlp_r2(Ctr, CYtr, Cte, CYte)
    print(f"  MLP          BAR: R2={m_r2:+.3f} corr={m_r:+.3f}   "
          f"CONTROL: R2={mc_r2:+.3f} corr={mc_r:+.3f}")
    # PIXEL baseline: same label from the raw frame => proves the label is learnable
    print("  (pixel-space baseline is by construction ~1.0: the label IS the pixel count)")
    # shuffled null
    rs = np.random.RandomState(0); Ysh = Ytr.copy(); rs.shuffle(Ysh)
    r2s, rsh = ridge_r2(Ztr, Ysh, Zte, Yte, 1e3)
    print(f"  shuffled-label null: R2={r2s:+.3f} corr={rsh:+.3f}  (must be ~0)")
    print(f"\n  n_train={len(Ytr)} n_held={len(Yte)}  shifts={SHIFTS}")
    print("  read: corr>~0.5 on BAR and clearly above CONTROL => fill is READABLE from one latent")


if __name__ == "__main__":
    main()
