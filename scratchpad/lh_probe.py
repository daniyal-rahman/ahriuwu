#!/usr/bin/env python3
"""Four probes on ONE last-hit event set — perception vs dynamics vs underdetermined.

  A      small CNN on the raw 352x352 frame at the commit frame  (raw-pixel ceiling)
  Acrop  same CNN on a 128x128 native-resolution crop centred on champion_screen
  B      v7 latent at the commit frame                            (single-frame)
  C      v7 latents over the preceding 16 frames                  (temporal window)
  D      agent token of the frozen Phase-2 dynamics, 16-frame window ending at t
  S      NON-VISUAL state oracle (champ level / hp / game time) — tests whether the
         label is determined by things that are NOT on a HUD-off screen
  cheat  Probe-A CNN on the frame at t+8, i.e. AFTER the gold landed. MUST score
         high; if it does not, the labels/alignment are broken and nothing else means
         anything.

Splits are by WHOLE GAME. `heldout_unseen` games were never seen by the v7
tokenizer nor by the Phase-2 dynamics (which was reward-supervised on gold over
every BC game, so `heldout_seen` is NOT a valid held-out set for probe D).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "src")
from lasthit_events import (auc_scores, boot_ci_cluster, latent_dir_for,  # noqa: E402
                            splits)

EVENTS = "scratchpad/lh_events.npz"
CACHE = "scratchpad/lh_cache"
os.makedirs(CACHE, exist_ok=True)


# ────────────────────────────── rows / splits ──────────────────────────────

ATK = "scratchpad/lh_atk.npz"


def anchor_mask(ev, anchor, atk_path=ATK):
    """Rows for `anchor`. For the gold anchoring both classes are forced to
    attack-STATE frames: its negatives are attack-state by construction, but only
    ~26% of its positives are, so the unfiltered set mostly measures "is he
    swinging". Filtering keeps the chained autos the commit anchoring misses."""
    m = ev["anchor"] == anchor
    if anchor == "gold" and os.path.exists(atk_path):
        m = m & np.load(atk_path)["atk"]
    return m


def load_rows(anchor, games, events=EVENTS):
    ev = np.load(events, allow_pickle=True)
    m = anchor_mask(ev, anchor) & np.isin(ev["mid"], list(games))
    return dict(mid=ev["mid"][m], frame=ev["frame"][m], y=ev["y"][m],
                csx=ev["csx"][m], csy=ev["csy"][m],
                level=ev["level"][m], hp=ev["hp"][m])


def cat_rows(rs):
    return {k: np.concatenate([r[k] for r in rs]) for k in rs[0]}


# ────────────────────────────── features ──────────────────────────────

def feat_latents(rows, sp, window, tag):
    """(N, window*32, 16, 16) float16 — frames [t-window+1 .. t], newest LAST."""
    path = f"{CACHE}/lat_{tag}_w{window}.npy"
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    N = len(rows["mid"])
    out = np.zeros((N, window * 32, 16, 16), dtype=np.float16)
    ok = np.zeros(N, bool)
    for mid in sorted(set(rows["mid"].tolist())):
        ldir = latent_dir_for(mid, sp)
        if ldir is None:                       # never tokenized -> no B/C/D row
            continue
        d = torch.load(os.path.join(ldir, f"{mid}.pt"), weights_only=True)
        lat = d["latents"].numpy()                     # (M,32,16,16) fp16
        fi = d["frame_indices"].numpy()
        pos = {int(f): i for i, f in enumerate(fi)}
        ii = np.where(rows["mid"] == mid)[0]
        for i in ii:
            t = int(rows["frame"][i])
            js = [pos.get(t - window + 1 + k) for k in range(window)]
            if any(j is None for j in js):
                continue
            out[i] = lat[np.array(js)].reshape(window * 32, 16, 16)
            ok[i] = True
        del d, lat
        print(f"    lat {mid}: {ok[ii].sum()}/{len(ii)}", flush=True)
    np.save(path, out)
    np.save(path.replace(".npy", "_ok.npy"), ok)
    return np.load(path, mmap_mode="r")


def feat_latents_ok(tag, window):
    return np.load(f"{CACHE}/lat_{tag}_w{window}_ok.npy")


def _load_phase2(path, device):
    from ahriuwu.models import RewardHead, create_dynamics  # noqa: F401
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a, cfg = ck.get("args", {}), ck.get("dynamics_config") or {}
    sd = ck["dynamics_state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    use_actions = cfg.get("use_actions", any("action_embed." in k for k in sd))
    dyn = create_dynamics(
        size=a.get("model_size", "medium"), latent_dim=cfg.get("latent_dim", 32),
        use_agent_tokens=True, use_actions=use_actions, num_tasks=1,
        agent_layers=a.get("agent_layers", 4), use_qk_norm=not a.get("no_qk_norm", False),
        soft_cap=a.get("soft_cap", 50.0) or None,
        num_register_tokens=a.get("num_register_tokens", 8),
        num_kv_heads=a.get("num_kv_heads", None)).to(device)
    miss, unexp = dyn.load_state_dict(sd, strict=False)
    miss = [m for m in miss if "rope" not in m.lower()]
    assert len(miss) + len(unexp) <= 10, f"ARCH MISMATCH {miss[:5]} {unexp[:5]}"
    dyn.eval().requires_grad_(False)
    print(f"[phase2] gs={ck.get('global_step')} epoch={ck.get('epoch')} "
          f"use_actions={use_actions} train_seq_len={a.get('seq_len')} "
          f"model_dim={dyn.model_dim}", flush=True)
    return dyn


def feat_agent(rows, sp, tag, ckpt, window=16, device="cuda", batch=4, sel=None):
    """Agent token at the last position of a `window`-frame context ending at t.

    `sel` subsamples which rows are actually pushed through the frozen dynamics —
    a 16x256-token forward is ~0.4 s on the 1060, so the full event set would cost
    hours for no extra statistical power."""
    path = f"{CACHE}/agent_{tag}_w{window}.npy"
    if os.path.exists(path):
        return np.load(path), np.load(path.replace(".npy", "_ok.npy"))
    dyn = _load_phase2(ckpt, device)
    lat_np = feat_latents(rows, sp, window, tag)
    ok = feat_latents_ok(tag, window).copy()
    if sel is not None:
        ok &= sel
    N = len(ok)
    out = np.zeros((N, dyn.model_dim), dtype=np.float32)
    idx = np.where(ok)[0]
    print(f"    agent: {len(idx)} rows to encode", flush=True)
    d_one = torch.ones(1, dtype=torch.long, device=device)
    t0 = time.time()
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        z = torch.from_numpy(np.ascontiguousarray(lat_np[b])).to(device).float()
        z = z.view(len(b), window, 32, 16, 16)
        tau = torch.ones(len(b), window, device=device)
        with torch.no_grad():
            _, agent_out = dyn(z, tau, step_size=d_one.expand(len(b)), actions=None)
        out[b] = agent_out[:, -1].float().cpu().numpy()
        if (s // batch) % 100 == 0:
            done = s + len(b)
            print(f"    agent {done}/{len(idx)}  {done/max(time.time()-t0,1e-9):.1f}/s",
                  flush=True)
    np.save(path, out)
    np.save(path.replace(".npy", "_ok.npy"), ok)
    return out, ok


def feat_state(rows, aux_path="scratchpad/lh_aux.npz", events=EVENTS):
    """NON-VISUAL oracle: off-screen champion state + the gold autocorrelation
    baseline. Nothing here comes from the screen, so any visual probe that does not
    beat this is reading nothing useful off the pixels."""
    lv = np.nan_to_num(rows["level"], nan=1.0)
    hp = np.nan_to_num(rows["hp"], nan=1.0)
    cols = [lv / 18.0, hp]
    names = ["level", "hp_frac"]
    if os.path.exists(aux_path):
        ev = np.load(events, allow_pickle=True)
        aux = np.load(aux_path, allow_pickle=True)
        A, an = aux["aux"], list(aux["names"])
        key = {(m, int(f)): i for m, f, i in
               zip(ev["mid"], ev["frame"], np.arange(len(ev["mid"])))}
        ii = np.array([key.get((m, int(f)), -1) for m, f in zip(rows["mid"], rows["frame"])])
        assert (ii >= 0).all(), "aux lookup miss"
        a = A[ii]
        cols += [np.clip(a[:, 0], 0, 400) / 100.0, np.log1p(np.clip(a[:, 0], 0, 400)),
                 a[:, 1], a[:, 2], a[:, 3] / 30.0, (a[:, 3] / 30.0) ** 2]
        names += [an[0] + "/100", "log_" + an[0], an[1], an[2], "minutes", "minutes^2"]
    return np.stack(cols, 1).astype(np.float32), names


# ────────────────────────────── models ──────────────────────────────

class ImgCNN(nn.Module):
    def __init__(self, chans=(32, 64, 96, 128, 192), in_ch=3, width=1.0, drop=0.2):
        super().__init__()
        c = [max(8, int(x * width)) for x in chans]
        layers, prev = [], in_ch
        for i, ch in enumerate(c):
            layers += [nn.Conv2d(prev, ch, 5 if i == 0 else 3, stride=2,
                                 padding=2 if i == 0 else 1),
                       nn.GroupNorm(min(8, ch), ch), nn.SiLU()]
            prev = ch
        self.body = nn.Sequential(*layers)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.body(x).mean((2, 3))
        return self.head(self.drop(h)).squeeze(-1)


class GridCNN(nn.Module):
    """For the 16x16 latent grid (channels = window*32)."""

    def __init__(self, in_ch, hidden=192, drop=0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.GroupNorm(8, hidden), nn.SiLU())
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        h = self.body(x)
        h = torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)
        return self.head(self.drop(h)).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, d, hidden=0, drop=0.3):
        super().__init__()
        self.net = (nn.Linear(d, 1) if hidden == 0 else
                    nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(drop),
                                  nn.Linear(hidden, 1)))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ────────────────────────────── train / eval ──────────────────────────────

def fit(model, get_batch, ntr, y_tr, evals, epochs, lr, wd, bs, device, seed=0,
        log_every=1, pos_weight=None):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * ((ntr + bs - 1) // bs)), pct_start=0.25)
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    yt = torch.from_numpy(y_tr).float()
    hist = []
    best = (-1.0, None)     # model selection on the inner-val GAMES, never on held-out
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(ntr)
        tot = 0.0
        for i in range(0, ntr, bs):
            b = perm[i:i + bs].numpy()
            xb = get_batch(b, True).to(device, non_blocking=True)
            yb = yt[b].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tot += float(loss) * len(b)
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            sc = {k: auc_scores(predict(model, f, n, device), yy)
                  for k, (f, n, yy) in evals.items()}
            hist.append({"ep": ep + 1, "loss": tot / ntr, **sc})
            if "val" in sc and sc["val"] > best[0]:
                best = (sc["val"], {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()}, ep + 1)
            print("    ep%3d loss=%.4f " % (ep + 1, tot / ntr) +
                  " ".join(f"{k}AUC={v:.3f}" for k, v in sc.items()), flush=True)
    if best[1] is not None:
        print(f"    -> restoring best inner-val epoch {best[2]} (valAUC={best[0]:.3f})",
              flush=True)
        model.load_state_dict(best[1])
        hist.append({"selected_epoch": best[2], "selected_val_auc": best[0]})
    return model, hist


def predict(model, get_batch, n, device, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, n, bs):
            b = np.arange(i, min(i + bs, n))
            out.append(model(get_batch(b, False).to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def report(name, scores, y, groups, extra=None):
    a = auc_scores(scores, y)
    lo, hi = boot_ci_cluster(scores, y, groups)
    d = dict(name=name, auc=a, ci=[lo, hi], n=int(len(y)), pos=float(np.mean(y)),
             games=int(len(set(groups.tolist()))))
    if extra:
        d.update(extra)
    print(f"  {name:34s} n={d['n']:6d} games={d['games']:3d} pos={d['pos']:.3f}  "
          f"AUC={a:.3f}  [{lo:.3f},{hi:.3f}]", flush=True)
    return d
