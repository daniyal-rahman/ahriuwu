#!/usr/bin/env python3
"""Shared event-dataset builder for the last-hit causal triage.

ONE definition of the event set, used by every probe (A raw pixels, B single-frame
latents, C temporal latents, D agent tokens) so all AUCs are comparable.

Two independent anchorings (they are reported separately because the AA label is a
STATE TRANSITION and therefore misses chained autos):

  anchor="commit"  frames where labels action.type ENTERS "attack".
                   y = 1 if gold jump >= `gold_thresh` lands in [t+lo, t+hi].
  anchor="gold"    positives at (gold_frame - LAG) for every gold jump >= thresh;
                   negatives at attack-STATE frames (incl. chained autos) with no
                   gold jump anywhere in [t-8, t+16]. This covers the ~60% of
                   income events that no marked swing is attributable to.

Everything is keyed by (match_id, frame) so the four probes index the identical rows.
"""
import json
import os

import numpy as np

FRAMES_ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
LAG = 6              # swing commit -> gold arrival, median at 20 fps (established)
WIN_LO, WIN_HI = 4, 9   # label window [t+4, t+9] around the 5-8 frame arrival


def load_match(mid, frames_root=FRAMES_ROOT):
    with open(os.path.join(frames_root, mid, "labels.json")) as f:
        d = json.load(f)
    frames = d["frames"]
    T = len(frames)
    gold = np.full(T, np.nan)
    atype = np.array([""] * T, dtype=object)
    cs_x = np.full(T, np.nan)
    cs_y = np.full(T, np.nan)
    level = np.full(T, np.nan)
    hp = np.full(T, np.nan)
    fidx = np.zeros(T, dtype=np.int64)
    for i, fr in enumerate(frames):
        fidx[i] = int(fr["frame"])
        lab = fr.get("label")
        if not lab:
            continue
        st = lab.get("champion_stats") or {}
        g = st.get("gold_total")
        if g is not None:
            gold[i] = float(g)
        lv = st.get("level")
        if lv is not None:
            level[i] = float(lv)
        h = st.get("hp")
        if h is not None and st.get("hp_max"):
            hp[i] = float(h) / float(st["hp_max"])
        sc = lab.get("champion_screen")
        if sc:
            cs_x[i], cs_y[i] = float(sc[0]), float(sc[1])
        a = (lab.get("action") or {}).get("type")
        atype[i] = a if a else ""
    assert np.all(np.diff(fidx) == 1) and fidx[0] == 0, f"{mid}: non-contiguous frame index"
    return dict(mid=mid, T=T, gold=gold, atype=atype, cs_x=cs_x, cs_y=cs_y,
                level=level, hp=hp, meta=d)


def gold_jumps(gold, thresh=10.0):
    """Boolean (T,) — a gold_total increase of >= thresh landed AT this frame."""
    T = len(gold)
    jump = np.zeros(T, dtype=bool)
    prev = None
    for i in range(T):
        g = gold[i]
        if np.isnan(g):
            prev = None
            continue
        if prev is not None and (g - prev) >= thresh:
            jump[i] = True
        prev = g
    return jump


def _future_hit(jump, t, lo=WIN_LO, hi=WIN_HI):
    T = len(jump)
    c = np.concatenate([[0], np.cumsum(jump)])
    a = np.clip(t + lo, 0, T)
    b = np.clip(t + hi + 1, 0, T)
    return (c[b] - c[a]) > 0


def build_events(m, anchor="commit", gold_thresh=10.0, min_sep=4):
    """Return (frames, y, extra) for one match."""
    T, gold, atype = m["T"], m["gold"], m["atype"]
    jump = gold_jumps(gold, gold_thresh)
    is_atk = np.array([a == "attack" for a in atype])
    enters = is_atk & ~np.concatenate([[False], is_atk[:-1]])
    valid = ~np.isnan(gold) & ~np.isnan(m["cs_x"])

    if anchor == "commit":
        t = np.where(enters)[0]
        t = t[(t >= 20) & (t + 20 < T)]
        t = t[valid[t]]
        y = _future_hit(jump, t).astype(np.float32)
        return t, y, {}

    if anchor == "gold":
        g = np.where(jump)[0]
        pos = g - LAG
        pos = pos[(pos >= 20) & (pos + 20 < T)]
        pos = pos[valid[pos]]
        # negatives: attack-state frames far from any gold jump
        c = np.concatenate([[0], np.cumsum(jump)])
        cand = np.where(is_atk)[0]
        cand = cand[(cand >= 20) & (cand + 20 < T)]
        cand = cand[valid[cand]]
        if len(cand):
            lo = np.clip(cand - 8, 0, T)
            hi = np.clip(cand + 17, 0, T)
            cand = cand[(c[hi] - c[lo]) == 0]
        # thin both sets so near-duplicate frames don't dominate
        pos = _thin(pos, min_sep)
        cand = _thin(cand, min_sep)
        t = np.concatenate([pos, cand]).astype(np.int64)
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(cand))]).astype(np.float32)
        o = np.argsort(t)
        return t[o], y[o], {}

    raise ValueError(anchor)


def _thin(idx, min_sep):
    out = []
    last = -10 ** 9
    for i in sorted(idx):
        if i - last >= min_sep:
            out.append(i)
            last = i
    return np.array(out, dtype=np.int64)


def build_all(mids, anchor="commit", gold_thresh=10.0, verbose=False):
    rows = []
    for mid in mids:
        try:
            m = load_match(mid)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {mid}: {e}", flush=True)
            continue
        t, y, _ = build_events(m, anchor, gold_thresh)
        if len(t) == 0:
            continue
        rows.append(dict(mid=mid, frames=t, y=y,
                         cs=np.stack([m["cs_x"][t], m["cs_y"][t]], 1),
                         level=m["level"][t], hp=m["hp"][t]))
        if verbose:
            print(f"  {mid}: n={len(t)} pos={y.mean():.3f}", flush=True)
    return rows


# ---------------------------------------------------------------- metrics

def auc(pos, neg):
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(len(allv), dtype=float)
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def auc_scores(s, y):
    s = np.asarray(s, float)
    y = np.asarray(y).astype(bool)
    return auc(s[y], s[~y])


def boot_ci_cluster(s, y, groups, n=2000, seed=0):
    """Cluster bootstrap over GAMES — adjacent events inside a game are correlated,
    so a naive per-sample bootstrap gives CIs that are far too tight."""
    rng = np.random.default_rng(seed)
    s = np.asarray(s, float)
    y = np.asarray(y).astype(bool)
    groups = np.asarray(groups)
    uq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uq, len(uq), replace=True)
        ii = np.concatenate([idx_by_g[g] for g in pick])
        a = auc(s[ii][y[ii]], s[ii][~y[ii]])
        if not np.isnan(a):
            vals.append(a)
    if len(vals) < 10:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, .025)), float(np.quantile(vals, .975)))


# ---------------------------------------------------------------- splits

LAT_BC = "/srv/nfs/datasets/replay_latents_v7_bc"
LAT_HO = "/srv/nfs/datasets/replay_latents_v7_heldout"
LAT_UNSEEN = "/srv/nfs/projects/ahriuwu/scratchpad/lh_latents_unseen"


def splits(events_npz="scratchpad/lh_events.npz", n_train=100, seed=0):
    """train / heldout_seen come from the 125 tokenizer-TRAIN games; heldout_unseen
    are games the v7 tokenizer and the Phase-2 dynamics never saw (I encoded them
    sparsely for this experiment). Split is by WHOLE GAME everywhere."""
    ev = np.load(events_npz, allow_pickle=True)
    games = sorted(set(ev["mid"].tolist()))
    bc = sorted(f[:-3] for f in os.listdir(LAT_BC) if f.endswith(".pt"))
    bc = [g for g in bc if g in games]
    unseen_all = [g for g in games if g not in bc]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(bc))
    bc = [bc[i] for i in perm]
    have_unseen = []
    if os.path.isdir(LAT_UNSEEN):
        have_unseen = sorted(f[:-3] for f in os.listdir(LAT_UNSEEN) if f.endswith(".pt"))
    if os.path.isdir(LAT_HO):
        have_unseen += sorted(f[:-3] for f in os.listdir(LAT_HO) if f.endswith(".pt"))
    have_unseen = [g for g in sorted(set(have_unseen)) if g in unseen_all]
    # heldout_unseen is ALWAYS every non-tokenized game: probes A/Acrop/cheat need
    # no latents and can use all of them, while the latent probes are automatically
    # restricted to the subset that has latents by their `ok` mask. The aggregator
    # intersects by game so the cross-probe comparison stays apples-to-apples.
    return {"train": bc[:n_train], "heldout_seen": bc[n_train:],
            "heldout_unseen": unseen_all, "with_latents": have_unseen,
            "unseen_all": unseen_all}


def latent_dir_for(mid, sp):
    """Directory holding this match's v7 latents, or None if it has none."""
    for d in (LAT_BC, LAT_UNSEEN, LAT_HO):
        if os.path.exists(os.path.join(d, f"{mid}.pt")):
            return d
    return None
