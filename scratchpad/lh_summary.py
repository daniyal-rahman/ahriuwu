#!/usr/bin/env python3
"""Aggregate the probe score dumps into ONE comparable table.

Every probe dumps per-row scores keyed by (match, frame), so the held-out AUCs can
be recomputed on the exact same rows — which matters because the latent probes can
only score the unseen games I actually tokenized, while the pixel probes can score
all of them.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lasthit_events import auc_scores, boot_ci_cluster  # noqa: E402


def strat_auc(s, y, mid, frame, minutes_per_bin=2.0):
    """AUC computed ONLY between positives and negatives inside the same game AND the
    same 2-minute slice of that game.

    This matters: the non-visual oracle (probe S) already reaches ~0.70 raw, almost
    all of it from champion level / game time — early Garen autos rarely convert,
    late ones usually do. Any pixel or latent probe gets that clock for free off the
    screen. Only the within-(game, time) concordance answers "does it know THIS swing
    is the one".
    """
    s = np.asarray(s, float)
    y = np.asarray(y).astype(bool)
    key = np.array([f"{m}|{int(fr / 20.0 / 60.0 / minutes_per_bin)}"
                    for m, fr in zip(mid, frame)])
    num = den = 0.0
    for k in np.unique(key):
        m = key == k
        p, n = s[m & y], s[m & ~y]
        if len(p) == 0 or len(n) == 0:
            continue
        c = (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
        num += c
        den += len(p) * len(n)
    return (num / den) if den else float("nan"), int(den)


def _strat_ids(mid, frame, minutes_per_bin=2.0):
    b = (frame / 20.0 / 60.0 / minutes_per_bin).astype(np.int64)
    _, gi = np.unique(mid, return_inverse=True)
    _, sid = np.unique(gi * 10000 + b, return_inverse=True)
    return sid


def _fast_strat_auc(s, y, sid):
    """Mann-Whitney U pooled over strata, vectorised (ties broken by sort order —
    negligible here, the probes emit continuous logits)."""
    order = np.lexsort((s, sid))
    ss, yy = sid[order], y[order]
    start = np.searchsorted(ss, np.arange(ss[-1] + 1), "left")
    cnt = np.bincount(ss, minlength=ss[-1] + 1)
    rank = np.arange(len(ss)) - np.repeat(start, cnt) + 1
    npos = np.bincount(ss, weights=yy, minlength=len(cnt))
    rsum = np.bincount(ss, weights=rank * yy, minlength=len(cnt))
    nneg = cnt - npos
    u = rsum - npos * (npos + 1) / 2.0
    den = (npos * nneg).sum()
    return (u.sum() / den if den else float("nan")), int(den)


def strat_ci(s, y, mid, frame, n=400, seed=0):
    rng = np.random.default_rng(seed)
    uq, gi = np.unique(mid, return_inverse=True)
    idx = [np.where(gi == k)[0] for k in range(len(uq))]
    sid0 = _strat_ids(mid, frame)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uq), len(uq))
        ii = np.concatenate([idx[k] for k in pick])
        # re-key so a game drawn twice does not merge into one stratum
        off = np.concatenate([np.full(len(idx[k]), j) for j, k in enumerate(pick)])
        _, sid = np.unique(sid0[ii] * 1000 + off, return_inverse=True)
        a, d = _fast_strat_auc(s[ii].astype(float), y[ii].astype(float), sid)
        if d and not np.isnan(a):
            vals.append(a)
    if len(vals) < 10:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, .025)), float(np.quantile(vals, .975)))


def load(tag):
    p = f"scratchpad/lh_scores_{tag}.npz"
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    out = {}
    for k in ["train", "val", "ho_seen", "ho_unseen"]:
        if f"{k}_s" in d:
            out[k] = dict(s=d[f"{k}_s"], mid=d[f"{k}_mid"], frame=d[f"{k}_frame"],
                          y=d[f"{k}_y"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--strat-ci", action="store_true")
    ap.add_argument("--common-unseen", action="store_true",
                    help="restrict ho_unseen to the games EVERY probe scored")
    args = ap.parse_args()
    tags = args.tags or sorted(
        os.path.basename(p)[len("lh_scores_"):-4] for p in glob.glob("scratchpad/lh_scores_*.npz"))
    res = {t: load(t) for t in tags}
    res = {t: v for t, v in res.items() if v}

    common = None
    if args.common_unseen:
        for t, v in res.items():
            if "ho_unseen" in v:
                g = set(v["ho_unseen"]["mid"].tolist())
                common = g if common is None else (common & g)
        print(f"common unseen games ({len(common or [])}): {sorted(common or [])}\n")

    print(f"{'probe':28s} {'split':10s} {'n':>6s} {'gm':>3s} {'pos':>5s} "
          f"{'AUC':>6s} {'95%CI(game-boot)':>18s} | {'AUCstrat':>8s} {'95%CI':>16s} {'pairs':>8s}")
    for t in tags:
        v = res.get(t)
        if not v:
            continue
        for k in ["train", "val", "ho_seen", "ho_unseen"]:
            if k not in v:
                continue
            d = v[k]
            s, y, mid, fr = d["s"], d["y"], d["mid"], d["frame"]
            if k == "ho_unseen" and common:
                m = np.isin(mid, sorted(common))
                s, y, mid, fr = s[m], y[m], mid[m], fr[m]
            if len(y) < 20 or len(set(y.tolist())) < 2:
                continue
            a = auc_scores(s, y)
            lo, hi = boot_ci_cluster(s, y, mid)
            sa, npair = _fast_strat_auc(s.astype(float), y.astype(float),
                                        _strat_ids(mid, fr))
            slo, shi = strat_ci(s.astype(float), y.astype(float), mid, fr, n=2000)
            print(f"{t:28s} {k:10s} {len(y):6d} {len(set(mid.tolist())):3d} {y.mean():5.3f} "
                  f"{a:6.3f} [{lo:6.3f},{hi:6.3f}] | {sa:8.3f} [{slo:6.3f},{shi:6.3f}] "
                  f"{npair:8d}")
        print()


if __name__ == "__main__":
    main()


def compare(tag_a, tag_b, split="ho_seen", n=2000, seed=0):
    """PAIRED game-level bootstrap of (stratified AUC of A) - (of B) on the rows both
    probes scored. Comparing two independent CIs would be the wrong test — the probes
    see the same events, so the difference has far less variance than either AUC."""
    va, vb = load(tag_a), load(tag_b)
    if not va or not vb or split not in va or split not in vb:
        return None
    ka = {(m, int(f)): i for i, (m, f) in enumerate(zip(va[split]["mid"], va[split]["frame"]))}
    ib, ia = [], []
    for j, (m, f) in enumerate(zip(vb[split]["mid"], vb[split]["frame"])):
        i = ka.get((m, int(f)))
        if i is not None:
            ia.append(i)
            ib.append(j)
    ia, ib = np.array(ia), np.array(ib)
    if len(ia) < 50:
        return None
    sa = va[split]["s"][ia].astype(float)
    sb = vb[split]["s"][ib].astype(float)
    y = va[split]["y"][ia].astype(float)
    mid, fr = va[split]["mid"][ia], va[split]["frame"][ia]
    assert (y == vb[split]["y"][ib].astype(float)).all(), "row alignment broken"
    sid0 = _strat_ids(mid, fr)
    base = (_fast_strat_auc(sa, y, sid0)[0], _fast_strat_auc(sb, y, sid0)[0])
    rng = np.random.default_rng(seed)
    uq, gi = np.unique(mid, return_inverse=True)
    idx = [np.where(gi == k)[0] for k in range(len(uq))]
    d = []
    for _ in range(n):
        pick = rng.integers(0, len(uq), len(uq))
        ii = np.concatenate([idx[k] for k in pick])
        off = np.concatenate([np.full(len(idx[k]), j) for j, k in enumerate(pick)])
        _, sid = np.unique(sid0[ii] * 1000 + off, return_inverse=True)
        x1 = _fast_strat_auc(sa[ii], y[ii], sid)[0]
        x2 = _fast_strat_auc(sb[ii], y[ii], sid)[0]
        if not (np.isnan(x1) or np.isnan(x2)):
            d.append(x1 - x2)
    d = np.array(d)
    return dict(n=len(ia), games=len(uq), a=base[0], b=base[1], diff=base[0] - base[1],
                ci=[float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                p_gt0=float((d <= 0).mean()))
