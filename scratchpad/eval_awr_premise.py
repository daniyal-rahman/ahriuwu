#!/usr/bin/env python3
"""Does advantage-weighted BC (AWR) have anything to grip on in these replays?

AWR reweights each demonstrated action by exp(A_t / beta). That only teaches
anything if A_t VARIES across frames in a way tied to the behaviour we care
about. Two ways it can be vacuous:

  1. A_t ~= 0 everywhere (all frames look equally valuable) -> weights uniform.
  2. A_t varies, but ONLY with something a value function trivially predicts
     from the clock (gold accrues, so the remaining discounted return falls
     monotonically through the game). A learned V absorbs that entirely, leaving
     no advantage. This is the trap: raw std(G_t) looks big and means nothing.

So we report BOTH the raw return spread AND the spread that survives a
time-local baseline (the strongest clock-only V a critic could learn), plus the
AWR effective sample size, plus -- the decision-relevant one -- whether the
surviving advantage actually up-weights the frames just before a last-hit.

CPU only; reads labels.json directly (same formula as rewards/reward.py).

  python scratchpad/eval_awr_premise.py --n-matches 25
"""
import argparse
import json
import os
import sys

import numpy as np

FPS = 20.0


def match_reward(labels_path, gold_scale=1e-3, death_penalty=-0.2):
    """Replicates rewards/reward.py _dense_solo_gold + _death_event."""
    with open(labels_path) as f:
        d = json.load(f)
    frames = d.get("frames") or []
    T = len(frames)
    r = np.zeros(T, dtype=np.float64)
    prev = None
    prev_hp = None
    for i, fr in enumerate(frames):
        lab = fr.get("label")
        cs = lab.get("champion_stats") if lab else None
        if cs is None:
            prev, prev_hp = None, None
            continue
        g = cs.get("gold_total")
        if g is None:
            prev = None
        else:
            if prev is not None:
                r[i] += gold_scale * (float(g) - prev)
            prev = float(g)
        hp = cs.get("hp")
        if hp is not None:
            if prev_hp is not None and prev_hp > 0 and float(hp) <= 0:
                r[i] += death_penalty
            prev_hp = float(hp)
        else:
            prev_hp = None
    return r


def discounted_return(r, gamma):
    G = np.zeros_like(r)
    acc = 0.0
    for i in range(len(r) - 1, -1, -1):
        acc = r[i] + gamma * acc
        G[i] = acc
    return G


def moving_average(x, w):
    """Centered moving average, edge-padded (the clock-only value baseline)."""
    if w <= 1:
        return x.copy()
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    k = np.ones(w) / w
    return np.convolve(xp, k, mode="valid")[: len(x)]


def ess(weights):
    """AWR effective sample size / N. 1.0 = uniform weights = no reweighting."""
    w = weights / weights.sum()
    return float(1.0 / (len(w) * np.sum(w ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", default="/srv/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--match-list", default="scratchpad/_usable_matches.txt")
    ap.add_argument("--n-matches", type=int, default=25)
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--thresh", type=float, default=0.010)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.match_list) as f:
        matches = [l.strip() for l in f if l.strip()][: args.n_matches]

    R, G, A_local, A_global = [], [], [], []
    ev_all = []
    per_match = []
    for m in matches:
        p = os.path.join(args.labels_root, m, "labels.json")
        if not os.path.exists(p):
            continue
        r = match_reward(p)
        if len(r) < 2000:
            continue
        g = discounted_return(r, args.gamma)
        # clock-only value baselines of two horizons
        v_short = moving_average(g, 201)    # +/- 5 s
        v_long = moving_average(g, 801)     # +/- 20 s
        R.append(r); G.append(g)
        A_local.append(g - v_short)
        A_global.append(g - g.mean())
        ev_all.append(r >= args.thresh)
        per_match.append({
            "match": m, "T": len(r),
            "std_G": float(g.std()),
            "std_A_local": float((g - v_short).std()),
            "std_A_long": float((g - v_long).std()),
            "r2_clock": float(1 - ((g - v_long) ** 2).mean() / (g.var() + 1e-12)),
        })
        print(f"  {m}: T={len(r)} ({len(r)/FPS/60:.1f} min)  std(G)={g.std():.4f}  "
              f"std(G-Vlocal)={(g-v_short).std():.4f}", flush=True)

    r = np.concatenate(R); g = np.concatenate(G)
    al = np.concatenate(A_local); ag = np.concatenate(A_global)
    ev = np.concatenate(ev_all)
    N = len(r)
    hl = np.log(0.5) / np.log(args.gamma)
    print(f"\n=== AWR premise | {len(per_match)} matches, {N} frames "
          f"({N/FPS/3600:.2f} h) | gamma={args.gamma} "
          f"(horizon 1/(1-g)={1/(1-args.gamma):.0f} frames = {1/(1-args.gamma)/FPS:.1f}s, "
          f"half-life {hl:.0f} frames = {hl/FPS:.1f}s) ===")

    print("\n--- per-frame reward r_t ---")
    print(f"  exactly zero : {float((r == 0).mean()):.2%}")
    print(f"  in (0, .01)  : {float(((r > 0) & (r < args.thresh)).mean()):.2%}  (passive gold trickle)")
    print(f"  >= .01       : {float((r >= args.thresh).mean()):.3%}  (last-hits / kills)")
    print(f"  negative     : {float((r < 0).mean()):.4%}  (deaths)")
    print(f"  mean={r.mean():.6f}  std={r.std():.6f}  p99={np.quantile(r,.99):.5f}  max={r.max():.3f}")

    print(f"\n--- discounted return G_t (gamma={args.gamma}) ---")
    q = np.quantile(g, [.01, .25, .5, .75, .99])
    print(f"  mean={g.mean():.4f}  std={g.std():.4f}")
    print(f"  p1={q[0]:.4f} p25={q[1]:.4f} p50={q[2]:.4f} p75={q[3]:.4f} p99={q[4]:.4f}")
    print(f"  coefficient of variation = {g.std()/abs(g.mean()):.3f}")

    print("\n--- how much of that spread is just the game clock? ---")
    print("  (V_clock = centered moving average of G; a critic learns this for free)")
    for name, a in [("A = G - mean(G)      [no baseline]", ag),
                    ("A = G - V_local(+-5s)  ", al)]:
        print(f"  {name}: std={a.std():.5f}  "
              f"|A|>1 minion (0.02): {float((np.abs(a) > 0.02).mean()):.1%}  "
              f"|A|>0.005: {float((np.abs(a) > 0.005).mean()):.1%}")
    var_expl = 1 - al.var() / g.var()
    print(f"  variance of G explained by the local clock baseline: {var_expl:.1%}")

    print("\n--- AWR weights w = min(exp(A/beta), wmax); ESS/N: 1.0 = uniform = useless ---")
    print("  (unclipped ESS is meaningless here: r has a heavy tail (kills/objectives,")
    print("   max r = %.2f) so a handful of frames would eat all the weight)" % r.max())
    print("  baseline     beta          wmax   ESS/N   w_p99/w_p50  top1%%_weight_share")
    for name, a in [("G-mean(G)", ag), ("G-V_local", al)]:
        for bmul in [1.0, 0.5]:
            beta = max(a.std() * bmul, 1e-9)
            for wmax in [20.0, np.inf]:
                w = np.minimum(np.exp(np.clip(a / beta, -20, 20)), wmax)
                wq = np.quantile(w, [.5, .99])
                srt = np.sort(w)[::-1]
                share = srt[: max(1, len(w) // 100)].sum() / w.sum()
                print(f"  {name:11s}  {beta:.5f}({bmul:g}sd)  {wmax:5.0f}  {ess(w):.3f}   "
                      f"{wq[1]/max(wq[0],1e-12):9.2f}   {share:.3f}")

    print("\n--- DECISION-RELEVANT: does the surviving advantage up-weight the")
    print("    frames just BEFORE a last-hit (i.e. would AWR teach last-hitting)? ---")
    idx = np.where(ev)[0]
    print("   lead(frames) lead(ms)     mean A(G-V_local)   z vs global")
    mu, sd = al.mean(), al.std()
    for d in [20, 10, 6, 4, 3, 2, 1, 0, -1, -3, -10]:
        t = idx - d
        t = t[(t >= 0) & (t < N)]
        v = al[t]
        print(f"     {d:+5d}     {int(1000*d/FPS):+6d}        {v.mean():+.5f}        "
              f"{(v.mean()-mu)/(sd+1e-12):+.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"per_match": per_match,
                       "gamma": args.gamma, "N": N,
                       "std_G": float(g.std()), "std_A_local": float(al.std()),
                       "var_explained_by_clock": float(var_expl)}, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
