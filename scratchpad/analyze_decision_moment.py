#!/usr/bin/env python3
"""THE decisive test: does the reward head see the last-hit BEFORE the human commits?

The raw anticipation AUCs (0.90 at 150-200 ms lead) have a fatal alternative
explanation. use_actions=False, so the model cannot read the AA keypress -- but
it reads the SCREEN, and at 100-300 ms before the gold arrives Garen's attack
WINDUP ANIMATION is already playing and the minion is already being hit. Seeing
"an attack is in flight" is a visible consequence of a decision the human
already made. A policy cannot use that: by the time the windup is on screen it
is too late to decide to attack.

So we ask three progressively harder questions:

  Q1 DECISION MOMENT. For each income event, find the AA press that caused it.
     What is the head's score at (press - 1), i.e. just BEFORE the human
     committed? If elevated -> the head sees the opportunity, not the windup.

  Q2 AA-FREE ANTICIPATION. Restrict to frames with NO AA press in [t-W, t].
     The windup cannot be on screen. Does forward income still get predicted?

  Q3 AA TRIAGE (the one v1.5 actually needs). Given the human pressed AA at t,
     can the head separate the presses that landed a last-hit from the ones that
     did not? This is "is NOW the right time to swing", scored only on the
     frames where swinging was on the table. Base rate here is ~40%, not 4%, so
     precision is meaningful instead of being crushed by class imbalance.

Reads the .npz dumped by eval_reward_anticipation.py --dump.
"""
import argparse
import numpy as np

FPS = 20.0


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(len(allv), dtype=float)
    sv = allv[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def boot_ci(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    if len(pos) < 3 or len(neg) < 3:
        return (float("nan"), float("nan"))
    vals = [auc(rng.choice(pos, len(pos), replace=True),
                rng.choice(neg, len(neg), replace=True)) for _ in range(n)]
    return (float(np.quantile(vals, .025)), float(np.quantile(vals, .975)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--thresh", type=float, default=0.010)
    ap.add_argument("--tag", default="")
    ap.add_argument("--attack-window", type=int, default=12,
                    help="max frames from AA press to gold for attributing an event")
    args = ap.parse_args()

    d = np.load(args.npz)
    R, P, AA = d["r"], d["p"], d["aa"]        # (W,T), (W,T,L), (W,T)
    W, T, L = P.shape
    # forward score = the head's total 'income in the next 50..400 ms' mass.
    # offsets 1..L-1 only -> strictly future, strictly causal.
    S = P[:, :, 1:].sum(axis=2)               # (W,T)
    flat_s = S.reshape(-1)
    mu, sd = flat_s.mean(), flat_s.std()
    print(f"=== {args.tag or args.npz} ===")
    print(f"windows={W} T={T} frames={W*T} ({W*T/FPS:.0f}s)  "
          f"events>={args.thresh}: {int((R>=args.thresh).sum())}  "
          f"AA press rate {AA.mean():.2%}")
    print(f"forward score: mean={mu:.5f} sd={sd:.5f}\n")

    # ---------------- Q1: decision moment ----------------
    print("--- Q1  score just BEFORE the AA press that caused each last-hit ---")
    print("    (if the head only watches the windup, this sits at the global mean)")
    pre_scores, press_scores, gold_scores = [], [], []
    lags = []
    for w in range(W):
        ev = np.where(R[w] >= args.thresh)[0]
        for e in ev:
            lo = max(0, e - args.attack_window)
            presses = np.where(AA[w][lo:e + 1] > 0)[0]
            if len(presses) == 0:
                continue                       # not attributable to an AA (Q kill, assist)
            a = lo + presses[-1]               # last AA press before the gold
            lags.append(e - a)
            if a - 1 >= 0:
                pre_scores.append(S[w][a - 1])
            press_scores.append(S[w][a])
            gold_scores.append(S[w][e])
    pre = np.array(pre_scores); prs = np.array(press_scores); gld = np.array(gold_scores)
    lags = np.array(lags)
    if len(lags):
        print(f"    attributed events: {len(lags)}  "
              f"press->gold lag: median {np.median(lags):.0f} frames "
              f"({np.median(lags)*50:.0f} ms), p90 {np.quantile(lags,.9):.0f}")
        for name, v in [("score at press-1 (DECISION)", pre),
                        ("score at press      ", prs),
                        ("score at gold arrival", gld)]:
            if len(v) == 0:
                continue
            pct = float((flat_s < v.mean()).mean())
            print(f"    {name}: mean={v.mean():.5f}  z={(v.mean()-mu)/sd:+.2f}  "
                  f"pctile_of_global={pct:.3f}  n={len(v)}")
    print()

    # ---------------- Q2: AA-free anticipation ----------------
    print("--- Q2  anticipation on frames with NO AA press in [t-W_aa, t] ---")
    print("    (no windup can be on screen; if AUC holds, the head reads the world)")
    print("    W_aa  k  ms   npos  nneg    AUC   [95% CI]")
    for w_aa in [4, 8, 12]:
        for k in [3, 5, 10]:
            pos, neg = [], []
            for w in range(W):
                r, s, aa = R[w], S[w], AA[w]
                csum = np.concatenate([[0.0], np.cumsum(r)])
                t = np.arange(w_aa, T - k)
                # no AA press in [t-w_aa, t]
                aac = np.concatenate([[0], np.cumsum(aa > 0)])
                no_aa = (aac[t + 1] - aac[t + 1 - w_aa - 1]) == 0
                # no income already flowing
                clean = (csum[t + 1] - csum[np.maximum(t + 1 - 6, 0)]) <= 0
                m = no_aa & clean
                fut = csum[t + k + 1] - csum[t + 1]
                y = fut >= args.thresh
                pos.append(s[t][m & y]); neg.append(s[t][m & ~y])
            pos = np.concatenate(pos); neg = np.concatenate(neg)
            a = auc(pos, neg); lo, hi = boot_ci(pos, neg)
            print(f"     {w_aa:2d}   {k:2d} {int(1000*k/FPS):4d} {len(pos):6d} {len(neg):6d}  "
                  f"{a:.3f}  [{lo:.3f},{hi:.3f}]")
    print()

    # ---------------- Q3: AA triage ----------------
    print("--- Q3  AA TRIAGE: given the human pressed AA at t, will it get gold? ---")
    print("    score at t-1 (pre-decision) and at t; label = income>=thresh in (t, t+k]")
    print("     use    k  ms   n_press  base   AUC   [95% CI]   prec@top50%  prec@top25%")
    for use in ["t-1", "t"]:
        for k in [8, 12, 16]:
            sc, lab = [], []
            for w in range(W):
                r, s, aa = R[w], S[w], AA[w]
                csum = np.concatenate([[0.0], np.cumsum(r)])
                t = np.arange(1, T - k)
                m = aa[t] > 0
                tt = t[m]
                if len(tt) == 0:
                    continue
                fut = csum[tt + k + 1] - csum[tt + 1]
                lab.append(fut >= args.thresh)
                sc.append(s[tt - 1] if use == "t-1" else s[tt])
            if not sc:
                continue
            sc = np.concatenate(sc); lab = np.concatenate(lab)
            a = auc(sc[lab], sc[~lab]); lo, hi = boot_ci(sc[lab], sc[~lab])
            p50 = lab[sc >= np.quantile(sc, .5)].mean() if len(sc) > 4 else float("nan")
            p25 = lab[sc >= np.quantile(sc, .75)].mean() if len(sc) > 4 else float("nan")
            print(f"     {use:4s}  {k:2d} {int(1000*k/FPS):4d}  {len(sc):6d}  {lab.mean():.3f}  "
                  f"{a:.3f}  [{lo:.3f},{hi:.3f}]     {p50:.3f}       {p25:.3f}")
    print()

    # ---------------- lead-time profile with CI ----------------
    # ISOLATED events only: minions die in clusters, so the score at lead +20
    # before event B is often lead -5 after event A. Requiring no other income
    # within +/-`iso` frames removes that contamination in both directions.
    for iso in [0, 25]:
        print(f"--- lead-time profile, score vs GOLD ARRIVAL "
              f"({'all events' if iso == 0 else f'ISOLATED events only (+/-{iso} frames)'}) ---")
        print("    lead  ms      n    mean      z    pctile")
        for dlead in [20, 15, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0, -2, -5]:
            v = []
            for w in range(W):
                r = R[w]
                ev = np.where(r >= args.thresh)[0]
                if iso:
                    csum = np.concatenate([[0], np.cumsum(r >= args.thresh)])
                    lo = np.maximum(ev - iso, 0); hi = np.minimum(ev + iso + 1, T)
                    ev = ev[(csum[hi] - csum[lo]) == 1]
                t = ev - dlead
                t = t[(t >= 0) & (t < T)]
                v.append(S[w][t])
            v = np.concatenate(v)
            if len(v) == 0:
                continue
            print(f"    {dlead:+4d} {int(1000*dlead/FPS):+5d} {len(v):5d}  {v.mean():.5f} "
                  f"{(v.mean()-mu)/sd:+6.2f}   {float((flat_s < v.mean()).mean()):.3f}")
        print()

    # ---------------- profile relative to the AA PRESS ----------------
    print("--- profile relative to the AA PRESS that scored (0 = press frame) ---")
    print("    negative = before the human committed; the plan needs elevation there")
    prof = {}
    for w in range(W):
        r, aa = R[w], AA[w]
        ev = np.where(r >= args.thresh)[0]
        for e in ev:
            lo = max(0, e - args.attack_window)
            pr = np.where(aa[lo:e + 1] > 0)[0]
            if len(pr) == 0:
                continue
            a = lo + pr[-1]
            for off in range(-10, 11):
                t = a + off
                if 0 <= t < T:
                    prof.setdefault(off, []).append(S[w][t])
    print("    off(frames)  ms      n    mean      z")
    for off in sorted(prof):
        v = np.array(prof[off])
        print(f"      {off:+4d}     {int(1000*off/FPS):+5d} {len(v):5d}  {v.mean():.5f} "
              f"{(v.mean()-mu)/sd:+6.2f}")


if __name__ == "__main__":
    main()
