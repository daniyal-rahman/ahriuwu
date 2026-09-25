#!/usr/bin/env python3
"""Corpus analyses over scratchpad/audit_cache/*.npz. Prints a numbered report."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

CACHE = Path("/srv/nfs/projects/ahriuwu/scratchpad/audit_cache")
BINS = 21
DEADBAND = 0.01


def load_all():
    out = []
    for p in sorted(CACHE.glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        m = json.loads(str(z["meta"][0]))
        out.append((m, z))
    return out


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def pct(a, b):
    return 100.0 * a / max(b, 1)


def main():
    data = load_all()
    print(f"{len(data)} matches loaded")

    # ---------------- 0. corpus basics ----------------
    sec("0. CORPUS BASICS")
    Ts = np.array([m["T"] for m, _ in data])
    durs = np.array([(z["gt"][-1] - z["gt"][0]) for _, z in data])
    print(f"frames/match: min {Ts.min()} med {int(np.median(Ts))} max {Ts.max()} total {Ts.sum():,}")
    print(f"game duration (s): min {durs.min():.0f} med {np.median(durs):.0f} max {durs.max():.0f}")
    print(f"  matches < 10 min: {(durs < 600).sum()}  < 15 min: {(durs < 900).sum()}")
    res = Counter(tuple(m["screen_resolution"]) for m, _ in data)
    fres = Counter(tuple(m["frame_resolution"]) for m, _ in data)
    print("screen_resolution:", dict(res), " frame_resolution:", dict(fres))
    print("fps:", Counter(m["fps"] for m, _ in data))
    bad_npng = [(m["match_id"], m["T"], m["n_png"], m["total_frames"])
                for m, _ in data if m["n_png"] != m["T"] or m["total_frames"] != m["T"]]
    print(f"T != n_png or != total_frames: {len(bad_npng)} matches {bad_npng[:5]}")
    # dt uniformity
    bad_dt = []
    for m, z in data:
        d = np.diff(z["gt"])
        if not np.allclose(d, 1.0 / m["fps"], atol=1e-6):
            bad_dt.append((m["match_id"], float(d.min()), float(d.max())))
    print(f"non-uniform gt spacing: {len(bad_dt)} matches {bad_dt[:5]}")
    lab = np.array([z["labeled"].mean() for _, z in data])
    print(f"labeled fraction: min {lab.min():.4f} mean {lab.mean():.4f}")

    # ---------------- 1. corrupt stats ----------------
    sec("1. CORRUPT / ABSURD STAT READS")
    fields = {
        "gold": (0, 5e4), "gold_total": (0, 5e4), "hp": (-1e3, 1e5),
        "hp_max": (1, 1e5), "level": (1, 18),
    }
    tot = Counter()
    n_lab_tot = 0
    per_match_bad = Counter()
    for m, z in data:
        L = z["labeled"]
        n_lab_tot += L.sum()
        for f, (lo, hi) in fields.items():
            v = z[f][L]
            bad = ~np.isfinite(v) | (v < lo) | (v > hi)
            tot[f] += int(bad.sum())
            if bad.any():
                per_match_bad[f] += 1
    print(f"labeled frames total: {n_lab_tot:,}")
    for f in fields:
        print(f"  {f:11s} out-of-range: {tot[f]:>10,}  ({pct(tot[f], n_lab_tot):6.3f}%)  "
              f"in {per_match_bad[f]}/{len(data)} matches")
    # magnitude of the gold corruption
    allgold = np.concatenate([z["gold"][z["labeled"]] for _, z in data])
    bad = ~np.isfinite(allgold) | (allgold < 0) | (allgold > 5e4)
    print(f"  champion_stats.gold: {pct(bad.sum(), len(allgold)):.2f}% corrupt; "
          f"|value| range of corrupt: {np.nanmin(np.abs(allgold[bad])):.3g} .. "
          f"{np.nanmax(np.abs(allgold[bad])):.3g}")
    gtot = np.concatenate([z["gold_total"][z["labeled"]] for _, z in data])
    b2 = ~np.isfinite(gtot) | (gtot < 0) | (gtot > 5e4)
    print(f"  gold_total: {pct(b2.sum(), len(gtot)):.4f}% corrupt "
          f"(range {np.nanmin(gtot):.4g} .. {np.nanmax(gtot):.4g})")

    # gold_total monotonicity (reward assumes monotone)
    n_dec = 0
    n_pairs = 0
    worst = []
    for m, z in data:
        g = z["gold_total"].copy()
        g[~z["labeled"]] = np.nan
        d = np.diff(g)
        d = d[np.isfinite(d)]
        n_pairs += len(d)
        n_dec += int((d < -1e-6).sum())
        if len(d):
            worst.append((float(d.min()), float(d.max()), m["match_id"]))
    worst.sort()
    print(f"gold_total decreases: {n_dec:,}/{n_pairs:,} ({pct(n_dec, n_pairs):.4f}%) "
          f"-> reward assumes monotone (delta>=0)")
    print(f"  most negative delta: {worst[0]}   largest positive delta: {max(worst, key=lambda t: t[1])}")

    # reward magnitude
    sec("1b. REWARD (solo-gold) MAGNITUDE")
    gs = 1e-3
    rmax, rsum = [], []
    for m, z in data:
        g = z["gold_total"].copy()
        g[~z["labeled"]] = np.nan
        d = np.diff(g) * gs
        d = d[np.isfinite(d)]
        rmax.append(float(np.abs(d).max()) if len(d) else 0.0)
        rsum.append(float(d.sum()))
    rmax, rsum = np.array(rmax), np.array(rsum)
    print(f"per-frame |reward| max over corpus: {rmax.max():.4f} (med per-match {np.median(rmax):.4f})")
    print(f"episode return (sum dense): med {np.median(rsum):.2f} max {rsum.max():.2f} "
          f"(value head uses +-3 symlog buckets)")
    # what a 64-frame window return looks like
    win = []
    for m, z in data:
        g = z["gold_total"].copy()
        g[~z["labeled"]] = np.nan
        d = np.nan_to_num(np.diff(g) * gs)
        c = np.convolve(d, np.ones(64), "valid")
        win.append(c)
    win = np.concatenate(win)
    print(f"64-frame windowed return: mean {win.mean():.5f} p50 {np.percentile(win,50):.5f} "
          f"p99 {np.percentile(win,99):.5f} max {win.max():.4f}  "
          f"frac exactly 0: {pct((win==0).sum(), len(win)):.1f}%")

    # positions
    sec("1c. POSITION SANITY")
    for name, xs, ys, lo, hi in [
        ("champion_screen", "cs_x", "cs_y", (0, 0), (1280, 720)),
        ("cursor.screen", "cur_sx", "cur_sy", (0, 0), (1280, 720)),
        ("champion_world", "cw_x", "cw_y", (-500, -500), (16500, 16500)),
    ]:
        X = np.concatenate([z[xs][z["labeled"]] for _, z in data])
        Y = np.concatenate([z[ys][z["labeled"]] for _, z in data])
        ok = np.isfinite(X) & np.isfinite(Y)
        oob = ok & ((X < lo[0]) | (X > hi[0]) | (Y < lo[1]) | (Y > hi[1]))
        print(f"  {name:16s} present {pct(ok.sum(), len(X)):5.1f}%  "
              f"out-of-bounds {pct(oob.sum(), max(ok.sum(),1)):5.2f}%  "
              f"x[{np.nanmin(X):.0f},{np.nanmax(X):.0f}] y[{np.nanmin(Y):.0f},{np.nanmax(Y):.0f}]")

    # ---------------- 2. movement quantization ----------------
    sec("2. MOVEMENT TARGET: DEAD-BAND + 21-BIN QUANTIZATION")
    all_d = []
    n_frames = n_none = 0
    n_db_kill = n_db_keep = 0
    n_bin_trans = n_bin_hold = 0
    n_kept_but_same_bin = 0
    fallback_frames = 0
    bin_hist = np.zeros((BINS, BINS), dtype=np.int64)
    for m, z in data:
        sw, sh = m["screen_resolution"]
        x, y = z["cur_sx"] / sw, z["cur_sy"] / sh
        T = len(x)
        n_frames += T
        have = np.isfinite(x) & np.isfinite(y)
        n_none += int((~have).sum())
        # replicate _parse_movement
        last = None
        mv = np.full((T, 2), 0.5)
        first_seen = None
        for i in range(T):
            if have[i]:
                nx, ny = x[i], y[i]
                if last is None or abs(nx - last[0]) > DEADBAND or abs(ny - last[1]) > DEADBAND:
                    if last is not None:
                        n_db_keep += 1
                    last = (nx, ny)
                    if first_seen is None:
                        first_seen = i
                else:
                    n_db_kill += 1
                    all_d.append(max(abs(nx - last[0]), abs(ny - last[1])))
            if last is not None:
                mv[i] = last
        fallback_frames += (first_seen or T)
        idx = np.clip(np.round(mv * (BINS - 1)).astype(int), 0, BINS - 1)
        np.add.at(bin_hist, (idx[:, 0], idx[:, 1]), 1)
        trans = (idx[1:] != idx[:-1]).any(1)
        n_bin_trans += int(trans.sum())
        n_bin_hold += int((~trans).sum())
        # command changes that survived the dead-band but land in the same bin
        chg = (np.abs(np.diff(mv, axis=0)) > 1e-12).any(1)
        same_bin = chg & ~trans
        n_kept_but_same_bin += int(same_bin.sum())

    print(f"frames total {n_frames:,}; cursor.screen MISSING on "
          f"{pct(n_none, n_frames):.2f}% of frames")
    print(f"held-forward fallback (0.5,0.5) before first cursor: {fallback_frames:,} frames "
          f"({pct(fallback_frames, n_frames):.2f}%)")
    tot_upd = n_db_kill + n_db_keep
    print(f"cursor reads with a nonzero delta: {tot_upd:,}")
    print(f"  killed by 1% dead-band: {n_db_kill:,} ({pct(n_db_kill, tot_upd):.2f}%)")
    print(f"  passed dead-band:       {n_db_keep:,} ({pct(n_db_keep, tot_upd):.2f}%)")
    print(f"  of the passed ones, landing in the SAME 21-bin cell as before: "
          f"{n_kept_but_same_bin:,} ({pct(n_kept_but_same_bin, n_db_keep):.2f}%)")
    print(f"bin-space transitions: {n_bin_trans:,}/{n_frames:,} = "
          f"{pct(n_bin_trans, n_frames):.2f}% of frames (this is `trans_frac` the gate sees)")
    print(f"  => {pct(n_bin_hold, n_frames):.2f}% of BC movement targets are 'repeat previous bin'")
    print(f"distinct (x,y) bin cells actually used: {(bin_hist>0).sum()}/{BINS*BINS}; "
          f"top-1 cell holds {pct(bin_hist.max(), bin_hist.sum()):.1f}% of frames; "
          f"top-10 cells hold {pct(np.sort(bin_hist.ravel())[-10:].sum(), bin_hist.sum()):.1f}%")
    ent = -(bin_hist / bin_hist.sum() * np.log(np.maximum(bin_hist / bin_hist.sum(), 1e-30))).sum()
    print(f"movement-target entropy: {ent:.3f} nats = {np.exp(ent):.1f} effective cells "
          f"(uniform over 441 = 6.089 nats)")
    if all_d:
        ad = np.array(all_d)
        print(f"dead-banded deltas: p50 {np.median(ad)*100:.3f}% p90 {np.percentile(ad,90)*100:.3f}% "
              f"of screen")
    np.save(CACHE.parent / "audit_bin_hist.npy", bin_hist)

    # bin width in px
    print(f"bin width = 1/{BINS-1} = {100/(BINS-1):.1f}% of screen = "
          f"{1280/(BINS-1):.0f} px in x, {720/(BINS-1):.0f} px in y  <-- ANISOTROPIC")

    # ---------------- 3. attack undercount ----------------
    sec("3. ATTACK LABEL UNDERCOUNT")
    ATT = 2
    n_att_frames = n_att_trans = 0
    runlens = []
    n_lasthit = 0
    n_games = len(data)
    per_game = []
    for m, z in data:
        a = z["atype"]
        att = a == ATT
        n_att_frames += int(att.sum())
        tr = att[1:] & ~att[:-1]
        ntr = int(tr.sum()) + int(att[0])
        n_att_trans += ntr
        # run lengths
        d = np.diff(np.concatenate([[0], att.view(np.int8), [0]]))
        s = np.where(d == 1)[0]
        e = np.where(d == -1)[0]
        runlens.append(e - s)
        # last-hit-like gold events
        g = z["gold_total"].copy()
        g[~z["labeled"]] = np.nan
        dg = np.diff(g)
        lh = np.isfinite(dg) & (dg >= 10) & (dg <= 120)
        n_lasthit += int(lh.sum())
        per_game.append((m["match_id"], ntr, int(att.sum()), int(lh.sum()),
                         float(z["gt"][-1] - z["gt"][0])))
    runlens = np.concatenate(runlens)
    print(f"frames with action.type=='attack': {n_att_frames:,} "
          f"({pct(n_att_frames, n_frames):.2f}% of all frames)")
    print(f"AA labels emitted (state transitions into 'attack'): {n_att_trans:,}")
    print(f"attack-run lengths (frames @20fps): p50 {np.median(runlens):.0f} "
          f"p90 {np.percentile(runlens,90):.0f} max {runlens.max()} mean {runlens.mean():.1f}")
    print(f"  => median attack run = {np.median(runlens)/20:.2f}s of continuous 'attack' state, "
          f"labeled as ONE AA press")
    # Garen AS ~0.625 base -> ~0.72-1.0 by mid game; assume 0.8 attacks/s while in attack state
    for AS in (0.625, 0.8, 1.0):
        est = n_att_frames / 20.0 * AS
        print(f"  at {AS} attacks/s while in attack state -> ~{est:,.0f} true attacks vs "
              f"{n_att_trans:,} labeled  => undercount factor {est/max(n_att_trans,1):.2f}x "
              f"({pct(n_att_trans, est):.1f}% labeled)")
    print(f"minion-kill-sized gold events (delta gold_total in [10,120]): {n_lasthit:,} "
          f"= {n_lasthit/n_games:.0f}/game")
    print(f"  AA labels per game: {n_att_trans/n_games:.0f}; attack-state seconds per game: "
          f"{n_att_frames/20/n_games:.0f}s")
    print(f"  ratio last-hit-gold-events : AA labels = {n_lasthit/max(n_att_trans,1):.2f}")

    # ---------------- 4. cast / ability labels ----------------
    sec("4. ABILITY / CAST LABELS")
    _SPELL = {"GarenQ": "Q", "GarenW": "W", "GarenR": "R", "GarenE": "E",
              "GarenECancel": "E", "SummonerFlash": "Flash", "SummonerDot": "Ignite",
              "recall": "Recall"}
    names = Counter()
    n_cast = n_unmapped = n_notime = n_oor = 0
    key_counts = Counter()
    per_key_frames = Counter()
    collisions = 0
    for m, z in data:
        ct, cn = z["casts_t"], z["casts_name"]
        gt0 = z["gt"][0]
        step = 1.0 / m["fps"]
        T = m["T"]
        occupied = {}
        for t, nm in zip(ct, cn):
            n_cast += 1
            names[nm] += 1
            if not np.isfinite(t):
                n_notime += 1
                continue
            k = _SPELL.get(nm)
            if k is None:
                n_unmapped += 1
                continue
            i = int(round((t - gt0) / step))
            if 0 <= i < T:
                key_counts[k] += 1
                if (k, i) in occupied:
                    collisions += 1
                occupied[(k, i)] = 1
            else:
                n_oor += 1
        per_key_frames["frames"] += T
    print(f"cast events total {n_cast:,}; spell_name histogram (top 20):")
    for nm, c in names.most_common(20):
        mp = _SPELL.get(nm)
        print(f"   {nm:26s} {c:>7,}  -> {mp if mp else 'DROPPED (unmapped)'}")
    print(f"unmapped dropped: {n_unmapped:,} ({pct(n_unmapped, n_cast):.1f}%); "
          f"no-time {n_notime:,}; out-of-range {n_oor:,} ({pct(n_oor, n_cast):.2f}%)")
    print(f"two casts of the same key rounding to the SAME frame (label collision, "
          f"one is lost): {collisions:,}")
    print("per-key positive-label counts and base rate:")
    for k, c in key_counts.most_common():
        print(f"   {k:8s} {c:>7,}  {pct(c, n_frames):.4f}% of frames "
              f"(pos_weight for balance would be {n_frames/max(c,1):.0f})")
    print(f"   AA       {n_att_trans:>7,}  {pct(n_att_trans, n_frames):.4f}% of frames")

    # stride
    n_stride = 0
    for m, z in data:
        lf = z["stride_lf"]
        prev = None
        for i in range(len(lf)):
            v = lf[i]
            if np.isfinite(v) and prev is not None and v > prev + 1e-6:
                n_stride += 1
            if np.isfinite(v):
                prev = v
    print(f"   Stride   {n_stride:>7,}  {pct(n_stride, n_frames):.4f}% of frames")

    # ---------------- 5. action.type distribution ----------------
    sec("5. action.type DISTRIBUTION")
    c = Counter()
    for m, z in data:
        u, n = np.unique(z["atype"], return_counts=True)
        for a, b in zip(u, n):
            c[int(a)] += int(b)
    inv = {0: "none/unlabeled", 1: "idle", 2: "attack", 3: "ability", 4: "recall", 5: "move", 9: "OTHER"}
    for k, v in sorted(c.items(), key=lambda t: -t[1]):
        print(f"   {inv.get(k, k):16s} {v:>10,}  {pct(v, n_frames):5.2f}%")
    print("NB: there is no 'move' state -> moving is indistinguishable from idle in action.type")

    # ---------------- 6. aux state targets ----------------
    sec("6. AUX STATE TARGETS")
    hp = np.concatenate([z["hp"][z["labeled"]] for _, z in data])
    hpm = np.concatenate([z["hp_max"][z["labeled"]] for _, z in data])
    ok = np.isfinite(hp) & np.isfinite(hpm) & (hpm > 0)
    frac = hp[ok] / hpm[ok]
    print(f"own_hp_frac: range [{frac.min():.3f},{frac.max():.3f}]  "
          f">1: {pct((frac>1).sum(), ok.sum()):.3f}%  <0: {pct((frac<0).sum(), ok.sum()):.3f}%")
    lv = np.concatenate([z["level"][z["labeled"]] for _, z in data])
    lv = lv[np.isfinite(lv)]
    print(f"level: min {lv.min()} max {lv.max()} (target = level/18)")
    ohp = np.concatenate([z["opp_hp"] for _, z in data])
    ohpm = np.concatenate([z["opp_hp_max"] for _, z in data])
    seen = np.isfinite(ohp)
    print(f"lane opponent in visible_heroes: {pct(seen.sum(), n_frames):.1f}% of frames "
          f"(enemy_visible target base rate)")
    oscr = np.concatenate([z["opp_screen_ok"] for _, z in data])
    on = np.isfinite(oscr) & (oscr > 0)
    print(f"  ...and has a SCREEN coord (actually on-screen): "
          f"{pct(on.sum(), n_frames):.2f}% of frames")
    print("  NB: `enemy_visible` target = 'entry exists in visible_heroes', NOT 'on screen'.")
    print(f"  entries with no screen coord but counted visible: "
          f"{pct(seen.sum()-on.sum(), max(seen.sum(),1)):.1f}% of 'visible' frames")

    # ---------------- 7. inventory ----------------
    sec("7. MISC")
    ni = np.concatenate([z["n_items"][z["labeled"]] for _, z in data])
    print(f"inventory items held: mean {ni.mean():.2f} max {ni.max()}")
    ad = Counter()
    for m, _ in data:
        for k, v in (m["action_distribution"] or {}).items():
            ad[k] += v
    print("labels.json action_distribution (as recorded):", dict(ad))


if __name__ == "__main__":
    main()
