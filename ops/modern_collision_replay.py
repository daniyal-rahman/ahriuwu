#!/usr/bin/env python3
"""Creep-block and champion-champion overlap statistics from 16.9 (= 26.9) replays.

Two measurements, both from the replay corpus (frames/ is never read):

  A  creep-block proxy for the recorded champion (labels.json + clicks.json):
     2-14 min, alive, idle (no attack/ability/recall within 0.3 s), latest
     move click <= 1 s old and >= 300 u away.  Speed = champion_world chord over
     a 0.25 s (and 0.5 s) forward window, divided by the nominal speed (modal
     straight-line 1 s-window speed of the same life segment between base visits,
     the modern_replay_fidelity "steady" method).  Split in-lane (<= LANE_R of a
     26.19 geometry.json lane path) vs jungle/river (>= JUNGLE_R from any lane)
     and "clean" (no HP loss in the prior 2 s, no alive enemy champion within
     1200 u: a CC/slow proxy -- labels carry no buff data).
     raw_mem.json carries no minion positions, so the minion-conditioned split is
     impossible; lane vs jungle is the control.
  B  champion-champion centre distances (raw_mem.json, all 10 heroes, unique gt):
     alive (hp > 0), neither within 1100 u of a fountain; ally / enemy pairs;
     histogram < 200 u, thresholds, and every < 100 u run (duration, min d, both
     heroes' mean speed during the run, spells seen).

  extract GAME_DIR... --out DIR [--jobs N]   one DIR/<match>.json.gz per game (~1.5 GB RSS)
  analyze --out DIR --json PATH              aggregate + print
"""
import argparse
import gzip
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SCHEMA = "modern_collision_replay/v1"
FOUNTAIN = np.array([[394.0, 461.0], [14340.0, 14391.0]])
GEOM = Path(__file__).resolve().parents[1] / "lanerl_jax/data/modern/26.19/geometry.json"
P = {
    "t_range": (120.0, 840.0),
    "win_frames": (5, 10),          # 0.25 s / 0.5 s at 20 Hz
    "click_max_age_s": 1.0,
    "click_min_dist": 300.0,
    "action_pad_frames": 6,         # 0.3 s either side of a non-idle action
    "lane_r": 800.0,
    "jungle_r": 1200.0,
    "fountain_min": 3000.0,
    "enemy_r": 1200.0,
    "hp_loss_lookback_frames": 40,  # 2 s
    "stall_ratio": 0.6,
    "stall_min_frames": 4,          # >= 0.2 s
    "steady_win_frames": 20,        # 1 s, straightness >= 0.999
    "fountain_radius": 1100.0,
    "close_run_d": 100.0,
    "rear_margin": 600.0,
}
RATIO_BINS = np.linspace(0.0, 2.0, 201)


def _lanes():
    g = json.loads(GEOM.read_text())
    return [np.asarray(v, float) for v in g["lane_paths"].values()]


def dist_to_lanes(pts, lanes):
    best = np.full(len(pts), np.inf)
    for path in lanes:
        a, b = path[:-1], path[1:]
        ab = b - a
        L2 = (ab ** 2).sum(1)
        t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
        proj = a[None] + t[..., None] * ab[None]
        best = np.minimum(best, np.linalg.norm(pts[:, None, :] - proj, axis=-1).min(1))
    return best


def _runs(mask):
    """[(start, end_exclusive)] of True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _mode(hist):
    if sum(hist.values()) < 30:
        return None
    return max(hist, key=lambda k: sum(hist.get(k + d, 0) for d in (-1, 0, 1)))


# ------------------------------------------------------------------------------- A
def pov_arrays(game_dir, lanes):
    with open(os.path.join(game_dir, "labels.json")) as f:
        d = json.load(f)
    champ, team = d["champion"], d.get("team")
    fr = [f for f in d["frames"] if f.get("label") and f["label"].get("champion_world")]
    del d
    n = len(fr)
    gt = np.array([f["gt"] for f in fr])
    pos = np.array([f["label"]["champion_world"][:2] for f in fr], float)
    hp = np.array([(f["label"].get("champion_stats") or {}).get("hp", 1.0) or 0.0 for f in fr])
    act = [((f["label"].get("action") or {}).get("type")) for f in fr]
    mspd = np.array([((f["label"].get("movement") or {}).get("speed") or np.nan) for f in fr], float)
    names = [h["name"] for h in fr[0]["label"]["visible_heroes"]]
    vh = np.full((n, len(names), 3), np.nan)
    for i, f in enumerate(fr):
        for h in f["label"]["visible_heroes"]:
            if h["name"] in names and h.get("world"):
                k = names.index(h["name"])
                vh[i, k] = (h["world"][0], h["world"][1], h.get("hp") or 0.0)
    del fr
    # teams of others: nearest fountain at the first frame
    f0 = vh[0, :, :2]
    side = np.argmin(np.linalg.norm(f0[:, None] - FOUNTAIN[None], axis=-1), 1)
    my_side = side[names.index(champ)] if champ in names else (0 if team == "blue" else 1)
    enemy = side != my_side
    own_f = FOUNTAIN[my_side]

    with open(os.path.join(game_dir, "clicks.json")) as f:
        cl = json.load(f).get("clicks") or []
    ct = np.array([c["game_t"] for c in cl]) if cl else np.zeros(0)
    cxy = np.array([[c["x"], c["z"]] for c in cl]) if cl else np.zeros((0, 2))

    step = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    dtf = np.r_[0.05, np.diff(gt)]
    jump = (step > 300) | (dtf > 0.12)
    nonidle = np.array([a not in ("idle",) for a in act])
    recall = np.array([a == "recall" for a in act])
    pad = P["action_pad_frames"]
    busy = np.convolve(nonidle.astype(float), np.ones(2 * pad + 1), "same") > 0
    alive = hp > 0
    fdist = np.minimum(np.linalg.norm(pos - FOUNTAIN[0], axis=1), np.linalg.norm(pos - FOUNTAIN[1], axis=1))
    ldist = dist_to_lanes(pos, lanes)

    # life segments: split at teleports (recall) and at deaths
    seg = np.cumsum((step > 2000) | (np.r_[False, ~alive[:-1] & alive[1:]]))
    # Homeguard ends on combat: only count a segment after its first attack/ability or HP loss
    combat = (nonidle & ~recall) | np.r_[False, np.diff(hp) < -1.0]
    post_combat = np.zeros(n, bool)
    for sid in np.unique(seg):
        m = np.flatnonzero(seg == sid)
        post_combat[m] = np.cumsum(combat[m]) > 0

    # nominal: modal 1 s straight-window speed per segment, alive & idle & away from fountain
    W = P["steady_win_frames"]
    cum = np.r_[0.0, np.cumsum(step[1:])]
    nominal_seg, game_hist = {}, Counter()
    seg_hist = {}
    for i in range(0, n - W, 2):
        j = i + W
        if jump[i + 1:j + 1].any() or not alive[i:j + 1].all() or busy[i:j + 1].any() or not post_combat[i] \
                or np.linalg.norm(pos[i] - own_f) < P["fountain_min"]:
            continue
        chord = np.linalg.norm(pos[j] - pos[i]); path = cum[j] - cum[i]
        dt = gt[j] - gt[i]
        if path <= 0 or chord / path < 0.999 or dt < 0.9 or chord / dt <= 100:
            continue
        v = round(chord / dt)
        seg_hist.setdefault(seg[i], Counter())[v] += 1
        if P["t_range"][0] <= gt[i] <= P["t_range"][1]:
            game_hist[v] += 1
    game_mode = _mode(game_hist)
    for s, h in seg_hist.items():
        nominal_seg[s] = _mode(h) or game_mode
    nom = np.array([nominal_seg.get(s, game_mode) or np.nan for s in seg], float)

    # latest click
    k = np.searchsorted(ct, gt, side="right") - 1
    has_click = k >= 0
    kk = np.clip(k, 0, max(len(ct) - 1, 0))
    click_age = np.where(has_click, gt - ct[kk] if len(ct) else np.inf, np.inf)
    click_d = np.where(has_click, np.linalg.norm(cxy[kk] - pos, axis=1) if len(ct) else 0, 0)
    # last non-idle action time (for "no action since the click")
    last_busy_t = np.full(n, -np.inf)
    lb = -np.inf
    for i in range(n):
        if nonidle[i]:
            lb = gt[i]
        last_busy_t[i] = lb

    # enemy proximity / hp loss
    epos = vh[:, enemy, :2]; ehp = vh[:, enemy, 2]
    ed = np.linalg.norm(epos - pos[:, None], axis=-1)
    ed = np.where(ehp > 0, ed, np.inf)
    enemy_near = np.nanmin(ed, axis=1) < P["enemy_r"]
    L = P["hp_loss_lookback_frames"]
    drop = np.r_[False, np.diff(hp) < -1.0]
    hp_loss = np.convolve(drop.astype(float), np.ones(L), "full")[:n] > 0

    lo, hi = P["t_range"]
    base_ok = (gt >= lo) & (gt <= hi) & alive & ~busy & ~recall & np.isfinite(nom) \
        & has_click & (click_age <= P["click_max_age_s"]) & (click_d >= P["click_min_dist"]) \
        & (last_busy_t < np.where(has_click, ct[kk] if len(ct) else 0, 0)) \
        & (fdist > P["fountain_min"]) & post_combat
    in_lane = ldist <= P["lane_r"]
    in_jg = ldist >= P["jungle_r"]
    clean = ~hp_loss & ~enemy_near
    rear = in_lane & lane_rear(pos, my_side)

    arr = {"champion": champ, "team": team, "n": n, "gt": gt, "pos": pos, "nom": nom, "mspd": mspd,
           "act": act, "game_mode": game_mode, "nominal_seg": nominal_seg, "speeds": {}, "q": {},
           "cats": {"lane": in_lane, "lane_front": in_lane & ~rear, "lane_rear": rear, "offlane": in_jg},
           "clean": clean}
    jc = np.r_[0, np.cumsum(jump[1:])]
    alive_c = np.r_[0, np.cumsum(~alive)]
    busy_c = np.r_[0, np.cumsum(busy)]
    for w in P["win_frames"]:
        idx = np.arange(n - w)
        ok_w = np.zeros(n, bool)
        chord = np.full(n, np.nan); path = np.full(n, np.nan)
        dt = gt[idx + w] - gt[idx]
        ok_w[idx] = (jc[idx + w] - jc[idx] == 0) & (alive_c[idx + w + 1] - alive_c[idx] == 0) \
            & (busy_c[idx + w + 1] - busy_c[idx] == 0) & (np.abs(dt - 0.05 * w) < 0.02)
        chord[idx] = np.linalg.norm(pos[idx + w] - pos[idx], axis=1) / np.maximum(dt, 1e-6)
        path[idx] = (cum[idx + w] - cum[idx]) / np.maximum(dt, 1e-6)
        arr["speeds"][w] = {"path": path, "chord": chord}
        arr["q"][w] = base_ok & ok_w
    return arr




def lane_rear(pos, my_side):
    """In-lane samples behind the own outer turret by >= rear_margin along the lane path."""
    g = json.loads(GEOM.read_text())
    names = list(g["lane_paths"])
    lane_id = {"bot": 0, "mid": 1, "top": 2}
    out = np.zeros(len(pos), bool)
    best = np.full(len(pos), np.inf)
    for li, nm in enumerate(names):
        path = np.asarray(g["lane_paths"][nm], float)
        a, b = path[:-1], path[1:]
        ab = b - a; L = np.linalg.norm(ab, axis=1); s0 = np.r_[0, np.cumsum(L)][:-1]
        t = np.clip(((pos[:, None] - a[None]) * ab[None]).sum(-1) / (L ** 2)[None], 0, 1)
        proj = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(pos[:, None] - proj, axis=-1)
        k = d.argmin(1); dm = d[np.arange(len(pos)), k]
        s = s0[k] + t[np.arange(len(pos)), k] * L[k]
        tur = [tt["position"] for tt in g["turrets"] if tt["team"] == my_side and tt["tier"] == "outer"
               and tt["lane"] == lane_id[nm]][0]
        tt_ = np.asarray(tur, float)
        # turret arc length: project turret onto the path
        tt_t = np.clip(((tt_[None] - a) * ab).sum(-1) / L ** 2, 0, 1)
        tt_d = np.linalg.norm(tt_[None] - (a + tt_t[:, None] * ab), axis=1)
        j = tt_d.argmin(); s_t = s0[j] + tt_t[j] * L[j]
        rear = (s < s_t - P["rear_margin"]) if my_side == 0 else (s > s_t + P["rear_margin"])
        better = dm < best
        best = np.where(better, dm, best); out = np.where(better, rear, out)
    return out


def extract_pov(game_dir, lanes):
    arr = pov_arrays(game_dir, lanes)
    out = {"champion": arr["champion"], "team": arr["team"], "n_frames": arr["n"],
           "game_mode_speed": arr["game_mode"],
           "nominal_segments": {int(s): v for s, v in arr["nominal_seg"].items()}, "windows": {}}
    nom, mspd = arr["nom"], arr["mspd"]
    for w in P["win_frames"]:
        q = arr["q"][w]
        res = {}
        for cbase, cm in arr["cats"].items():
            for suffix, extra in (("", True), ("_clean", arr["clean"])):
                m = q & cm & extra
                r_entry = {"n": int(m.sum()), "seconds": float(m.sum() * 0.05)}
                for kind in ("path", "chord"):
                    sp = arr["speeds"][w][kind]
                    ratio = sp / nom
                    stalls = []
                    for a, b in _runs(m):
                        for s0, s1 in _runs(ratio[a:b] < P["stall_ratio"]):
                            if s1 - s0 >= P["stall_min_frames"]:
                                stalls.append(round((s1 - s0) * 0.05, 3))
                    r_entry[kind] = {"hist": np.histogram(np.clip(ratio[m], 0, 1.999), RATIO_BINS)[0].tolist(),
                                     "stalls": stalls}
                r_entry["mspeed_hist"] = np.histogram(np.clip(mspd[m] / nom[m], 0, 1.999), RATIO_BINS)[0].tolist()
                sp = arr["speeds"][w]["path"]
                r_entry["mspeed_over_path_median"] = float(np.nanmedian(mspd[m] / np.maximum(sp[m], 1))) \
                    if m.any() else None
                res[cbase + suffix] = r_entry
        out["windows"][str(w)] = res
    return out


# ------------------------------------------------------------------------------- B
def extract_pairs(game_dir):
    with open(os.path.join(game_dir, "raw_mem.json")) as f:
        samples = json.load(f)
    names = sorted({h for s in samples[:50] for h in s.get("heroes", {})})
    rows, last = [], None
    for s in samples:
        g = s.get("gt")
        if g is None or g == last:
            continue
        last = g
        r = [g]
        for h in names:
            v = s["heroes"].get(h)
            if v and v.get("pos") and v.get("hp") is not None:
                r += [v["pos"][0], v["pos"][-1], v["hp"], 1.0 if v.get("spell") else 0.0]
                if v.get("spell"):
                    r.append(v["spell"])
            else:
                r += [np.nan, np.nan, np.nan, 0.0]
        rows.append(r)
    del samples
    # split spell strings out
    T = len(rows)
    H = len(names)
    gt = np.empty(T); X = np.full((T, H, 2), np.nan); HP = np.full((T, H), np.nan)
    spell = [[None] * H for _ in range(T)]
    for t, r in enumerate(rows):
        gt[t] = r[0]; k = 1
        for h in range(H):
            X[t, h] = r[k], r[k + 1]; HP[t, h] = r[k + 2]; has = r[k + 3]; k += 4
            if has:
                spell[t][h] = r[k]; k += 1
    del rows
    side = np.argmin(np.linalg.norm(X[0][:, None] - FOUNTAIN[None], axis=-1), 1)
    fd = np.min(np.linalg.norm(X[:, :, None] - FOUNTAIN[None, None], axis=-1), axis=-1)
    valid = (HP > 0) & (fd > P["fountain_radius"]) & np.isfinite(X[..., 0])
    dtt = np.r_[np.diff(gt), 0.0]
    vel = np.full((T, H), np.nan)
    vel[1:] = np.linalg.norm(np.diff(X, axis=0), axis=-1) / np.maximum(np.diff(gt), 1e-3)[:, None]

    edges = np.arange(0, 201, 1.0)
    thr = (130, 113, 100, 65, 30)
    out = {"heroes": names, "side": side.tolist(), "n_samples": T,
           "gt_span": [float(gt[0]), float(gt[-1])], "pairs": {}}
    agg = {rel: {"n": 0, "secs": 0.0, "hist": np.zeros(200, np.int64), "lt": {t: 0 for t in thr},
                 "lt_secs": {t: 0.0 for t in thr}, "min": None, "runs": []} for rel in ("ally", "enemy")}
    for i in range(H):
        for j in range(i + 1, H):
            rel = "ally" if side[i] == side[j] else "enemy"
            A = agg[rel]
            v = valid[:, i] & valid[:, j]
            d = np.linalg.norm(X[:, i] - X[:, j], axis=-1)
            dv = d[v]
            A["n"] += int(v.sum()); A["secs"] += float(dtt[v].sum())
            A["hist"] += np.histogram(dv, edges)[0]
            for t in thr:
                A["lt"][t] += int((dv < t).sum()); A["lt_secs"][t] += float(dtt[v & (d < t)].sum())
            if dv.size:
                m = float(dv.min())
                A["min"] = m if A["min"] is None else min(A["min"], m)
            for a, b in _runs(v & (d < P["close_run_d"])):
                seg = slice(a, b)
                k = a + int(np.argmin(d[seg]))
                sp_i = [s for s in {spell[t][i] for t in range(max(0, a - 10), b)} if s]
                sp_j = [s for s in {spell[t][j] for t in range(max(0, a - 10), b)} if s]
                A["runs"].append({"a": names[i], "b": names[j], "gt": round(float(gt[a]), 3),
                                  "dur": round(float(gt[b - 1] - gt[a] + dtt[b - 1]), 3),
                                  "n": int(b - a), "dmin": round(float(d[k]), 1),
                                  "dmed": round(float(np.median(d[seg])), 1),
                                  "va": round(float(np.nanmedian(vel[seg, i])), 1),
                                  "vb": round(float(np.nanmedian(vel[seg, j])), 1),
                                  "spells": sp_i + sp_j})
    for rel, A in agg.items():
        A["hist"] = A["hist"].tolist()
        A["lt"] = {str(k): v for k, v in A["lt"].items()}
        A["lt_secs"] = {str(k): v for k, v in A["lt_secs"].items()}
    out["pairs"] = agg
    return out


def extract_game(game_dir, lanes):
    gid = os.path.basename(os.path.normpath(game_dir))
    return {"schema": SCHEMA, "game_id": gid, "params": P,
            "pov": extract_pov(game_dir, lanes), "pairs": extract_pairs(game_dir)}


def _extract_one(args):
    game_dir, out_dir = args
    gid = os.path.basename(os.path.normpath(game_dir))
    dst = os.path.join(out_dir, gid + ".json.gz")
    if os.path.exists(dst):
        return gid, "cached"
    try:
        res = extract_game(game_dir, _lanes())
    except Exception as e:
        import traceback
        return gid, "error: %r %s" % (e, traceback.format_exc()[-400:])
    tmp = dst + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(res, f, separators=(",", ":"))
    os.replace(tmp, dst)
    return gid, "ok"


def cmd_extract(a):
    os.makedirs(a.out, exist_ok=True)
    jobs = [(g, a.out) for g in a.games
            if all(os.path.isfile(os.path.join(g, x)) for x in ("raw_mem.json", "labels.json", "clicks.json"))]
    if a.jobs > 1:
        from multiprocessing import Pool
        with Pool(a.jobs, maxtasksperchild=1) as pool:
            for gid, st in pool.imap_unordered(_extract_one, jobs):
                print(gid, st, flush=True)
    else:
        for j in jobs:
            print(*_extract_one(j), flush=True)


# ------------------------------------------------------------------------------- analyze
def _pct_from_hist(h, qs):
    h = np.asarray(h, float)
    c = np.cumsum(h)
    if c[-1] == 0:
        return {f"p{q}": None for q in qs}
    centers = (RATIO_BINS[:-1] + RATIO_BINS[1:]) / 2
    return {f"p{q}": round(float(centers[np.searchsorted(c, q / 100 * c[-1])]), 3) for q in qs}


def cmd_analyze(a):
    games = []
    for n in sorted(os.listdir(a.out)):
        if n.endswith(".json.gz"):
            with gzip.open(os.path.join(a.out, n), "rt") as f:
                games.append(json.load(f))
    res = {"n_games": len(games), "params": P}
    print(f"games: {len(games)}")
    qs = (1, 5, 10, 25, 50)
    res["A"] = {}
    cats = [c + s for c in ("lane", "lane_front", "lane_rear", "offlane") for s in ("", "_clean")]
    for w in map(str, P["win_frames"]):
        res["A"][w] = {}
        for cat in cats:
            for kind in ("path", "chord"):
                H = np.zeros(200); MH = np.zeros(200); secs = 0.0; stalls = []; mom = []
                for g in games:
                    r = g["pov"]["windows"][w][cat]
                    H += r[kind]["hist"]; MH += r["mspeed_hist"]; secs += r["seconds"]
                    stalls += r[kind]["stalls"]
                    if r["mspeed_over_path_median"] is not None:
                        mom.append(r["mspeed_over_path_median"])
                st = np.asarray(stalls)
                row = {"minutes": round(secs / 60, 1), **_pct_from_hist(H, qs),
                       "frac_below_0p6": round(float(H[:60].sum() / max(H.sum(), 1)), 4),
                       "frac_below_0p9": round(float(H[:90].sum() / max(H.sum(), 1)), 4),
                       "stalls": int(st.size),
                       "stalls_per_min": round(float(st.size / (secs / 60)), 3) if secs else None,
                       "stall_dur_median": float(np.median(st)) if st.size else None,
                       "stall_dur_p90": round(float(np.percentile(st, 90)), 3) if st.size else None}
                if kind == "path":
                    row["movement_speed_field_ratio_pcts"] = _pct_from_hist(MH, qs)
                    row["movement_speed_field_over_path_speed_median_of_games"] = \
                        float(np.median(mom)) if mom else None
                res["A"][w].setdefault(cat, {})[kind] = row
                print(f"A w={w:2s} {kind:5s} {cat:18s} {row['minutes']:7.1f} min  "
                      + " ".join(f"{k}={v}" for k, v in row.items() if k.startswith("p") and k[1:].isdigit())
                      + f"  <0.6:{row['frac_below_0p6']} stalls/min {row['stalls_per_min']} "
                        f"dur med {row['stall_dur_median']} p90 {row['stall_dur_p90']}")
    nom = [(g["pov"]["champion"], g["pov"]["game_mode_speed"]) for g in games]
    res["A"]["nominal_game_modes"] = nom
    res["A"]["n_games_with_nominal"] = sum(1 for _, v in nom if v)

    res["B"] = {}
    for rel in ("ally", "enemy"):
        n = sum(g["pairs"]["pairs"][rel]["n"] for g in games)
        secs = sum(g["pairs"]["pairs"][rel]["secs"] for g in games)
        hist = np.sum([g["pairs"]["pairs"][rel]["hist"] for g in games], axis=0)
        lt = {t: sum(g["pairs"]["pairs"][rel]["lt"][t] for g in games) for t in games[0]["pairs"]["pairs"][rel]["lt"]}
        mins = [g["pairs"]["pairs"][rel]["min"] for g in games if g["pairs"]["pairs"][rel]["min"] is not None]
        runs = [dict(r, game=g["game_id"]) for g in games for r in g["pairs"]["pairs"][rel]["runs"]]
        dur = np.asarray([r["dur"] for r in runs])
        r65 = [r for r in runs if r["dmin"] < 65]
        d65 = np.asarray([r["dur"] for r in r65]) if r65 else np.zeros(0)
        still65 = [r for r in r65 if r["va"] < 30 and r["vb"] < 30 and r["dur"] >= 0.5]
        row = {"pair_samples": n, "pair_seconds": round(secs, 1), "min_distance": min(mins) if mins else None,
               "frac_lt": {t: v / n for t, v in lt.items()},
               "hist_10u_below_200": [int(hist[i:i + 10].sum()) for i in range(0, 200, 10)],
               "runs_lt100": int(dur.size),
               "runs_lt100_dur_median": float(np.median(dur)) if dur.size else None,
               "runs_lt100_dur_p90": float(np.percentile(dur, 90)) if dur.size else None,
               "runs_lt100_ge1s": int((dur >= 1).sum()),
               "runs_dmin_lt65": len(r65),
               "runs_dmin_lt65_dur_median": float(np.median(d65)) if d65.size else None,
               "runs_dmin_lt65_ge1s": int((d65 >= 1).sum()),
               "runs_dmin_lt65_both_still_ge0p5s": len(still65),
               "closest": sorted(runs, key=lambda r: r["dmin"])[:15],
               "longest_lt65": sorted(r65, key=lambda r: -r["dur"])[:10],
               "champs_in_lt30_runs": Counter(c for r in runs if r["dmin"] < 30
                                              for c in (r["a"], r["b"])).most_common(15),
               "spells_in_lt30_runs": Counter(s for r in runs if r["dmin"] < 30
                                              for s in r["spells"]).most_common(15)}
        res["B"][rel] = row
        print(f"B {rel}: pair-samples {n} ({secs/3600:.1f} pair-h) min {row['min_distance']:.1f}  "
              + " ".join(f"<{t}:{v:.2e}" for t, v in row["frac_lt"].items())
              + f"  runs<100 {dur.size} (med {row['runs_lt100_dur_median']} s, p90 {row['runs_lt100_dur_p90']}, "
                f">=1s {row['runs_lt100_ge1s']}); dmin<65 runs {len(r65)} (>=1s {row['runs_dmin_lt65_ge1s']}, "
                f"both still>=0.5s {len(still65)})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("games", nargs="+")
    e.add_argument("--out", required=True)
    e.add_argument("--jobs", type=int, default=1)
    n = sub.add_parser("analyze")
    n.add_argument("--out", required=True)
    n.add_argument("--json")
    a = ap.parse_args(argv)
    return {"extract": cmd_extract, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(0 if main() is not None or True else 1)
