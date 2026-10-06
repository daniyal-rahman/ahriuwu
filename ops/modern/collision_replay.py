#!/usr/bin/env python3
"""Creep-block and champion-champion overlap statistics from 16.9 (= 26.9) replays (docs/modern/COLLISION.md).

  A  creep-block proxy for the recorded champion (labels.json + clicks.json): 2-14 min, alive, idle (no
     attack/ability/recall within 0.3 s), latest move click <= 1 s old and >= 300 u away. Speed = path or
     chord speed over a 0.25 s / 0.5 s forward window / nominal speed (modal straight 1 s-window speed of the
     life segment, the replay_fidelity "steady" method). Split in-lane (<= lane_r of a 26.19 lane path, front
     / behind the own outer turret) vs off-lane (>= jungle_r), and "clean" (no HP loss in the prior 2 s, no
     live enemy champion within 1200 u: a CC/slow proxy). raw_mem.json has no minion positions, so lane vs
     jungle is the control.
  B  champion-champion centre distances (raw_mem.json, all 10 heroes): alive, neither within 1100 u of a
     fountain; ally / enemy pairs; histogram < 200 u, thresholds and every < 100 u run (duration, min d,
     both heroes' speed, spells seen).

  extract GAME_DIR... --out DIR [--jobs N]   one DIR/<match>.json.gz per game (~1.5 GB RSS)
  analyze --out DIR --json PATH              aggregate + print
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ops.modern.replay_fidelity import REPO, corpus_main  # noqa: E402

SCHEMA = "ops.modern.collision_replay/v1"
FOUNTAIN = np.array([[394.0, 461.0], [14340.0, 14391.0]])
GEOM = REPO / "lanerl_jax/modern/data/26.19/geometry.json"
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
LANE_ID = {"bot": 0, "mid": 1, "top": 2}


def project(pts, path):
    """Per point: distance to the polyline ``path`` and arc length of the closest point on it."""
    a, ab = path[:-1], np.diff(path, axis=0)
    L = np.linalg.norm(ab, axis=1)
    t = np.clip(((pts[:, None] - a[None]) * ab[None]).sum(-1) / (L ** 2)[None], 0, 1)
    d = np.linalg.norm(pts[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1)
    k, i = d.argmin(1), np.arange(len(pts))
    return d[i, k], np.r_[0, np.cumsum(L)][:-1][k] + t[i, k] * L[k]


def lane_position(pos, my_side, geom):
    """``(distance to the nearest lane, in-lane and >= rear_margin behind the own outer turret)``."""
    best, rear = np.full(len(pos), np.inf), np.zeros(len(pos), bool)
    for name, path in geom["lane_paths"].items():
        path = np.asarray(path, float)
        d, s = project(pos, path)
        tur = next(np.asarray(t["position"], float) for t in geom["turrets"]
                   if t["team"] == my_side and t["tier"] == "outer" and t["lane"] == LANE_ID[name])
        s_t = project(tur[None], path)[1][0]
        behind = (s < s_t - P["rear_margin"]) if my_side == 0 else (s > s_t + P["rear_margin"])
        rear = np.where(d < best, behind, rear)
        best = np.minimum(best, d)
    return best, rear


def _runs(mask):
    """[(start, end_exclusive)] of True runs."""
    d = np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _mode(hist):
    if sum(hist.values()) < 30:
        return None
    return max(hist, key=lambda k: sum(hist.get(k + d, 0) for d in (-1, 0, 1)))


# ------------------------------------------------------------------------------- A
def load_pov(game_dir):
    with open(os.path.join(game_dir, "labels.json")) as f:
        d = json.load(f)
    fr = [f["label"] for f in d["frames"] if f.get("label") and f["label"].get("champion_world")]
    gt = np.array([f["gt"] for f in d["frames"] if f.get("label") and f["label"].get("champion_world")])
    names = [h["name"] for h in fr[0]["visible_heroes"]]
    vh = np.full((len(fr), len(names), 3), np.nan)
    for i, f in enumerate(fr):
        for h in f["visible_heroes"]:
            if h["name"] in names and h.get("world"):
                vh[i, names.index(h["name"])] = (h["world"][0], h["world"][1], h.get("hp") or 0.0)
    with open(os.path.join(game_dir, "clicks.json")) as f:
        cl = json.load(f).get("clicks") or []
    return dict(champ=d["champion"], team=d.get("team"), gt=gt, names=names, vh=vh,
                pos=np.array([f["champion_world"][:2] for f in fr], float),
                hp=np.array([(f.get("champion_stats") or {}).get("hp", 1.0) or 0.0 for f in fr]),
                act=[(f.get("action") or {}).get("type") for f in fr],
                mspd=np.array([((f.get("movement") or {}).get("speed") or np.nan) for f in fr], float),
                ct=np.array([c["game_t"] for c in cl]) if cl else np.zeros(0),
                cxy=np.array([[c["x"], c["z"]] for c in cl]) if cl else np.zeros((0, 2)))


def pov_arrays(game_dir, geom):
    v = load_pov(game_dir)
    champ, team, gt, pos, hp, act, ct, cxy = (v[k] for k in ("champ", "team", "gt", "pos", "hp", "act", "ct", "cxy"))
    n, vh, names = len(gt), v["vh"], v["names"]
    side = np.argmin(np.linalg.norm(vh[0, :, :2][:, None] - FOUNTAIN[None], axis=-1), 1)   # nearest fountain
    my_side = side[names.index(champ)] if champ in names else (0 if team == "blue" else 1)
    own_f = FOUNTAIN[my_side]

    step = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    jump = (step > 300) | (np.r_[0.05, np.diff(gt)] > 0.12)
    nonidle = np.array([a not in ("idle",) for a in act])
    recall = np.array([a == "recall" for a in act])
    pad = P["action_pad_frames"]
    busy = np.convolve(nonidle.astype(float), np.ones(2 * pad + 1), "same") > 0
    alive = hp > 0
    fdist = np.minimum(np.linalg.norm(pos - FOUNTAIN[0], axis=1), np.linalg.norm(pos - FOUNTAIN[1], axis=1))
    ldist, behind = lane_position(pos, my_side, geom)

    # Life segments split at teleports and deaths. Homeguard ends on combat: a segment counts only after its
    # first attack/ability or HP loss.
    seg = np.cumsum((step > 2000) | (np.r_[False, ~alive[:-1] & alive[1:]]))
    combat = (nonidle & ~recall) | np.r_[False, np.diff(hp) < -1.0]
    post_combat = np.zeros(n, bool)
    for sid in np.unique(seg):
        m = np.flatnonzero(seg == sid)
        post_combat[m] = np.cumsum(combat[m]) > 0

    # Nominal speed: modal 1 s straight-window speed per segment, alive, idle, away from the own fountain.
    W = P["steady_win_frames"]
    cum = np.r_[0.0, np.cumsum(step[1:])]
    game_hist, seg_hist = Counter(), {}
    for i in range(0, n - W, 2):
        j = i + W
        if jump[i + 1:j + 1].any() or not alive[i:j + 1].all() or busy[i:j + 1].any() or not post_combat[i] \
                or np.linalg.norm(pos[i] - own_f) < P["fountain_min"]:
            continue
        chord, path, dt = np.linalg.norm(pos[j] - pos[i]), cum[j] - cum[i], gt[j] - gt[i]
        if path <= 0 or chord / path < 0.999 or dt < 0.9 or chord / dt <= 100:
            continue
        seg_hist.setdefault(seg[i], Counter())[round(chord / dt)] += 1
        if P["t_range"][0] <= gt[i] <= P["t_range"][1]:
            game_hist[round(chord / dt)] += 1
    game_mode = _mode(game_hist)
    nominal_seg = {s: _mode(h) or game_mode for s, h in seg_hist.items()}
    nom = np.array([nominal_seg.get(s, game_mode) or np.nan for s in seg], float)

    # Latest click, and the last non-idle action ("no action since the click").
    k = np.searchsorted(ct, gt, side="right") - 1
    has_click = k >= 0
    kk = np.clip(k, 0, max(len(ct) - 1, 0))
    click_age = np.where(has_click, gt - ct[kk] if len(ct) else np.inf, np.inf)
    click_d = np.where(has_click, np.linalg.norm(cxy[kk] - pos, axis=1) if len(ct) else 0, 0)
    last = np.maximum.accumulate(np.where(nonidle, np.arange(n), -1))
    last_busy_t = np.where(last >= 0, gt[np.maximum(last, 0)], -np.inf)

    enemy = side != my_side
    ed = np.where(vh[:, enemy, 2] > 0, np.linalg.norm(vh[:, enemy, :2] - pos[:, None], axis=-1), np.inf)
    enemy_near = np.nanmin(ed, axis=1) < P["enemy_r"]
    drop = np.r_[False, np.diff(hp) < -1.0]
    hp_loss = np.convolve(drop.astype(float), np.ones(P["hp_loss_lookback_frames"]), "full")[:n] > 0

    lo, hi = P["t_range"]
    base_ok = (gt >= lo) & (gt <= hi) & alive & ~busy & ~recall & np.isfinite(nom) \
        & has_click & (click_age <= P["click_max_age_s"]) & (click_d >= P["click_min_dist"]) \
        & (last_busy_t < np.where(has_click, ct[kk] if len(ct) else 0, 0)) \
        & (fdist > P["fountain_min"]) & post_combat
    in_lane = ldist <= P["lane_r"]
    rear = in_lane & behind
    arr = {"champion": champ, "team": team, "n": n, "nom": nom, "mspd": v["mspd"], "game_mode": game_mode,
           "nominal_seg": nominal_seg, "speeds": {}, "q": {}, "clean": ~hp_loss & ~enemy_near,
           "cats": {"lane": in_lane, "lane_front": in_lane & ~rear, "lane_rear": rear,
                    "offlane": ldist >= P["jungle_r"]}}
    jc, alive_c, busy_c = np.r_[0, np.cumsum(jump[1:])], np.r_[0, np.cumsum(~alive)], np.r_[0, np.cumsum(busy)]
    for w in P["win_frames"]:
        idx = np.arange(n - w)
        ok_w = np.zeros(n, bool)
        chord, path = np.full(n, np.nan), np.full(n, np.nan)
        dt = gt[idx + w] - gt[idx]
        ok_w[idx] = (jc[idx + w] - jc[idx] == 0) & (alive_c[idx + w + 1] - alive_c[idx] == 0) \
            & (busy_c[idx + w + 1] - busy_c[idx] == 0) & (np.abs(dt - 0.05 * w) < 0.02)
        chord[idx] = np.linalg.norm(pos[idx + w] - pos[idx], axis=1) / np.maximum(dt, 1e-6)
        path[idx] = (cum[idx + w] - cum[idx]) / np.maximum(dt, 1e-6)
        arr["speeds"][w] = {"path": path, "chord": chord}
        arr["q"][w] = base_ok & ok_w
    return arr


def extract_pov(game_dir, geom):
    arr = pov_arrays(game_dir, geom)
    out = {"champion": arr["champion"], "team": arr["team"], "n_frames": arr["n"],
           "game_mode_speed": arr["game_mode"],
           "nominal_segments": {int(s): v for s, v in arr["nominal_seg"].items()}, "windows": {}}
    nom, mspd = arr["nom"], arr["mspd"]
    for w in P["win_frames"]:
        res = {}
        for cbase, cm in arr["cats"].items():
            for suffix, extra in (("", True), ("_clean", arr["clean"])):
                m = arr["q"][w] & cm & extra
                entry = {"n": int(m.sum()), "seconds": float(m.sum() * 0.05)}
                for kind in ("path", "chord"):
                    ratio = arr["speeds"][w][kind] / nom
                    stalls = [round((s1 - s0) * 0.05, 3) for a, b in _runs(m)
                              for s0, s1 in _runs(ratio[a:b] < P["stall_ratio"]) if s1 - s0 >= P["stall_min_frames"]]
                    entry[kind] = {"hist": np.histogram(np.clip(ratio[m], 0, 1.999), RATIO_BINS)[0].tolist(),
                                   "stalls": stalls}
                entry["mspeed_hist"] = np.histogram(np.clip(mspd[m] / nom[m], 0, 1.999), RATIO_BINS)[0].tolist()
                sp = arr["speeds"][w]["path"]
                entry["mspeed_over_path_median"] = float(np.nanmedian(mspd[m] / np.maximum(sp[m], 1))) \
                    if m.any() else None
                res[cbase + suffix] = entry
        out["windows"][str(w)] = res
    return out


# ------------------------------------------------------------------------------- B
def extract_pairs(game_dir):
    with open(os.path.join(game_dir, "raw_mem.json")) as f:
        samples = json.load(f)
    names = sorted({h for s in samples[:50] for h in s.get("heroes", {})})
    uniq, last = [], None
    for s in samples:
        if s.get("gt") is not None and s["gt"] != last:
            last = s["gt"]
            uniq.append(s)
    del samples
    T, H = len(uniq), len(names)
    gt, X, HP = np.empty(T), np.full((T, H, 2), np.nan), np.full((T, H), np.nan)
    spell = [[None] * H for _ in range(T)]
    for t, s in enumerate(uniq):
        gt[t] = s["gt"]
        for h, name in enumerate(names):
            v = s["heroes"].get(name)
            if v and v.get("pos") and v.get("hp") is not None:
                X[t, h], HP[t, h] = (v["pos"][0], v["pos"][-1]), v["hp"]
                spell[t][h] = v.get("spell") or None
    side = np.argmin(np.linalg.norm(X[0][:, None] - FOUNTAIN[None], axis=-1), 1)
    fd = np.min(np.linalg.norm(X[:, :, None] - FOUNTAIN[None, None], axis=-1), axis=-1)
    valid = (HP > 0) & (fd > P["fountain_radius"]) & np.isfinite(X[..., 0])
    dtt = np.r_[np.diff(gt), 0.0]
    vel = np.full((T, H), np.nan)
    vel[1:] = np.linalg.norm(np.diff(X, axis=0), axis=-1) / np.maximum(np.diff(gt), 1e-3)[:, None]

    thr = (130, 113, 100, 65, 30)
    agg = {rel: {"n": 0, "secs": 0.0, "hist": np.zeros(200, np.int64), "lt": {t: 0 for t in thr},
                 "lt_secs": {t: 0.0 for t in thr}, "min": None, "runs": []} for rel in ("ally", "enemy")}
    for i in range(H):
        for j in range(i + 1, H):
            A = agg["ally" if side[i] == side[j] else "enemy"]
            v = valid[:, i] & valid[:, j]
            d = np.linalg.norm(X[:, i] - X[:, j], axis=-1)
            dv = d[v]
            A["n"] += int(v.sum())
            A["secs"] += float(dtt[v].sum())
            A["hist"] += np.histogram(dv, np.arange(0, 201, 1.0))[0]
            for t in thr:
                A["lt"][t] += int((dv < t).sum())
                A["lt_secs"][t] += float(dtt[v & (d < t)].sum())
            if dv.size:
                A["min"] = float(dv.min()) if A["min"] is None else min(A["min"], float(dv.min()))
            for a, b in _runs(v & (d < P["close_run_d"])):
                seg = slice(a, b)
                k = a + int(np.argmin(d[seg]))
                seen = [s for h in (i, j) for s in {spell[t][h] for t in range(max(0, a - 10), b)} if s]
                A["runs"].append({"a": names[i], "b": names[j], "gt": round(float(gt[a]), 3),
                                  "dur": round(float(gt[b - 1] - gt[a] + dtt[b - 1]), 3),
                                  "n": int(b - a), "dmin": round(float(d[k]), 1),
                                  "dmed": round(float(np.median(d[seg])), 1),
                                  "va": round(float(np.nanmedian(vel[seg, i])), 1),
                                  "vb": round(float(np.nanmedian(vel[seg, j])), 1), "spells": seen})
    for A in agg.values():
        A["hist"] = A["hist"].tolist()
        A["lt"] = {str(k): v for k, v in A["lt"].items()}
        A["lt_secs"] = {str(k): v for k, v in A["lt_secs"].items()}
    return {"heroes": names, "side": side.tolist(), "n_samples": T,
            "gt_span": [float(gt[0]), float(gt[-1])], "pairs": agg}


def extract_game(game_dir):
    geom = json.loads(GEOM.read_text())
    return {"schema": SCHEMA, "game_id": os.path.basename(os.path.normpath(game_dir)), "params": P,
            "pov": extract_pov(game_dir, geom), "pairs": extract_pairs(game_dir)}


# ------------------------------------------------------------------------------- analyze
def _pct_from_hist(h, qs):
    c = np.cumsum(np.asarray(h, float))
    if c[-1] == 0:
        return {f"p{q}": None for q in qs}
    centers = (RATIO_BINS[:-1] + RATIO_BINS[1:]) / 2
    return {f"p{q}": round(float(centers[np.searchsorted(c, q / 100 * c[-1])]), 3) for q in qs}


def _med(x):
    return float(np.median(x)) if len(x) else None


def analyze(games):
    res = {"n_games": len(games), "params": P, "A": {}, "B": {}}
    print(f"games: {len(games)}")
    qs = (1, 5, 10, 25, 50)
    cats = [c + s for c in ("lane", "lane_front", "lane_rear", "offlane") for s in ("", "_clean")]
    for w in map(str, P["win_frames"]):
        res["A"][w] = {}
        for cat in cats:
            for kind in ("path", "chord"):
                rs = [g["pov"]["windows"][w][cat] for g in games]
                H = np.zeros(200) + np.sum([r[kind]["hist"] for r in rs], axis=0)
                secs = sum(r["seconds"] for r in rs)
                st = np.asarray([s for r in rs for s in r[kind]["stalls"]])
                row = {"minutes": round(secs / 60, 1), **_pct_from_hist(H, qs),
                       "frac_below_0p6": round(float(H[:60].sum() / max(H.sum(), 1)), 4),
                       "frac_below_0p9": round(float(H[:90].sum() / max(H.sum(), 1)), 4),
                       "stalls": int(st.size),
                       "stalls_per_min": round(float(st.size / (secs / 60)), 3) if secs else None,
                       "stall_dur_median": _med(st),
                       "stall_dur_p90": round(float(np.percentile(st, 90)), 3) if st.size else None}
                if kind == "path":
                    MH = np.zeros(200) + np.sum([r["mspeed_hist"] for r in rs], axis=0)
                    row["movement_speed_field_ratio_pcts"] = _pct_from_hist(MH, qs)
                    row["movement_speed_field_over_path_speed_median_of_games"] = _med(
                        [r["mspeed_over_path_median"] for r in rs if r["mspeed_over_path_median"] is not None])
                res["A"][w].setdefault(cat, {})[kind] = row
                print(f"A w={w:2s} {kind:5s} {cat:18s} {row['minutes']:7.1f} min  "
                      + " ".join(f"{k}={v}" for k, v in row.items() if k.startswith("p") and k[1:].isdigit())
                      + f"  <0.6:{row['frac_below_0p6']} stalls/min {row['stalls_per_min']} "
                        f"dur med {row['stall_dur_median']} p90 {row['stall_dur_p90']}")
    nom = [(g["pov"]["champion"], g["pov"]["game_mode_speed"]) for g in games]
    res["A"]["nominal_game_modes"] = nom
    res["A"]["n_games_with_nominal"] = sum(1 for _, v in nom if v)

    for rel in ("ally", "enemy"):
        ps = [g["pairs"]["pairs"][rel] for g in games]
        n, secs = sum(p["n"] for p in ps), sum(p["secs"] for p in ps)
        hist = np.sum([p["hist"] for p in ps], axis=0)
        lt = {t: sum(p["lt"][t] for p in ps) for t in ps[0]["lt"]}
        mins = [p["min"] for p in ps if p["min"] is not None]
        runs = [dict(r, game=g["game_id"]) for g in games for r in g["pairs"]["pairs"][rel]["runs"]]
        dur = np.asarray([r["dur"] for r in runs])
        r65 = [r for r in runs if r["dmin"] < 65]
        d65 = np.asarray([r["dur"] for r in r65])
        still65 = [r for r in r65 if r["va"] < 30 and r["vb"] < 30 and r["dur"] >= 0.5]
        row = {"pair_samples": n, "pair_seconds": round(secs, 1), "min_distance": min(mins) if mins else None,
               "frac_lt": {t: v / n for t, v in lt.items()},
               "hist_10u_below_200": [int(hist[i:i + 10].sum()) for i in range(0, 200, 10)],
               "runs_lt100": int(dur.size),
               "runs_lt100_dur_median": _med(dur),
               "runs_lt100_dur_p90": float(np.percentile(dur, 90)) if dur.size else None,
               "runs_lt100_ge1s": int((dur >= 1).sum()),
               "runs_dmin_lt65": len(r65),
               "runs_dmin_lt65_dur_median": _med(d65),
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
    return res


if __name__ == "__main__":
    corpus_main(__doc__, extract_game, analyze, required=("raw_mem.json", "labels.json", "clicks.json"))
