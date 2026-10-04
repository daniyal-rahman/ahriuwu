#!/usr/bin/env python3
"""Replay fidelity checks of the 26.19 modern sim against 16.9 (= 26.9) replays.

Complements the economy replay oracle (ops/replay_oracle_extract.py ->
lanerl_jax/data/modern/oracle/replay_16_9_observations.json.gz, tested in
lanerl_jax/sim/tests/test_modern_economy_oracle.py), which already covers
starting/ambient gold phase, death-timer table + scaling, kill/assist gold,
fountain regen and level-up HP.  This tool adds what that extract does not
record, all from ``raw_mem.json`` positions/HP/gold (frames/ is never read):

  dead_gold   gold_total slope while dead (ambient rate over the whole game)
  exits       speed vs time since leaving the fountain (Homeguard / Deathguard)
  recalls     stand-still time before a recall teleport (channel length)
  steady      steady straight-line walking speeds in lane, 90-200 s
  respawns    respawn HP == max HP, respawn point
  pov_recall  recorded champion's labels.json 'recall' action runs (cast -> teleport)

Two stages:

  extract GAME_DIR... --out DIR [--jobs N]
      Pure standard library (runs on the desktop's system python too); one
      DIR/<match>.json.gz per game.  ~1.5 GB RSS per worker on a 114 MB raw_mem.
      Login node:  ops/login_capped.sh 8G 3 .venv-jax/bin/python \
                       ops/modern_replay_fidelity.py extract <dataset>/NA1_* --out DIR --jobs 2
  analyze --out DIR [--json PATH]
      Imports the simulator (modern_economy, modern_stat_pipeline,
      modern_world constants; no tick compile) and the Riot match-v5 extract
      (lanerl_jax/data/modern/oracle/riot_16_9_frames.json.gz) and prints the
      fidelity table.

Results and interpretation: docs/modern/REPLAY_FIDELITY.md.
"""
import argparse
import gzip
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

SCHEMA = "modern_replay_fidelity/v1"
FOUNTAIN = {"blue": (394.0, 461.0), "red": (14340.0, 14391.0)}   # = modern_world.FOUNTAINS
P = {
    "fountain_radius": 1100.0,     # sp_RegenRadius (fountain_regen / in_fountain)
    "exit_window_s": 30.0,         # speed series length after leaving the fountain
    "speed_window_s": 0.5,         # displacement window for one speed sample
    "speed_step_s": 0.25,
    "straight_min": 0.995,         # chord / path length to call a window straight
    "jump_units": 2000.0,          # one-sample displacement = teleport
    "still_units": 1.0,            # stand-still tolerance before a recall jump
    "dead_min_s": 5.0,             # ignore hp==0 flickers
    "dead_margin_s": (2.5, 0.3),   # skip the kill payout at death / respawn edge
    "big_increment": 3.0,          # gold_total step > this is non-ambient income
    "steady_window_s": 1.0,
    "steady_t": (90.0, 200.0),
    "steady_min_fountain_dist": 3000.0,
}


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def load_series(path):
    """Per-hero columns over unique game times: gt, x, y, hp, hp_max, gold_total, level."""
    with open(path) as f:
        samples = json.load(f)
    cols = defaultdict(lambda: defaultdict(list))
    last = None
    for s in samples:
        gt = s.get("gt")
        if gt is None or gt == last:
            continue
        last = gt
        for h, v in s.get("heroes", {}).items():
            p = v.get("pos")
            if not p or v.get("hp") is None:
                continue
            c = cols[h]
            c["gt"].append(gt)
            c["x"].append(p[0])
            c["y"].append(p[-1])            # pos is [x, z] (or [x, y, z]); ground plane = first/last
            c["hp"].append(v["hp"])
            c["hp_max"].append(v["hp_max"])
            c["gold_total"].append(v.get("gold_total"))
            c["level"].append(v.get("level"))
    del samples
    return {h: dict(c) for h, c in cols.items()}


def team_of(c):
    p = (c["x"][0], c["y"][0])
    return "blue" if _d(p, FOUNTAIN["blue"]) < _d(p, FOUNTAIN["red"]) else "red"


def dead_runs(c):
    """[(i_first_dead, i_first_alive)] for hp<=0 runs lasting >= dead_min_s."""
    out, i, n = [], 0, len(c["gt"])
    while i < n:
        if c["hp"][i] <= 0:
            j = i
            while j < n and c["hp"][j] <= 0:
                j += 1
            if j < n and c["gt"][j] - c["gt"][i] >= P["dead_min_s"]:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def speed_at(c, i, window):
    """Speed over [gt[i], gt[i]+window] and straightness; None if a jump/gap."""
    gt, x, y = c["gt"], c["x"], c["y"]
    j, path = i, 0.0
    while j + 1 < len(gt) and gt[j] - gt[i] < window:
        step = math.hypot(x[j + 1] - x[j], y[j + 1] - y[j])
        if step > 300 or gt[j + 1] - gt[j] > 0.2:
            return None
        path += step
        j += 1
    dt = gt[j] - gt[i]
    if dt < 0.8 * window:
        return None
    chord = math.hypot(x[j] - x[i], y[j] - y[i])
    return chord / dt, (chord / path if path > 0 else 0.0)


def extract_dead_gold(c):
    out = []
    a, b = P["dead_margin_s"]
    for i0, i1 in dead_runs(c):
        t0, t1 = c["gt"][i0] + a, c["gt"][i1] - b
        if t1 - t0 < 2.0:
            continue
        idx = [k for k in range(i0, i1) if t0 <= c["gt"][k] <= t1]
        g = [c["gold_total"][k] for k in idx]
        if len(g) < 10 or any(v is None for v in g):
            continue
        big = sum(d for d in (q - p for p, q in zip(g, g[1:])) if d > P["big_increment"])
        out.append({"gt_death": c["gt"][i0], "t0": c["gt"][idx[0]], "t1": c["gt"][idx[-1]],
                    "delta": round(g[-1] - g[0], 2), "big": round(big, 2), "level": c["level"][i0]})
    return out


def extract_respawns(c, team):
    out = []
    for i0, i1 in dead_runs(c):
        out.append({"gt": c["gt"][i1], "dead_s": round(c["gt"][i1] - c["gt"][i0], 3),
                    "hp": c["hp"][i1], "hp_max": c["hp_max"][i1],
                    "pos": [c["x"][i1], c["y"][i1]],
                    "dist_fountain": round(_d((c["x"][i1], c["y"][i1]), FOUNTAIN[team]), 1)})
    return out


def extract_recalls(c, team):
    out = []
    gt, x, y, hp = c["gt"], c["x"], c["y"], c["hp"]
    f = FOUNTAIN[team]
    for i in range(1, len(gt)):
        if hp[i] <= 0 or hp[i - 1] <= 0:
            continue
        if math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) < P["jump_units"] or _d((x[i], y[i]), f) > 600:
            continue
        # the last ~0.1 s before the teleport is often nudged a few units, so the
        # stand-still reference is the position 0.25 s before the jump
        r = i - 1
        while r > 0 and gt[r] > gt[i] - 0.25:
            r -= 1
        k = r
        while k - 1 >= 0 and math.hypot(x[k - 1] - x[r], y[k - 1] - y[r]) <= P["still_units"] \
                and hp[k - 1] > 0:
            k -= 1
        took_damage = any(hp[m + 1] < hp[m] - 0.5 for m in range(k, i - 1))
        out.append({"gt": gt[i], "still_s": round(gt[i] - gt[k], 3), "gt_prev": gt[i - 1],
                    "from_dist": round(_d((x[i - 1], y[i - 1]), f), 1), "damaged": took_damage,
                    "hp_before": hp[i - 1], "hp_max": c["hp_max"][i]})
    return out


def extract_exits(c, team, recalls, respawns):
    f, R = FOUNTAIN[team], P["fountain_radius"]
    gt, x, y = c["gt"], c["x"], c["y"]
    dist = [_d((a, b), f) for a, b in zip(x, y)]
    arrivals = [(r["gt"], "recall") for r in recalls] + [(r["gt"], "respawn") for r in respawns]
    out = []
    for i in range(1, len(gt)):
        if not (dist[i - 1] <= R < dist[i]) or gt[i] < 25.0 or c["hp"][i] <= 0:
            continue
        prior = [a for a in arrivals if a[0] <= gt[i]]
        kind, arrived = (max(prior)[1], max(prior)[0]) if prior else ("game_start", 0.0)
        # entering the fountain on foot after the last arrival makes it a walk-in
        if any(dist[k - 1] > R >= dist[k] and math.hypot(x[k] - x[k - 1], y[k] - y[k - 1]) < 300
               for k in range(max(1, i - 2000), i) if gt[k] > arrived):
            kind = "walked_in"
        series, j = [], i
        while j < len(gt) and gt[j] - gt[i] <= P["exit_window_s"]:
            if c["hp"][j] <= 0:
                break
            s = speed_at(c, j, P["speed_window_s"])
            if s is not None:
                series.append([round(gt[j] - gt[i] + P["speed_window_s"] / 2, 3), round(s[0], 1), round(s[1], 4),
                               round(dist[j], 0)])
            t_next = gt[j] + P["speed_step_s"]
            while j < len(gt) and gt[j] < t_next:
                j += 1
        out.append({"gt": gt[i], "kind": kind, "arrived": arrived, "in_fountain_s": round(gt[i] - arrived, 2),
                    "level": c["level"][i], "series": series})
    return out


def extract_steady(c, team, stop):
    """Steady-speed histogram in lane before the first recall/death (no Homeguard,
    no Homestart, starting items only)."""
    lo, hi = P["steady_t"]
    hi = min(hi, stop)
    gt = c["gt"]
    f = FOUNTAIN[team]
    hist = Counter()
    i = 0
    while i < len(gt) and gt[i] < lo:
        i += 1
    while i < len(gt) and gt[i] < hi:
        if c["hp"][i] > 0 and _d((c["x"][i], c["y"][i]), f) > P["steady_min_fountain_dist"]:
            s = speed_at(c, i, P["steady_window_s"])
            if s is not None and s[1] >= 0.999 and s[0] > 100:
                hist[round(s[0])] += 1
        i += 2
    return dict(hist)


def pov_recall_runs(path):
    """Recorded champion only: [start_gt, end_gt] runs of labels.json action type 'recall'."""
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = json.load(f)
    runs, prev = [], None
    for fr in d.get("frames") or []:
        a = ((fr.get("label") or {}).get("action") or {}).get("type")
        if a == "recall" and prev != "recall":
            runs.append([fr["gt"], None])
        elif a != "recall" and prev == "recall":
            runs[-1][1] = fr["gt"]
        prev = a
    return {"champion": d.get("champion"), "runs": [r for r in runs if r[1] is not None]}


def extract_game(game_dir):
    gid = os.path.basename(os.path.normpath(game_dir))
    cols = load_series(os.path.join(game_dir, "raw_mem.json"))
    out = {"schema": SCHEMA, "game_id": gid, "params": P, "heroes": {},
           "pov_recall": pov_recall_runs(os.path.join(game_dir, "labels.json"))}
    for h, c in cols.items():
        if len(c["gt"]) < 1000:
            continue
        team = team_of(c)
        rec, rsp = extract_recalls(c, team), extract_respawns(c, team)
        out["heroes"][h] = {
            "team": team,
            "hp_max_first": c["hp_max"][0], "level_first": c["level"][0], "gt_first": c["gt"][0],
            "dead_gold": extract_dead_gold(c), "respawns": rsp, "recalls": rec,
            "exits": extract_exits(c, team, rec, rsp),
            "steady": extract_steady(c, team, min([r["gt"] for r in rec]
                                                  + [r["gt"] - r["dead_s"] for r in rsp] + [1e9]))}
    return out


def _extract_one(args):
    game_dir, out_dir = args
    gid = os.path.basename(os.path.normpath(game_dir))
    dst = os.path.join(out_dir, gid + ".json.gz")
    if os.path.exists(dst):
        return gid, "cached"
    try:
        res = extract_game(game_dir)
    except Exception as e:  # keep going over the corpus; report at the end
        return gid, "error: %r" % (e,)
    tmp = dst + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(res, f, separators=(",", ":"))
    os.replace(tmp, dst)
    return gid, "ok"


def cmd_extract(a):
    os.makedirs(a.out, exist_ok=True)
    jobs = [(g, a.out) for g in a.games if os.path.isfile(os.path.join(g, "raw_mem.json"))]
    if a.jobs > 1:
        from multiprocessing import Pool
        with Pool(a.jobs) as pool:
            for gid, st in pool.imap_unordered(_extract_one, jobs):
                print(gid, st, flush=True)
    else:
        for j in jobs:
            print(*_extract_one(j), flush=True)


# ------------------------------------------------------------------------------- analyze

def _load_games(out_dir):
    games = []
    for n in sorted(os.listdir(out_dir)):
        if n.endswith(".json.gz"):
            with gzip.open(os.path.join(out_dir, n), "rt") as f:
                games.append(json.load(f))
    return games


def _riot():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "lanerl_jax/data/modern/oracle/riot_16_9_frames.json.gz"
    d = json.load(gzip.open(p))
    F = d["fields"]
    ms_i = d["stat_names"].index("movementSpeed")
    by = defaultdict(list)    # (match, champion) -> [(t_s, ms, items, runes, shards)]
    for r in d["frames"]:
        row = dict(zip(F, r))
        by[(row["match"], row["champion"])].append((row["t_ms"] / 1000.0, row["stats"][ms_i], row["items"],
                                                    row["runes"], row["shards"]))
    return by


def _norm(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def soft_cap(raw):
    if raw > 490:
        return 0.5 * raw + 230.0
    if raw > 415:
        return 0.8 * raw + 83.0
    return raw


def inv_soft_cap(v):
    if v > 475:                       # soft_cap(490) = 475
        return (v - 230.0) / 0.5
    if v > 415:
        return (v - 83.0) / 0.8
    return v


def _plateaus(series, t_lo, t_hi, straight=0.999):
    v = [s[1] for s in series if t_lo <= s[0] < t_hi and s[2] >= straight and s[1] > 150]
    return statistics.median(v) if len(v) >= 3 else None


def _post_plateau(series, v_hg):
    """Steady speed after Homeguard drops: first run of >= 6 straight windows,
    after t >= 5 s, all at least 8% below the Homeguard plateau and within 1.5% of each other."""
    run = []
    for t, v, st, _ in series:
        if t < 5.0:
            continue
        ok = st >= 0.999 and 150 < v < 0.92 * v_hg
        if ok and (not run or abs(v - statistics.median(run)) <= 0.015 * statistics.median(run)):
            run.append(v)
            if len(run) >= 6:
                return statistics.median(run)
        else:
            run = [v] if ok else []
    return None


def cmd_analyze(a):
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from lanerl_jax.sim import modern_economy as E
    from lanerl_jax.sim import modern_stat_pipeline as SP
    from lanerl_jax.sim import modern_world as W

    games = _load_games(a.out)
    res = {"n_games": len(games)}
    print(f"games: {len(games)}")

    # -- respawn: full HP at the fountain centre (sim: modern_step _timers, FOUNTAINS)
    rs = [r for g in games for h in g["heroes"].values() for r in h["respawns"]]
    full = np.mean([abs(r["hp"] - r["hp_max"]) <= 0.15 for r in rs])
    at = np.mean([r["dist_fountain"] <= 1.0 for r in rs])
    res["respawn"] = dict(n=len(rs), full_hp=float(full), at_fountain=float(at),
                          sim_fountains=[list(map(float, p)) for p in W.FOUNTAINS])
    print(f"respawn: n={len(rs)} hp==hp_max {full:.3f}  at fountain centre (<=1u) {at:.3f}  "
          f"sim FOUNTAINS {W.FOUNTAINS}")

    # -- death durations vs E.death_time on this extract (independent of the summary's)
    obs, pred = [], []
    for g in games:
        for h in g["heroes"].values():
            dg = {round(d["gt_death"], 3): d["level"] for d in h["dead_gold"]}
            for r in h["respawns"]:
                t_d = r["gt"] - r["dead_s"]
                lv = dg.get(round(t_d, 3))
                if lv is None:
                    continue
                obs.append(r["dead_s"])
                pred.append(float(E.death_time(lv, t_d)))
    err = np.asarray(obs) - np.asarray(pred)
    res["death_time"] = dict(n=len(obs), within_0p15=float(np.mean(np.abs(err) <= 0.15)),
                             median_err=float(np.median(err)))
    print(f"death time: n={len(obs)} |obs-sim|<=0.15s {np.mean(np.abs(err) <= 0.15):.3f} "
          f"median err {np.median(err):+.3f}s")

    # -- ambient gold while dead (whole game)
    rates, rates_clean, by_t = [], [], defaultdict(list)
    for g in games:
        for h in g["heroes"].values():
            for d in h["dead_gold"]:
                dt = d["t1"] - d["t0"]
                sim = float(E.ambient_payments(d["t0"], d["t1"]))
                r = (d["delta"] - d["big"]) / dt
                rates.append(r)
                if d["big"] == 0:
                    rates_clean.append((d["delta"], sim, dt))
                    by_t[min(int(d["t0"] // 600), 4)].append(r)
    dl = np.asarray([x[0] for x in rates_clean]); sl = np.asarray([x[1] for x in rates_clean])
    res["dead_gold"] = dict(n=len(rates_clean), sim_rate=float(E._const("ai_AmbientGoldAmount")
                                                               / E._const("ai_AmbientGoldInterval")),
                            median_rate=float(np.median([d / t for d, _, t in rates_clean])),
                            within_1p5g=float(np.mean(np.abs(dl - sl) <= 1.5)),
                            median_rate_by_10min={int(k) * 10: float(np.median(v)) for k, v in sorted(by_t.items())})
    print(f"ambient gold while dead: n={len(rates_clean)} median {res['dead_gold']['median_rate']:.4f} g/s "
          f"(sim {res['dead_gold']['sim_rate']:.4f}); |obs-sim|<=1.5 g {res['dead_gold']['within_1p5g']:.3f}; "
          f"by 10 min {res['dead_gold']['median_rate_by_10min']}")

    # -- recall channel: stand-still before the teleport
    rc = [r for g in games for h in g["heroes"].values() for r in h["recalls"] if not r["damaged"]]
    still = np.asarray([r["still_s"] for r in rc])
    res["recall"] = dict(n=len(rc), sim_channel=E.RECALL_CHANNEL,
                         p01=float(np.percentile(still, 1)), p05=float(np.percentile(still, 5)),
                         median=float(np.median(still)),
                         frac_lt_7p9=float(np.mean(still < 7.9)), frac_lt_4p1=float(np.mean(still < 4.1)),
                         hist_le_9={f"{b:.1f}": int(np.sum((still >= b) & (still < b + 0.5)))
                                    for b in np.arange(0, 9, 0.5)})
    main = still[(still >= 7.9) & (still < 9.0)]
    emp = still[(still >= 3.9) & (still < 5.0)]
    res["recall"].update(mode_8s_bucket=dict(n=int(main.size), p1=float(np.percentile(main, 1)),
                                             median=float(np.median(main))),
                         mode_4s_bucket=dict(n=int(emp.size), median=float(np.median(emp)) if emp.size else None))
    # recorded champion: labels.json 'recall' action start -> teleport
    pov = []
    for g in games:
        pr = g.get("pov_recall")
        h = pr and g["heroes"].get(pr["champion"])
        if not h:
            continue
        jumps = [r["gt"] for r in h["recalls"]]
        for a0, a1 in pr["runs"]:
            if any(abs(a1 - j) <= 0.2 for j in jumps):
                pov.append(a1 - a0)
    pov = np.asarray(pov)
    if pov.size:
        res["recall"]["pov_cast_to_teleport"] = dict(
            n=int(pov.size), median_8s=float(np.median(pov[(pov > 7) & (pov < 10)])),
            n_8s=int(np.sum((pov > 7) & (pov < 10))),
            median_4s=float(np.median(pov[(pov > 3.5) & (pov < 5.5)])) if np.any((pov > 3.5) & (pov < 5.5)) else None,
            n_4s=int(np.sum((pov > 3.5) & (pov < 5.5))))
        print(f"recall (recorded champion, labels cast->teleport): {res['recall']['pov_cast_to_teleport']}")
    print(f"recall stand-still 7.9-9 s bucket: {res['recall']['mode_8s_bucket']}; 3.9-5 s bucket: "
          f"{res['recall']['mode_4s_bucket']}")
    print(f"recall stand-still: n={len(rc)} p1 {res['recall']['p01']:.2f} p5 {res['recall']['p05']:.2f} "
          f"median {res['recall']['median']:.2f}  <7.9s {res['recall']['frac_lt_7p9']:.3f} "
          f"(sim channel {E.RECALL_CHANNEL})")

    # -- Homeguard / Deathguard speed after leaving the fountain
    riot = _riot()
    cb_cache = {}

    def base_ms(name):
        if not cb_cache:
            st = json.load(open(Path(__file__).resolve().parents[1]
                                / "lanerl_jax/data/modern/oracle/client_16_9_stats.json"))["champions"]
            cb_cache.update({k: v["base_ms"] for k, v in st.items()})
        return cb_cache.get(_norm(name))

    hg_rows = []
    for g in games:
        for name, h in g["heroes"].items():
            b = base_ms(name)
            if b is None:
                continue
            for e in h["exits"]:
                if e["kind"] not in ("recall", "respawn"):
                    continue
                s = e["series"]
                v_hg = _plateaus(s, 4.5, 8.0)
                v_pk = _plateaus(s, 0.25, 0.75, straight=0.995)
                if v_hg is None:
                    continue
                v0 = _post_plateau(s, v_hg)
                if v0 is None or v0 > 600:
                    continue
                late = e["gt"] >= E.HOMEGUARD_SWITCH
                h_floor = float(E.homeguard_bonus_ms(e["gt"], 10.0))
                h_peak = float(E.homeguard_bonus_ms(e["gt"], 0.5))
                row = dict(kind=e["kind"], late=bool(late), v0=v0, base=b, v_hg=v_hg, v_pk=v_pk,
                           sim_floor=v0 + h_floor * b,                         # modern_step.py: post-cap add
                           precap_floor=soft_cap(inv_soft_cap(v0) * (1 + h_floor)),
                           sim_peak=v0 + h_peak * b,
                           precap_peak=soft_cap(inv_soft_cap(v0) * (1 + h_peak)),
                           h_precap_obs=inv_soft_cap(v_hg) / inv_soft_cap(v0) - 1.0,
                           h_sim_obs=(v_hg - v0) / b)
                hg_rows.append(row)
    res["homeguard"] = {}
    for key in [("recall", False), ("recall", True), ("respawn", False), ("respawn", True)]:
        rows = [r for r in hg_rows if (r["kind"], r["late"]) == key]
        if not rows:
            continue
        def mae(k, ref="v_hg"):
            v = [abs(r[k] - r[ref]) for r in rows if r[ref] is not None]
            return float(np.median(v)) if v else None
        summ = dict(n=len(rows), median_v0=float(np.median([r["v0"] for r in rows])),
                    median_v_floor=float(np.median([r["v_hg"] for r in rows])),
                    floor_abs_err_sim=mae("sim_floor"), floor_abs_err_precap=mae("precap_floor"),
                    peak_abs_err_sim=mae("sim_peak", "v_pk"), peak_abs_err_precap=mae("precap_peak", "v_pk"),
                    h_obs_precap_median=float(np.median([r["h_precap_obs"] for r in rows])),
                    h_obs_precap_iqr=[float(np.percentile([r["h_precap_obs"] for r in rows], q)) for q in (25, 75)],
                    h_obs_simform_median=float(np.median([r["h_sim_obs"] for r in rows])),
                    sim_floor_h=float(E.homeguard_bonus_ms(1500.0 if key[1] else 300.0, 10.0)))
        # fraction of exits that the soft-cap-first model predicts within 10 u
        summ["floor_within10_precap"] = float(np.mean([abs(r["precap_floor"] - r["v_hg"]) <= 10 for r in rows]))
        summ["floor_within10_sim"] = float(np.mean([abs(r["sim_floor"] - r["v_hg"]) <= 10 for r in rows]))
        res["homeguard"]["%s_%s" % (key[0], "late" if key[1] else "early")] = summ
        print(f"homeguard {key[0]:7s} {'>=14:00' if key[1] else '<14:00 '}: n={len(rows)} v0~{summ['median_v0']:.0f} "
              f"floor~{summ['median_v_floor']:.0f}  |err| sim(post-cap) {summ['floor_abs_err_sim']:.1f} "
              f"vs %MS-before-softcap {summ['floor_abs_err_precap']:.1f}; within10 "
              f"{summ['floor_within10_sim']:.2f}/{summ['floor_within10_precap']:.2f}; implied bonus "
              f"{summ['h_obs_precap_median']:.3f} (sim {summ['sim_floor_h']:.2f})")

    # -- steady walking speed in lane (90-200 s) vs Riot movementSpeed and base MS
    st_rows = []
    for g in games:
        for name, h in g["heroes"].items():
            hist = Counter({int(k): v for k, v in h["steady"].items()})
            if sum(hist.values()) < 30:
                continue
            mode = max(hist, key=lambda k: sum(hist.get(k + d, 0) for d in (-1, 0, 1)))
            fr = riot.get((g["game_id"], name)) or riot.get((g["game_id"], name.replace(" ", "")))
            if not fr:
                continue
            ms2 = [f[1] for f in fr if 110 <= f[0] <= 130]
            if not ms2:
                continue
            st_rows.append(dict(name=name, mode=mode, riot=ms2[0], base=base_ms(name)))
    within = np.mean([abs(r["mode"] - r["riot"]) <= 2 for r in st_rows]) if st_rows else float("nan")
    res["steady_speed"] = dict(n=len(st_rows), mode_within2_of_riot_ms=float(within))
    print(f"steady lane speed 90-200 s: n={len(st_rows)} modal speed within 2 u of Riot movementSpeed@2:00 "
          f"{within:.3f}")
    # Garen/Jax: sim 26.19 records
    for champ in ("Garen", "Jax"):
        cb = SP.champion_base([champ])
        rows = [r for r in st_rows if r["name"] == champ]
        sim_b = float(cb.base_ms[0])
        res["steady_speed"][champ] = dict(n=len(rows), sim_base_ms=sim_b, base_16_9=base_ms(champ),
                                          riot_ms=sorted(Counter(r["riot"] for r in rows).items()),
                                          replay_modes=sorted(Counter(r["mode"] for r in rows).items()))
        print(f"  {champ}: sim 26.19 base_ms {sim_b} / 16.9 {base_ms(champ)}; Riot MS@2:00 "
              f"{res['steady_speed'][champ]['riot_ms']}; replay modal {res['steady_speed'][champ]['replay_modes']}")

    # -- level-1 max HP at load (before purchases) for Garen/Jax vs sim records (+0/10/20 shard)
    for champ in ("Garen", "Jax"):
        cb = SP.champion_base([champ])
        b = float(cb.base_hp[0])
        hps = [h["hp_max_first"] for g in games for n, h in g["heroes"].items()
               if n == champ and h["level_first"] == 1 and h["gt_first"] < 5]
        # rune shards at level 1: scaling health +10 (flex and/or defense row), flat health +65 (defense)
        ok = np.mean([min(abs(v - (b + k)) for k in (0, 10, 20, 65, 75)) <= 0.15 for v in hps]) if hps else float("nan")
        res.setdefault("hp_l1", {})[champ] = dict(n=len(hps), sim_base_hp=b, observed=sorted(Counter(hps).items()),
                                                 match_base_plus_shards=float(ok))
        print(f"  {champ} level-1 hp_max at load: n={len(hps)} sim base {b} -> {sorted(Counter(hps).items())} "
              f"(= base + shards {ok:.3f})")

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
    main()
