#!/usr/bin/env python3
"""Extract RAW ground-truth observations from recorded LoL replays (patch 16.9).

Independent oracle: this script only records what the client memory reader
saw.  It deliberately encodes no game formula (no gold/xp/regen models); the
only thresholds are the extraction parameters listed in PARAMS below, which are
written into every output file.

Inputs per game dir (opened by exact path, never globbed -- frames/ is huge):
    raw_mem.json    list of ~40 Hz samples
                    {"wall", "gt", "heroes": {Name: {"pos", "hp", "hp_max",
                     "gold", "gold_total", "level"}}}
    labels.json     header fields only: match_id, champion, team, slot, fps
    item_track.json optional, recorded champion only (see item_timeline())

Modes:
    extract  GAME_DIR [GAME_DIR ...] --out OUT_DIR   one <match>.json.gz each
    index    --out OUT_DIR                            OUT_DIR/index.json
    summarize --out OUT_DIR --summary PATH            compact all-game summary

Pure standard library (the desktop's system python has no numpy).
"""
import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter

SCRIPT_VERSION = 2
PATCH = "16.9"
PARAMS = {
    "chunk_threshold": 3.0,          # gold_total sample-to-sample increase > this = chunk
    "early_window_s": [0.0, 150.0],  # full-res gold_total series + all increments
    "payout_window_rel_s": [-0.5, 1.5],
    "fountain_radius": 1200.0,
    "fountain_segment_cap_s": 20.0,
    "recall_jump_units": 2000.0,     # one-sample displacement > this = teleport-like
    "gap_report_s": 0.25,            # gt steps larger than this are listed as gaps
    "spawn_blue": [400.0, 400.0],
    "spawn_red": [14300.0, 14400.0],
    "levelup_after_probe_s": 0.5,
    "near_levelup_s": 1.0,
    "summary_fountain_hz": 10.0,
}
CURRENT_GOLD_SANE = (-1.0, 1.0e5)    # outside this the raw current-gold read is garbage


def sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def r(x, nd=3):
    if x is None:
        return None
    if isinstance(x, float):
        if not math.isfinite(x):
            return repr(x)
        return round(x, nd)
    return x


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pos2(p):
    if p is None:
        return None
    return [r(float(p[0]), 1), r(float(p[1]), 1)]


def sane_gold(g):
    return g is not None and isinstance(g, (int, float)) and math.isfinite(g) \
        and CURRENT_GOLD_SANE[0] <= g <= CURRENT_GOLD_SANE[1]


# --------------------------------------------------------------------------
# loading

FIELDS = ("hp", "hp_max", "gold", "gold_total", "level")


def load_raw(path):
    """Return (gts, walls, heroes, per_hero_columns, load_notes).

    Columns are python lists aligned to gts; a missing hero in a sample is None.
    """
    with open(path) as f:
        samples = json.load(f)
    heroes = []
    seen = set()
    for s in samples[:50]:
        for name in s.get("heroes", {}):
            if name not in seen:
                seen.add(name)
                heroes.append(name)
    n = len(samples)
    gts = [None] * n
    walls = [None] * n
    cols = {h: {k: [None] * n for k in FIELDS + ("x", "y")} for h in heroes}
    extra_heroes = Counter()
    pos_dims = Counter()
    for i, s in enumerate(samples):
        gts[i] = s.get("gt")
        walls[i] = s.get("wall")
        hs = s.get("heroes", {})
        for name, d in hs.items():
            c = cols.get(name)
            if c is None:
                extra_heroes[name] += 1
                continue
            for k in FIELDS:
                c[k][i] = d.get(k)
            p = d.get("pos")
            if p is not None:
                pos_dims[len(p)] += 1
                c["x"][i] = p[0]
                c["y"][i] = p[1]
    del samples
    notes = {"n_samples": n, "pos_dims": dict(pos_dims),
             "heroes_not_in_first_50_samples": dict(extra_heroes)}
    return gts, walls, heroes, cols, notes


def load_labels_header(path):
    with open(path) as f:
        d = json.load(f)
    frames = d.get("frames") or []
    out = {k: d.get(k) for k in ("match_id", "champion", "team", "slot", "fps",
                                  "total_frames")}
    out["n_frames"] = len(frames)
    out["first_frame_gt"] = frames[0].get("gt") if frames else None
    out["last_frame_gt"] = frames[-1].get("gt") if frames else None
    # recorded champion's own stats per frame (20 fps): used only to check
    # whether the label-side current gold is a live value (raw_mem's is not).
    stats = []
    for fr in frames:
        cs = (fr.get("label") or {}).get("champion_stats") or {}
        stats.append((fr.get("gt"), cs.get("gold"), cs.get("gold_total"), cs.get("hp"),
                      cs.get("hp_max"), cs.get("level")))
    del d, frames
    return out, stats


def labels_gold_observations(stats):
    """Current-gold behaviour of the recorded champion in labels.json."""
    gvals = [x[1] for x in stats if sane_gold(x[1])]
    drops, gains_vs_total = [], []
    prev = None
    n_total_drop = 0
    for x in stats:
        if x[0] is None or not sane_gold(x[1]) or x[2] is None:
            continue
        if prev is not None:
            dg, dt = x[1] - prev[1], x[2] - prev[2]
            if dt < 0:
                n_total_drop += 1
            if dg < -1e-6:
                drops.append({"gt": r(x[0]), "gt_prev": r(prev[0]), "gold_before": r(prev[1]),
                              "gold_after": r(x[1]), "gold_total_before": r(prev[2]),
                              "gold_total_after": r(x[2]), "hp_max_before": r(prev[4]),
                              "hp_max_after": r(x[4])})
            elif dg > 1e-6:
                gains_vs_total.append(r(dg - dt, 3))
        prev = x
    return {"n_frames": len(stats), "n_sane_gold": len(gvals),
            "n_distinct_gold": len(set(gvals)),
            "gold_min": r(min(gvals)) if gvals else None, "gold_max": r(max(gvals)) if gvals else None,
            "n_gold_drops": len(drops), "gold_drops": drops,
            "n_gold_total_drops": n_total_drop,
            "gold_gain_minus_total_gain_hist": [[k, v] for k, v in Counter(gains_vs_total).most_common(10)]}


# --------------------------------------------------------------------------
# helpers on columns

def nearest_index_le(gts, t, lo=0):
    """Largest i with gts[i] <= t (gts assumed nondecreasing); -1 if none."""
    import bisect
    return bisect.bisect_right(gts, t, lo) - 1


def value_at(col, i):
    return col[i] if 0 <= i < len(col) else None


def sample_rate_stats(gts, walls):
    steps = [gts[i + 1] - gts[i] for i in range(len(gts) - 1)
             if gts[i] is not None and gts[i + 1] is not None]
    pos_steps = [s for s in steps if s > 0]
    gaps = [{"gt_before": r(gts[i]), "gt_after": r(gts[i + 1]), "step": r(gts[i + 1] - gts[i])}
            for i in range(len(gts) - 1)
            if gts[i] is not None and gts[i + 1] is not None
            and gts[i + 1] - gts[i] > PARAMS["gap_report_s"]]
    back = [{"index": i + 1, "gt_before": r(gts[i]), "gt_after": r(gts[i + 1])}
            for i in range(len(gts) - 1)
            if gts[i] is not None and gts[i + 1] is not None and gts[i + 1] < gts[i]]
    wsteps = [walls[i + 1] - walls[i] for i in range(len(walls) - 1)
              if walls[i] is not None and walls[i + 1] is not None]
    rate = None
    if len(gts) > 1 and walls[0] is not None and walls[-1] is not None and walls[-1] != walls[0]:
        rate = (gts[-1] - gts[0]) / (walls[-1] - walls[0])
    step_hist = Counter(round(s, 3) for s in steps)
    return {
        "n_samples": len(gts),
        "gt_first": r(gts[0]) if gts else None,
        "gt_last": r(gts[-1]) if gts else None,
        "gt_step_median": r(statistics.median(steps), 4) if steps else None,
        "gt_step_min": r(min(steps), 4) if steps else None,
        "gt_step_max": r(max(steps), 4) if steps else None,
        "gt_step_mean": r(sum(steps) / len(steps), 4) if steps else None,
        "n_zero_steps": sum(1 for s in steps if s == 0),
        "n_negative_steps": len(back),
        "negative_steps": back[:50],
        "n_gaps_gt_%.2fs" % PARAMS["gap_report_s"]: len(gaps),
        "gaps": gaps[:200],
        "wall_step_median": r(statistics.median(wsteps), 4) if wsteps else None,
        "gt_per_wall_second": r(rate, 4),
        "gt_step_top10": [[k, v] for k, v in step_hist.most_common(10)],
        "positive_step_median": r(statistics.median(pos_steps), 4) if pos_steps else None,
    }


def value_quirks(col):
    """Observational stats on a numeric column (rounding, granularity)."""
    vals = [v for v in col if isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return {"n": 0}
    n_int = sum(1 for v in vals if float(v).is_integer())
    frac = Counter()
    for v in vals:
        f = round(abs(v) - math.floor(abs(v)), 4)
        frac[f] += 1
    return {"n": len(vals), "frac_integer": r(n_int / len(vals), 4),
            "n_distinct_fractional_parts": len(frac),
            "top_fractional_parts": [[k, c] for k, c in frac.most_common(6)],
            "min": r(min(vals)), "max": r(max(vals))}


# --------------------------------------------------------------------------
# extraction

def team_of(x, y):
    db = dist((x, y), PARAMS["spawn_blue"])
    dr = dist((x, y), PARAMS["spawn_red"])
    return ("blue" if db < dr else "red"), db, dr


def extract_game(game_dir, hash_files=True):
    t0 = datetime.datetime.now(datetime.timezone.utc)
    game_dir = game_dir.rstrip("/")
    gid = os.path.basename(game_dir)
    paths = {k: os.path.join(game_dir, k + ".json")
             for k in ("raw_mem", "labels", "item_track")}
    present = {k: os.path.isfile(p) for k, p in paths.items()}
    out = {
        "schema": "replay_oracle_game/v1",
        "script": "ops/replay_oracle_extract.py",
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "patch": PATCH,
        "game_dir": game_dir,
        "game_id": gid,
        "files_present": present,
        "params": PARAMS,
        "extracted_at": t0.isoformat(timespec="seconds"),
    }
    if hash_files:
        out["source_sha256"] = {k + ".json": sha256_file(p)
                                for k, p in paths.items() if present[k]}
        out["source_bytes"] = {k + ".json": os.path.getsize(p)
                               for k, p in paths.items() if present[k]}
    labels = None
    label_stats = None
    if present["labels"]:
        try:
            labels, label_stats = load_labels_header(paths["labels"])
            out["labels_recorded_gold"] = labels_gold_observations(label_stats)
        except Exception as e:  # noqa: BLE001 - record, continue
            out["labels_error"] = repr(e)
    out["labels_header"] = labels
    if not present["raw_mem"]:
        out["status"] = "missing_raw_mem"
        return out

    gts, walls, heroes, C, load_notes = load_raw(paths["raw_mem"])
    n = len(gts)
    out["load_notes"] = load_notes
    if n < 2:
        out["status"] = "too_few_samples"
        return out

    # ---- metadata / teams
    first_i = {}
    for h in heroes:
        for i in range(n):
            if C[h]["x"][i] is not None:
                first_i[h] = i
                break
    teams, team_evidence = {}, {}
    for h in heroes:
        i = first_i.get(h)
        if i is None:
            continue
        t, db, dr = team_of(C[h]["x"][i], C[h]["y"][i])
        teams[h] = t
        team_evidence[h] = {"first_pos": pos2((C[h]["x"][i], C[h]["y"][i])),
                            "first_gt": r(gts[i]), "dist_blue": r(db, 1), "dist_red": r(dr, 1)}
    rec = labels.get("champion") if labels else None
    meta = {
        "match_id": (labels or {}).get("match_id") or gid,
        "champions": heroes,
        "team": teams,
        "team_counts": dict(Counter(teams.values())),
        "team_evidence": team_evidence,
        "recorded_champion": rec,
        "recorded_team_labels": (labels or {}).get("team"),
        "recorded_slot_labels": (labels or {}).get("slot"),
        "recorded_team_inferred": teams.get(rec),
        "recorded_team_matches_labels": (teams.get(rec) == labels.get("team")) if labels and rec in teams else None,
        "recorded_champion_in_raw_mem": rec in C if rec else None,
        "gt_first_sample": r(gts[0]),
        "sampling": sample_rate_stats(gts, walls),
    }
    out["meta"] = meta

    # Indices sorted (stable) by gt are only needed if gt goes backwards; we keep
    # the recorded sample order and report backward steps in meta.sampling.
    lo, hi = PARAMS["early_window_s"]
    early_idx = [i for i in range(n) if gts[i] is not None and lo <= gts[i] <= hi]

    # ---- gold
    gold = {"initial": {}, "early_series": {"gt": [r(gts[i]) for i in early_idx]},
            "early_increments": {}, "chunks": [], "gold_total_decreases": {},
            "current_gold_checks": {}, "small_increment_hist": {},
            "gold_total_value_quirks": {}}
    series = {}
    for h in heroes:
        gt_col, g_col = C[h]["gold_total"], C[h]["gold"]
        i0 = first_i.get(h, 0)
        gold["initial"][h] = {"gt": r(gts[i0]), "gold_total": r(gt_col[i0]), "gold": r(g_col[i0])}
        series[h] = [r(gt_col[i]) for i in early_idx]
        incs, decs, small = [], [], Counter()
        n_cur_drop, n_cur_insane, cur_drops_with_total_drop = 0, 0, 0
        distinct_cur = set()
        prev_i = None
        for i in range(n):
            v = gt_col[i]
            if v is None:
                continue
            if prev_i is not None:
                pv = gt_col[prev_i]
                d = v - pv
                if d != 0 and gts[i] is not None and lo <= gts[i] <= hi:
                    incs.append([r(gts[i]), r(d, 4)])
                if d > PARAMS["chunk_threshold"]:
                    gold["chunks"].append({
                        "gt": r(gts[i]), "gt_prev": r(gts[prev_i]), "hero": h, "delta": r(d, 3),
                        "gold_total": r(v), "pos": pos2((C[h]["x"][i], C[h]["y"][i])),
                        "level": C[h]["level"][i], "hp": r(C[h]["hp"][i], 1),
                        "alive": (C[h]["hp"][i] or 0) > 0})
                elif 0 < d <= PARAMS["chunk_threshold"]:
                    small[round(d, 2)] += 1
                if d < 0:
                    decs.append([r(gts[i]), r(d, 4), r(pv), r(v)])
                cg, pcg = g_col[i], g_col[prev_i]
                if sane_gold(cg) and sane_gold(pcg) and cg < pcg - 1e-6:
                    n_cur_drop += 1
                    if d < 0:
                        cur_drops_with_total_drop += 1
            if not sane_gold(g_col[i]):
                n_cur_insane += 1
            distinct_cur.add(g_col[i] if sane_gold(g_col[i]) else "garbage")
            prev_i = i
        gold["early_increments"][h] = incs
        gold["gold_total_decreases"][h] = {"n": len(decs), "first": decs[:50]}
        gold["current_gold_checks"][h] = {
            "n_current_gold_drops": n_cur_drop,
            "n_current_drops_coinciding_with_total_drop": cur_drops_with_total_drop,
            "n_samples_current_gold_garbage": n_cur_insane,
            "frac_current_gold_garbage": r(n_cur_insane / n, 4),
            "n_distinct_current_gold_values": len(distinct_cur),
            "distinct_current_gold_values_first10": sorted(str(v) for v in distinct_cur)[:10]}
        gold["small_increment_hist"][h] = [[k, c] for k, c in sorted(small.items())]
        gold["gold_total_value_quirks"][h] = value_quirks(gt_col)
    gold["early_series"]["gold_total"] = series
    gold["n_chunks"] = len(gold["chunks"])
    gold["chunks"].sort(key=lambda c: (c["gt"], c["hero"]))
    out["gold"] = gold

    # ---- deaths / respawns
    def gold_total_at(h, t):
        i = nearest_index_le(gts, t)
        if i < 0:
            i = 0
        # walk back over missing values
        col = C[h]["gold_total"]
        while i > 0 and col[i] is None:
            i -= 1
        return col[i], i

    deaths = []
    for h in heroes:
        hp = C[h]["hp"]
        prev_alive = None
        i = 0
        while i < n:
            v = hp[i]
            if v is None:
                i += 1
                continue
            alive = v > 0
            if prev_alive is True and not alive:
                di = i
                j = di + 1
                while j < n and not (hp[j] is not None and hp[j] > 0):
                    j += 1
                dgt = gts[di]
                w0, w1 = dgt + PARAMS["payout_window_rel_s"][0], dgt + PARAMS["payout_window_rel_s"][1]
                payouts = {}
                for o in heroes:
                    a, ia = gold_total_at(o, w0)
                    b, ib = gold_total_at(o, w1)
                    payouts[o] = {"delta": r(b - a, 3) if a is not None and b is not None else None,
                                  "team": teams.get(o)}
                # individual chunks of others inside the window
                win_chunks = []
                k0 = max(0, nearest_index_le(gts, w0))
                k1 = nearest_index_le(gts, w1)
                for o in heroes:
                    col = C[o]["gold_total"]
                    for k in range(k0 + 1, k1 + 1):
                        if col[k] is not None and col[k - 1] is not None and col[k] - col[k - 1] != 0:
                            win_chunks.append([o, r(gts[k]), r(col[k] - col[k - 1], 3)])
                rec_d = {
                    "hero": h, "team": teams.get(h), "gt": r(dgt), "gt_prev_alive": r(gts[di - 1]) if di > 0 else None,
                    "level": C[h]["level"][di], "pos": pos2((C[h]["x"][di], C[h]["y"][di])),
                    "pos_prev_alive": pos2((C[h]["x"][di - 1], C[h]["y"][di - 1])) if di > 0 else None,
                    "hp_at_death_sample": r(v, 2), "hp_prev_sample": r(hp[di - 1], 2) if di > 0 else None,
                    "hp_max": r(C[h]["hp_max"][di], 2),
                    "gold_total": r(C[h]["gold_total"][di]),
                    "payout_window": [r(w0), r(w1)],
                    "payout_deltas": payouts,
                    "window_gold_changes": sorted(win_chunks, key=lambda x: (x[1], x[0])),
                }
                if j < n:
                    rec_d["respawn"] = {
                        "gt": r(gts[j]), "pos": pos2((C[h]["x"][j], C[h]["y"][j])),
                        "hp": r(hp[j], 2), "hp_max": r(C[h]["hp_max"][j], 2),
                        "level": C[h]["level"][j], "gold_total": r(C[h]["gold_total"][j]),
                        "gold": r(C[h]["gold"][j]),
                        "last_dead_gt": r(gts[j - 1]),
                        "hp_equals_hp_max": hp[j] == C[h]["hp_max"][j],
                        "next_samples": [[r(gts[k]), pos2((C[h]["x"][k], C[h]["y"][k])), r(hp[k], 2)]
                                         for k in range(j + 1, min(n, j + 6))],
                    }
                    rec_d["dead_duration"] = r(gts[j] - dgt, 3)
                    dead_hp = [hp[k] for k in range(di, j) if hp[k] is not None]
                    rec_d["dead_hp_values"] = sorted(set(r(x, 2) for x in dead_hp))[:10]
                    rec_d["n_dead_samples"] = j - di
                    # position while dead: did it move?
                    rec_d["pos_last_dead_sample"] = pos2((C[h]["x"][j - 1], C[h]["y"][j - 1]))
                else:
                    rec_d["respawn"] = None
                    rec_d["dead_duration"] = None
                    rec_d["n_dead_samples"] = n - di
                deaths.append(rec_d)
                prev_alive = False
                i = j
                continue
            prev_alive = alive
            i += 1
    deaths.sort(key=lambda d: (d["gt"], d["hero"]))
    out["deaths"] = deaths

    # ---- fountains from respawn positions (fallback: game-start positions)
    fountains = {}
    for t in ("blue", "red"):
        # only "full" respawns: >= 5 s at hp <= 0 and back at hp == hp_max.  Short
        # hp<=0 blips (dead_duration < 2 s records) keep their field position.
        rp = [d["respawn"]["pos"] for d in deaths
              if d.get("respawn") and d["team"] == t and (d.get("dead_duration") or 0) >= 5.0
              and d["respawn"].get("hp_equals_hp_max")]
        src = "respawn_median(dead>=5s,hp==hp_max)"
        if not rp:
            rp = [d["respawn"]["pos"] for d in deaths if d.get("respawn") and d["team"] == t]
            src = "respawn_median(all)"
        if not rp:
            rp = [team_evidence[h]["first_pos"] for h in heroes if teams.get(h) == t]
            src = "first_sample_median_fallback"
        if rp:
            fountains[t] = {"pos": [r(statistics.median(p[0] for p in rp), 1),
                                    r(statistics.median(p[1] for p in rp), 1)],
                            "source": src, "n": len(rp),
                            "spread_max": r(max(dist(p, (statistics.median(q[0] for q in rp),
                                                          statistics.median(q[1] for q in rp))) for p in rp), 1)}
    out["fountains"] = fountains

    # ---- fountain segments
    segs = []
    R = PARAMS["fountain_radius"]
    cap = PARAMS["fountain_segment_cap_s"]
    for h in heroes:
        f = fountains.get(teams.get(h))
        if not f:
            continue
        fp = f["pos"]
        X, Y, HP = C[h]["x"], C[h]["y"], C[h]["hp"]
        inside_prev = False
        cur = None
        for i in range(n):
            if X[i] is None or HP[i] is None:
                continue
            inside = HP[i] > 0 and dist((X[i], Y[i]), fp) <= R
            if inside and not inside_prev:
                # classify the start
                pi = i - 1
                while pi >= 0 and (X[pi] is None or HP[pi] is None):
                    pi -= 1
                if pi < 0:
                    kind, jump = "game_start", None
                else:
                    jump = dist((X[i], Y[i]), (X[pi], Y[pi]))
                    k = pi
                    recent_dead = False
                    while k >= 0 and gts[k] is not None and gts[i] - gts[k] <= 2.0:
                        if HP[k] is not None and HP[k] <= 0:
                            recent_dead = True
                            break
                        k -= 1
                    if recent_dead:
                        kind = "respawn"
                    elif jump > PARAMS["recall_jump_units"]:
                        kind = "recall"
                    else:
                        kind = "walked_in"
                cur = {"hero": h, "team": teams.get(h), "start_gt": r(gts[i]), "kind": kind,
                       "entry_jump": r(jump, 1), "start_pos": pos2((X[i], Y[i])),
                       "prev_pos": pos2((X[pi], Y[pi])) if pi >= 0 else None,
                       "prev_hp": r(HP[pi], 2) if pi >= 0 else None,
                       "series": {"gt": [], "hp": [], "hp_max": [], "gold": [], "level": []},
                       "_start": gts[i]}
                segs.append(cur)
            if inside:
                if gts[i] - cur["_start"] <= cap:
                    s = cur["series"]
                    s["gt"].append(r(gts[i]))
                    s["hp"].append(r(HP[i], 3))
                    s["hp_max"].append(r(C[h]["hp_max"][i], 3))
                    s["gold"].append(r(C[h]["gold"][i], 3))
                    s["level"].append(C[h]["level"][i])
                cur["end_gt"] = r(gts[i])
            elif inside_prev and cur is not None:
                cur["exit_reason"] = "died" if HP[i] <= 0 else "left_radius"
            inside_prev = inside
    for s in segs:
        s.pop("_start", None)
        s["duration"] = r(s["end_gt"] - s["start_gt"], 3)
        s["capped"] = s["duration"] > cap
        s.setdefault("exit_reason", "game_end")
    segs.sort(key=lambda s: (s["start_gt"], s["hero"]))
    out["fountain_segments"] = segs

    # ---- level-ups and hp_max changes
    levelups, hpmax_changes = [], []
    probe = PARAMS["levelup_after_probe_s"]
    for h in heroes:
        L, HM, HP, G = C[h]["level"], C[h]["hp_max"], C[h]["hp"], C[h]["gold"]
        prev = None
        for i in range(n):
            if L[i] is None or HM[i] is None:
                continue
            if prev is not None:
                if L[i] > L[prev]:
                    k = nearest_index_le(gts, gts[i] + probe)
                    levelups.append({
                        "hero": h, "gt": r(gts[i]), "gt_prev": r(gts[prev]),
                        "old_level": L[prev], "new_level": L[i],
                        "hp_before": r(HP[prev], 3), "hp_max_before": r(HM[prev], 3),
                        "hp_after": r(HP[i], 3), "hp_max_after": r(HM[i], 3),
                        "hp_probe": r(HP[k], 3) if k > i else None,
                        "hp_max_probe": r(HM[k], 3) if k > i else None,
                        "probe_gt": r(gts[k]) if k > i else None,
                        "alive": (HP[i] or 0) > 0})
                elif L[i] < L[prev]:
                    levelups.append({"hero": h, "gt": r(gts[i]), "old_level": L[prev],
                                     "new_level": L[i], "level_decrease": True})
                if HM[i] != HM[prev] and L[i] == L[prev]:
                    hpmax_changes.append({
                        "hero": h, "gt": r(gts[i]), "gt_prev": r(gts[prev]),
                        "old_hp_max": r(HM[prev], 3), "new_hp_max": r(HM[i], 3),
                        "delta": r(HM[i] - HM[prev], 3),
                        "hp_before": r(HP[prev], 3), "hp_after": r(HP[i], 3),
                        "gold_before": r(G[prev], 3), "gold_after": r(G[i], 3),
                        "level": L[i], "pos": pos2((C[h]["x"][i], C[h]["y"][i])),
                        "alive_before": (HP[prev] or 0) > 0, "alive_after": (HP[i] or 0) > 0})
            prev = i
    # flag hp_max changes close to a level-up of the same hero
    lu_by_hero = {}
    for lu in levelups:
        lu_by_hero.setdefault(lu["hero"], []).append(lu["gt"])
    for c in hpmax_changes:
        c["near_levelup"] = any(abs(c["gt"] - g) <= PARAMS["near_levelup_s"]
                                for g in lu_by_hero.get(c["hero"], []))
    levelups.sort(key=lambda d: (d["gt"], d["hero"]))
    hpmax_changes.sort(key=lambda d: (d["gt"], d["hero"]))
    out["levelups"] = levelups
    out["hp_max_changes"] = hpmax_changes

    # ---- value granularity quirks
    out["value_quirks"] = {h: {"hp": value_quirks(C[h]["hp"]), "hp_max": value_quirks(C[h]["hp_max"])}
                           for h in heroes}
    # gold_total change cadence: intervals between consecutive changes (whole game, alive or not)
    cad = {}
    for h in heroes:
        col = C[h]["gold_total"]
        ch = [gts[i] for i in range(1, n) if col[i] is not None and col[i - 1] is not None and col[i] != col[i - 1]]
        iv = [b - a for a, b in zip(ch, ch[1:])]
        early_ch = [t for t in ch if lo <= t <= hi]
        cad[h] = {"n_changes": len(ch),
                  "interval_median": r(statistics.median(iv), 4) if iv else None,
                  "interval_top": [[k, v] for k, v in Counter(round(x, 2) for x in iv).most_common(8)],
                  "first_change_gt": r(ch[0]) if ch else None,
                  "n_changes_early_window": len(early_ch)}
    out["gold_total_change_cadence"] = cad

    # ---- labels.json recorded-champion stats vs raw_mem (nearest earlier sample)
    if label_stats and rec in C:
        diffs = {"gold_total": [], "hp": [], "hp_max": [], "level": []}
        for x in label_stats[::20]:
            if x[0] is None:
                continue
            i = nearest_index_le(gts, x[0])
            if i < 0:
                continue
            for k, li in (("gold_total", 2), ("hp", 3), ("hp_max", 4), ("level", 5)):
                a, b = x[li], C[rec][k][i]
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    diffs[k].append(abs(a - b))
        out["labels_vs_raw_mem_recorded"] = {
            k: {"n": len(v), "max_abs_diff": r(max(v)) if v else None,
                "frac_exact": r(sum(1 for d in v if d == 0) / len(v), 4) if v else None}
            for k, v in diffs.items()}
        out["labels_vs_raw_mem_recorded"]["note"] = "labels frame gt matched to last raw_mem sample with gt <= it, every 20th frame"

    # ---- item_track
    if present["item_track"]:
        try:
            out["item_track"] = item_timeline(paths["item_track"])
        except Exception as e:  # noqa: BLE001
            out["item_track"] = {"error": repr(e)}
    else:
        out["item_track"] = None

    out["counts"] = {"deaths": len(deaths), "respawns": sum(1 for d in deaths if d.get("respawn")),
                     "levelups": sum(1 for l in levelups if not l.get("level_decrease")),
                     "level_decreases": sum(1 for l in levelups if l.get("level_decrease")),
                     "fountain_segments": len(segs), "hp_max_changes": len(hpmax_changes),
                     "hp_max_changes_not_near_levelup": sum(1 for c in hpmax_changes if not c["near_levelup"]),
                     "gold_chunks": gold["n_chunks"]}
    out["status"] = "ok"
    out["elapsed_s"] = r((datetime.datetime.now(datetime.timezone.utc) - t0).total_seconds(), 1)
    return out


def item_timeline(path):
    """item_track.json (recorded champion only) schema, as observed:
    {match_id, champion, scanner:{speed, poll_interval_s, snapshot_every_gt,
     patch_mod_size, scanned_at}, n_polls, n_drifts,
     events:[{gt, slot, type in acquired|removed|changed|consume|active_fire|
              active_flag_rise, item_id, [from_id], [from_uc,to_uc], [fire_t]}],
     snapshots:[{gt, inv:[7 x null | {id, use_counter, last_fire_t, active_flag}]}]}
    Slot 6 is the trinket slot.  We keep header + all events verbatim and the
    snapshots reduced to inventory-id changes only.
    """
    with open(path) as f:
        d = json.load(f)
    snaps = d.get("snapshots") or []
    inv_changes = []
    last = None
    for s in snaps:
        ids = [x.get("id") if isinstance(x, dict) else None for x in (s.get("inv") or [])]
        if ids != last:
            inv_changes.append([r(s.get("gt")), ids])
            last = ids
    ev = []
    types = Counter()
    for e in d.get("events") or []:
        e2 = {k: (r(v) if isinstance(v, float) else v) for k, v in e.items()}
        ev.append(e2)
        types[e.get("type")] += 1
    return {"champion": d.get("champion"), "match_id": d.get("match_id"),
            "scanner": d.get("scanner"), "n_polls": d.get("n_polls"), "n_drifts": d.get("n_drifts"),
            "n_snapshots": len(snaps),
            "snapshot_gt_first": r(snaps[0].get("gt")) if snaps else None,
            "snapshot_gt_last": r(snaps[-1].get("gt")) if snaps else None,
            "top_level_keys": sorted(d.keys()),
            "event_type_counts": dict(types), "events": ev,
            "inventory_id_changes": inv_changes}


# --------------------------------------------------------------------------
# index / summary

def write_json_gz(obj, path):
    tmp = path + ".tmp_partial"
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def read_json_gz(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def game_outputs(out_dir):
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                  if f.endswith(".json.gz") and f.startswith("NA1_"))


def build_index(out_dir, dataset_root=None):
    entries = []
    for p in game_outputs(out_dir):
        g = read_json_gz(p)
        entries.append({"game_id": g["game_id"], "file": os.path.basename(p), "status": g.get("status"),
                        "files_present": g.get("files_present"),
                        "source_sha256": g.get("source_sha256"),
                        "recorded_champion": (g.get("meta") or {}).get("recorded_champion"),
                        "recorded_team_matches_labels": (g.get("meta") or {}).get("recorded_team_matches_labels"),
                        "counts": g.get("counts"), "elapsed_s": g.get("elapsed_s")})
    missing = []
    if dataset_root:
        have = {e["game_id"] for e in entries}
        for d in sorted(os.listdir(dataset_root)):
            if d.startswith("NA1_") and d not in have:
                missing.append(d)
    idx = {"schema": "replay_oracle_index/v1", "patch": PATCH, "dataset_root": dataset_root,
           "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "n_games": len(entries), "games": entries, "game_dirs_without_output": missing}
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(idx, f, indent=1)
    return idx


def downsample_series(series, hz):
    gt = series["gt"]
    keep, last = [], None
    for i, t in enumerate(gt):
        if last is None or t - last >= 1.0 / hz - 1e-9 or i == len(gt) - 1:
            keep.append(i)
            last = t
    return {k: [v[i] for i in keep] for k, v in series.items()}


LEVELUP_FIELDS = ["hero", "gt", "gt_prev", "old_level", "new_level", "hp_before", "hp_max_before",
                  "hp_after", "hp_max_after", "hp_probe", "hp_max_probe", "probe_gt", "alive",
                  "level_decrease"]
HPMAX_FIELDS = ["hero", "gt", "gt_prev", "old_hp_max", "new_hp_max", "delta", "hp_before", "hp_after",
                "gold_before", "gold_after", "level", "pos", "alive_before", "alive_after",
                "near_levelup"]


def table(rows, fields):
    return {"fields": fields, "rows": [[row.get(f) for f in fields] for row in rows]}


def build_summary(out_dir, summary_path, dataset_root):
    games = []
    hashes = {}
    totals = Counter()
    per_game_sha = set()
    for p in game_outputs(out_dir):
        g = read_json_gz(p)
        gid = g["game_id"]
        hashes[gid] = (g.get("source_sha256") or {}).get("raw_mem.json")
        if g.get("script_sha256"):
            per_game_sha.add(g["script_sha256"])
        e = {"game_id": gid, "status": g.get("status"), "files_present": g.get("files_present")}
        if g.get("status") == "ok":
            meta = dict(g["meta"])
            samp = dict(meta["sampling"])
            samp["gaps"] = samp["gaps"][:20]
            meta["sampling"] = samp
            gold = g["gold"]
            champs = meta["champions"]
            fs = []
            for s in g["fountain_segments"]:
                s2 = {k: v for k, v in s.items() if k != "series"}
                ser = downsample_series(s["series"], PARAMS["summary_fountain_hz"])
                for k in ("hp_max", "gold", "level"):
                    if len(set(map(str, ser[k]))) <= 1:   # constant -> scalar
                        ser[k] = ser[k][0] if ser[k] else None
                t0s = s["start_gt"]
                ser["dt"] = [r(t - t0s, 3) for t in ser.pop("gt")]   # seconds since start_gt
                s2["series"] = ser
                fs.append(s2)
            deaths = []
            for d in g["deaths"]:
                d2 = {k: v for k, v in d.items() if k not in ("payout_deltas", "window_gold_changes")}
                d2["payout_deltas"] = [d["payout_deltas"].get(h, {}).get("delta") for h in champs]
                d2["window_gold_chunks"] = [c for c in d["window_gold_changes"]
                                            if abs(c[2]) > PARAMS["chunk_threshold"]]
                if d2.get("respawn"):
                    d2["respawn"] = dict(d2["respawn"])
                    d2["respawn"]["next_samples"] = d2["respawn"]["next_samples"][:2]
                deaths.append(d2)
            e.update({
                "meta": meta,
                "fountains": g["fountains"],
                "initial_gold": gold["initial"],
                "early_gold_total_increments": gold["early_increments"],
                "gold_total_decreases": {h: v["n"] for h, v in gold["gold_total_decreases"].items()},
                "current_gold_checks": gold["current_gold_checks"],
                "small_increment_hist": gold["small_increment_hist"],
                "gold_total_change_cadence": g["gold_total_change_cadence"],
                "n_gold_chunks": gold["n_chunks"],
                "deaths": deaths,
                "fountain_segments": fs,
                "levelups": table(g["levelups"], LEVELUP_FIELDS),
                "hp_max_changes_not_at_levelup": table(g["hp_max_changes"], HPMAX_FIELDS),
                "item_timeline": ({k: g["item_track"].get(k) for k in
                                   ("champion", "events", "inventory_id_changes", "n_drifts")}
                                  if isinstance(g.get("item_track"), dict) else None),
                "labels_recorded_gold": g.get("labels_recorded_gold"),
                "labels_vs_raw_mem_recorded": g.get("labels_vs_raw_mem_recorded"),
                "counts": g["counts"],
            })
            for k, v in g["counts"].items():
                totals[k] += v
            totals["games_ok"] += 1
        games.append(e)
    summary = {
        "header": {
            "schema": "replay_oracle_summary/v2",
            "description": "Raw observations from recorded 16.9 replays (client memory reads). "
                           "Independent of any simulator formula.",
            "data_source": dataset_root,
            "patch": PATCH, "patch_alias": "26.9",
            "extraction_script": "ops/replay_oracle_extract.py",
            "extraction_script_sha256_summary_stage": sha256_file(os.path.abspath(__file__)),
            "extraction_script_sha256_per_game": sorted(per_game_sha),
            "script_version": SCRIPT_VERSION,
            "per_game_outputs": out_dir,
            "date": datetime.date.today().isoformat(),
            "params": PARAMS,
            "raw_mem_sha256": hashes,
            "totals": dict(totals),
            "notes": [
                "fountain segment series downsampled to <= %.0f Hz here; full res in per-game files"
                % PARAMS["summary_fountain_hz"],
                "gold chunks (>3.0) for the whole game are in the per-game files only",
                "full-res [0,150]s gold_total series are in the per-game files only",
                "deaths[].payout_deltas is a list aligned to meta.champions "
                "(gold_total delta over payout_window)",
                "deaths[].window_gold_chunks: [hero, gt, delta] for |delta| > chunk_threshold inside "
                "the window (per-game files also list the small periodic increments)",
                "fountain series: dt = seconds since start_gt; hp_max/gold/level given as a scalar when constant over the stretch",
                "levelups / hp_max_changes_not_at_levelup are {fields, rows} tables; hp_max changes "
                "with near_levelup=true are within 1 s of a level-up of the same hero (kept, flagged)",
                "raw_mem 'gold' (current gold) is constant per hero or garbage in this dataset; "
                "see current_gold_checks and labels_recorded_gold",
                "deaths with dead_duration < 2 s are hp<=0 blips: they come back with hp < hp_max "
                "at the same position",
            ],
        },
        "games": games,
    }
    tmp = summary_path + ".tmp_partial"
    if summary_path.endswith(".gz"):
        with gzip.open(tmp, "wt", compresslevel=9) as f:
            json.dump(summary, f, separators=(",", ":"))
    else:
        with open(tmp, "w") as f:
            json.dump(summary, f, separators=(",", ":"))
    os.replace(tmp, summary_path)
    return summary["header"]["totals"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["extract", "index", "summarize"])
    ap.add_argument("game_dirs", nargs="*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-root", default="/mnt/nfs/datasets/lol_replays_16_9_772")
    ap.add_argument("--summary")
    ap.add_argument("--no-hash", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "extract":
        rc = 0
        for gd in a.game_dirs:
            gid = os.path.basename(gd.rstrip("/"))
            try:
                res = extract_game(gd, hash_files=not a.no_hash)
            except Exception as e:  # noqa: BLE001 - record and continue
                import traceback
                res = {"schema": "replay_oracle_game/v1", "game_id": gid, "game_dir": gd,
                       "status": "error", "error": repr(e), "traceback": traceback.format_exc()}
                rc = 1
            write_json_gz(res, os.path.join(a.out, gid + ".json.gz"))
            print(gid, res.get("status"), json.dumps(res.get("counts")), res.get("elapsed_s"), flush=True)
        return rc
    if a.mode == "index":
        idx = build_index(a.out, a.dataset_root)
        print("indexed", idx["n_games"], "missing", idx["game_dirs_without_output"])
        return 0
    if a.mode == "summarize":
        if not a.summary:
            ap.error("--summary required")
        print(json.dumps(build_summary(a.out, a.summary, a.dataset_root)))
        return 0


if __name__ == "__main__":
    sys.exit(main())
