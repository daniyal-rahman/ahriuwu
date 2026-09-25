#!/usr/bin/env python3
"""Corpus scan: distill every labels.json + clicks.json into compact npz arrays.

Writes <out>/<match_id>.npz with per-frame arrays so all downstream audit
analyses (movement quantization, attack undercount, corrupt stats, timestamp
gaps, alignment) run off the cache instead of re-parsing 23MB JSONs.
"""
import json
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    import orjson

    def jload(p):
        with open(p, "rb") as f:
            return orjson.loads(f.read())
except ImportError:
    def jload(p):
        with open(p) as f:
            return json.load(f)

ROOT = Path("/srv/nfs/datasets/lol_replays_16_9_772")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/srv/nfs/projects/ahriuwu/scratchpad/audit_cache")
OUT.mkdir(parents=True, exist_ok=True)

ATYPE = {None: 0, "idle": 1, "attack": 2, "ability": 3, "recall": 4, "move": 5}


def f(x):
    return np.float64(x) if x is not None else np.nan


def scan(mid):
    outp = OUT / f"{mid}.npz"
    if outp.exists():
        return mid, "cached"
    d = ROOT / mid
    L = jload(d / "labels.json")
    frames = L.get("frames") or []
    T = len(frames)
    cols = {}
    for k in ("gt", "frame_field", "hp", "hp_max", "gold", "gold_total", "level",
              "cs_x", "cs_y", "cw_x", "cw_y", "cur_sx", "cur_sy", "cur_wx", "cur_wy",
              "act_sx", "act_sy", "mv_hx", "mv_hy", "mv_speed",
              "opp_hp", "opp_hp_max", "opp_screen_ok", "n_vis", "wp_x", "wp_y"):
        cols[k] = np.full(T, np.nan)
    labeled = np.zeros(T, dtype=bool)
    atype = np.zeros(T, dtype=np.int8)
    spell = []
    opp_name = L.get("lane_opponent")
    stride_lf = np.full(T, np.nan)
    n_items = np.zeros(T, dtype=np.int8)

    for i, fr in enumerate(frames):
        cols["gt"][i] = f(fr.get("gt"))
        ff = fr.get("frame")
        cols["frame_field"][i] = f(ff)
        lab = fr.get("label")
        sp = None
        if not lab:
            spell.append("")
            continue
        labeled[i] = True
        cs = lab.get("champion_stats") or {}
        cols["hp"][i] = f(cs.get("hp"))
        cols["hp_max"][i] = f(cs.get("hp_max"))
        cols["gold"][i] = f(cs.get("gold"))
        cols["gold_total"][i] = f(cs.get("gold_total"))
        cols["level"][i] = f(cs.get("level"))
        scr = lab.get("champion_screen")
        if scr and len(scr) == 2:
            cols["cs_x"][i], cols["cs_y"][i] = f(scr[0]), f(scr[1])
        wld = lab.get("champion_world")
        if wld and len(wld) == 2:
            cols["cw_x"][i], cols["cw_y"][i] = f(wld[0]), f(wld[1])
        cur = lab.get("cursor") or {}
        c = cur.get("screen")
        if c and len(c) == 2:
            cols["cur_sx"][i], cols["cur_sy"][i] = f(c[0]), f(c[1])
        c = cur.get("world")
        if c and len(c) == 2:
            cols["cur_wx"][i], cols["cur_wy"][i] = f(c[0]), f(c[1])
        act = lab.get("action") or {}
        atype[i] = ATYPE.get(act.get("type"), 9)
        sp = act.get("spell")
        a = act.get("screen")
        if a and len(a) == 2:
            cols["act_sx"][i], cols["act_sy"][i] = f(a[0]), f(a[1])
        mv = lab.get("movement") or {}
        hs = mv.get("heading_screen")
        if hs and len(hs) == 2:
            cols["mv_hx"][i], cols["mv_hy"][i] = f(hs[0]), f(hs[1])
        cols["mv_speed"][i] = f(mv.get("speed"))
        wp = lab.get("waypoint")
        if wp and len(wp) == 2:
            cols["wp_x"][i], cols["wp_y"][i] = f(wp[0]), f(wp[1])
        vh = lab.get("visible_heroes") or []
        cols["n_vis"][i] = len(vh)
        for h in vh:
            if h.get("name") == opp_name:
                cols["opp_hp"][i] = f(h.get("hp"))
                cols["opp_hp_max"][i] = f(h.get("hp_max"))
                cols["opp_screen_ok"][i] = 1.0 if h.get("screen") else 0.0
                break
        inv = lab.get("inventory") or []
        n_items[i] = sum(1 for it in inv if it)
        for it in inv:
            if it and it.get("id") == 6631:
                stride_lf[i] = f(it.get("lf"))
                break
        spell.append(sp or "")

    # clicks.json
    casts_t, casts_name = [], []
    cp = d / "clicks.json"
    clicks_keys = []
    n_clicks = 0
    if cp.exists():
        C = jload(cp)
        clicks_keys = list(C.keys())
        for c in (C.get("casts") or []):
            gt = c.get("game_t")
            if gt is None:
                gt = c.get("game_time")
            casts_t.append(np.nan if gt is None else float(gt))
            casts_name.append(str(c.get("spell_name")))
        n_clicks = len(C.get("clicks") or []) if isinstance(C.get("clicks"), list) else 0

    meta = {
        "match_id": mid,
        "fps": L.get("fps"),
        "screen_resolution": L.get("screen_resolution"),
        "frame_resolution": L.get("frame_resolution"),
        "total_frames": L.get("total_frames"),
        "T": T,
        "champion": L.get("champion"),
        "team": L.get("team"),
        "lane_opponent": opp_name,
        "action_distribution": L.get("action_distribution"),
        "projection": L.get("projection"),
        "label_top_keys": list(L.keys()),
        "clicks_keys": clicks_keys,
        "n_click_events": n_clicks,
        "n_png": len(list((d / "frames").glob("*.png"))) if (d / "frames").is_dir() else -1,
    }
    np.savez_compressed(
        outp,
        labeled=labeled, atype=atype, spell=np.array(spell),
        stride_lf=stride_lf, n_items=n_items,
        casts_t=np.array(casts_t, dtype=np.float64),
        casts_name=np.array(casts_name),
        meta=np.array([json.dumps(meta)]),
        **cols,
    )
    return mid, f"ok T={T}"


def _safe(mid):
    try:
        return scan(mid)
    except Exception:
        return mid, "ERR " + traceback.format_exc(limit=3)


if __name__ == "__main__":
    valid = json.load(open("/srv/nfs/projects/ahriuwu/scratchpad/valid_games.json"))
    mids = valid["both"] if isinstance(valid, dict) else valid
    mids = sorted(set(mids))
    print(f"{len(mids)} matches -> {OUT}", flush=True)
    nw = int(os.environ.get("NW", "6"))
    with Pool(nw) as pool:
        for i, (mid, st) in enumerate(pool.imap_unordered(_safe, mids)):
            print(f"[{i+1}/{len(mids)}] {mid}: {st}", flush=True)
