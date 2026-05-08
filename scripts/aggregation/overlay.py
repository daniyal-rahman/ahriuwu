#!/usr/bin/env python3
"""
Render an overlay video from a pipeline.py output directory.

One stage: reads labels.json + raw_cam.json + raw_mem.json + frames/*.png,
writes overlay.mp4 with:
  - frame upsized 352x352 -> out-w x out-h (default 1280x720, undoes the
    pipeline's 1280x720 -> 352x352 squish)
  - HUD top-left (game time, level, gold, hp)
  - rolling text log top-right (recent actions, exact name + timing)
  - hero crosshair at the per-frame screen pos pipeline already recorded
  - click markers projected from world coords (--click-events optional)

Camera comes from raw_cam.json. raw_cam samples are wall-time-stamped from
pass1; we map wall->gt via raw_mem.json (which has both), then per-frame
look up cam by gt. Frames whose gt falls outside cam coverage (by more than
--cam-tolerance-s) draw a red "NO CAM SAMPLE" warning and skip click/cast
projection — the HUD + hero crosshair still render since those use
pre-projected screen coords from labels.json.

Projection magic numbers default from labels.json's "projection" block but
can be overridden via CLI for tuning.

Usage:
  python scripts/overlay.py --match-dir C:\\tmp\\replay_data\\NA1_5553742746
"""
import argparse
import bisect
import json
import math
import os
import sys

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


# ─────────────────── projection ───────────────────
def make_projector(out_w, out_h, fov_v_deg, tilt_deg, cam_y, floor_y):
    fov_v = math.radians(fov_v_deg)
    fov_h = 2 * math.atan(math.tan(fov_v / 2) * out_w / out_h)
    tan_h, tan_v = math.tan(fov_h / 2), math.tan(fov_v / 2)
    cos_t = math.cos(math.radians(tilt_deg))
    sin_t = math.sin(math.radians(tilt_deg))

    def project(map_x, map_z, cam_x, cam_z):
        # cam_x, cam_z come from /replay/render (raw_cam.json) — already
        # offset south-of-hero by the engine, so no extra _CAM_Z_OFFSET here.
        dx = map_x - cam_x
        dy = floor_y - cam_y
        dz = map_z - cam_z
        vy = dy * cos_t + dz * sin_t
        vz = -dy * sin_t + dz * cos_t
        if vz <= 10:
            return None
        nx = 0.5 + (dx / vz) / tan_h * 0.5
        ny = 0.5 - (vy / vz) / tan_v * 0.5
        return int(nx * out_w), int(ny * out_h)

    return project


# ─────────────────── wall <-> gt + cam lookups ───────────────────
def build_wall_to_gt(raw_mem):
    """Linear interpolator wall -> gt from raw_mem samples (sorted ascending)."""
    walls, gts = [], []
    last_w = -1.0
    for s in raw_mem:
        w, g = s.get("wall"), s.get("gt")
        if w is None or g is None or w <= last_w:
            continue
        walls.append(w)
        gts.append(g)
        last_w = w
    if len(walls) < 2:
        raise SystemExit("raw_mem.json needs >=2 (wall, gt) samples to build mapping")

    def f(wall):
        i = bisect.bisect_left(walls, wall)
        if i <= 0:
            slope = (gts[1] - gts[0]) / (walls[1] - walls[0])
            return gts[0] + slope * (wall - walls[0])
        if i >= len(walls):
            slope = (gts[-1] - gts[-2]) / (walls[-1] - walls[-2])
            return gts[-1] + slope * (wall - walls[-1])
        f_ = (wall - walls[i - 1]) / (walls[i] - walls[i - 1])
        return gts[i - 1] + f_ * (gts[i] - gts[i - 1])

    return f


def build_gt_to_cam(raw_cam, wall_to_gt):
    """Sort cam samples by mapped gt, return interpolator gt -> (cam_x, cam_z)."""
    pairs = []
    for s in raw_cam:
        w = s.get("wall")
        if w is None:
            continue
        g = wall_to_gt(w)
        pairs.append((g, s["cx"], s["cz"]))
    pairs.sort(key=lambda p: p[0])
    if not pairs:
        raise SystemExit("raw_cam.json mapped to 0 gt samples — check raw_mem coverage")
    gts = [p[0] for p in pairs]
    cxs = [p[1] for p in pairs]
    czs = [p[2] for p in pairs]

    def cam_at(gt):
        i = bisect.bisect_left(gts, gt)
        if i <= 0:
            return cxs[0], czs[0]
        if i >= len(gts):
            return cxs[-1], czs[-1]
        f_ = (gt - gts[i - 1]) / (gts[i] - gts[i - 1])
        return cxs[i - 1] + f_ * (cxs[i] - cxs[i - 1]), czs[i - 1] + f_ * (czs[i] - czs[i - 1])

    return cam_at, gts[0], gts[-1]


# ─────────────────── action extraction ───────────────────
def extract_actions(frames):
    """Walk per-frame action labels, dedupe (type, spell) within 0.3s.
    Returns list of dicts: {gt, type, label, screen}. label is the raw spell
    name from memory — not collapsed — so you can see exactly what was read.
    """
    out = []
    last_emit = {}
    for fr in frames:
        lab = fr.get("label")
        if not lab:
            continue
        act = lab.get("action") or {}
        atype = act.get("type") or "idle"
        if atype == "idle":
            continue
        gt = fr["gt"]
        spell = act.get("spell") or atype.upper()
        screen = act.get("screen")  # pre-projected by pipeline.py
        slabel = spell  # show the actual spell name from memory verbatim
        key = (atype, slabel)
        if gt - last_emit.get(key, -10) < 0.3:
            continue
        last_emit[key] = gt
        out.append({"gt": gt, "type": atype, "label": slabel, "screen": screen})
    return out


# ─────────────────── drawing ───────────────────
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
COLOR = {
    "AA": (180, 180, 180),
    "Q": (0, 200, 255),
    "W": (255, 200, 0),
    "E": (0, 255, 0),
    "R": (0, 0, 255),
    "B": (255, 100, 255),
    "CLICK": (0, 255, 255),
}


def color_for(label):
    """Pick a marker color from a raw or short spell label.
    Match the QWER slot from a champion-spell name like "GarenQ", "GarenQAttack"
    (Q with an attack-on-cast modifier), "BelvethW", or a cast-log entry like
    "Q GarenQ" / "D SummonerFlash"."""
    s = (label or "").lower()
    if "basicattack" in s or s == "aa":
        return COLOR["AA"]
    if "recall" in s or s == "b":
        return COLOR["B"]
    # Strip the "attack" suffix that wraps Q-on-hit modifiers (GarenQAttack → GarenQ).
    if s.endswith("attack"):
        s = s[:-6]
    # Champion-spell name: slot is the last char (GarenQ → q, BelvethW → w).
    if s and s[-1] in "qwer":
        return COLOR[s[-1].upper()]
    # Cast-log "<slot> <spell_name>" form: slot is first char.
    if s and s[0] in "qwer":
        return COLOR[s[0].upper()]
    return WHITE


def fmt_clock(gt):
    m = int(gt // 60)
    return f"{m}:{gt - 60 * m:05.2f}"


def put_text(img, text, xy, scale=0.5, color=WHITE, thickness=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_log(img, log, w, log_n, scale=0.45):
    if not log:
        return
    show = log[-log_n:]
    x = w - 220
    y = 25
    for i, (gt, label) in enumerate(show):
        put_text(img, f"{fmt_clock(gt)}  {label}", (x, y + i * 20), scale, color_for(label))


def draw_hud(img, gt, frame_idx, total, stats):
    put_text(img, fmt_clock(gt), (12, 28), 0.7, WHITE, 2)
    put_text(img, f"F{frame_idx}/{total}", (12, 50), 0.4, (200, 200, 200))
    if not stats:
        return
    y = 75
    if stats.get("level") is not None:
        put_text(img, f"LVL {stats['level']}", (12, y), 0.5, (100, 255, 255))
        y += 20
    gc = stats.get("gold")          # current / unspent
    gtot = stats.get("gold_total")  # lifetime earned
    if gc is not None and gtot is not None:
        put_text(img, f"GOLD {gc:.0f}/{gtot:.0f}", (12, y), 0.5, (0, 215, 255))
        y += 20
    elif gtot is not None:
        put_text(img, f"GOLD {gtot:.0f}", (12, y), 0.5, (0, 215, 255))
        y += 20
    elif gc is not None:
        put_text(img, f"GOLD {gc:.0f}", (12, y), 0.5, (0, 215, 255))
        y += 20
    hp, hpm = stats.get("hp"), stats.get("hp_max")
    if hp is not None:
        if hpm and abs(hpm) < 1e7:
            put_text(img, f"HP {hp:.0f}/{hpm:.0f}", (12, y), 0.5, (0, 255, 0))
        else:
            put_text(img, f"HP {hp:.0f}", (12, y), 0.5, (0, 255, 0))


def draw_marker(img, sx, sy, color, radius=10):
    h, w = img.shape[:2]
    if not (-50 <= sx <= w + 50 and -50 <= sy <= h + 50):
        return
    sx = max(0, min(int(sx), w - 1))
    sy = max(0, min(int(sy), h - 1))
    cv2.circle(img, (sx, sy), radius, color, 2, cv2.LINE_AA)
    cv2.circle(img, (sx, sy), 2, color, -1)


def draw_hero(img, sx, sy):
    if sx is None or sy is None:
        return
    sx, sy = int(sx), int(sy)
    cv2.circle(img, (sx, sy), 14, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(img, (sx - 10, sy), (sx + 10, sy), (0, 255, 0), 1)
    cv2.line(img, (sx, sy - 10), (sx, sy + 10), (0, 255, 0), 1)


# ─────────────────── main ───────────────────
def main():
    ap = argparse.ArgumentParser(description="Render overlay video from pipeline.py output")
    ap.add_argument("--match-dir", required=True)
    ap.add_argument("--output", default=None, help="default <match-dir>/overlay.mp4")
    ap.add_argument("--out-w", type=int, default=1280)
    ap.add_argument("--out-h", type=int, default=720)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="cap output length in real seconds (= max-seconds * fps frames). "
                         "Overrides --max-frames if set. 0 = full video.")
    ap.add_argument("--click-events", default=None,
                    help="optional click events JSON (poll_click_alloc.py format)")
    ap.add_argument("--log-len", type=int, default=12)
    ap.add_argument("--attack-persist-frames", type=int, default=3)
    ap.add_argument("--ability-persist-frames", type=int, default=8)
    ap.add_argument("--recall-persist-frames", type=int, default=60)
    # projection — defaults read from labels.projection
    ap.add_argument("--fov-v-deg", type=float, default=None)
    ap.add_argument("--tilt-deg", type=float, default=None)
    ap.add_argument("--cam-y", type=float, default=None)
    ap.add_argument("--floor-y", type=float, default=52.0)
    ap.add_argument("--cam-tolerance-s", type=float, default=1.0,
                    help="flag a frame as no-cam if its gt is further than this from any cam sample")
    args = ap.parse_args()

    md = args.match_dir
    out_path = args.output or os.path.join(md, "overlay.mp4")

    print(f"Loading {md}...")

    def _load_json(name):
        p = os.path.join(md, name)
        if not os.path.exists(p):
            raise SystemExit(f"FATAL: {p} missing — pipeline output incomplete?")
        with open(p) as f:
            return json.load(f)

    labels = _load_json("labels.json")
    raw_cam = _load_json("raw_cam.json")
    raw_mem = _load_json("raw_mem.json")
    if "frames" not in labels:
        raise SystemExit("FATAL: labels.json has no 'frames' key — schema mismatch")
    frames = labels["frames"]

    proj_meta = labels.get("projection", {})
    if not proj_meta:
        print(f"  WARN: labels.json has no 'projection' block — using defaults / CLI", flush=True)

    def _pick(cli_val, meta_key, default, src_label):
        if cli_val is not None:
            return cli_val, f"{src_label}=CLI"
        if meta_key in proj_meta:
            return proj_meta[meta_key], f"{src_label}=labels"
        return default, f"{src_label}=default"

    fov_v, fov_src = _pick(args.fov_v_deg, "fov_v_deg", 40.0, "fov_v")
    tilt,  tilt_src = _pick(args.tilt_deg, "tilt_deg", 56.0, "tilt")
    cam_y, camy_src = _pick(args.cam_y, "cam_y", 1912.0, "cam_y")
    print(f"  {len(frames)} frames, {len(raw_cam)} cam samples, {len(raw_mem)} mem samples")
    print(f"  champion (label key): {labels.get('champion')!r}")
    print(f"  projection: fov_v={fov_v} tilt={tilt} cam_y={cam_y} floor_y={args.floor_y} "
          f"({fov_src} {tilt_src} {camy_src})")
    print(f"  output: {args.out_w}x{args.out_h} @ {args.fps}fps -> {out_path}")

    print("Building wall->gt (from raw_mem) and gt->cam (from raw_cam)...")
    wall_to_gt = build_wall_to_gt(raw_mem)
    cam_at, cam_gt_min, cam_gt_max = build_gt_to_cam(raw_cam, wall_to_gt)
    print(f"  cam coverage: gt {cam_gt_min:.1f} .. {cam_gt_max:.1f}s")
    if frames:
        print(f"  frame coverage: gt {frames[0]['gt']:.1f} .. {frames[-1]['gt']:.1f}s")

    # actions snapped to frame index
    actions = extract_actions(frames)
    print(f"Actions extracted: {len(actions)}")
    actions_by_frame = {}
    if frames:
        gt0 = frames[0]["gt"]
        step = 1.0 / args.fps
        for a in actions:
            i = int((a["gt"] - gt0) / step)
            if 0 <= i < len(frames):
                actions_by_frame.setdefault(i, []).append(a)

    # click events optional — default to <match-dir>/clicks.json if present.
    click_path = args.click_events
    if not click_path:
        auto_path = os.path.join(md, "clicks.json")
        if os.path.exists(auto_path):
            click_path = auto_path
            print(f"  auto-using {auto_path}")
        else:
            print(f"  no clicks.json found at {auto_path} — click/cast markers disabled")
    click_by_frame = {}
    cast_by_frame = {}
    n_dropped_no_t = 0
    if click_path:
        with open(click_path) as f:
            ce = json.load(f)
        clicks = ce.get("clicks", []) + ce.get("events", [])
        casts = ce.get("casts", [])
        gt0 = frames[0]["gt"] if frames else 0.0
        step = 1.0 / args.fps
        for c in clicks:
            t = c.get("game_time", c.get("game_t"))
            if t is None:
                n_dropped_no_t += 1
                continue
            i = int((t - gt0) / step)
            if 0 <= i < len(frames):
                click_by_frame.setdefault(i, []).append({
                    "world_x": c["x"], "world_z": c["z"], "gt": t,
                })
        for c in casts:
            t = c.get("game_time", c.get("game_t"))
            if t is None:
                n_dropped_no_t += 1
                continue
            i = int((t - gt0) / step)
            if 0 <= i < len(frames):
                cast_by_frame.setdefault(i, []).append({
                    "slot": c.get("slot", "?"),
                    "spell_name": c.get("spell_name") or "",
                    "hero_x": c.get("hero_x"),
                    "hero_z": c.get("hero_z"),
                    "gt": t,
                })
        print(f"Click events: {len(clicks)} clicks + {len(casts)} casts from {click_path}")
        if n_dropped_no_t:
            print(f"  WARN: {n_dropped_no_t} click/cast event(s) dropped (no game_time field)")

    project = make_projector(args.out_w, args.out_h, fov_v, tilt, cam_y, args.floor_y)
    if "screen_resolution" not in labels:
        print(f"  WARN: labels.json has no 'screen_resolution' — assuming 1280x720. "
              f"If pipeline ran at a different resolution, all action markers will be mis-scaled.")
    pipe_w, pipe_h = labels.get("screen_resolution", [1280, 720])
    sx_scale = args.out_w / pipe_w
    sy_scale = args.out_h / pipe_h

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, args.fps, (args.out_w, args.out_h))
    if not vw.isOpened():
        print(f"ERROR: cannot open VideoWriter at {out_path}. "
              f"Check codec (mp4v expected), parent-dir permissions, and free disk space.")
        return 1

    frames_dir = os.path.join(md, "frames")
    if args.max_seconds > 0:
        n = int(args.max_seconds * args.fps)
    elif args.max_frames > 0:
        n = args.max_frames
    else:
        n = len(frames)
    n = min(n, len(frames))
    log = []  # (gt, label)
    pending_markers = []  # [{sx,sy,color,label,frames_left}]
    PERSIST = {  # frames-to-show, by action type
        "attack":   args.attack_persist_frames,
        "ability":  args.ability_persist_frames,
        "recall":   args.recall_persist_frames,
        "summoner": args.ability_persist_frames,
        "other":    args.ability_persist_frames,
    }
    n_skip_png = 0
    n_no_cam = 0
    n_dropped_proj = 0       # click/cast world coords behind cam (project returned None)
    n_cast_no_hero = 0       # cast event missing hero_x/hero_z
    n_fallback_vh = 0        # frames where champion_screen/_stats fell back to visible_heroes
    INLINE_LOG_LIMIT = 3
    print(f"\nRendering {n} frames...")
    for i in range(n):
        fr = frames[i]
        gt = fr["gt"]
        idx = fr["frame"]
        png = os.path.join(frames_dir, f"{idx:06d}.png")
        img = cv2.imread(png)
        if img is None:
            if n_skip_png < INLINE_LOG_LIMIT:
                print(f"  WARN: unreadable PNG at frame {idx} ({png})")
            n_skip_png += 1
            img = np.zeros((args.out_h, args.out_w, 3), dtype=np.uint8)
        elif (img.shape[1], img.shape[0]) != (args.out_w, args.out_h):
            img = cv2.resize(img, (args.out_w, args.out_h), interpolation=cv2.INTER_LINEAR)

        cam_ok = (cam_gt_min - args.cam_tolerance_s) <= gt <= (cam_gt_max + args.cam_tolerance_s)
        if cam_ok:
            cam_x, cam_z = cam_at(gt)
        else:
            if n_no_cam < INLINE_LOG_LIMIT:
                print(f"  WARN: no cam sample for frame {idx} gt={gt:.2f} "
                      f"(cam covers {cam_gt_min:.1f}..{cam_gt_max:.1f}, tolerance={args.cam_tolerance_s}s)")
            cam_x = cam_z = 0.0
            n_no_cam += 1

        for a in actions_by_frame.get(i, []):
            log.append((a["gt"], a["label"]))
            scr = a.get("screen")
            if scr and len(scr) == 2:
                # action.screen comes from pipeline at 1280x720 — scale to output
                sx_, sy_ = scr[0] * sx_scale, scr[1] * sy_scale
                pending_markers.append({
                    "sx": sx_, "sy": sy_,
                    "color": color_for(a["label"]),
                    "label": a["label"],
                    "frames_left": PERSIST.get(a["type"], args.ability_persist_frames),
                })

        if cam_ok:
            for c in click_by_frame.get(i, []):
                pos = project(c["world_x"], c["world_z"], cam_x, cam_z)
                if pos:
                    log.append((c["gt"], "CLICK"))
                    pending_markers.append({
                        "sx": pos[0], "sy": pos[1],
                        "color": COLOR["CLICK"], "label": "CLICK",
                        "frames_left": args.ability_persist_frames,
                    })
                else:
                    n_dropped_proj += 1
            for c in cast_by_frame.get(i, []):
                slot = c["slot"]; nm = c["spell_name"]
                lbl = f"{slot} {nm}" if nm else slot
                log.append((c["gt"], lbl))
                hx, hz = c.get("hero_x"), c.get("hero_z")
                if hx is None or hz is None:
                    n_cast_no_hero += 1
                    continue
                pos = project(hx, hz, cam_x, cam_z)
                if pos:
                    pending_markers.append({
                        "sx": pos[0], "sy": pos[1],
                        "color": color_for(slot), "label": lbl,
                        "frames_left": args.ability_persist_frames,
                    })
                else:
                    n_dropped_proj += 1

        # draw + age pending action markers (screen-locked, don't track cam —
        # they fired on a one-off frame and stay where they were drawn)
        next_pending = []
        for mk in pending_markers:
            draw_marker(img, mk["sx"], mk["sy"], mk["color"])
            put_text(img, mk["label"], (int(mk["sx"]) + 14, int(mk["sy"]) - 8),
                     0.45, mk["color"], 1)
            mk["frames_left"] -= 1
            if mk["frames_left"] > 0:
                next_pending.append(mk)
        pending_markers = next_pending

        lab = fr.get("label") or {}
        # champion_screen/champion_stats are the focus champion's per-frame
        # data. Older labels.json may still have it under garen_screen — try
        # both. If both missing or stats all-zero, look up the focus champion
        # by name in visible_heroes (gold/gold_total aren't on visible_heroes
        # entries so HUD will be partial in this fallback).
        gs = lab.get("champion_screen") or lab.get("garen_screen")
        stats = lab.get("champion_stats") or lab.get("garen_stats")
        if (not gs) or (stats and not any(stats.values())):
            focus_name = labels.get("champion")
            vh_list = lab.get("visible_heroes") or []
            vh = next((v for v in vh_list if v.get("name") == focus_name), None)
            if vh is None and vh_list:
                vh = vh_list[0]
            if vh:
                if n_fallback_vh < INLINE_LOG_LIMIT:
                    print(f"  WARN: frame {idx} fell back to visible_heroes for HUD stats "
                          f"(focus={focus_name!r}, matched={vh.get('name')!r})")
                n_fallback_vh += 1
                gs = vh.get("screen") or gs
                stats = {
                    "level": vh.get("level"),
                    "hp": vh.get("hp"),
                    "hp_max": vh.get("hp_max"),
                    "gold_total": vh.get("gold_total") or vh.get("gold"),
                }
        if gs:
            draw_hero(img, gs[0] * sx_scale, gs[1] * sy_scale)
        draw_hud(img, gt, idx, n, stats)
        draw_log(img, log, args.out_w, args.log_len)
        if not cam_ok:
            put_text(img, "NO CAM SAMPLE", (args.out_w // 2 - 80, 25), 0.6, (0, 0, 255), 2)

        vw.write(img)
        if (i + 1) % 500 == 0:
            print(f"  frame {i+1}/{n}  gt={gt:.1f}  log_total={len(log)}")

    vw.release()
    print("\nDone.")
    print(f"  output: {out_path}")
    print(f"  written: {n}")
    print(f"  unreadable_png:    {n_skip_png}")
    print(f"  no_cam_coverage:   {n_no_cam}")
    print(f"  dropped_projection:{n_dropped_proj}  (click/cast world coord behind cam)")
    print(f"  cast_no_hero_pos:  {n_cast_no_hero}  (cast event lacked hero_x/hero_z)")
    print(f"  visible_heroes_fallback: {n_fallback_vh}  (frames using fallback HUD source)")
    print(f"  total actions logged: {len(log)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
