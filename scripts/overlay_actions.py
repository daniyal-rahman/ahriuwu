#!/usr/bin/env python3
"""Render the GROUND-TRUTH action labels onto replay frames, to see what the
model is actually being asked to imitate.

Not a model-prediction overlay (that is scripts/overlay_e2e.py). This draws only
what is in the data:

  * champion screen position                       (cyan circle)
  * cursor position, when the label has one        (yellow crosshair)
  * every click event, projected into THIS frame's camera, flashing white on the
    frame it fires and fading over ~0.5s           (white ring + trail)
  * the HELD movement target the BC loss uses      (green cross)
  * abilities / attacks from label.action          (HUD strip, flashes on fire)
  * a rolling event tape of the last few seconds   (bottom strip)

The point is to answer, visually: are the labels sparse because the PLAYER is
idle, or because our label construction holds a value between events? The click
markers are the raw event stream; the green cross is what we derived from it.

    PYTHONPATH=src python scripts/overlay_actions.py --match NA1_5549981347 \
        --start 2000 --frames 600 --out actions.mp4
"""
import argparse
import glob
import json
import os
import sys
from bisect import bisect_left

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ahriuwu.data.replay_dataset import _Projection   # noqa: E402

ROOT_DEFAULT = "/srv/nfs/datasets/lol_replays_16_9_772"
CYAN, YELLOW, WHITE, GREEN, RED, GREY = ((255, 255, 0), (0, 255, 255), (255, 255, 255),
                                         (0, 255, 0), (0, 0, 255), (150, 150, 150))


def to_px(x, y, lw, lh, S):
    """label space (1280x720) -> frame pixels (SxS, aspect-squished like training)."""
    return int(round(x / lw * S)), int(round(y / lh * S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="NA1_5549981347")
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--out", default="actions_overlay.mp4")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--scale", type=int, default=3, help="upscale factor for legibility")
    a = ap.parse_args()

    mdir = f"{a.root}/{a.match}"
    labels = json.load(open(f"{mdir}/labels.json"))
    fr = labels["frames"]
    lw, lh = labels["screen_resolution"]
    proj = _Projection(labels)

    cj = json.load(open(f"{mdir}/clicks.json"))
    clicks = cj if isinstance(cj, list) else cj.get("clicks", [])
    clicks = sorted(clicks, key=lambda c: c["game_t"])
    ctimes = [c["game_t"] for c in clicks]

    files = sorted(glob.glob(f"{mdir}/frames/*.png"))
    lo = max(0, a.start)
    hi = min(len(files), len(fr), lo + a.frames)
    S = cv2.imread(files[lo]).shape[0]
    up = a.scale
    W = S * up
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, W + 70 * up // 3))
    if not vw.isOpened():
        raise SystemExit(f"could not open {a.out} for writing")

    # map each click to its frame (nearest gt), same timebase the dataset uses
    gts = [f.get("gt", np.nan) for f in fr]
    click_frame = {}
    for c in clicks:
        i = bisect_left(gts, c["game_t"])
        i = min(max(i, 0), len(gts) - 1)
        if i > 0 and abs(gts[i - 1] - c["game_t"]) < abs(gts[i] - c["game_t"]):
            i -= 1
        click_frame.setdefault(i, []).append(c)

    tape = []           # (frame_idx, text) recent events
    n_click = n_cast = 0
    held = None         # last click's world point -> the "held" BC target

    for i in range(lo, hi):
        im = cv2.imread(files[i])
        if im is None:
            continue
        im = cv2.resize(im, (W, W), interpolation=cv2.INTER_NEAREST)
        lab = fr[i].get("label") or {}
        gt = fr[i].get("gt", 0.0)

        # --- camera for THIS frame, recovered from champion world->screen ---
        cam = None
        cw, cs = lab.get("champion_world"), lab.get("champion_screen")
        if isinstance(cw, list) and isinstance(cs, list):
            cam = proj.invert(cw[0], cw[1], cs[0] + 0.5, cs[1] + 0.5)
            px, py = to_px(cs[0], cs[1], lw, lh, S)
            cv2.circle(im, (px * up, py * up), 7 * up, CYAN, 2)

        # --- cursor (this is the aim point at keypress time) ---
        cur = (lab.get("cursor") or {}).get("screen")
        if isinstance(cur, list) and len(cur) == 2:
            qx, qy = to_px(cur[0], cur[1], lw, lh, S)
            cv2.drawMarker(im, (qx * up, qy * up), YELLOW, cv2.MARKER_CROSS, 9 * up, 2)

        # --- new click events on this frame ---
        for c in click_frame.get(i, []):
            held = (c["x"], c["z"])
            n_click += 1
            tape.append((i, f"CLICK ({c['x']:.0f},{c['z']:.0f})"))
        # fading trail of recent clicks (0.5s)
        for back in range(0, 11):
            for c in click_frame.get(i - back, []):
                if cam is None:
                    continue
                nx, ny = proj.project_norm(c["x"], c["z"], cam[0], cam[1])
                mx, my = int(nx * S * up), int(ny * S * up)
                r = int((3 + back) * up)
                shade = int(255 * (1 - back / 12))
                cv2.circle(im, (mx, my), r, (shade, shade, shade), 2)

        # --- the HELD target the BC movement loss actually uses ---
        if held is not None and cam is not None:
            nx, ny = proj.project_norm(held[0], held[1], cam[0], cam[1])
            hx, hy = int(nx * S * up), int(ny * S * up)
            cv2.drawMarker(im, (hx, hy), GREEN, cv2.MARKER_TILTED_CROSS, 11 * up, 2)
            if cs is not None:
                px, py = to_px(cs[0], cs[1], lw, lh, S)
                cv2.line(im, (px * up, py * up), (hx, hy), GREEN, 1)

        # --- abilities / attacks ---
        act = lab.get("action") or {}
        atype, spell = act.get("type"), act.get("spell")
        if atype and atype != "idle":
            n_cast += 1
            tape.append((i, f"{atype.upper()} {spell or ''}".strip()))
            cv2.putText(im, f"{atype.upper()} {spell or ''}", (8 * up, 18 * up),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35 * up, RED, 1, cv2.LINE_AA)

        # --- HUD ---
        hud = np.zeros((70 * up // 3, W, 3), np.uint8)
        el = max(i - lo + 1, 1) / a.fps
        cv2.putText(hud, f"f{i}  t={gt:6.1f}s   clicks={n_click} ({n_click/el:4.2f}/s)"
                         f"   actions={n_cast} ({n_cast/el:4.2f}/s)",
                    (6 * up, 10 * up), cv2.FONT_HERSHEY_SIMPLEX, 0.30 * up, WHITE, 1, cv2.LINE_AA)
        tape = [(f_, t) for f_, t in tape if i - f_ < 40]
        cv2.putText(hud, " | ".join(t for _, t in tape[-4:])[:110],
                    (6 * up, 20 * up), cv2.FONT_HERSHEY_SIMPLEX, 0.26 * up, GREY, 1, cv2.LINE_AA)
        cv2.putText(hud, "cyan=champ  yellow=cursor  white=click event  green=HELD target (BC label)",
                    (6 * up, 20 * up + 9 * up), cv2.FONT_HERSHEY_SIMPLEX, 0.24 * up, GREY, 1, cv2.LINE_AA)
        vw.write(np.vstack([im, hud]))

    vw.release()
    dur = (hi - lo) / a.fps
    print(f"wrote {a.out}  frames {lo}..{hi}  ({dur:.1f}s of gameplay)")
    print(f"  clicks in window : {n_click}  = {n_click/dur:.2f}/s")
    print(f"  actions in window: {n_cast}  = {n_cast/dur:.2f}/s")
    print(f"  combined         : {(n_click+n_cast)/dur:.2f}/s   (APM {(n_click+n_cast)*60/dur:.0f})")


if __name__ == "__main__":
    main()
