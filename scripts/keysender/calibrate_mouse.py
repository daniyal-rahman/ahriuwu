#!/usr/bin/env python3
"""Measure the relative-mouse gadget's units->pixels mapping, from the video feed.

The Pi gadget is a RELATIVE mouse: we can only send deltas, never ask where the
cursor is. So dead reckoning needs one number per axis — how many mouse units it
takes to cross the screen (MOUSE_SPAN in keysender/hybrid_sender.py). This script
measures it instead of guessing:

  1. Slam the cursor into the top-left corner. The OS clamps there, so that is
     the ONE position we can know without observing anything.
  2. From that known origin, send an exact number of axis-aligned reports.
  3. Read the cursor's real position back off the game's video stream by
     template-matching the League cursor.
  4. Repeat over a sweep and fit pixels-per-unit per axis.

Step 3 is the only ground truth available. The detector was validated against
hand-measured cursor positions to within ~3 px.

WHY AXIS-ALIGNED: with Windows "Enhance pointer precision" on, pixels travelled
scale with a report's MAGNITUDE, so a diagonal (30,30) carries each axis further
than a (30,0). Mixing them makes units-per-screen depend on the path taken. This
sweep therefore moves one axis at a time, matching what _mouse_loop does live.

Nothing is clicked — the cursor only moves, so this is safe to run mid-game.

    python scripts/calibrate_mouse.py --host 192.168.1.144
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "keysender"))
from play_live import StreamCapture                      # noqa: E402
from hybrid_sender import HybridKeyboard                 # noqa: E402

TPL_TIP = (3, 3)          # cursor hotspot offset inside cursor_tpl.png


class CursorFinder:
    """Locate the League cursor in a frame.

    Search is restricted to the game world (ROI). The HUD sidebar, the minimap
    and especially the practice-tool cheat panel contain high-contrast glyphs
    that template-match the cursor well enough to beat it, and a false positive
    is worse than a miss here: it is STATIC, so two measurements return the same
    coordinate and the move reads as zero travel.
    """

    def __init__(self, tpl_path, roi=(60, 40, 1040, 600), min_score=0.70):
        self.tpl = cv2.imread(tpl_path)
        if self.tpl is None:
            raise SystemExit(f"cursor template not found: {tpl_path}")
        self.roi = roi
        self.min_score = min_score

    def find(self, frame_rgb01):
        """-> (x, y, score) in FULL-frame coords. Frame is RGB float 0..1."""
        img = (np.clip(frame_rgb01, 0, 1) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        x0, y0, x1, y1 = self.roi
        res = cv2.matchTemplate(img[y0:y1, x0:x1], self.tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        return x0 + loc[0] + TPL_TIP[0], y0 + loc[1] + TPL_TIP[1], float(score)


def slam_corner(kb, chunk, reps):
    """Drive hard into the top-left corner; the clamp makes the position known."""
    for _ in range(reps):
        kb._send(f"mouse {-chunk} 0")
        time.sleep(0.002)
    for _ in range(reps):
        kb._send(f"mouse 0 {-chunk}")
        time.sleep(0.002)


def step(kb, chunk, n, axis):
    for _ in range(n):
        kb._send(f"mouse {chunk} 0" if axis == "x" else f"mouse 0 {chunk}")
        time.sleep(0.002)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.1.144")
    ap.add_argument("--udp-port", type=int, default=5000)
    ap.add_argument("--stream-size", default="1280x720")
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--settle", type=float, default=0.9,
                    help="Seconds to wait for the moved cursor to show up in the stream "
                         "(encode + network + decode latency).")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(__file__), "cursor_tpl.png"))
    ap.add_argument("--out", default=None, help="Write raw samples here as JSON.")
    args = ap.parse_args()

    w, h = map(int, args.stream_size.split("x"))
    reps = int(max(w, h) / args.chunk) + 6          # generous: guarantees the clamp
    finder = CursorFinder(args.template)
    cap = StreamCapture(port=args.udp_port, size=(w, h))
    cap.wait_first()
    kb = HybridKeyboard(host=args.host, mouse=False)  # drive raw; no chase loop
    time.sleep(0.3)

    # Measure DIFFERENTIALLY between two mid-screen points. The clamped corner is
    # still the origin (so runs never accumulate error), but it is never itself
    # detected: at (0,0) the cursor sits over the dark shop sidebar where the
    # template matches poorly. Both measured points stay in the game world.
    # Move clear of the corner on BOTH axes before measuring. Stepping only the
    # axis under test leaves the other at 0, i.e. the cursor pinned to a screen
    # edge and outside the game-world ROI, where it cannot be seen at all.
    BASE_X, BASE_Y = 4, 3
    plan = ([("x", d) for d in (4, 6, 8, 10, 12)] +
            [("y", d) for d in (2, 3, 4, 5, 6)])
    samples = []
    print(f"\n{'axis':4s} {'d_rep':>6s} {'units':>6s} {'px':>6s} {'units/px':>9s} {'score':>6s}  note")
    print("-" * 58)
    for axis, d in plan:
        slam_corner(kb, args.chunk, reps)
        step(kb, args.chunk, BASE_X, "x")               # clear of BOTH edges
        step(kb, args.chunk, BASE_Y, "y")
        time.sleep(args.settle)
        x0, y0, s0 = finder.find(cap.grab_rgb01())
        step(kb, args.chunk, d, axis)
        time.sleep(args.settle)
        x1, y1, s1 = finder.find(cap.grab_rgb01())
        units = d * args.chunk
        px = (x1 - x0) if axis == "x" else (y1 - y0)
        note = ""
        ok = True
        if min(s0, s1) < finder.min_score:
            ok, note = False, "low confidence"
        elif px <= 0:
            # identical/backwards reading => the detector locked onto something
            # static, not the cursor. Never fold this into the fit.
            ok, note = False, "no travel detected (false positive?)"
        print(f"{axis:4s} {d:6d} {units:6d} {px:6d} {units/px if px > 0 else 0:9.4f} "
              f"{min(s0,s1):6.3f}  {note}")
        if ok:
            samples.append({"axis": axis, "units": units, "px": px,
                            "p0": [x0, y0], "p1": [x1, y1], "score": min(s0, s1)})

    print("\n--- fit (units to cross the full screen) ---")
    out = {}
    for axis, extent in (("x", w), ("y", h)):
        pts = [s for s in samples if s["axis"] == axis]
        if len(pts) < 2:
            print(f"{axis}: not enough good samples")
            continue
        # units-per-pixel through the origin (the corner IS the origin, so no intercept)
        upp = sum(p["units"] for p in pts) / sum(p["px"] for p in pts)
        span = upp * extent
        # linearity: if pointer acceleration is distorting things, per-sample
        # ratios drift with distance instead of staying constant.
        ratios = [p["units"] / p["px"] for p in pts]
        spread = (max(ratios) - min(ratios)) / (sum(ratios) / len(ratios)) * 100
        print(f"{axis}: units/px={upp:.4f}  SPAN={span:.1f}  "
              f"nonlinearity={spread:.1f}%  (n={len(pts)})")
        out[axis] = {"units_per_px": upp, "span": span, "nonlinearity_pct": spread,
                     "n": len(pts)}
    if "x" in out and "y" in out:
        print(f"\nMOUSE_SPAN = ({out['x']['span']:.1f}, {out['y']['span']:.1f})")
        print("Put that in scripts/keysender/hybrid_sender.py (or pass --mouse-span).")
        if max(out["x"]["nonlinearity_pct"], out["y"]["nonlinearity_pct"]) > 8:
            print("\nWARNING: >8% nonlinearity — pointer acceleration is likely ON in\n"
                  "Windows. A single span constant cannot be exact while it is. Turn off\n"
                  "'Enhance pointer precision' and re-run for a clean linear mapping.")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"samples": samples, "fit": out}, fh, indent=2)
        print(f"\nraw samples -> {args.out}")

    kb.running = False
    time.sleep(0.2)
    try:
        kb.sock.close()
    except OSError:
        pass
    cap.close()


if __name__ == "__main__":
    main()
