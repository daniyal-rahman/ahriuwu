#!/usr/bin/env python
"""Render the capture as transparent overlay frames, to composite onto video.

Input is the JSONL from ``lanerl/viz_capture.py``; output is a directory of
RGBA PNGs plus the exact ffmpeg command that lays them over the screen
recording.

Why PNGs and not a finished video
---------------------------------
The recording is made on Windows and this runs on Linux, so the two halves are
not in the same place at the same time. Emitting frames plus a command means
the render does not need the video, the composite does not need Python, and
re-rendering after a design change costs no game time.

What gets drawn
---------------
``target``  a ring on every entity slot the policy could act on, at that
            slot's SCREEN position, with radius and opacity set by the target
            head's probability for it. This is the interesting one: it is
            literally "which minion am I about to click", drawn on the minion.
            The argmax gets a filled marker.
``move``    an arrow from screen centre along the chosen (move_x, move_z)
            bins, converted back through the lane frame. The champion is at
            screen centre by construction -- that is what centred_on means.
``button``  a small bar chart of the button distribution, with the sampled one
            highlighted, so a confident policy and a dithering one look
            different at a glance.
``value``   the critic's estimate as a moving trace.
``order``   the literal wire order, as text, so what the agent DID is legible
            next to what it was considering.

Alignment
---------
Records carry the game clock in ms. ``--t0-ms`` is the game time of the first
VIDEO frame; everything else follows from ``--fps``. Get t0 from the in-game
clock visible in the recording -- guessing it puts the whole overlay out of
sync with the thing it is annotating.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("viz_overlay")

TEAM_COLOUR = {100: "#4da6ff", 200: "#ff5c5c"}


def load(path: Path) -> list:
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def pick(records: list, t_ms: float) -> dict | None:
    """The record whose game clock is nearest ``t_ms``.

    Nearest rather than last-before: decisions are 33 ms apart and video
    frames 16-33 ms, so 'last before' systematically lags by up to a full
    decision, which is visible as the overlay trailing the action.
    """
    if not records:
        return None
    i = int(np.searchsorted([r["t"] for r in records], t_ms))
    cands = [c for c in (i - 1, i) if 0 <= c < len(records)]
    if not cands:
        return None
    # Plain argmin. The `... and records[...]` one-liner this replaces was
    # wrong at index 0, which is falsy -- it would have returned 0 instead of
    # the first record, i.e. dropped the overlay on the opening frames only.
    best = min(cands, key=lambda c: abs(records[c]["t"] - t_ms))
    return records[best]


def render_frame(rec: dict, side: str, cfg, values: list, out: Path) -> None:
    w, h = cfg.width, cfg.height
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # screen y grows DOWNWARD, same as the projection
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    s = (rec.get("sides") or {}).get(side)
    if s is None:
        fig.savefig(out, transparent=True)
        plt.close(fig)
        return

    heads = s.get("heads") or {}
    tgt = heads.get("target") or []
    slots = s.get("slots") or []

    best = max(range(len(tgt)), key=lambda i: tgt[i]) if tgt else None
    for e in slots:
        sx, sy = e["screen"]
        if not (-0.1 <= sx <= 1.1 and -0.1 <= sy <= 1.1):
            continue  # off screen: it has a real coordinate, but nothing to draw on
        p = tgt[e["slot"]] if e["slot"] < len(tgt) else 0.0
        col = TEAM_COLOUR.get(e["team"], "#cccccc")
        ax.add_patch(plt.Circle((sx, sy), 0.006 + 0.05 * p, fill=False,
                                lw=1.0 + 3.0 * p, color=col, alpha=0.25 + 0.75 * p))
        if best is not None and e["slot"] == best and p > 0.05:
            ax.add_patch(plt.Circle((sx, sy), 0.012, color="#ffe066", alpha=0.9))
            ax.text(sx + 0.018, sy, f"{p:.0%}", color="#ffe066", fontsize=9,
                    va="center", fontweight="bold")

    # Move intent, from screen centre: the champion is centred by construction.
    mx, mz = heads.get("move_x"), heads.get("move_z")
    if mx and mz:
        n = len(mx)
        ix = int(np.argmax(mx)) - (n - 1) / 2.0
        iz = int(np.argmax(mz)) - (n - 1) / 2.0
        mag = float(np.hypot(ix, iz)) or 1.0
        ax.arrow(0.5, 0.5, 0.10 * ix / mag, -0.10 * iz / mag,
                 width=0.002, head_width=0.012, color="#7CFC00", alpha=0.85)

    # Button distribution.
    btn = heads.get("button") or []
    if btn:
        names = ["noop", "move", "atkmv", "recall", "q", "w", "e", "r"][: len(btn)]
        # Bars start at x=0.07, not 0.02: the labels are right-aligned to the
        # left of the bar, so at 0.02 every one of them ran off the frame and
        # rendered as "oop"/"ove"/"call".
        x0, y0, bh = 0.07, 0.06, 0.028
        for i, (nm, p) in enumerate(zip(names, btn)):
            y = y0 + i * bh
            ax.add_patch(plt.Rectangle((x0, y), 0.12 * p, bh * 0.7,
                                       color="#ffffff", alpha=0.35 + 0.55 * p))
            ax.text(x0 - 0.006, y + bh * 0.35, nm, color="#ffffff", fontsize=8,
                    ha="right", va="center", alpha=0.9)

    v = s.get("value")
    if v is not None:
        values.append(v)
        tail = values[-240:]
        xs = np.linspace(0.72, 0.96, len(tail))
        lo, hi = min(tail), max(tail)
        rng = (hi - lo) or 1.0
        # Band 0.09..0.15 rather than 0.04..0.10: screen y is inverted here, so
        # the smaller numbers are the TOP of the frame and the trace was
        # running off it whenever the value hit its running maximum.
        ys = [0.15 - 0.06 * ((y - lo) / rng) for y in tail]
        ax.plot(xs, ys, color="#ffe066", lw=1.4, alpha=0.9)
        ax.text(0.96, 0.185, f"V={v:+.2f}", color="#ffe066", fontsize=9, ha="right")

    order = s.get("order")
    if order:
        ax.text(0.5, 0.965, json.dumps(order), color="#ffffff", fontsize=10,
                ha="center", alpha=0.85, family="monospace")
    ax.text(0.02, 0.03, f"t={rec.get('t')}ms  {side}", color="#ffffff",
            fontsize=9, alpha=0.7, family="monospace")

    fig.savefig(out, transparent=True)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--out-dir", default="lanerl/logs/overlay_frames")
    ap.add_argument("--side", default="blue", choices=("blue", "red"))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--t0-ms", type=float, default=None,
                    help="game clock of the FIRST video frame. Defaults to the "
                         "first captured decision, which is only right if the "
                         "recording starts exactly there.")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    cfg = ap.parse_args()

    recs = load(Path(cfg.capture))
    if not recs:
        log.error("no records in %s", cfg.capture)
        return 2
    recs.sort(key=lambda r: r["t"])
    t0 = cfg.t0_ms if cfg.t0_ms is not None else recs[0]["t"]
    span_ms = (cfg.seconds * 1000.0) if cfg.seconds else (recs[-1]["t"] - t0)
    n_frames = max(1, int(span_ms / 1000.0 * cfg.fps))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values: list = []
    for i in range(n_frames):
        rec = pick(recs, t0 + i * 1000.0 / cfg.fps)
        if rec is None:
            continue
        render_frame(rec, cfg.side, cfg, values, out_dir / f"f{i:06d}.png")
        if i % 200 == 0:
            log.info("rendered %d/%d", i, n_frames)

    log.info("wrote %d frames to %s", n_frames, out_dir)
    print(
        "\ncomposite onto the recording with:\n"
        f"  ffmpeg -i RECORDING.mp4 -framerate {cfg.fps:g} -i {out_dir}/f%06d.png \\\n"
        "    -filter_complex '[0:v][1:v]overlay=0:0:format=auto' -c:a copy OUT.mp4\n"
        "\n(the overlay frames are already at the video's resolution; pass "
        "--width/--height if the recording is not 1280x720)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
