"""Render a recorded episode (``record.py`` .npz) to an mp4: top-down top lane, real time at the decision rate.

    python -m lanerl_jax.modern.bc.render episode.npz [--out episode.mp4] [--speed 1] [--size 900] [--fog blue|red|off]

Pure numpy + Pillow + ffmpeg (no JAX): run it on the login node or a cpu job. Blue is drawn in blue, red in
red; champions are the big discs with their initial, minions small discs, turrets squares, the Nexus a
diamond; health bars above living units. Each champion's order this decision: a line to its move point, a
ring on its attack or cast target, the cast letter at the cast point. ``--fog blue`` hides what blue's team
cannot see (the clone's own view); default shows everything and hollows out the units blue cannot see.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:                                              # pragma: no cover
    sys.exit("render.py needs Pillow (python3 -m pip install pillow)")

KIND_NONE, KIND_CHAMPION, KIND_MINION, KIND_TURRET, KIND_INHIBITOR, KIND_NEXUS, KIND_MONSTER, KIND_WARD = range(8)
TEAM_RGB = ((70, 130, 255), (235, 80, 70))
BG, WALL, LANE = (52, 66, 56), (16, 20, 18), (84, 100, 86)
PANEL_W = 300


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class View:
    """World (x, y) -> image pixels for a square window around the top lane (y up in the world, down on screen)."""

    def __init__(self, lane_path, size, margin=900.0):
        lo, hi = lane_path.min(0) - margin, lane_path.max(0) + margin
        side = float(max(hi - lo))
        centre = (lo + hi) / 2
        self.lo = centre - side / 2
        self.scale = size / side
        self.size = size

    def __call__(self, x, y):
        return (float(x) - self.lo[0]) * self.scale, self.size - (float(y) - self.lo[1]) * self.scale

    def r(self, units, floor=1.5):
        return max(float(units) * self.scale, floor)


def background(d, view, size):
    """The walkable mask sampled per pixel, and the lane path."""
    walk, (mx, mz), cell = d["walkable"].astype(bool), d["terrain_min"], float(d["terrain_cell"])
    h, w = walk.shape
    xs = view.lo[0] + (np.arange(size) + 0.5) / view.scale                 # world x of each pixel column
    ys = view.lo[1] + (size - (np.arange(size) + 0.5)) / view.scale        # world y of each pixel row
    cx = np.clip(((xs - mx) / cell).astype(int), 0, w - 1)
    cy = np.clip(((ys - mz) / cell).astype(int), 0, h - 1)
    inside = ((xs >= mx) & (xs < mx + w * cell))[None, :] & ((ys >= mz) & (ys < mz + h * cell))[:, None]
    open_ = walk[cy[:, None], cx[None, :]] & inside
    arr = np.where(open_[..., None], np.asarray(BG, np.uint8), np.asarray(WALL, np.uint8)).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.line([view(x, y) for x, y in d["lane_path"]], fill=LANE, width=max(int(view.r(300)), 2))
    return img


def hp_bar(draw, x, y, w, frac, colour):
    draw.rectangle([x - w / 2, y - 3, x + w / 2, y], fill=(20, 20, 20))
    draw.rectangle([x - w / 2, y - 3, x - w / 2 + w * max(min(frac, 1.0), 0.0), y], fill=colour)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("episode", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    p.add_argument("--size", type=int, default=900, help="map window pixels (square)")
    p.add_argument("--fog", choices=("blue", "red", "off"), default="off")
    p.add_argument("--start-s", type=float, default=None)
    p.add_argument("--end-s", type=float, default=None)
    p.add_argument("--crf", type=int, default=23)
    a = p.parse_args(argv)
    d = np.load(a.episode, allow_pickle=False)
    buttons = [str(b) for b in d["buttons"]]
    names = [str(n) for n in d["champions"]]
    T = d["t"].shape[0]
    per_step = "u_kind" in d.files                                # kind / team / tier per decision (minions spawn)
    fps = float(d["decision_hz"]) * a.speed
    view = View(np.concatenate([d["lane_path"], d["fountain"]]), a.size)
    base = background(d, view, a.size)
    font, small, big = _font(15), _font(12), _font(22)
    W, H = a.size + PANEL_W, a.size
    out = a.out or a.episode.with_suffix(".mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", f"{fps:.3f}", "-i", "-", "-an", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", str(a.crf), "-movflags", "+faststart", str(out)],
                          stdin=subprocess.PIPE)
    frames = 0
    t0 = float(d["t"][0]) if a.start_s is None else a.start_s
    t1 = float(d["t"][-1]) if a.end_s is None else a.end_s
    letters = "QWER"
    for i in range(T):
        t = float(d["t"][i])
        if t < t0 or t > t1:
            continue
        img = base.copy()
        draw = ImageDraw.Draw(img)
        x, y, hp, mhp, alive = d["u_x"][i], d["u_y"][i], d["u_hp"][i], d["u_max_hp"][i], d["u_alive"][i]
        kind, team, sub = ((d["u_kind"][i], d["u_team"][i], d["u_sub"][i]) if per_step
                           else (d["unit_kind"], d["unit_team"], d["unit_sub"]))
        vis = d["u_visible"][i]                                   # (2, N)
        seen = vis[{"blue": 0, "red": 1}[a.fog]] if a.fog != "off" else np.ones_like(alive, bool)
        order = range(len(kind))
        for j in sorted(order, key=lambda j: (kind[j] == KIND_CHAMPION, kind[j] == KIND_MINION)):
            k = kind[j]
            if k == KIND_NONE or not alive[j] or not seen[j]:
                continue
            cx, cy = view(x[j], y[j])
            col = TEAM_RGB[int(team[j])] if team[j] in (0, 1) else (200, 200, 120)
            hollow = a.fog == "off" and not vis[0][j]
            fill, outline = (None, col) if hollow else (col, (0, 0, 0))
            if k == KIND_CHAMPION:
                r = view.r(max(float(d["radius"][j]), 65.0), 9.0)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=2)
                draw.text((cx, cy), names[j][0], fill=(255, 255, 255), font=font, anchor="mm")
                hp_bar(draw, cx, cy - r - 4, 2.4 * r + 16, hp[j] / max(mhp[j], 1.0), col)
            elif k == KIND_MINION:
                r = view.r(45.0 if sub[j] < 2 else 60.0, 3.5)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline)
                hp_bar(draw, cx, cy - r - 2, 2 * r + 4, hp[j] / max(mhp[j], 1.0), col)
            elif k == KIND_TURRET:
                r = view.r(90.0, 6.0)
                draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill, outline=outline, width=2)
                hp_bar(draw, cx, cy - r - 4, 2 * r + 10, hp[j] / max(mhp[j], 1.0), col)
            elif k in (KIND_INHIBITOR, KIND_NEXUS):
                r = view.r(120.0, 7.0)
                draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=fill, outline=outline)
                hp_bar(draw, cx, cy - r - 4, 2 * r + 10, hp[j] / max(mhp[j], 1.0), col)
            elif k == KIND_WARD:
                r = view.r(60.0)
                draw.polygon([(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)], fill=fill, outline=outline)
        # orders this decision
        for c in (0, 1):
            if not alive[c]:
                continue
            col = TEAM_RGB[c]
            cx, cy = view(x[c], y[c])
            if d["o_move"][i][c]:
                mx, my = view(d["o_move_x"][i][c], d["o_move_y"][i][c])
                draw.line([cx, cy, mx, my], fill=col, width=1)
                draw.line([mx - 5, my - 5, mx + 5, my + 5], fill=col, width=2)
                draw.line([mx - 5, my + 5, mx + 5, my - 5], fill=col, width=2)
            tgt = int(d["o_attack"][i][c])
            if tgt >= 0:
                tx, ty = view(x[tgt], y[tgt])
                r = view.r(110.0, 8.0)
                draw.ellipse([tx - r, ty - r, tx + r, ty + r], outline=(255, 230, 60), width=2)
            slot = int(d["o_cast_slot"][i][c])
            if slot >= 0:
                ct = int(d["o_cast_target"][i][c])
                if ct >= 0:                                        # unit cast: ring the target
                    tx, ty = view(x[ct], y[ct])
                    r = view.r(110.0, 8.0)
                    draw.ellipse([tx - r, ty - r, tx + r, ty + r], outline=(255, 255, 120), width=2)
                draw.text((cx, cy - 22), letters[slot], fill=(255, 255, 120), font=big, anchor="mm")
                r = view.r(140.0, 12.0)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 120), width=3)
            ss = int(d["o_summoner_slot"][i][c])
            if ss >= 0:
                draw.text((cx, cy + 24), "D" if ss == 0 else "F", fill=(255, 160, 255), font=big, anchor="mm")
        # side panel
        canvas = Image.new("RGB", (W, H), (16, 16, 20))
        canvas.paste(img, (0, 0))
        img, draw = canvas, ImageDraw.Draw(canvas)
        m, s = divmod(int(t), 60)
        draw.text((a.size + 12, 10), f"{m:02d}:{s:02d}   x{a.speed:g}", fill=(230, 230, 230), font=big)
        yy = 50
        for c in (0, 1):
            col = TEAM_RGB[c]
            b = buttons[int(d["action"][i][c][0])]
            cd = d["c_cooldowns"][i][c]
            rk = d["c_ranks"][i][c]
            draw.text((a.size + 12, yy), f"{names[c]}  lvl {int(d['c_level'][i][c])}", fill=col, font=font)
            draw.text((a.size + 12, yy + 20), f"HP {int(hp[c])}/{int(mhp[c])}  CS {int(d['c_cs'][i][c])}  "
                      f"gold {int(d['c_gold'][i][c])}", fill=(220, 220, 220), font=small)
            draw.text((a.size + 12, yy + 38), f"{'dead' if d['c_dead'][i][c] else 'action'}: {b}",
                      fill=(255, 255, 120) if b not in ("noop", "move") else (200, 200, 200), font=small)
            for q in range(4):
                xq = a.size + 12 + q * 68
                ready = cd[q] <= 0.0 and rk[q] > 0
                draw.rectangle([xq, yy + 58, xq + 60, yy + 80], fill=(40, 90, 40) if ready else (60, 60, 60),
                               outline=(90, 90, 90))
                label = letters[q] + (f" {cd[q]:.0f}s" if cd[q] > 0 else "" if rk[q] > 0 else " -")
                draw.text((xq + 30, yy + 69), label, fill=(240, 240, 240), font=small, anchor="mm")
            yy += 100
        draw.text((a.size + 12, yy + 10), f"reward blue {float(d['reward'][:i + 1, 0].sum()):+.2f}",
                  fill=(180, 180, 180), font=small)
        draw.text((a.size + 12, H - 40), "yellow ring: attack/cast target\nx: move point", fill=(140, 140, 140),
                  font=small)
        raw = img.tobytes()
        if len(raw) != W * H * 3:
            raise RuntimeError(f"frame {i}: {img.mode} {img.size} gives {len(raw)} bytes, expected {W * H * 3}")
        ff.stdin.write(raw)
        frames += 1
    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit("ffmpeg failed")
    print(f"{frames} frames at {fps:.1f} fps -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
