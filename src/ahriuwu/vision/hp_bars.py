"""Classical-CV HP-bar reader for 352x352 League frames.

Perception side-channel: the v7 tokenizer provably does not carry HP-bar fill
(cross-game probe R2~0.16; minion bars vanish in recon), but the bars are
deterministic rendered UI — trivially separable by color+shape in pixel space.
This reads them directly: champion bars (~22-34px wide) and minion bars
(~6-20px), each with a fill fraction estimated from the colored run vs the
dark depleted remainder (LoL bars deplete right-to-left, fill anchored left).

Feeds: policy scalar side-channel, wave-state channels for the dynamics, and
validation against replay labels (own-champ hp_frac has exact ground truth).

No ML, no GPU. ~2ms/frame.
"""
from __future__ import annotations

from statistics import median

import cv2
import numpy as np

# HSV fill-color bands (OpenCV H in [0,180]). Bars are saturated red/green/blue.
_BANDS = {
    "red": [((0, 100, 80), (12, 255, 255)), ((168, 100, 80), (180, 255, 255))],
    "green": [((35, 90, 80), (85, 255, 255))],
    "blue": [((95, 90, 80), (130, 255, 255))],
}
_MIN_W, _MAX_H, _MIN_ASPECT = 3, 6, 1.5
_CHAMP_MIN_TOTAL_W = 22          # total bar width (fill+depleted) above this = champion
_DARK_V = 75                     # depleted-bar backing: dark pixels
_MAX_TOTAL_W = 60                # sanity cap on backing walk


def _dark_run_right(hsv: np.ndarray, y0: int, y1: int, x_from: int, max_px: int) -> int:
    """Length of the dark (depleted) run to the right of the colored fill."""
    H, W = hsv.shape[:2]
    v = hsv[max(y0, 0):min(y1, H), :, 2]
    n = 0
    for x in range(x_from, min(x_from + max_px, W)):
        col = v[:, x]
        if col.size and (col <= _DARK_V).mean() >= 0.6:
            n += 1
        else:
            break
    return n


def detect_bars(rgb: np.ndarray) -> list[dict]:
    """Detect HP bars in an RGB uint8 frame (any size; tuned at 352x352).

    Returns a list of dicts: {cx, cy, fill, total_w, kind ('champion'|'minion'),
    color ('red'|'green'|'blue')}. fill in [0,1] = colored / (colored + dark-right).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    out = []
    for color, bands in _BANDS.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in bands:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if w < _MIN_W or h > _MAX_H or w < _MIN_ASPECT * h or area < 3:
                continue
            if area < 0.55 * w * h:               # colored fill is a solid run
                continue
            dark = _dark_run_right(hsv, y, y + h, x + w, _MAX_TOTAL_W - w)
            total = w + dark
            fill = w / max(total, 1)
            kind = "champion" if total >= _CHAMP_MIN_TOTAL_W else "minion"
            out.append({"cx": float(cents[i][0]), "cy": float(cents[i][1]),
                        "fill": float(fill), "total_w": int(total),
                        "kind": kind, "color": color})
    return out


# Champion floating-bar geometry at 352x352, measured on held-out replays.
# Two camera zooms observed: bar interior (fill+depleted) is ~19px or ~30px
# wide; the fill block is 3-4 rows tall at the small zoom, 5-6 at the large.
# The colored fill of a FULL bar reads ~2px narrower than the interior (edge
# antialiasing falls out of the HSV bands). The fill's left edge sits at
# anchor_x - 13 .. +2; the bar floats ~38-70px above the champion_screen
# anchor depending on camera pitch and map position.
_FULL_W = (17, 28)               # colored width of a full bar (small/large zoom)
_EXP_LEFT = -9                   # expected fill-left minus anchor x
_LEFT_TOL = 15
_FILL_H = (3, 7)                 # fill block height bounds (rows)
_DEPLETED_V = 55                 # depleted interior / backing is near-black


def read_own_champ_hp(rgb: np.ndarray, champ_xy_352: tuple[float, float],
                      _debug: list | None = None) -> float | None:
    """Fill fraction of the champion's floating HP bar via a dedicated crop read.

    Structure-first read: the fill is a left-anchored solid colored run 3-6
    rows tall; the depleted remainder is near-black and abuts it on the right
    (after an antialiased blend / leading-edge glint of up to 2px). We group
    per-row colored runs into vertical fill blocks, then classify:
      A) bounded: colored + glint + dark run totals a known interior width
         -> exact fill = colored / total;
      B) (near-)full: colored alone matches a known full-bar width and no
         depleted run follows -> colored / full width;
      C) unbounded: the depleted run merges with dark terrain or shadow
         -> colored / full width for the zoom (by colored width, else by
         fill height).
    Champion-outline streaks (1-2 rows) and effect blobs (>7 rows) fail the
    height bounds; candidates missing the bar's dark strip 1-3 rows below
    the fill fail the strip gate; other champions' bars fail the anchor-
    distance gate. Returns None if nothing plausible remains (death,
    occlusion, anchor far off the bar).
    """
    H, W = rgb.shape[:2]
    cx, cy = int(round(champ_xy_352[0])), int(round(champ_xy_352[1]))
    x0, x1 = max(cx - 32, 0), min(cx + 42, W)
    y0, y1 = max(cy - 74, 0), min(max(cy - 28, 1), H)
    if x1 - x0 < 24 or y1 - y0 < 10:
        return None
    crop = rgb[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    colored = np.zeros(hsv.shape[:2], np.uint8)
    for bands in _BANDS.values():
        for lo, hi in bands:
            colored |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    c = colored > 0
    # drop horizontally isolated colored pixels (outline speckle)
    keep = np.zeros_like(c)
    keep[:, 1:] |= c[:, :-1]
    keep[:, :-1] |= c[:, 1:]
    c &= keep
    # bridge exactly-1px tick-mark gaps (2px gaps separate adjacent bars)
    closed = c.copy()
    closed[:, 1:-1] |= c[:, :-2] & c[:, 2:]
    v = hsv[..., 2]
    nrows, ncols = closed.shape

    # per-row maximal colored runs -> vertical groups (fill blocks); chain by
    # group median with a one-row gap allowance (sheen rows drop out of the
    # color bands occasionally)
    groups: list[dict] = []
    for r in range(nrows):
        x = 0
        row = closed[r]
        while x < ncols:
            if not row[x]:
                x += 1
                continue
            x2 = x
            while x2 < ncols and row[x2]:
                x2 += 1
            if x2 - x >= 2:
                for g in groups:
                    if r - g["last"] in (1, 2) \
                            and abs(x - median(g["s"])) <= 3 \
                            and abs((x2 - x) - median(g["l"])) <= 4:
                        g["s"].append(x)
                        g["l"].append(x2 - x)
                        g["last"] = r
                        break
                else:
                    groups.append({"s": [x], "l": [x2 - x], "first": r, "last": r})
            x = x2

    exp_s = (cx - x0) + _EXP_LEFT
    best = None                                        # (class, dist, fill)
    for g in groups:
        h = g["last"] - g["first"] + 1
        if not (_FILL_H[0] <= h <= _FILL_H[1]):
            continue
        # junk streaks above/below the fill can chain into the group; measure
        # on the most self-consistent 3-row window instead of the whole block
        n = len(g["s"])
        win = min(n, 3)
        bi = 0
        bscore = 10 ** 9
        for i in range(n - win + 1):
            ss, ll = g["s"][i:i + win], g["l"][i:i + win]
            score = (max(ss) - min(ss)) + (max(ll) - min(ll))
            if score <= bscore:                        # ties -> lower window
                bscore, bi = score, i
        s = int(median(g["s"][bi:bi + win]))
        l = int(median(g["l"][bi:bi + win]))
        r0 = g["first"] + bi
        r1 = min(g["first"] + bi + win, g["last"] + 1)  # window rows [r0, r1)
        if _debug is not None and l >= 3:
            _debug.append({"pre": True, "s": s, "l": l, "h": h,
                           "rows": (g["first"], g["last"]), "exp_s": exp_s})
        if l > 38 or abs(s - exp_s) > _LEFT_TOL:
            continue
        # depleted (dark) run to the right of the fill, over the window rows;
        # the fill/depleted boundary renders an antialiased blend plus a
        # bright leading-edge glint, so allow up to 2 non-dark pixels first —
        # but only in proportion to the fill (a glint cannot dominate a
        # near-empty bar; unbounded skips let 2px specks fake class-A reads)
        d = 0
        skip = 0
        max_skip = 0 if l < 4 else 1 if l < 10 else 2
        bounded = False
        xd = s + l
        while skip < max_skip and xd < ncols and (v[r0:r1, xd] <= _DEPLETED_V).mean() < 0.6:
            skip += 1
            xd += 1
        for x in range(xd, min(s + l + 40, ncols)):
            col = v[r0:r1, x]
            if (col <= _DEPLETED_V).mean() >= 0.6:
                d += 1
            else:
                bounded = True
                break
        if d == 0:
            skip = 0
            bounded = True
        # the bar always renders a near-black strip 1-3 rows below the fill;
        # colored streaks over terrain (champ outlines, effects) do not
        strip_x1 = min(s + max(l, 8), ncols)
        strip = False
        for rr in range(r1, min(g["last"] + 4, nrows)):
            seg = v[rr, s:strip_x1]
            if seg.size and (seg <= 60).mean() >= 0.55:
                strip = True
                break
        total = l + skip + d
        if _debug is not None:
            _debug.append({"s": s, "l": l, "d": d, "skip": skip, "h": h,
                           "rows": (g["first"], g["last"]), "win": (r0, r1),
                           "bounded": bounded, "strip": strip, "total": total})
        if not strip:
            continue
        cand = None
        if d >= 2 and bounded and (16 <= total <= 20 or 27 <= total <= 32):
            cand = (0, (l + skip / 2) / total)         # A: exact bounded read
        elif d <= 1 and 14 <= l <= 19:
            cand = (0, min(l / _FULL_W[0], 1.0))       # B: (near-)full, small
        elif d <= 1 and 25 <= l <= 38:
            cand = (0, min(l / _FULL_W[1], 1.0))       # B: (near-)full, large
        elif d <= 1 and 20 <= l <= 24:                 # zoom-ambiguous width:
            w = _FULL_W[1] if h >= 5 else _FULL_W[0]   # large partial w/ bright
            cand = (1, min(l / w, 1.0))                # depleted, or small full
        elif d >= 3:                                   # C: depleted run merges
            w = _FULL_W[1] if (l >= 25 or h >= 5) else _FULL_W[0]
            cand = (1, min(l / w, 1.0))                #    with dark bg/shadow
        if cand is None:
            continue
        key = (cand[0], abs(s - exp_s))
        if best is None or key < best[:2]:
            best = (*key, cand[1])
    if best is None:
        return None
    return float(min(best[2], 1.0))
