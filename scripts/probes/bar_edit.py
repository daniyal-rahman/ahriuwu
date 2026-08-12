#!/usr/bin/env python3
"""Programmatic minion-HP-bar edits — synthetic ground truth for the
"does minion HP survive the tokenizer" test, with NO labels required.

A real HP change is just the boundary between the coloured fill and the dark
depleted remainder moving. So the edit is: locate a bar, sample the dark colour
from that bar's OWN depleted section, and paint it over k columns of the fill
(shorten) or paint the fill colour over k columns of depleted (lengthen).
Pixel-identical to what the game renders at a different HP.

Localization uses a colour threshold. Note this needs only WHERE a bar is, not
what it reads — a miss degrades a sample to a control, it cannot bias the label.
"""
from __future__ import annotations

import cv2
import numpy as np

# Minion bar fill colours (HSV). Red = enemy, green = ally (blue = own team ally
# in some skins). Kept tight so we hit bars, not terrain.
_FILL_BANDS = [
    ((0, 120, 90), (10, 255, 255)),      # red low
    ((170, 120, 90), (180, 255, 255)),   # red high
    ((40, 100, 90), (85, 255, 255)),     # green
]
_MIN_W, _MAX_W, _MAX_H, _MIN_H = 5, 22, 4, 1     # minion bar: ~6-20 x 1-3 px
_DARK_V = 70                                      # depleted backing is near-black


def find_minion_bars(rgb: np.ndarray) -> list[dict]:
    """Candidate minion bars: {x,y,w,h,fill_bgr,dark_bgr}. Small, wide, solid."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in _FILL_BANDS:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    H, W = mask.shape
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (_MIN_W <= w <= _MAX_W and _MIN_H <= h <= _MAX_H):
            continue
        if w < 2.5 * h or area < 0.6 * w * h:          # wide + solid
            continue
        # depleted colour: the dark run immediately right of the fill
        x2 = x + w
        dark = None
        for dx in range(0, 12):
            if x2 + dx >= W:
                break
            col = rgb[y:y + h, x2 + dx]
            if col.size and col.max() <= _DARK_V:
                dark = col.reshape(-1, 3).mean(0)
                break
        if dark is None:
            continue                                    # full-HP bar: no depleted ref
        out.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "fill": rgb[y:y + h, x:x + w].reshape(-1, 3).mean(0),
                    "dark": dark})
    return out


def edit_bar(rgb: np.ndarray, bar: dict, k: int) -> np.ndarray:
    """k>0 shortens the fill by k px (lower HP); k<0 lengthens it (higher HP).
    Colours are sampled from the bar itself, so no seam/colour tell."""
    out = rgb.copy()
    x, y, w, h = bar["x"], bar["y"], bar["w"], bar["h"]
    if k > 0:
        k = min(k, w - 1)                               # never erase the whole bar
        out[y:y + h, x + w - k:x + w] = bar["dark"].astype(np.uint8)
    elif k < 0:
        k = -k
        x2 = min(x + w + k, out.shape[1])
        out[y:y + h, x + w:x2] = bar["fill"].astype(np.uint8)
    return out


def control_edit(rgb: np.ndarray, bar: dict, k: int, offset_y: int = 26) -> np.ndarray:
    """Same pixel count + same colours, painted on nearby TERRAIN instead of a
    bar. Calibrates 'the encoder noticed *an* edit' vs '*this* edit'."""
    out = rgb.copy()
    x, y, w, h = bar["x"], bar["y"], bar["w"], bar["h"]
    yy = min(max(y + offset_y, 0), out.shape[0] - h)
    kk = min(abs(k), w - 1)
    out[yy:yy + h, x + w - kk:x + w] = bar["dark"].astype(np.uint8)
    return out


def pick_bar(bars: list[dict], rgb_shape) -> dict | None:
    """Prefer a bar with room to edit, nearest the frame centre (where the
    champion and the contested wave are)."""
    if not bars:
        return None
    cy, cx = rgb_shape[0] / 2, rgb_shape[1] / 2
    ok = [b for b in bars if b["w"] >= 6]
    if not ok:
        return None
    return min(ok, key=lambda b: (b["x"] + b["w"] / 2 - cx) ** 2 + (b["y"] - cy) ** 2)
