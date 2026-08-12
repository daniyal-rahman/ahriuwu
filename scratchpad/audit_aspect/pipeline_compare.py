"""Squish vs letterbox vs bigger-square: sampling arithmetic + two empirical tests.

Test A (bounded): real 352 squish frame -> un-squish to 1280x720 -> re-run each
                  pipeline. Bounded by already-lost detail; measures resampling only.
Test B (exact):   synthesise a 1280x720 chart with minion HP bars at the measured
                  NATIVE geometry, run each pipeline, count recoverable HP levels.
"""
import json, math
import numpy as np, cv2

OUT = "/srv/nfs/projects/ahriuwu/scratchpad/audit_aspect"
SRC_W, SRC_H = 1280, 720
FRAME = "/srv/nfs/datasets/lol_replays_16_9_772/NA1_5549995114/frames/005000.png"

MINION_W_352 = 9        # measured, hand-verified
MINION_H_352 = 1
CHAMP_W_352 = 16
SX_SQUISH = 352 / SRC_W  # 0.275
NATIVE_MINION_W = MINION_W_352 / SX_SQUISH
NATIVE_MINION_H = MINION_H_352 / (352 / SRC_H)
NATIVE_CHAMP_W = CHAMP_W_352 / SX_SQUISH

print("=" * 72)
print("SAMPLING ARITHMETIC")
print("=" * 72)
print(f"source                     : {SRC_W}x{SRC_H} (16:9)")
print(f"measured minion bar in 352 : {MINION_W_352} px wide x {MINION_H_352} px tall")
print(f"=> implied NATIVE minion bar: {NATIVE_MINION_W:.1f} x {NATIVE_MINION_H:.1f} px @1280x720")
print(f"=> implied NATIVE champ bar : {NATIVE_CHAMP_W:.1f} px @1280x720")
print()

rows = []
def variant(name, cw, ch, canvas_px):
    sx, sy = cw / SRC_W, ch / SRC_H
    rows.append(dict(name=name, cw=cw, ch=ch, sx=sx, sy=sy, canvas=canvas_px,
                     content=cw * ch,
                     bar_w=NATIVE_MINION_W * sx, bar_h=NATIVE_MINION_H * sy))

variant("squish 352x352 (CURRENT)", 352, 352, 352 * 352)
variant("letterbox in 352x352 canvas", 352, 198, 352 * 352)
variant("aspect-preserved @ equal px budget (469x264)", 469, 264, 469 * 264)
variant("squish 448x448", 448, 448, 448 * 448)
variant("letterbox in 448x448 canvas", 448, 252, 448 * 448)
variant("squish 512x512", 512, 512, 512 * 512)
variant("letterbox in 512x512 canvas", 512, 288, 512 * 512)
variant("native 1280x720", 1280, 720, 1280 * 720)

hdr = f"{'variant':<46}{'sx':>7}{'sy':>7}{'content px':>12}{'%canvas':>9}{'bar w':>8}{'bar h':>7}{'HP steps':>10}{'%HP/px':>8}"
print(hdr); print("-" * len(hdr))
for r in rows:
    steps = r["bar_w"]
    print(f"{r['name']:<46}{r['sx']:>7.3f}{r['sy']:>7.3f}{r['content']:>12,}"
          f"{100*r['content']/r['canvas']:>8.0f}%{r['bar_w']:>8.1f}{r['bar_h']:>7.2f}"
          f"{steps:>10.1f}{100/max(steps,1e-9):>7.1f}%")

print()
print("KEY: squish-N and letterbox-N have IDENTICAL horizontal scale (N/1280).")
print("     Letterbox only LOWERS vertical scale and wastes 43.75% of the canvas.")
print()

# ── Test A: real frame reconstruction ───────────────────────────────────────
print("=" * 72)
print("TEST A - real frame (bounded by already-lost detail)")
print("=" * 72)
sq = cv2.imread(FRAME)
recon = cv2.resize(sq, (SRC_W, SRC_H), interpolation=cv2.INTER_CUBIC)   # un-squish

def pipe_squish(img, n):
    return cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)

def pipe_letterbox(img, n):
    ch = int(round(n * SRC_H / SRC_W))
    small = cv2.resize(img, (n, ch), interpolation=cv2.INTER_AREA)
    out = np.zeros((n, n, 3), np.uint8)
    top = (n - ch) // 2
    out[top:top + ch] = small
    return out, (top, top + ch)

def hgrad_energy(img, region=None):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if region:
        g = g[region[0]:region[1]]
    gx = np.abs(np.diff(g, axis=1))
    return float(gx.mean()), float((gx ** 2).mean())

a = pipe_squish(recon, 352)
b, reg = pipe_letterbox(recon, 352)
ma, ea = hgrad_energy(a)
mb, eb = hgrad_energy(b, reg)          # content region only - fair comparison
print(f"squish 352      : mean|dI/dx|={ma:.3f}  energy={ea:.1f}   over {a.shape[0]}x{a.shape[1]}")
print(f"letterbox 352   : mean|dI/dx|={mb:.3f}  energy={eb:.1f}   over content {reg[1]-reg[0]}x352")
print(f"-> horizontal detail per row is ~identical ({ma:.3f} vs {mb:.3f}); letterbox")
print(f"   simply keeps {reg[1]-reg[0]} content rows instead of 352 (-{100*(1-(reg[1]-reg[0])/352):.0f}% rows).")

# vertical gradient (the axis that actually differs)
def vgrad_energy(img, region=None):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if region:
        g = g[region[0]:region[1]]
    return float(np.abs(np.diff(g, axis=0)).mean())
print(f"vertical  mean|dI/dy| : squish={vgrad_energy(a):.3f}  letterbox={vgrad_energy(b,reg):.3f}")

# ── Test B: synthetic native bars ──────────────────────────────────────────
print()
print("=" * 72)
print("TEST B - synthetic 1280x720 chart with NATIVE-geometry minion bars")
print("=" * 72)
BW, BH = int(round(NATIVE_MINION_W)), max(1, int(round(NATIVE_MINION_H)))
fracs = np.arange(1, 11) / 10.0
chart = np.full((SRC_H, SRC_W, 3), (40, 70, 40), np.uint8)   # grass-ish
positions = []
for i, f in enumerate(fracs):
    cx = 90 + (i % 5) * 240
    cy = 150 + (i // 5) * 260
    cv2.rectangle(chart, (cx - 1, cy - 1), (cx + BW, cy + BH), (10, 10, 10), -1)
    fw = max(1, int(round(BW * f)))
    cv2.rectangle(chart, (cx, cy), (cx + fw - 1, cy + BH - 1), (230, 130, 40), -1)
    positions.append((cx, cy, fw, f))
cv2.imwrite(f"{OUT}/synth_native_1280x720.png", chart)

def measure_bar_widths(img, sx, sy, xoff=0, yoff=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    mk = ((H >= 95) & (H <= 125) & (S >= 120) & (V >= 100)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(mk, 8)
    got = []
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        got.append((x, y, w, h))
    return got

print(f"native bar: {BW} x {BH} px; 10 fill levels 10%..100%")
results = {}
for name, n in (("squish", 352), ("letterbox", 352), ("squish", 448), ("letterbox", 448),
                ("squish", 512), ("letterbox", 512), ("native", 1280)):
    if name == "squish":
        img = pipe_squish(chart, n); sx, sy = n / SRC_W, n / SRC_H
    elif name == "letterbox":
        img, _ = pipe_letterbox(chart, n); sx = sy = n / SRC_W
    else:
        img = chart.copy(); sx = sy = 1.0
    got = measure_bar_widths(img, sx, sy)
    got = [g for g in got if g[2] >= 1]
    ws = sorted(g[2] for g in got)
    hs = sorted(g[3] for g in got)
    nd = len(got)
    distinct = len(set(ws))
    results[f"{name}-{n}"] = dict(n_detected=nd, widths=ws, heights=hs, distinct_widths=distinct)
    print(f"  {name}-{n:<5}: detected {nd}/10 bars; widths={ws}; heights={sorted(set(hs))}; "
          f"distinct widths={distinct}/10")

json.dump(dict(arith=rows, synth=results), open(f"{OUT}/pipeline_compare.json", "w"), default=float)

# ── Part 4: required resolution ────────────────────────────────────────────
print()
print("=" * 72)
print("PART 4 - square-resize resolution needed for a minion bar of N px")
print("=" * 72)
print(f"{'target bar px':>14}{'squish NxN':>14}{'letterbox NxN':>16}{'aspect-pres. WxH':>20}")
for tgt in (8, 16, 32):
    n_sq = math.ceil(tgt * SRC_W / NATIVE_MINION_W)
    n_lb = n_sq                                    # identical horizontal scale
    # aspect preserved, non-square: W = tgt*1280/native_w, H = W*9/16
    Wp = math.ceil(tgt * SRC_W / NATIVE_MINION_W)
    Hp = math.ceil(Wp * 9 / 16)
    print(f"{tgt:>14}{n_sq:>14}{n_lb:>16}{f'{Wp}x{Hp}':>20}")
print()
print("current tokenizer img_size = 352, patch_size = 16  -> 22x22 = 484 patches")
print(f"a {MINION_W_352}x{MINION_H_352}px minion bar occupies "
      f"{MINION_W_352*MINION_H_352/(16*16)*100:.1f}% of ONE 16x16 patch")
