"""Audit summary figure: measured minion-bar widths + resolution requirement."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/srv/nfs/projects/ahriuwu/scratchpad/audit_aspect"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # validated slots 1-3
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"

rows = json.load(open(f"{OUT}/final_geom.json"))
W = np.array([r[0] for r in rows]); H = np.array([r[1] for r in rows])
minion = W[(H == 1) & (W <= 12)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0), facecolor=SURF)
for ax in (ax1, ax2):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(axis="y", color="#e5e4e0", lw=0.8)
    ax.set_axisbelow(True)

# ── Panel 1: measured minion HP-bar fill widths in the shipped 352 frames ──
bins = np.arange(2.5, 13.5, 1)
cnt, _ = np.histogram(minion, bins=bins)
xs = np.arange(3, 13)
ax1.bar(xs, cnt, width=0.78, color=S1, zorder=3)
ax1.bar([9], [cnt[9 - 3]], width=0.78, color=S2, zorder=4)
ax1.annotate(f"full bar = 9 px\n({cnt[9-3]} obs)", xy=(9, cnt[9 - 3]), xytext=(10.4, cnt[9 - 3] * 0.92),
             color=INK, fontsize=9.5, fontweight="bold",
             arrowprops=dict(arrowstyle="-", color=S2, lw=1.6))
ax1.set_xlabel("minion HP-bar fill width (px, in the shipped 352x352 frame)", color=INK2, fontsize=10)
ax1.set_ylabel("connected components", color=INK2, fontsize=10)
ax1.set_title("Only 9 distinguishable HP levels survive\n"
              "measured, 12 matches x 41 mid/late frames", color=INK, fontsize=12,
              fontweight="bold", loc="left", pad=12)
ax1.set_xticks(xs)

# ── Panel 2: bar width vs canvas size ──
N = np.arange(224, 1300, 4)
NATIVE = 32.7
sq = NATIVE * N / 1280.0                       # squish NxN  (== letterbox NxN)
ap = NATIVE * (N * 4 / 3) / 1280.0             # aspect-preserved at equal px budget
ax2.plot(N, sq, color=S1, lw=2.0, zorder=3)
ax2.plot(N, sq, color=S2, lw=2.0, ls=(0, (5, 5)), zorder=4)
ax2.plot(N, ap, color=S3, lw=2.0, zorder=3)
for y, lab, c, dy in ((8, "8 px", MUTED, 3), (16, "16 px", MUTED, 3), (32, "32 px", MUTED, 3)):
    ax2.axhline(y, color="#cfcec9", lw=1.0, ls=":", zorder=1)
    ax2.text(1290, y + dy, lab, color=MUTED, fontsize=8.5, ha="right")
ax2.scatter([352], [NATIVE * 352 / 1280], s=60, color=S1, zorder=6,
            edgecolor=SURF, linewidth=2)
ax2.annotate("CURRENT\n352 squish -> 9.0 px", xy=(352, 9.0), xytext=(400, 15.5),
             color=INK, fontsize=9.5, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
ax2.text(1180, NATIVE * 1180 / 1280 + 1.2, "squish NxN", color=S1, fontsize=10,
         fontweight="bold", ha="right")
ax2.text(1180, NATIVE * 1180 / 1280 - 3.4, "letterbox NxN (identical)", color=S2,
         fontsize=10, fontweight="bold", ha="right")
ax2.text(940, NATIVE * (940 * 4 / 3) / 1280 + 1.4, "aspect-preserved,\nsame pixel budget",
         color=S3, fontsize=10, fontweight="bold", ha="right")
ax2.set_xlabel("square canvas size N (NxN input to the tokenizer)", color=INK2, fontsize=10)
ax2.set_ylabel("minion HP-bar width (px)", color=INK2, fontsize=10)
ax2.set_title("Letterboxing buys ZERO horizontal detail\n"
              "only total resolution moves the bar width", color=INK, fontsize=12,
              fontweight="bold", loc="left", pad=12)
ax2.set_xlim(224, 1300); ax2.set_ylim(0, 38)

fig.tight_layout(pad=1.6)
fig.savefig(f"{OUT}/aspect_audit_summary.png", dpi=150, facecolor=SURF)
print("wrote", f"{OUT}/aspect_audit_summary.png")
