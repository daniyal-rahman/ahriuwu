#!/usr/bin/env python3
"""Generate README figures into docs/assets/ (2026-08-01 state).

Palette: validated 3-slot categorical (blue #2a78d6, orange #eb6834, aqua #1baf7a)
on the light surface #fcfcfb; ink/chrome per the reference palette.
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OUT = "docs/assets"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "figure.titlesize": 13,
    "legend.frameon": False, "font.family": "DejaVu Sans",
})

PAT = re.compile(r"Epoch (\d+) \[(\d+)/(\d+)\] loss=([\d.]+) bc=([\d.]+) "
                 r"\(abil=([\d.]+) move=([\d.]+)\) rew=([\d.]+)")


def parse(paths, batch):
    rows = []
    for p in paths:
        for line in open(p, errors="ignore"):
            m = PAT.search(line)
            if m:
                rows.append([float(m.group(6)), float(m.group(7)), float(m.group(8))])
    a = np.array(rows)                      # (N, [abil, move, rew]) in run order
    x = np.arange(len(a)) * 50 * batch / 1e6  # samples seen (M), 50 batches/log line
    return x, a


def smooth(y, w=51):
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same") / np.convolve(np.ones_like(y), k, mode="same") * 1.0


def fig_loss_curves():
    runs = [
        ("Phase-2 BC — old no-action backbone (GTX 1060, batch 6)",
         parse(["scratchpad/bc_logs_archive/bc_1060_prev_ep0-3.log",
                "scratchpad/bc_logs_archive/bc_1060_0731_pause.log",
                "scratchpad/bc_1060.log"], 6)),
        ("Phase-2 BC — act8775 action-conditioned backbone (RTX 5080, batch 16)",
         parse(["scratchpad/bc_5080_act8775.log"], 16)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    series = [("movement NLL", 1, BLUE), ("ability BCE", 0, ORANGE), ("reward twohot", 2, AQUA)]
    for ax, (title, (x, a)) in zip(axes, runs):
        for name, j, c in series:
            ax.plot(x, a[:, j], color=c, lw=0.6, alpha=0.15)
            ax.plot(x, smooth(a[:, j]), color=c, lw=1.8, label=name)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("training samples seen (M, approx — restarts re-cover some)")
    axes[0].set_ylabel("loss component (log scale)")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("BC training loss components — old backbone vs act8775 (as of 2026-08-01)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/bc_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for t, (x, a) in runs:
        print(f"  parsed {len(x)} log lines for: {t.split('(')[0].strip()} (last move={a[-1,1]:.2f})")


def fig_eval_binacc():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = ["ep1", "ep2", "ep3", "ep4 (mid)", "walk-to-lane\nwindow", "mid-laning\nwindow"]
    vals = [59.4, 63.4, 64.5, 68.5, 11.2, 7.5]
    xs = [0, 1, 2, 3, 4.6, 5.6]
    cols = [BLUE] * 4 + [ORANGE] * 2
    bars = ax.bar(xs, vals, width=0.8, color=cols, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center",
                fontsize=9, color=INK2)
    ax.plot([-0.5, 3.5], [59, 59], ls="--", lw=1.2, color=MUTED, zorder=4)
    ax.annotate("predict-center baseline ≈59%\n(human stands still 59% of the\nin-training window)",
                xy=(3.5, 59), xytext=(5.1, 38), ha="center", fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xticks(xs, labels, fontsize=9)
    ax.set_ylabel("movement bin accuracy (%)")
    ax.set_ylim(0, 78)
    ax.set_title("BC policy (old backbone): in-training game (blue) vs held-out game (orange)\n"
                 "held-out windows are ~100% moving — no stand-still credit  ·  2026-07-31")
    fig.tight_layout()
    fig.savefig(f"{OUT}/bc_eval_binacc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_probe_r2():
    targets = ["Garen HP\nfraction", "level", "champ\nscreen x", "champ\nscreen y"]
    mlp = [0.16, -0.65, 0.47, 0.27]
    lin = [0.11, -1.29, 0.27, 0.17]
    CLIP = -0.72
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(4)
    b1 = ax.bar(x - 0.19, mlp, width=0.36, color=BLUE, label="MLP probe", zorder=3)
    b2 = ax.bar(x + 0.19, np.clip(lin, CLIP, None), width=0.36, color=ORANGE,
                label="linear probe", zorder=3)
    for xi, v in zip(x - 0.19, mlp):
        ax.text(xi, v + 0.025 if v >= 0 else v - 0.07, f"{v:+.2f}", ha="center", fontsize=9, color=INK2)
    for xi, v in zip(x + 0.19, lin):
        shown = max(v, CLIP)
        txt = f"{v:+.2f}" + (" (clipped)" if v < CLIP else "")
        ax.text(xi, shown + 0.025 if v >= 0 else shown - 0.07, txt, ha="center", fontsize=8.5, color=INK2)
    ax.axhline(0, color=BASE, lw=1.2, zorder=2)
    ax.set_xticks(x, targets, fontsize=9)
    ax.set_ylabel("held-out-game R²")
    ax.set_ylim(CLIP - 0.18, 0.62)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Can probes read game state from v7 latents? (cross-game, 8 games, 3 held-out folds)\n"
                 "R²≤0 = no better than predicting the mean  ·  2026-07-31")
    fig.tight_layout()
    fig.savefig(f"{OUT}/probe_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_reward_auc():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), width_ratios=[1.15, 1])
    # left: event AUC dots with reference lines
    for y, lab in [(0.5, "0.50 blind"), (0.75, "0.75 usable"), (0.9, "0.90 strong")]:
        ax1.axhline(y, ls="--", lw=1.0, color=MUTED)
        ax1.text(1.62, y + 0.006, lab, fontsize=8, color=MUTED)
    ax1.scatter([0, 1], [0.902, 0.956], s=110, color=BLUE, zorder=4)
    for xi, v in [(0, 0.902), (1, 0.956)]:
        ax1.text(xi, v + 0.016, f"{v:.3f}", ha="center", fontsize=10, color=INK)
    ax1.set_xlim(-0.6, 2.3)
    ax1.set_ylim(0.45, 1.02)
    ax1.set_xticks([0, 1], ["in-training\ngames (2)", "HELD-OUT\ngame"], fontsize=9)
    ax1.set_ylabel("event AUC")
    ax1.set_title("Reward head ranks income events\n(Δgold ≥ 10) above no-income frames", fontsize=10)
    # right: mean predicted reward, event vs no-event
    x = np.arange(2)
    ev = [0.00194, 0.00267]
    nv = [0.00020, 0.00025]
    ax2.bar(x - 0.19, ev, width=0.36, color=BLUE, label="income-event frames", zorder=3)
    ax2.bar(x + 0.19, nv, width=0.36, color=ORANGE, label="no-income frames", zorder=3)
    for xi, v in list(zip(x - 0.19, ev)) + list(zip(x + 0.19, nv)):
        ax2.text(xi, v + 0.00006, f"{v:.4f}", ha="center", fontsize=8, color=INK2)
    ax2.set_xticks(x, ["in-training", "held-out"], fontsize=9)
    ax2.set_ylabel("mean predicted reward")
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.set_title("~10× separation in mean prediction\n(magnitude under-calibrated, rank is what PMPO needs)",
                  fontsize=10)
    fig.suptitle("Phase-3 go/no-go: solo-gold reward IS readable from latents  ·  2026-08-01", y=1.04)
    fig.tight_layout()
    fig.savefig(f"{OUT}/reward_head_auc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def copy_images():
    import cv2
    m = cv2.imread("scratchpad/hp_recon_stills/hp_recon_montage.png")
    h, w = m.shape[:2]
    if w > 1400:
        m = cv2.resize(m, (1400, int(h * 1400 / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{OUT}/hp_recon_montage.png", m)
    d = cv2.imread("scratchpad/dreamq_stills/ro_h008.png")
    cv2.imwrite(f"{OUT}/dream_vs_gt_h8.png", d)
    print(f"  montage -> {m.shape}, dream still -> {d.shape}")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_loss_curves()
    fig_eval_binacc()
    fig_probe_r2()
    fig_reward_auc()
    copy_images()
    print("done ->", OUT)
