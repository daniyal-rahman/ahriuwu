#!/usr/bin/env python
"""Figures 7-12: reward, MTP, Phase 3, data/labels, scale, pipeline."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *          # noqa
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

setup()
R = "/srv/nfs/projects/ahriuwu"
FD = f"{R}/reports/figdata"
J = lambda p: json.load(open(p))
corpus = J(f"{FD}/corpus_stats.json")


def vlabel(ax, bars, fmt="{:.3f}", dy=0.0, fs=6.9):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fs)


# =====================================================================
# F7  What the reward actually is
# =====================================================================
def fig_reward():
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.85))

    ax = axes[0]
    dec = [d for d in corpus["reward_decomp"] if d["band"] not in ("zero", "deaths (negative)")]
    labs = ["passive tick\n<2.1 g", "2.1-10 g", "melee minion\n10-25 g", "caster\n25-45 g",
            "cannon/plate\n45-100 g", "kill/turret\n>100 g"]
    x = np.arange(len(dec)); w = 0.38
    fr = [d["frac_frames"] * 100 for d in dec]
    sh = [d["share_pos_reward"] * 100 for d in dec]
    ax.bar(x - w / 2, fr, w, label="% of all frames", color=MARGINAL, edgecolor=INK, linewidth=0.6)
    ax.bar(x + w / 2, sh, w, label="% of all positive reward", color=MODEL, hatch="xx",
           edgecolor=INK, linewidth=0.6)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 200)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6.1, rotation=38, ha="right")
    ax.set_ylabel("percent (log scale)")
    ax.set_title("(a) Where the reward lives", loc="left")
    ax.legend(loc="upper right", fontsize=6.4)
    despine(ax); ax.grid(axis="x", visible=False)

    ax = axes[1]
    n = np.arange(9)
    auc = [0.9041, 0.8553, 0.8268, 0.8503, 0.8833, 0.8601, 0.8608, 0.7942, 0.7340]
    ax.plot(n * 0.05, auc, color=MODEL, lw=1.6, marker="o", ms=4, mfc="white", mew=1.1)
    ax.axhline(0.5, color=ALERT, lw=1.0, ls="--")
    ax.text(0.02, 0.52, "chance (validated: noise probe 0.506)", fontsize=6.3, color=ALERT)
    ax.axhline(0.90, color=BLIND, lw=0.8, ls=":")
    ax.text(0.40, 0.912, "acceptance gate G2 = 0.90", fontsize=6.3, color=BLIND, ha="right")
    ax.set_xlabel("MTP offset (seconds of lead at 20 fps)")
    ax.set_ylabel("held-out income-event AUC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("(b) The reward head is not blind", loc="left")
    despine(ax)

    ax = axes[2]
    k = np.array([-4, -2, -1, 0, 1, 2, 3, 4, 5]) * 0.05
    a = [0.557, 0.748, 0.832, 0.904, 0.914, 0.837, 0.768, 0.656, 0.539]
    ax.plot(k, a, color=PROBE, lw=1.6, marker="s", ms=4, mfc="white", mew=1.1)
    ax.axhline(0.5, color=ALERT, lw=1.0, ls="--")
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("target shifted by k (seconds)")
    ax.set_ylabel("offset-0 head AUC")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("(c) ... it is a sharp detector", loc="left")
    despine(ax)
    fig.tight_layout()
    return save(fig, "f07_reward.png")


# =====================================================================
# F8  MTP offset 0 is dead, in every checkpoint including today's
# =====================================================================
def fig_mtp():
    nm = J(f"{FD}/mtp_norms.json")
    keys = list(nm.keys())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharey=True)
    for ax, tag in zip(axes, keys):
        d = nm[tag]
        groups = [("ability heads", d["heads"], MODEL, ""),
                  ("movement heads", d["movement_heads"], PROBE, "xx"),
                  ("gate heads", d["gate_heads"], MARGINAL, ""),
                  ("reward heads (n=0 IS trained)", d["reward_heads"], ORACLE, "..")]
        x = np.arange(9); w = 0.2
        for i, (lab, dd, c, h) in enumerate(groups):
            v = np.array([dd[str(j)] if str(j) in dd else dd[j] for j in range(9)], float)
            ax.bar(x + (i - 1.5) * w, np.maximum(v, 1e-3), w * 0.92, label=lab,
                   color=c, hatch=h, edgecolor=INK, linewidth=0.5)
        ax.set_yscale("log"); ax.set_ylim(1e-3, 3e2)
        ax.set_xticks(x); ax.set_xlabel("MTP offset n")
        ax.axvspan(-0.5, 0.5, color=ALERT, alpha=0.12, lw=0)
        ax.text(0.75, 2.0e-3, "exactly 0.0\n(never touched)", ha="left", fontsize=6.2, color=ALERT)
        ax.set_title(f"({'ab'[keys.index(tag)]}) {tag}", loc="left", fontsize=8.2)
        despine(ax); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("output-weight Frobenius norm (log)")
    axes[0].legend(loc="upper left", fontsize=6.2, ncol=2)
    fig.tight_layout()
    return save(fig, "f08_mtp_dead_head.png")


# =====================================================================
# F9  The Phase-3 blocker chain
# =====================================================================
def fig_phase3():
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    def box(x, y, w, h, text, fc, ec=INK, fs=6.9, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    linewidth=0.9, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2, c=INK, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=8, linewidth=0.9, color=c,
                                     shrinkA=2, shrinkB=2))

    box(1, 38, 20, 8, "Phase-2 checkpoint\n(5 on disk)", "#f0f0f0")
    steps = [
        ("1. sticky movement gate\n`log_prob` hard-raises", "SystemExit at load\n3 of 5 checkpoints"),
        ("2. joint_noop: imagine() stores\ncontinuous xy, log_prob wants ints", "TypeError"),
        ("3. factorized_policy_kl indexes a\n`movement_dim` a joint head lacks", "IndexError"),
        ("4. prior read at MTP offset 0,\npolicy read at offset 1", "KL = 7.287 nats\nwhere it must be 0"),
    ]
    y = 38
    for i, (what, res) in enumerate(steps):
        box(25, y, 40, 8, what, "#f7ecec" if i < 3 else "#f5dede",
            ec=ALERT if i == 3 else INK)
        box(69, y, 30, 8, res, "white", ec=MUTED, fs=6.6,
            tc=ALERT if i == 3 else INK)
        arrow(65, y + 4.0, 69, y + 4.0, MUTED)
        if i < 3:
            arrow(45, y, 45, y - 1.5)
        y -= 9.5
    arrow(21, 42, 25, 42)

    ax.text(1, 1.0,
            "Blockers 1-3 were fixed on 2026-08-28 (`4ca79e0`); blocker 4 on 2026-08-27 (`af72fa2`).\n"
            "Nothing downstream of them has ever been measured: Phase 3 has never completed a training run.",
            fontsize=6.6, color=MUTED, va="bottom")
    ax.set_title("Four independent blockers, in the order they fire", loc="left", fontsize=9.0)
    fig.tight_layout()
    return save(fig, "f09_phase3_blockers.png")


# =====================================================================
# F10  What the labels do and do not cover
# =====================================================================
def fig_labels():
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.85))

    ax = axes[0]
    pf = np.load(f"{FD}/prefirst_lens.npy") / 20.0
    ax.hist(pf, bins=24, color=BLIND, edgecolor=INK, linewidth=0.5)
    ax.axvline(np.median(pf), color=ALERT, lw=1.2, ls="--")
    ax.text(np.median(pf) + 1.0, ax.get_ylim()[1] * 0.86,
            f"median {np.median(pf):.1f} s", color=ALERT, fontsize=6.6)
    ax.axvline(264 / 20.0, color=MODEL, lw=1.2)
    ax.text(264 / 20.0 + 1.2, ax.get_ylim()[1] * 0.62,
            "13.2 s once the\nwalk-out is labelled\nfrom heading_screen",
            color=MODEL, fontsize=6.3, ha="left")
    ax.set_xlabel("time to the first recorded click (s)")
    ax.set_ylabel("games (of 125)")
    ax.set_title("(a) The walk-out was unsupervised", loc="left")
    despine(ax)

    ax = axes[1]
    cov = corpus["state_mask_coverage"]
    labs = ["own\nHP", "own\nlevel", "enemy\nHP", "enemy\nvisible"]
    cols = [MODEL if c > 0.9 else ALERT for c in cov]
    b = ax.bar(range(4), [c * 100 for c in cov], color=cols,
               hatch=["", "", "xx", ""], edgecolor=INK, linewidth=0.6, width=0.6)
    vlabel(ax, b, "{:.1f}%", 1.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs)
    ax.set_ylabel("% of frames with a valid target")
    ax.set_ylim(0, 118)
    ax.set_title("(b) Aux-state label coverage", loc="left")
    despine(ax); ax.grid(axis="x", visible=False)

    ax = axes[2]
    pr = corpus["ability_press_rate"]
    order = sorted(pr, key=lambda k: -pr[k])
    b = ax.bar(range(len(order)), [pr[k] * 100 for k in order],
               color=[MODEL if pr[k] > 1e-3 else ALERT for k in order],
               edgecolor=INK, linewidth=0.5, width=0.66)
    ax.set_yscale("log"); ax.set_ylim(5e-4, 3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, ha="right", fontsize=6.4)
    ax.set_ylabel("% of frames pressed (log)")
    ax.axhline(0.1, color=ALERT, lw=0.9, ls="--")
    ax.text(len(order) - 0.4, 0.115, "0.1% of frames", fontsize=6.3, color=ALERT, ha="right")
    ax.set_title("(c) Every ability is a rare event", loc="left")
    despine(ax); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, "f10_labels.png")


# =====================================================================
# F11  Scale, against the paper
# =====================================================================
def fig_scale():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    rows = [("tokenizer parameters", 206e6, 400e6, "206 M", "400 M"),
            ("dynamics parameters", 146e6, 1.6e9, "146 M", "1.6 B"),
            ("patch tokens per frame", 484, 960, "484", "960"),
            ("action-labelled video (hours)", 49.4, 100, "49.4 h", "100 h"),
            ("unlabelled video (hours)", 0.5, 2441, "0 h", "2,441 h"),
            ("accelerators", 1, 256, "1x RTX 5080", "256-1024x TPU-v5p")]
    y = np.arange(len(rows))[::-1]
    for i, (yy, (lab, ours, theirs, lo, lt)) in enumerate(zip(y, rows)):
        ax.barh(yy + 0.18, theirs, 0.34, color=MARGINAL, edgecolor=INK, linewidth=0.6)
        ax.barh(yy - 0.18, ours, 0.34, color=MODEL, edgecolor=INK, linewidth=0.6)
        ax.text(theirs * 1.45, yy + 0.18, lt, va="center", fontsize=6.5, color=MUTED)
        ax.text(ours * 1.45, yy - 0.18, lo, va="center", fontsize=6.5, color=MODEL)
        if i == 0:
            ax.text(0.45, yy + 0.18, "Dreamer 4 (paper)", va="center", ha="left",
                    fontsize=6.4, color=INK)
            ax.text(0.45, yy - 0.18, "ahriuwu", va="center", ha="left",
                    fontsize=6.4, color="white", fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(0.3, 3e11)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("log scale; units differ per row -- read each row on its own")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_title("The budget the whole result sits inside", loc="left")
    despine(ax, left=False); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, "f11_scale.png")


# =====================================================================
# F12  The pipeline, and where the leak is
# =====================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec=INK, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                    linewidth=0.9, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="center",
                fontsize=7.4, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h / 2 - 2.4, body, ha="center", va="center",
                fontsize=6.4, color=INK)

    def arrow(x1, y1, x2, y2, c=INK, ls="-", lab=None, labdy=1.6):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                     linewidth=1.0, color=c, linestyle=ls, shrinkA=2, shrinkB=2))
        if lab:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + labdy, lab, ha="center",
                    fontsize=6.2, color=c)

    box(1, 30, 21, 18, "Phase 1a  tokenizer",
        "frozen v7, 206 M params\n352x352 -> 484 patches\n-> 512 latents x 16 dim\nreshaped 32 x 16 x 16", "#f2f2f2")
    box(26, 30, 24, 18, "Phase 1b  dynamics",
        "146 M, 18 blocks, dim 768\ndiffusion forcing (shortcut OFF)\naction-conditioned\nglobal_step 8,775 = <1 epoch", "#f2f2f2")
    box(54, 30, 22, 18, "Phase 2  BC + reward",
        "backbone FROZEN\n31.7 M agent blocks + heads\n9 MTP offsets, n=1..8\nthis is what we deploy", "#e8eef4", ec=MODEL)
    box(80, 30, 19, 18, "Phase 3  imagination",
        "PMPO in the dream\nH = 8 frames = 0.4 s\nNEVER RUN", "#f7ecec", ec=ALERT, tc=ALERT)

    arrow(22, 39, 26, 39); arrow(50, 39, 54, 39); arrow(76, 39, 80, 39, ALERT, "--")

    box(26, 6, 24, 15, "action conditioning",
        "the movement action is\nappended as a token in\nframe t's token set\n(dynamics.py:776)", "#fdf3ec", ec=PROBE)
    box(54, 6, 22, 15, "the BC target",
        "predict a_{t+n}, n>=1\nbut a_{t+1} == a_t on\n87.3% of frames\n(measured, 3.55M frames)", "#fdf3ec", ec=PROBE)

    arrow(38, 21, 38, 30, PROBE)
    arrow(65, 21, 65, 30, PROBE)
    ax.add_patch(FancyArrowPatch((50, 13.5), (54, 13.5), arrowstyle="-|>", mutation_scale=9,
                                 linewidth=1.4, color=ALERT, shrinkA=2, shrinkB=2))
    ax.text(52, 17.6, "THE LEAK", ha="center", fontsize=7.0, color=ALERT, fontweight="bold")
    ax.text(52, 3.0, "the answer to the question is sitting in the model's own input",
            ha="center", fontsize=6.6, color=ALERT, style="italic")
    ax.set_title("The stack, and the one edge that broke it", loc="left", fontsize=9.0)
    fig.tight_layout()
    return save(fig, "f12_pipeline.png")




# =====================================================================
# F13  The enemy-gold stream, re-measured for this report
# =====================================================================
def fig_gold():
    g = J(f"{FD}/gold_stats.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))

    ax = axes[0]
    e = np.array(g["edges"]); po = np.array(g["p_own"]); pe = np.array(g["p_opp"])
    m = e <= 90
    ax.step(e[m], po[m] * 100, where="mid", color=MODEL, lw=1.5, label="Garen (the control)")
    ax.step(e[m], pe[m] * 100, where="mid", color=PROBE, lw=1.5, ls="--",
            label="lane opponent")
    for gv, lab, tx, ty in ((14, "caster minion", 6.0, 20), (20, "melee minion", 24.0, 25)):
        ax.axvline(gv, color=MUTED, lw=0.6, ls=":")
        ax.annotate(f"{gv} g\n{lab}", xy=(gv, ty), xytext=(tx, ty), fontsize=6.3, color=MUTED,
                    va="center", ha="right" if tx < gv else "left",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlabel("single-frame gold gain (gold)")
    ax.set_ylabel("% of all lumps >= 3 g")
    ax.set_xlim(3, 90)
    ax.set_title(f"(a) Lump spectra, TVD = {g['tvd']:.3f}", loc="left")
    ax.legend(loc="upper right", fontsize=6.8)
    despine(ax)

    ax = axes[1]
    b = ax.bar([0, 1], [g["rate_on"] * 100, g["rate_off"] * 100],
               color=[MODEL, PROBE], hatch=["", "xx"], edgecolor=INK, linewidth=0.6, width=0.5)
    vlabel(ax, b, "{:.2f}%", 0.16)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["opponent ON screen", "opponent OFF screen"])
    ax.set_ylabel("% of frames where opponent gold moves")
    ax.set_ylim(0, 14)
    ax.text(0.5, 12.4, f"off/on ratio {g['rate_ratio']:.3f}\n"
                       f"longest constant hold: own {g['hold_own_max_frames']} frames,\n"
                       f"opponent {g['hold_opp_max_frames']} frames",
            ha="center", fontsize=6.6, color=MUTED)
    ax.set_title("(b) Visibility does not gate the read", loc="left")
    despine(ax); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, "f13_enemy_gold.png")


if __name__ == "__main__":
    for f in (fig_reward, fig_mtp, fig_phase3, fig_labels, fig_scale, fig_pipeline, fig_gold):
        f()
