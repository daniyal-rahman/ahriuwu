#!/usr/bin/env python
"""Every figure in the ahriuwu report, built from real measurements.

Layout convention: long category names always go on a horizontal axis (hbar),
titles are one short line, and the detail lives in the report caption.
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *          # noqa
import matplotlib.pyplot as plt

setup()
R = "/srv/nfs/projects/ahriuwu"
FD = f"{R}/reports/figdata"
D0 = f"{R}/scratchpad/decision0"
J = lambda p: json.load(open(p))

corpus = J(f"{FD}/corpus_stats.json")
cursor = J(f"{FD}/cursor_stats.json")
lane   = J(f"{FD}/lane_table.json")


def hbar(ax, labels, vals, cols, hats, fmt="{:.3f}", pad=None, xlim=None):
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=cols, hatch=hats, edgecolor=INK, linewidth=0.6, height=0.66)
    span = (xlim[1] - xlim[0]) if xlim else max(vals)
    pad = pad if pad is not None else span * 0.012
    for yy, v in zip(y, vals):
        ax.text(v + pad, yy, fmt.format(v), va="center", ha="left", fontsize=6.9)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    if xlim: ax.set_xlim(*xlim)
    despine(ax, left=False)
    ax.grid(axis="y", visible=False)
    return y


def vlabel(ax, bars, fmt="{:.3f}", dy=0.0, fs=6.9):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fs)


# =====================================================================
# F1  model vs a no-pixel lookup table vs chance
# =====================================================================
def fig_core():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    ax = axes[0]
    labels = ["uniform (chance)", "corpus marginal", "trained model,\nACTION INPUT CUT",
              "blind 21x21 table\n(no pixels)", "TRAINED MODEL\n(146M stack)"]
    vals = [6.089, 5.216, 5.450, 4.357, 4.365]
    y = hbar(ax, labels, vals, [CHANCE, MARGINAL, PROBE, BLIND, MODEL],
             ["", "", "xx", "//", ""], xlim=(0, 7.6), pad=0.09)
    ax.plot([4.36, 4.36], [y[3], y[4]], color=ALERT, lw=1.2)
    ax.annotate("0.008 nats\napart", xy=(4.36, (y[3] + y[4]) / 2), xytext=(5.5, (y[3] + y[4]) / 2),
                fontsize=6.8, color=ALERT, va="center",
                arrowprops=dict(arrowstyle="->", color=ALERT, lw=0.8))
    ax.set_xlabel("event-frame movement cross-entropy (nats)")
    ax.set_title("(a) The measurement that started it", loc="left")

    ax = axes[1]
    cell = J(f"{D0}/ce_lcell.json")["cell|heldout6"]
    g = lambda k: cell[k]["ce"]
    labels = ["uniform (chance)", "corpus marginal", "blind table (30-game fit)",
              "ORACLE full ground-truth state", "DEPLOYED model",
              "probe on ablated features", "ORACLE future path (cheats)"]
    vals = [g("uniform"), g("marginal"), g("blind_table"), g("ORACLE_state+mlp"),
            g("DEPLOYED_head_act"), g("agent_noact+mlp"), g("ORACLE_future+mlp")]
    y = hbar(ax, labels, vals, [CHANCE, MARGINAL, BLIND, ORACLE, MODEL, PROBE, "#3a3a3a"],
             ["", "", "//", "..", "", "xx", "\\\\"], xlim=(0, 7.6), pad=0.09)
    ax.axvspan(g("ORACLE_state+mlp"), g("blind_table"), color=ALERT, alpha=0.30, lw=0)
    ax.annotate("the whole visible headroom\nin this metric: 0.043 nats",
                xy=(4.11, 1.6), xytext=(4.75, 0.35), fontsize=6.6, color=ALERT,
                arrowprops=dict(arrowstyle="->", color=ALERT, lw=0.8))
    ax.set_xlabel("event-frame movement cross-entropy (nats)")
    ax.set_title("(b) The same test at 7.6x the data", loc="left")
    fig.tight_layout()
    return save(fig, "f01_core_result.png")


# =====================================================================
# F2  Walk to lane, before and after cutting the channel
# =====================================================================
def fig_lane():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    ax = axes[0]
    rows = [("human", lane["HUMAN"], "#3a3a3a", ""),
            ("model, action HELD (deployed)", lane["held_executed"], MODEL, ""),
            ("model, action CUT (no retrain)", lane["none_executed"], PROBE, "xx")]
    lanes = ["TOP", "MID", "BOT"]
    x = np.arange(3); w = 0.26
    for i, (lab, cnt, c, h) in enumerate(rows):
        b = ax.bar(x + (i - 1) * w, [cnt[L] for L in lanes], w * 0.9, label=lab,
                   color=c, hatch=h, edgecolor=INK, linewidth=0.6)
        vlabel(ax, b, "{:.0f}", 0.5)
    ax.set_xticks(x); ax.set_xticklabels(["TOP  (correct)", "MID", "BOT"])
    ax.set_ylabel("games (of 40)"); ax.set_ylim(0, 46)
    ax.set_title("(a) Lane the executed command points down", loc="left")
    ax.legend(loc="upper right", fontsize=6.8)
    despine(ax); ax.grid(axis="x", visible=False)

    ax = axes[1]
    pg = J(f"{D0}/head_walkout.json")["per_game"]
    for lab, key, c, ls, mk in [("movement action CUT", "noact_med_err", PROBE, "-", "o"),
                                ("as deployed (action in)", "act_med_err", MODEL, "-", "s"),
                                ("CENTER control", "CENTER_med_err", MARGINAL, "--", "^")]:
        v = np.sort([g[key] for g in pg if g.get(key) is not None])
        yy = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, yy, where="post", color=c, lw=1.5, ls=ls, label=lab)
        idx = np.linspace(0, len(v) - 1, 7).astype(int)
        ax.plot(v[idx], yy[idx], mk, color=c, ms=3.2, mfc="white", mew=1.0)
    for key, c, dx in (("noact_med_err", PROBE, 3), ("act_med_err", MODEL, 3)):
        m = np.median([g[key] for g in pg])
        ax.axvline(m, color=c, lw=0.7, ls=":", alpha=0.9)
        ax.text(m + dx, 0.03, f"median {m:.1f}°", color=c, fontsize=6.5, rotation=90, va="bottom")
    ax.set_xlabel("per-game median direction error vs the human's heading (deg)")
    ax.set_ylabel("fraction of the 112 games")
    ax.set_xlim(0, 180); ax.set_ylim(0, 1.03); ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_title("(b) Commanded direction, open loop", loc="left")
    ax.legend(loc="lower right", fontsize=6.8)
    despine(ax)
    fig.tight_layout()
    return save(fig, "f02_walk_to_lane.png")


# =====================================================================
# F3  The crutch, quantified
# =====================================================================
def fig_crutch():
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.85))

    ax = axes[0]
    vals = [corpus["movement"]["exact_repeat_frac"], cursor["cursor"]["exact_repeat"]]
    b = ax.bar([0, 1], vals, color=[MODEL, PROBE], hatch=["", "xx"],
               edgecolor=INK, linewidth=0.6, width=0.5)
    vlabel(ax, b, "{:.1%}", 0.015)
    lo, hi = corpus["movement"]["per_game_repeat_min"], corpus["movement"]["per_game_repeat_max"]
    ax.plot([0.30, 0.30], [lo, hi], color=INK, lw=1.0)
    for v in (lo, hi):
        ax.plot([0.26, 0.34], [v, v], color=INK, lw=1.0)
    ax.text(0.38, (lo + hi) / 2, "per-game\nrange", fontsize=6.3, va="center", color=MUTED)
    ax.set_xticks([0, 1]); ax.set_xlim(-0.55, 1.55)
    ax.set_xticklabels(["movement\n(held click target)\nIS AN INPUT",
                        "cursor\n(pointer position)\nobservation only"], fontsize=7.0)
    ax.set_ylabel("P(value at t+1 identical to t)"); ax.set_ylim(0, 1.05)
    ax.set_title("(a) The two spatial channels", loc="left")
    despine(ax); ax.grid(axis="x", visible=False)

    ax = axes[1]
    runs = np.load(f"{FD}/hold_runs.npy")
    ax.hist(runs, bins=np.logspace(0, np.log10(runs.max() + 1), 44),
            color=BLIND, edgecolor=INK, linewidth=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    for q, lab in ((np.median(runs), "median\n4 frames"), (np.percentile(runs, 99), "p99\n77 frames")):
        ax.axvline(q, color=ALERT, lw=0.9, ls="--")
        ax.text(q * 1.3, 1.2e4, lab, color=ALERT, fontsize=6.4)
    ax.set_xlabel("hold-run length (frames at 20 fps)")
    ax.set_ylabel("number of runs")
    ax.set_title("(b) How long one order is held", loc="left")
    despine(ax)

    ax = axes[2]
    dc = corpus["dropout_curve"]
    ax.plot(dc["p"], dc["available"], color=MODEL, lw=1.8)
    for pv, c, off in ((0.15, ALERT, (10, -4)), (0.50, MUTED, (8, -6)), (0.95, MUTED, (-72, 14))):
        av = corpus["dropout_at"][str(pv)]
        ax.plot([pv], [av], "o", color=c, ms=5, mfc="white", mew=1.4)
        ax.annotate(f"p = {pv}  ->  {av:.1%}", (pv, av), textcoords="offset points",
                    xytext=off, fontsize=6.5, color=c)
    ax.set_xlabel("per-frame action-dropout rate p")
    ax.set_ylabel("P(the answer survives somewhere\nin the 16-frame context)")
    ax.set_ylim(0, 1.06); ax.set_xlim(0, 1.0)
    ax.set_title("(c) Why per-frame dropout cannot work", loc="left")
    despine(ax)
    fig.tight_layout()
    return save(fig, "f03_the_crutch.png")


# =====================================================================
# F4  Decision 0
# =====================================================================
def fig_decision0():
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))

    ax = axes[0]
    w = J(f"{D0}/ce_wdir8.json")["wdir|heldout6"]
    arms = [("corpus marginal", "marginal", MARGINAL, ""),
            ("ORACLE side + game time", "ORACLE_sidetime+mlp", ORACLE, ".."),
            ("ORACLE true map position", "ORACLE_pos+mlp", ORACLE, ".."),
            ("ORACLE full ground-truth state", "ORACLE_state+mlp", ORACLE, ".."),
            ("agent token, action ABLATED", "agent_noact+mlp", PROBE, "xx"),
            ("block-11 tokens, ABLATED", "d11_mean_noact+mlp", MODEL, "")]
    y = np.arange(len(arms))[::-1]
    for yy, (lab, key, c, h) in zip(y, arms):
        d = w[key]["vs_base"]["delta"]
        ax.barh(yy, d, color=c, hatch=h, edgecolor=INK, linewidth=0.6, height=0.62)
        per = list(w[key]["vs_base"]["per_game"].values())
        ax.plot(per, [yy] * len(per), "o", color="white", ms=3.0,
                markeredgecolor=INK, markeredgewidth=0.55, zorder=5)
        ax.text(d - 0.035 if d < 0 else d + 0.035, yy - 0.40, f"{d:+.3f}",
                va="center", ha="right" if d < 0 else "left", fontsize=6.7)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels([a[0] for a in arms])
    ax.set_xlabel("CE change vs the blind persistence table (nats)")
    ax.set_xlim(-0.46, 0.70)
    ax.text(-0.44, len(arms) - 0.55, "BETTER  <--", fontsize=6.4, color=MUTED)
    ax.text(0.66, len(arms) - 0.55, "--> WORSE", fontsize=6.4, color=MUTED, ha="right")
    ax.set_title("(a) Walk-out direction, 8-way", loc="left")
    despine(ax, left=False); ax.grid(axis="y", visible=False)

    ax = axes[1]
    labels = ["train marginal", "best constant per game (oracle)", "map-position table 32x32",
              "ridge on ONE frozen latent", "small CNN on one latent",
              "previous-click table (THE CRUTCH)"]
    vals = [0.1842, 0.2027, 0.2732, 0.3776, 0.3819, 0.5634]
    yy = hbar(ax, labels, vals, [MARGINAL, MARGINAL, BLIND, MODEL, MODEL, "#3a3a3a"],
              ["", "..", "//", "", "xx", "\\\\"], fmt="{:.1%}", xlim=(0, 0.78), pad=0.009)
    null, nsd = 0.1539, 0.0033
    ax.axvspan(null - 2 * nsd, null + 2 * nsd, color=ALERT, alpha=0.20, lw=0)
    ax.axvline(null, color=ALERT, lw=1.0, ls="--")
    ax.annotate("MEASURED null 15.4%\n(within-game permutation;\nuniform would say 12.5%)",
                xy=(null, yy[2] - 0.35), xytext=(0.30, yy[1] - 0.55), fontsize=6.4, color=ALERT,
                arrowprops=dict(arrowstyle="->", color=ALERT, lw=0.8))
    ax.set_xlabel("next-click octant accuracy")
    ax.set_title("(b) Independent replication, 10,982 events", loc="left")
    fig.tight_layout()
    return save(fig, "f04_decision0.png")


# =====================================================================
# F5  The evaluation trap
# =====================================================================
def fig_eval_trap():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.95))

    ax = axes[0]
    labels = ["uniform head (known-answer check)", "DEPLOYED head (a proven copier)",
              "ridge probe on frozen latents", "blind copier (repeat, re-aimed)",
              "perfect head"]
    vals = [0.1246, 0.3770, 0.3776, 0.4585, 1.0]
    y = hbar(ax, labels, vals, [CHANCE, MODEL, PROBE, BLIND, "#3a3a3a"],
             ["", "", "xx", "//", ".."], fmt="{:.1%}", xlim=(0, 1.20), pad=0.014)
    ax.axvline(0.1386, color=ALERT, lw=1.0, ls="--")
    ax.annotate("measured permutation null, 13.9%", xy=(0.1386, y[4]), xytext=(0.42, y[4]),
                fontsize=6.5, color=ALERT, va="center",
                arrowprops=dict(arrowstyle="->", color=ALERT, lw=0.8))
    ax.set_xlabel("commanded-direction octant accuracy")
    ax.set_title("(a) A blind copier clears the null", loc="left")

    ax = axes[1]
    names = ["uniform", "marginal\nonly", "DEPLOYED", "blind\ncopier"]
    b = ax.bar(range(4), [0.0, 0.04, 26.4, 53.0], color=[CHANCE, MARGINAL, MODEL, BLIND],
               hatch=["", "", "", "//"], edgecolor=INK, linewidth=0.6, width=0.56)
    vlabel(ax, b, "{:+.2f}", 0.9)
    ax.axhline(2.0, color=ALERT, lw=1.0, ls="--")
    ax.text(-0.36, 7.5, "z = 2  (a naive\nsignificance bar)",
            fontsize=6.5, color=ALERT, ha="left")
    ax.set_xticks(range(4)); ax.set_xticklabels(names)
    ax.set_ylabel("z against the measured permutation null (sd)")
    ax.set_ylim(0, 62)
    ax.set_title("(b) The null is not a test of vision", loc="left")
    despine(ax); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, "f05_eval_trap.png")


# =====================================================================
# F6  What the retrain showed
# =====================================================================
def fig_retrain():
    txt = open(f"{R}/ops/bronze_retrain_run.log").read()
    blocks = re.findall(
        r"\[(?:VAL @ step (\d+)|EPOCH 0 VAL)\][\s\S]*?"
        r"(?:move_event_ce=([\d.]+)[\s\S]*?)?"
        r"dir_octant=([\d.]+)%.*?null ([\d.]+)\+-([\d.]+)%[\s\S]*?"
        r"dir_median_angle=([\d.]+)deg", txt)
    steps = np.array([int(b[0]) if b[0] else 25283 for b in blocks])
    ces   = np.array([float(b[1]) if b[1] else np.nan for b in blocks])
    dirs  = np.array([float(b[2]) for b in blocks])
    nulls = np.array([float(b[3]) for b in blocks])
    nsds  = np.array([float(b[4]) for b in blocks])
    meds  = np.array([float(b[5]) for b in blocks])
    json.dump(dict(steps=steps.tolist(), dir_octant=dirs.tolist(), null=nulls.tolist(),
                   null_sd=nsds.tolist(), move_event_ce=ces.tolist(),
                   median_angle=meds.tolist()),
              open(f"{FD}/retrain_series.json", "w"), indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))

    ax = axes[0]
    ax.fill_between(steps, nulls - 2 * nsds, nulls + 2 * nsds, color=ALERT, alpha=0.15, lw=0,
                    label="measured null $\\pm$ 2 sd")
    ax.plot(steps, nulls, color=ALERT, lw=0.9, ls="--")
    ax.plot(steps, dirs, color=MODEL, lw=1.5, marker="o", ms=3.0, mfc="white", mew=0.9,
            label="retrained head, action channel removed")
    ax.axhline(50.68, color=BLIND, lw=1.5)
    ax.text(25000, 51.9, "ACCEPTANCE BAR: blind copier, 50.7%", fontsize=6.5,
            color=BLIND, ha="right")
    ax.axhline(37.70, color=MUTED, lw=0.8, ls=":")
    ax.text(25000, 38.8, "the deployed copier it replaced, 37.7%",
            fontsize=6.3, color=MUTED, ha="right")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("commanded-direction octant accuracy (%)")
    ax.set_ylim(0, 58)
    ax.set_title("(a) One full epoch with the crutch removed", loc="left")
    ax.legend(loc="lower left", fontsize=6.6)
    despine(ax)

    ax = axes[1]
    ok = ~np.isnan(ces)
    ax.plot(steps[ok], ces[ok], color=MODEL, lw=1.4, marker="o", ms=2.8, mfc="white", mew=0.8,
            label="retrained head")
    ax.axhline(5.232, color=MARGINAL, lw=1.0, ls="-.", label="corpus marginal, 5.232")
    ax.axhline(4.0706, color=BLIND, lw=1.5, label="blind table on this split, 4.071")
    ax.axhline(4.1212, color=MUTED, lw=0.8, ls=":", label="deployed copier, 4.121")
    for x, v, ty, lab in ((5000, 4.827, 4.62, "A/B arm R0: control (action held)"),
                          (5000, 5.192, 5.46, "A/B arm R1: action channel cut")):
        ax.plot([x], [v], "^", color=PROBE, ms=5.5, mfc="white", mew=1.2)
        ax.annotate(lab, xy=(x, v), xytext=(x + 900, ty), fontsize=6.4, color=PROBE,
                    va="center", arrowprops=dict(arrowstyle="-", color=PROBE, lw=0.6))
    ax.set_xlabel("optimizer step"); ax.set_ylabel("event-frame movement CE (nats)")
    ax.set_ylim(3.8, 6.1)
    ax.set_title("(b) The old acceptance metric, same run", loc="left")
    ax.legend(loc="upper right", fontsize=6.3)
    despine(ax)
    fig.tight_layout()
    return save(fig, "f06_retrain.png")


if __name__ == "__main__":
    for f in (fig_core, fig_lane, fig_crutch, fig_decision0, fig_eval_trap, fig_retrain):
        f()
