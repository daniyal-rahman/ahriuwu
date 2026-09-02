"""Shared figure style for the ahriuwu report.

Design constraints (from the brief): readable in greyscale, axis labels with
units, no chartjunk.  The palette is therefore chosen for LIGHTNESS separation
rather than hue separation, and every categorical encoding is doubled with a
hatch or a marker so that a monochrome print still distinguishes the series.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- palette -------------------------------------------------------------
# Neutrals carry the *baselines* (chance -> marginal -> blind table): a single
# sequential ramp, light to dark, because they encode a magnitude ordering.
CHANCE   = "#d9d9d9"   # L* 86
MARGINAL = "#adadad"   # L* 71
BLIND    = "#6b6b6b"   # L* 45  -- the bar everything is judged against
# Accents carry *identities*: the trained model, and a probe/oracle.
MODEL    = "#1f4e79"   # L* 33
PROBE    = "#9c4a1a"   # L* 41
ORACLE   = "#5b7f5b"   # L* 50
ALERT    = "#8a1c1c"   # L* 32
INK      = "#1a1a1a"
MUTED    = "#666666"
GRID     = "#d0d0d0"

HATCH = {"blind": "//", "model": "", "probe": "xx", "oracle": "..", "chance": ""}

def setup():
    rcParams.update({
        "figure.dpi": 190,
        "savefig.dpi": 190,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.2,
        "axes.edgecolor": "#9a9a9a",
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.9,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7.4,
        "legend.frameon": False,
        "hatch.linewidth": 0.6,
        "text.color": INK,
        "axes.labelcolor": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

def despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)

def save(fig, name):
    import os
    p = os.path.join("/srv/nfs/projects/ahriuwu/reports/figures", name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)
    return p
