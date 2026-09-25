"""Audit D: what does the HUD mask actually cover, and what does the data look like."""
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUTDIR = "/srv/nfs/projects/ahriuwu/scratchpad/audit_mask"
os.makedirs(OUTDIR, exist_ok=True)
MASK_PT = "/srv/nfs/projects/ahriuwu/scratchpad/hud_valid_mask_352.pt"
REPLAY = "/srv/nfs/datasets/lol_replays_16_9_772/NA1_5549995114/frames"
YT_EVAL = "/srv/nfs/datasets/yt_eval_frames_352"
N = 500

valid = torch.load(MASK_PT, weights_only=True).numpy()          # 1=content 0=HUD
masked = valid == 0


def stack_stats(paths):
    acc = None
    acc2 = None
    mn = None
    n = 0
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        g = img.astype(np.float64) / 255.0
        g = g.mean(axis=2) if g.ndim == 3 else g
        if acc is None:
            acc = np.zeros_like(g)
            acc2 = np.zeros_like(g)
            mn = np.full_like(g, 1e9)
        acc += g
        acc2 += g * g
        mn = np.minimum(mn, g)
        n += 1
    mean = acc / n
    var = np.maximum(acc2 / n - mean * mean, 0.0)
    return mean, var, mn, n


# ---- replay ----
rf = sorted(f for f in os.listdir(REPLAY) if f.endswith(".png"))
sel = [os.path.join(REPLAY, rf[i]) for i in np.linspace(1000, len(rf) - 1, N).astype(int)]
r_mean, r_var, r_min, r_n = stack_stats(sel)

# ---- yt eval ----
ytdirs = sorted(d for d in os.listdir(YT_EVAL) if d.startswith("yt_"))
ysel = []
for d in ytdirs[:2]:
    fs = sorted(os.listdir(os.path.join(YT_EVAL, d)))
    fs = [f for f in fs if f.endswith((".jpg", ".png"))]
    ysel += [os.path.join(YT_EVAL, d, fs[i]) for i in np.linspace(0, len(fs) - 1, N // 2).astype(int)]
y_mean, y_var, y_min, y_n = stack_stats(ysel)

EPS = 1e-12
r_zero = r_var <= EPS
y_zero = y_var <= EPS
r_black = r_max_black = (r_mean <= 0.02) & (r_var <= 1e-8)
y_black = (y_mean <= 0.02) & (y_var <= 1e-8)


def frac(a):
    return float(a.mean())


# ---- named regions on the 352x352 canvas (League 16:9 letterboxed to square) ----
# Frame is 1280x720 squashed/letterboxed into 352x352. Report the mask coverage
# of each region as measured, not assumed.
regions = {
    "top_bar_y0_40": (slice(0, 40), slice(0, 352)),
    "left_pillar_x0_28": (slice(0, 352), slice(0, 28)),
    "right_pillar_x324_352": (slice(0, 352), slice(324, 352)),
    "bottom_strip_y280_352": (slice(280, 352), slice(0, 352)),
    "bottom_center_hud_y280_352_x110_240": (slice(280, 352), slice(110, 240)),
    "bottom_right_minimap_y255_352_x255_352": (slice(255, 352), slice(255, 352)),
    "center_playfield_y60_260_x40_310": (slice(60, 260), slice(40, 310)),
}

report = {"mask_masked_frac": frac(masked), "replay_n": r_n, "yt_n": y_n}
report["regions"] = {}
for name, (ys, xs) in regions.items():
    report["regions"][name] = {
        "mask_masked_frac": frac(masked[ys, xs]),
        "replay_zero_var_frac": frac(r_zero[ys, xs]),
        "replay_black_frac": frac(r_black[ys, xs]),
        "replay_mean_var": float(r_var[ys, xs].mean()),
        "yt_zero_var_frac": frac(y_zero[ys, xs]),
        "yt_black_frac": frac(y_black[ys, xs]),
        "yt_mean_var": float(y_var[ys, xs].mean()),
    }

report["global"] = {
    "replay_zero_var_frac": frac(r_zero), "replay_black_frac": frac(r_black),
    "yt_zero_var_frac": frac(y_zero), "yt_black_frac": frac(y_black),
    "mask_vs_replay_zero_iou": float((masked & r_zero).sum() / max((masked | r_zero).sum(), 1)),
    "mask_vs_yt_black_iou": float((masked & y_black).sum() / max((masked | y_black).sum(), 1)),
    "masked_but_replay_varying_frac": float((masked & ~r_zero).sum() / max(masked.sum(), 1)),
    "replay_var_inside_mask_mean": float(r_var[masked].mean()),
    "replay_var_outside_mask_mean": float(r_var[~masked].mean()),
}

# exact row/col extent of the fully-masked bottom band
rowfrac = masked.mean(axis=1)
colfrac = masked.mean(axis=0)
full_rows = np.nonzero(rowfrac > 0.99)[0]
report["fully_masked_rows"] = [int(full_rows.min()), int(full_rows.max())] if len(full_rows) else []
report["n_fully_masked_rows"] = int(len(full_rows))
report["row_mask_frac_every16"] = {int(i): round(float(rowfrac[i]), 3) for i in range(0, 352, 16)}
report["col_mask_frac_every16"] = {int(i): round(float(colfrac[i]), 3) for i in range(0, 352, 16)}

json.dump(report, open(os.path.join(OUTDIR, "audit_d_report.json"), "w"), indent=1)
print(json.dumps(report, indent=1))

# ---- visualization ----
sample_r = cv2.cvtColor(cv2.imread(sel[len(sel) // 2]), cv2.COLOR_BGR2RGB)
sample_y = cv2.cvtColor(cv2.imread(ysel[len(ysel) // 2]), cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
ax[0, 0].imshow(sample_r); ax[0, 0].set_title("REPLAY sample frame (352x352)")
ax[0, 1].imshow(np.log10(r_var + 1e-10), cmap="magma"); ax[0, 1].set_title("REPLAY log10 per-pixel variance (500 frames)")
ax[0, 2].imshow(r_zero, cmap="gray"); ax[0, 2].set_title(f"REPLAY zero-variance pixels ({frac(r_zero)*100:.1f}%)")
ov = sample_r.copy(); ov[masked] = (ov[masked] * 0.25).astype(np.uint8)
ax[0, 3].imshow(ov); ax[0, 3].set_title("REPLAY + hud_valid_mask (dim = masked)")
ax[1, 0].imshow(sample_y); ax[1, 0].set_title("YT sample frame (352x352)")
ax[1, 1].imshow(np.log10(y_var + 1e-10), cmap="magma"); ax[1, 1].set_title("YT log10 per-pixel variance (500 frames)")
ax[1, 2].imshow(masked, cmap="gray"); ax[1, 2].set_title(f"hud_valid_mask_352.pt masked ({frac(masked)*100:.1f}%)")
ovy = sample_y.copy(); ovy[masked] = (ovy[masked] * 0.25).astype(np.uint8)
ax[1, 3].imshow(ovy); ax[1, 3].set_title("YT + hud_valid_mask (dim = masked)")
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "hud_mask_audit.png"), dpi=110)
print("WROTE", os.path.join(OUTDIR, "hud_mask_audit.png"))

np.savez_compressed(os.path.join(OUTDIR, "stats.npz"),
                    replay_var=r_var.astype(np.float32), replay_mean=r_mean.astype(np.float32),
                    yt_var=y_var.astype(np.float32), yt_mean=y_mean.astype(np.float32),
                    mask_valid=valid.astype(np.uint8))
