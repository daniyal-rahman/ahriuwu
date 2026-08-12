"""Audit D part 2: is there ANY HUD chrome in the replay frames? Zoom the regions
where League draws minimap / champion HP bar / ability row, in replay vs YT."""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/srv/nfs/projects/ahriuwu/scratchpad/audit_mask"
R = "/srv/nfs/datasets/lol_replays_16_9_772/NA1_5549995114/frames"
Y = "/srv/nfs/datasets/yt_eval_frames_352"

rf = sorted(f for f in os.listdir(R) if f.endswith(".png"))
rsel = [os.path.join(R, rf[i]) for i in np.linspace(2000, len(rf) - 1, 6).astype(int)]
yd = sorted(d for d in os.listdir(Y) if d.startswith("yt_"))[0]
yfs = sorted(f for f in os.listdir(os.path.join(Y, yd)) if f.endswith((".jpg", ".png")))
ysel = [os.path.join(Y, yd, yfs[i]) for i in np.linspace(0, len(yfs) - 1, 6).astype(int)]

fig, ax = plt.subplots(2, 6, figsize=(24, 8))
for j, p in enumerate(rsel):
    ax[0, j].imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
    ax[0, j].set_title(f"REPLAY {os.path.basename(p)}", fontsize=9)
for j, p in enumerate(ysel):
    ax[1, j].imshow(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
    ax[1, j].set_title(f"YT {os.path.basename(p)}", fontsize=9)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
plt.suptitle("Replay frames (HUD disabled at record time) vs YT frames (HUD blacked out)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "hud_presence_replay_vs_yt.png"), dpi=100)
print("WROTE", os.path.join(OUT, "hud_presence_replay_vs_yt.png"))

# zoom bottom-right (minimap) + bottom-center (HP/ability bar) for replay
img = cv2.cvtColor(cv2.imread(rsel[2]), cv2.COLOR_BGR2RGB)
imy = cv2.cvtColor(cv2.imread(ysel[2]), cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(2, 3, figsize=(13, 9))
for row, (im, lab) in enumerate([(img, "REPLAY"), (imy, "YT")]):
    ax[row, 0].imshow(im); ax[row, 0].set_title(f"{lab} full 352x352")
    ax[row, 1].imshow(im[250:352, 250:352]); ax[row, 1].set_title(f"{lab} y250-352 x250-352 (minimap slot)")
    ax[row, 2].imshow(im[270:352, 100:250]); ax[row, 2].set_title(f"{lab} y270-352 x100-250 (HP/ability slot)")
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT, "hud_region_zoom.png"), dpi=110)
print("WROTE", os.path.join(OUT, "hud_region_zoom.png"))
