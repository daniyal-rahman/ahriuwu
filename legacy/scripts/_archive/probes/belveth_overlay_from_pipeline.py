"""Overlay script for pipeline.py output.
Reads labels.json + frames/*.png (352x352) and renders a 1920x1080 video with:
  - frame upscaled 352->1080, centered on 1920x1080 canvas (black bars)
  - HUD: game time, champion stats, action
  - Action markers (attack / click)
Writes: C:\\tmp\\replay_data\\NA1_5545727197\\overlay.mp4
"""
import os, json, cv2, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

DIR = r"C:\tmp\replay_data\NA1_5545727197"
OUT = os.path.join(DIR, "overlay.mp4")
FPS = 20
OUT_W, OUT_H = 1920, 1080

labels = json.load(open(os.path.join(DIR, "labels.json")))
frames = labels["frames"]
print(f"{len(frames)} frames, champion={labels['champion']}, fps={labels['fps']}")

# Upscale params: 352 -> 1080 square; place at x = (1920-1080)/2 = 420
FR_SIZE = 1080  # square render size (height of output)
PAD_X   = (OUT_W - FR_SIZE) // 2  # 420

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(OUT, fourcc, FPS, (OUT_W, OUT_H))

# Action colors
COLOR = {"idle": (120,120,120), "attack": (0,200,255), "Q": (255,200,0),
         "W": (200,255,200), "E": (0,255,0), "R": (0,0,255), "cast": (255,100,255)}

for fr in frames:
    idx = fr["frame"]
    png = os.path.join(DIR, "frames", f"{idx:06d}.png")
    img = cv2.imread(png)
    if img is None: continue

    # Upscale 352->1080
    img = cv2.resize(img, (FR_SIZE, FR_SIZE), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    canvas[:, PAD_X:PAD_X+FR_SIZE] = img

    lab = fr.get("label")
    gt = fr["gt"]
    # HUD strings
    lines = [f"gt={gt:6.2f}s  frame={idx}"]
    action = "-"
    if lab:
        stats = lab.get("garen_stats", {})
        lines.append(f"HP {stats.get('hp',0):.0f}/{stats.get('hp_max',0):.0f}  "
                     f"Gold {stats.get('gold',0):.0f}  Lvl {stats.get('level',0)}")
        gw = lab.get("garen_world", [0,0])
        lines.append(f"world: ({gw[0]:.0f}, {gw[1]:.0f})")
        action = lab.get("action", {}).get("type", "-")
        lines.append(f"action: {action}")
    else:
        lines.append("(unlabeled)")

    # Action indicator on right side
    ac = COLOR.get(action, (100,100,100))
    cv2.rectangle(canvas, (OUT_W-180, 30), (OUT_W-30, 100), ac, -1)
    cv2.putText(canvas, action.upper(), (OUT_W-170, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(canvas, action.upper(), (OUT_W-170, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 1, cv2.LINE_AA)

    # HUD text — top-left
    for j, line in enumerate(lines):
        cv2.putText(canvas, line, (20, 40+j*36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(canvas, line, (20, 40+j*36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,255,255), 2, cv2.LINE_AA)

    # Center marker (Bel'Veth is at screen center because cam was locked)
    cv2.circle(canvas, (OUT_W//2, OUT_H//2), 28, (0,255,0), 2)
    cv2.line(canvas, (OUT_W//2-22, OUT_H//2), (OUT_W//2+22, OUT_H//2), (0,255,0), 2)
    cv2.line(canvas, (OUT_W//2, OUT_H//2-22), (OUT_W//2, OUT_H//2+22), (0,255,0), 2)

    vw.write(canvas)
    if idx % 200 == 0:
        print(f"  frame {idx}/{len(frames)}  gt={gt:.1f} action={action}")

vw.release()
sz = os.path.getsize(OUT)
print(f"Wrote {OUT} ({sz/(1024*1024):.1f} MB)")
