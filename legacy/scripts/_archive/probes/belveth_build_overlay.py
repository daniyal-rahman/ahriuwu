"""Post-process: read PNG frames + mem/cam samples, build overlay video.
Runs on Windows. Downscales 4K → 1080p for manageable output.
"""
import os, sys, glob, json, bisect, math
import cv2, numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

import glob as _g
# recurse to find whatever subdir League auto-created
_candidates = sorted(_g.glob(r"C:\tmp\belveth_frames\**\*.png", recursive=True))
if _candidates:
    PNG_DIR = os.path.dirname(_candidates[0])
else:
    PNG_DIR = r"C:\tmp\belveth_frames"
SAMPLES   = r"C:\tmp\belveth_run2\samples.json"
OUT_PATH  = r"C:\tmp\belveth_run2\overlay.mp4"
SPEED     = 2.0
REC_FPS   = 40
START_GT  = 0.5
OUT_W     = 1920  # downscale from 3840
OUT_H     = 1080

def main():
    pngs = sorted(glob.glob(os.path.join(PNG_DIR, "*.png")))
    if not pngs:
        print(f"No PNGs in {PNG_DIR}"); return
    samples = json.load(open(SAMPLES))
    sample_gts = [s["gt"] for s in samples]
    print(f"Frames: {len(pngs)}  Samples: {len(samples)}  gt range: {sample_gts[0]:.2f}..{sample_gts[-1]:.2f}")

    first = cv2.imread(pngs[0])
    src_h, src_w = first.shape[:2]
    print(f"Source: {src_w}x{src_h} → output {OUT_W}x{OUT_H}")

    play_fps = REC_FPS / SPEED   # 20 game-fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(OUT_PATH, fourcc, play_fps, (OUT_W, OUT_H))

    MAP_VIEW_WIDTH = 12000.0
    TILT = math.cos(math.radians(56.0))
    scale = OUT_W / MAP_VIEW_WIDTH

    for i, png in enumerate(pngs):
        f = cv2.imread(png)
        if f is None: continue
        if (f.shape[1], f.shape[0]) != (OUT_W, OUT_H):
            f = cv2.resize(f, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        gt = START_GT + i / play_fps
        idx = bisect.bisect_left(sample_gts, gt)
        if 0 <= idx < len(samples):
            s = samples[idx]
            bv, cam = s["bv"], s["cam"]
            sx = int((bv[0] - cam[0]) * scale + OUT_W / 2)
            sy = int((cam[2] - bv[2]) * scale * TILT + OUT_H / 2)
            cv2.circle(f, (sx, sy), 24, (0, 255, 0), 3)
            cv2.line(f, (sx-20, sy), (sx+20, sy), (0,255,0), 3)
            cv2.line(f, (sx, sy-20), (sx, sy+20), (0,255,0), 3)
            lines = [
                f"gt={gt:6.2f}s  frame={i}",
                f"BelVeth world: ({bv[0]:7.1f}, {bv[1]:6.1f}, {bv[2]:7.1f})",
                f"Camera  world: ({cam[0]:7.1f}, {cam[1]:6.1f}, {cam[2]:7.1f})",
                f"delta bv-cam:  ({bv[0]-cam[0]:+6.1f}, {bv[2]-cam[2]:+6.1f})",
            ]
            for j, line in enumerate(lines):
                cv2.putText(f, line, (12, 32+j*32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,0), 5, cv2.LINE_AA)
                cv2.putText(f, line, (12, 32+j*32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100,255,255), 2, cv2.LINE_AA)
        vw.write(f)
        if i % 100 == 0:
            print(f"  {i}/{len(pngs)}  gt={gt:.1f}")
    vw.release()
    print(f"Saved -> {OUT_PATH}")

if __name__ == "__main__":
    main()
