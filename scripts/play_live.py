#!/usr/bin/env python3
"""Live Garen play (Windows at-home entrypoint).

Thin I/O wrapper around the tested inference core (scripts/agent_infer.GarenAgent):

    mss screen capture -> GarenAgent.act(frame) -> pynput key/mouse injection

The MODEL side is identical to the offline test in agent_infer.py, so anything
that works there works here; only capture + injection are new (and Windows-only).
ALWAYS start with --dry-run to eyeball the action stream before sending inputs.

    # dry run — prints actions, sends nothing
    python scripts/play_live.py --phase2-ckpt <ckpt> --tokenizer-ckpt <v7> --dry-run

    # live (practice tool / custom game only)
    python scripts/play_live.py --phase2-ckpt <ckpt> --tokenizer-ckpt <v7> \
        --capture-region 0,0,1920,1080 --target-fps 20

Requires (Windows): pip install mss pynput opencv-python. Default LoL keybinds
assumed (Q/W/E/R abilities, D=Flash, F=Ignite, B=Recall, item slot for Stride,
right-click = move / AA). Remap with the flags below.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch

from ahriuwu.constants import ABILITY_KEYS
from agent_infer import GarenAgent  # scripts/ on sys.path[0]


# LoL default binds; ABILITY_KEYS = [Q,W,E,R,Flash,Ignite,AA,Recall,Stride]
DEFAULT_KEYS = {"Q": "q", "W": "w", "E": "e", "R": "r", "Flash": "d",
                "Ignite": "f", "Recall": "b", "Stride": "3"}  # AA handled via right-click


class ScreenCapture:
    def __init__(self, region):
        import mss
        self.sct = mss.mss()
        self.mon = ({"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
                    if region else self.sct.monitors[1])
        print(f"capture region: {self.mon}")

    def grab_rgb01(self):
        img = np.array(self.sct.grab(self.mon))                 # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB).astype(np.float32) / 255.0


class InputController:
    def __init__(self, region, dry_run, keys):
        self.dry, self.keys = dry_run, keys
        self.left, self.top = region[0], region[1]
        self.w, self.h = region[2], region[3]
        if not dry_run:
            from pynput.keyboard import Controller as KB
            from pynput.mouse import Controller as MC, Button
            self.kb, self.mouse, self.Button = KB(), MC(), Button

    def send(self, action):
        mx = self.left + int(action["movement"][0] * self.w)
        my = self.top + int(action["movement"][1] * self.h)
        pressed = [k for k, v in action["abilities"].items() if v]
        if self.dry:
            return mx, my, pressed
        # movement / AA: right-click at the cursor target
        self.mouse.position = (mx, my)
        self.mouse.click(self.Button.right)
        for k in pressed:
            if k == "AA":
                continue  # AA = the right-click above
            key = self.keys.get(k)
            if key:
                self.kb.press(key); self.kb.release(key)
        return mx, my, pressed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-ckpt", required=True)
    ap.add_argument("--tokenizer-ckpt", required=True)
    ap.add_argument("--capture-region", default=None, help="x,y,w,h (default: primary monitor)")
    ap.add_argument("--context", type=int, default=32)
    ap.add_argument("--target-fps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.0, help="0=greedy (steadiest)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    region = tuple(map(int, args.capture_region.split(","))) if args.capture_region else None
    cap = ScreenCapture(region)
    if region is None:
        region = (cap.mon["left"], cap.mon["top"], cap.mon["width"], cap.mon["height"])
    ctrl = InputController(region, args.dry_run, DEFAULT_KEYS)
    agent = GarenAgent(args.phase2_ckpt, tokenizer_ckpt=args.tokenizer_ckpt,
                       context=args.context, device=args.device)
    agent.reset()

    dt_target = 1.0 / args.target_fps
    hist = deque(maxlen=60)
    n = 0
    print(f"\nRunning ({'DRY-RUN' if args.dry_run else 'LIVE — sending inputs'}). Ctrl+C to stop.\n")
    try:
        while True:
            t0 = time.perf_counter()
            frame = cap.grab_rgb01()
            lat = agent.encode_frame(frame)
            action = agent.act_from_latent(lat, temperature=args.temperature)
            mx, my, pressed = ctrl.send(action)
            hist.append((time.perf_counter() - t0) * 1000)
            n += 1
            if n % args.target_fps == 0:
                fps = 1000 / (sum(hist) / len(hist))
                print(f"frame {n:6d} | move=({action['movement'][0]:.2f},{action['movement'][1]:.2f})"
                      f"->({mx},{my}) keys={pressed or '-'} rew={action['reward_pred']:+.2f} | {fps:4.1f} fps")
            slack = dt_target - (time.perf_counter() - t0)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        print(f"\nstopped after {n} frames.")


if __name__ == "__main__":
    main()
