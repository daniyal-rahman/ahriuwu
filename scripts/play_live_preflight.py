#!/usr/bin/env python3
"""Pre-game readiness check for the live agent. Run on the WINDOWS box before a
practice game. Verifies each stage independently so a failure is localized, and
NEVER sends a game input (HID moves go to a safe desktop corner).

    python scripts/play_live_preflight.py --phase2-ckpt <ckpt> --tokenizer-ckpt <v7> \
        [--capture-region x,y,w,h] [--inject hid --hid-host <ip> --desktop 1920x1080]
"""
import argparse
import sys
import time

import numpy as np


def check(name, fn):
    try:
        msg = fn()
        print(f"  [OK]   {name}: {msg}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-ckpt", required=True)
    ap.add_argument("--tokenizer-ckpt", required=True)
    ap.add_argument("--capture-region", default=None)
    ap.add_argument("--inject", choices=["dry", "pynput", "hid"], default="dry")
    ap.add_argument("--hid-host", default="127.0.0.1")
    ap.add_argument("--desktop", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sys.path.insert(0, "scripts")
    ok = True
    print("=== play_live preflight ===")

    def cap_check():
        from play_live import ScreenCapture
        region = tuple(map(int, args.capture_region.split(","))) if args.capture_region else None
        cap = ScreenCapture(region)
        t = time.perf_counter()
        f, _fresh = cap.grab_rgb01()
        ms = (time.perf_counter() - t) * 1e3
        assert f.ndim == 3 and f.shape[2] == 3, f"bad frame shape {f.shape}"
        return f"grabbed {f.shape} in {ms:.0f}ms, range [{f.min():.2f},{f.max():.2f}]"
    ok &= check("screen capture", cap_check)

    holder = {}

    def agent_check():
        from agent_infer import GarenAgent
        ag = GarenAgent(args.phase2_ckpt, tokenizer_ckpt=args.tokenizer_ckpt, device=args.device)
        ag.reset()
        holder["ag"] = ag
        gated = getattr(ag.policy, "movement_gate", False)
        return f"loaded (use_actions={ag.use_actions}, movement_gate={gated}, bf16={ag.amp})"
    ok &= check("agent load", agent_check)

    def latency_check():
        ag = holder["ag"]
        frame = np.random.rand(352, 352, 3).astype(np.float32)
        for _ in range(5):
            ag.act_from_latent(ag.encode_frame(frame), temperature=1.0)
        t = time.perf_counter()
        N = 40
        for _ in range(N):
            ag.act_from_latent(ag.encode_frame(frame), temperature=1.0)
        fps = N / (time.perf_counter() - t)
        verdict = "OK for 20fps" if fps >= 20 else "BELOW 20fps — needs optimization"
        return f"encode+act {fps:.1f} fps ({verdict})"
    ok &= check("inference speed", latency_check)

    if args.inject == "hid":
        def hid_check():
            import socket
            import json
            s = socket.create_connection((args.hid_host, 9999), timeout=5)
            dw, dh = (map(int, args.desktop.split("x")) if args.desktop else (1920, 1080))
            # move to a harmless corner + a no-op key tap; NO click into the game
            s.sendall((json.dumps({"t": "move", "x": 100, "y": 100}) + "\n").encode())
            s.sendall((json.dumps({"t": "reset"}) + "\n").encode())
            s.close()
            return f"hid_server reachable at {args.hid_host}:9999 (moved pointer to corner, no click)"
        ok &= check("HID injection", hid_check)

    def state_api_check():
        from ahriuwu.live import read_own_state
        s = read_own_state()
        return f"live client state: {s}" if s else "no game running (expected pre-game; will work in-game)"
    ok &= check("live client API", state_api_check)

    print("=== " + ("ALL CRITICAL CHECKS PASSED" if ok else "SOME CHECKS FAILED — see above") + " ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
