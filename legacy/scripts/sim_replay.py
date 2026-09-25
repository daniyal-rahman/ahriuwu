#!/usr/bin/env python3
"""Offline replay-sim for the live agent (AV-style: validate on recorded logs
BEFORE ever going live again).

Feeds RECORDED frames through the exact live inference path (same encode, same
policy, same gate->click decode) and scores behavior against in-domain
reference bands measured from the replay corpus the policy cloned. Any change
to the inference pipeline must PASS here before a live session.

Sources:
  --session <dir>   a recorded live session (recordings/session_*/), i.e. the
                    frames the model ACTUALLY saw in a real game
  --replay <match>  an in-domain replay game (the "known-good" control)

Color transform (--fix): the recorded live frames were decoded WITHOUT
limited->full range expansion (the bug), so they are ~35% dark. `--fix range`
applies the exact expansion ffmpeg's `scale=in_range=tv:out_range=pc` performs,
simulating the fixed receive path on the OLD recording — no new capture needed.

METRICS + PASS BANDS (from 45 min of Masters+ replays / in-domain agent runs):
  move_diversity  unique movement targets / frames   in-domain ~0.16   FAIL <0.05
  click_rate      gate firings per second            human 2-5/s      FAIL <1 or >10
  cast_rate       frames with any ability            human 1.36%      WARN outside 0.4-3%
  brightness      mean pixel                         training 0.203   WARN <0.17

    PYTHONPATH=src python scripts/sim_replay.py --session <dir> --fix range
    PYTHONPATH=src python scripts/sim_replay.py --replay NA1_5549981347   # control
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

REF = {"move_diversity": 0.16, "click_rate": 3.5, "cast_rate": 0.0136, "brightness": 0.203}
FPS = 20


def expand_range(f01):
    """Exact TV(16-235) -> PC(0-255) expansion, per channel, on [0,1] floats.
    This is what `scale=in_range=tv:out_range=pc` does and what the broken
    receive path omitted."""
    return np.clip((f01 * 255.0 - 16.0) * (255.0 / 219.0) / 255.0, 0.0, 1.0)


def load_session(d, n):
    mp4 = os.path.join(d, "model_view_352.mp4")
    cap = cv2.VideoCapture(mp4)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(np.linspace(tot // 5, tot - 1, min(n, tot)).astype(int).tolist())
    out, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
        i += 1
    cap.release()
    return out


def load_replay(match, n, root="/mnt/nfs/datasets/lol_replays_16_9_772"):
    fs = sorted(glob.glob(f"{root}/{match}/frames/*.png"))
    if not fs:
        fs = sorted(glob.glob(f"/srv/nfs/datasets/lol_replays_16_9_772/{match}/frames/*.png"))
    idx = np.linspace(2000, len(fs) - 1, n).astype(int)
    out = []
    for i in idx:
        im = cv2.imread(fs[int(i)])
        r = cv2.resize(im, (352, 352), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None, help="recordings/session_* dir")
    ap.add_argument("--replay", default=None, help="in-domain replay match id (control)")
    ap.add_argument("--fix", choices=["none", "range", "gamma", "matchmean"], default="none",
                    help="color transform to simulate on the recorded frames. 'matchmean' "
                         "rescales brightness to --target-mean (the training statistic) — the "
                         "principled correction; 'range' (TV->PC) is kept only to document that "
                         "it CRUSHES SHADOWS and made things worse.")
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--target-mean", type=float, default=0.203, help="training brightness")
    ap.add_argument("--darken-to", type=float, default=None,
                    help="ABLATION: rescale frames DOWN to this mean (e.g. 0.136 = live) to test "
                         "whether darkness alone causes the degradation on in-domain frames.")
    ap.add_argument("--hud", choices=["none", "mask", "add"], default="none",
                    help="HUD ablation. 'mask' blacks the live HUD regions; 'add' PAINTS those "
                         "regions black on in-domain replay frames (which were captured HUD-off) "
                         "— the decisive reverse test for whether HUD occlusion causes the drop.")
    ap.add_argument("--gate-bias", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--phase2-ckpt", default="/mnt/storage/ahriuwu-live/checkpoints/phase2_bc.pt")
    ap.add_argument("--tokenizer-ckpt", default="/mnt/storage/ahriuwu-live/checkpoints/tokenizer_v7.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not (args.session or args.replay):
        raise SystemExit("need --session or --replay")
    frames = (load_session(args.session, args.frames) if args.session
              else load_replay(args.replay, args.frames))
    if args.fix == "range":
        frames = [expand_range(f) for f in frames]
    elif args.fix == "gamma":
        frames = [np.clip(f, 0, 1) ** args.gamma for f in frames]
    elif args.fix == "matchmean":
        cur = float(np.mean([f.mean() for f in frames]))
        g = np.log(max(args.target_mean, 1e-6)) / np.log(max(cur, 1e-6))  # gamma s.t. mean matches
        frames = [np.clip(np.clip(f, 0, 1) ** g, 0, 1) for f in frames]
        print(f"[matchmean] mean {cur:.3f} -> target {args.target_mean:.3f} via gamma {g:.3f}")
    if args.hud in ("mask", "add"):
        # live HUD layout in 352-space: left ability column, bottom bar, minimap, top-right
        hm = np.ones((352, 352, 1), np.float32)
        for y0, y1, x0, x1 in [(0, 352, 0, 30), (325, 352, 0, 352),
                               (240, 352, 275, 352), (0, 22, 300, 352)]:
            hm[y0:y1, x0:x1] = 0.0
        frames = [f * hm for f in frames]               # same op either way; differs by source
        print(f"[hud {args.hud}] blacked {(1 - hm).mean():.1%} of the frame")
    if args.darken_to is not None:                      # ablation, applied last
        cur = float(np.mean([f.mean() for f in frames]))
        g = np.log(max(args.darken_to, 1e-6)) / np.log(max(cur, 1e-6))
        frames = [np.clip(np.clip(f, 0, 1) ** g, 0, 1) for f in frames]
        print(f"[darken] mean {cur:.3f} -> {args.darken_to:.3f} via gamma {g:.3f}")

    from agent_infer import GarenAgent
    ag = GarenAgent(args.phase2_ckpt, tokenizer_ckpt=args.tokenizer_ckpt, device=args.device)
    ag.reset()

    moves, fires, casts_any, per_key = [], [], 0, {}
    supports_bias = "gate_bias" in ag.act_from_latent.__code__.co_varnames
    for f in frames:
        kw = {"temperature": args.temperature}
        if supports_bias:
            kw["gate_bias"] = args.gate_bias
        a = ag.act_from_latent(ag.encode_frame(f), **kw)
        moves.append(tuple(round(float(x), 3) for x in a["movement"]))
        fires.append(bool(a.get("gate", True)))
        on = [k for k, v in a["abilities"].items() if v]
        casts_any += bool(on)
        for k in on:
            per_key[k] = per_key.get(k, 0) + 1

    n = len(frames)
    m = {
        "brightness": float(np.mean([f.mean() for f in frames])),
        "move_diversity": len(set(moves)) / n,
        "click_rate": float(np.mean(fires)) * FPS,
        "cast_rate": casts_any / n,
    }
    src = args.session or f"replay:{args.replay}"
    print(f"\n=== SIM {src}  fix={args.fix}  gate_bias={args.gate_bias}  n={n} ===")

    def line(k, val, fmt, band, ok):
        print(f"  {k:15s} {format(val, fmt):>9s}   ref {band:16s} {'PASS' if ok else '** FAIL **'}")

    ok_b = m["brightness"] > 0.17
    ok_d = m["move_diversity"] >= 0.05
    ok_c = 1.0 <= m["click_rate"] <= 10.0
    ok_a = 0.004 <= m["cast_rate"] <= 0.03
    line("brightness", m["brightness"], ".3f", f"{REF['brightness']:.3f} (train)", ok_b)
    line("move_diversity", m["move_diversity"], ".3f", f"~{REF['move_diversity']:.2f} in-domain", ok_d)
    line("click_rate/s", m["click_rate"], ".1f", "2-5 human", ok_c)
    line("cast_rate", m["cast_rate"], ".3%", f"{REF['cast_rate']:.2%} human", ok_a)
    if per_key:
        print("  casts by key:", {k: round(v / n, 4) for k, v in sorted(per_key.items(), key=lambda x: -x[1])})
    all_ok = ok_b and ok_d and ok_c
    print(f"\n  VERDICT: {'READY for a live session' if all_ok else 'NOT READY — fix before going live'}"
          f"{'' if ok_a else '  (cast rate off-band: policy undertrained, not a blocker)'}")
    print(json.dumps(m))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
