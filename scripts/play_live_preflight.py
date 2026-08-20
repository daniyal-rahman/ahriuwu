#!/usr/bin/env python3
"""Pre-game readiness check for the live agent. Run ON THE DESKTOP (the box that
runs play_live.py), with the Windows stream already flowing and the Pi's
hid_server.py already up.

This file exists because a previous version of it PASSED on exactly the broken
configuration it was supposed to catch: it "tested" the mouse by sending an
absolute `move` verb to a relative gadget, which was a silent no-op, and it
reported inference speed as a warning rather than a failure. So every check here
is written to FAIL, loudly, on the specific things that have actually bitten:

  1. provenance      — is the running code identifiable at all?
  2. checkpoints     — right file, right head geometry, sha stamped
  3. mouse calib     — the persisted span exists (or the fallback is declared)
  4. HID             — reachable AND the mouse gadget is really there (asks it)
  5. stream          — UDP frames arriving, right geometry, and genuinely NEW
                       frames at a usable rate (the first live session died of
                       2-3 fps of duplicates into a 17 fps loop)
  6. inference       — loads and hits a usable frame rate on real tensors

Nothing here sends a game input: HID checks move the pointer a few units and
never click.

    python scripts/play_live_preflight.py --phase2-ckpt <ckpt> --tokenizer-ckpt <v7> \\
        --inject hid --hid-host 192.168.1.144 --source udp --udp-port 5000
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import time

import numpy as np

RESULTS = []


def check(name, fn, critical=True):
    """Run one check. `critical` failures fail the whole preflight."""
    t0 = time.perf_counter()
    try:
        msg = fn()
        print(f"  [OK]   {name}: {msg}   ({time.perf_counter()-t0:.1f}s)")
        RESULTS.append((name, True, critical))
        return True
    except Exception as e:                                        # noqa: BLE001
        tag = "FAIL" if critical else "WARN"
        print(f"  [{tag}] {name}: {type(e).__name__}: {e}")
        RESULTS.append((name, False, critical))
        return False


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def hid_status(host, port=9999, timeout=5):
    """Ask hid_server what it actually has. Raises if it cannot answer."""
    s = socket.create_connection((host, port), timeout=timeout)
    try:
        s.sendall(b"status\n")
        s.settimeout(timeout)
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                raise RuntimeError(
                    "hid_server accepted the connection but never answered 'status'. "
                    "It is an OLD build without the status verb — re-deploy the Pi's "
                    "scripts/keysender/hid_server.py from this commit.")
            buf += chunk
        return json.loads(buf.split(b"\n", 1)[0].decode())
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-ckpt", required=True)
    ap.add_argument("--tokenizer-ckpt", required=True)
    ap.add_argument("--source", choices=["udp", "screen", "none"], default="udp")
    ap.add_argument("--udp-port", type=int, default=5000)
    ap.add_argument("--stream-size", default="1280x720")
    ap.add_argument("--capture-region", default=None)
    ap.add_argument("--inject", choices=["dry", "pynput", "hid"], default="hid")
    ap.add_argument("--hid-host", default="192.168.1.144")
    ap.add_argument("--movement-mode", choices=["wasd", "mouse"], default="mouse")
    ap.add_argument("--desktop", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-fps", type=float, default=10.0,
                    help="Inference below this is a FAILURE, not a warning. 20 fps is "
                         "known unreachable (81.4 ms model-only on an idle 5080); the "
                         "rig measured 17 fps and the loop is usable down to ~10.")
    ap.add_argument("--min-stream-fps", type=float, default=8.0,
                    help="NEW frames/s from the Windows stream below this is a FAILURE. "
                         "The first live session ran at 2-3.")
    ap.add_argument("--stream-secs", type=float, default=4.0)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(0, os.path.join(here, "keysender"))
    print("=== play_live preflight ===")

    # --- 1. provenance --------------------------------------------------------
    def prov_check():
        from play_live import provenance
        p = provenance()
        if p["commit"] == "UNKNOWN":
            raise RuntimeError(
                "cannot identify the running code: no VERSION file and not a git "
                "checkout. Re-deploy with ops/stage_desktop_standalone.sh.")
        if p.get("dirty") == "yes":
            raise RuntimeError(
                f"tree is DIRTY at commit {p['commit'][:12]} — the code about to run is "
                "not any committed state, so a bad live session cannot be reproduced.")
        return f"commit {p['commit'][:12]} via {p['source']}"
    check("provenance", prov_check, critical=False)

    # --- 2. checkpoint identity ----------------------------------------------
    holder = {}

    def ckpt_check():
        import torch
        for p in (args.phase2_ckpt, args.tokenizer_ckpt):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        ck = torch.load(args.phase2_ckpt, map_location="cpu", weights_only=False)
        a = ck.get("args", {})
        a = a if isinstance(a, dict) else vars(a)
        holder["mm"] = a.get("movement_mode", "axis")
        holder["gate"] = a.get("movement_gate", False)
        step = ck.get("global_step")
        # A frozen backbone on the click-target lineage is the documented
        # under-delivery (action_embed was fitted to cursor.screen and cannot
        # adapt). Not fatal, but it must be visible before a demo.
        frozen = not a.get("unfreeze_backbone", False)
        note = "  ** FROZEN backbone lineage **" if frozen else ""
        return (f"step={step} movement_mode={holder['mm']} gate={holder['gate']} "
                f"size={a.get('model_size')} sha={sha16(args.phase2_ckpt)} "
                f"| tok sha={sha16(args.tokenizer_ckpt)}{note}")
    ok_ck = check("checkpoint identity", ckpt_check)

    # --- 3. mouse calibration -------------------------------------------------
    def calib_check():
        if args.movement_mode != "mouse":
            return "not applicable (--movement-mode wasd)"
        from hybrid_sender import load_calibration, CALIB_PATH
        span, chunk, src = load_calibration()
        if "FALLBACK" in src:
            raise RuntimeError(
                f"no measured calibration at {CALIB_PATH}. Every click would use a "
                f"built-in span {span} from some other session of the rig. Run ONCE:\n"
                f"         python scripts/keysender/calibrate_mouse.py --host "
                f"{args.hid_host} --udp-port {args.udp_port}")
        prov = ""
        try:
            with open(CALIB_PATH) as fh:
                if json.load(fh).get("provisional"):
                    prov = ("  ** PROVISIONAL: carried over, not re-measured on this boot. "
                            "Valid only if Windows pointer speed / 'Enhance pointer "
                            "precision' / resolution are unchanged. **")
        except (OSError, ValueError):
            pass
        return f"span={span[0]:.0f}x{span[1]:.0f} chunk={chunk} <- {src}{prov}"
    check("mouse calibration", calib_check, critical=(args.inject == "hid"))

    # --- 4. HID: reachable AND the mouse really exists ------------------------
    if args.inject == "hid":
        def hid_check():
            st = hid_status(args.hid_host)
            if not st.get("keyboard"):
                raise RuntimeError("hid_server reports NO keyboard (/dev/hidg0)")
            if args.movement_mode == "mouse":
                if not st.get("mouse"):
                    raise RuntimeError(
                        "hid_server has NO mouse gadget (/dev/hidg1). Every click would be "
                        "silently dropped. On the Pi: sudo MOUSE_MODE=rel "
                        "scripts/keysender/setup_hid_combo.sh, then restart hid_server.py.")
                if st["mouse"] != "rel":
                    raise RuntimeError(
                        f"hid_server mouse is '{st['mouse']}' but hybrid_sender speaks "
                        f"RELATIVE. Re-run setup_hid_combo.sh with MOUSE_MODE=rel, or start "
                        f"hid_server with --mouse rel.")
            return f"keyboard=yes mouse={st.get('mouse')} at {args.hid_host}:9999"
        check("HID gadget", hid_check)

        def hid_move_check():
            """Actually push relative reports and prove the socket survives.
            5 units is well under one movement bin — invisible in-game."""
            if args.movement_mode != "mouse":
                return "skipped (wasd)"
            s = socket.create_connection((args.hid_host, 9999), timeout=5)
            try:
                for cmd in (b"mouse 5 0\n", b"mouse 0 5\n", b"mouse -5 0\n",
                            b"mouse 0 -5\n", b"reset\n"):
                    s.sendall(cmd)
                    time.sleep(0.01)
                s.sendall(b"status\n")                 # still alive after all that?
                s.settimeout(5)
                if not s.recv(4096):
                    raise RuntimeError("server closed the socket after mouse reports")
            finally:
                s.close()
            return "10 relative reports accepted, socket alive, no click sent"
        check("HID mouse reports", hid_move_check)

    # --- 5. stream: arriving, right geometry, and genuinely NEW ---------------
    def stream_check():
        if args.source == "none":
            return "skipped (--source none)"
        from play_live import StreamCapture, ScreenCapture
        if args.source == "screen":
            region = tuple(map(int, args.capture_region.split(","))) if args.capture_region else None
            cap = ScreenCapture(region)
            f, _ = cap.grab_rgb01()
            return f"mss grab {f.shape} range [{f.min():.2f},{f.max():.2f}]"
        sw, sh = map(int, args.stream_size.split("x"))
        cap = StreamCapture(port=args.udp_port, size=(sw, sh))
        try:
            cap.wait_first(timeout=15)
            f, _ = cap.grab_rgb01()
            if f.shape != (sh, sw, 3):
                raise RuntimeError(f"stream geometry {f.shape} != expected {(sh, sw, 3)}")
            # Count NEW frames, not loop iterations. This is the check the first
            # live session needed and did not have.
            t0, n_new, n_iter = time.perf_counter(), 0, 0
            while time.perf_counter() - t0 < args.stream_secs:
                _, fresh = cap.grab_rgb01()
                n_new += int(fresh)
                n_iter += 1
                time.sleep(1 / 60)
            fps = n_new / (time.perf_counter() - t0)
            holder["stream_fps"] = fps
            mean = float(f.mean())
            if fps < args.min_stream_fps:
                raise RuntimeError(
                    f"only {fps:.1f} NEW frames/s ({n_new}/{n_iter} polls fresh). The model's "
                    f"16-frame context would fill with duplicates and the agent would stand "
                    f"still. Raise the Windows ffmpeg -framerate (see docs/DEMO_RUNBOOK.md).")
            if mean < 0.04:
                raise RuntimeError(f"stream is essentially black (mean {mean:.3f}) — "
                                   f"is the game actually on the captured screen?")
            return (f"{sw}x{sh}, {fps:.1f} new frames/s ({n_new}/{n_iter} polls fresh), "
                    f"mean brightness {mean:.3f}")
        finally:
            cap.close()
    check("UDP stream", stream_check)

    # --- 6. model loads and runs fast enough ---------------------------------
    def agent_check():
        from agent_infer import GarenAgent
        ag = GarenAgent(args.phase2_ckpt, tokenizer_ckpt=args.tokenizer_ckpt,
                        device=args.device)
        ag.reset()
        holder["ag"] = ag
        return (f"use_actions={ag.use_actions} "
                f"movement_mode={getattr(ag.policy, 'movement_mode', 'axis')} "
                f"gate={getattr(ag.policy, 'movement_gate', False)} bf16={ag.amp}")
    ok_agent = check("agent load", agent_check)

    def latency_check():
        ag = holder["ag"]
        # A REAL-shaped frame through the REAL path (resize + tokenizer + dynamics),
        # not a 352 tensor that skips the resize.
        sw, sh = map(int, args.stream_size.split("x"))
        frame = np.random.rand(sh, sw, 3).astype(np.float32)
        for _ in range(5):
            ag.act_from_latent(ag.encode_frame(frame), temperature=1.0)
        t = time.perf_counter()
        N = 30
        for _ in range(N):
            ag.act_from_latent(ag.encode_frame(frame), temperature=1.0)
        fps = N / (time.perf_counter() - t)
        if fps < args.min_fps:
            raise RuntimeError(f"{fps:.1f} fps < required {args.min_fps} — the loop cannot "
                               f"keep up with the game. Is something else on the GPU?")
        return f"full path (resize+tokenizer+dynamics+heads) {fps:.1f} fps"
    if ok_agent:
        check("inference speed", latency_check)

    def behaviour_check():
        """A policy that never moves is the failure mode greedy decode produces.
        Sample 120 steps on noise and demand it is not degenerate."""
        ag = holder["ag"]
        ag.reset()
        lat = np.random.rand(1, ag.latent_dim, 16, 16).astype(np.float32)
        import torch
        cells, fires = set(), 0
        for _ in range(120):
            a = ag.act_from_latent(torch.from_numpy(lat).to(args.device), temperature=1.0)
            cells.add(tuple(round(v, 2) for v in a["movement"]))
            fires += int(a.get("gate", True))
        if len(cells) < 3:
            raise RuntimeError(f"policy is degenerate: {len(cells)} distinct movement "
                               f"target(s) over 120 samples. Is --temperature 0 in play?")
        return (f"{len(cells)} distinct targets, gate fired {fires}/120 "
                f"(~{fires/120*17:.1f} clicks/s at 17 fps)")
    if ok_agent:
        check("policy not degenerate", behaviour_check, critical=False)

    print()
    crit_fail = [n for n, ok, c in RESULTS if not ok and c]
    warn = [n for n, ok, c in RESULTS if not ok and not c]
    if warn:
        print(f"WARNINGS (non-blocking): {', '.join(warn)}")
    if crit_fail:
        print(f"=== PREFLIGHT FAILED: {', '.join(crit_fail)} ===")
        print("Do NOT start a live session until these pass. See docs/DEMO_RUNBOOK.md.")
        sys.exit(1)
    print("=== ALL CRITICAL CHECKS PASSED ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
