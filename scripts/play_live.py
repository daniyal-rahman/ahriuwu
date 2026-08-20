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
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch

from ahriuwu.constants import ABILITY_KEYS
from agent_infer import GarenAgent  # scripts/ on sys.path[0]


# LoL default binds; ABILITY_KEYS = [Q,W,E,R,Flash,Ignite,AA,Recall,Stride].
# Movement is WASD (keyboard-only rig): abilities MUST NOT collide with w/a/s/d,
# so Flash/Ignite move off d/f and abilities use their standard letters (Q/W/E/R
# unchanged — LoL binds those to the champion, WASD to movement, no conflict in
# WASD-movement mode). AA -> attack key (default None = rely on auto-attack /
# attack-move-on-move; set --attack-key to bind it).
# Garen W is on 'w' by LoL default -> collides with move-up. Abilities relocated
# OFF w/a/s/d (must match in-game binds). Q/E/R are free of WASD; only W moves.
DEFAULT_KEYS = {"Q": "q", "W": "v", "E": "e", "R": "r", "Flash": "g",
                "Ignite": "f", "Recall": "b", "Stride": "3"}
# 8-way WASD decode. Screen: +x right, +y down; "up on screen" = W.
_WASD_DIRS = [
    (0.0, "d"), (45.0, "wd"), (90.0, "w"), (135.0, "wa"),
    (180.0, "a"), (225.0, "sa"), (270.0, "s"), (315.0, "sd"),
]


def provenance():
    """Where this code came from -> dict, printed at startup and recorded in
    every session's meta.json.

    The deployed rig was for ten days an unmanaged FORK of this repo, so every
    fix made in git was absent from the thing that actually played and nothing
    in the logs said so. Resolve identity from the VERSION file written by
    ops/stage_desktop_standalone.sh, or from git when running out of a checkout.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vf = os.path.join(root, "VERSION")
    if os.path.exists(vf):
        d = {}
        with open(vf) as fh:
            for line in fh:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    d[k] = v
        d["source"] = "VERSION (staged)"
        return d
    try:
        sha = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "-C", root, "status", "--porcelain",
                                         "--", "src", "scripts"],
                                        stderr=subprocess.DEVNULL, text=True).strip()
        return {"commit": sha, "dirty": "yes" if dirty else "no",
                "source": "git checkout at " + root}
    except (subprocess.CalledProcessError, OSError):
        return {"commit": "UNKNOWN", "source": "UNRESOLVABLE — not staged, not a git checkout"}


class ScreenCapture:
    def __init__(self, region):
        import mss
        self.sct = mss.mss()
        self.mon = ({"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
                    if region else self.sct.monitors[1])
        print(f"capture region: {self.mon}")

    def grab_rgb01(self):
        img = np.array(self.sct.grab(self.mon))                 # BGRA
        # (frame, is_fresh) — mss always grabs live pixels, so always fresh.
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB).astype(np.float32) / 255.0, True


class StreamCapture:
    """Receive the Windows game stream (gdigrab -> x264 zerolatency -> mpegts/UDP)
    and hand the model the FRESHEST decoded frame.

    An ffmpeg subprocess decodes the UDP stream to raw rgb24; a reader thread
    drains the pipe continuously and keeps only the newest frame, so the 20fps
    control loop never acts on a stale/backlogged frame (realtime > completeness).
    """

    def __init__(self, port=5000, size=(1280, 720), expand_range=False, gamma=1.0):
        self.w, self.h = size
        self.frame_bytes = self.w * self.h * 3
        # NOTE (2026-08-10): range expansion defaults OFF. It was added on a
        # wrong diagnosis ("the stream is TV-range => dark") and REFUTED by the
        # offline sim (scripts/sim_replay.py): expansion CRUSHES SHADOWS on this
        # source (mean 0.136 -> 0.090) and neither brightness nor HUD masking
        # changed the policy's behavior. The real live failure was STALE FRAMES
        # (58% of consecutive frames byte-identical) — see the staleness guard
        # below. Keep the flag only for A/B experiments.
        vf = "scale=in_range=tv:out_range=pc" if expand_range else "null"
        self.gamma_lut = None
        if gamma != 1.0:
            self.gamma_lut = (np.clip((np.arange(256) / 255) ** gamma, 0, 1) * 255).astype(np.uint8)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-fflags", "nobuffer", "-flags", "low_delay",
            "-probesize", "32", "-analyzeduration", "0",
            "-i", f"udp://@:{port}?fifo_size=1000000&overrun_nonfatal=1",
            "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, bufsize=self.frame_bytes)
        self.latest = None
        self.n_recv = 0
        self._last_served_n = -1
        self.stale_serves = 0
        self.lock = threading.Lock()
        self.run = True
        print(f"stream: listening udp :{port} for {self.w}x{self.h} — waiting for first frame...")
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        buf = b""
        while self.run:
            chunk = self.proc.stdout.read(self.frame_bytes - len(buf))
            if not chunk:
                break                                            # ffmpeg died / stream ended
            buf += chunk
            if len(buf) < self.frame_bytes:
                continue
            fr = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            if self.gamma_lut is not None:
                fr = self.gamma_lut[fr]
            with self.lock:
                self.latest = fr
                self.n_recv += 1
            buf = b""

    def grab_rgb01(self):
        """Freshest frame + whether it is NEW since the last call.

        STALENESS GUARD: the first live session silently fed the model a frozen
        world — 58% of consecutive frames were byte-identical because the Windows
        stream delivered new content only ~2-3x/s while this loop ran at 17fps,
        and `latest` simply repeats. A world model shown no change predicts no
        change, so the agent stood still. Serving stale frames must never be
        invisible again: `stale_serves` is counted and surfaced in the HUD line.
        """
        with self.lock:
            fr = self.latest
            n = self.n_recv
        if fr is None:
            return None, False
        fresh = n != self._last_served_n
        if fresh:
            self._last_served_n = n
        else:
            self.stale_serves += 1
        return fr.astype(np.float32) / 255.0, fresh

    def wait_first(self, timeout=30.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest is not None:
                return True
            if self.proc.poll() is not None:
                raise RuntimeError("ffmpeg exited before any frame (no stream? wrong port/size?)")
            time.sleep(0.1)
        raise TimeoutError(f"no stream frame within {timeout}s on the UDP port")

    def close(self):
        self.run = False
        self.proc.terminate()


class InputController:
    """action dict -> real inputs. Two backends:

    - ``pynput`` (backend='pynput'): synthetic events from *inside* Windows.
      Simple, but Vanguard (LoL's kernel anti-cheat) rejects synthetic input in
      real games — usable only for desktop/notepad plumbing tests.
    - ``hid`` (backend='hid'): JSON to the external USB-HID gadget's hid_server
      (scripts/keysender/hid_server.py). Indistinguishable from a real kb+mouse
      to the host — this is the path for an actual game. Absolute mouse coords
      are 0..32767 across the FULL desktop, so we map screen fraction -> desktop
      logical, independent of the capture region's offset.
    """

    def __init__(self, region, backend, keys, desktop, hid_host="127.0.0.1",
                 deadzone=0.06, attack_key=None, movement_mode="wasd",
                 click_min_interval=0.12):
        self.backend, self.keys = backend, keys
        self.left, self.top = region[0], region[1]
        self.w, self.h = region[2], region[3]
        self.dw, self.dh = desktop
        self.deadzone = deadzone           # |target - center| below this = stand still
        self.attack_key = attack_key       # AA bind in WASD mode (None = no-op)
        self.held = set()                  # currently-held WASD movement keys
        # "wasd" = keyboard-only rig; "mouse" = right-click-to-move, the real
        # League primitive the policy was actually trained on (click targets).
        self.movement_mode = movement_mode
        # Floor on time between clicks. The gate normally paces these (~2/s), but
        # an ungated checkpoint reports gate=True every frame, which at 20 fps
        # would be 20 clicks/s -- nothing like a human, and it cancels its own
        # orders before the champion moves.
        self.click_min_interval = click_min_interval
        self._last_click_t = 0.0
        self.clicks_sent = 0
        self.clicks_suppressed = 0
        self.mouse = None
        if backend == "pynput":
            from pynput.keyboard import Controller as KB
            self.kb = KB()
            if movement_mode == "mouse":
                from pynput.mouse import Controller as MC
                self.mouse = MC()
        elif backend == "hid":
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "keysender"))
            from hybrid_sender import HybridKeyboard
            # mouse=True spawns the corner-relative pointer loop. Without it,
            # move_click() sets a target that nothing ever consumes -- the exact
            # silent no-op that let a live session "run" while never aiming.
            self.kb = HybridKeyboard(host=hid_host, mouse=(movement_mode == "mouse"))

    def _set_movement(self, want):
        if self.backend == "pynput":
            for k in self.held - want:
                self.kb.release(k)
            for k in want - self.held:
                self.kb.press(k)
            self.held = want
        else:                              # hid: reconcile-loop handles timing
            self.kb.set_movement(want)

    def _tap(self, k):
        if self.backend == "pynput":
            self.kb.press(k); self.kb.release(k)
        else:
            self.kb.tap(k)

    def _wasd_keys(self, mvx, mvy):
        """Movement vector (screen fraction, champion assumed screen-center) ->
        set of WASD keys. Deadzone -> empty set (stand still)."""
        dx, dy = mvx - 0.5, mvy - 0.5
        if (dx * dx + dy * dy) ** 0.5 < self.deadzone:
            return set()
        ang = math.degrees(math.atan2(-dy, dx)) % 360   # -dy: up-screen = +angle
        _, combo = min(_WASD_DIRS, key=lambda d: min(abs(ang - d[0]), 360 - abs(ang - d[0])))
        return set(combo)

    def _click_move(self, mx, my, action):
        """Right-click-to-move. Fires only when the policy's sticky gate says a
        NEW command was issued -- holding a command between gate firings is the
        whole point of the gate, and re-clicking every frame would restart the
        order 20x/s. Returns True if a click actually went out."""
        if not action.get("gate", True):
            return False
        now = time.time()
        if now - self._last_click_t < self.click_min_interval:
            self.clicks_suppressed += 1
            return False
        self._last_click_t = now
        self.clicks_sent += 1
        if self.backend == "hid":
            # fractions of the FULL desktop, which is what absolute HID wants
            self.kb.move_click(mx / self.dw, my / self.dh)
        elif self.backend == "pynput" and self.mouse is not None:
            from pynput.mouse import Button
            self.mouse.position = (mx, my)
            self.mouse.click(Button.right)
        return True

    def send(self, action):
        mx = self.left + int(action["movement"][0] * self.w)
        my = self.top + int(action["movement"][1] * self.h)
        pressed = [k for k, v in action["abilities"].items() if v]
        if self.backend == "dry":
            want = (set() if self.movement_mode == "mouse"
                    else self._wasd_keys(*action["movement"]))
            return mx, my, pressed, sorted(want)

        # --- movement ---
        if self.movement_mode == "mouse":
            want = set()
            self._click_move(mx, my, action)
        else:
            # set held WASD keys (backend handles press/release diff)
            want = self._wasd_keys(*action["movement"])
            self._set_movement(want)

        # --- abilities: taps (AA -> optional attack key) ---
        for k in pressed:
            if k == "AA":
                if self.attack_key:
                    self._tap(self.attack_key)
                continue
            key = self.keys.get(k)
            if key:
                self._tap(key)
        return mx, my, pressed, sorted(want)

    def close(self):
        if self.backend == "hid":
            self.kb.close()
        elif self.backend == "pynput":
            for k in list(self.held):      # never leave a movement key stuck
                self.kb.release(k)
            self.held.clear()


class Recorder:
    """Persist what the MODEL saw (352x352, post-transform) + every action, for
    later debug/analysis. Not the raw 720p — the transformed frame is what the
    policy actually conditioned on. One timestamped dir per session:
      model_view_352.mp4  — the exact frames fed to the tokenizer
      actions.jsonl       — per-frame {i, t, movement, wasd, casts, reward, ms}
      meta.json           — ckpts, args, stream size, model config
    """

    def __init__(self, root, fps, meta):
        os.makedirs(root, exist_ok=True)
        self.dir = os.path.join(root, "session_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.vw = cv2.VideoWriter(os.path.join(self.dir, "model_view_352.mp4"),
                                  cv2.VideoWriter_fourcc(*"mp4v"), fps, (352, 352))
        self.actions = open(os.path.join(self.dir, "actions.jsonl"), "w")
        with open(os.path.join(self.dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        self.n = 0
        print(f"recording -> {self.dir}")

    def write(self, frame352_rgb01, rec):
        img = (np.clip(frame352_rgb01, 0, 1) * 255).astype(np.uint8)
        self.vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.actions.write(json.dumps(rec) + "\n")
        self.n += 1

    def close(self):
        try:
            self.vw.release()
            self.actions.close()
            print(f"recorded {self.n} frames -> {self.dir}")
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2-ckpt", required=True)
    ap.add_argument("--tokenizer-ckpt", required=True)
    ap.add_argument("--source", choices=["screen", "udp"], default="udp",
                    help="udp=receive the Windows ffmpeg stream (2-machine rig); "
                         "screen=local mss grab (single-machine/dev).")
    ap.add_argument("--udp-port", type=int, default=5000, help="UDP port to listen on (--source udp).")
    ap.add_argument("--stream-size", default="1280x720", help="WxH the Windows ffmpeg sends.")
    # DEFAULT OFF. action="store_false" without an explicit default yields True,
    # which silently kept this ON for every live run even after the offline sim
    # REFUTED it (mean brightness 0.136 -> 0.090; it crushes shadows). The class
    # docstring above has said "defaults OFF" since then; it was not true.
    ap.add_argument("--range-expand", dest="expand_range", action="store_true", default=False,
                    help="Limited->full colour-range expansion on decode. Default OFF: the "
                         "offline sim measured this making frames DARKER, not brighter. "
                         "Kept only for A/B experiments.")
    ap.add_argument("--no-range-expand", dest="expand_range", action="store_false",
                    help="Explicitly disable range expansion (already the default).")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Extra gamma on incoming frames (<1 brightens). Only if range-expand "
                         "isn't enough; verified ~0.6 rescues an uncorrected dark stream.")
    ap.add_argument("--capture-region", default=None, help="x,y,w,h for --source screen.")
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--target-fps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0=sample (calibrated casts, matches training); 0=greedy.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--inject", choices=["dry", "pynput", "hid"], default="dry",
                    help="dry=print only; pynput=in-Windows synthetic (Vanguard blocks in-game); "
                         "hid=external Pi keyboard gadget via hybrid_sender (real games).")
    ap.add_argument("--hid-host", default="192.168.1.144", help="Pi HID server address for --inject hid.")
    ap.add_argument("--deadzone", type=float, default=0.06,
                    help="Movement target within this fraction of screen-center = stand still "
                         "(no WASD key). Champion is assumed camera-locked to center.")
    ap.add_argument("--attack-key", default=None,
                    help="Key to tap for AA in WASD mode (e.g. an attack-move bind). "
                         "Default None: rely on the mode's auto-attack.")
    ap.add_argument("--desktop", default=None, metavar="WxH",
                    help="TRUE desktop size for absolute-HID mapping. Distinct from the "
                         "capture region: HID coords are 0..32767 across the WHOLE desktop, "
                         "so a region with a non-zero origin needs the desktop size to map "
                         "correctly. Previously the region SIZE was passed here, which sent "
                         "every click to the screen edge for any offset region. "
                         "Default: derived from the region (correct only when origin is 0,0).")
    ap.add_argument("--gate-bias", type=float, default=0.0,
                    help="Shift the movement FIRING RATE without changing where it clicks. "
                         "Every offline eval exposes this; the live entrypoint did not, so a "
                         "live run was pinned at the checkpoint's raw calibration.")
    ap.add_argument("--ability-thresh", type=float, default=0.0,
                    help="Greedy-mode cast threshold on the ability logit. At --temperature 0 "
                         "the default 0.0 means NEVER cast (trained logits sit at -3.5..-5). "
                         "Offline evals run -3.6..-4.0.")
    ap.add_argument("--movement-mode", choices=["wasd", "mouse"], default="mouse",
                    help="mouse=right-click-to-move, the primitive the policy is "
                         "trained on (click targets) and the default. wasd=keyboard-only "
                         "rig (no /dev/hidg1 on the Pi); the 8-way decode throws away "
                         "the target's distance and off-axis angle.")
    ap.add_argument("--click-min-interval", type=float, default=0.12,
                    help="Floor on seconds between right-clicks in mouse mode. The gate "
                         "normally paces these; this protects against an ungated "
                         "checkpoint clicking every frame.")
    ap.add_argument("--act-on-stale", action="store_true", default=False,
                    help="Feed the model repeated frames instead of skipping them. "
                         "DEFAULT OFF: the first live session filled its 16-frame context "
                         "with duplicates (58%% byte-identical) and the agent stood still. "
                         "Skipping means the agent runs at the STREAM's real rate.")
    ap.add_argument("--record", action="store_true", default=True,
                    help="Record model-view (352) video + per-frame actions (default on).")
    ap.add_argument("--no-record", dest="record", action="store_false")
    ap.add_argument("--record-dir", default="recordings", help="Root for session recordings.")
    ap.add_argument("--dry-run", action="store_true", help="alias for --inject dry")
    ap.add_argument("--allow-unstamped", action="store_true",
                    help="Permit --inject hid from a tree whose commit cannot be resolved. "
                         "Off by default: the deployed rig silently drifted 10 days from "
                         "the repo once, and an unidentifiable tree is how that happens.")
    args = ap.parse_args()
    if args.dry_run:
        args.inject = "dry"

    prov = provenance()
    print(f"[version] commit={prov.get('commit', '?')[:12]} dirty={prov.get('dirty', '?')} "
          f"<- {prov['source']}")
    if prov.get("staged_at"):
        print(f"[version] staged {prov['staged_at']} from {prov.get('staged_from')} | "
              f"ckpt sha {prov.get('phase2_sha256_16')} tok sha {prov.get('tokenizer_sha256_16')}")
    if prov["commit"] == "UNKNOWN" and args.inject == "hid" and not args.allow_unstamped:
        raise SystemExit(
            "REFUSING --inject hid: this tree has no VERSION file and is not a git\n"
            "checkout, so there is no way to know what code is about to drive real\n"
            "hardware. Re-deploy with ops/stage_desktop_standalone.sh (which stamps\n"
            "VERSION), or pass --allow-unstamped if you accept not knowing.")
    if prov.get("dirty") == "yes":
        print("[version] WARNING: working tree is DIRTY — the running code is not the commit above.")

    # WASD movement holds w/a/s/d down, so no ability may be bound to them (it
    # would fire while walking). Irrelevant in mouse mode, where movement uses no
    # keys at all and League's default Q/W/E/R binds are correct.
    collide = {k: v for k, v in DEFAULT_KEYS.items() if v in ("w", "a", "s", "d")}
    if collide and args.inject != "dry" and args.movement_mode == "wasd":
        raise SystemExit(f"ability keys collide with WASD movement: {collide}. "
                         "Rebind these abilities (in-game AND in DEFAULT_KEYS) off w/a/s/d, "
                         "or use --movement-mode mouse.")
    _mv = ("right-click (mouse)" if args.movement_mode == "mouse" else "WASD")
    print(f"keybinds (set these in-game): movement={_mv}  " +
          "  ".join(f"{a}={k}" for a, k in DEFAULT_KEYS.items()) +
          f"  AA={'(' + args.attack_key + ')' if args.attack_key else 'auto'}")
    if args.movement_mode == "mouse" and args.inject == "hid":
        print("mouse mode needs /dev/hidg1 on the Pi (setup_hid_combo.sh) — "
              "hid_server prints 'mouse gadget: ...' at startup; if it says "
              "unavailable, clicks are silently dropped.")

    if args.source == "udp":
        sw, sh = map(int, args.stream_size.split("x"))
        cap = StreamCapture(port=args.udp_port, size=(sw, sh),
                            expand_range=args.expand_range, gamma=args.gamma)
        region = (0, 0, sw, sh)
    else:
        region = tuple(map(int, args.capture_region.split(","))) if args.capture_region else None
        cap = ScreenCapture(region)
        if region is None:
            region = (cap.mon["left"], cap.mon["top"], cap.mon["width"], cap.mon["height"])
    if args.desktop:
        _dw, _dh = map(int, args.desktop.lower().split("x"))
    else:
        _dw, _dh = region[2], region[3]
        if (region[0] or region[1]):
            raise SystemExit(
                f"--capture-region has origin ({region[0]},{region[1]}) but --desktop was not "
                f"given. Absolute HID coords span the whole desktop, so without the real "
                f"desktop size every click maps past the edge and clamps. Pass --desktop WxH.")
    ctrl = InputController(region, args.inject, DEFAULT_KEYS, (_dw, _dh),
                           args.hid_host, deadzone=args.deadzone, attack_key=args.attack_key,
                           movement_mode=args.movement_mode,
                           click_min_interval=args.click_min_interval)
    agent = GarenAgent(args.phase2_ckpt, tokenizer_ckpt=args.tokenizer_ckpt,
                       context=args.context, device=args.device,
                       ability_thresh=args.ability_thresh)
    agent.reset()

    rec = None
    if args.record:
        rec = Recorder(args.record_dir, args.target_fps, {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase2_ckpt": args.phase2_ckpt, "tokenizer_ckpt": args.tokenizer_ckpt,
            "source": args.source, "stream_size": args.stream_size, "inject": args.inject,
            "temperature": args.temperature, "deadzone": args.deadzone,
            "keybinds": DEFAULT_KEYS, "use_actions": agent.use_actions,
            "movement_gate": getattr(agent.policy, "movement_gate", False),
            "movement_mode_head": getattr(agent.policy, "movement_mode", "axis"),
            "movement_mode_input": args.movement_mode,
            "gate_bias": args.gate_bias, "desktop": [_dw, _dh], "region": list(region),
            "act_on_stale": args.act_on_stale,
            "provenance": prov,
        })

    dt_target = 1.0 / args.target_fps
    cap_ms, enc_ms, act_ms, inj_ms = (deque(maxlen=60) for _ in range(4))
    n = 0
    n_stale = 0            # frames served that were IDENTICAL to the previous one
    recent = deque(maxlen=200)   # rolling freshness, so the readout tracks NOW
    n_iter = 0
    last_fresh_t = time.perf_counter()
    action = None
    mode = {"dry": "DRY-RUN (no input)", "pynput": "LIVE pynput", "hid": "LIVE HID gadget"}[args.inject]
    if args.source == "udp":
        cap.wait_first()                          # block until the stream is flowing
    print(f"\nRunning ({mode}, source={args.source}). Ctrl+C to stop.\n")
    t_start = time.perf_counter()      # for the clicks/s rate readout
    try:
        while True:
            t0 = time.perf_counter()
            frame, fresh = cap.grab_rgb01()
            if frame is None:                     # stream hiccup — hold, don't act on nothing
                time.sleep(dt_target)
                continue
            n_iter += 1
            recent.append(1 if fresh else 0)
            # --- STALENESS GUARD (the first live session died of this) ---------
            # The Windows stream delivered ~2-3 new frames/s into a 17 fps loop,
            # so 58% of consecutive frames were byte-identical. Feeding those to
            # the model fills its 16-frame context with duplicates: a world model
            # shown no change predicts no change, and the agent stands still.
            # A repeated frame carries ZERO new information, so the correct
            # response is to HOLD the previous action -- keyboard holds persist
            # on their own and the mouse must not re-click -- and simply not
            # advance the context. The agent then runs at the STREAM's true rate,
            # which is the rate at which the world actually changes.
            if not fresh:
                n_stale += 1
                if not args.act_on_stale:
                    dead = time.perf_counter() - last_fresh_t
                    if dead > 3.0:
                        print(f"  ** NO NEW FRAME FOR {dead:.1f}s — is the Windows "
                              f"ffmpeg still streaming? ** (holding last action)")
                        last_fresh_t = time.perf_counter()
                    slack = dt_target - (time.perf_counter() - t0)
                    if slack > 0:
                        time.sleep(slack)
                    continue
            else:
                last_fresh_t = time.perf_counter()
            t1 = time.perf_counter()
            lat = agent.encode_frame(frame)
            t2 = time.perf_counter()
            action = agent.act_from_latent(lat, temperature=args.temperature, gate_bias=args.gate_bias)
            t3 = time.perf_counter()
            mx, my, pressed, wasd = ctrl.send(action)
            t4 = time.perf_counter()
            cap_ms.append((t1 - t0) * 1e3); enc_ms.append((t2 - t1) * 1e3)
            act_ms.append((t3 - t2) * 1e3); inj_ms.append((t4 - t3) * 1e3)
            n += 1
            if rec is not None:
                casts_r = [k for k in pressed if k != "AA"]
                rec.write(agent.last_input352, {
                    "i": n, "t": round(t0, 4),
                    "movement": [round(float(x), 4) for x in action["movement"]],
                    "wasd": sorted(wasd), "casts": casts_r,
                    "aa": "AA" in pressed, "reward_pred": round(float(action["reward_pred"]), 4),
                    "ms": {"cap": round((t1 - t0) * 1e3, 1), "enc": round((t2 - t1) * 1e3, 1),
                           "act": round((t3 - t2) * 1e3, 1), "inj": round((t4 - t3) * 1e3, 1)},
                })
            if n % args.target_fps == 0:
                tot = sum(map(lambda d: sum(d) / len(d), (cap_ms, enc_ms, act_ms, inj_ms)))
                casts = [k for k in pressed if k != "AA"]
                # ROLLING, not cumulative: what matters is whether the stream is
                # healthy NOW, and a cumulative average hides a stream that just
                # died behind ten good minutes.
                stale_pct = 1.0 - (sum(recent) / max(len(recent), 1))
                acted_fps = n / max(time.perf_counter() - t_start, 1e-6)
                warn = ("  ** STALE STREAM: acting at %.1f fps, raise the Windows "
                        "ffmpeg framerate **" % acted_fps) if stale_pct > 0.25 else ""
                if args.movement_mode == "mouse":
                    # clicks/s is the number to watch: humans issue ~2/s, and the
                    # trained gate sits near 0.2/s without --gate-bias.
                    rate = ctrl.clicks_sent / max(time.perf_counter() - t_start, 1e-6)
                    mv_s = f"clicks={ctrl.clicks_sent:4d}({rate:4.2f}/s)"
                else:
                    mv_s = f"wasd={''.join(wasd).upper() or 'stand':5s}"
                print(f"frame {n:6d} | {mv_s} "
                      f"casts={casts or '-'} rew={action['reward_pred']:+.2f} | "
                      f"{acted_fps:4.1f}fps acted (model cap {1000/tot:4.1f}) "
                      f"[cap{sum(cap_ms)/len(cap_ms):.0f} enc{sum(enc_ms)/len(enc_ms):.0f} "
                      f"act{sum(act_ms)/len(act_ms):.0f} inj{sum(inj_ms)/len(inj_ms):.0f}ms] "
                      f"stale={stale_pct:.0%}{warn}")
            slack = dt_target - (time.perf_counter() - t0)
            if slack > 0:
                time.sleep(slack)
    except KeyboardInterrupt:
        el = time.perf_counter() - t_start
        print(f"\nstopped after {n} acted frames in {el:.0f}s ({n/max(el,1e-6):.1f} fps acted); "
              f"{n_stale}/{n_iter} loop iterations saw a repeated frame "
              f"({n_stale/max(n_iter,1):.0%}).")
        if args.movement_mode == "mouse" and args.inject == "hid":
            print(f"mouse: {getattr(ctrl.kb, 'mouse_stats', lambda: {})()}  "
                  f"clicks sent={ctrl.clicks_sent} suppressed={ctrl.clicks_suppressed}")
    finally:
        ctrl.close()
        if hasattr(cap, "close"):
            cap.close()
        if rec is not None:
            rec.close()


if __name__ == "__main__":
    main()
