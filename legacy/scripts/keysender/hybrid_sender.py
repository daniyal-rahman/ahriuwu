#!/usr/bin/env python3
"""Desktop-side keyboard+mouse sender for the live agent -> Pi HID gadget.

Speaks the Pi server's line protocol ("press <key>\\n" / "release <key>\\n" /
"mouse <dx> <dy>\\n" / "click <btn>\\n") over one persistent TCP connection.
Keyboard: desired/actual reconciliation with hold timing sampled from real League
play. Abilities are momentary taps; WASD movement is a persistent held set.

THE MOUSE PROBLEM AND ITS SOLUTION
----------------------------------
The policy emits an ABSOLUTE normalized (fx, fy) in the capture region's space.
The Pi gadget is a RELATIVE mouse: every report is a signed-byte delta, and the
cursor's position can never be read back. Bridging the two by dead reckoning
(track a believed position, send the difference) accumulates error forever --
every rounding residue and every bit of pointer-acceleration nonlinearity is
permanent, and there is no observation to correct it with.

So we do not dead-reckon across commands. Each command is CORNER-RELATIVE:

    1. slam hard into the top-left corner. The OS clamps there, so that is the
       one position we can know without observing anything.
    2. from that known origin travel exactly fx*span_x, fy*span_y.

Every move's error is then INDEPENDENT of every previous move. Nothing
accumulates, so the calibration only has to be roughly right, and a wrong span
shows up as a constant scale error rather than a cursor that walks off the
screen after a minute of play.

AXIS-ALIGNED ONLY. With Windows "Enhance pointer precision" on, the pixels a
report travels scale with its MAGNITUDE, so a diagonal (30,30) carries each axis
further than an axis-aligned (30,0) does. Serializing the axes keeps every
travel report at the same magnitude the calibration was measured at. This is
also why MOUSE_CHUNK must match the chunk calibrate_mouse.py used -- changing it
silently invalidates the span.

The SLAM is exempt: it ends at a hardware clamp, so where it lands does not
depend on how far each report travelled. It therefore uses the largest legal
report (SLAM_CHUNK) purely to finish faster.

Calibration is LOADED FROM DISK (mouse_calibration.json next to this file,
written by calibrate_mouse.py), not hardcoded, so a re-calibration on the rig
takes effect without editing code. The constants below are only the fallback.

Standalone self-tests (only with the Pi NOT plugged into a live game):
    python scripts/keysender/hybrid_sender.py --host 192.168.1.144
    python scripts/keysender/hybrid_sender.py --host 192.168.1.144 --mouse-test
"""
import argparse
import json
import os
import random
import socket
import threading
import time

PI_IP = "192.168.1.144"
PORT = 9999

# Hold-duration distribution from the real League input log.
HOLD_MEDIAN, HOLD_STD, HOLD_MIN, HOLD_MAX = 0.132, 0.090, 0.070, 0.450

# --- relative-mouse geometry (FALLBACK ONLY -- see load_calibration) ----------
# MOUSE_SPAN: how many mouse units it takes to cross the screen, measured on the
# rig by driving N axis-aligned reports out of the clamped top-left corner and
# reading the cursor back off the video stream.
MOUSE_SPAN = (649.0, 367.0)
MOUSE_CHUNK = 30                # MUST match what the span was calibrated with
SLAM_CHUNK = 127                # max legal signed-byte report; slam only
MOUSE_INTERVAL = 0.002          # gap between mouse reports (~500 Hz ceiling)

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "mouse_calibration.json")


def load_calibration(path=CALIB_PATH):
    """-> (span_xy, chunk, source_string). Falls back to the constants above.

    Never raises: a missing or corrupt calibration must degrade to the last
    known-good numbers rather than take the demo down. The returned source
    string is printed at startup so an unnoticed fallback is impossible.
    """
    try:
        with open(path) as fh:
            d = json.load(fh)
        span = (float(d["span"][0]), float(d["span"][1]))
        chunk = int(d.get("chunk", MOUSE_CHUNK))
        if not (span[0] > 1 and span[1] > 1 and chunk > 0):
            raise ValueError(f"implausible calibration: span={span} chunk={chunk}")
        return span, chunk, f"{path} (measured {d.get('measured_at', '?')})"
    except FileNotFoundError:
        return MOUSE_SPAN, MOUSE_CHUNK, "BUILT-IN FALLBACK (no calibration file)"
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"[hid] calibration file unusable ({e}); using fallback")
        return MOUSE_SPAN, MOUSE_CHUNK, f"BUILT-IN FALLBACK ({type(e).__name__})"


def save_calibration(span, chunk, path=CALIB_PATH, **extra):
    """Persist a measured calibration where HybridKeyboard will find it."""
    d = {"span": [round(float(span[0]), 2), round(float(span[1]), 2)],
         "chunk": int(chunk),
         "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2)
    return path


def sample_hold() -> float:
    return min(HOLD_MAX, max(HOLD_MIN, random.gauss(HOLD_MEDIAN, HOLD_STD)))


class HybridKeyboard:
    """Persistent, self-healing sender. desired = keys that SHOULD be down; a
    background loop reconciles actual->desired with small human-timed gaps.
    With mouse=True a second loop serves corner-relative move/click commands."""

    def __init__(self, host=PI_IP, port=PORT, connect_timeout=5, mouse=False,
                 span=None, chunk=None, calib_path=CALIB_PATH,
                 interval=MOUSE_INTERVAL):
        self.host, self.port = host, port
        self.connect_timeout = connect_timeout
        self.sock = None
        self.desired = set()
        self.actual = set()
        self._move = set()            # the WASD subset of desired (persistent holds)
        self.lock = threading.Lock()
        # The keyboard reconcile loop and the mouse loop both write to this one
        # socket (the Pi server accepts a single connection), so serialize writes
        # or two commands interleave mid-line and the server sees garbage.
        self.send_lock = threading.Lock()
        self.running = True

        # --- mouse state (only used when mouse=True) ---
        cal_span, cal_chunk, self.calib_source = load_calibration(calib_path)
        self.span = tuple(span) if span else cal_span
        self.chunk = int(chunk) if chunk else cal_chunk
        self.interval = interval
        self._m_target = None         # (fx, fy) screen fraction, or None
        self._m_click = None          # button name queued for arrival, or None
        self._m_pos = None            # believed position in units; None = unknown
        self._m_slams = 0
        self._m_moves = 0
        self._m_last_ms = 0.0         # wall time of the last completed command

        self._connect()
        self.worker = threading.Thread(target=self._reconcile_loop, daemon=True)
        self.worker.start()
        self.mouse_worker = None
        if mouse:
            print(f"[hid] mouse: corner-relative, span={self.span[0]:.0f}x{self.span[1]:.0f} "
                  f"units chunk={self.chunk} <- {self.calib_source}")
            self.mouse_worker = threading.Thread(target=self._mouse_loop, daemon=True)
            self.mouse_worker.start()

    def _connect(self):
        while self.running:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(self.connect_timeout)
                s.connect((self.host, self.port))
                s.settimeout(None)
                self.sock = s
                print(f"[hid] connected {self.host}:{self.port}")
                return
            except OSError as e:
                print(f"[hid] connect failed: {e} - retry 2s")
                time.sleep(2)

    def _send(self, cmd: str):
        with self.send_lock:
            try:
                self.sock.sendall((cmd.strip() + "\n").encode())
            except OSError as e:
                print(f"[hid] send failed: {e}; reconnecting")
                self._connect()
                self.sock.sendall((cmd.strip() + "\n").encode())
                self._m_pos = None            # cursor unknown after a reconnect

    # --- public API used by play_live ---
    def set_movement(self, want: set):
        """Replace the held WASD set; press/release only the difference."""
        want = {k.lower().strip() for k in want}
        with self.lock:
            for k in self._move - want:
                self.desired.discard(k)
            for k in want - self._move:
                self.desired.add(k)
            self._move = want

    def tap(self, key: str, hold: float | None = None):
        """Momentary keypress with a human hold duration (abilities)."""
        key = key.lower().strip()
        with self.lock:
            self.desired.add(key)
        threading.Timer(hold if hold is not None else sample_hold(),
                        self._tap_release, args=(key,)).start()

    def _tap_release(self, key: str):
        with self.lock:
            # never release a key that movement is currently holding
            if key not in self._move:
                self.desired.discard(key)

    # --- mouse API (relative gadget, corner-relative addressing) ---
    def move_click(self, fx, fy, button="right"):
        """Aim at screen fraction (fx, fy) and click once the cursor ARRIVES.

        Returns immediately: the travel takes ~100 ms of reports and firing the
        click from the caller's thread would land it mid-flight. Only the LATEST
        request is served -- if the policy issues a new target while the previous
        one is still travelling, the stale one is dropped rather than queued,
        which is what you want at 17 fps.
        """
        with self.lock:
            self._m_target = (min(1.0, max(0.0, float(fx))), min(1.0, max(0.0, float(fy))))
            self._m_click = button

    def move_to(self, fx, fy):
        """Aim without clicking (hover -- e.g. so a later cast lands here)."""
        with self.lock:
            self._m_target = (min(1.0, max(0.0, float(fx))), min(1.0, max(0.0, float(fy))))
            self._m_click = None

    # Back-compat aliases: the older JSON-absolute API named these move/click.
    def move(self, fx, fy):
        self.move_to(fx, fy)

    def click(self, fx, fy, button="right"):
        self.move_click(fx, fy, button)

    def believed(self):
        """Dead-reckoned cursor as a screen fraction, or None if unknown.

        Valid only WITHIN one command (we re-zero at the corner before each), so
        this is a debug readout, not something to plan a move from.
        """
        p = self._m_pos
        return None if p is None else (p[0] / self.span[0], p[1] / self.span[1])

    def mouse_stats(self):
        return {"moves": self._m_moves, "slams": self._m_slams,
                "last_cmd_ms": round(self._m_last_ms * 1e3, 1)}

    def _slam_corner(self):
        """Drive hard into the top-left corner; the clamp makes the position known.

        Uses the LARGEST legal report rather than the calibrated chunk: the slam
        terminates at a hardware clamp, so how far each report travels is
        irrelevant -- only that the total is comfortably past the edge. At chunk
        30 this was ~22 reports/axis (~90 ms); at 127 it is ~9 (~35 ms).
        """
        n = int(max(self.span) / SLAM_CHUNK) + 4
        for _ in range(n):
            self._send(f"mouse {-SLAM_CHUNK} 0")
            time.sleep(self.interval)
        for _ in range(n):
            self._send(f"mouse 0 {-SLAM_CHUNK}")
            time.sleep(self.interval)
        self._m_pos = [0.0, 0.0]
        self._m_slams += 1

    def _travel_axis(self, dist, axis):
        """Send `dist` units along one axis as chunk-sized reports + a remainder.

        Full chunks travel exactly the calibrated distance. The single remainder
        report has a smaller magnitude, so under pointer acceleration it
        undershoots slightly -- bounded by one chunk (~5% of screen width) and,
        because we re-zero at the corner every command, never accumulated.
        """
        sign = 1 if dist >= 0 else -1
        remaining = abs(dist)
        while remaining >= 1.0:
            step = self.chunk if remaining >= self.chunk else int(round(remaining))
            if step < 1:
                break
            d = sign * step
            self._send(f"mouse {d} 0" if axis == "x" else f"mouse 0 {d}")
            self._m_pos[0 if axis == "x" else 1] += d
            remaining -= step
            time.sleep(self.interval)

    def _mouse_loop(self):
        while self.running:
            with self.lock:
                tgt, click = self._m_target, self._m_click
                self._m_target, self._m_click = None, None      # serve latest only
            if tgt is None:
                time.sleep(0.004)
                continue
            t0 = time.perf_counter()
            # CORNER-RELATIVE: re-zero, then travel the absolute target from a
            # known origin. This is the whole reason errors do not accumulate.
            self._slam_corner()
            self._travel_axis(tgt[0] * self.span[0], "x")
            self._travel_axis(tgt[1] * self.span[1], "y")
            if click and self.running:
                self._send(f"click {click}")
            self._m_moves += 1
            self._m_last_ms = time.perf_counter() - t0

    def _reconcile_loop(self):
        while self.running:
            with self.lock:
                to_press = self.desired - self.actual
                to_release = self.actual - self.desired
            for key in to_release:
                self._send(f"release {key}")
                self.actual.discard(key)
                time.sleep(random.uniform(0.006, 0.018))
            for key in to_press:
                self._send(f"press {key}")
                self.actual.add(key)
                time.sleep(random.uniform(0.008, 0.025))
            time.sleep(0.007)          # ~140 Hz reconcile

    def close(self):
        self.running = False
        with self.lock:
            self.desired.clear()
            self._move.clear()
            self._m_target = None          # stop the mouse loop chasing a target
            self._m_click = None
        time.sleep(0.15)               # let the loops flush releases
        try:
            for k in list(self.actual):
                self._send(f"release {k}")
            self._send("reset")        # belt and braces: clear keys AND buttons
            if self.sock:
                self.sock.close()
        except OSError:
            pass
        print("[hid] closed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=PI_IP)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--mouse-test", action="store_true",
                    help="Move the pointer to the four corners and the centre, NO clicks. "
                         "Watch the screen: it should hit each spot within a few percent.")
    args = ap.parse_args()
    if args.mouse_test:
        kb = HybridKeyboard(args.host, args.port, mouse=True)
        for fx, fy in ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.5, 0.5)):
            print(f"  -> ({fx}, {fy})")
            kb.move_to(fx, fy)
            time.sleep(1.2)
        print(f"  stats: {kb.mouse_stats()}")
        kb.close()
        return
    kb = HybridKeyboard(args.host, args.port)
    print("self-test: q w e taps + a WASD hold burst")
    time.sleep(0.4)
    for k in ("q", "w", "e"):
        kb.tap(k); time.sleep(random.uniform(0.12, 0.2))
    kb.set_movement({"w", "d"}); time.sleep(0.5)
    kb.set_movement({"a"}); time.sleep(0.4)
    kb.set_movement(set()); time.sleep(0.2)
    kb.close()
    print("done")


if __name__ == "__main__":
    main()
