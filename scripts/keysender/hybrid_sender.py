#!/usr/bin/env python3
"""Desktop-side keyboard sender for the live agent -> Pi HID gadget.

Speaks the Pi server's line protocol ("press <key>\\n" / "release <key>\\n")
with a persistent connection, a desired/actual reconciliation loop, and hold
timing sampled from real League play. This is the user's hardened HybridKeyboard,
generalized for the agent: WASD movement is a persistent held set (set_movement);
abilities are momentary taps with a realistic press duration (tap).

Standalone self-test (sends a small burst to the configured Pi — only run when the
Pi is NOT plugged into a live game):
    python scripts/keysender/hybrid_sender.py --host 192.168.1.144
"""
import argparse
import json
import random
import socket
import threading
import time

PI_IP = "192.168.1.144"
PORT = 9999

# Hold-duration distribution from the real League input log.
HOLD_MEDIAN, HOLD_STD, HOLD_MIN, HOLD_MAX = 0.132, 0.090, 0.070, 0.450

# USB HID absolute-pointer range. The gadget reports 0..32767 across the whole
# desktop, so the agent's fractional screen coords scale straight onto it.
ABS_MAX = 32767


def sample_hold() -> float:
    return min(HOLD_MAX, max(HOLD_MIN, random.gauss(HOLD_MEDIAN, HOLD_STD)))


def _to_abs(f: float) -> int:
    """Fractional screen coord (0..1) -> absolute HID units, clamped."""
    return max(0, min(ABS_MAX, int(round(float(f) * ABS_MAX))))


class HybridKeyboard:
    """Persistent, self-healing sender. desired = keys that SHOULD be down; a
    background loop reconciles actual->desired with small human-timed gaps."""

    def __init__(self, host=PI_IP, port=PORT, connect_timeout=5):
        self.host, self.port = host, port
        self.connect_timeout = connect_timeout
        self.sock = None
        self.desired = set()
        self.actual = set()
        self._move = set()            # the WASD subset of desired (persistent holds)
        self.lock = threading.Lock()
        # Guards the socket itself. The reconcile thread writes key lines while
        # the caller thread writes mouse lines; without this the two can
        # interleave mid-line and the server sees corrupt commands.
        self.send_lock = threading.Lock()
        self.running = True
        self._connect()
        self.worker = threading.Thread(target=self._reconcile_loop, daemon=True)
        self.worker.start()

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

    # --- mouse (League is right-click-to-move; needs /dev/hidg1 on the Pi) ---
    # These go out as JSON because the text line protocol has no mouse verbs.
    # hid_server._parse accepts both formats on the same socket.
    def move(self, fx: float, fy: float):
        """Move the pointer to a fractional screen position (0..1, 0..1)."""
        x, y = _to_abs(fx), _to_abs(fy)
        self._send(json.dumps({"t": "move", "x": x, "y": y}))

    def click(self, fx: float, fy: float, button: str = "right"):
        """Click at a fractional screen position. Right-click = move/AA order."""
        x, y = _to_abs(fx), _to_abs(fy)
        self._send(json.dumps({"t": "click", "b": button, "x": x, "y": y}))

    def move_click(self, fx: float, fy: float):
        """The League movement primitive: right-click at (fx, fy)."""
        self.click(fx, fy, "right")

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
        time.sleep(0.15)               # let the loop flush releases
        try:
            for k in list(self.actual):
                self._send(f"release {k}")
            if self.sock:
                self.sock.close()
        except OSError:
            pass
        print("[hid] closed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=PI_IP)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
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
