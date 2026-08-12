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
import random
import socket
import threading
import time

PI_IP = "192.168.1.144"
PORT = 9999

# Hold-duration distribution from the real League input log.
HOLD_MEDIAN, HOLD_STD, HOLD_MIN, HOLD_MAX = 0.132, 0.090, 0.070, 0.450


def sample_hold() -> float:
    return min(HOLD_MAX, max(HOLD_MIN, random.gauss(HOLD_MEDIAN, HOLD_STD)))


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
