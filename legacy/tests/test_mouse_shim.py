#!/usr/bin/env python3
"""Offline verification of the relative-mouse shim — no Pi, no game, no GPU.

The mouse path cannot be tested on the real rig without a live game, so
everything that CAN be checked without hardware is checked here, by pushing real
bytes through the real code:

  * hid_server._parse over every wire form the sender actually emits, plus the
    malformed input that must never cost us the socket.
  * RelMouse's report packing against the 4-byte descriptor in
    setup_hid_combo.sh (a wrong-width report is a silent no-op in-game).
  * HybridKeyboard's corner-relative addressing, driven through a fake socket
    into a SIMULATED SCREEN that clamps like a real one — so we can measure
    where the cursor would actually land, including the thing this design exists
    for: that error does NOT accumulate across commands.

    PYTHONPATH=src python tests/test_mouse_shim.py
"""
import io
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KS = os.path.join(HERE, os.pardir, "scripts", "keysender")
sys.path.insert(0, KS)

import hid_server                                            # noqa: E402
from hybrid_sender import HybridKeyboard, load_calibration   # noqa: E402

FAILED = []


def ck(cond, label, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
    if not cond:
        FAILED.append(label)


# --------------------------------------------------------------------------- #
# 1. wire protocol
# --------------------------------------------------------------------------- #
def test_parse():
    print("\n[1] hid_server._parse over the real wire forms")
    p = hid_server._parse
    ck(p("press w") == {"t": "key", "k": "w", "d": 1}, "press")
    ck(p("release w") == {"t": "key", "k": "w", "d": 0}, "release")
    ck(p("tap q") == {"t": "tap", "k": "q"}, "tap")
    ck(p("reset") == {"t": "reset"}, "reset")
    ck(p("mouse -30 0") == {"t": "rel", "dx": -30, "dy": 0}, "mouse negative dx")
    ck(p("mouse 0 27") == {"t": "rel", "dx": 0, "dy": 27}, "mouse positive dy")
    ck(p("click right") == {"t": "click", "b": "right"}, "click right")
    ck(p('{"t": "reset"}') == {"t": "reset"}, "json still accepted")
    # malformed input must be ignored, never raise (it would drop the socket
    # mid-game and clear every held key).
    for bad in ("", "   ", "mouse", "mouse a b", "mouse 1", "click", "{", "{bad json}",
                "\x00\xff garbage", "press"):
        try:
            ck(p(bad) is None, f"malformed {bad!r} -> None")
        except Exception as e:                                # noqa: BLE001
            ck(False, f"malformed {bad!r} raised", repr(e))
    # every line hybrid_sender can emit must parse
    emitted = ["mouse -127 0", "mouse 0 -127", "mouse 30 0", "mouse 0 12",
               "click right", "press a", "release a", "reset"]
    ck(all(p(x) is not None for x in emitted), "every sender-emitted line parses")


def test_report_packing():
    print("\n[2] RelMouse report width vs the gadget descriptor")
    sink = io.BytesIO()
    ms = hid_server.RelMouse.__new__(hid_server.RelMouse)
    ms.fd, ms.buttons = sink, 0
    ms.move_rel(30, 0)
    ms.move_rel(-127, 0)
    ms.move_rel(500, -500)                     # must clamp into signed-byte range
    data = sink.getvalue()
    ck(len(data) == 12, "3 reports x 4 bytes", f"got {len(data)}")
    r0 = struct.unpack("<Bbbb", data[0:4])
    r2 = struct.unpack("<Bbbb", data[8:12])
    ck(r0 == (0, 30, 0, 0), "report encodes (buttons, dx, dy, wheel)", str(r0))
    ck(r2 == (0, 127, -127, 0), "out-of-range deltas clamp, not wrap", str(r2))
    # click: press report then release report, buttons bit 2 = right
    sink2 = io.BytesIO()
    ms.fd, ms.buttons = sink2, 0
    ms.click("right")
    d2 = sink2.getvalue()
    ck(len(d2) == 8, "click = 2 reports", f"got {len(d2)}")
    ck(struct.unpack("<Bbbb", d2[0:4])[0] == 2, "right button bit set")
    ck(struct.unpack("<Bbbb", d2[4:8])[0] == 0, "button released again")
    ck(ms.buttons == 0, "no button left stuck")


# --------------------------------------------------------------------------- #
# 2. corner-relative addressing, against a simulated clamping screen
# --------------------------------------------------------------------------- #
class FakeSock:
    """Captures the lines the sender writes instead of talking to a Pi."""

    def __init__(self):
        self.lines = []

    def sendall(self, b):
        for ln in b.decode().split("\n"):
            if ln.strip():
                self.lines.append(ln.strip())

    def close(self):
        pass


class Screen:
    """A screen that clamps, and (optionally) a nonlinear pointer-acceleration
    curve so we can check the design's robustness claim rather than assume it.

    px travelled for a report of magnitude m = m * k * (m/chunk)**accel_exp.
    accel_exp=0 is 'Enhance pointer precision' OFF (perfectly linear).
    """

    def __init__(self, w, h, span, chunk, accel_exp=0.0, scale=1.0):
        self.w, self.h, self.chunk = w, h, chunk
        self.kx = (w / span[0]) * scale        # true px per unit; scale != 1 =>
        self.ky = (h / span[1]) * scale        # the stored calibration is WRONG
        self.accel_exp = accel_exp
        self.x = self.y = 0.0

    def _px(self, units, k):
        m = abs(units)
        if m == 0:
            return 0.0
        gain = (m / self.chunk) ** self.accel_exp
        return (units * k) * gain

    def apply(self, line):
        parts = line.split()
        if parts[0] != "mouse":
            return
        dx, dy = int(parts[1]), int(parts[2])
        self.x = min(self.w, max(0.0, self.x + self._px(dx, self.kx)))
        self.y = min(self.h, max(0.0, self.y + self._px(dy, self.ky)))

    def frac(self):
        return self.x / self.w, self.y / self.h


def drive(span, chunk, targets, accel_exp=0.0, scale=1.0, screen_wh=(1280, 720)):
    """Run real HybridKeyboard mouse commands through the fake socket + screen.
    -> list of (target_frac, landed_frac, clicked_bool)."""
    kb = HybridKeyboard.__new__(HybridKeyboard)          # no socket, no threads
    kb.sock, kb.running, kb.lock = FakeSock(), True, __import__("threading").Lock()
    kb.send_lock = __import__("threading").Lock()
    kb.span, kb.chunk, kb.interval = span, chunk, 0.0
    kb._m_pos = kb._m_target = kb._m_click = None
    kb._m_slams = kb._m_moves = 0
    kb._m_last_ms = 0.0

    scr = Screen(*screen_wh, span, chunk, accel_exp, scale)
    out = []
    for fx, fy in targets:
        kb.sock.lines.clear()
        kb.move_click(fx, fy)
        # run exactly one iteration of the loop body
        with kb.lock:
            tgt, click = kb._m_target, kb._m_click
            kb._m_target, kb._m_click = None, None
        kb._slam_corner()
        kb._travel_axis(tgt[0] * kb.span[0], "x")
        kb._travel_axis(tgt[1] * kb.span[1], "y")
        if click:
            kb._send(f"click {click}")
        clicked = False
        for ln in kb.sock.lines:
            if ln.startswith("click"):
                clicked = True
            scr.apply(ln)
        out.append(((fx, fy), scr.frac(), clicked))
    return out, kb, scr


def test_geometry():
    span, chunk = (649.0, 367.0), 30
    grid = [(x / 4, y / 4) for x in range(5) for y in range(5)]

    print("\n[3] corner-relative addressing, IDEAL rig (linear, calibration exact)")
    res, kb, _ = drive(span, chunk, grid)
    err = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b, _ in res]
    ck(max(err) < 0.01, "every target within 1% of screen", f"max err {max(err):.4f}")
    ck(all(c for _, _, c in res), "a click is emitted for every move_click")
    ck(kb._m_slams == len(grid), "exactly one corner slam per command",
       f"{kb._m_slams} slams / {len(grid)} cmds")

    print("\n[4] the actual claim: error does NOT accumulate")
    # 200 commands bouncing across the screen, with accel ON and a 12%-wrong
    # calibration. If anything accumulated, late errors would exceed early ones.
    many = [((i * 7 % 9) / 8, (i * 5 % 9) / 8) for i in range(200)]
    res, _, _ = drive(span, chunk, many, accel_exp=0.15, scale=1.12)
    e = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b, _ in res]
    first20, last20 = sum(e[:20]) / 20, sum(e[-20:]) / 20
    ck(last20 <= first20 * 1.10, "late error is not worse than early error",
       f"first20 {first20:.4f} vs last20 {last20:.4f}")
    ck(max(e[-20:]) < 0.25, "bounded even with a 12%-wrong span + accel",
       f"max {max(e[-20:]):.4f}")
    # and the same target always lands in the same place -> no drift
    rep, _, _ = drive(span, chunk, [(0.7, 0.3)] * 40, accel_exp=0.15, scale=1.12)
    lands = {(round(b[0], 6), round(b[1], 6)) for _, b, _ in rep}
    ck(len(lands) == 1, "repeating one target is perfectly repeatable",
       f"{len(lands)} distinct landing spots")

    print("\n[5] contrast: recovery after the cursor is knocked off-belief")
    # HONEST NOTE: under a pure systematic scale error, dead reckoning is not
    # automatically worse — it scales a short DELTA rather than the full
    # corner-to-target distance, so its per-move error can be smaller. Measured:
    # dead-reckon 0.0625 vs corner-relative 0.0650 on the sweep above. That is
    # NOT the property the corner slam is bought for.
    #
    # What it IS bought for is that a relative mouse's believed position can be
    # corrupted with no way to notice: an edge clamp, a dropped USB report, a
    # socket reconnect, or a human bumping the physical mouse at a demo. Dead
    # reckoning has no observation to correct with and is wrong FOREVER after
    # one such event; corner-relative is correct again on the very next command.
    # That is what this measures.
    def run(mode, disturb_at, n=40):
        kbd = HybridKeyboard.__new__(HybridKeyboard)
        kbd.sock, kbd.running = FakeSock(), True
        kbd.lock = kbd.send_lock = __import__("threading").Lock()
        kbd.span, kbd.chunk, kbd.interval = span, chunk, 0.0
        kbd._m_pos = None
        kbd._m_slams = kbd._m_moves = 0
        s = Screen(1280, 720, span, chunk)          # ideal rig: isolate the disturbance
        errs = []
        for i in range(n):
            fx, fy = (i * 7 % 9) / 8, (i * 5 % 9) / 8
            kbd.sock.lines.clear()
            if mode == "corner" or kbd._m_pos is None:
                kbd._slam_corner()
            tx, ty = fx * span[0], fy * span[1]
            if mode == "corner":
                kbd._travel_axis(tx, "x"); kbd._travel_axis(ty, "y")
            else:
                kbd._travel_axis(tx - kbd._m_pos[0], "x")
                kbd._travel_axis(ty - kbd._m_pos[1], "y")
            for ln in kbd.sock.lines:
                s.apply(ln)
            if i == disturb_at:                     # somebody bumps the mouse
                s.x, s.y = s.w * 0.8, s.h * 0.8
            errs.append(max(abs(fx - s.frac()[0]), abs(fy - s.frac()[1])))
        return errs

    c_err, d_err = run("corner", 10), run("dead", 10)
    c_after = max(c_err[11:])
    d_after = max(d_err[11:])
    ck(c_after < 0.02, "corner-relative is accurate again on the NEXT command",
       f"max err after the bump {c_after:.4f}")
    ck(d_after > 0.15, "dead reckoning never recovers from it",
       f"max err after the bump {d_after:.4f}")

    print("\n[6] report budget per command (timing sanity)")
    res, kb, _ = drive(span, chunk, [(1.0, 1.0)])
    kb.sock.lines.clear()
    kb._m_pos = None
    kb._slam_corner()
    n_slam = len(kb.sock.lines)
    kb.sock.lines.clear()
    kb._travel_axis(span[0], "x"); kb._travel_axis(span[1], "y")
    n_travel = len(kb.sock.lines)
    ms = (n_slam + n_travel) * 2.0            # MOUSE_INTERVAL = 2 ms
    ck(n_slam <= 24, "slam is cheap with the max-magnitude report", f"{n_slam} reports")
    ck(ms < 200, "worst-case command under 200 ms at 2 ms/report",
       f"{n_slam}+{n_travel} reports = {ms:.0f} ms")


def test_calibration_file():
    print("\n[7] calibration loading")
    span, chunk, src = load_calibration("/nonexistent/mouse_calibration.json")
    ck(span == (649.0, 367.0) and chunk == 30, "missing file -> measured fallback")
    ck("FALLBACK" in src, "fallback is announced, not silent", src)
    import json
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"span": [700.5, 400.25], "chunk": 25, "measured_at": "x"}, fh)
        good = fh.name
    span, chunk, src = load_calibration(good)
    ck(span == (700.5, 400.25) and chunk == 25, "file wins over the fallback")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        fh.write("{not json")
        bad = fh.name
    span, chunk, src = load_calibration(bad)
    ck(span == (649.0, 367.0), "corrupt file -> fallback, no crash")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"span": [0, 0], "chunk": 30}, fh)
        zero = fh.name
    span, _, _ = load_calibration(zero)
    ck(span == (649.0, 367.0), "implausible span rejected")
    for f in (good, bad, zero):
        os.unlink(f)

    # the file the demo will actually use
    real = os.path.join(KS, "mouse_calibration.json")
    span, chunk, src = load_calibration(real)
    print(f"       repo calibration: span={span} chunk={chunk} <- {src}")


if __name__ == "__main__":
    t0 = time.time()
    test_parse()
    test_report_packing()
    test_geometry()
    test_calibration_file()
    print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'ALL PASS'}  ({time.time()-t0:.1f}s)")
    sys.exit(1 if FAILED else 0)
