#!/usr/bin/env python3
"""Network HID injection server for the live agent (runs on the gadget device).

Completes the keysender rig: the original hid_keyboard.py keeps /dev/hidg0 open
with clean press/release semantics but its network listener was a stub, and there
was no mouse — League is right-click-to-move. This serves BOTH functions of the
combo gadget (setup_hid_combo.sh) over one TCP socket.

Protocol: line-oriented on TCP :9999 (one connection at a time, e.g.
play_live.py with --inject hid). Coordinates are absolute logical 0..32767
across the desktop.

    {"t": "tap",   "k": "q"}                      # press+release a key
    {"t": "key",   "k": "q", "d": 1}              # hold (d=1) / release (d=0)
    {"t": "click", "b": "right", "x": 20000, "y": 15000}
    {"t": "move",  "x": 20000, "y": 15000}
    {"t": "reset"}                                # release everything

BOTH wire formats are accepted, because the rig has two senders that predate
each other: hybrid_sender.py speaks the original plain-text line protocol
("press w" / "release w" / "tap q" / "reset"), while the mouse commands only
exist in JSON. Parsing JSON only would make the first keystroke from
hybrid_sender raise JSONDecodeError, drop the connection and clear every input
-- i.e. the agent silently does nothing in-game. _parse handles either.

On disconnect all keys/buttons are released (no stuck inputs mid-game).
"""
import argparse
import json
import socket
import struct
import time

from hid_keyboard import PersistentKeyboard

MOUSE_PATH = "/dev/hidg1"
BTN = {"left": 1, "right": 2, "middle": 4}


class RelMouse:
    """RELATIVE pointer: signed-byte deltas, 4-byte report [buttons, dx, dy, wheel].

    This is what the rig's gadget actually is. A relative mouse cannot be asked
    where the cursor is, so the CALLER dead-reckons (see hybrid_sender) and
    re-zeroes by slamming a screen corner, where the OS clamps and the position
    becomes knowable. Deltas are clipped to the signed-byte range the descriptor
    declares; the caller is expected to chunk larger moves.
    """

    def __init__(self, path=MOUSE_PATH):
        self.fd = open(path, "wb")
        self.buttons = 0

    def _send(self, dx=0, dy=0, wheel=0):
        self.fd.write(struct.pack("<Bbbb", self.buttons,
                                  max(-127, min(127, int(dx))),
                                  max(-127, min(127, int(dy))),
                                  max(-127, min(127, int(wheel)))))
        self.fd.flush()

    def move_rel(self, dx, dy):
        self._send(dx, dy)

    def click(self, btn, dx=0, dy=0):
        if dx or dy:
            self._send(dx, dy)
        self.buttons |= BTN.get(btn, 2)
        self._send()
        time.sleep(0.008)
        self.buttons &= ~BTN.get(btn, 2)
        self._send()

    def reset(self):
        self.buttons = 0
        self._send()


def _parse(line: str) -> dict | None:
    """Accept either a JSON command or the plain-text line protocol.

    Text forms (hybrid_sender.py): "press <k>", "release <k>", "tap <k>",
    "reset". Anything unrecognised returns None and is ignored rather than
    killing the connection -- a malformed line must never cost us the socket
    mid-game.
    """
    line = line.strip()
    if not line:
        return None
    if line[0] == "{":
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None
    parts = line.split()
    verb = parts[0].lower()
    if verb in ("press", "release") and len(parts) >= 2:
        return {"t": "key", "k": parts[1], "d": 1 if verb == "press" else 0}
    if verb == "tap" and len(parts) >= 2:
        return {"t": "tap", "k": parts[1]}
    if verb == "reset":
        return {"t": "reset"}
    # relative move: "mouse <dx> <dy>". The dead-reckoning sender emits this for
    # every chunk of a move; it is the highest-rate command on the wire.
    if verb == "mouse" and len(parts) >= 3:
        try:
            return {"t": "rel", "dx": int(float(parts[1])), "dy": int(float(parts[2]))}
        except ValueError:
            return None
    if verb == "click" and len(parts) >= 2:
        return {"t": "click", "b": parts[1]}
    return None


class AbsMouse:
    def __init__(self, path=MOUSE_PATH):
        self.fd = open(path, "wb")
        self.x = self.y = 16384
        self.buttons = 0

    def _send(self, wheel=0):
        self.fd.write(struct.pack("<BhhB", self.buttons, self.x, self.y, wheel & 0xFF))
        self.fd.flush()

    def move(self, x, y):
        self.x, self.y = max(0, min(32767, int(x))), max(0, min(32767, int(y)))
        self._send()

    def click(self, btn, x=None, y=None):
        if x is not None:
            self.move(x, y)
        self.buttons |= BTN.get(btn, 2)
        self._send()
        time.sleep(0.008)                      # ~real click duration
        self.buttons &= ~BTN.get(btn, 2)
        self._send()

    def reset(self):
        self.buttons = 0
        self._send()


def probe_mouse_mode():
    """-> "rel" | "abs" | None, read off the gadget's own report descriptor.

    The report LENGTH is baked into the descriptor by setup_hid_combo.sh (4 for
    relative, 6 for absolute) and a mismatched writer is a silent failure, so
    ask configfs rather than assume. Returns None if configfs isn't readable
    (not root, gadget made some other way) -- the caller then falls back to the
    hardware's actual mode, which is relative.
    """
    import glob
    for p in glob.glob("/sys/kernel/config/usb_gadget/*/functions/hid.usb1/report_length"):
        try:
            with open(p) as fh:
                n = int(fh.read().strip())
            return {4: "rel", 6: "abs"}.get(n)
        except (OSError, ValueError):
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mouse", choices=["auto", "rel", "abs", "off"], default="auto",
                    help="Pointer report format. auto=probe the gadget descriptor and "
                         "fall back to rel (what the rig's hardware is). Getting this "
                         "wrong makes every mouse command a silent no-op.")
    ap.add_argument("--port", type=int, default=9999)
    args = ap.parse_args()

    kb = PersistentKeyboard()
    # Mouse is optional: WASD keyboard-only inference has no /dev/hidg1.
    mode = args.mouse
    if mode == "auto":
        probed = probe_mouse_mode()
        mode = probed or "rel"
        print(f"mouse mode: {mode} ({'probed from report_desc' if probed else 'default; descriptor not readable'})")
    ms = None
    if mode != "off":
        try:
            ms = (RelMouse() if mode == "rel" else AbsMouse())
            print(f"mouse gadget: /dev/hidg1 ({'relative' if mode == 'rel' else 'absolute'})")
        except OSError as e:
            # FileNotFoundError = keyboard-only rig; PermissionError = gadget exists
            # but we aren't root. Either way, degrade to keyboard rather than dying
            # at startup -- the keyboard half is what a WASD run actually needs.
            print(f"mouse gadget: unavailable ({e.__class__.__name__}: {e}) "
                  "— keyboard-only / WASD mode")
    is_rel = isinstance(ms, RelMouse)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)
    print(f"hid_server: kb=/dev/hidg0 mouse={'/dev/hidg1 ' + mode if ms else 'NONE'}, "
          f"listening :{args.port}")
    while True:
        conn, addr = srv.accept()
        print(f"client {addr}")
        buf = b""
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    m = _parse(line.decode("utf-8", "replace"))
                    if m is None:
                        continue
                    t = m.get("t")
                    try:
                        if t == "tap":
                            kb.press(m["k"]); time.sleep(0.012); kb.release(m["k"])
                        elif t == "key":
                            (kb.press if m.get("d") else kb.release)(m["k"])
                        elif t == "click":
                            if ms is None:
                                print("DROPPED click: no mouse gadget")
                            elif is_rel:
                                # a relative mouse cannot be told WHERE to click;
                                # the sender has already walked the cursor there.
                                ms.click(m.get("b", "right"))
                            else:
                                ms.click(m.get("b", "right"), m.get("x"), m.get("y"))
                        elif t == "rel":
                            if ms is None:
                                print("DROPPED mouse move: no mouse gadget")
                            elif is_rel:
                                ms.move_rel(m["dx"], m["dy"])
                            else:
                                print("DROPPED 'mouse dx dy': gadget is ABSOLUTE but the "
                                      "sender speaks RELATIVE — re-run setup_hid_combo.sh "
                                      "(MOUSE_MODE=rel) or start with --mouse rel")
                        elif t == "move":
                            if ms is None:
                                print("DROPPED move: no mouse gadget")
                            elif is_rel:
                                # Loud, not silent: an absolute 'move' on a relative
                                # gadget is exactly the no-op that made a preflight
                                # pass on a rig that could not aim.
                                print("DROPPED absolute 'move': gadget is RELATIVE. "
                                      "Use hybrid_sender's corner-relative move_to/move_click.")
                            else:
                                ms.move(m["x"], m["y"])
                        elif t == "reset":
                            kb.pressed.clear(); kb.modifiers = 0; kb._send()
                            if ms:
                                ms.reset()
                    except (KeyError, TypeError, ValueError) as e:
                        # a malformed command must not cost us the socket
                        print(f"bad command {m!r}: {e}")
        except (ConnectionResetError, OSError) as e:
            print(f"client error: {e}")
        finally:
            # never leave inputs stuck when the agent dies mid-game
            kb.pressed.clear(); kb.modifiers = 0; kb._send()
            if ms:
                ms.reset()
            conn.close()
            print("client gone; inputs cleared")


if __name__ == "__main__":
    main()
