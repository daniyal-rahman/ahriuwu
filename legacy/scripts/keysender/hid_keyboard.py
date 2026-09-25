#!/usr/bin/env python3
"""
Persistent Das Keyboard emulator.
Keeps /dev/hidg0 open forever and only sends reports when needed.
"""

import time
import struct
import threading
import socket
from collections import defaultdict

HID_PATH = "/dev/hidg0"

# Same KEY_MAP as before...
KEY_MAP = {
    'a': 0x04, 'b': 0x05, 'c': 0x06, 'd': 0x07, 'e': 0x08,
    'f': 0x09, 'g': 0x0A, 'h': 0x0B, 'i': 0x0C, 'j': 0x0D,
    'k': 0x0E, 'l': 0x0F, 'm': 0x10, 'n': 0x11, 'o': 0x12,
    'p': 0x13, 'q': 0x14, 'r': 0x15, 's': 0x16, 't': 0x17,
    'u': 0x18, 'v': 0x19, 'w': 0x1A, 'x': 0x1B, 'y': 0x1C,
    'z': 0x1D,
    '1': 0x1E, '2': 0x1F, '3': 0x20, '4': 0x21, '5': 0x22,
    '6': 0x23, '7': 0x24, '8': 0x25, '9': 0x26, '0': 0x27,
    'enter': 0x28, 'esc': 0x29, 'backspace': 0x2A, 'tab': 0x2B,
    'space': 0x2C,
    'lctrl': 0x01, 'lshift': 0x02, 'lalt': 0x04, 'lgui': 0x08,
    'rctrl': 0x10, 'rshift': 0x20, 'ralt': 0x40, 'rgui': 0x80,
}

class PersistentKeyboard:
    def __init__(self):
        self.fd = open(HID_PATH, "wb")
        self.pressed = set()
        self.modifiers = 0
        self.lock = threading.Lock()
        self.repeat_delay = 0.50
        self.repeat_rate = 0.033
        self._stop = threading.Event()
        self._repeat_thread = None

    def _send(self):
        keys = list(self.pressed)[:6] + [0] * 6
        report = struct.pack("BBBBBBBB",
                             self.modifiers, 0,
                             keys[0], keys[1], keys[2],
                             keys[3], keys[4], keys[5])
        self.fd.write(report)
        self.fd.flush()

    def press(self, name):
        with self.lock:
            if name in ('lctrl','rctrl','lshift','rshift','lalt','ralt','lgui','rgui'):
                self.modifiers |= KEY_MAP[name]
            else:
                code = KEY_MAP.get(name.lower())
                if code:
                    self.pressed.add(code)
            self._send()
            self._restart_repeat()

    def release(self, name):
        with self.lock:
            if name in ('lctrl','rctrl','lshift','rshift','lalt','ralt','lgui','rgui'):
                self.modifiers &= ~KEY_MAP[name]
            else:
                code = KEY_MAP.get(name.lower())
                if code:
                    self.pressed.discard(code)
            self._send()
            if not self.pressed and self.modifiers == 0:
                self._stop_repeat()

    def _restart_repeat(self):
        self._stop_repeat()
        if self.pressed:
            self._stop.clear()
            self._repeat_thread = threading.Thread(target=self._repeat_loop, daemon=True)
            self._repeat_thread.start()

    def _stop_repeat(self):
        self._stop.set()
        if self._repeat_thread:
            self._repeat_thread.join(timeout=0.1)

    def _repeat_loop(self):
        if self._stop.wait(self.repeat_delay):
            return
        while not self._stop.is_set():
            with self.lock:
                if self.pressed:
                    self._send()
            if self._stop.wait(self.repeat_rate):
                break

    def close(self):
        self._stop_repeat()
        # Send empty report on exit so no keys are left stuck
        self.pressed.clear()
        self.modifiers = 0
        self._send()
        self.fd.close()


# ========== Simple network listener example ==========
# You can feed it commands from Windows screen-share side or from local input

def main():
    kb = PersistentKeyboard()
    print("Persistent keyboard running. Keeping /dev/hidg0 open...")
    print("Send commands via TCP 9999 or just keep the process alive.")

    try:
        while True:
            time.sleep(1)   # just keep alive – replace with real input source
    except KeyboardInterrupt:
        print("Shutting down cleanly...")
        kb.close()

if __name__ == "__main__":
    main()
