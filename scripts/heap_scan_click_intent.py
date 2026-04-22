"""Broad heap scan searching for Vec3 fields with click-intent signature.

Look for patterns:
  - Vec3 that matches hero position at some game time
  - Then changes discretely (step) at another time
  - Consistent with a click destination that "jumps" vs. smoothly updating

Runs against paused replay (via Replay API) to get atomic samples.
"""
import ctypes
import struct
import subprocess
import sys
import json
import time
import math
import urllib.request
import ssl
from ctypes import wintypes

# Signature battery: look for Vec3s that match these patterns
# 1. Vec3 ≈ hero position at t0
# 2. Vec3 changes to new location at t1
# 3. Change is discrete (large jump, not gradual drift)

def is_map(v):
    """Is this a valid map coordinate (Vec3)?"""
    if v is None or len(v) != 3:
        return False
    x, y, z = v
    if abs(x) < 0.1 and abs(z) < 0.1:
        return False
    return -500 < x < 16000 and -500 < y < 1500 and -500 < z < 16000

def dist2d(a, b):
    """2D distance in x,z plane."""
    if not a or not b:
        return None
    return math.sqrt((a[0] - b[0])**2 + (a[2] - b[2])**2)

def find_pid():
    """Find League of Legends process ID."""
    r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq League of Legends.exe', '/FO', 'CSV', '/NH'],
                      capture_output=True, text=True)
    for line in r.stdout.strip().split('\n'):
        if 'league' in line.lower():
            return int(line.strip('"').split('","')[1])
    return None

def find_base(pid):
    """Find League of Legends module base."""
    _k = ctypes.WinDLL("kernel32", use_last_error=True)

    class ME32(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("a", wintypes.DWORD), ("pid", wintypes.DWORD),
                    ("b", wintypes.DWORD), ("c", wintypes.DWORD),
                    ("modBase", ctypes.POINTER(ctypes.c_byte)), ("modSize", wintypes.DWORD),
                    ("hMod", wintypes.HMODULE), ("szMod", ctypes.c_char*256),
                    ("szPath", ctypes.c_char*260)]

    snap = _k.CreateToolhelp32Snapshot(0x18, pid)
    me = ME32()
    me.dwSize = ctypes.sizeof(ME32)
    if _k.Module32First(snap, ctypes.byref(me)):
        while True:
            if b'league' in me.szMod.lower():
                b = ctypes.cast(me.modBase, ctypes.c_void_p).value
                _k.CloseHandle(snap)
                return b, me.modSize
            if not _k.Module32Next(snap, ctypes.byref(me)):
                break
    _k.CloseHandle(snap)
    return None, None

class Mem:
    """Windows memory reader."""
    def __init__(self, pid):
        _k = ctypes.WinDLL("kernel32", use_last_error=True)
        self.h = _k.OpenProcess(0x0410, False, pid)
        self._k = _k

    def _r(self, a, sz):
        buf = ctypes.create_string_buffer(sz)
        n = ctypes.c_size_t(0)
        ok = self._k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None

    def vec3(self, a):
        d = self._r(a, 12)
        return struct.unpack("<fff", d) if d else None

    def block(self, a, sz):
        return self._r(a, sz)

    def close(self):
        self._k.CloseHandle(self.h)

def pause_replay():
    """Pause replay for atomic sampling."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        urllib.request.urlopen(urllib.request.Request(
            "https://127.0.0.1:2999/replay/playback",
            method="POST",
            data=json.dumps({"speed": 0.0, "paused": True}).encode(),
            headers={"Content-Type": "application/json"}), context=ctx, timeout=3).read()
        time.sleep(0.5)
        return True
    except Exception as e:
        print(f"Pause failed: {e}")
        return False

def scan_heap_vec3s(mem, base, msize, target_pos, threshold=50):
    """Scan committed memory for Vec3s that match target position.

    Returns list of (address, vec3_value) tuples.
    """
    hits = []
    chunk_size = 0x100000  # 1MB chunks
    scan_addr = base

    print(f"Scanning heap from 0x{base:X} for Vec3s within {threshold}u of target...")
    while scan_addr < base + msize:
        try:
            chunk = mem.block(scan_addr, chunk_size)
            if not chunk:
                scan_addr += chunk_size
                continue

            for off in range(0, len(chunk) - 12, 4):
                try:
                    v = struct.unpack_from("<fff", chunk, off)
                    if is_map(v) and dist2d(v, target_pos) < threshold:
                        hits.append((scan_addr + off, v))
                except:
                    pass

            scan_addr += chunk_size
            if scan_addr % 0x1000000 == 0:
                print(f"  ... scanned to 0x{scan_addr:X}")
        except:
            scan_addr += chunk_size

    return hits

def main():
    print("=" * 70)
    print("Heap Vec3 Scanner: Looking for Click-Intent Signatures")
    print("=" * 70)

    pid = find_pid()
    if not pid:
        print("ERROR: League not running")
        return

    base, msize = find_base(pid)
    if not base:
        print("ERROR: Could not find League module base")
        return

    print(f"PID={pid} base=0x{base:X} size=0x{msize:X}\n")

    mem = Mem(pid)

    # Pause replay
    if not pause_replay():
        print("ERROR: Could not pause replay")
        return

    # For now, just demonstrate the scanning infrastructure
    # TODO: integrate with hero position reading to actually scan
    print("\nHeap scan infrastructure ready.")
    print("TODO: Integrate with hero position reading to scan for Vec3s matching/changing position.")
    print("\nTo use this scanner:")
    print("1. Pause the replay at a known hero position")
    print("2. Call scan_heap_vec3s(mem, base, msize, hero_pos, threshold=50)")
    print("3. Time-sample each hit to detect discrete jumps (click signature)")
    print("4. Report candidates with step-change pattern\n")

    mem.close()

if __name__ == "__main__":
    main()
