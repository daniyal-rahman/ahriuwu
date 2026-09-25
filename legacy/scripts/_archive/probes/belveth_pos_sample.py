"""Sample Bel'Veth world position via the camera-locked 'focused hero' pointer
at module+0x1E13490, derived via find_belveth.py. Writes samples @ ~10Hz with
replay game_time to C:\\tmp\\belveth_samples.json.

Verified offsets for patch 16.8.766.8562:
  position      @ hero + 0x200
  champion_name @ hero + 0x4360
"""
import ctypes, struct, subprocess, sys, json, time, ssl, urllib.request
from ctypes import wintypes

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

FOCUSED_HERO_PTR_RVA = 0x1E13490
POSITION             = 0x200
CHAMPION_NAME        = 0x4360
SAMPLE_HZ            = 10
DURATION_S           = 25
OUT                  = r"C:\tmp\belveth_samples.json"

_k = ctypes.WinDLL("kernel32", use_last_error=True)
class ME32(ctypes.Structure):
    _fields_ = [("dwSize",wintypes.DWORD),("a",wintypes.DWORD),("pid",wintypes.DWORD),
                ("b",wintypes.DWORD),("c",wintypes.DWORD),
                ("modBase",ctypes.POINTER(ctypes.c_byte)),("modSize",wintypes.DWORD),
                ("hMod",wintypes.HMODULE),("szMod",ctypes.c_char*256),
                ("szPath",ctypes.c_char*260)]

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

def find_base(pid):
    snap = _k.CreateToolhelp32Snapshot(0x18, pid)
    me = ME32(); me.dwSize = ctypes.sizeof(ME32)
    if _k.Module32First(snap, ctypes.byref(me)):
        while True:
            if b'league' in me.szMod.lower():
                b = ctypes.cast(me.modBase, ctypes.c_void_p).value
                _k.CloseHandle(snap); return b, me.modSize
            if not _k.Module32Next(snap, ctypes.byref(me)): break
    _k.CloseHandle(snap); return None, None

class Mem:
    def __init__(self, pid):  self.h = _k.OpenProcess(0x0410, False, pid)
    def _r(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok else b""
    def u64(self, a):
        d = self._r(a, 8);  return struct.unpack("<Q", d)[0] if len(d)==8 else None
    def vec3(self, a):
        d = self._r(a, 12); return struct.unpack("<fff", d) if len(d)==12 else (0,0,0)
    def name(self, a):
        d = self._r(a, 32); return d.split(b"\x00",1)[0].decode("ascii", "replace")

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def replay_time():
    try:
        with urllib.request.urlopen("https://127.0.0.1:2999/replay/playback", context=_ctx, timeout=1) as r:
            return json.loads(r.read()).get("time")
    except Exception:
        return None

def main():
    pid = find_pid()
    base, size = find_base(pid)
    print(f"PID={pid} base=0x{base:X} size=0x{size:X}")
    m = Mem(pid)
    hero = m.u64(base + FOCUSED_HERO_PTR_RVA)
    if not hero:
        print("ERR: focused-hero pointer is null — is camera still locked on Bel'Veth?"); return
    name = m.name(hero + CHAMPION_NAME)
    x, y, z = m.vec3(hero + POSITION)
    print(f"Focused hero: 0x{hero:X}  name='{name}'  pos=({x:.0f},{y:.0f},{z:.0f})")
    if name.lower() != "belveth":
        print(f"WARN: focused hero is '{name}' not Bel'Veth — continuing anyway")

    samples = []
    dt = 1.0 / SAMPLE_HZ
    t_end = time.time() + DURATION_S
    while time.time() < t_end:
        t0 = time.time()
        gt = replay_time()
        # re-read the pointer in case struct is relocated between frames
        h = m.u64(base + FOCUSED_HERO_PTR_RVA) or hero
        x, y, z = m.vec3(h + POSITION)
        samples.append({"wall_time": round(t0, 4), "game_time": gt, "x": x, "y": y, "z": z})
        if len(samples) % 10 == 0:
            print(f"  gt={gt:.1f}s  pos=({x:.0f},{y:.0f},{z:.0f})  n={len(samples)}")
        sleep = dt - (time.time() - t0)
        if sleep > 0: time.sleep(sleep)

    with open(OUT, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} samples to {OUT}")

if __name__ == "__main__":
    main()
