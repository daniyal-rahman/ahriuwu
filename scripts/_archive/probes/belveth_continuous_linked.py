"""Continuous-play linked-allocation snapshot.

Seeks once to gt=34, plays 20s. Every 400ms:
  1. Re-resolve focused-hero pointer (should be stable across play)
  2. Read Bel'Veth struct
  3. Dump every heap pointer inside the struct and read 16KB at each address
  4. Save all to C:\\tmp\\bv_linked\\

Then offline we find:
  - Pointers that are stable across all snapshots (same addresses)
  - Vec3 offsets within those allocations whose (x,z) matches destinations
    during each click window
"""
import ctypes, struct, subprocess, sys, json, time, os
from ctypes import wintypes
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

FOCUSED_HERO_PTR_RVA = 0x1E13490
POSITION             = 0x200
STRUCT_SIZE          = 128 * 1024
LINKED_SCAN_SIZE     = 32 * 1024
OUT_DIR              = r"C:\tmp\bv_linked"
SEEK_TO   = 34.0
DURATION  = 20.0
SNAP_EVERY = 0.4
os.makedirs(OUT_DIR, exist_ok=True)

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
    def __init__(self, pid): self.h = _k.OpenProcess(0x0410, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok else b""
    def u64(self, a): d = self.read(a, 8); return struct.unpack("<Q", d)[0] if len(d)==8 else None
    def vec3(self, a): d = self.read(a, 12); return struct.unpack("<fff", d) if len(d)==12 else (0,0,0)

import ssl, urllib.request
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def _post(obj):
    req = urllib.request.Request("https://127.0.0.1:2999/replay/playback",
            data=json.dumps(obj).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def _get():
    with urllib.request.urlopen("https://127.0.0.1:2999/replay/playback", context=_ctx, timeout=2) as r:
        return json.loads(r.read())

def main():
    for f in os.listdir(OUT_DIR):
        try: os.remove(os.path.join(OUT_DIR, f))
        except: pass
    pid = find_pid(); base, _ = find_base(pid); m = Mem(pid)
    print(f"PID={pid} base=0x{base:X}")
    _post({"time": SEEK_TO, "speed": 1.0, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = _get()
        if not st["seeking"] and st["paused"]: break
    print(f"at gt={st['time']:.2f}, unpausing")
    _post({"speed": 1.0, "paused": False})

    idx = 0; t_end = time.time() + DURATION; index = []
    while time.time() < t_end:
        t0 = time.time()
        hero = m.u64(base + FOCUSED_HERO_PTR_RVA)
        if not hero: time.sleep(SNAP_EVERY); idx += 1; continue
        px, py, pz = m.vec3(hero + POSITION)
        hero_raw = m.read(hero, STRUCT_SIZE)
        # pull heap pointers
        ptrs = set()
        for off in range(0, STRUCT_SIZE - 8, 8):
            v = struct.unpack_from("<Q", hero_raw, off)[0]
            # heap range on Win10/11: allocations typically in 0x200...000 region
            if 0x10000000000 < v < 0x7FF000000000 and v != hero:
                ptrs.add(v)
        # dump hero + each linked
        dir_ = os.path.join(OUT_DIR, f"s{idx:03d}")
        os.makedirs(dir_, exist_ok=True)
        with open(os.path.join(dir_, "hero.bin"), "wb") as f: f.write(hero_raw)
        linked_map = {}
        for p in ptrs:
            b = m.read(p, LINKED_SCAN_SIZE)
            if b:
                with open(os.path.join(dir_, f"p_{p:X}.bin"), "wb") as f: f.write(b)
                linked_map[hex(p)] = len(b)
        try:
            gt = _get()["time"]
        except: gt = None
        rec = {"idx": idx, "wall": round(t0,4), "gt": gt, "hero": hex(hero),
               "pos": [px, py, pz], "linked": linked_map}
        index.append(rec)
        if idx % 4 == 0:
            print(f"  [{idx}] gt={gt:.2f} pos=({px:.0f},{pz:.0f}) linked={len(linked_map)}")
        idx += 1
        sleep = SNAP_EVERY - (time.time() - t0)
        if sleep > 0: time.sleep(sleep)
    _post({"speed": 1.0, "paused": True})
    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nDone. {len(index)} snapshots @ {OUT_DIR}")

if __name__ == "__main__":
    main()
