"""Continuously snapshot Bel'Veth's 128KB struct while replay plays.

Seeks once to gt=34, plays at 1.0× for ~20s. Every 300ms records:
  (wall, game_t, hero_ptr, hero_pos, struct_bytes)

Saves a packed .bin per snapshot + an index.json for offline heap-diff.
"""
import ctypes, struct, subprocess, sys, json, time, ssl, urllib.request, os
from ctypes import wintypes
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

FOCUSED_HERO_PTR_RVA = 0x1E13490
POSITION             = 0x200
CHAMPION_NAME        = 0x4360
STRUCT_SIZE          = 128 * 1024
OUT_DIR              = r"C:\tmp\bv_snaps"
os.makedirs(OUT_DIR, exist_ok=True)

SEEK_TO   = 34.0
DURATION  = 20.0
SNAP_EVERY = 0.3

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
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok else b""
    def u64(self, a):  d = self.read(a, 8);  return struct.unpack("<Q", d)[0] if len(d)==8 else None
    def vec3(self, a): d = self.read(a, 12); return struct.unpack("<fff", d) if len(d)==12 else (0,0,0)
    def name(self, a): return self.read(a, 32).split(b"\x00",1)[0].decode("ascii", "replace")
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def _post(obj):
    req = urllib.request.Request("https://127.0.0.1:2999/replay/playback",
            data=json.dumps(obj).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def _get():
    with urllib.request.urlopen("https://127.0.0.1:2999/replay/playback", context=_ctx, timeout=2) as r:
        return json.loads(r.read())

def main():
    # clean out dir
    for f in os.listdir(OUT_DIR):
        try: os.remove(os.path.join(OUT_DIR, f))
        except: pass

    pid = find_pid(); base, sz = find_base(pid)
    print(f"PID={pid} base=0x{base:X}")
    m = Mem(pid)

    print(f"seek to {SEEK_TO}, pause")
    _post({"time": SEEK_TO, "speed": 1.0, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = _get()
        if not st["seeking"] and st["paused"]: break
    print(f"  at gt={st['time']:.2f}")

    print(f"unpausing, sampling every {SNAP_EVERY}s for {DURATION}s")
    _post({"speed": 1.0, "paused": False})
    t_end = time.time() + DURATION
    idx = 0
    index = []
    while time.time() < t_end:
        t0 = time.time()
        hero = m.u64(base + FOCUSED_HERO_PTR_RVA)
        if not hero:
            print(f"  [{idx}] hero ptr null, skip"); time.sleep(SNAP_EVERY); idx += 1; continue
        name = m.name(hero + CHAMPION_NAME)
        px, py, pz = m.vec3(hero + POSITION)
        raw = m.read(hero, STRUCT_SIZE)
        try:
            st = _get()
            gt = st["time"]
        except:
            gt = None
        path = os.path.join(OUT_DIR, f"snap_{idx:03d}.bin")
        with open(path, "wb") as f: f.write(raw)
        rec = {"idx": idx, "wall": round(t0, 4), "gt": gt, "hero": hex(hero),
               "name": name, "pos": [px, py, pz], "path": path, "size": len(raw)}
        index.append(rec)
        if idx % 4 == 0:
            print(f"  [{idx}] gt={gt:.2f} name={name} pos=({px:.0f},{py:.0f},{pz:.0f}) saved {len(raw)}B")
        idx += 1
        sleep = SNAP_EVERY - (time.time() - t0)
        if sleep > 0: time.sleep(sleep)

    _post({"speed": 1.0, "paused": True})
    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\nDone. {len(index)} snapshots @ {OUT_DIR}")

if __name__ == "__main__":
    main()
