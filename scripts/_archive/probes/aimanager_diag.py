"""Diagnostic: is the hero struct live in replay?
   Then: where in the hero struct does hero.pos appear as a Vec3 mirror?
   Any such mirror is a candidate for ServerPos inside AiManager (or inline).
"""
import ctypes, struct, subprocess, sys, json, time, math
from ctypes import wintypes
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

HERO_ARRAY_RVA = 0x1DD7128
POSITION       = 0x25C
CHAMPION_NAME  = 0x4360

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
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u64(self, a): d=self._r(a,8);  return struct.unpack("<Q",d)[0] if d else None
    def u32(self, a): d=self._r(a,4);  return struct.unpack("<I",d)[0] if d else None
    def i32(self, a): d=self._r(a,4);  return struct.unpack("<i",d)[0] if d else None
    def f32(self, a): d=self._r(a,4);  return struct.unpack("<f",d)[0] if d else None
    def vec3(self,a): d=self._r(a,12); return struct.unpack("<fff",d)  if d else None
    def string(self, a, n=64):
        d = self._r(a,n);  return d.split(b'\x00')[0].decode('ascii',errors='replace') if d else None
    def block(self, a, sz): return self._r(a, sz)

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF
def is_map(v):
    if v is None: return False
    x,y,z=v
    if abs(x)<0.1 and abs(z)<0.1: return False
    return -500<x<16000 and -500<y<1500 and -500<z<16000

def dist2d(a,b): return math.sqrt((a[0]-b[0])**2+(a[2]-b[2])**2)

def main():
    pid = find_pid();  base,_ = find_base(pid);  m = Mem(pid)
    print(f"PID={pid} base=0x{base:X}")
    arr = m.u64(base + HERO_ARRAY_RVA)
    heroes = []
    for i in range(10):
        hp = m.u64(arr + i*8)
        if not is_heap(hp): continue
        name = m.string(hp + CHAMPION_NAME, 40) or f"h{i}"
        pos  = m.vec3(hp + POSITION)
        heroes.append({"ptr": hp, "name": name, "pos": pos})
    print(f"Found {len(heroes)} heroes")

    # === Diagnostic 1: is hero.pos changing? ===
    print("\n--- DIAG 1: is hero.pos changing over 6 samples @ 0.5s? ---")
    series = {h["name"]:[] for h in heroes}
    for t in range(6):
        for h in heroes:
            series[h["name"]].append(m.vec3(h["ptr"] + POSITION))
        time.sleep(0.5)
    for name, s in series.items():
        pts = [p for p in s if p]
        if len(pts) < 2: continue
        moved = max(dist2d(pts[0], p) for p in pts[1:])
        print(f"  {name:10s} start={tuple(round(x,0) for x in pts[0])} max_move={moved:.1f}u over 3s")

    # === Diagnostic 2: find every Vec3 mirror of hero.pos INSIDE the hero struct ===
    # Freshly re-read pos + struct block back-to-back (hero moves ~400u/s, so
    # stale sampling kills everything). Pause the replay first for stability.
    print("\n--- DIAG 2: pausing replay to sample hero struct atomically ---")
    try:
        import urllib.request, ssl
        c = ssl.create_default_context(); c.check_hostname=False; c.verify_mode=ssl.CERT_NONE
        urllib.request.urlopen(urllib.request.Request(
            "https://127.0.0.1:2999/replay/playback", method="POST",
            data=b'{"speed":0.0,"paused":true}',
            headers={"Content-Type":"application/json"}), context=c, timeout=3).read()
        time.sleep(0.5)
    except Exception as e:
        print(f"  pause failed: {e}")

    jayce = heroes[0]
    hp_ptr = jayce["ptr"]
    hp_pos = m.vec3(hp_ptr + POSITION)   # FRESH
    print(f"  Jayce pos={hp_pos}  ptr=0x{hp_ptr:X}")

    HERO_SIZE = 0x20000
    block = m.block(hp_ptr, HERO_SIZE)
    if not block:
        print("  failed to read hero struct"); return

    # Inline mirrors
    inline_mirrors = []
    for off in range(0, HERO_SIZE - 12, 4):
        v = struct.unpack_from("<fff", block, off)
        if is_map(v) and dist2d(v, hp_pos) < 5:
            inline_mirrors.append((off, v))
    print(f"  inline Vec3 mirrors (match within 5u): {len(inline_mirrors)}")
    for off, v in inline_mirrors[:40]:
        print(f"    hero+0x{off:04X}  =  ({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})")
    if len(inline_mirrors) > 40:
        print(f"    ... +{len(inline_mirrors)-40} more")

    # 1-hop pointer deref: for each 8-aligned pointer in the hero struct,
    # read +0x474 (ServerPos candidate) and check for hero.pos match.
    # Also try +0x25C (maybe AiManager inherits GameObject layout).
    print("\n--- DIAG 3: 1-hop deref — any ptr in hero struct whose target contains hero.pos? ---")
    hits = []
    for off in range(0x100, HERO_SIZE - 8, 8):
        ptr = struct.unpack_from("<Q", block, off)[0]
        if not is_heap(ptr): continue
        # Probe a few candidate ServerPos offsets
        for probe_off, label in [(0x474,"+0x474"), (0x25C,"+0x25C"), (0x34,"+0x34"),
                                  (0x33C,"+0x33C"), (0x330,"+0x330"), (0x0,"+0x0"),
                                  (0x20,"+0x20"), (0x10,"+0x10")]:
            v = m.vec3(ptr + probe_off)
            if v and is_map(v) and dist2d(v, hp_pos) < 5:
                hits.append((off, ptr, probe_off, label, v))
                break
    print(f"  pointer-deref hits (ptr+X ≈ hero.pos): {len(hits)}")
    for off, ptr, po, lbl, v in hits[:60]:
        print(f"    hero+0x{off:04X} -> 0x{ptr:X} {lbl}  =  ({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})")
    if len(hits) > 60:
        print(f"    ... +{len(hits)-60} more")

    out = {
        "position_series": {n: [list(p) if p else None for p in s] for n,s in series.items()},
        "inline_mirrors":  [{"off": f"0x{o:X}", "vec": list(v)} for o,v in inline_mirrors],
        "deref_hits":      [{"off": f"0x{o:X}", "ptr": f"0x{p:X}", "probe_off": lbl, "vec": list(v)}
                            for o,p,_,lbl,v in hits],
    }
    with open(r"C:\Users\daniz\aimanager_diag_out.json","w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote C:\\Users\\daniz\\aimanager_diag_out.json")

if __name__ == "__main__":
    main()
