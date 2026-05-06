"""Follow pointers inside Bel'Veth's hero struct and scan linked allocations
for a Vec3 that matches the expected click destinations at the right times.

Uses the snapshots already on disk at C:\\tmp\\bv_snaps\\snap_NNN.bin but
additionally, at SAMPLE_INDICES below, re-opens the live process and for
each 8-aligned u64 inside the hero struct that looks like a heap pointer,
reads 16KB at that address and scans it.
"""
import ctypes, struct, subprocess, sys, json, time, os
from ctypes import wintypes
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

FOCUSED_HERO_PTR_RVA = 0x1E13490
POSITION             = 0x200
STRUCT_SIZE          = 128 * 1024
LINKED_SCAN_SIZE     = 16 * 1024  # bytes per linked allocation
# Probe times and expected destinations (x, z) from the continuous sample
PROBES = [
    (40.0, None,       None),  # transit — no constraint (baseline)
    (41.5, 3124, 8122),        # arrived at click 1 dest
    (46.0, 3736, 8358),        # arrived at click 2 dest
    (49.5, 4398, 8444),        # mid-move to click 3 dest
]
XZ_TOL = 50.0

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
    def u64(self, a): d = self.read(a, 8);  return struct.unpack("<Q", d)[0] if len(d)==8 else None
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

def seek_then_play_to(target_gt):
    """Seek to a bit before target_gt, unpause, wait till we're near target."""
    pre = max(target_gt - 2.0, 0.5)
    _post({"time": pre, "speed": 1.0, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = _get()
        if not st["seeking"] and st["paused"]: break
    _post({"speed": 1.0, "paused": False})
    for _ in range(40):
        time.sleep(0.1)
        st = _get()
        if st["time"] >= target_gt:
            _post({"speed": 1.0, "paused": True})
            return st
    return _get()

def snap_linked(m, hero_ptr):
    """Return dict {linked_addr: bytes} for every distinct heap pointer in the hero struct."""
    hero_raw = m.read(hero_ptr, STRUCT_SIZE)
    ptrs = set()
    for off in range(0, STRUCT_SIZE - 8, 8):
        v = struct.unpack_from("<Q", hero_raw, off)[0]
        # heap range filter: skip nulls, modules (<0x7FF000000000), and garbage
        if 0x100000000000 < v < 0x7FF000000000:
            ptrs.add(v)
    print(f"  {len(ptrs)} distinct heap pointers")
    dumps = {}
    for p in list(ptrs):
        b = m.read(p, LINKED_SCAN_SIZE)
        if b: dumps[p] = b
    return hero_raw, dumps

def main():
    pid = find_pid(); base, _ = find_base(pid)
    m = Mem(pid)

    snapshots = []  # list of (gt, expected_xz, hero_ptr, hero_raw, linked_dict)
    for gt, ex, ez in PROBES:
        print(f"\n--- seek+play to gt≈{gt} ---")
        seek_then_play_to(gt)
        hero = m.u64(base + FOCUSED_HERO_PTR_RVA)
        px, py, pz = m.vec3(hero + POSITION)
        st = _get()
        print(f"  gt={st['time']:.2f} hero=0x{hero:X} pos=({px:.0f},{py:.0f},{pz:.0f})  exp={ex},{ez}")
        hero_raw, linked = snap_linked(m, hero)
        snapshots.append((st['time'], (ex, ez), hero, hero_raw, linked))

    # For each linked pointer that is present in ALL snapshots (same address) AND
    # contains a Vec3 at some offset matching expected xz in every constrained snapshot:
    common_ptrs = set(snapshots[0][4].keys())
    for _, _, _, _, d in snapshots[1:]:
        common_ptrs &= set(d.keys())
    print(f"\n{len(common_ptrs)} heap pointers present in all {len(snapshots)} snapshots")

    constrained_idx = [i for i, s in enumerate(snapshots) if s[1][0] is not None]
    print(f"Constrained snapshots: {constrained_idx}")

    def read_xyz(buf, off):
        if off + 12 > len(buf): return None
        return struct.unpack_from("<fff", buf, off)
    def match(v, ex, ez):
        if v is None: return False
        x, y, z = v
        if x != x or z != z or abs(x) > 1e7 or abs(z) > 1e7: return False
        return abs(x - ex) < XZ_TOL and abs(z - ez) < XZ_TOL

    hits = []  # list of (ptr, offset_in_alloc, values_across_snaps)
    for p in common_ptrs:
        bufs = [s[4][p] for s in snapshots]
        L = min(len(b) for b in bufs)
        for off in range(0, L - 12, 4):
            ok = True
            for i in constrained_idx:
                v = read_xyz(bufs[i], off)
                ex, ez = snapshots[i][1]
                if not match(v, ex, ez):
                    ok = False; break
            if ok:
                vals = [read_xyz(b, off) for b in bufs]
                hits.append((p, off, vals))
    print(f"\n{len(hits)} hits in linked memory (Vec3 matching destination in all constrained snaps)")
    for p, off, vals in hits[:40]:
        vstr = "  ".join(f"({v[0]:.0f},{v[1]:.0f},{v[2]:.0f})" if v else "-" for v in vals)
        print(f"  alloc 0x{p:X}+0x{off:X}  [{vstr}]")

    # Also: hero_struct itself (included for completeness)
    print("\nSearching hero struct Vec3 offsets:")
    hero_hits = []
    for off in range(0, STRUCT_SIZE - 12, 4):
        ok = True
        for i in constrained_idx:
            v = read_xyz(snapshots[i][3], off)
            ex, ez = snapshots[i][1]
            if not match(v, ex, ez):
                ok = False; break
        if ok:
            hero_hits.append(off)
    print(f"  {len(hero_hits)} hero-struct hits (should include position mirrors)")

    report = {"hits_linked": [{"alloc": hex(p), "off": hex(off),
                               "vals": vals} for p, off, vals in hits],
              "hits_hero_struct": [hex(o) for o in hero_hits],
              "snapshots": [{"gt": s[0], "exp": s[1], "hero": hex(s[2])} for s in snapshots]}
    with open(r"C:\tmp\bv_linked_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nReport: C:\\tmp\\bv_linked_report.json")

if __name__ == "__main__":
    main()
