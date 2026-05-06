"""Heap-diff probe to find AI-waypoint / click-destination field in memory.

Snapshots Bel'Veth's 128KB hero struct at 4 replay game_times, then finds
4-byte-aligned offsets where the Vec3 (x, z) at that offset matches the
EXPECTED click destination for that snapshot:

    snapshot 0: gt=37.0  — idle, no recent click     → no constraint (record)
    snapshot 1: gt=40.0  — moving to (3124, 8122)    → Vec3.x∈[3100,3160], z∈[8100,8160]
    snapshot 2: gt=44.8  — moving to (3736, 8358)    → Vec3.x∈[3710,3770], z∈[8340,8400]
    snapshot 3: gt=49.2  — moving to (4398, 8444)    → Vec3.x∈[4370,4430], z∈[8420,8480]

Probe also re-resolves the focused-hero pointer each snapshot since the
struct is reallocated on replay seek. Uses /replay/playback to drive
seek+pause; snapshot is taken while paused to avoid mid-write torn reads.
Dumps raw struct bytes per snapshot to C:\\tmp\\bvheap\\S{i}.bin for
offline analysis, plus a ranked candidate list.
"""
import ctypes, struct, subprocess, sys, json, time, ssl, urllib.request, os
from ctypes import wintypes
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

FOCUSED_HERO_PTR_RVA = 0x1E13490
POSITION             = 0x200
CHAMPION_NAME        = 0x4360
STRUCT_SIZE          = 128 * 1024
OUT_DIR              = r"C:\tmp\bvheap"
os.makedirs(OUT_DIR, exist_ok=True)

# Probe points: (game_time, label, expected_x, expected_z) or None if no constraint
PROBES = [
    (37.0, "idle_before_click1", None, None),
    (40.0, "moving_to_click1",  3124, 8122),
    (44.8, "moving_to_click2",  3736, 8358),
    (49.2, "moving_to_click3",  4398, 8444),
]
XZ_TOL = 30.0  # units

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
def _post_playback(obj):
    req = urllib.request.Request("https://127.0.0.1:2999/replay/playback",
            data=json.dumps(obj).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def _get_playback():
    with urllib.request.urlopen("https://127.0.0.1:2999/replay/playback", context=_ctx, timeout=2) as r:
        return json.loads(r.read())

def seek_and_pause(gt):
    _post_playback({"time": gt, "speed": 1.0, "paused": True})
    for _ in range(40):  # up to 20s
        time.sleep(0.5)
        st = _get_playback()
        if not st.get("seeking") and st.get("paused") and abs(st.get("time", 0) - gt) < 0.5:
            return st
    return _get_playback()

def main():
    pid = find_pid()
    base, size = find_base(pid)
    print(f"PID={pid} base=0x{base:X} size=0x{size:X}")
    m = Mem(pid)

    snaps = []   # list of (label, gt_actual, hero_addr, raw_bytes, exp_xz)
    for gt, label, ex, ez in PROBES:
        print(f"\n--- snap {label}: seeking to gt={gt} ---")
        seek_and_pause(gt)
        # Briefly unpause to let interpolator settle (100ms)
        _post_playback({"speed": 1.0, "paused": False})
        time.sleep(0.25)
        _post_playback({"speed": 1.0, "paused": True})
        time.sleep(0.3)
        hero = m.u64(base + FOCUSED_HERO_PTR_RVA)
        if not hero:
            print("  ERR: focused hero pointer null"); return
        name = m.name(hero + CHAMPION_NAME)
        px, py, pz = m.vec3(hero + POSITION)
        st = _get_playback()
        print(f"  hero=0x{hero:X} name='{name}' pos=({px:.0f},{py:.0f},{pz:.0f}) gt={st['time']:.2f}")
        raw = m.read(hero, STRUCT_SIZE)
        path = os.path.join(OUT_DIR, f"S{len(snaps)}_{label}.bin")
        with open(path, "wb") as f: f.write(raw)
        print(f"  dumped {len(raw)} bytes -> {path}")
        snaps.append((label, st['time'], hero, raw, (ex, ez), (px, py, pz)))

    # Diff: for each 4-byte-aligned offset, try (x=f32 @ off, z=f32 @ off+8) as a Vec3,
    # and check if snapshot 1/2/3 match their expected (x, z) within tolerance.
    def read_xz(buf, off):
        if off + 12 > len(buf): return None
        x, y, z = struct.unpack_from("<fff", buf, off)
        return (x, y, z)

    # Build a list of candidates that pass all 3 constrained snapshots
    def match(v, ex, ez):
        if v is None: return False
        x, y, z = v
        # reject NaN / inf / obvious garbage
        for c in (x, y, z):
            if c != c or abs(c) > 1e7: return False
        return abs(x - ex) < XZ_TOL and abs(z - ez) < XZ_TOL

    constrained = [s for s in snaps if s[4][0] is not None]
    print(f"\nScanning {STRUCT_SIZE} bytes @ 4-byte alignment for Vec3 matching all {len(constrained)} constraints...")
    candidates = []
    for off in range(0, STRUCT_SIZE - 12, 4):
        if all(match(read_xz(s[3], off), s[4][0], s[4][1]) for s in constrained):
            candidates.append(off)
    print(f"{len(candidates)} candidate offsets passed all 3 constraints.")
    for off in candidates[:40]:
        vals = []
        for s in snaps:
            v = read_xz(s[3], off)
            vals.append(f"({v[0]:.0f},{v[1]:.0f},{v[2]:.0f})" if v else "-")
        print(f"  hero+0x{off:04X}  " + "  ".join(vals))

    # Also: relaxed — show candidates that match ANY 2 of 3 (possible truncated stores)
    if not candidates:
        print("\nNo 3/3 matches. Listing 2/3 matches (may reveal partial writes):")
        partial = []
        for off in range(0, STRUCT_SIZE - 12, 4):
            hits = sum(1 for s in constrained if match(read_xz(s[3], off), s[4][0], s[4][1]))
            if hits >= 2:
                partial.append((hits, off))
        partial.sort(reverse=True)
        for hits, off in partial[:40]:
            vals = [f"({read_xz(s[3], off)[0]:.0f},{read_xz(s[3], off)[2]:.0f})" for s in snaps if read_xz(s[3], off)]
            print(f"  hero+0x{off:04X}  hits={hits}/3  " + "  ".join(vals))

    # Write candidate report
    report = {"snaps": [{"label": s[0], "gt": s[1], "hero_addr": hex(s[2]),
                          "pos": s[5], "expected_xz": s[4]} for s in snaps],
              "candidates": [{"offset": hex(off), "values": [read_xz(s[3], off) for s in snaps]}
                             for off in candidates]}
    with open(os.path.join(OUT_DIR, "diff_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {OUT_DIR}\\diff_report.json")

if __name__ == "__main__":
    main()
