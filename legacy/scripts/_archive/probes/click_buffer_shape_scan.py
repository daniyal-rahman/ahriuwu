"""Shape-only scan for click-destination buffer allocations.

Signature (established empirically from Bel'Veth, reproduced across runs):
  At some offset p in heap:
    Vec3 at p+0x000 == Vec3 at p+0x308 == Vec3 at p+0x374
    All three Vec3s are in map range (x, z ∈ (100, 16000); y ∈ (-300, 300))

Every hit is a candidate click-destination buffer. We then attribute each hit
to a hero by matching its held Vec3 to live hero positions, resolved by
byte-scanning for champion names in memory.

No click ground truth / no keylog / no arrivals used — pure structure scan.
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, json, time, ssl, urllib.request
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

STRIDE_B = 0x308     # offset to 2nd mirror from 1st
STRIDE_C = 0x374     # offset to 3rd mirror from 1st
CHAMP_OFF = 0x4360
POS_OFF   = 0x200
CHAMPS = ["Irelia", "Belveth", "Malzahar", "KaiSa", "Nautilus",
          "Illaoi", "Shyvana", "Anivia", "Ezreal", "Karma"]

_k = ctypes.windll.kernel32
class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong), ("__a", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong),
                ("__b", ctypes.c_ulong)]
_k.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
_k.VirtualQueryEx.restype = ctypes.c_size_t
MEM_COMMIT = 0x1000; MEM_PRIVATE = 0x20000
PAGE_RW = 0x04 | 0x08 | 0x40

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

def enum_rw(h):
    addr = 0; mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)): break
        base = mbi.BaseAddress or 0; size = mbi.RegionSize
        if (mbi.State == MEM_COMMIT and mbi.Type == MEM_PRIVATE and (mbi.Protect & PAGE_RW)):
            yield base, size
        addr = base + size
        if addr <= base: break

def read_region(h, base, size, chunk=4*1024*1024):
    if size > 512*1024*1024: return None
    out = bytearray(size); view = memoryview(out); off = 0
    while off < size:
        n = min(chunk, size - off)
        buf = (ctypes.c_char * n)()
        rd = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(h, ctypes.c_void_p(base + off), buf, n, ctypes.byref(rd))
        if not ok or rd.value == 0: return None if off == 0 else bytes(view[:off])
        view[off:off+rd.value] = buf[:rd.value]
        off += rd.value
    return bytes(out)

def read(h, a, sz):
    buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
    return buf.raw[:n.value] if ok else b""

def scan_shape(data, base_addr):
    """Yield (abs_addr, (x,y,z)) for positions where the 3-mirror pattern holds."""
    import numpy as np
    # Need data[p..p+0x374+12] accessible
    end = len(data) - (STRIDE_C + 12)
    if end <= 0: return
    # Read all floats as numpy
    floats = np.frombuffer(data, dtype=np.float32)
    # indices are 4-byte positions; stride in float units = byte/4
    B = STRIDE_B // 4   # 0x308/4 = 194
    C = STRIDE_C // 4   # 0x374/4 = 221
    # We need f[i], f[i+1], f[i+2] (Vec3 at p) == f[i+B], f[i+B+1], f[i+B+2] (at p+0x308) == f[i+C], f[i+C+1], f[i+C+2]
    max_i = len(floats) - (C + 3)
    if max_i <= 0: return
    # exact bit-equality is enough — Riot writes the same Vec3 to 3 slots
    x = floats[:max_i]
    y = floats[1:max_i+1]
    z = floats[2:max_i+2]
    xB = floats[B:B+max_i]; yB = floats[B+1:B+1+max_i]; zB = floats[B+2:B+2+max_i]
    xC = floats[C:C+max_i]; yC = floats[C+1:C+1+max_i]; zC = floats[C+2:C+2+max_i]
    # equality between Vec3 components (bit-equal)
    eq_B = (x == xB) & (y == yB) & (z == zB)
    eq_C = (x == xC) & (y == yC) & (z == zC)
    # map range filter
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    in_map = (x > 100) & (x < 16000) & (z > 100) & (z < 16000) & (y > -300) & (y < 300)
    mask = eq_B & eq_C & finite & in_map
    idx = np.nonzero(mask)[0]
    for i in idx:
        i = int(i)
        yield base_addr + i * 4, (float(x[i]), float(y[i]), float(z[i]))

def find_hero_positions(h, champs):
    """Byte-scan for each champion name, read position @+0x200 for each candidate,
    pick all candidates with x,z in map range (avoid 0,0 or 1,1 templates)."""
    out = {}  # champ -> list of (hero_addr, pos)
    pats = {c: (c + "\x00").encode() for c in champs}
    for base, size in enum_rw(h):
        if size > 256*1024*1024: continue
        data = read_region(h, base, size)
        if not data: continue
        for name, pat in pats.items():
            i = 0
            while True:
                j = data.find(pat, i)
                if j < 0: break
                i = j + 1
                hb = base + j - CHAMP_OFF
                if hb <= 0: continue
                p = read(h, hb + POS_OFF, 12)
                if len(p) != 12: continue
                x, y, z = struct.unpack("<fff", p)
                if 100 < x < 16000 and 100 < z < 16000 and -300 < y < 300:
                    out.setdefault(name, []).append((hb, (x, y, z)))
    return out

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def _post(o):
    req = urllib.request.Request("https://127.0.0.1:2999/replay/playback",
            data=json.dumps(o).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def _get():
    with urllib.request.urlopen("https://127.0.0.1:2999/replay/playback", context=_ctx, timeout=2) as r:
        return json.loads(r.read())

def main():
    pid = find_pid(); h = _k.OpenProcess(0x0410, False, pid)
    print(f"PID={pid}")

    # Ensure replay is paused for stable reads
    st = _get()
    if not st["paused"]:
        _post({"speed":1.0, "paused": True}); time.sleep(0.3)
    st = _get()
    print(f"Replay paused at gt={st['time']:.2f}")

    # Step 1: shape scan
    print("\nStep 1: shape scan for 3-mirror Vec3 pattern (offsets +0, +0x308, +0x374)")
    regions = list(enum_rw(h))
    total = sum(s for _, s in regions)
    print(f"  {len(regions)} regions, {total/(1024*1024):.0f} MB")
    hits = []
    t0 = time.time()
    for base, size in regions:
        data = read_region(h, base, size)
        if not data: continue
        for addr, vec in scan_shape(data, base):
            hits.append((addr, vec))
    print(f"  {len(hits)} hits in {time.time()-t0:.1f}s")

    # Step 2: find hero positions
    print("\nStep 2: find live hero positions")
    heroes = find_hero_positions(h, CHAMPS)
    # Dedupe per-champ: pick the candidate with position that's unique (not 0/1 template etc).
    # Just report all unique (position) values per champion.
    print("  Live heroes (position in map range):")
    live_pos = {}  # champ -> set of positions
    for name, cands in heroes.items():
        unique = {}
        for hb, pos in cands:
            unique.setdefault(pos, hb)
        live_pos[name] = unique
        for pos, hb in list(unique.items())[:3]:
            print(f"    {name:<12} 0x{hb:X} pos=({pos[0]:.0f},{pos[2]:.0f})")

    # Step 3: attribute each shape-scan hit to a champion by matching Vec3
    print(f"\nStep 3: attribute {len(hits)} shape hits to champions")
    attributed = {}  # champ -> list of (hit_addr, vec)
    unattributed = []
    for addr, vec in hits:
        x, _, z = vec
        matched = None
        for name, positions in live_pos.items():
            for pos in positions:
                if abs(x - pos[0]) < 30 and abs(z - pos[2]) < 30:
                    matched = name; break
            if matched: break
        if matched:
            attributed.setdefault(matched, []).append((addr, vec))
        else:
            unattributed.append((addr, vec))

    print("\n=== RESULTS ===")
    for name in CHAMPS:
        hs = attributed.get(name, [])
        if hs:
            print(f"\n  {name}: {len(hs)} click-buffer candidate(s)")
            for addr, vec in hs[:5]:
                print(f"    alloc_addr=0x{addr:X}  vec=({vec[0]:.0f},{vec[1]:.2f},{vec[2]:.0f})")
        else:
            print(f"\n  {name}: NO click-buffer found (or position unknown)")
    if unattributed:
        print(f"\n  Unattributed hits: {len(unattributed)}")
        for addr, vec in unattributed[:10]:
            print(f"    0x{addr:X}  ({vec[0]:.0f},{vec[1]:.2f},{vec[2]:.0f})")

    with open(r"C:\tmp\shape_scan_result.json", "w") as f:
        json.dump({
            "paused_gt": _get()["time"],
            "attributed": {n: [[hex(a), list(v)] for a, v in hs] for n, hs in attributed.items()},
            "unattributed": [[hex(a), list(v)] for a, v in unattributed],
        }, f, indent=2)
    print("\nSaved -> C:\\tmp\\shape_scan_result.json")

if __name__ == "__main__":
    main()
