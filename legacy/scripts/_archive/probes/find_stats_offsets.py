"""
Find memory offsets for stats that have API ground truth:
  level, cs, kills, deaths, assists

Strategy: cross-champion consistency.
  At the correct offset, every champion's memory value must match their
  ground-truth value from /liveclientdata/allgamedata.

For each u32 offset in [0, 0x10000):
    if all(hero_blob[i].u32(off) == gt[i][field] for i in 0..9):
        offset is a candidate for `field`

Items, summoners, HP, gold use pointer indirection and are handled separately.
"""
import base64, ctypes, ctypes.wintypes as wt, json, ssl
import struct, subprocess, sys, time, urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

EXPECTED_MOD_SIZE = 0x207C000
HERO_ARRAY_RVA = 0x1DBEEE8
CHAMPION_NAME_OFF = 0x4328
LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"
REPLAY_ID = "5528069928"
BLOB_SIZE = 0x20000  # 128KB per hero

_ctx = ssl.create_default_context()
_ctx.check_hostname = False; _ctx.verify_mode = ssl.CERT_NONE
_k = ctypes.windll.kernel32


def replay_get(ep):
    with urllib.request.urlopen(urllib.request.Request(f"https://127.0.0.1:2999{ep}"), context=_ctx, timeout=5) as r:
        return json.loads(r.read())

def replay_post(ep, d):
    with urllib.request.urlopen(urllib.request.Request(f"https://127.0.0.1:2999{ep}",
        data=json.dumps(d).encode(), headers={"Content-Type": "application/json"}), context=_ctx, timeout=5) as r:
        return json.loads(r.read() or b"null")

def lcu_post(ep, body=None):
    parts = open(LOCKFILE).read().strip().split(":")
    port = parts[2]
    auth = f"Basic {base64.b64encode(f'riot:{parts[3]}'.encode()).decode()}"
    req = urllib.request.Request(f"https://127.0.0.1:{port}{ep}", method="POST",
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": auth, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=_ctx, timeout=15) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


class Mem:
    def __init__(self, pid):
        self.h = _k.OpenProcess(0x0410, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u64(self, a):
        d = self.read(a, 8); return struct.unpack('<Q', d)[0] if d else None
    def string(self, a, n=64):
        d = self.read(a, n)
        if not d: return None
        t = d.split(b'\x00')[0]
        try: return t.decode('ascii')
        except: return None


def find_pid():
    r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq League of Legends.exe', '/FO', 'CSV', '/NH'],
                       capture_output=True, text=True)
    for line in r.stdout.strip().split('\n'):
        if 'league' in line.lower():
            return int(line.strip('"').split('","')[1])
    return None

def find_base(pid):
    class ME(ctypes.Structure):
        _fields_=[("dwSize",ctypes.c_ulong),("th32ModuleID",ctypes.c_ulong),
            ("th32ProcessID",ctypes.c_ulong),("GlblcntUsage",ctypes.c_ulong),
            ("ProccntUsage",ctypes.c_ulong),("modBaseAddr",ctypes.POINTER(ctypes.c_byte)),
            ("modBaseSize",ctypes.c_ulong),("hModule",ctypes.c_void_p),
            ("szModule",ctypes.c_char*256),("szExePath",ctypes.c_char*260)]
    snap = _k.CreateToolhelp32Snapshot(0x18, pid)
    me = ME(); me.dwSize = ctypes.sizeof(ME)
    if _k.Module32First(snap, ctypes.byref(me)):
        while True:
            if b'league' in me.szModule.lower():
                base = ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value
                size = me.modBaseSize
                _k.CloseHandle(snap)
                return base, size
            if not _k.Module32Next(snap, ctypes.byref(me)): break
    _k.CloseHandle(snap)
    return None, None


def init_heroes(m, base):
    arr_ptr = m.u64(base + HERO_ARRAY_RVA)
    heroes = {}
    for i in range(10):
        hp = m.u64(arr_ptr + i * 8)
        if not hp or hp < 0x10000: continue
        name = m.string(hp + CHAMPION_NAME_OFF)
        if name and len(name) >= 2 and name[0].isalpha():
            heroes[name] = hp
    return heroes


def read_blob(m, ptr, size=BLOB_SIZE):
    """Read up to `size` bytes, falling back through smaller sizes + chunks."""
    for s in (size, 0x8000, 0x6000, 0x5000):
        b = m.read(ptr, s)
        if b: return b
    out = b""
    for off in range(0, size, 0x1000):
        c = m.read(ptr + off, 0x1000)
        if not c: break
        out += c
    return out


def play_until(target_gt, max_wall=300):
    try: replay_post("/replay/playback", {"speed": 2.0, "paused": False})
    except: pass
    t0 = time.time()
    while time.time() - t0 < max_wall:
        try:
            gt = replay_get("/liveclientdata/gamestats").get("gameTime", 0)
            if gt >= target_gt: break
        except: pass
        time.sleep(1.0)
    try: replay_post("/replay/playback", {"speed": 0.0, "paused": True})
    except: pass
    time.sleep(1.0)


def find_offsets_by_consistency(blobs_by_champ, gt_values, dtype='u32'):
    """Find offsets where for all champions, the value at offset matches
    ground truth. Returns sorted list of candidate offsets."""
    candidates = []
    # Get the first blob length as reference
    any_blob = next(iter(blobs_by_champ.values()))
    n = len(any_blob)
    step = 4 if dtype in ('u32', 'i32', 'f32') else 8
    fmt = {'u32': '<I', 'i32': '<i', 'f32': '<f', 'u64': '<Q'}[dtype]
    tol = 0.01 if dtype == 'f32' else 0

    for off in range(0, n - step + 1, 4):
        match = True
        for mk, blob in blobs_by_champ.items():
            if mk not in gt_values: continue
            target = gt_values[mk]
            try:
                val = struct.unpack_from(fmt, blob, off)[0]
            except struct.error:
                match = False; break
            if dtype == 'f32':
                if abs(val - target) > max(tol, abs(target) * 0.001):
                    match = False; break
            else:
                if val != target:
                    match = False; break
        if match:
            candidates.append(off)
    return candidates


def main():
    print("=" * 60)
    print("Cross-champion consistency scanner")
    print("=" * 60)

    # Launch replay
    try:
        lcu_post(f"/lol-replays/v1/rofls/{REPLAY_ID}/watch", {"componentType": "string"})
    except Exception as e:
        print(f"launch: {e}")
    for i in range(90):
        try:
            if replay_get("/liveclientdata/gamestats").get("gameTime", 0) > 5:
                break
        except: pass
        time.sleep(2)
    print("[loaded]")

    pid = find_pid()
    base, size = find_base(pid)
    print(f"[mem] pid={pid} base=0x{base:x} size=0x{size:x}")
    m = Mem(pid)
    heroes = init_heroes(m, base)
    print(f"[mem] heroes: {list(heroes.keys())}")

    # Advance so champs have non-zero CS / KDA / level
    print("[play] advancing to gt=300s...")
    play_until(300.0, max_wall=200)

    # Get ground truth
    data = replay_get("/liveclientdata/allgamedata")
    api_champs = {}
    for p in data.get("allPlayers", []):
        name = p.get("championName", "").replace(" ", "").replace("'", "").replace(".", "")
        sc = p.get("scores", {})
        api_champs[name] = {
            "level": int(p.get("level", 0)),
            "cs": int(sc.get("creepScore", 0)),
            "kills": int(sc.get("kills", 0)),
            "deaths": int(sc.get("deaths", 0)),
            "assists": int(sc.get("assists", 0)),
            "ward_score_f32": float(sc.get("wardScore", 0)),
        }
    # Map memory keys to api keys
    def match_key(mk):
        for ak in api_champs:
            if ak.lower() == mk.lower(): return ak
        return None

    gt_level = {}
    gt_cs = {}
    gt_k = {}
    gt_d = {}
    gt_a = {}
    gt_ward = {}
    for mk in heroes:
        ak = match_key(mk)
        if not ak:
            print(f"[warn] no api match for {mk}")
            continue
        v = api_champs[ak]
        gt_level[mk] = v["level"]
        gt_cs[mk] = v["cs"]
        gt_k[mk] = v["kills"]
        gt_d[mk] = v["deaths"]
        gt_a[mk] = v["assists"]
        gt_ward[mk] = v["ward_score_f32"]

    print(f"\n[gt] level:   {gt_level}")
    print(f"[gt] cs:      {gt_cs}")
    print(f"[gt] kills:   {gt_k}")
    print(f"[gt] deaths:  {gt_d}")
    print(f"[gt] assists: {gt_a}")
    print(f"[gt] ward:    {gt_ward}")

    # Read blobs
    print("\n[read] hero blobs...")
    blobs = {}
    for mk in heroes:
        b = read_blob(m, heroes[mk], BLOB_SIZE)
        if b:
            blobs[mk] = b
            print(f"  {mk}: {len(b)} bytes")
        else:
            print(f"  {mk}: FAIL")

    # Find offsets
    print("\n" + "=" * 60)
    print("Scanning for consistent offsets")
    print("=" * 60)

    fields_u32 = [
        ("level",   gt_level),
        ("cs",      gt_cs),
        ("kills",   gt_k),
        ("deaths",  gt_d),
        ("assists", gt_a),
    ]
    for name, gt in fields_u32:
        t0 = time.time()
        cands = find_offsets_by_consistency(blobs, gt, 'u32')
        print(f"  {name:8s}: {len(cands):4d} candidate(s) in {time.time()-t0:.1f}s", end="")
        if cands:
            print(f"  → {[hex(c) for c in cands[:20]]}")
        else:
            print()

    # CS as f32
    gt_cs_f32 = {mk: float(v) for mk, v in gt_cs.items()}
    t0 = time.time()
    cands = find_offsets_by_consistency(blobs, gt_cs_f32, 'f32')
    print(f"  cs_f32  : {len(cands):4d} candidate(s) in {time.time()-t0:.1f}s", end="")
    if cands:
        print(f"  → {[hex(c) for c in cands[:20]]}")
    else:
        print()

    # Deaths as u32 — may be filtered by zero matches. Try with exclusion of zero-death champs
    nonzero_deaths = {mk: v for mk, v in gt_d.items() if v > 0}
    if nonzero_deaths:
        t0 = time.time()
        cands = find_offsets_by_consistency({mk: blobs[mk] for mk in nonzero_deaths if mk in blobs},
                                            nonzero_deaths, 'u32')
        print(f"  deaths*: {len(cands):4d} candidate(s) in {time.time()-t0:.1f}s (nonzero champs only)", end="")
        if cands:
            print(f"  → {[hex(c) for c in cands[:20]]}")
        else:
            print()

    # Neighborhood around kills
    kills_off = 0x5428
    print(f"\n[neighborhood] u32 values near kills (0x{kills_off:X}):")
    for mk in list(blobs.keys())[:5]:
        vals = []
        for off in range(kills_off - 0x20, kills_off + 0x80, 4):
            try:
                v = struct.unpack_from('<I', blobs[mk], off)[0]
                vals.append(f"+{off-kills_off:+3d}:{v}")
            except: pass
        print(f"  {mk:12s}: {' '.join(vals[:15])}")

    # Ward f32: cross-reference with ward scores that are distinct
    # Filter out offsets where all champs have 0 (noise)
    distinct_ward = {mk: v for mk, v in gt_ward.items() if v > 0.01}
    if distinct_ward:
        t0 = time.time()
        cands = find_offsets_by_consistency({mk: blobs[mk] for mk in distinct_ward if mk in blobs},
                                            distinct_ward, 'f32')
        print(f"\n  ward* : {len(cands):4d} candidate(s) (nonzero only)")
        if cands:
            print(f"   → {[hex(c) for c in cands[:20]]}")

    # ── Item scan: cross-champion match on first item slot ──
    print(f"\n{'='*60}\nITEM SCAN: first-slot item via pointer deref")
    print("=" * 60)
    first_items = {}
    for p in data.get("allPlayers", []):
        name = p.get("championName", "").replace(" ", "").replace("'", "").replace(".", "")
        its = p.get("items", [])
        if its:
            its = sorted(its, key=lambda x: x.get("slot", 99))
            first_items[name] = int(its[0].get("itemID", 0))
    # Map to mem keys
    first_items_mem = {}
    for mk in heroes:
        ak = match_key(mk)
        if ak and ak in first_items:
            first_items_mem[mk] = first_items[ak]
    print(f"  first item per champ: {first_items_mem}")

    # For each hero offset, deref and check if struct contains first item ID at same sub-offset
    # We iterate in sync: at offset X, hero_ptr + X is a u64 pointer, deref it,
    # then find a sub-offset Y such that deref[Y:Y+4] == first_items[mk] for all champs
    if len(first_items_mem) >= 5:
        any_blob = next(iter(blobs.values()))
        n = len(any_blob)
        print(f"  scanning {n//8} qword positions...")
        t0 = time.time()
        # Precompute deref'd structs per champ for every qword in the blob
        # That's 16K * 10 = 160K ReadProcessMemory calls. Expensive but finite.
        # Instead, we iterate per-offset and read only if needed
        item_cands = []
        n_checked = 0
        for off in range(0, n - 8, 8):
            # First champ check — must point to valid memory
            mk0 = next(iter(first_items_mem))
            p0 = struct.unpack_from('<Q', blobs[mk0], off)[0]
            if not (0x10000 < p0 < 0x7FFFFFFFFFFF):
                continue
            # Read 0x80 from first champ's deref
            sub0 = m.read(p0, 0x80)
            if not sub0:
                continue
            n_checked += 1
            # Find candidate sub_offsets where sub0 has mk0's first item
            target0 = first_items_mem[mk0]
            for so in range(0, len(sub0) - 4, 4):
                v = struct.unpack_from('<I', sub0, so)[0]
                if v != target0: continue
                # Found a match at (off, so) for mk0. Check other champs.
                all_match = True
                for mk in first_items_mem:
                    if mk == mk0: continue
                    p = struct.unpack_from('<Q', blobs[mk], off)[0]
                    if not (0x10000 < p < 0x7FFFFFFFFFFF):
                        all_match = False; break
                    sub = m.read(p + so, 4)
                    if not sub:
                        all_match = False; break
                    vi = struct.unpack_from('<I', sub, 0)[0]
                    if vi != first_items_mem[mk]:
                        all_match = False; break
                if all_match:
                    item_cands.append((off, so))
        print(f"  checked {n_checked} pointers in {time.time()-t0:.1f}s")
        print(f"  item candidates (hero_off, sub_off): {item_cands[:20]}")
    else:
        print(f"  too few champs with items: {len(first_items_mem)}")

    print("\n[done]")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(2)
