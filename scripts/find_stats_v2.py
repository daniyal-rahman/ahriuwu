"""
Comprehensive stats offset scanner — finds deaths, CS, HP, gold, items.

Strategies:
  deaths/CS: cross-champion consistency with multiple type encodings (u32/i32/f32/u16/u8)
  HP: plausible-range scan + snapshot differencing (HP changes over time)
  Gold: plausible-range + monotonic-increase over snapshots
  Items: 3-level pointer chain scan (hero → Inventory → Slot → Item → itemID)
"""
import base64, ctypes, ctypes.wintypes as wt, json, ssl
import struct, subprocess, sys, time, urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

HERO_ARRAY_RVA = 0x1DBEEE8
CHAMPION_NAME_OFF = 0x4328
LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"
REPLAY_ID = "5528069928"
BLOB_SIZE = 0x20000

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
_k = ctypes.windll.kernel32


def rget(ep):
    r = urllib.request.urlopen(urllib.request.Request(f"https://127.0.0.1:2999{ep}"), context=_ctx, timeout=5)
    return json.loads(r.read())
def rpost(ep, d):
    r = urllib.request.urlopen(urllib.request.Request(f"https://127.0.0.1:2999{ep}",
        data=json.dumps(d).encode(), headers={"Content-Type":"application/json"}), context=_ctx, timeout=5)
    return json.loads(r.read() or b"null")
def lcu_launch():
    parts = open(LOCKFILE).read().strip().split(":")
    req = urllib.request.Request(f"https://127.0.0.1:{parts[2]}/lol-replays/v1/rofls/{REPLAY_ID}/watch",
        method="POST", data=json.dumps({"componentType":"string"}).encode(),
        headers={"Authorization": f"Basic {base64.b64encode(f'riot:{parts[3]}'.encode()).decode()}",
                 "Content-Type":"application/json"})
    urllib.request.urlopen(req, context=_ctx, timeout=10)


class Mem:
    def __init__(self, pid): self.h = _k.OpenProcess(0x0410, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u64(self, a): d=self.read(a,8); return struct.unpack('<Q',d)[0] if d else None
    def u32(self, a): d=self.read(a,4); return struct.unpack('<I',d)[0] if d else None
    def f32(self, a): d=self.read(a,4); return struct.unpack('<f',d)[0] if d else None
    def string(self, a, n=64):
        d=self.read(a,n)
        if not d: return None
        try: return d.split(b'\x00')[0].decode('ascii')
        except: return None


def find_pid():
    r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq League of Legends.exe', '/FO', 'CSV', '/NH'],
                       capture_output=True, text=True)
    for line in r.stdout.strip().split('\n'):
        if 'league' in line.lower(): return int(line.strip('"').split('","')[1])
    return None

def find_base(pid):
    class ME(ctypes.Structure):
        _fields_=[("dwSize",ctypes.c_ulong),("th32ModuleID",ctypes.c_ulong),("th32ProcessID",ctypes.c_ulong),
            ("GlblcntUsage",ctypes.c_ulong),("ProccntUsage",ctypes.c_ulong),("modBaseAddr",ctypes.POINTER(ctypes.c_byte)),
            ("modBaseSize",ctypes.c_ulong),("hModule",ctypes.c_void_p),("szModule",ctypes.c_char*256),("szExePath",ctypes.c_char*260)]
    snap=_k.CreateToolhelp32Snapshot(0x18,pid); me=ME(); me.dwSize=ctypes.sizeof(ME)
    if _k.Module32First(snap, ctypes.byref(me)):
        while True:
            if b'league' in me.szModule.lower():
                base = ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value
                _k.CloseHandle(snap); return base
            if not _k.Module32Next(snap, ctypes.byref(me)): break
    return None


def read_blob(m, ptr, size=BLOB_SIZE):
    out = b""
    for off in range(0, size, 0x1000):
        c = m.read(ptr + off, 0x1000)
        if not c: break
        out += c
    return out


def play_until(target_gt, max_wall=300):
    try: rpost("/replay/playback", {"speed": 2.0, "paused": False})
    except: pass
    t0 = time.time()
    while time.time() - t0 < max_wall:
        try:
            gt = rget("/liveclientdata/gamestats").get("gameTime", 0)
            if gt >= target_gt: break
        except: pass
        time.sleep(1.0)
    try: rpost("/replay/playback", {"speed": 0.0, "paused": True})
    except: pass
    time.sleep(1.0)


def scan_consistent(blobs, gt_vals, dtype):
    """Find offsets where all champs have their gt_vals."""
    step_fmt = {'u32':('<I',4),'i32':('<i',4),'f32':('<f',4),'u16':('<H',2),'u8':('<B',1),'u64':('<Q',8)}
    fmt, step = step_fmt[dtype]
    any_blob = next(iter(blobs.values()))
    n = len(any_blob)
    cands = []
    for off in range(0, n - step + 1, step):
        ok = True
        for mk, blob in blobs.items():
            if mk not in gt_vals: continue
            try: v = struct.unpack_from(fmt, blob, off)[0]
            except: ok = False; break
            tgt = gt_vals[mk]
            if dtype == 'f32':
                if abs(v - tgt) > max(0.01, abs(tgt) * 0.001):
                    ok = False; break
            else:
                if v != tgt:
                    ok = False; break
        if ok: cands.append(off)
    return cands


def scan_plausible_f32(blobs, low, high):
    """Find offsets where all champs have distinct f32 values in [low, high]."""
    any_blob = next(iter(blobs.values()))
    n = len(any_blob)
    cands = []
    for off in range(0, n - 4, 4):
        vals = []
        ok = True
        for mk, blob in blobs.items():
            try: v = struct.unpack_from('<f', blob, off)[0]
            except: ok = False; break
            if not (low <= v <= high):
                ok = False; break
            if v != v:  # NaN
                ok = False; break
            vals.append(v)
        if ok and len(set(round(v, 1) for v in vals)) >= 3:
            cands.append((off, vals))
    return cands


def main():
    print("=" * 60)
    print("Comprehensive stats scanner")
    print("=" * 60)

    lcu_launch()
    for i in range(90):
        try:
            if rget("/liveclientdata/gamestats").get("gameTime", 0) > 5: break
        except: pass
        time.sleep(2)
    print("[loaded]")

    pid = find_pid()
    base = find_base(pid)
    m = Mem(pid)
    arr_ptr = m.u64(base + HERO_ARRAY_RVA)
    heroes = []
    for i in range(10):
        hp = m.u64(arr_ptr + i * 8)
        if hp:
            name = m.string(hp + CHAMPION_NAME_OFF)
            if name: heroes.append((name, hp))
    print(f"[mem] heroes: {[h[0] for h in heroes]}")

    # Play to gt=300, read snapshot A
    print("[play] advancing to gt=300...")
    play_until(300.0)
    data_a = rget("/liveclientdata/allgamedata")
    print("[read] snapshot A")
    blobs_a = {}
    for name, hp in heroes:
        b = read_blob(m, hp, BLOB_SIZE)
        if len(b) >= 0x4000:
            blobs_a[name] = b
    print(f"  {len(blobs_a)} blobs, min_len=0x{min(len(b) for b in blobs_a.values()):X}")

    # Play to gt=450, read snapshot B
    print("\n[play] advancing to gt=450...")
    play_until(450.0)
    data_b = rget("/liveclientdata/allgamedata")
    print("[read] snapshot B")
    blobs_b = {}
    for name, hp in heroes:
        b = read_blob(m, hp, BLOB_SIZE)
        if len(b) >= 0x4000:
            blobs_b[name] = b

    # Truncate blobs to common min length
    min_len = min(min(len(b) for b in blobs_a.values()), min(len(b) for b in blobs_b.values()))
    blobs_a = {k: v[:min_len] for k, v in blobs_a.items()}
    blobs_b = {k: v[:min_len] for k, v in blobs_b.items()}
    print(f"[truncated] blobs to 0x{min_len:X} bytes each")

    def gt_from_api(data, field):
        result = {}
        for p in data.get("allPlayers", []):
            name = p["championName"].replace(" ", "").replace("'", "").replace(".", "")
            if field in ("kills", "deaths", "assists", "cs"):
                sc = p.get("scores", {})
                k = {"cs":"creepScore"}.get(field, field)
                result[name] = int(sc.get(k, 0))
            elif field == "level":
                result[name] = int(p.get("level", 0))
        # map to mem keys
        mem_result = {}
        for name, _ in heroes:
            for ak in result:
                if ak.lower() == name.lower():
                    mem_result[name] = result[ak]
                    break
        return mem_result

    # ── STATIC SCANS (snapshot A) ──
    print("\n" + "=" * 60)
    print("Snapshot A scans (deaths, cs)")
    print("=" * 60)

    gt_deaths = gt_from_api(data_a, "deaths")
    gt_cs = gt_from_api(data_a, "cs")
    gt_kills = gt_from_api(data_a, "kills")
    gt_assists = gt_from_api(data_a, "assists")
    gt_level = gt_from_api(data_a, "level")

    print(f"  deaths gt: {gt_deaths}")
    print(f"  cs gt:     {gt_cs}")
    print(f"  level gt:  {gt_level}")

    # Sanity: confirm known offsets still work
    sanity = {"level": (0x4D10, gt_level, 'u32'),
              "kills": (0x5428, gt_kills, 'u32'),
              "assists": (0x5488, gt_assists, 'u32')}
    for n, (off, gt, dt) in sanity.items():
        fmt = {'u32':'<I'}[dt]
        ok_count = 0
        for mk, blob in blobs_a.items():
            if mk not in gt: continue
            v = struct.unpack_from(fmt, blob, off)[0]
            if v == gt[mk]: ok_count += 1
        print(f"  sanity {n} @ 0x{off:X}: {ok_count}/{len(gt)} match")

    # Deaths — multiple encodings
    for dtype in ('u32', 'i32', 'u16', 'u8'):
        cands = scan_consistent(blobs_a, gt_deaths, dtype)
        print(f"  deaths [{dtype}]: {len(cands)} cand{' → ' + str([hex(c) for c in cands[:10]]) if cands else ''}")

    # CS — multiple encodings
    for dtype in ('u32', 'i32', 'f32', 'u16'):
        cands = scan_consistent(blobs_a, gt_cs, dtype)
        print(f"  cs     [{dtype}]: {len(cands)} cand{' → ' + str([hex(c) for c in cands[:10]]) if cands else ''}")

    # Also cs as f32 excluding zero-cs champs
    cs_nonzero = {mk: float(v) for mk, v in gt_cs.items() if v > 0}
    cs_blobs_nonzero = {mk: blobs_a[mk] for mk in cs_nonzero if mk in blobs_a}
    cands = scan_consistent(cs_blobs_nonzero, cs_nonzero, 'f32')
    print(f"  cs [f32 nonzero-only]: {len(cands)} cand" + (f" → {[hex(c) for c in cands[:10]]}" if cands else ""))

    # ── HP: plausible range ──
    print("\n" + "=" * 60)
    print("HP scan (plausible range + change-over-time)")
    print("=" * 60)

    # Candidates: f32 in [400, 2500], all champs distinct
    print("  finding f32 in plausible HP range [400, 2500]...")
    t0 = time.time()
    hp_plausible = scan_plausible_f32(blobs_a, 400.0, 2500.0)
    print(f"  {len(hp_plausible)} plausible offsets in {time.time()-t0:.1f}s")

    # Narrow: offsets where values at snapshot B differ from A (or are the same if no combat)
    # AND also satisfies: max_hp at hero+4 or hero+8 >= current_hp
    hp_candidates = []
    for off, vals_a in hp_plausible[:5000]:  # cap
        # Check snapshot B for same offset — value should be similar range
        ok = True
        any_change = False
        for mk in blobs_a:
            va = struct.unpack_from('<f', blobs_a[mk], off)[0]
            vb = struct.unpack_from('<f', blobs_b[mk], off)[0]
            if not (400 <= vb <= 2500):
                ok = False; break
            if abs(vb - va) > 0.1:
                any_change = True
        if ok and any_change:
            hp_candidates.append(off)
    print(f"  narrowed by snapshot B plausibility + any change: {len(hp_candidates)}")

    # Show top 30 with change magnitude
    for off in hp_candidates[:30]:
        delta = []
        for mk in list(blobs_a.keys())[:5]:
            va = struct.unpack_from('<f', blobs_a[mk], off)[0]
            vb = struct.unpack_from('<f', blobs_b[mk], off)[0]
            delta.append(f"{mk[:4]}:{va:.0f}→{vb:.0f}")
        print(f"    0x{off:04X}: {delta}")

    # ── Gold: plausible range + monotonic increase ──
    print("\n" + "=" * 60)
    print("Gold scan (plausible range + monotonic increase)")
    print("=" * 60)

    print("  finding f32 in plausible gold range [400, 5000]...")
    t0 = time.time()
    gold_plausible = scan_plausible_f32(blobs_a, 400.0, 5000.0)
    print(f"  {len(gold_plausible)} plausible offsets in {time.time()-t0:.1f}s")

    # Narrow: in snapshot B, value should be >= snapshot A for all champs (monotonic)
    # AND increase should be reasonable (5-500 over 150s of game time)
    gold_candidates = []
    for off, vals_a in gold_plausible[:5000]:
        ok = True
        n_increased = 0
        for mk in blobs_a:
            va = struct.unpack_from('<f', blobs_a[mk], off)[0]
            vb = struct.unpack_from('<f', blobs_b[mk], off)[0]
            if vb < va - 1:  # strictly decreasing = not gold
                ok = False; break
            if not (400 <= vb <= 8000):
                ok = False; break
            if vb - va > 1:
                n_increased += 1
        if ok and n_increased >= 5:  # most champs should earn gold
            gold_candidates.append(off)
    print(f"  narrowed by monotonic-increase: {len(gold_candidates)}")
    for off in gold_candidates[:30]:
        delta = []
        for mk in list(blobs_a.keys())[:5]:
            va = struct.unpack_from('<f', blobs_a[mk], off)[0]
            vb = struct.unpack_from('<f', blobs_b[mk], off)[0]
            delta.append(f"{mk[:4]}:{va:.0f}→{vb:.0f}")
        print(f"    0x{off:04X}: {delta}")

    # Try gold as u32 too
    print("\n  gold as u32 (same method):")
    u32_gold = []
    any_blob = next(iter(blobs_a.values()))
    for off in range(0, len(any_blob) - 4, 4):
        ok = True
        n_inc = 0
        va_list = []
        for mk in blobs_a:
            va = struct.unpack_from('<I', blobs_a[mk], off)[0]
            vb = struct.unpack_from('<I', blobs_b[mk], off)[0]
            if not (400 <= va <= 20000) or not (400 <= vb <= 20000):
                ok = False; break
            if vb < va:
                ok = False; break
            if vb - va > 0 and vb - va < 2000:
                n_inc += 1
            va_list.append(va)
        if ok and n_inc >= 5 and len(set(va_list)) >= 4:
            u32_gold.append(off)
    print(f"  {len(u32_gold)} u32 gold candidates")
    for off in u32_gold[:30]:
        delta = []
        for mk in list(blobs_a.keys())[:5]:
            va = struct.unpack_from('<I', blobs_a[mk], off)[0]
            vb = struct.unpack_from('<I', blobs_b[mk], off)[0]
            delta.append(f"{mk[:4]}:{va}→{vb}")
        print(f"    0x{off:04X}: {delta}")

    print("\n[done]")
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except Exception:
        import traceback; traceback.print_exc()
        sys.exit(2)
