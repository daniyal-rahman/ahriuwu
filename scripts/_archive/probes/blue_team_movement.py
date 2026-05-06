"""Find live-updating hero structs for blue-team teammates (Irelia, Malzahar,
KaiSa, Nautilus) by:
  1. Scanning process memory for each champion name
  2. For each candidate, reading position TWICE with a play step between
  3. Keeping candidates whose position CHANGES (live) vs STATIC (template)

Then sample trajectory for 20s starting at gt=34 and find arrivals.
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, json, time, ssl, urllib.request
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

CHAMP_OFF = 0x4360
POS_OFF   = 0x200
BLUE_TEAM = ["Belveth", "Irelia", "Malzahar", "KaiSa", "Nautilus"]

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
def read(h, a, sz):
    buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
    return buf.raw[:n.value] if ok else b""
def vec3(h, a):
    d = read(h, a, 12); return struct.unpack("<fff", d) if len(d)==12 else (0,0,0)

def enum_rw(h):
    addr = 0; mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)): break
        base = mbi.BaseAddress or 0; size = mbi.RegionSize
        if (mbi.State == MEM_COMMIT and mbi.Type == MEM_PRIVATE and (mbi.Protect & PAGE_RW)):
            yield base, size
        addr = base + size
        if addr <= base: break

def find_candidates(h, names):
    """Byte-scan for each name. Return {name: [hero_base_candidate_addrs]}."""
    out = {name: [] for name in names}
    pats = {name: (name + "\x00").encode() for name in names}
    for base, size in enum_rw(h):
        if size > 256*1024*1024: continue
        data = read(h, base, size)
        if not data: continue
        for name, pat in pats.items():
            start = 0
            while True:
                i = data.find(pat, start)
                if i < 0: break
                start = i + 1
                hb = base + i - CHAMP_OFF
                if hb > 0:
                    out[name].append(hb)
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

    # Seek to 34 paused
    _post({"time": 34.0, "speed": 1.0, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = _get()
        if not st["seeking"] and st["paused"]: break
    # warmup
    _post({"speed":1.0, "paused": False}); time.sleep(0.4); _post({"speed":1.0, "paused": True})

    print("Finding hero-struct candidates by name...")
    cands = find_candidates(h, BLUE_TEAM)
    for n, cs in cands.items():
        print(f"  {n}: {len(cs)} candidates")

    # Read position ONCE while paused at 34
    print("\nPhase 1: pos @ gt=34 (paused)")
    pos_t1 = {name: {c: vec3(h, c + POS_OFF) for c in cs} for name, cs in cands.items()}
    # Advance play to gt=40
    _post({"speed":1.0, "paused": False})
    while True:
        time.sleep(0.1)
        st = _get()
        if st["time"] >= 40.0: break
    _post({"speed":1.0, "paused": True}); time.sleep(0.3)
    print("Phase 2: pos @ gt=40 (paused)")
    pos_t2 = {name: {c: vec3(h, c + POS_OFF) for c in cs} for name, cs in cands.items()}

    # For each hero, find candidates whose pos changed between t1 and t2 (live update)
    print("\nLive (position-changing) candidates per blue-team hero:")
    live_picks = {}
    for name in BLUE_TEAM:
        live = []
        for c in cands[name]:
            p1 = pos_t1[name][c]
            p2 = pos_t2[name][c]
            # require both positions in map (not 0,0,0 or 1,1,1)
            def map_valid(p):
                return (200 < p[0] < 16000) and (200 < p[2] < 16000) and (-300 < p[1] < 300)
            if map_valid(p1) and map_valid(p2):
                moved = abs(p1[0]-p2[0]) + abs(p1[2]-p2[2])
                live.append((c, p1, p2, moved))
        live.sort(key=lambda x: -x[3])  # largest movement first
        print(f"\n  {name}:")
        for c, p1, p2, moved in live[:5]:
            print(f"    0x{c:X}  gt34=({p1[0]:.0f},{p1[2]:.0f}) gt40=({p2[0]:.0f},{p2[2]:.0f})  moved={moved:.0f}")
        if live:
            live_picks[name] = live[0][0]  # top candidate

    # For chosen live heroes, sample trajectory for ~20s
    if not live_picks:
        print("\nNo live-updating heroes found beyond Bel'Veth — replay vision too restrictive at this time.")
        return
    print(f"\nSampling live heroes: {list(live_picks.keys())}")
    _post({"time": 34.0, "speed": 1.0, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = _get()
        if not st["seeking"] and st["paused"]: break
    _post({"speed":1.0, "paused": False})
    series = {name: [] for name in live_picks}
    t_end = time.time() + 20.0
    while time.time() < t_end:
        t0 = time.time()
        try: gt = _get()["time"]
        except: gt = None
        for name, hb in live_picks.items():
            x, y, z = vec3(h, hb + POS_OFF)
            series[name].append({"gt": gt, "x": x, "y": y, "z": z})
        sl = 0.1 - (time.time() - t0)
        if sl > 0: time.sleep(sl)
    _post({"speed":1.0, "paused": True})

    # Detect arrivals (≥4 stationary samples)
    def arrivals(samps):
        out = []; i = 0
        while i < len(samps) - 1:
            # advance while moving
            j = i
            while j + 1 < len(samps) and abs(samps[j+1]['x']-samps[j]['x']) + abs(samps[j+1]['z']-samps[j]['z']) > 10:
                j += 1
            # measure stationary run
            k = j
            while k + 1 < len(samps) and abs(samps[k+1]['x']-samps[j]['x']) + abs(samps[k+1]['z']-samps[j]['z']) < 30:
                k += 1
            if k - j >= 4 and j >= 1:  # stationary after some motion
                out.append({"gt_in": samps[j]['gt'], "gt_out": samps[k]['gt'],
                            "x": samps[j]['x'], "z": samps[j]['z']})
            i = max(k + 1, j + 1)
        return out

    print("\nArrivals:")
    for name in series:
        arrs = arrivals(series[name])
        if arrs:
            print(f"  {name}: {len(arrs)} arrivals — " + "; ".join(f"gt={a['gt_in']:.1f}→{a['gt_out']:.1f} ({a['x']:.0f},{a['z']:.0f})" for a in arrs[:5]))
        else:
            print(f"  {name}: no arrivals (stayed moving or stayed put)")
    with open(r"C:\tmp\blue_team.json", "w") as f:
        json.dump({"series": series, "picks": {n: hex(h) for n, h in live_picks.items()}}, f, indent=2)

if __name__ == "__main__":
    main()
