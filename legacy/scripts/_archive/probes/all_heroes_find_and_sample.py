"""Find all 10 hero structs in process memory by byte-searching for their
champion names, then sample each one's position during a forward play of the
replay. Detect arrival (stop) points per hero.

Champion internal names strip apostrophes: "Belveth", "KaiSa".
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, json, time, ssl, urllib.request
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

CHAMPION_NAME_OFFSET = 0x4360
POSITION_OFFSET      = 0x200
CHAMPS = ["Irelia", "Belveth", "Malzahar", "KaiSa", "Nautilus",
          "Illaoi", "Shyvana", "Anivia", "Ezreal", "Karma"]

SEEK_TO   = 34.0
DURATION  = 20.0
SAMPLE_HZ = 10

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

def enum_rw_private(h):
    addr = 0; mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)): break
        base = mbi.BaseAddress or 0; size = mbi.RegionSize
        if (mbi.State == MEM_COMMIT and mbi.Type == MEM_PRIVATE and (mbi.Protect & PAGE_RW)):
            yield base, size
        addr = base + size
        if addr <= base: break

def find_hero_structs(h, champs):
    """Byte-scan heap regions. For each occurrence of a champion name preceded
    by exactly 0x4360 bytes we consider the start of that block the hero struct.

    Instead of requiring a known struct base, scan for the name bytes and pick
    candidates where position @+0x200 (= name_addr - 0x4360 + 0x200) looks like
    a valid map coord, then dedupe per champion.
    """
    found = {}  # name -> list of candidate hero_base addresses
    patterns = {name: (name + "\x00").encode() for name in champs}
    for base, size in enum_rw_private(h):
        if size > 256*1024*1024: continue
        data = read(h, base, size)
        if not data: continue
        for name, pat in patterns.items():
            off = 0
            while True:
                i = data.find(pat, off)
                if i < 0: break
                off = i + 1
                # candidate hero base
                hb = base + i - CHAMPION_NAME_OFFSET
                if hb < 0: continue
                # validate: read +0x200 as Vec3, check (x,z) in map range [0, 16000], y in [-300, 300]
                x, y, z = vec3(h, hb + POSITION_OFFSET)
                if 0 < x < 16000 and 0 < z < 16000 and -300 < y < 300:
                    found.setdefault(name, set()).add(hb)
    return found

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

    # Seek, pause, then scan for hero structs
    print(f"\nSeeking to gt={SEEK_TO} paused")
    _post({"time": SEEK_TO, "speed": 1.0, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = _get()
        if not st["seeking"] and st["paused"]: break
    # Let replay warm up slightly
    _post({"speed":1.0, "paused": False}); time.sleep(0.4); _post({"speed":1.0, "paused": True})

    print("\nScanning process for hero structs by champion name...")
    t0 = time.time()
    found = find_hero_structs(h, CHAMPS)
    print(f"Scan done in {time.time()-t0:.1f}s")

    # Dedupe: pick the candidate per champion whose position varies least from frame to frame
    # (the network-snapshot buffer copies tend to lag slightly)
    hero_of = {}  # name -> chosen hero struct addr
    for name in CHAMPS:
        cands = found.get(name, set())
        if not cands:
            print(f"  {name}: NOT FOUND")
            continue
        # If multiple, pick the one with the lowest address (typically the canonical allocation)
        print(f"  {name}: {len(cands)} candidates")
        # Validate each by reading position and showing; pick first
        for c in list(cands)[:5]:
            x, y, z = vec3(h, c + POSITION_OFFSET)
            print(f"    0x{c:X} pos=({x:.0f},{y:.0f},{z:.0f})")
        hero_of[name] = min(cands)

    # Continuously sample all chosen hero positions during play
    print(f"\nSampling {len(hero_of)} heroes for {DURATION}s starting at gt={SEEK_TO}")
    _post({"speed":1.0, "paused": False})
    series = {name: [] for name in hero_of}
    dt = 1.0 / SAMPLE_HZ
    t_end = time.time() + DURATION
    while time.time() < t_end:
        t0 = time.time()
        try: gt = _get()["time"]
        except: gt = None
        for name, hb in hero_of.items():
            x, y, z = vec3(h, hb + POSITION_OFFSET)
            series[name].append({"gt": gt, "wall": round(t0,4), "x": x, "y": y, "z": z})
        sl = dt - (time.time() - t0)
        if sl > 0: time.sleep(sl)
    _post({"speed":1.0, "paused": True})

    # Detect arrivals per hero: find runs of >= 4 consecutive samples where position changes by < 10u
    def arrivals(samps):
        out = []
        i = 0
        while i < len(samps):
            # skip until motion stops
            j = i
            while j + 1 < len(samps):
                d = abs(samps[j+1]['x'] - samps[j]['x']) + abs(samps[j+1]['z'] - samps[j]['z'])
                if d < 10: break
                j += 1
            # find run length of stationary
            k = j
            while k + 1 < len(samps):
                d = abs(samps[k+1]['x'] - samps[j]['x']) + abs(samps[k+1]['z'] - samps[j]['z'])
                if d >= 30: break
                k += 1
            if k - j >= 4:  # >=4 stationary samples = ~0.4s = arrival
                out.append({"arrive_gt": samps[j]['gt'], "leave_gt": samps[k]['gt'],
                            "x": samps[j]['x'], "z": samps[j]['z']})
                i = k + 1
            else:
                i = j + 1
        return out

    out_path = r"C:\tmp\all_heroes_sample.json"
    report = {"series": series, "arrivals": {}}
    print("\nArrivals per hero (gt + (x,z)):")
    for name in hero_of:
        arrs = arrivals(series[name])
        report["arrivals"][name] = arrs
        if arrs:
            print(f"  {name:<12}: {len(arrs)} arrivals — " + "; ".join(f"gt={a['arrive_gt']:.1f}→{a['leave_gt']:.1f} ({a['x']:.0f},{a['z']:.0f})" for a in arrs[:5]))
        else:
            print(f"  {name:<12}: no arrivals (maybe stationary the whole time)")
    with open(out_path, "w") as f: json.dump(report, f, indent=2)
    print(f"\nSaved -> {out_path}")

if __name__ == "__main__":
    main()
