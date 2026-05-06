"""
Targeted scan: find current HP + confirm gold offset.

Known: hp_max @ 0x10A8, gold candidates @ 0x2858 and 0x4CE8.
Strategy:
  1. Launch replay, sample at gt=120, gt=300, gt=500
  2. Around 0x10A8 (±0x200), find f32 that's in [100, hp_max] and changes
  3. At 0x2858 and 0x4CE8, check if values monotonically increase
     across 3 snapshots with reasonable gold income rate
"""
import base64, ctypes, ctypes.wintypes as wt, json, ssl, struct, subprocess, sys, time, urllib.request
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

HERO_ARRAY_RVA = 0x1DBEEE8
CHAMPION_NAME_OFF = 0x4328
LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"
REPLAY_ID = "5528069928"

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
    p = open(LOCKFILE).read().strip().split(":")
    req = urllib.request.Request(f"https://127.0.0.1:{p[2]}/lol-replays/v1/rofls/{REPLAY_ID}/watch",
        method="POST", data=json.dumps({"componentType":"string"}).encode(),
        headers={"Authorization": f"Basic {base64.b64encode(f'riot:{p[3]}'.encode()).decode()}",
                 "Content-Type":"application/json"})
    urllib.request.urlopen(req, context=_ctx, timeout=10)


class Mem:
    def __init__(self, pid): self.h = _k.OpenProcess(0x0410, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u64(self, a): d=self.read(a,8); return struct.unpack('<Q',d)[0] if d else None
    def f32(self, a): d=self.read(a,4); return struct.unpack('<f',d)[0] if d else None
    def string(self, a, n=64):
        d=self.read(a,n)
        if not d: return None
        try: return d.split(b'\x00')[0].decode('ascii')
        except: return None

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'], capture_output=True, text=True)
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


def snapshot(m, heroes):
    """Read values at known-candidate offsets for each hero."""
    snap = {}
    for name, hp in heroes:
        d = {}
        # Read f32 values around 0x10A8 (±0x100)
        for off in range(0x1000, 0x11A0, 4):
            v = m.f32(hp + off)
            if v is not None:
                d[off] = v
        # Also read gold candidates
        d[0x2858] = m.f32(hp + 0x2858) or 0
        d[0x4CE8] = m.f32(hp + 0x4CE8) or 0
        snap[name] = d
    return snap


def main():
    print("=" * 60)
    print("Targeted scan: current HP + gold disambiguation")
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
        hp = m.u64(arr_ptr + i*8)
        if hp:
            name = m.string(hp + CHAMPION_NAME_OFF)
            if name: heroes.append((name, hp))
    print(f"[mem] heroes: {[h[0] for h in heroes]}")

    # Sample 3 snapshots
    targets = [150, 300, 500]
    snapshots = []
    for tgt in targets:
        print(f"\n[play] advancing to gt={tgt}...")
        play_until(float(tgt))
        actual = rget("/liveclientdata/gamestats").get("gameTime", 0)
        print(f"  actual gt: {actual:.1f}")
        snap = snapshot(m, heroes)
        snapshots.append((actual, snap))

    # ── Analyze HP region (0x1000-0x11A0) ──
    print("\n" + "=" * 60)
    print("HP region (0x1000-0x11A0): find current HP (changes, in [100, 3000])")
    print("=" * 60)

    # For each offset, print values at all 3 snapshots for Garen + LeeSin
    print(f"\n{'off':>6} | {'Garen t1':>10} {'t2':>10} {'t3':>10} | {'LeeSin t1':>12} {'t2':>12} {'t3':>12}")
    offsets = sorted(snapshots[0][1]["Garen"].keys())
    for off in offsets:
        if off > 0x11A0: continue
        garen_vals = [s[1]["Garen"].get(off, 0) for s in snapshots]
        lee_vals = [s[1]["LeeSin"].get(off, 0) for s in snapshots]
        # Filter: only print if at least one value is in plausible HP range
        all_vals = garen_vals + lee_vals
        if any(100 < v < 3000 for v in all_vals):
            changed = any(abs(garen_vals[0] - garen_vals[i]) > 1 for i in range(1, 3))
            mark = " *" if changed else ""
            print(f"0x{off:04X} | {garen_vals[0]:10.1f} {garen_vals[1]:10.1f} {garen_vals[2]:10.1f} | {lee_vals[0]:12.1f} {lee_vals[1]:12.1f} {lee_vals[2]:12.1f}{mark}")

    # ── Gold disambiguation ──
    print("\n" + "=" * 60)
    print("Gold candidates: 0x2858 vs 0x4CE8")
    print("=" * 60)
    print(f"\n{'Champ':<12} | {'0x2858 t1':>12} {'t2':>12} {'t3':>12} | {'0x4CE8 t1':>12} {'t2':>12} {'t3':>12}")
    for name, _ in heroes:
        a = [s[1][name].get(0x2858, 0) for s in snapshots]
        b = [s[1][name].get(0x4CE8, 0) for s in snapshots]
        print(f"{name:<12} | {a[0]:12.0f} {a[1]:12.0f} {a[2]:12.0f} | {b[0]:12.0f} {b[1]:12.0f} {b[2]:12.0f}")

    print("\n[done]")
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except: import traceback; traceback.print_exc(); sys.exit(2)
