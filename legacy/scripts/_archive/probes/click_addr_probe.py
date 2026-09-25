"""Self-contained probe: launch replay → find Garen → vtable-scan → identify
candidates → dump each candidate's parent struct (256 bytes) + follow every
pointer in the parent for an inner-struct preview, looking for distinguishing
features (champion-name strings, owner pointers, vtables, etc.).

Output: C:\\tmp\\click_addr_probe.json with per-candidate structural data.
Designed to find a robust disambiguator BEFORE post-hoc density filtering.
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, struct, json, time, ssl, urllib.request, base64, os
from collections import defaultdict
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

CHAMPION = b"Garen"
HERO_POS_OFF = 0x200
HERO_NAME_OFF = 0x4360
VTABLE_RVA = 0x192BF90
VEC3_OFFSET_FROM_VPTR = 0x14
SB = 0x308; SC = 0x374
P1_DURATION_S = 25
DUMP_BYTES = 256

GAME_ID = os.environ.get("GAREN_GAME_ID", "5547830059")
LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"

_k = ctypes.windll.kernel32
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
REPLAY = "https://127.0.0.1:2999"

class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong), ("__a", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong),
                ("__b", ctypes.c_ulong)]
_k.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
_k.VirtualQueryEx.restype = ctypes.c_size_t
MEM_COMMIT=0x1000; MEM_PRIVATE=0x20000; PAGE_RW=0x04|0x08|0x40

def lcu_post(ep, body=None):
    parts = open(LOCKFILE).read().strip().split(":")
    port = parts[2]; auth = base64.b64encode(f"riot:{parts[3]}".encode()).decode()
    req = urllib.request.Request(f"https://127.0.0.1:{port}{ep}", method="POST",
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Basic {auth}", "Content-Type":"application/json"})
    with urllib.request.urlopen(req, context=_ctx, timeout=10) as r:
        raw = r.read(); return json.loads(raw) if raw else None

def replay_post(p, body):
    req = urllib.request.Request(f"{REPLAY}{p}", method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return r.read()

def api_get(p):
    try:
        with urllib.request.urlopen(f"{REPLAY}{p}", context=_ctx, timeout=2) as r:
            return json.loads(r.read())
    except: return None

def find_pid():
    r=subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                     capture_output=True,text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

def module_range(pid):
    psapi = ctypes.WinDLL("psapi.dll")
    h = _k.OpenProcess(0x0410, False, pid)
    HMODULE = wt.HMODULE
    psapi.EnumProcessModulesEx.argtypes = [wt.HANDLE, ctypes.POINTER(HMODULE), wt.DWORD, ctypes.POINTER(wt.DWORD), wt.DWORD]
    psapi.GetModuleFileNameExW.argtypes = [wt.HANDLE, HMODULE, wt.LPWSTR, wt.DWORD]
    class MINFO(ctypes.Structure):
        _fields_ = [("lpBaseOfDll", ctypes.c_void_p), ("SizeOfImage", wt.DWORD), ("EntryPoint", ctypes.c_void_p)]
    psapi.GetModuleInformation.argtypes = [wt.HANDLE, HMODULE, ctypes.POINTER(MINFO), wt.DWORD]
    mods = (HMODULE * 1024)(); needed = wt.DWORD(0)
    psapi.EnumProcessModulesEx(h, mods, ctypes.sizeof(mods), ctypes.byref(needed), 3)
    n = needed.value // ctypes.sizeof(HMODULE)
    for i in range(n):
        name = ctypes.create_unicode_buffer(260)
        psapi.GetModuleFileNameExW(h, mods[i], name, 260)
        if name.value.lower().endswith("league of legends.exe"):
            mi = MINFO()
            psapi.GetModuleInformation(h, mods[i], ctypes.byref(mi), ctypes.sizeof(mi))
            _k.CloseHandle(h)
            return mi.lpBaseOfDll, mi.SizeOfImage
    _k.CloseHandle(h); return None, None

def read_bytes(h, addr, n):
    buf=(ctypes.c_char*n)(); r=ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(r))
    return bytes(buf[:r.value]) if ok else None

def read_vec3(h, addr):
    b=read_bytes(h, addr, 12)
    return struct.unpack("<fff", b) if b and len(b)==12 else None

def regions(h, max_size=64*1024*1024):
    addr=0; mbi=MBI()
    while addr<0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h,ctypes.c_void_p(addr),ctypes.byref(mbi),ctypes.sizeof(mbi)): break
        b=mbi.BaseAddress or 0; s=mbi.RegionSize
        if (mbi.State==MEM_COMMIT and mbi.Type==MEM_PRIVATE
                and (mbi.Protect&PAGE_RW) and s<=max_size):
            yield b,s
        addr=b+s
        if addr<=b: break

def read_region(h, base, size):
    out=bytearray(size); v=memoryview(out); o=0; CH=4*1024*1024
    while o<size:
        n=min(CH,size-o)
        buf=(ctypes.c_char*n)(); r=ctypes.c_size_t(0)
        if not _k.ReadProcessMemory(h,ctypes.c_void_p(base+o),buf,n,ctypes.byref(r)) or r.value==0:
            return None if o==0 else bytes(v[:o])
        v[o:o+r.value]=buf[:r.value]; o+=r.value
    return bytes(out)

def find_hero_by_name(h, name_bytes):
    needle = name_bytes + b"\x00"
    for base, size in regions(h):
        data = read_region(h, base, size)
        if not data: continue
        off = 0
        while True:
            j = data.find(needle, off)
            if j == -1: break
            cand = base + j - 0x4360
            pos = read_vec3(h, cand + HERO_POS_OFF)
            if pos and 100 < pos[0] < 15000 and 100 < pos[2] < 15000 and 45 < pos[1] < 65:
                nm = read_bytes(h, cand + 0x4360, len(name_bytes)+1)
                if nm and nm.split(b"\x00")[0] == name_bytes:
                    return cand
            off = j + 1
    return None

def vtable_scan(h, target_vptr):
    target_bytes = struct.pack("<Q", target_vptr)
    out = []
    for base, size in regions(h):
        data = read_region(h, base, size)
        if not data: continue
        n = len(data) // 8 * 8
        if n < 8: continue
        arr = np.frombuffer(data[:n], dtype=np.uint64)
        idxs = np.nonzero(arr == np.uint64(target_vptr))[0]
        for i in idxs:
            vptr_addr = base + int(i)*8
            vec3_addr = vptr_addr + VEC3_OFFSET_FROM_VPTR
            bv = read_bytes(h, vec3_addr, 12)
            if not bv or len(bv) < 12: continue
            x, y, z = struct.unpack("<fff", bv)
            if not (100 < x < 15000 and 100 < z < 15000 and 45 < y < 65): continue
            bb = read_bytes(h, vec3_addr + SB, 12)
            bc = read_bytes(h, vec3_addr + SC, 12)
            if not (bb and bc and len(bb)==12 and len(bc)==12): continue
            xb,yb,zb = struct.unpack("<fff", bb)
            xc,yc,zc = struct.unpack("<fff", bc)
            if (abs(x-xb)<0.01 and abs(y-yb)<0.01 and abs(z-zb)<0.01
                and abs(x-xc)<0.01 and abs(z-zc)<0.01):
                out.append((vptr_addr, vec3_addr, (x,y,z)))
    return out

def identify_top_k(h, hero_ptr, target_vptr, duration_s, k=10):
    hero_hist = []
    cand_hist = defaultdict(list)
    t0 = time.time()
    next_scan = t0
    last_pos_t = 0
    while time.time() - t0 < duration_s:
        now = time.time()
        t_rel = now - t0
        pb = api_get("/replay/playback"); gt = pb.get("time", 0) if pb else 0
        if now - last_pos_t >= 1.0:
            pos = read_vec3(h, hero_ptr + HERO_POS_OFF)
            if pos and 100 < pos[0] < 15000:
                hero_hist.append((t_rel, gt, pos))
            last_pos_t = now
        if now >= next_scan:
            cands = vtable_scan(h, target_vptr)
            for vptr, vec3_addr, v in cands:
                cand_hist[vec3_addr].append((t_rel, gt, v))
            next_scan = now + 4.0
        time.sleep(0.2)
    scores = []
    for addr, hist in cand_hist.items():
        if len(hist) < 3 or len(set(v for _,_,v in hist)) < 2: continue
        dists = []
        for t_c, gt_c, v_c in hist:
            future = [p for t,gt,p in hero_hist if gt_c <= gt <= gt_c + 15.0]
            if future:
                dists.append(min(((p[0]-v_c[0])**2 + (p[2]-v_c[2])**2)**0.5 for p in future))
        if not dists: continue
        avg = sum(dists)/len(dists)
        if avg < 1500:
            scores.append((addr, avg, len(hist)))
    scores.sort(key=lambda r: r[1])
    return scores[:k], hero_hist

def follow_pointer(h, ptr, hero_ptr, base):
    """For a candidate pointer, check if it's:
    - A vtable (target's first u64 is in module text)
    - A hero struct (champion_name @ +0x4360 is ascii)
    - The known Garen hero pointer
    - Some other recognizable struct
    Returns dict with structural metadata."""
    info = {"ptr_hex": hex(ptr)}
    if not (0x10000 <= ptr <= 0x7FFFFFFFFFFF):
        info["valid"] = False; return info
    if ptr == hero_ptr:
        info["match"] = "= GAREN HERO POINTER"; return info
    head = read_bytes(h, ptr, 32)
    if not head:
        info["read_failed"] = True; return info
    info["head_hex"] = head[:16].hex()
    info["head_ascii"] = "".join(chr(b) if 32<=b<127 else '.' for b in head)
    # If first u64 looks like a vptr (in module range)
    vptr = struct.unpack("<Q", head[:8])[0]
    if base <= vptr <= base + 0x3000000:
        info["vptr_rva"] = hex(vptr - base)
    # Try to read champion-name string at +0x4360 (hero struct heuristic)
    nm_b = read_bytes(h, ptr + HERO_NAME_OFF, 24)
    if nm_b:
        nul = nm_b.find(b"\x00")
        nm = nm_b[:nul if nul >= 0 else 24]
        if nm and all(32 <= b < 127 for b in nm) and len(nm) >= 3 and nm.isascii() and nm.decode().isalpha():
            info["champion_name_at_0x4360"] = nm.decode()
    return info

def launch_replay():
    print(f"launching replay {GAME_ID} via LCU...", flush=True)
    try: lcu_post(f"/lol-replays/v1/rofls/{GAME_ID}/watch", {"componentType": "replay"})
    except Exception as e: print(f"  LCU: {e}", flush=True)
    for i in range(120):
        gs = api_get("/liveclientdata/gamestats")
        if gs and gs.get("gameTime") is not None:
            print(f"  loaded ({i*2}s)", flush=True); return True
        time.sleep(2)
    return False

def setup_2x():
    replay_post("/replay/playback", {"paused": False, "speed": 2.0, "time": 1.0})
    time.sleep(1)

def main():
    if not launch_replay(): print("ERR launch"); return 1
    setup_2x()
    pid = find_pid(); base, _ = module_range(pid)
    target_vptr = base + VTABLE_RVA
    print(f"pid={pid} module=0x{base:X} target_vptr=0x{target_vptr:X}", flush=True)
    h = _k.OpenProcess(0x0410, False, pid)

    hero_ptr = None
    for attempt in range(15):
        hero_ptr = find_hero_by_name(h, CHAMPION)
        if hero_ptr: break
        time.sleep(2)
    if not hero_ptr: print("ERR no Garen"); return 1
    print(f"hero=0x{hero_ptr:X}", flush=True)

    print(f"identify ({P1_DURATION_S}s)...", flush=True)
    top, hero_hist = identify_top_k(h, hero_ptr, target_vptr, P1_DURATION_S, k=10)
    print(f"got {len(top)} candidates", flush=True)

    cand_data = []
    for addr, avg_d, n in top:
        parent = addr - VEC3_OFFSET_FROM_VPTR
        raw = read_bytes(h, parent, DUMP_BYTES)
        if not raw or len(raw) < DUMP_BYTES: continue
        # Extract u64 slots + pointer chase each likely-pointer
        slots = []
        for off in range(0, DUMP_BYTES, 8):
            v = struct.unpack("<Q", raw[off:off+8])[0]
            entry = {"off": off, "u64": hex(v)}
            if 0x10000 <= v <= 0x7FFFFFFFFFFF:
                entry["chase"] = follow_pointer(h, v, hero_ptr, base)
            # Also check if it looks like a float vec component (rough heuristic)
            slots.append(entry)
        # Vec3 reading at +0x14
        v3 = read_vec3(h, addr)
        cand_data.append({
            "vec3_addr": hex(addr), "parent_addr": hex(parent),
            "avg_d_to_hero": round(avg_d, 1), "n_samples": n,
            "current_vec3": list(v3) if v3 else None,
            "raw_hex": raw.hex(),
            "slots": slots,
        })
        print(f"\n--- vec3@{hex(addr)}  parent={hex(parent)}  avg_d={avg_d:.0f}  n={n} ---", flush=True)
        for s in slots:
            mark = ""
            chase = s.get("chase")
            if chase:
                if chase.get("match"): mark = f" ★ {chase['match']}"
                elif chase.get("champion_name_at_0x4360"): mark = f" ← hero={chase['champion_name_at_0x4360']}"
                elif chase.get("vptr_rva"): mark = f" vtable rva={chase['vptr_rva']}"
            print(f"  +0x{s['off']:>3X}  {s['u64']:>18}{mark}")

    out = {
        "champion": CHAMPION.decode(),
        "game_id": GAME_ID,
        "hero_addr": hex(hero_ptr),
        "module_base": hex(base),
        "vtable_addr": hex(target_vptr),
        "candidates": cand_data,
    }
    os.makedirs(r"C:\tmp", exist_ok=True)
    with open(r"C:\tmp\click_addr_probe.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote C:\\tmp\\click_addr_probe.json ({len(cand_data)} candidates)", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
