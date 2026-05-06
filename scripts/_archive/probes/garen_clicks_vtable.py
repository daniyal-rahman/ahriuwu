"""Garen click-destination extractor using the vtable signature.

Uses the 16.8.766 click-dest-class signature found via static RE:
  vptr at parent+0x00 == module_base + 0x192BF90
  Vec3 at parent+0x14 (3 floats x, y, z)
  triple-mirror at +0x308 / +0x374

Procedure:
  1. Find Garen hero via "Garen\\0" heap scan (verify position+champion_name).
  2. Vtable-scan heap for click-dest-class objects (~2s, 18-ish candidates).
  3. Identify-phase (60s wall ≈ 120s game at 2x): track each candidate's Vec3
     and Garen's position; correlate trajectories; pick top-K by lowest avg-dist
     and low simul-ratio (avoid Garen's own-position mirror).
  4. Tight-poll-phase: every 100ms, read Vec3 from the top-K candidates; record
     deltas > 50u as click events.
  5. Re-verify every 3 min: if half the watched addrs go invalid, redo identify.

Output: C:\\tmp\\garen_clicks_vtable.json
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, struct, json, time, ssl, urllib.request, base64, os
from collections import defaultdict
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import builtins
_orig=builtins.print
def print(*a,**k): k.setdefault("flush",True); _orig(*a,**k)
builtins.print=print

CHAMPION = b"Garen"
HERO_POS_OFF = 0x200
VTABLE_RVA = 0x192BF90
VEC3_OFFSET_FROM_VPTR = 0x14
OWNER_PTR_OFF = 0x68 - VEC3_OFFSET_FROM_VPTR  # 0x54 — relative to vec3 addr
                                              # parent+0x68 = back-pointer to owner hero
SB = 0x308; SC = 0x374
POS_POLL_S = 0.5
POLL_MS = 30   # ~30Hz target (HTTP RTT floor ≈ 13ms; achieved ~27Hz)
P1_DURATION_S = 25
REVERIFY_S = 180
DELTA_UNITS = 50.0
RUN_DURATION_S = 360   # 6 min wall ≈ 12 min game at 2x

# Spellbook cast-detection (catches Q-activation, W, E, R, D, F via cd_expire jumps)
SPELLBOOK_OFF = 0x3120
SLOT_ARRAY_OFF = 0xAE0   # relative to spellbook
SLOT_CD_EXPIRE_OFF = 0x30
SLOT_TOTAL_CD_OFF = 0x74
SLOT_SPELL_INFO_OFF = 0x130
SPELL_NAME_PTR_OFF = 0x28
SLOT_NAMES = ["Q", "W", "E", "R", "D", "F"]   # 6 slots: champion abilities + summoners
ACTIVE_SPELL_OFF = 0x3158   # current channeling spell ptr — catches recall

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
    """Scan heap for the champion-name string and verify hero struct."""
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
    """Return list of (vptr_addr, vec3_addr, vec3) for objects whose vptr matches
    and which have a valid Vec3 + triple-mirror at vec3_addr."""
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

def valid_vec(v):
    return v and 100 < v[0] < 15000 and 45 < v[1] < 65 and 100 < v[2] < 15000

def identify_top_k(h, hero_ptr, target_vptr, duration_s, k=5):
    hero_hist = []   # [(t_rel, gt, pos), ...]
    cand_hist = defaultdict(list)
    t0 = time.time()
    next_scan = t0
    last_pos_t = 0
    print(f"  [identify] running {duration_s}s, scanning every 4s...")
    while time.time() - t0 < duration_s:
        now = time.time()
        t_rel = now - t0
        pb = api_get("/replay/playback"); gt = pb.get("time", 0) if pb else 0
        if now - last_pos_t >= 1.0:
            pos = read_vec3(h, hero_ptr + HERO_POS_OFF)
            if valid_vec(pos):
                hero_hist.append((t_rel, gt, pos))
            last_pos_t = now
        if now >= next_scan:
            cands = vtable_scan(h, target_vptr)
            for vptr, vec3_addr, v in cands:
                cand_hist[vec3_addr].append((t_rel, gt, v))
            next_scan = now + 4.0
        time.sleep(0.2)
    # Score
    scores = []
    for addr, hist in cand_hist.items():
        if len(hist) < 3 or len(set(v for _,_,v in hist)) < 2: continue
        dists = []
        simul = 0
        for t_c, gt_c, v_c in hist:
            future = [p for t,gt,p in hero_hist if gt_c <= gt <= gt_c + 15.0]
            if future:
                dists.append(min(((p[0]-v_c[0])**2 + (p[2]-v_c[2])**2)**0.5 for p in future))
            same = [p for t,gt,p in hero_hist if abs(gt - gt_c) < 1.0]
            if same and ((same[0][0]-v_c[0])**2 + (same[0][2]-v_c[2])**2)**0.5 < 50:
                simul += 1
        if not dists: continue
        avg = sum(dists)/len(dists)
        sr = simul/len(hist)
        if sr < 0.4 and avg < 1500:
            scores.append((addr, avg, sr, len(hist)))
    scores.sort(key=lambda r: r[1])
    return scores[:k], hero_hist

GAME_ID = os.environ.get("GAREN_GAME_ID", "5547184086")
LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"

def lcu_post(ep, body=None):
    parts = open(LOCKFILE).read().strip().split(":")
    port = parts[2]; auth = base64.b64encode(f"riot:{parts[3]}".encode()).decode()
    req = urllib.request.Request(
        f"https://127.0.0.1:{port}{ep}", method="POST",
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, context=_ctx, timeout=10) as r:
        raw = r.read(); return json.loads(raw) if raw else None

def launch_replay():
    print(f"launching replay {GAME_ID} via LCU...")
    try:
        lcu_post(f"/lol-replays/v1/rofls/{GAME_ID}/watch", {"componentType": "replay"})
    except Exception as e:
        print(f"  LCU launch returned: {e} (may be benign if already loading)")
    for i in range(120):
        gs = api_get("/liveclientdata/gamestats")
        if gs is not None and isinstance(gs, dict) and gs.get("gameTime") is not None:
            print(f"  game loaded ({i*2}s, gameTime={gs.get('gameTime')})")
            return True
        time.sleep(2)
    return False

def setup_playback():
    """Run replay at 2x from start; do NOT advance the clock. We want click
    coverage from the earliest moment the alloc resolves."""
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{REPLAY}/replay/playback",
            data=json.dumps({"paused": False, "speed": 2.0, "time": 1.0}).encode(),
            headers={"Content-Type":"application/json"}), context=_ctx, timeout=5) as r: r.read()
        return True
    except Exception as e:
        print(f"playback setup failed: {e}"); return False

def read_spellbook_slots(h, hero_ptr):
    """Return list of (slot_name, cd_expire, total_cd) for Q/W/E/R."""
    slots = []
    sb_addr = hero_ptr + SPELLBOOK_OFF
    slot_array_addr = sb_addr + SLOT_ARRAY_OFF
    for i, name in enumerate(SLOT_NAMES):
        slot_ptr = read_u64(h, slot_array_addr + i * 8)
        if not slot_ptr or slot_ptr < 0x10000 or slot_ptr > 0x7FFFFFFFFFFF:
            slots.append((name, None, None)); continue
        cd_e = read_f32(h, slot_ptr + SLOT_CD_EXPIRE_OFF)
        cd_t = read_f32(h, slot_ptr + SLOT_TOTAL_CD_OFF)
        slots.append((name, cd_e, cd_t))
    return slots

def read_u64(h, addr):
    b = read_bytes(h, addr, 8)
    return struct.unpack("<Q", b)[0] if b and len(b)==8 else None

def read_f32(h, addr):
    b = read_bytes(h, addr, 4)
    return struct.unpack("<f", b)[0] if b and len(b)==4 else None

def read_cstr(h, addr, max_len=64):
    b = read_bytes(h, addr, max_len)
    if not b: return None
    n = b.find(b"\x00")
    return b[:n if n >= 0 else max_len].decode("ascii", errors="replace")

def read_slot_spell_name(h, slot_ptr):
    """Resolve slot's spell name via slot+0x130 → +0x28 → cstr."""
    if not slot_ptr: return None
    si = read_u64(h, slot_ptr + SLOT_SPELL_INFO_OFF)
    if not si or si < 0x10000 or si > 0x7FFFFFFFFFFF: return None
    np_ = read_u64(h, si + SPELL_NAME_PTR_OFF)
    if not np_ or np_ < 0x10000 or np_ > 0x7FFFFFFFFFFF: return None
    return read_cstr(h, np_, 96)

def read_active_spell_name(h, hero_ptr):
    """Read the current active spell name (channels: recall, summoners, AAs)."""
    sp = read_u64(h, hero_ptr + ACTIVE_SPELL_OFF)
    if not sp or sp < 0x10000 or sp > 0x7FFFFFFFFFFF: return None
    si = read_u64(h, sp + 0x008)   # spell_info
    if not si or si < 0x10000 or si > 0x7FFFFFFFFFFF: return None
    np_ = read_u64(h, si + SPELL_NAME_PTR_OFF)
    if not np_ or np_ < 0x10000 or np_ > 0x7FFFFFFFFFFF: return None
    return read_cstr(h, np_, 96)

def main():
    if not launch_replay():
        print("ERR: replay did not load"); return 1
    if not setup_playback():
        print("ERR: playback setup failed"); return 1
    pid = find_pid()
    if not pid: print("ERR no League"); return 1
    base, _ = module_range(pid)
    if not base: print("ERR module"); return 1
    target_vptr = base + VTABLE_RVA
    print(f"pid={pid}  module=0x{base:X}  target_vptr=0x{target_vptr:X}")

    h = _k.OpenProcess(0x0410, False, pid)
    print(f"finding {CHAMPION.decode()} hero...")
    # Hero struct isn't allocated until a few seconds in. Retry up to 30s.
    hero_ptr = None
    for attempt in range(15):
        hero_ptr = find_hero_by_name(h, CHAMPION)
        if hero_ptr: break
        gs = api_get("/liveclientdata/gamestats")
        gt_now = gs.get("gameTime") if gs else None
        print(f"  attempt {attempt+1}: no hero yet (gt={gt_now}), waiting 2s...")
        time.sleep(2)
    if not hero_ptr:
        print(f"ERR: could not find {CHAMPION.decode()} hero in heap"); return 1
    print(f"  hero=0x{hero_ptr:X}")
    pos = read_vec3(h, hero_ptr + HERO_POS_OFF)
    print(f"  pos=({pos[0]:.0f},{pos[1]:.1f},{pos[2]:.0f})")

    # P1: identify (collect ALL viable click-dest objects, not just top-k)
    top, hero_hist = identify_top_k(h, hero_ptr, target_vptr, P1_DURATION_S, k=20)
    if not top:
        print("ERR: identify produced no candidates"); return 1
    print(f"\n  candidates after P1 ({len(top)}):")
    for a, avg, sr, n in top:
        owner = read_u64(h, a + OWNER_PTR_OFF)
        owner_mark = " ★ GAREN" if owner == hero_ptr else f" owner=0x{owner or 0:X}"
        print(f"    vec3@0x{a:X}  avg_d={avg:.0f}  simul={sr:.2f}  n={n}{owner_mark}")

    # Disambiguate by owner pointer (parent+0x68): keep ONLY candidates whose
    # owner matches the Garen hero struct. This is the structural disambiguator
    # found via click_addr_probe.py — replaces the brittle post-hoc density vote.
    garen_owned = [(a,*r) for (a,*r) in top
                   if read_u64(h, a + OWNER_PTR_OFF) == hero_ptr]
    if not garen_owned:
        print("WARN: no candidate owned by Garen — falling back to top by avg_d")
        garen_owned = top[:1]
    watched = [a for a,*_ in garen_owned]
    print(f"  after owner-filter: watching {len(watched)} addr(s)")
    prev_vec = {a: read_vec3(h, a) for a in watched}

    # Resolve slot spell-names so D/F are tagged Flash/Ignite/TP/etc.
    slot_array_addr = hero_ptr + SPELLBOOK_OFF + SLOT_ARRAY_OFF
    slot_real_names = {}
    for i, name in enumerate(SLOT_NAMES):
        slot_ptr = read_u64(h, slot_array_addr + i*8)
        nm = read_slot_spell_name(h, slot_ptr) if slot_ptr else None
        slot_real_names[name] = nm
        print(f"  slot {name}: {nm}")

    # P2: tight poll (clicks + spell-cast detection)
    all_clicks = []
    all_casts = []   # (gt, slot_name, hero_pos)
    prev_cd = {n: None for n in SLOT_NAMES}
    prev_active = None
    last_recall_t = -10.0
    t_loop0 = time.time()
    next_reverify = t_loop0 + REVERIFY_S
    last_print = t_loop0
    last_gt = 0
    print(f"\n== P2: tight-poll {len(watched)} addrs at {1000//POLL_MS}Hz + spell cd ==")
    while True:
        now = time.time()
        if now - t_loop0 > RUN_DURATION_S: break
        # Read vec3s FIRST (cheap mem reads), then query gt — drift_probe showed
        # gt-before-vec3 introduced +6ms positive bias; gt-after gives a tighter
        # bracket on the actual click time.
        hero_pos = read_vec3(h, hero_ptr + HERO_POS_OFF)
        cur_vecs = {addr: read_vec3(h, addr) for addr in watched}
        pb = api_get("/replay/playback")
        gt = pb.get("time", last_gt) if pb else last_gt
        last_gt = gt
        for addr in watched:
            v = cur_vecs[addr]
            if not valid_vec(v):
                continue
            prev = prev_vec.get(addr)
            if prev:
                dx, dz = v[0]-prev[0], v[2]-prev[2]
                d = (dx*dx + dz*dz) ** 0.5
                if d > DELTA_UNITS:
                    all_clicks.append({
                        "game_t": gt, "addr": hex(addr),
                        "x": v[0], "y": v[1], "z": v[2], "delta": round(d, 1),
                        "hero_x": hero_pos[0] if hero_pos else None,
                        "hero_z": hero_pos[2] if hero_pos else None,
                    })
            prev_vec[addr] = v
        # Spell cast detection: cd_expire jumps to (gt + total_cd) on cast.
        slots = read_spellbook_slots(h, hero_ptr)
        for name, cd_e, cd_t in slots:
            if cd_e is None or cd_t is None: continue
            prev = prev_cd[name]
            # Cast: cd_expire jumped forward by ~total_cd (allow ±1.5s slop on
            # gt sampling jitter). Skip the first sample (no prev).
            if prev is not None and cd_e - prev > max(1.0, cd_t * 0.5):
                all_casts.append({
                    "game_t": gt, "slot": name,
                    "spell_name": slot_real_names.get(name),
                    "hero_x": hero_pos[0] if hero_pos else None,
                    "hero_z": hero_pos[2] if hero_pos else None,
                    "cd_expire": cd_e, "total_cd": cd_t,
                })
            prev_cd[name] = cd_e
        # Active-spell channel detection: recall (and any other channeled spell
        # that doesn't update a spellbook slot's cd_expire).
        active_name = read_active_spell_name(h, hero_ptr)
        if active_name and active_name != prev_active:
            nl = active_name.lower()
            if "recall" in nl and gt - last_recall_t > 1.0:
                all_casts.append({
                    "game_t": gt, "slot": "B",
                    "spell_name": active_name,
                    "hero_x": hero_pos[0] if hero_pos else None,
                    "hero_z": hero_pos[2] if hero_pos else None,
                })
                last_recall_t = gt
        prev_active = active_name
        if now >= next_reverify:
            alive = [a for a in watched if valid_vec(read_vec3(h, a))]
            if len(alive) < max(1, len(watched)//2):
                print(f"  reverify at gt={gt:.0f}: {len(alive)}/{len(watched)} alive, redoing identify")
                top, _ = identify_top_k(h, hero_ptr, target_vptr, 30, k=20)
                garen_owned = [(a,*r) for (a,*r) in top
                               if read_u64(h, a + OWNER_PTR_OFF) == hero_ptr]
                if garen_owned:
                    watched = [a for a,*_ in garen_owned]
                    prev_vec = {a: read_vec3(h, a) for a in watched}
                    print(f"  new watched (owner-filtered): {[hex(a) for a in watched]}")
            else:
                print(f"  reverify gt={gt:.0f}: {len(alive)}/{len(watched)} alive OK")
            next_reverify = now + REVERIFY_S
        if now - last_print > 30:
            print(f"  wall={now-t_loop0:6.1f}  gt={gt:7.1f}  clicks={len(all_clicks)}  casts={len(all_casts)}")
            last_print = now
        time.sleep(POLL_MS/1000.0)

    out = {
        "champion": CHAMPION.decode(),
        "hero_addr": hex(hero_ptr),
        "module_base": hex(base),
        "vtable_addr": hex(target_vptr),
        "watched_addrs": [hex(a) for a in watched],
        "total_clicks": len(all_clicks),
        "clicks": all_clicks,
        "total_casts": len(all_casts),
        "casts": all_casts,
    }
    import os
    os.makedirs(r"C:\tmp", exist_ok=True)
    with open(r"C:\tmp\garen_clicks_vtable.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote C:\\tmp\\garen_clicks_vtable.json  ({len(all_clicks)} clicks)")
    return 0

if __name__=="__main__":
    sys.exit(main())
