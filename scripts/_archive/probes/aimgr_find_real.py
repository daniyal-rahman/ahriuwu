"""
Find the REAL AiManager pointer in the hero struct.

The previous finding was WRONG: hero+0x4628 -> +0x10 is a stats struct (GOLD_SPENT etc),
NOT AiManager. The position at inner+0x000 was just a coincidence.

We need to scan the hero struct (offsets 0x3000-0x5000) for a pointer to a struct that:
1. Contains the hero's current position somewhere (server pos)
2. Contains movement speed float (~300-400 for most champs)
3. Contains a movement target Vec3 that differs from current pos
4. Has fields that change when the hero moves

Strategy: For each candidate pointer in the hero struct, read 0x500 bytes and look
for the AiManager signature: position data + velocity + path data.
"""

import ctypes
import ctypes.wintypes
import struct
import sys
import time
from ctypes import wintypes

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
MAX_MODULE_NAME32 = 255
MAX_PATH = 260
HERO_ARRAY_RVA = 0x1DBEEE8

class PROCESSENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD),("cntUsage", wintypes.DWORD),("th32ProcessID", wintypes.DWORD),("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),("th32ModuleID", wintypes.DWORD),("cntThreads", wintypes.DWORD),("th32ParentProcessID", wintypes.DWORD),("pcPriClassBase", ctypes.c_long),("dwFlags", wintypes.DWORD),("szExeFile", ctypes.c_char * MAX_PATH)]

class MODULEENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD),("th32ModuleID", wintypes.DWORD),("th32ProcessID", wintypes.DWORD),("GlblcntUsage", wintypes.DWORD),("ProccntUsage", wintypes.DWORD),("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),("modBaseSize", wintypes.DWORD),("hModule", wintypes.HMODULE),("szModule", ctypes.c_char * (MAX_MODULE_NAME32 + 1)),("szExePath", ctypes.c_char * MAX_PATH)]

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

def find_league():
    s = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    pe = PROCESSENTRY32(); pe.dwSize = ctypes.sizeof(PROCESSENTRY32)
    kernel32.Process32First(s, ctypes.byref(pe))
    while True:
        if b"League of Legends" in pe.szExeFile:
            pid = pe.th32ProcessID; kernel32.CloseHandle(s); return pid
        if not kernel32.Process32Next(s, ctypes.byref(pe)): break
    kernel32.CloseHandle(s); return None

def find_base(pid):
    s = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    me = MODULEENTRY32(); me.dwSize = ctypes.sizeof(MODULEENTRY32)
    kernel32.Module32First(s, ctypes.byref(me))
    while True:
        if b"league of legends" in me.szModule.lower():
            b = ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value; sz = me.modBaseSize; kernel32.CloseHandle(s); return b, sz
        if not kernel32.Module32Next(s, ctypes.byref(me)): break
    kernel32.CloseHandle(s); return None, None

class Mem:
    def __init__(s, pid):
        s.h = kernel32.OpenProcess(PROCESS_VM_READ|PROCESS_QUERY_INFORMATION, False, pid)
    def read(s, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = kernel32.ReadProcessMemory(s.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u32(s,a): d=s.read(a,4); return struct.unpack("<I",d)[0] if d else None
    def u64(s,a): d=s.read(a,8); return struct.unpack("<Q",d)[0] if d else None
    def f32(s,a): d=s.read(a,4); return struct.unpack("<f",d)[0] if d else None
    def vec3(s,a): d=s.read(a,12); return struct.unpack("<fff",d) if d else None
    def string(s,a,n=64):
        d=s.read(a,n)
        if not d: return None
        return d.split(b'\x00')[0].decode('ascii',errors='replace') or None
    def close(s): kernel32.CloseHandle(s.h)

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF
def is_map_pos(v):
    if v is None: return False
    x, y, z = v
    return (-500 < x < 16000 and -500 < y < 1000 and -500 < z < 16000
            and not (x == 0 and z == 0)
            and not (abs(x - 1.0) < 0.01 and abs(z - 1.0) < 0.01))

def score_aimgr(data, hero_pos):
    """Score how likely a 0x500-byte buffer is an AiManager struct.
    Returns (score, details_dict). Higher = more likely."""
    if not data or len(data) < 0x400:
        return 0, {}

    hx, hy, hz = hero_pos if hero_pos else (0, 0, 0)
    score = 0
    details = {}

    # Check known AiManager field offsets from hacker_logs
    # TargetPosition at +0x34
    tp = struct.unpack("<fff", data[0x34:0x34+12])
    if is_map_pos(tp):
        score += 10
        details["target_0x34"] = f"({tp[0]:.0f},{tp[2]:.0f})"

    # ServerPos at +0x474
    if len(data) >= 0x474 + 12:
        sp = struct.unpack("<fff", data[0x474:0x474+12])
        if is_map_pos(sp):
            dist = ((sp[0]-hx)**2 + (sp[2]-hz)**2)**0.5
            if dist < 100:
                score += 20  # Server pos matches hero pos = strong signal
                details["serverpos_0x474"] = f"({sp[0]:.0f},{sp[2]:.0f}) dist={dist:.0f}"

    # Velocity at +0x318 (typical move speed 300-500)
    if len(data) >= 0x31C:
        vel = struct.unpack("<f", data[0x318:0x31C])[0]
        if 100 < vel < 1000:
            score += 15
            details["velocity_0x318"] = f"{vel:.0f}"

    # IsMoving at +0x31C
    if len(data) >= 0x320:
        moving = struct.unpack("<f", data[0x31C:0x320])[0]
        if moving == 0.0 or moving == 1.0:
            score += 5
            details["isMoving_0x31C"] = f"{moving}"

    # SegmentsCount at +0x350 (0-50)
    if len(data) >= 0x354:
        seg = struct.unpack("<i", data[0x350:0x354])[0]
        if 0 <= seg <= 50:
            score += 5
            details["segments_0x350"] = str(seg)

    # PathStart at +0x330
    if len(data) >= 0x330 + 12:
        ps = struct.unpack("<fff", data[0x330:0x330+12])
        if is_map_pos(ps):
            score += 5
            details["pathStart_0x330"] = f"({ps[0]:.0f},{ps[2]:.0f})"

    # PathEnd at +0x33C
    if len(data) >= 0x33C + 12:
        pe = struct.unpack("<fff", data[0x33C:0x33C+12])
        if is_map_pos(pe):
            score += 5
            details["pathEnd_0x33C"] = f"({pe[0]:.0f},{pe[2]:.0f})"

    # NavArray ptr at +0x348 (should be a heap pointer)
    if len(data) >= 0x350:
        nav = struct.unpack("<Q", data[0x348:0x350])[0]
        if is_heap(nav):
            score += 5
            details["navArray_0x348"] = f"0x{nav:X}"

    # Position at +0x000 (some AiManagers have current pos here)
    cp = struct.unpack("<fff", data[0x00:0x0C])
    if is_map_pos(cp):
        dist = ((cp[0]-hx)**2 + (cp[2]-hz)**2)**0.5
        if dist < 100:
            score += 10
            details["pos_0x000"] = f"({cp[0]:.0f},{cp[2]:.0f}) dist={dist:.0f}"

    # Count how many map-position Vec3s are in the struct (AiManager has several)
    map_pos_count = 0
    for off in range(0, min(len(data)-12, 0x500), 4):
        v = struct.unpack("<fff", data[off:off+12])
        if is_map_pos(v):
            map_pos_count += 1
    if map_pos_count >= 3:
        score += map_pos_count
        details["map_pos_count"] = map_pos_count

    return score, details


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 70)
    print("Find REAL AiManager in Hero Struct")
    print("=" * 70)

    pid = find_league()
    if not pid: print("League not found!"); return
    base, _ = find_base(pid)
    if not base: print("Module not found!"); return
    print(f"PID={pid} Base=0x{base:X}")

    m = Mem(pid)
    assert m.read(base, 2) == b'MZ'

    arr_ptr = m.u64(base + HERO_ARRAY_RVA)
    heroes = []
    for i in range(10):
        hp = m.u64(arr_ptr + i * 8)
        if not is_heap(hp): continue
        name = m.string(hp + 0x4328, 32) or f"h{i}"
        pos = m.vec3(hp + 0x25C)
        heroes.append({"ptr": hp, "name": name, "pos": pos, "idx": i})

    print(f"Found {len(heroes)} heroes")
    test_hero = heroes[0]  # LeeSin
    print(f"Testing with: {test_hero['name']} at ({test_hero['pos'][0]:.0f}, {test_hero['pos'][2]:.0f})")

    # ================================================================
    # PHASE 1: Scan hero struct for AiManager-like pointers
    # Try both direct and double-deref patterns
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: Scanning hero struct 0x3000-0x5000 for AiManager candidates")
    print("=" * 70)

    candidates = []

    for off in range(0x3000, 0x5000, 8):
        ptr = m.u64(test_hero["ptr"] + off)
        if not is_heap(ptr): continue

        # Pattern A: Direct AiManager
        data = m.read(ptr, 0x500)
        if data:
            score_a, det_a = score_aimgr(data, test_hero["pos"])
            if score_a >= 15:
                candidates.append({
                    "hero_off": off, "ptr": ptr, "mode": "direct",
                    "score": score_a, "details": det_a
                })

        # Pattern B: LeagueObfuscation double-deref (+0x10)
        inner = m.u64(ptr + 0x10)
        if is_heap(inner):
            data_b = m.read(inner, 0x500)
            if data_b:
                score_b, det_b = score_aimgr(data_b, test_hero["pos"])
                if score_b >= 15:
                    candidates.append({
                        "hero_off": off, "ptr": inner, "mode": "inner+0x10",
                        "score": score_b, "details": det_b
                    })

        # Pattern C: Inner at +0x08
        inner2 = m.u64(ptr + 0x08)
        if is_heap(inner2) and inner2 != inner:
            data_c = m.read(inner2, 0x500)
            if data_c:
                score_c, det_c = score_aimgr(data_c, test_hero["pos"])
                if score_c >= 15:
                    candidates.append({
                        "hero_off": off, "ptr": inner2, "mode": "inner+0x08",
                        "score": score_c, "details": det_c
                    })

    # Sort by score
    candidates.sort(key=lambda c: c["score"], reverse=True)

    print(f"\nFound {len(candidates)} candidates with score >= 15:")
    for c in candidates[:20]:
        print(f"\n  hero+0x{c['hero_off']:X} [{c['mode']}] score={c['score']}")
        print(f"    ptr=0x{c['ptr']:X}")
        for k, v in c["details"].items():
            print(f"    {k}: {v}")

    if not candidates:
        print("No candidates found! Trying wider scan with lower threshold...")
        for off in range(0x2000, 0x6000, 8):
            ptr = m.u64(test_hero["ptr"] + off)
            if not is_heap(ptr): continue
            data = m.read(ptr, 0x500)
            if data:
                score, det = score_aimgr(data, test_hero["pos"])
                if score >= 5:
                    print(f"  hero+0x{off:X} [direct] score={score} {det}")
            inner = m.u64(ptr + 0x10)
            if is_heap(inner):
                data_b = m.read(inner, 0x500)
                if data_b:
                    score, det = score_aimgr(data_b, test_hero["pos"])
                    if score >= 5:
                        print(f"  hero+0x{off:X} [inner+0x10] score={score} {det}")

    # ================================================================
    # PHASE 2: Validate top candidates across all heroes
    # ================================================================
    if candidates:
        print(f"\n\n{'='*70}")
        print("PHASE 2: Validating top 3 candidates across all heroes")
        print("=" * 70)

        for c in candidates[:3]:
            off = c["hero_off"]
            mode = c["mode"]
            print(f"\n  --- hero+0x{off:X} [{mode}] ---")

            for hero in heroes:
                raw_ptr = m.u64(hero["ptr"] + off)
                if not is_heap(raw_ptr):
                    print(f"    {hero['name']:12s} RAW=NULL"); continue

                if "inner" in mode:
                    inner_off = int(mode.split("+")[1], 16)
                    aimgr_ptr = m.u64(raw_ptr + inner_off)
                    if not is_heap(aimgr_ptr):
                        print(f"    {hero['name']:12s} INNER=NULL"); continue
                else:
                    aimgr_ptr = raw_ptr

                data = m.read(aimgr_ptr, 0x500)
                if not data: continue

                score, det = score_aimgr(data, hero["pos"])
                det_str = ", ".join(f"{k}={v}" for k, v in det.items())
                print(f"    {hero['name']:12s} score={score:3d} {det_str}")

    # ================================================================
    # PHASE 3: For the best candidate, do temporal analysis
    # ================================================================
    if candidates:
        best = candidates[0]
        off = best["hero_off"]
        mode = best["mode"]

        print(f"\n\n{'='*70}")
        print(f"PHASE 3: Temporal analysis of best candidate: hero+0x{off:X} [{mode}]")
        print("=" * 70)
        print("Taking 6 snapshots over 12 seconds...")

        snapshots = []
        for t in range(6):
            snap = {}
            for hero in heroes:
                raw_ptr = m.u64(hero["ptr"] + off)
                if not is_heap(raw_ptr): continue

                if "inner" in mode:
                    inner_off_val = int(mode.split("+")[1], 16)
                    aimgr_ptr = m.u64(raw_ptr + inner_off_val)
                    if not is_heap(aimgr_ptr): continue
                else:
                    aimgr_ptr = raw_ptr

                hero_pos = m.vec3(hero["ptr"] + 0x25C)
                data = m.read(aimgr_ptr, 0x500)
                snap[hero["name"]] = {"pos": hero_pos, "data": data, "aimgr": aimgr_ptr}
            snapshots.append(snap)
            if t < 5:
                print(f"  Snapshot {t}...")
                time.sleep(2.5)

        # Analyze changes for key offsets
        key_offsets = [0x000, 0x00C, 0x018, 0x024, 0x030, 0x034, 0x040,
                       0x048, 0x050, 0x058, 0x060, 0x068, 0x070, 0x078,
                       0x080, 0x088, 0x090, 0x098, 0x0A0,
                       0x318, 0x31C, 0x320, 0x324, 0x328, 0x32C,
                       0x330, 0x33C, 0x348, 0x350, 0x354,
                       0x360, 0x384, 0x3A8,
                       0x474, 0x480]

        for hero in heroes[:5]:
            all_data = [s.get(hero["name"], {}).get("data") for s in snapshots]
            all_pos = [s.get(hero["name"], {}).get("pos") for s in snapshots]
            if not all(d is not None for d in all_data): continue

            first_pos = all_pos[0]
            last_pos = all_pos[-1]
            if first_pos and last_pos:
                pos_dist = ((last_pos[0]-first_pos[0])**2 + (last_pos[2]-first_pos[2])**2)**0.5
            else:
                pos_dist = 0

            print(f"\n  {hero['name']} [moved {pos_dist:.0f}]:")

            for koff in key_offsets:
                if koff + 12 > len(all_data[0]): continue
                vals = [struct.unpack("<fff", d[koff:koff+12]) for d in all_data]
                max_delta = max(
                    abs(vals[t][c] - vals[0][c])
                    for t in range(1, 6) for c in range(3)
                )

                is_map = all(is_map_pos(v) for v in vals)
                v0 = vals[0]; vn = vals[-1]

                if max_delta > 0.5 or is_map:
                    tag = ""
                    if is_map and max_delta > 0.5: tag = "CHANGING_MAP"
                    elif is_map: tag = "STATIC_MAP"
                    elif max_delta > 0.5: tag = "CHANGING_NON_MAP"

                    # Also show as single floats for velocity/flags
                    f0 = struct.unpack("<f", all_data[0][koff:koff+4])[0]
                    fn = struct.unpack("<f", all_data[-1][koff:koff+4])[0]

                    if is_map:
                        print(f"    +0x{koff:03X}: ({v0[0]:8.1f},{v0[2]:8.1f}) -> ({vn[0]:8.1f},{vn[2]:8.1f})  delta={max_delta:.1f}  [{tag}]")
                    else:
                        print(f"    +0x{koff:03X}: f32={f0:.2f} -> {fn:.2f}  delta={max_delta:.1f}  [{tag}]")

        # Also scan ALL changing offsets (not just key ones)
        print(f"\n\n  --- Full scan of all changing map positions ---")
        for hero in heroes[:3]:
            all_data = [s.get(hero["name"], {}).get("data") for s in snapshots]
            if not all(d is not None for d in all_data): continue

            first_pos = snapshots[0].get(hero["name"], {}).get("pos")
            last_pos = snapshots[-1].get(hero["name"], {}).get("pos")
            if first_pos and last_pos:
                pos_dist = ((last_pos[0]-first_pos[0])**2 + (last_pos[2]-first_pos[2])**2)**0.5
            else:
                pos_dist = 0

            print(f"\n  {hero['name']} [moved {pos_dist:.0f}]:")

            for off_scan in range(0, 0x4F0, 4):
                vals = [struct.unpack("<fff", d[off_scan:off_scan+12]) for d in all_data]
                max_delta = max(
                    abs(vals[t][c] - vals[0][c])
                    for t in range(1, 6) for c in range(3)
                )
                if max_delta > 1.0 and all(is_map_pos(v) for v in vals):
                    v0 = vals[0]; vn = vals[-1]
                    hp0 = snapshots[0].get(hero["name"], {}).get("pos") or (0,0,0)
                    dist0 = ((v0[0]-hp0[0])**2 + (v0[2]-hp0[2])**2)**0.5
                    tag = "SRVPOS" if dist0 < 10 else "TARGET?" if dist0 < 2000 else "FAR"
                    print(f"    +0x{off_scan:03X}: ({v0[0]:8.1f},{v0[2]:8.1f}) -> ({vn[0]:8.1f},{vn[2]:8.1f})  delta={max_delta:.1f} distFromHero={dist0:.0f} [{tag}]")

    # ================================================================
    # PHASE 4: Also scan for ActiveSpell — try different pointer chains
    # ================================================================
    print(f"\n\n{'='*70}")
    print("PHASE 4: ActiveSpell offset scan")
    print("=" * 70)

    # Check various offsets around 0x4578 for ActiveSpell
    for hero in heroes[:3]:
        print(f"\n  {hero['name']}:")
        for spell_off in range(0x4500, 0x4700, 8):
            spell_ptr = m.u64(hero["ptr"] + spell_off)
            if not is_heap(spell_ptr): continue

            # Try to read a spell name at various sub-offsets
            for name_chain in [
                # Direct name
                [(0x28, None)],
                [(0x30, None)],
                [(0x38, None)],
                # SpellInfo chain: ptr -> SpellInfo -> name
                [(0x038, 0x28)],
                [(0x038, 0x30)],
                [(0x008, 0x28)],
                [(0x010, 0x28)],
                [(0x018, 0x28)],
                [(0x020, 0x28)],
            ]:
                final_ptr = spell_ptr
                for step_off, sub_off in name_chain:
                    if sub_off is not None:
                        intermediate = m.u64(final_ptr + step_off)
                        if is_heap(intermediate):
                            name = m.string(intermediate + sub_off, 48)
                            if name and len(name) > 3 and all(c.isalnum() or c in '_' for c in name):
                                # Found a spell name!
                                cast_pos = m.vec3(spell_ptr + 0x108)
                                cast_time = m.f32(spell_ptr + 0x1E8)
                                cp_str = f"({cast_pos[0]:.0f},{cast_pos[2]:.0f})" if cast_pos and is_map_pos(cast_pos) else "N/A"
                                ct_str = f"{cast_time:.1f}" if cast_time and 0 < cast_time < 5000 else "?"
                                print(f"    hero+0x{spell_off:X} -> +0x{step_off:X} -> +0x{sub_off:X}: '{name}' pos={cp_str} t={ct_str}")
                    else:
                        name = m.string(final_ptr + step_off, 48)
                        if name and len(name) > 3 and all(c.isalnum() or c in '_' for c in name):
                            print(f"    hero+0x{spell_off:X} -> +0x{step_off:X}: '{name}'")

    # Also try SpellBook approach: hero + 0x30E8 -> ActiveSpellCast at +0x38
    print(f"\n  --- SpellBook approach (hero+0x30E8) ---")
    for hero in heroes[:5]:
        sb = m.u64(hero["ptr"] + 0x30E8)
        if not is_heap(sb): continue

        # ActiveSpellCast = SpellBook + 0x38 (i.e., hero + 0x3120)
        active = m.u64(hero["ptr"] + 0x3120)  # Direct from hero base
        if not is_heap(active):
            active = m.u64(sb + 0x38)  # Via spellbook
        if not is_heap(active): continue

        # Scan the active spell struct for name strings
        data = m.read(active, 0x200)
        if not data: continue

        # Look for pointers to name strings
        for sub_off in range(0, 0x100, 8):
            ptr = struct.unpack("<Q", data[sub_off:sub_off+8])[0]
            if is_heap(ptr):
                name = m.string(ptr, 48)
                if name and len(name) > 3 and all(c.isalnum() or c in '_' for c in name):
                    print(f"    {hero['name']:12s} hero+0x3120 -> +0x{sub_off:X} -> '{name}'")

                # Also deref one more level
                ptr2 = m.u64(ptr)
                if is_heap(ptr2):
                    name2 = m.string(ptr2, 48)
                    if name2 and len(name2) > 3 and all(c.isalnum() or c in '_' for c in name2):
                        print(f"    {hero['name']:12s} hero+0x3120 -> +0x{sub_off:X} -> deref -> '{name2}'")

                # Try name at ptr + various offsets
                for n_off in [0x28, 0x30, 0x38, 0x40]:
                    name3 = m.string(ptr + n_off, 48)
                    if name3 and len(name3) > 3 and all(c.isalnum() or c in '_' for c in name3):
                        print(f"    {hero['name']:12s} hero+0x3120 -> +0x{sub_off:X} -> +0x{n_off:X} -> '{name3}'")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
