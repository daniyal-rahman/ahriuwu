#!/usr/bin/env python3
"""Automated offset scanner for League of Legends replay pipeline.
Run once per patch to discover all memory offsets.

Input: Running replay + Replay API on port 2999
Output: offsets.json with verified offsets for current patch

Usage:
    python scripts/scan_offsets.py [--anchors prev_offsets.json]
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, time, json
import ssl, urllib.request, math, os, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
_k = ctypes.windll.kernel32

# ── Hacker log anchors (starting guesses, will be adjusted) ──
DEFAULT_ANCHORS = {
    "hero_array":     0x1DBEEE8,  # will scan ±0x10000
    "game_time":      0x1DCD1E0,
    "position":       0x25C,
    "hp_current":     0x1080,
    "hp_max":         0x10A8,
    "gold_current":   0x2830,
    "gold_earned":    0x2858,
    "champion_name":  0x4330,     # hacker value (we found 0x4328)
    "level":          0x4D18,     # hacker value (we found 0x4D10)
    "vision_score":   0x55E0,     # hacker value (we found 0x55D8)
    "active_spell":   0x3120,
    "spellbook":      0x30E8,
    "slot_array":     0xAE0,
    "stats_base":     0x1B88,
}

# ── Memory access ──
class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong), ("__alignment1", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong),
                ("__alignment2", ctypes.c_ulong)]
_k.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
_k.VirtualQueryEx.restype = ctypes.c_size_t

class Mem:
    def __init__(self, pid): self.h = _k.OpenProcess(0x0410, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u64(self, a): d=self.read(a,8); return struct.unpack('<Q',d)[0] if d else None
    def u32(self, a): d=self.read(a,4); return struct.unpack('<I',d)[0] if d else None
    def f32(self, a): d=self.read(a,4); return struct.unpack('<f',d)[0] if d else None
    def vec3(self, a): d=self.read(a,12); return struct.unpack('<fff',d) if d and len(d)==12 else None
    def string(self, a, n=80):
        d=self.read(a,n)
        if not d: return None
        try: return d.split(b'\x00')[0].decode('ascii')
        except: return None

def enum_regions(h):
    addr = 0; mbi = MBI(); regions = []
    while addr < 0x7FFFFFFFFFFF:
        rc = _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi))
        if not rc: break
        a = mbi.BaseAddress or 0; sz = mbi.RegionSize or 0
        if sz == 0: break
        if mbi.State == 0x1000 and (mbi.Protect & 0xEE) and not (mbi.Protect & 0x100):
            regions.append((a, sz))
        addr = a + sz
    return regions

def rpost(ep, d):
    return json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://127.0.0.1:2999{ep}", data=json.dumps(d).encode(),
        headers={"Content-Type":"application/json"}), context=_ctx, timeout=5).read())
def rget(ep):
    return json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://127.0.0.1:2999{ep}"), context=_ctx, timeout=5).read())

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'], capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
def find_base(pid):
    class ME(ctypes.Structure):
        _fields_=[("dwSize",ctypes.c_ulong),("a",ctypes.c_ulong),("b",ctypes.c_ulong),
            ("c",ctypes.c_ulong),("d",ctypes.c_ulong),("modBaseAddr",ctypes.POINTER(ctypes.c_byte)),
            ("modBaseSize",ctypes.c_ulong),("hModule",ctypes.c_void_p),
            ("szModule",ctypes.c_char*256),("szExePath",ctypes.c_char*260)]
    snap=_k.CreateToolhelp32Snapshot(0x18,pid);me=ME();me.dwSize=ctypes.sizeof(ME)
    if _k.Module32First(snap,ctypes.byref(me)):
        while True:
            if b'league' in me.szModule.lower():
                _k.CloseHandle(snap)
                return ctypes.cast(me.modBaseAddr,ctypes.c_void_p).value, me.modBaseSize
            if not _k.Module32Next(snap,ctypes.byref(me)): break
    return None, None

CHAMP_NAMES = {"Garen","Mordekaiser","Graves","Viktor","Kaisa","Rell","Naafiri",
    "Brand","Ezreal","Alistar","LeeSin","Karthus","Hwei","Smolder","Nami",
    "Malphite","Azir","Ashe","Seraphine","Aatrox","Ahri","Akali","Amumu",
    "Anivia","Annie","Aphelios","Bard","Blitzcrank","Braum","Caitlyn",
    "Camille","Cassiopeia","Corki","Darius","Diana","DrMundo","Draven",
    "Ekko","Elise","Evelynn","Fiora","Fizz","Galio","Gangplank","Gragas",
    "Hecarim","Heimerdinger","Illaoi","Irelia","Ivern","Janna","JarvanIV",
    "Jax","Jayce","Jhin","Jinx","Kalista","Karma","Kassadin","Katarina",
    "Kayn","Kennen","Khazix","Kindred","Kled","KogMaw","Leblanc","Lillia",
    "Lissandra","Lucian","Lulu","Lux","Malzahar","Maokai","MasterYi",
    "MissFortune","Morgana","Nasus","Nautilus","Neeko","Nidalee","Nocturne",
    "Nunu","Olaf","Orianna","Ornn","Pantheon","Poppy","Pyke","Qiyana",
    "Quinn","Rakan","Rammus","RekSai","Rell","Renata","Renekton","Rengar",
    "Riven","Rumble","Ryze","Samira","Sejuani","Senna","Seraphine","Sett",
    "Shaco","Shen","Shyvana","Singed","Sion","Sivir","Skarner","Sona",
    "Soraka","Swain","Sylas","Syndra","TahmKench","Taliyah","Talon","Taric",
    "Teemo","Thresh","Tristana","Trundle","Tryndamere","TwistedFate",
    "Twitch","Udyr","Urgot","Varus","Vayne","Veigar","Velkoz","Vex",
    "Vi","Viego","Vladimir","Volibear","Warwick","Wukong","Xayah",
    "Xerath","XinZhao","Yasuo","Yone","Yorick","Yuumi","Zac","Zed",
    "Zeri","Ziggs","Zilean","Zoe","Zyra"}
LO, HI = 0x10000000, 0x7FFFFFFFFFFF

# ═══════════════════════════════════════════════════════════════
# Phase 1: Find hero_array RVA and game_time RVA
# ═══════════════════════════════════════════════════════════════
def find_hero_array(m, base, mod_size, anchor, name_anchor=0x4328):
    """Scan module data section for hero array pointer."""
    print("\n[Phase 1a] Finding hero_array RVA...")
    # Widened to ±0x200000 — modern patches can shift globals by >64KB
    # Candidate champion_name offsets: anchor ±0x80 in 8-byte steps
    name_offsets = [name_anchor + d for d in range(-0x80, 0x88, 8)]
    for drift in range(0, 0x200000, 8):
        for rva in [anchor + drift, anchor - drift]:
            if rva < 0 or rva > mod_size: continue
            ptr = m.u64(base + rva)
            if not ptr or not (LO < ptr < HI): continue
            # Check if it points to an array of 10 hero-like structs
            heroes_found = set()  # distinct names
            for i in range(10):
                hp = m.u64(ptr + i*8)
                if not hp or not (LO < hp < HI): continue
                for name_off in name_offsets:
                    name = m.string(hp + name_off, 40)
                    if name and name in CHAMP_NAMES:
                        heroes_found.add((i, name))
                        break
            # Require ≥8 distinct slots with champion names
            distinct_names = set(n for _, n in heroes_found)
            if len(heroes_found) >= 8 and len(distinct_names) == len(heroes_found):
                print(f"  ✓ hero_array at RVA 0x{rva:X} ({len(heroes_found)}/10 distinct heroes)")
                return rva
    print("  ✗ hero_array not found")
    return None

def find_game_time(m, base, mod_size, anchor, api_gt):
    """Scan module data section for game time float."""
    print("\n[Phase 1b] Finding game_time RVA...")
    for drift in range(0, 0x200000, 4):
        for rva in [anchor + drift, anchor - drift]:
            if rva < 0 or rva > mod_size: continue
            v = m.f32(base + rva)
            if v and abs(v - api_gt) < 1.0:
                print(f"  ✓ game_time at RVA 0x{rva:X} (read={v:.1f}, api={api_gt:.1f})")
                return rva
    print("  ✗ game_time not found")
    return None

# ═══════════════════════════════════════════════════════════════
# Phase 2: Find hero struct fields via API cross-check
# ═══════════════════════════════════════════════════════════════
def _get_hero_and_api(m, base, hero_array_rva, name_off, target_name="Garen"):
    """Helper: get hero ptr + API data for target champion."""
    arr_ptr = m.u64(base + hero_array_rva)
    hp = None
    for i in range(10):
        h = m.u64(arr_ptr + i*8)
        if h and m.string(h + name_off, 40) == target_name:
            hp = h; break
    if not hp:
        for i in range(10):
            h = m.u64(arr_ptr + i*8)
            if h: hp = h; break
    pl = rget("/liveclientdata/playerlist")
    api = next((p for p in pl if p.get("championName") == target_name), pl[0] if pl else None)
    return hp, api

def find_hero_fields(m, base, hero_array_rva, anchors):
    """Find hero struct fields using 2-time-point verification."""
    print("\n[Phase 2] Finding hero struct fields (multi-time verification)...")
    arr_ptr = m.u64(base + hero_array_rva)

    # Find champion_name offset
    name_off = None
    for drift in range(-16, 24, 4):
        test_off = anchors["champion_name"] + drift
        found = 0
        for i in range(10):
            hp = m.u64(arr_ptr + i*8)
            if hp:
                name = m.string(hp + test_off, 40)
                if name and name in CHAMP_NAMES: found += 1
        if found >= 5:
            name_off = test_off
            print(f"  ✓ champion_name at hero+0x{name_off:X} ({found}/10 match)")
            break
    if not name_off:
        print("  ✗ champion_name not found"); return {}

    results = {"champion_name": name_off}

    # Capture at TWO time points — read ALL candidate values IMMEDIATELY
    # (hero pointers go stale after seek, so must read at capture time)
    SEARCH_RANGE = 96  # ±96 bytes from anchor (widened for non-uniform patch shifts)
    candidate_offsets = {}
    for field, anchor_key, dtype in [
        ("position", "position", "vec3"), ("level", "level", "u32"),
        ("hp_current", "hp_current", "f32"), ("hp_max", "hp_max", "f32"),
        ("gold_current", "gold_current", "f32"), ("gold_earned", "gold_earned", "f32"),
        ("vision_score", "vision_score", "f32"), ("active_spell", "active_spell", "u64"),
    ]:
        for drift in range(-SEARCH_RANGE, SEARCH_RANGE+4, 4 if dtype != "u64" else 8):
            candidate_offsets.setdefault(field, []).append(anchors.get(anchor_key, 0) + drift)

    snapshots = []
    for gt_target in [200, 600]:
        rpost("/replay/playback", {"time": float(gt_target), "speed": 0.0, "paused": True})
        time.sleep(1.5)
        rpost("/replay/playback", {"speed": 1.0, "paused": False})
        time.sleep(2)
        rpost("/replay/playback", {"speed": 0.0, "paused": True})
        time.sleep(0.3)
        hp, api = _get_hero_and_api(m, base, hero_array_rva, name_off)
        gt_rva = anchors.get("game_time", 0x1DCD1E0)
        gt = m.f32(base + gt_rva) or 0
        if not hp or not api: continue
        sc = api.get("scores", {})
        snap = {"api": api, "gt": gt, "vals": {}}
        # Read ALL candidate offsets NOW while hero pointer is fresh
        for field, offsets_list in candidate_offsets.items():
            for off in offsets_list:
                if field == "position":
                    v = m.vec3(hp + off)
                elif field in ("level",):
                    v = m.u32(hp + off)
                elif field == "active_spell":
                    v = m.u64(hp + off)
                else:
                    v = m.f32(hp + off)
                snap["vals"][(field, off)] = v
        # Also read active spell name chain
        for off in candidate_offsets.get("active_spell", []):
            ptr = m.u64(hp + off)
            if ptr and LO < ptr < HI:
                si = m.u64(ptr + 0x8)
                if si and LO < si < HI:
                    np_ = m.u64(si + 0x28)
                    if np_ and LO < np_ < HI:
                        sn = m.string(np_)
                        if sn and len(sn) > 3:
                            snap["vals"][("active_spell_name", off)] = sn
        snapshots.append(snap)
        print(f"  snapshot gt={gt:.0f}: level={api.get('level')} ward={sc.get('wardScore',0):.1f}")

    if len(snapshots) < 2:
        print("  ✗ Could not get 2 snapshots"); return results
    s1, s2 = snapshots[0], snapshots[1]

    # Position
    for off in candidate_offsets["position"]:
        v1 = s1["vals"].get(("position", off))
        v2 = s2["vals"].get(("position", off))
        if (v1 and v2 and 100 < v1[0] < 16000 and 30 < v1[1] < 200 and 100 < v1[2] < 16000
                and 100 < v2[0] < 16000):
            results["position"] = off
            print(f"  ✓ position at hero+0x{off:X}")
            break

    # Level: match API at BOTH times, values must differ
    al1, al2 = s1["api"].get("level",0), s2["api"].get("level",0)
    if al1 > 0 and al2 > 0 and al1 != al2:
        for off in candidate_offsets["level"]:
            v1 = s1["vals"].get(("level", off))
            v2 = s2["vals"].get(("level", off))
            if v1 == al1 and v2 == al2:
                results["level"] = off
                print(f"  ✓ level at hero+0x{off:X}: {v1}→{v2} (api {al1}→{al2})")
                break

    # HP
    for off in candidate_offsets["hp_current"]:
        v1 = s1["vals"].get(("hp_current", off))
        v2 = s2["vals"].get(("hp_current", off))
        if v1 and v2 and 50 < v1 < 10000 and 50 < v2 < 10000:
            results["hp_current"] = off
            for off2 in candidate_offsets["hp_max"]:
                vm1 = s1["vals"].get(("hp_max", off2))
                vm2 = s2["vals"].get(("hp_max", off2))
                if vm1 and vm2 and vm1 >= v1*0.8 and vm2 >= v2*0.8 and off2 != off:
                    results["hp_max"] = off2
                    print(f"  ✓ hp at hero+0x{off:X}/0x{off2:X}: {v1:.0f}/{vm1:.0f} → {v2:.0f}/{vm2:.0f}")
                    break
            break

    # Gold earned: must increase
    for off in candidate_offsets["gold_earned"]:
        v1 = s1["vals"].get(("gold_earned", off))
        v2 = s2["vals"].get(("gold_earned", off))
        if v1 is not None and v2 is not None and v2 > v1 > 100:
            results["gold_earned"] = off
            print(f"  ✓ gold_earned at hero+0x{off:X}: {v1:.0f}→{v2:.0f}")
            break

    # Gold current: plausible range
    for off in candidate_offsets["gold_current"]:
        v1 = s1["vals"].get(("gold_current", off))
        v2 = s2["vals"].get(("gold_current", off))
        if v1 is not None and v2 is not None and 0 < v1 < 20000 and 0 < v2 < 20000:
            results["gold_current"] = off
            print(f"  ✓ gold_current at hero+0x{off:X}: {v1:.0f}→{v2:.0f}")
            break

    # Vision score: match API wardScore (require at least one > 0)
    ws1 = s1["api"].get("scores",{}).get("wardScore",0)
    ws2 = s2["api"].get("scores",{}).get("wardScore",0)
    if ws2 > 0:
        for off in candidate_offsets["vision_score"]:
            v1 = s1["vals"].get(("vision_score", off))
            v2 = s2["vals"].get(("vision_score", off))
            if v2 and abs(v2 - ws2) < 1.0:
                if ws1 > 0 and v1 and abs(v1 - ws1) < 1.0:
                    results["vision_score"] = off
                    print(f"  ✓ vision_score at hero+0x{off:X}: {v1:.1f}→{v2:.1f} (api {ws1:.1f}→{ws2:.1f})")
                    break
                elif ws1 == 0:
                    results["vision_score"] = off
                    print(f"  ✓ vision_score at hero+0x{off:X}: ?→{v2:.1f} (api ?→{ws2:.1f})")
                    break

    # Active spell — check both snapshots (hero might not be casting at one)
    for snap in [s2, s1]:
        found = False
        for off in candidate_offsets["active_spell"]:
            sn = snap["vals"].get(("active_spell_name", off))
            if sn:
                results["active_spell"] = off
                print(f"  ✓ active_spell at hero+0x{off:X}: {sn}")
                found = True; break
        if found: break
    if "active_spell" not in results:
        # Fallback: if spellbook anchor is known, active_spell = spellbook + 0x38
        sb_anchor = anchors.get("spellbook", 0x30E8)
        fallback = sb_anchor + 0x38
        results["active_spell"] = fallback
        print(f"  ~ active_spell at hero+0x{fallback:X} (fallback: spellbook+0x38)")

    return results

# ═══════════════════════════════════════════════════════════════
# Phase 3: Find SpellBook
# ═══════════════════════════════════════════════════════════════
def find_spellbook(m, garen_hp, anchors, offsets=None):
    """Find inline spellbook by looking for spell name deref chains."""
    print("\n[Phase 3] Finding SpellBook...")
    expected = {"GarenQ", "GarenW", "GarenE", "GarenR"}
    for sb_drift in range(-0x100, 0x100, 8):
        sb_off = anchors["spellbook"] + sb_drift
        for sa_drift in range(-0x60, 0x60, 8):
            sa_off = anchors["slot_array"] + sa_drift
            # Check if hero+sb_off+sa_off is an array of spell slot pointers
            found_names = set()
            best_info_off = None
            for i in range(4):
                slot_ptr = m.u64(garen_hp + sb_off + sa_off + i*8)
                if not slot_ptr or not (LO < slot_ptr < HI): continue
                for info_off in [0x128, 0x130, 0x120, 0x138]:
                    si = m.u64(slot_ptr + info_off)
                    if not si or not (LO < si < HI): continue
                    np = m.u64(si + 0x28)
                    if not np or not (LO < np < HI): continue
                    sn = m.string(np)
                    if sn and sn in expected:
                        found_names.add(sn)
                        best_info_off = info_off
                        break
            spells_found = len(found_names)
            if spells_found >= 4:
                # Cross-check: spellbook + 0x38 should == active_spell offset
                # This disambiguates the base vs slot_array split
                active_spell_off = offsets.get("active_spell")
                if active_spell_off and sb_off + 0x38 != active_spell_off:
                    # Wrong split — recalculate
                    correct_sb = active_spell_off - 0x38
                    correct_sa = (sb_off + sa_off) - correct_sb
                    print(f"  ✓ spellbook at hero+0x{correct_sb:X} (fixed via active_spell), slot_array at +0x{correct_sa:X} ({spells_found}/4 spells)")
                    sb_off, sa_off = correct_sb, correct_sa
                else:
                    print(f"  ✓ spellbook at hero+0x{sb_off:X}, slot_array at +0x{sa_off:X} ({spells_found}/4 spells)")
                # Find the spell_info offset within slot
                slot0 = m.u64(garen_hp + sb_off + sa_off)
                for info_off in [0x128, 0x130, 0x120, 0x138]:
                    si = m.u64(slot0 + info_off)
                    if si and LO < si < HI:
                        np = m.u64(si + 0x28)
                        if np and LO < np < HI:
                            sn = m.string(np)
                            if sn and sn.startswith("Garen"):
                                print(f"    slot_spell_info at +0x{info_off:X}")
                                return {"spellbook": sb_off, "slot_array": sa_off, "slot_spell_info": info_off}
                return {"spellbook": sb_off, "slot_array": sa_off}
    print("  ✗ SpellBook not found")
    return {}

# ═══════════════════════════════════════════════════════════════
# Phase 4: Find click destination via global scan
# ═══════════════════════════════════════════════════════════════
def find_click_dest(m, base, garen_hp):
    """Play a known movement, global scan for destination vec3."""
    print("\n[Phase 4] Finding click destination...")
    # Play from gt=35, observe where Garen stops
    rpost("/replay/playback", {"time": 35.0, "speed": 0.0, "paused": True})
    time.sleep(1.5)
    rpost("/replay/playback", {"speed": 1.0, "paused": False})

    # Sample position at 2Hz for 15s to find stop point
    positions = []
    t0 = time.time()
    while time.time() - t0 < 15:
        pos = m.vec3(garen_hp + 0x25C)  # use position we already know
        if pos and 100 < pos[0] < 16000:
            positions.append((round(pos[0],1), round(pos[2],1)))
        time.sleep(0.5)

    # Find where Garen stopped (consecutive same position)
    dest = None
    for i in range(len(positions)-2):
        if positions[i] == positions[i+1] == positions[i+2]:
            dest = positions[i]
            break
    if not dest:
        dest = positions[-1] if positions else None

    if not dest:
        print("  ✗ Could not determine stop position")
        rpost("/replay/playback", {"speed": 0.0, "paused": True})
        return {}

    print(f"  Garen stops at ({dest[0]}, {dest[1]})")

    # Replay the movement, pause mid-way, global scan for dest
    rpost("/replay/playback", {"time": 36.0, "speed": 0.0, "paused": True})
    time.sleep(1.5)
    rpost("/replay/playback", {"speed": 1.0, "paused": False})
    time.sleep(2)
    rpost("/replay/playback", {"speed": 0.0, "paused": True})
    time.sleep(0.3)

    dest_x, dest_z = dest
    BUF = 30
    regions = enum_regions(m.h)
    print(f"  Scanning {len(regions)} regions for ({dest_x}±{BUF}, ?, {dest_z}±{BUF})...")

    hits = []
    for ra, rsz in regions:
        data = m.read(ra, rsz)
        if not data: continue
        for i in range(0, len(data)-12, 4):
            x = struct.unpack_from('<f', data, i)[0]
            if not (dest_x - BUF < x < dest_x + BUF): continue
            z = struct.unpack_from('<f', data, i+8)[0]
            if not (dest_z - BUF < z < dest_z + BUF): continue
            y = struct.unpack_from('<f', data, i+4)[0]
            hits.append((ra + i, round(x,1), round(y,1), round(z,1)))

    print(f"  {len(hits)} global hits for destination")

    if not hits:
        return {}

    # Backref: find what points to these addresses
    for dest_addr, x, y, z in hits[:10]:
        for offset_guess in [0x00, 0x10, 0x20, 0x30, 0x34, 0x40]:
            struct_base = dest_addr - offset_guess
            needle = struct.pack('<Q', struct_base)
            # Check module data section
            mod_data = m.read(base, 0x2000000)  # first 32MB of module
            if mod_data:
                idx = mod_data.find(needle)
                if idx >= 0 and idx % 8 == 0:
                    print(f"  ✓ click_dest: module+0x{idx:X} → struct+0x{offset_guess:X} = dest ({x},{z})")
                    return {"click_dest_rva": idx, "click_dest_off": offset_guess}

    print("  ✗ No stable pointer chain found for click destination")
    return {}

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Automated offset scanner")
    parser.add_argument("--anchors", help="Previous offsets JSON for drift-based search")
    parser.add_argument("--output", "-o", default="offsets.json")
    args = parser.parse_args()

    anchors = DEFAULT_ANCHORS.copy()
    if args.anchors:
        with open(args.anchors) as f:
            anchors.update(json.load(f))

    print("=== League Offset Scanner ===")

    pid = find_pid()
    if not pid: print("ERROR: League not running"); return
    base, mod_size = find_base(pid)
    m = Mem(pid)
    print(f"PID={pid} base=0x{base:x} mod_size=0x{mod_size:x}")

    # Wait for replay API
    gt = rget("/liveclientdata/gamestats").get("gameTime", 0)
    print(f"Game time: {gt:.1f}s")

    # Play to gt=100 for stable state (heroes in lane)
    rpost("/replay/playback", {"time": 100.0, "speed": 0.0, "paused": True})
    time.sleep(2)
    rpost("/replay/playback", {"speed": 1.0, "paused": False})
    time.sleep(2)
    rpost("/replay/playback", {"speed": 0.0, "paused": True})
    time.sleep(0.5)
    gt = rget("/liveclientdata/gamestats").get("gameTime", 0)
    print(f"Stabilized at gt={gt:.1f}")

    offsets = {"_scanner_version": 1, "_patch_mod_size": mod_size}

    # Phase 1: RVAs
    hero_rva = find_hero_array(m, base, mod_size, anchors["hero_array"],
                                name_anchor=anchors.get("champion_name", 0x4328))
    if hero_rva: offsets["hero_array"] = hero_rva

    gt_rva = find_game_time(m, base, mod_size, anchors["game_time"], gt)
    if gt_rva: offsets["game_time"] = gt_rva

    if not hero_rva:
        print("\nFATAL: Cannot find hero array. Aborting.")
        return

    # Phase 2: Hero fields (uses 2 time points internally)
    if gt_rva:
        anchors["game_time"] = gt_rva  # pass verified RVA for internal use
    hero_fields = find_hero_fields(m, base, hero_rva, anchors)
    offsets.update(hero_fields)

    # Phase 3: SpellBook — seek to mid-game where spells are leveled
    rpost("/replay/playback", {"time": 600.0, "speed": 0.0, "paused": True})
    time.sleep(1.5)
    rpost("/replay/playback", {"speed": 1.0, "paused": False})
    time.sleep(2)
    rpost("/replay/playback", {"speed": 0.0, "paused": True})
    time.sleep(0.3)
    arr_ptr = m.u64(base + hero_rva)
    garen_hp = None
    name_off = offsets.get("champion_name", 0x4328)
    for i in range(10):
        hp = m.u64(arr_ptr + i*8)
        if hp and m.string(hp + name_off) == "Garen":
            garen_hp = hp; break
    if not garen_hp:
        for i in range(10):
            hp = m.u64(arr_ptr + i*8)
            if hp: garen_hp = hp; break

    if garen_hp:
        sb = find_spellbook(m, garen_hp, anchors, offsets)
        offsets.update(sb)

    # Phase 4: Click destination
    if garen_hp and "position" in offsets:
        click = find_click_dest(m, base, garen_hp)
        offsets.update(click)

    rpost("/replay/playback", {"speed": 0.0, "paused": True})

    # Save
    out_path = args.output
    with open(out_path, "w") as f:
        json.dump(offsets, f, indent=2)
    print(f"\n=== Done! Saved to {out_path} ===")
    for k, v in offsets.items():
        if k.startswith("_"):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: 0x{v:X}" if isinstance(v, int) else f"  {k}: {v}")

if __name__ == "__main__":
    main()
