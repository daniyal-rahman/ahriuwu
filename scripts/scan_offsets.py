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
    """Scan module data section for hero_array RVA.

    Tries TWO layouts:
      - 'deref' (16.8.766 and earlier): m.u64(base+rva) → arr_ptr → m.u64(arr_ptr+i*8) → hero_ptr (stride 8)
      - 'inline' (16.9.772+): hero_i = m.u64(base+rva + i*0x108) (in-place table, stride 0x108)

    Validates by reading champion_name at hero+name_anchor and matching CHAMP_NAMES
    in the live game. Requires ≥8 distinct slot/name pairs.

    Returns (rva, layout, stride) or (None, None, None).
    """
    print("\n[Phase 1a] Finding hero_array RVA...")
    name_offsets = [name_anchor + d for d in range(-0x80, 0x88, 8)]

    def _score(get_hero_ptr_at_index):
        """Returns count of distinct (slot, name) pairs (max 10). Names must be in CHAMP_NAMES."""
        heroes_found = set()
        for i in range(10):
            hp = get_hero_ptr_at_index(i)
            if not hp or not (LO < hp < HI): continue
            for name_off in name_offsets:
                name = m.string(hp + name_off, 40)
                if name and name in CHAMP_NAMES:
                    heroes_found.add((i, name))
                    break
        distinct_names = set(n for _, n in heroes_found)
        # Reject layouts where the same hero appears in multiple slots (false positives)
        return len(heroes_found) if len(distinct_names) == len(heroes_found) else 0

    # Score both layouts at every candidate RVA; return the one with the highest hero count.
    best = (0, None, None, None)  # (score, rva, layout, stride)
    for drift in range(0, 0x200000, 8):
        for rva in [anchor + drift, anchor - drift]:
            if rva < 0 or rva > mod_size: continue

            # Layout A: pointer-to-array, stride 8 (16.8.766 and earlier)
            arr_ptr = m.u64(base + rva)
            if arr_ptr and (LO < arr_ptr < HI):
                s = _score(lambda i: m.u64(arr_ptr + i*8))
                if s > best[0]:
                    best = (s, rva, "deref", 8)
                    if s == 10: break

            # Layout B: in-place, stride 0x108 (16.9.772+)
            s = _score(lambda i: m.u64(base + rva + i*0x108))
            if s > best[0]:
                best = (s, rva, "inline", 0x108)
                if s == 10: break
        if best[0] == 10: break

    if best[0] >= 8:
        score, rva, layout, stride = best
        print(f"  ✓ hero_array at RVA 0x{rva:X} ({layout} layout, stride 0x{stride:X}, {score}/10 heroes)")
        return rva, layout, stride
    print(f"  ✗ hero_array not found (best score {best[0]}/10)")
    return None, None, None

def find_game_time(m, base, mod_size, anchor, api_gt):
    """Scan module for game_time float. Multi-time verified — reject stale snapshots.

    Phase 1: collect ALL f32 candidates near api_gt within drift window.
    Phase 2: sleep 5s while replay plays; pick the one whose value advanced ~5s.
    """
    print("\n[Phase 1b] Finding game_time RVA (multi-time verified)...")
    cands = []
    for drift in range(0, 0x200000, 4):
        for rva in [anchor + drift, anchor - drift]:
            if rva < 0 or rva > mod_size: continue
            v = m.f32(base + rva)
            if v is not None and abs(v - api_gt) < 1.0:
                cands.append((rva, v))
    print(f"  found {len(cands)} candidates near api_gt={api_gt:.1f}")
    if not cands:
        print("  ✗ game_time not found"); return None

    # Wait for replay to advance, then re-read
    SLEEP = 5.0
    print(f"  sleeping {SLEEP}s for replay to advance...")
    time.sleep(SLEEP)
    api_t1 = rget("/liveclientdata/gamestats").get("gameTime", 0)
    delta = api_t1 - api_gt
    print(f"  api advanced by {delta:.2f}s (from {api_gt:.1f} to {api_t1:.1f})")
    if delta < 1.0:
        print("  ⚠ replay paused — picking first match without verification")
        rva, v = cands[0]
        print(f"  ✓ game_time at RVA 0x{rva:X} (read={v:.1f}, api={api_gt:.1f}, UNVERIFIED)")
        return rva
    survivors = []
    for rva, _ in cands:
        v2 = m.f32(base + rva)
        if v2 is None: continue
        if abs(v2 - api_t1) < 1.0:
            survivors.append((rva, v2))
    print(f"  {len(survivors)} survived multi-time check (advanced ~{delta:.1f}s)")
    if not survivors:
        print("  ✗ no live game_time — all candidates were stale snapshots")
        return None
    survivors.sort(key=lambda x: abs(x[1] - api_t1))
    rva, v = survivors[0]
    print(f"  ✓ game_time at RVA 0x{rva:X} (read={v:.2f}, api={api_t1:.2f}, advanced ✓)")
    return rva

# ═══════════════════════════════════════════════════════════════
# Phase 2: Find hero struct fields via API cross-check
# ═══════════════════════════════════════════════════════════════
def _iter_hero_ptrs(m, base, hero_array_rva, layout, stride):
    """Yield hero pointers for a 10-slot hero array, layout-agnostic."""
    if layout == "inline":
        for i in range(10):
            yield m.u64(base + hero_array_rva + i * stride)
    else:  # deref
        arr_ptr = m.u64(base + hero_array_rva)
        if not arr_ptr: return
        for i in range(10):
            yield m.u64(arr_ptr + i * stride)

def _norm_champ(s):
    """Normalize champion names for matching. API returns display form ('Bel'Veth',
    'Dr. Mundo', 'Cho'Gath') while the in-memory struct uses the internal form
    ('Belveth', 'DrMundo', 'Chogath'). Strip apostrophes/dots/spaces, lowercase."""
    return (s or "").replace("'", "").replace(".", "").replace(" ", "").lower()

def _get_hero_and_api(m, base, hero_array_rva, name_off, layout="deref", stride=8, target_name="Garen"):
    """Helper: get hero ptr + API data for target champion.

    Returns (None, None) if the champion is not in the hero_array — caller must skip
    the snapshot rather than read garbage from a different hero.
    """
    target_norm = _norm_champ(target_name)
    hp = None
    ptrs = list(_iter_hero_ptrs(m, base, hero_array_rva, layout, stride))
    for h in ptrs:
        if h and _norm_champ(m.string(h + name_off, 40)) == target_norm:
            hp = h; break
    if not hp:
        live_names = [m.string(h + name_off, 40) for h in ptrs if h]
        print(f"  ⚠ {target_name!r} not in hero_array (live: {[n for n in live_names if n]}); skipping snapshot")
        return None, None
    pl = rget("/liveclientdata/playerlist")
    api = next((p for p in pl if _norm_champ(p.get("championName")) == target_norm), None)
    if api is None:
        live_api = [p.get("championName") for p in pl]
        print(f"  ⚠ {target_name!r} not in /liveclientdata/playerlist (live: {live_api}); skipping snapshot")
        return None, None
    return hp, api

def find_hero_fields(m, base, hero_array_rva, anchors, layout="deref", stride=8, target_name="Garen"):
    """Find hero struct fields using 2-time-point verification."""
    print("\n[Phase 2] Finding hero struct fields (multi-time verification)...")

    # Find champion_name offset (±0x80 from anchor, step 4 — patches have shifted by +0x38)
    name_off = None
    best_found = 0
    for drift in range(-0x80, 0x84, 4):
        test_off = anchors["champion_name"] + drift
        found = 0
        for hp in _iter_hero_ptrs(m, base, hero_array_rva, layout, stride):
            if hp:
                name = m.string(hp + test_off, 40)
                if name and name in CHAMP_NAMES: found += 1
        if found > best_found:
            best_found = found
            name_off = test_off
            if found >= 8: break  # very confident
    if name_off and best_found >= 5:
        print(f"  ✓ champion_name at hero+0x{name_off:X} ({best_found}/10 match)")
    else:
        print(f"  ✗ champion_name not found (best={best_found}/10)"); return {}

    results = {"champion_name": name_off}

    # Capture at TWO time points — read ALL candidate values IMMEDIATELY
    # (hero pointers go stale after seek, so must read at capture time)
    SEARCH_RANGE = 192  # ±192 bytes from anchor (widened — patches have shifted by up to +0x80)
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
        hp, api = _get_hero_and_api(m, base, hero_array_rva, name_off, layout=layout, stride=stride, target_name=target_name)
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
        # Also read active spell name chain. The chain dereferences through:
        #   active_spell_ptr (hp + active_spell_off)
        #     +0x008 → spell_info struct (spell_info offset within cast struct)
        #       +0x028 → spell_name_ptr (spell_name_ptr offset within spell_info)
        # If the chain produces a valid champion-spell name, both spell_info=0x008
        # and spell_name_ptr=0x028 are transitively verified for this patch.
        # Also probe sp + 0..0x300 for a Vec3 (cast_target) matching hero position
        # (self-cast) or somewhere on the map.
        SPELL_INFO_HOPS = [0x008, 0x010, 0x018]
        for off in candidate_offsets.get("active_spell", []):
            ptr = m.u64(hp + off)
            if ptr and LO < ptr < HI:
                for si_hop in SPELL_INFO_HOPS:
                    si = m.u64(ptr + si_hop)
                    if si and LO < si < HI:
                        np_ = m.u64(si + 0x28)
                        if np_ and LO < np_ < HI:
                            sn = m.string(np_)
                            if sn and len(sn) > 3:
                                snap["vals"][("active_spell_name", off)] = sn
                                snap["vals"][("spell_info_hop", off)] = si_hop
                                # Probe for cast_target Vec3 within the cast struct
                                # (sp + 0..0x300, 4-byte aligned). Map bounds = 200..15800.
                                pos = snap["vals"].get(("position", anchors.get("position", 0x200)))
                                hx = pos[0] if pos else None
                                hz = pos[2] if pos else None
                                for ct_off in range(0x40, 0x300, 4):
                                    v = m.vec3(ptr + ct_off)
                                    if not v: continue
                                    x, _, z = v
                                    if not (-2 < x < 16000 and -2 < z < 16000): continue
                                    # Prefer match to hero pos (self-cast) but accept any plausible
                                    if hx is not None and hz is not None and abs(x - hx) < 10 and abs(z - hz) < 10:
                                        snap["vals"][("cast_target_off", off)] = ct_off
                                        snap["vals"][("cast_target_match", off)] = "self"
                                        break
                                    # Fallback: any plausible non-zero Vec3 inside map
                                    if 200 < x < 15800 and 200 < z < 15800:
                                        if ("cast_target_off", off) not in snap["vals"]:
                                            snap["vals"][("cast_target_off", off)] = ct_off
                                            snap["vals"][("cast_target_match", off)] = "map"
                                break  # found chain via this si_hop, stop trying more hops
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

    # Gold earned: increases monotonically, plausible total-gold range (0..200k).
    # Tighter bounds avoid huge-value garbage candidates (e.g. raw bit pattern of a pointer
    # interpreted as f32 yields 1e30+).
    for off in candidate_offsets["gold_earned"]:
        v1 = s1["vals"].get(("gold_earned", off))
        v2 = s2["vals"].get(("gold_earned", off))
        if (v1 is not None and v2 is not None
                and 100 < v1 < 200_000 and 100 < v2 < 200_000
                and v2 > v1 and (v2 - v1) < 100_000):
            results["gold_earned"] = off
            print(f"  ✓ gold_earned at hero+0x{off:X}: {v1:.0f}→{v2:.0f}")
            break

    # Gold current: plausible spending-gold range, must be positive at least once.
    for off in candidate_offsets["gold_current"]:
        v1 = s1["vals"].get(("gold_current", off))
        v2 = s2["vals"].get(("gold_current", off))
        if (v1 is not None and v2 is not None
                and 0 <= v1 < 30_000 and 0 <= v2 < 30_000
                and (v1 > 10 or v2 > 10)):  # at least one snapshot has spendable gold
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
                # Transitively verified: spell_info hop and spell_name_ptr.
                si_hop = snap["vals"].get(("spell_info_hop", off))
                if si_hop is not None:
                    results["spell_info"] = si_hop
                    print(f"  ✓ spell_info at active_spell+0x{si_hop:X} (transitive via name chain)")
                results["spell_name_ptr"] = 0x28
                # cast_target probe (only valid when actually casting)
                ct = snap["vals"].get(("cast_target_off", off))
                ctm = snap["vals"].get(("cast_target_match", off))
                if ct is not None:
                    results["cast_target"] = ct
                    print(f"  ✓ cast_target at active_spell+0x{ct:X} (match={ctm})")
                else:
                    print(f"  ⚠ cast_target NOT probed (no Vec3 in cast struct matched hero pos / map bounds)")
                found = True; break
        if found: break
    if "active_spell" not in results:
        print(f"  ~ active_spell, spell_info, cast_target deferred — Phase 3.6 (live-cast diff probe) is sole source of truth")

    return results

# ═══════════════════════════════════════════════════════════════
# Phase 3: Find SpellBook
# ═══════════════════════════════════════════════════════════════
def find_spellbook(m, hero_ptr, anchors, offsets=None, champion="Garen"):
    """Find inline spellbook by looking for spell name deref chains.

    Champion-adaptive: spell names are <Champion><Q|W|E|R> (e.g. "BelvethQ").
    Also searches the slot_spell_info hop (slot+?) and spell_name_ptr hop (spell_info+?)
    rather than only trying a small fixed list, so patches that shift these still resolve.
    """
    print(f"\n[Phase 3] Finding SpellBook for {champion}...")
    expected = {f"{champion}{k}" for k in "QWER"}
    # Widened search ranges 2026-05-05: spellbook shifted by 0x50 across patches,
    # slot_array shifted by 0x58 — old ±0x100/±0x60 windows missed champions whose
    # base SpellBook RVA differs from the anchor.
    info_offsets = [0x130, 0x128, 0x120, 0x138, 0x140, 0x148, 0x150, 0x158, 0x118, 0x110, 0x108, 0x160]
    name_offsets = [0x28, 0x20, 0x30, 0x38, 0x18, 0x10, 0x08, 0x40, 0x48, 0x50]
    for sb_drift in range(-0x200, 0x200, 8):
        sb_off = anchors["spellbook"] + sb_drift
        for sa_drift in range(-0x100, 0x100, 8):
            sa_off = anchors["slot_array"] + sa_drift
            found_names = set()
            best_info_off = None
            best_name_off = None
            for i in range(4):
                slot_ptr = m.u64(hero_ptr + sb_off + sa_off + i*8)
                if not slot_ptr or not (LO < slot_ptr < HI): continue
                for info_off in info_offsets:
                    si = m.u64(slot_ptr + info_off)
                    if not si or not (LO < si < HI): continue
                    for name_off in name_offsets:
                        np = m.u64(si + name_off)
                        if not np or not (LO < np < HI): continue
                        sn = m.string(np)
                        if sn and sn in expected:
                            found_names.add(sn)
                            best_info_off = info_off
                            best_name_off = name_off
                            break
                    if best_info_off == info_off: break
            spells_found = len(found_names)
            if spells_found >= 4:
                # Trust the (sb_off, sa_off) that produced 4 valid spell-name derefs.
                # The old "fixed via active_spell" rewrite path assumed sb_off + 0x38 ==
                # active_spell — that convention BROKE on 16.9.772. Phase 3.6 verifies
                # active_spell separately via live-cast diff probe.
                print(f"  ✓ spellbook at hero+0x{sb_off:X}, slot_array at +0x{sa_off:X} ({spells_found}/4 spells)")
                print(f"    slot_spell_info at +0x{best_info_off:X}, spell_name_ptr at +0x{best_name_off:X}")
                return {
                    "spellbook": sb_off, "slot_array": sa_off,
                    "slot_spell_info": best_info_off,
                    "spell_name_ptr": best_name_off,
                }
    print(f"  ✗ SpellBook not found (looking for {sorted(expected)})")
    return {}

# ═══════════════════════════════════════════════════════════════
# Phase 3.6: Live-cast probe to derive spell_info + cast_target
# ═══════════════════════════════════════════════════════════════
def probe_live_cast(m, base, hero_array_rva, layout, stride, name_off,
                    position_off, champion="Garen",
                    seek_targets=(300.0, 500.0, 700.0, 900.0, 1100.0, 1300.0,
                                  1500.0, 1700.0, 1900.0, 2100.0),
                    poll_seconds=15.0,
                    scan_lo=0x2C00, scan_hi=0x3400):
    """Snapshot-diff probe to derive active_spell + spell_info + cast_target.

    Strategy:
      1. Pause at seek_gt; snapshot ALL u64 fields at hero+0x3000..0x3300 (paused, baseline).
      2. Resume; poll those same offsets at 50Hz.
      3. Any offset that goes non-null AND derefs (via hop 0x008/0x010/0x018) to a
         heap struct whose +0x028 is a ptr to "<champion>X" (Q/W/E/R) string is our
         live active_spell offset. Capture spell_info hop + scan for cast_target Vec3.

    Re-resolves hero_ptr after each seek (engine reallocs heroes).
    Returns {active_spell, spell_info, cast_target, spell_name_ptr} on success, {} on miss.
    """
    print("\n[Phase 3.6] Live-cast diff probe (active_spell + spell_info + cast_target)...")
    SPELL_INFO_HOPS = [0x008, 0x010, 0x018, 0x020, 0x028]
    SCAN_STEP = 8
    SCAN_OFFSETS = list(range(scan_lo, scan_hi, SCAN_STEP))

    target_norm = _norm_champ(champion)

    def _find_target_hp():
        for hp in _iter_hero_ptrs(m, base, hero_array_rva, layout, stride):
            if hp and LO < hp < HI and _norm_champ(m.string(hp + name_off, 40)) == target_norm:
                return hp
        return None

    expected_spells = {f"{champion}{k}" for k in "QWER"}

    for seek_gt in seek_targets:
        rpost("/replay/playback", {"time": float(seek_gt), "speed": 0.0, "paused": True})
        time.sleep(1.5)
        hero_ptr = _find_target_hp()
        if not hero_ptr:
            print(f"  seek gt={seek_gt:.0f}: ⚠ {champion} not found in hero_array post-seek")
            continue
        # Snapshot baseline (paused) — record which offsets were non-null already
        baseline_nonnull = set()
        for off in SCAN_OFFSETS:
            v = m.u64(hero_ptr + off)
            if v: baseline_nonnull.add(off)
        rpost("/replay/playback", {"speed": 1.0, "paused": False})
        t_start = time.time()
        sn_caught = None
        cast_struct = None
        si_hop = None
        ac_off = None
        # Track which offsets have toggled to a non-null value during play
        toggled_offsets = set()
        while time.time() - t_start < poll_seconds:
            for off in SCAN_OFFSETS:
                v = m.u64(hero_ptr + off)
                if not v or not (LO < v < HI): continue
                if off in baseline_nonnull and off not in toggled_offsets:
                    # Was non-null at baseline AND still non-null — could be permanent struct ptr.
                    # Still try the chain in case it derefs to a spell.
                    pass
                # Attempt chain
                for hop in SPELL_INFO_HOPS:
                    si = m.u64(v + hop)
                    if not si or not (LO < si < HI): continue
                    np_ = m.u64(si + 0x28)
                    if not np_ or not (LO < np_ < HI): continue
                    sn = m.string(np_, 32)
                    if sn and sn in expected_spells:
                        sn_caught = sn
                        cast_struct = v
                        si_hop = hop
                        ac_off = off
                        break
                if sn_caught: break
                # If this offset wasn't in baseline_nonnull, it just toggled
                if off not in baseline_nonnull:
                    toggled_offsets.add(off)
            if sn_caught: break
            time.sleep(0.02)
        rpost("/replay/playback", {"speed": 0.0, "paused": True})
        if not sn_caught:
            print(f"  seek gt={seek_gt:.0f}: no cast in {poll_seconds:.0f}s "
                  f"({len(toggled_offsets)} offsets toggled non-null but none derefed to {sorted(expected_spells)[0]}…)")
            continue

        print(f"  seek gt={seek_gt:.0f}: caught cast {sn_caught!r} via hero+0x{ac_off:X}")
        print(f"  ✓ active_spell at hero+0x{ac_off:X} (diff-probe verified)")
        print(f"  ✓ spell_info at active_spell+0x{si_hop:X} (chain verified)")
        result = {"active_spell": ac_off, "spell_info": si_hop, "spell_name_ptr": 0x28}

        # Probe cast_target Vec3 within the cast struct
        hpos = m.vec3(hero_ptr + position_off)
        hx, hz = (hpos[0], hpos[2]) if hpos else (None, None)
        ct_self = None
        ct_map = None
        for ct_off in range(0x40, 0x300, 4):
            v = m.vec3(cast_struct + ct_off)
            if not v: continue
            x, _, z = v
            if not (-2 < x < 16000 and -2 < z < 16000): continue
            if hx is not None and abs(x - hx) < 10 and abs(z - hz) < 10:
                ct_self = ct_off
                break
            if 200 < x < 15800 and 200 < z < 15800 and ct_map is None:
                ct_map = ct_off
        if ct_self is not None:
            result["cast_target"] = ct_self
            print(f"  ✓ cast_target at active_spell+0x{ct_self:X} (self-cast match)")
        elif ct_map is not None:
            result["cast_target"] = ct_map
            print(f"  ✓ cast_target at active_spell+0x{ct_map:X} (in-map Vec3)")
        else:
            print(f"  ⚠ cast_target NOT found in cast struct (no plausible Vec3)")
        return result

    print(f"  ⚠ no cast captured across {len(seek_targets)} seeks; spell_info/cast_target remain MISSING")
    return {}

# ═══════════════════════════════════════════════════════════════
# Phase 3.5: Find click-dest VTABLE_RVA via runtime triple-mirror scan
# ═══════════════════════════════════════════════════════════════
def find_click_vtable(m, base, mod_size, hero_ptrs):
    """Find click-dest class vptr (16.9.772+) via heap-scan for triple-Vec3-mirror.

    Click-dest object layout (across all known patches):
      parent+0x00       vptr (the class signature we're after)
      parent+0x14       Vec3 click destination (xfile)
      parent+0x14+0x308 Vec3 mirror B
      parent+0x14+0x374 Vec3 mirror C

    Plus an owner pointer at fixed offset within the parent (parent+0x68 on 16.8.766).
    Owner offset is also derived dynamically by checking which fixed offset in
    parent reads back one of the known hero pointers.
    """
    print("\n[Phase 3.5] Finding click-dest VTABLE_RVA + owner offset...")
    regions = [(a, sz) for (a, sz) in enum_regions(m.h)
               if sz <= 64*1024*1024]  # skip giant regions

    hero_ptr_set = set(hero_ptrs.values()) if isinstance(hero_ptrs, dict) else set(hero_ptrs)

    # Pass 1: find candidate parents via triple-mirror Vec3 pattern
    parents = []  # list of (parent_addr, vec3)
    for ra, rsz in regions:
        data = m.read(ra, rsz)
        if not data or len(data) < 0x400: continue
        # Iterate Vec3 candidates (4-byte aligned)
        for i in range(0, len(data) - 0x380, 4):
            x = struct.unpack_from('<f', data, i)[0]
            if not (200 < x < 15800): continue
            z = struct.unpack_from('<f', data, i+8)[0]
            if not (200 < z < 15800): continue
            y = struct.unpack_from('<f', data, i+4)[0]
            if not (-10 < y < 250): continue
            # Triple-mirror check
            xb, yb, zb = struct.unpack_from('<fff', data, i + 0x308)
            if xb != x or yb != y or zb != z: continue
            xc, yc, zc = struct.unpack_from('<fff', data, i + 0x374)
            if xc != x or zc != z: continue
            # parent = vec_addr - 0x14 (vec lives at parent+0x14)
            if i < 0x14: continue
            parent_addr = ra + i - 0x14
            parents.append((parent_addr, (x, y, z)))

    if not parents:
        print("  ✗ no triple-mirror candidates found")
        return {}
    print(f"  found {len(parents)} triple-mirror parent candidates")

    # Pass 2: top vptr value among parents → VTABLE_RVA
    from collections import Counter
    vptr_counts = Counter()
    for parent_addr, _ in parents:
        vp = m.u64(parent_addr)
        if vp and base <= vp < base + mod_size:
            vptr_counts[vp - base] += 1
    if not vptr_counts:
        print("  ✗ no module-resident vptrs")
        return {}
    vtable_rva, n_hits = vptr_counts.most_common(1)[0]
    print(f"  ✓ click_vtable_rva = 0x{vtable_rva:X}  (n={n_hits})")

    # Pass 3: derive owner offset by checking common candidates against hero_ptrs.
    # Filter to parents whose vptr matches the chosen VTABLE.
    matching_parents = [p for (p, _) in parents
                        if m.u64(p) == base + vtable_rva]
    print(f"  {len(matching_parents)} parents have VTABLE vptr; testing owner offsets...")
    owner_offset = None
    if hero_ptr_set and matching_parents:
        # Try a range of fixed offsets in parent; pick the offset that points to
        # a known hero ptr in the most parents.
        owner_match_counts = Counter()
        for off in range(0x10, 0x200, 8):
            n = 0
            for p in matching_parents:
                if m.u64(p + off) in hero_ptr_set:
                    n += 1
            if n > 0:
                owner_match_counts[off] = n
        if owner_match_counts:
            owner_offset, n = owner_match_counts.most_common(1)[0]
            print(f"  ✓ click_owner_offset = 0x{owner_offset:X}  (matches {n}/{len(matching_parents)} parents)")
        else:
            print(f"  ✗ no owner offset found (parent+0x?? → hero_ptr)")
    return {
        "click_vtable_rva": vtable_rva,
        **({"click_owner_offset": owner_offset} if owner_offset is not None else {}),
    }


# ═══════════════════════════════════════════════════════════════
# Phase 4: Find attack_target_pos on the hero struct (Vec3 of AA target unit)
# ═══════════════════════════════════════════════════════════════
def find_attack_target_pos(m, base, hero_array_rva, layout, stride, name_off,
                           position_off, active_spell_off, spell_info_off,
                           spell_name_ptr_off, champion="Garen",
                           seek_targets=(60.0, 120.0, 200.0, 300.0, 420.0),
                           target_aa_ticks=60, poll_seconds_per_seek=50.0,
                           scan_lo=0x3F00, scan_hi=0x4900):
    """Find hero+attack_target_pos: the Vec3 of the unit being auto-attacked.

    On 16.9.x the AA cast struct's `cast_target` field is the cast-origin (= hero
    pos), not the target unit. The real target's world pos lives on the HERO
    struct. We find its offset by:

      1) Seeking through laning windows; for each tick, check whether the target
         champion's `active_spell` resolves to "<Champion>BasicAttack(2)".
      2) On AA ticks, snapshot u32s in hero[scan_lo..scan_hi].
      3) Score each offset: count of ticks where (offset, offset+4, offset+8) is
         a valid Vec3 in map bounds AND distinct from hero pos (>=50u).
      4) Among offsets passing 70% threshold, prefer the one with MAX median
         distance (the trajectory buffer also clears the threshold but stays
         within ~60u of hero — AA target lives further out).

    No cam-lock player-key required: the spell-name-startswith-champion filter
    rejects spell-pointer mirroring artifacts (Garen's struct momentarily holding
    XinZhao's cast struct etc.)."""
    print("\n[Phase 4] Finding attack_target_pos (cam-lock-free AA-filter probe)...")
    target_norm = _norm_champ(champion)

    def _find_target_hp():
        for hp in _iter_hero_ptrs(m, base, hero_array_rva, layout, stride):
            if hp and LO < hp < HI and _norm_champ(m.string(hp + name_off, 40)) == target_norm:
                return hp
        return None

    aa_ticks = []  # list of {pos, u32s}
    for seek_gt in seek_targets:
        if len(aa_ticks) >= target_aa_ticks: break
        rpost("/replay/playback", {"time": float(seek_gt), "paused": True})
        time.sleep(0.6)
        rpost("/replay/render", {"selectionName": champion})
        time.sleep(0.3)
        rpost("/replay/playback", {"speed": 2.0, "paused": False})
        time.sleep(0.8)

        deadline = time.time() + poll_seconds_per_seek
        captured_at_seek = 0
        while len(aa_ticks) < target_aa_ticks and time.time() < deadline:
            time.sleep(0.04)
            hp = _find_target_hp()
            if not hp: continue
            sp = m.u64(hp + active_spell_off)
            if not sp or not (LO < sp < HI): continue
            si = m.u64(sp + spell_info_off)
            if not si or not (LO < si < HI): continue
            np_ = m.u64(si + spell_name_ptr_off)
            if not np_ or not (LO < np_ < HI): continue
            sn = m.string(np_, 48)
            if not sn or not sn.startswith(champion) or "asicAttack" not in sn:
                continue
            pos = m.vec3(hp + position_off)
            if not pos: continue
            d = m.read(hp + scan_lo, scan_hi - scan_lo)
            if not d: continue
            u32s = struct.unpack(f"<{len(d)//4}I", d)
            aa_ticks.append({"pos": pos, "u32s": u32s})
            captured_at_seek += 1
        print(f"  seek gt={seek_gt:.0f}: +{captured_at_seek} {champion}-AA ticks "
              f"(total {len(aa_ticks)}/{target_aa_ticks})")

    rpost("/replay/playback", {"speed": 0.0, "paused": True})

    if not aa_ticks:
        print(f"  ⚠ no {champion} AAs captured across {len(seek_targets)} seek windows")
        return {}

    # Score every aligned offset in the scan range as a candidate Vec3 start.
    n_off = (scan_hi - scan_lo) // 4
    scored = []
    for off_idx in range(n_off - 2):
        off = scan_lo + off_idx * 4
        valid = 0
        dists = []
        for tick in aa_ticks:
            fx = struct.unpack("<f", struct.pack("<I", tick["u32s"][off_idx]))[0]
            fy = struct.unpack("<f", struct.pack("<I", tick["u32s"][off_idx + 1]))[0]
            fz = struct.unpack("<f", struct.pack("<I", tick["u32s"][off_idx + 2]))[0]
            if not (-2000 < fx < 16500 and -2000 < fz < 16500 and -300 < fy < 600):
                continue
            dx = fx - tick["pos"][0]; dz = fz - tick["pos"][2]
            d2 = dx * dx + dz * dz
            if d2 < 50 * 50: continue   # too close to hero pos (hero-pos mirror)
            valid += 1
            dists.append(d2 ** 0.5)
        if valid >= len(aa_ticks) * 0.7 and dists:
            med = sorted(dists)[len(dists) // 2]
            scored.append((off, valid, med))

    if not scored:
        print(f"  ⚠ no offset cleared 70% valid-target Vec3 threshold")
        return {}
    # Prefer high valid count, then HIGH median distance (trajectory buffers also
    # clear 70% but cluster near hero — AA target is further).
    scored.sort(key=lambda s: (-s[1], -s[2]))
    off, valid, med = scored[0]
    print(f"  ✓ attack_target_pos at hero+0x{off:X} "
          f"(valid {valid}/{len(aa_ticks)}, med_dist={med:.0f}u)")
    return {"attack_target_pos": off}

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _load_anchors_file(path):
    """Load anchors from either the legacy flat format or the tagged format
    (where each offset is {"value": <int>, ...}). Skips _-prefixed metadata."""
    with open(path) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if k.startswith("_"): continue
        if isinstance(v, int):
            out[k] = v
        elif isinstance(v, dict) and isinstance(v.get("value"), int):
            out[k] = v["value"]
    return out

def main():
    parser = argparse.ArgumentParser(description="Automated offset scanner")
    parser.add_argument("--anchors", help="Previous offsets JSON for drift-based search")
    parser.add_argument("--output", "-o", default="offsets.json")
    parser.add_argument("--patch", default=None, help="Patch identifier tag (e.g., 16.9.772)")
    parser.add_argument("--champion", default="Garen", help="Champion name to lock onto for spellbook + click-dest discovery (e.g., Belveth)")
    args = parser.parse_args()

    anchors = DEFAULT_ANCHORS.copy()
    if args.anchors:
        anchors.update(_load_anchors_file(args.anchors))

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

    offsets = {"_scanner_version": 2, "_patch_mod_size": mod_size,
               "_scanned_at": _now_iso()}
    if args.patch: offsets["_patch"] = args.patch
    versions = {}  # field -> ISO8601 timestamp of last verification

    def _tag(*fields):
        ts = _now_iso()
        for f in fields:
            if f in offsets: versions[f] = ts

    # Phase 1: RVAs
    hero_rva, layout, stride = find_hero_array(
        m, base, mod_size, anchors["hero_array"],
        name_anchor=anchors.get("champion_name", 0x4328))
    if hero_rva is not None:
        offsets["hero_array"] = hero_rva
        offsets["hero_array_layout"] = layout
        offsets["hero_array_stride"] = stride
        _tag("hero_array", "hero_array_layout", "hero_array_stride")

    gt_rva = find_game_time(m, base, mod_size, anchors["game_time"], gt)
    if gt_rva:
        offsets["game_time"] = gt_rva
        _tag("game_time")

    if hero_rva is None:
        print("\nFATAL: Cannot find hero array. Aborting.")
        return

    # Phase 2: Hero fields (uses 2 time points internally)
    if gt_rva: anchors["game_time"] = gt_rva
    hero_fields = find_hero_fields(m, base, hero_rva, anchors,
                                    layout=layout, stride=stride,
                                    target_name=args.champion)
    offsets.update(hero_fields)
    _tag(*hero_fields.keys())

    # Phase 3: SpellBook — seek to mid-game where spells are leveled
    rpost("/replay/playback", {"time": 600.0, "speed": 0.0, "paused": True})
    time.sleep(1.5)
    rpost("/replay/playback", {"speed": 1.0, "paused": False})
    time.sleep(2)
    rpost("/replay/playback", {"speed": 0.0, "paused": True})
    time.sleep(0.3)

    # Layout-agnostic hero iteration; champion-match is normalization-tolerant
    # ("Bel'Veth" display form ↔ "Belveth" struct form).
    name_off = offsets.get("champion_name", 0x4328)
    target_hp = None
    hero_ptrs = {}  # slot index → hero_ptr (for click_vtable phase)
    target_norm = _norm_champ(args.champion)
    for i, hp in enumerate(_iter_hero_ptrs(m, base, hero_rva, layout, stride)):
        if hp and (LO < hp < HI):
            hero_ptrs[i] = hp
            nm = m.string(hp + name_off)
            if _norm_champ(nm) == target_norm:
                target_hp = hp
    if not target_hp:
        live = [m.string(hp + name_off) for hp in hero_ptrs.values()]
        print(f"\n⚠ champion {args.champion!r} not in hero_ptrs (live: {[n for n in live if n]}). "
              f"Phase 3/3.5/3.6 will be skipped — fields will remain MISSING.")

    if target_hp:
        sb = find_spellbook(m, target_hp, anchors, offsets, champion=args.champion)
        offsets.update(sb)
        _tag(*sb.keys())
        # Note: active_spell used to be set here as spellbook+0x38, but that convention
        # broke on 16.9.772. active_spell is now solely derived by Phase 3.6 (live-cast
        # diff probe). If Phase 3.6 fails to capture a cast, active_spell remains MISSING.

    # Phase 3.6: live-cast diff probe — sole source of truth for
    # active_spell + spell_info + cast_target on 16.9.772+.
    # Snapshots all u64 fields in hero+scan_lo..scan_hi while paused, then plays
    # at 1x and watches which offset toggles to a deref-able cast struct.
    if "position" in offsets and "champion_name" in offsets:
        cast_offs = probe_live_cast(
            m, base,
            hero_array_rva=hero_rva, layout=layout, stride=stride,
            name_off=offsets["champion_name"],
            position_off=offsets["position"],
            champion=args.champion,
        )
        for k, v in cast_offs.items():
            offsets[k] = v
            _tag(k)

    # Phase 3.5: click-dest VTABLE_RVA + owner offset (heap-scan)
    if hero_ptrs:
        cvt = find_click_vtable(m, base, mod_size, hero_ptrs)
        offsets.update(cvt)
        _tag(*cvt.keys())

    # Phase 4: attack_target_pos on hero struct (Vec3 of AA target unit).
    # Requires Phase 3.6 outputs (active_spell + spell_info + spell_name_ptr).
    needed_for_aa = ("position", "champion_name", "active_spell", "spell_info", "spell_name_ptr")
    if all(k in offsets for k in needed_for_aa):
        at = find_attack_target_pos(
            m, base,
            hero_array_rva=hero_rva, layout=layout, stride=stride,
            name_off=offsets["champion_name"],
            position_off=offsets["position"],
            active_spell_off=offsets["active_spell"],
            spell_info_off=offsets["spell_info"],
            spell_name_ptr_off=offsets["spell_name_ptr"],
            champion=args.champion,
        )
        offsets.update(at)
        _tag(*at.keys())
    else:
        missing_for_aa = [k for k in needed_for_aa if k not in offsets]
        print(f"\n[Phase 4] SKIP — missing prerequisites: {missing_for_aa}")

    rpost("/replay/playback", {"speed": 0.0, "paused": True})

    # Loud-fallback policy: list every field the scan failed to derive. Pipeline.py
    # defaults will fill these in but the user must be aware of every gap.
    # NOTE: click_dest_rva/click_dest_off are LEGACY AiWaypoints fields not used by
    # pipeline_merged.py (which uses click_vtable_rva + click_owner_offset directly).
    EXPECTED = {
        "hero_array", "hero_array_layout", "hero_array_stride",
        "game_time",
        "champion_name", "position", "level",
        "hp_current", "hp_max",
        "gold_current", "gold_earned",
        "vision_score", "active_spell",
        "spellbook", "slot_array", "slot_spell_info", "spell_name_ptr",
        "spell_info", "cast_target",
        "click_vtable_rva", "click_owner_offset",
        "attack_target_pos",
    }
    missing = sorted(f for f in EXPECTED if f not in offsets)
    if missing:
        offsets["_missing"] = missing

    offsets["_offset_versions"] = versions

    # Save
    out_path = args.output
    with open(out_path, "w") as f:
        json.dump(offsets, f, indent=2)
    print(f"\n=== Done! Saved to {out_path} ===")
    for k, v in offsets.items():
        if k == "_offset_versions": continue
        if k.startswith("_"):
            print(f"  {k}: {v}")
            continue
        tag = versions.get(k, "?")
        if isinstance(v, int):
            print(f"  {k}: 0x{v:X}  [{tag}]")
        else:
            print(f"  {k}: {v}  [{tag}]")
    if missing:
        print(f"\n⚠ MISSING ({len(missing)} fields not derived; pipeline.py will fall back to defaults):")
        for f in missing:
            print(f"    ⚠ {f}")

if __name__ == "__main__":
    main()
