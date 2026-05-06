"""
Find spellbook (Q/W/E/R/D/F) and inventory offsets for LoL patch 26.7.

Approach:
  The known ActiveSpell chain is 3 levels deep:
    hero_ptr + 0x3120 → ActiveSpell*
    ActiveSpell + 0x008 → SpellInfo*
    SpellInfo + 0x028 → name_ptr
    *name_ptr → "GarenQ", "SummonerFlash", etc.

  SpellBook slots (Q, W, E, R, D, F) are stored somewhere else in the
  hero struct using the SAME chain pattern. Find them by scanning every
  qword in the hero blob, following the chain, and keeping offsets whose
  chain resolves to a spell name string.

  For items: similar deep deref — look for chain that resolves to u32 itemID
  matching known inventory from allgamedata.
"""
import base64, ctypes, ctypes.wintypes as wt, json, os, ssl
import struct, subprocess, sys, time, urllib.request
from pynput.keyboard import Controller as KbController

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
_kb = KbController()

EXPECTED_MOD_SIZE = 0x207C000
OFFSETS = {
    "hero_array":    0x1DBEEE8,
    "game_time":     0x1DCD1E0,
    "position":      0x25C,
    "champion_name": 0x4328,
    "active_spell":  0x3120,   # known: ActiveSpell*
    "spell_info":    0x008,    # known: ActiveSpell + 0x008 → SpellInfo*
    "spell_name":    0x028,    # known: SpellInfo + 0x028 → name_ptr
}
LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"
REPLAY_ID = "5528069928"

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE
_k = ctypes.windll.kernel32


def replay_get(ep):
    req = urllib.request.Request(f"https://127.0.0.1:2999{ep}")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r:
        return json.loads(r.read())


def replay_post(ep, data):
    req = urllib.request.Request(
        f"https://127.0.0.1:2999{ep}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r:
        return json.loads(r.read() or b"null")


def lcu_auth():
    parts = open(LOCKFILE).read().strip().split(":")
    return parts[2], f"Basic {base64.b64encode(f'riot:{parts[3]}'.encode()).decode()}"


def lcu_post(ep, body=None):
    port, auth = lcu_auth()
    req = urllib.request.Request(
        f"https://127.0.0.1:{port}{ep}", method="POST",
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": auth, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, context=_ctx, timeout=15) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


# ── Mem reader ────────────────────────────────────────────────
class Mem:
    def __init__(self, pid):
        self.h = _k.OpenProcess(0x0410, False, pid)
        if not self.h: raise OSError(f"OpenProcess({pid}) failed")
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz)
        n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u64(self, a):
        d = self.read(a, 8)
        return struct.unpack('<Q', d)[0] if d else None
    def u32(self, a):
        d = self.read(a, 4)
        return struct.unpack('<I', d)[0] if d else None
    def string(self, a, n=64):
        d = self.read(a, n)
        if not d: return None
        t = d.split(b'\x00')[0]
        try: return t.decode('ascii')
        except Exception: return None
    def close(self):
        _k.CloseHandle(self.h)


def find_league_pid():
    r = subprocess.run(
        ['tasklist', '/FI', 'IMAGENAME eq League of Legends.exe', '/FO', 'CSV', '/NH'],
        capture_output=True, text=True,
    )
    for line in r.stdout.strip().split('\n'):
        if 'league' in line.lower():
            return int(line.strip('"').split('","')[1])
    return None


def find_module_base(pid):
    class ME(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong), ("th32ModuleID", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong), ("GlblcntUsage", ctypes.c_ulong),
            ("ProccntUsage", ctypes.c_ulong),
            ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
            ("modBaseSize", ctypes.c_ulong), ("hModule", ctypes.c_void_p),
            ("szModule", ctypes.c_char * 256), ("szExePath", ctypes.c_char * 260),
        ]
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
    arr_ptr = m.u64(base + OFFSETS["hero_array"])
    if not arr_ptr: return {}
    heroes = {}
    for i in range(10):
        hp = m.u64(arr_ptr + i * 8)
        if not hp or hp < 0x10000: continue
        name = m.string(hp + OFFSETS["champion_name"])
        if name and len(name) >= 2 and name[0].isalpha():
            heroes[name] = {"ptr": hp, "slot": i, "team": "blue" if i < 5 else "red"}
    return heroes


# ── Ground truth ──────────────────────────────────────────────
def get_ground_truth():
    data = replay_get("/liveclientdata/allgamedata")
    gt = {}
    for p in data.get("allPlayers", []):
        champ = p.get("championName") or p.get("rawChampionName", "")
        items = p.get("items", [])
        summs = p.get("summonerSpells", {})
        gt[champ] = {
            "items": [int(it.get("itemID", 0)) for it in items],
            "summ1": (summs.get("summonerSpellOne") or {}).get("displayName", ""),
            "summ2": (summs.get("summonerSpellTwo") or {}).get("displayName", ""),
        }
    return gt


# ── Playback control ─────────────────────────────────────────
def play_until(target_gt, max_wall=300):
    try:
        replay_post("/replay/playback", {"speed": 2.0, "paused": False})
    except Exception as e:
        print(f"[play] err: {e}")
        return 0.0
    t0 = time.time()
    last_gt = 0.0
    while time.time() - t0 < max_wall:
        try:
            d = replay_get("/liveclientdata/gamestats")
            last_gt = d.get("gameTime", 0.0)
            if last_gt >= target_gt:
                break
        except Exception:
            pass
        time.sleep(1.0)
    try:
        replay_post("/replay/playback", {"speed": 0.0, "paused": True})
    except Exception:
        pass
    time.sleep(1.0)
    return last_gt


def wait_for_loaded(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            d = replay_get("/liveclientdata/gamestats")
            if d.get("gameTime", 0) > 5:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def launch_replay():
    try:
        lcu_post(f"/lol-replays/v1/rofls/{REPLAY_ID}/watch", {"componentType": "string"})
    except Exception as e:
        print(f"[launch] err: {e}")


# ── CORE: deep deref scan ─────────────────────────────────────
def is_ptr(v):
    return v is not None and 0x10000 < v < 0x7FFFFFFFFFFF


def deref_wrapped_spell(m, ptr1):
    """3-level: ptr1 → +0x08 (SpellInfo*) → +0x28 (name_ptr) → string.
    Used for ActiveSpellCast-style wrappers."""
    if not is_ptr(ptr1): return None
    si = m.u64(ptr1 + OFFSETS["spell_info"])
    if not is_ptr(si): return None
    np_ = m.u64(si + OFFSETS["spell_name"])
    if not is_ptr(np_): return None
    return m.string(np_, 64)


def deref_direct_spell(m, ptr1):
    """2-level: ptr1 → +0x28 (name_ptr) → string.
    Used if ptr1 is a direct SpellInfo*."""
    if not is_ptr(ptr1): return None
    np_ = m.u64(ptr1 + OFFSETS["spell_name"])
    if not is_ptr(np_): return None
    return m.string(np_, 64)


def valid_spell_name(nm):
    if not nm or len(nm) < 3 or not nm.isascii(): return False
    if not nm[0].isalpha(): return False
    return any(c.isupper() for c in nm)


def scan_hero_for_spells(m, hero_ptr, blob, verbose=False):
    """Scan for qwords that resolve to spell names via EITHER:
       (a) 3-level wrapped chain (ActiveSpellCast → SpellInfo → name)
       (b) 2-level direct chain (SpellInfo → name)
    Returns dict of {hero_offset: (chain_type, spell_name)}."""
    results = {}
    n_ptr = 0
    for off in range(0, len(blob) - 8, 8):
        p1 = struct.unpack_from('<Q', blob, off)[0]
        if not is_ptr(p1): continue
        n_ptr += 1
        # Try 3-level first (wrapped)
        nm = deref_wrapped_spell(m, p1)
        if valid_spell_name(nm):
            results[off] = ("wrapped", nm); continue
        # Try 2-level direct
        nm = deref_direct_spell(m, p1)
        if valid_spell_name(nm):
            results[off] = ("direct", nm); continue
    if verbose:
        print(f"    tested {n_ptr} pointers, found {len(results)} names")
    return results


def scan_hero_for_spellbook(m, hero_ptr, blob, verbose=False):
    """4-level: hero_blob[off] → SpellBook* → SpellSlot* → SpellInfo* → name.
    Returns dict of {hero_offset: {slot_idx: spell_name}}."""
    results = {}
    n_ptr = 0
    n_sb = 0
    for off in range(0, len(blob) - 8, 8):
        p1 = struct.unpack_from('<Q', blob, off)[0]
        if not is_ptr(p1): continue
        n_ptr += 1
        # Read first 16 qwords from candidate SpellBook
        sb = m.read(p1, 16 * 8)
        if not sb: continue
        slots = struct.unpack('<16Q', sb)
        if sum(1 for s in slots if is_ptr(s)) < 2:
            continue  # not enough pointers to be a spellbook
        n_sb += 1
        slot_names = {}
        for slot_idx, slot_ptr in enumerate(slots):
            if not is_ptr(slot_ptr): continue
            # Try wrapped chain (slot → SpellInfo → name)
            nm = deref_wrapped_spell(m, slot_ptr)
            if valid_spell_name(nm):
                slot_names[slot_idx] = nm
                continue
            # Try direct chain
            nm = deref_direct_spell(m, slot_ptr)
            if valid_spell_name(nm):
                slot_names[slot_idx] = nm
        if len(slot_names) >= 2:
            results[off] = slot_names
    if verbose:
        print(f"    tested {n_ptr} pointers, {n_sb} candidate spellbooks, {len(results)} match")
    return results


# ── Item scan ─────────────────────────────────────────────────
def scan_hero_for_items(m, hero_ptr, blob, item_ids, verbose=False):
    """1-level: hero_blob[off] → struct, look for item IDs in first 0x40 bytes."""
    results = {}
    for off in range(0, len(blob) - 8, 8):
        p1 = struct.unpack_from('<Q', blob, off)[0]
        if not is_ptr(p1): continue
        sub = m.read(p1, 0x40)
        if not sub: continue
        hits = []
        for so in range(0, len(sub) - 4, 4):
            v = struct.unpack_from('<I', sub, so)[0]
            if v in item_ids:
                hits.append((so, v))
        if hits:
            results[off] = hits
    return results


def scan_hero_for_items_2level(m, hero_ptr, blob, item_ids, verbose=False):
    """2-level: hero_blob[off] → array pointer → dereferenced qwords → item struct.
    Expects inventory pointer → array of 7 Item* pointers → Item struct with itemID.
    Returns dict of {hero_off: list of (array_slot, item_id)}."""
    results = {}
    for off in range(0, len(blob) - 8, 8):
        p_inv = struct.unpack_from('<Q', blob, off)[0]
        if not is_ptr(p_inv): continue
        # Treat p_inv as an array of 7 Item* (56 bytes)
        arr = m.read(p_inv, 56)
        if not arr: continue
        slots = struct.unpack('<7Q', arr)
        if not any(is_ptr(s) for s in slots): continue
        # For each slot, deref and search for itemID
        slot_hits = []
        for slot_idx, slot_ptr in enumerate(slots):
            if not is_ptr(slot_ptr): continue
            sub = m.read(slot_ptr, 0x40)
            if not sub: continue
            for so in range(0, len(sub) - 4, 4):
                v = struct.unpack_from('<I', sub, so)[0]
                if v in item_ids:
                    slot_hits.append((slot_idx, v, so))
                    break
        if len(slot_hits) >= 2:
            results[off] = slot_hits
    return results


def main():
    print("=" * 60)
    print("Spellbook + inventory offset scanner (deep deref)")
    print("=" * 60)

    launch_replay()
    if not wait_for_loaded(240):
        print("[FAIL] game did not load")
        return 1

    pid = find_league_pid()
    base, size = find_module_base(pid)
    print(f"[mem] pid={pid} base=0x{base:x} size=0x{size:x}")
    if size != EXPECTED_MOD_SIZE:
        print(f"[WARN] module size mismatch")

    m = Mem(pid)
    heroes = init_heroes(m, base)
    print(f"[mem] heroes: {list(heroes.keys())}")

    # Advance to gt=300 to ensure items/summoners populated
    print("[play] advancing to gt=300s...")
    reached = play_until(300.0, max_wall=200)
    print(f"[play] gameTime reached: {reached:.1f}")

    gt1 = get_ground_truth()
    print(f"[gt] ground truth collected")
    for c, v in gt1.items():
        print(f"  {c}: items={v['items']} summs=({v['summ1']}, {v['summ2']})")

    def match_mem_key(api_name):
        if api_name in heroes: return api_name
        k = api_name.replace("'", "").replace(" ", "").replace(".", "")
        if k in heroes: return k
        for mk in heroes:
            if mk.lower() == k.lower(): return mk
        return None

    # ── Sanity check: verify known ActiveSpell chain works ──
    print("\n" + "=" * 60)
    print("Sanity: verify known ActiveSpell chain")
    print("=" * 60)
    for mk, info in list(heroes.items())[:3]:
        hp = info["ptr"]
        active = m.u64(hp + OFFSETS["active_spell"])
        if is_ptr(active):
            name = deref_wrapped_spell(m, active)
            print(f"  {mk}: active_spell @ hero+0x{OFFSETS['active_spell']:X} → {name}")
        else:
            print(f"  {mk}: active_spell = 0 (not casting)")

    # ── Scan Garen's hero blob for all spell chains ──
    print("\n" + "=" * 60)
    print("Pass 1: deep scan Garen's hero blob for spell chains")
    print("=" * 60)

    garen_key = match_mem_key("Garen")
    if not garen_key:
        print("[FAIL] no Garen in heroes"); return 1

    garen_ptr = heroes[garen_key]["ptr"]
    # Try decreasing blob sizes in case of page boundary
    garen_blob = None
    for size in (0x20000, 0x10000, 0x8000, 0x6000, 0x5000):
        garen_blob = m.read(garen_ptr, size)
        if garen_blob:
            print(f"  Garen ptr=0x{garen_ptr:x} blob={len(garen_blob)} bytes")
            break
    if not garen_blob:
        # Page-by-page fallback
        garen_blob = b''
        for chunk in range(0, 0x20000, 0x1000):
            c = m.read(garen_ptr + chunk, 0x1000)
            if c: garen_blob += c
            else: break
        print(f"  Garen ptr=0x{garen_ptr:x} blob={len(garen_blob)} bytes (chunked)")
    if not garen_blob or len(garen_blob) < 0x1000:
        print("[FAIL] could not read Garen blob"); return 1

    t0 = time.time()
    garen_spells = scan_hero_for_spells(m, garen_ptr, garen_blob, verbose=True)
    print(f"  2/3-level scan took {time.time()-t0:.1f}s")
    print(f"\n  {len(garen_spells)} spell chains found in Garen:")
    for off in sorted(garen_spells):
        chain, nm = garen_spells[off]
        print(f"    hero+0x{off:04X} [{chain}]: {nm}")

    # 4-level SpellBook scan
    print(f"\n  Running 4-level SpellBook scan...")
    t0 = time.time()
    garen_spellbook = scan_hero_for_spellbook(m, garen_ptr, garen_blob, verbose=True)
    print(f"  4-level scan took {time.time()-t0:.1f}s")
    print(f"  {len(garen_spellbook)} spellbook candidates:")
    for off, slots in sorted(garen_spellbook.items(), key=lambda kv: -len(kv[1]))[:20]:
        names = [f"s{i}={n}" for i, n in sorted(slots.items())]
        print(f"    hero+0x{off:04X}: {names}")

    # ── Cross-validate with LeeSin (first champ) ──
    print("\n" + "=" * 60)
    print("Pass 2: cross-validate on other champions")
    print("=" * 60)

    for cross_mk in list(heroes.keys())[:3]:
        if cross_mk == garen_key: continue
        cross_ptr = heroes[cross_mk]["ptr"]
        cross_blob = m.read(cross_ptr, 0x20000)
        if not cross_blob: continue
        print(f"\n  {cross_mk}:")
        for off in sorted(garen_spells.keys())[:30]:
            p1 = struct.unpack_from('<Q', cross_blob, off)[0]
            if not is_ptr(p1): continue
            chain, _ = garen_spells[off]
            name = deref_wrapped_spell(m, p1) if chain == "wrapped" else deref_direct_spell(m, p1)
            if valid_spell_name(name):
                print(f"    hero+0x{off:04X}: {name}")

    # ── Identify summoner spell slots ──
    print("\n" + "=" * 60)
    print("Summoner slots (Garen)")
    print("=" * 60)
    summoner_slots = {
        off: nmt for off, nmt in garen_spells.items() if nmt[1].startswith("Summoner")
    }
    for off, (chain, nm) in sorted(summoner_slots.items()):
        print(f"  hero+0x{off:04X} [{chain}]: {nm}")

    # ── Identify Q/W/E/R slots ──
    print("\n" + "=" * 60)
    print("Champ ability slots (Garen Q/W/E/R)")
    print("=" * 60)
    ability_prefix = garen_key  # "Garen"
    ability_slots = {
        off: nmt for off, nmt in garen_spells.items()
        if nmt[1].startswith(ability_prefix) and len(nmt[1]) > len(ability_prefix)
    }
    for off, (chain, nm) in sorted(ability_slots.items()):
        print(f"  hero+0x{off:04X} [{chain}]: {nm}")

    # ── Item scan ──
    print("\n" + "=" * 60)
    print("Pass 3: item scan (Garen)")
    print("=" * 60)
    gt_garen = gt1.get("Garen", {})
    items = set(i for i in gt_garen.get("items", []) if i > 0)
    if items:
        print(f"  known items: {items}")

        t0 = time.time()
        one_level = scan_hero_for_items(m, garen_ptr, garen_blob, items)
        print(f"  1-level scan: {time.time()-t0:.1f}s, {len(one_level)} offsets")
        for off, hits in sorted(one_level.items(), key=lambda kv: -len(kv[1]))[:15]:
            ids = [f"sub+0x{so:x}:id={iid}" for so, iid in hits[:5]]
            print(f"    hero+0x{off:04X}: {ids}")

        t0 = time.time()
        two_level = scan_hero_for_items_2level(m, garen_ptr, garen_blob, items)
        print(f"\n  2-level scan: {time.time()-t0:.1f}s, {len(two_level)} offsets")
        for off, hits in sorted(two_level.items(), key=lambda kv: -len(kv[1]))[:15]:
            desc = [f"slot{si}:id={iid}@sub+0x{so:x}" for si, iid, so in hits]
            print(f"    hero+0x{off:04X}: {desc}")
    else:
        print("  no items in inventory")

    m.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(2)
