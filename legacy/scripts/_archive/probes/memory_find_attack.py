"""
Find oBasicAttackBase by scanning hero struct for pointers
to objects containing attack-related strings or spell data.

Strategy:
1. Scan hero struct offsets 0x2000-0x5000 for heap pointers
2. For each pointer, scan the target object for strings like
   "BasicAttack", "Attack", spell names, champion names
3. Also follow pointer chains (ptr -> ptr -> string)
4. Cross-reference: the active attack should have the CASTER's
   position and a TARGET NetID matching another hero
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
    _fields_ = [("dwSize",wintypes.DWORD),("cntUsage",wintypes.DWORD),("th32ProcessID",wintypes.DWORD),("th32DefaultHeapID",ctypes.POINTER(ctypes.c_ulong)),("th32ModuleID",wintypes.DWORD),("cntThreads",wintypes.DWORD),("th32ParentProcessID",wintypes.DWORD),("pcPriClassBase",ctypes.c_long),("dwFlags",wintypes.DWORD),("szExeFile",ctypes.c_char*MAX_PATH)]
class MODULEENTRY32(ctypes.Structure):
    _fields_ = [("dwSize",wintypes.DWORD),("th32ModuleID",wintypes.DWORD),("th32ProcessID",wintypes.DWORD),("GlblcntUsage",wintypes.DWORD),("ProccntUsage",wintypes.DWORD),("modBaseAddr",ctypes.POINTER(ctypes.c_byte)),("modBaseSize",wintypes.DWORD),("hModule",wintypes.HMODULE),("szModule",ctypes.c_char*(MAX_MODULE_NAME32+1)),("szExePath",ctypes.c_char*MAX_PATH)]

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
def find_league():
    s=kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);pe=PROCESSENTRY32();pe.dwSize=ctypes.sizeof(PROCESSENTRY32);kernel32.Process32First(s,ctypes.byref(pe))
    while True:
        if b"League of Legends" in pe.szExeFile: pid=pe.th32ProcessID;kernel32.CloseHandle(s);return pid
        if not kernel32.Process32Next(s,ctypes.byref(pe)): break
    kernel32.CloseHandle(s);return None
def find_base(pid):
    s=kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE|TH32CS_SNAPMODULE32,pid);me=MODULEENTRY32();me.dwSize=ctypes.sizeof(MODULEENTRY32);kernel32.Module32First(s,ctypes.byref(me))
    while True:
        if b"league of legends" in me.szModule.lower(): b=ctypes.cast(me.modBaseAddr,ctypes.c_void_p).value;sz=me.modBaseSize;kernel32.CloseHandle(s);return b,sz
        if not kernel32.Module32Next(s,ctypes.byref(me)): break
    kernel32.CloseHandle(s);return None,None

class Mem:
    def __init__(s,pid): s.h=kernel32.OpenProcess(PROCESS_VM_READ|PROCESS_QUERY_INFORMATION,False,pid)
    def read(s,a,sz):
        buf=ctypes.create_string_buffer(sz);n=ctypes.c_size_t(0)
        ok=kernel32.ReadProcessMemory(s.h,ctypes.c_void_p(a),buf,sz,ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value==sz else None
    def u32(s,a): d=s.read(a,4); return struct.unpack("<I",d)[0] if d else None
    def u64(s,a): d=s.read(a,8); return struct.unpack("<Q",d)[0] if d else None
    def f32(s,a): d=s.read(a,4); return struct.unpack("<f",d)[0] if d else None
    def vec3(s,a): d=s.read(a,12); return struct.unpack("<fff",d) if d else None
    def string(s,a,n=128):
        d=s.read(a,n)
        if not d: return None
        return d.split(b'\x00')[0].decode('ascii',errors='replace') or None
    def close(s): kernel32.CloseHandle(s.h)

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF

def find_strings_in_object(m, ptr, size=0x400, keywords=None):
    """Read an object and find any readable strings, optionally filtering by keywords."""
    data = m.read(ptr, size)
    if not data:
        return []

    results = []
    # Check for inline strings (direct ASCII in the object)
    for i in range(0, len(data) - 4):
        # Check if this looks like a printable ASCII run
        if 32 <= data[i] < 127 and 32 <= data[i+1] < 127 and 32 <= data[i+2] < 127:
            end = i
            while end < len(data) and 32 <= data[end] < 127:
                end += 1
            s = data[i:end].decode('ascii', errors='replace')
            if len(s) >= 4:
                if keywords is None or any(kw.lower() in s.lower() for kw in keywords):
                    results.append((i, s, "inline"))
                i = end  # skip ahead

    # Check for pointer-to-string (each 8 bytes as a pointer, then read string)
    for i in range(0, len(data) - 8, 8):
        ptr_val = struct.unpack("<Q", data[i:i+8])[0]
        if is_heap(ptr_val):
            s = m.string(ptr_val, 64)
            if s and len(s) >= 4 and all(32 <= ord(c) < 127 for c in s[:4]):
                if keywords is None or any(kw.lower() in s.lower() for kw in keywords):
                    results.append((i, s, "ptr"))

    return results

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    pid = find_league(); base, _ = find_base(pid); m = Mem(pid)
    print(f"PID={pid} Base=0x{base:X}")

    arr_ptr = m.u64(base + HERO_ARRAY_RVA)
    heroes = []
    for i in range(10):
        hp = m.u64(arr_ptr + i*8)
        if not is_heap(hp): continue
        name = m.string(hp + 0x4328, 32) or f"h{i}"
        pos = m.vec3(hp + 0x25C)
        net_id = m.u32(hp + 0xCC)
        heroes.append({"ptr": hp, "name": name, "pos": pos, "net_id": net_id, "idx": i})

    print(f"Found {len(heroes)} heroes: {[h['name'] for h in heroes]}")

    # Attack-related keywords to search for
    attack_kw = ["Attack", "Basic", "Spell", "Cast", "Crit",
                 "Passive", "Melee", "Ranged", "Missile"]
    # Champion-specific attack names
    for h in heroes:
        attack_kw.append(h["name"])

    # ================================================================
    # SCAN 1: Broad scan of hero struct for attack-string pointers
    # ================================================================
    print("\n=== SCAN 1: Hero struct -> objects with attack strings ===")
    print("Scanning offsets 0x2000-0x5000...")

    hits = {}  # offset -> [(string, location_type)]
    test_heroes = heroes[:3]  # Test on first 3 heroes

    for hero in test_heroes:
        print(f"\n  {hero['name']} (ptr=0x{hero['ptr']:X}):")
        for off in range(0x2000, 0x5000, 8):
            ptr = m.u64(hero["ptr"] + off)
            if not is_heap(ptr):
                continue

            # Search this object for attack strings
            strings = find_strings_in_object(m, ptr, 0x200, attack_kw)
            if strings:
                if off not in hits:
                    hits[off] = []
                for str_off, s, stype in strings:
                    hits[off].append((hero["name"], str_off, s, stype))
                    if len(hits[off]) <= 6:  # Don't flood output
                        print(f"    +0x{off:04X} -> +0x{str_off:03X} ({stype}): '{s[:60]}'")

            # Also follow one level of pointer chain
            for sub_off in [0x0, 0x8, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38]:
                sub_ptr = m.u64(ptr + sub_off)
                if is_heap(sub_ptr):
                    strings2 = find_strings_in_object(m, sub_ptr, 0x200, attack_kw)
                    for str_off, s, stype in strings2:
                        key = (off, sub_off)
                        print(f"    +0x{off:04X} -> +0x{sub_off:02X} -> +0x{str_off:03X} ({stype}): '{s[:60]}'")

    # ================================================================
    # SCAN 2: Specifically check known offsets from hacker_logs
    # ================================================================
    print("\n\n=== SCAN 2: Check known offsets with broader string search ===")
    known_offsets = {
        0x2A70: "ActiveSpell (pandoras)",
        0x2C68: "BasicAttackBase",
        0x30E8: "SpellBook",
        0x2C0:  "BasicAttackOffset1 (from BA base)",
        0x2868: "close to BuffManager 0x28B8",
    }

    for hero in test_heroes[:1]:
        print(f"\n  {hero['name']}:")
        for off, label in known_offsets.items():
            ptr = m.u64(hero["ptr"] + off)
            print(f"    +0x{off:04X} ({label}): ", end="")
            if not is_heap(ptr):
                print(f"0x{ptr:X} (not heap)" if ptr else "NULL")
                continue
            print(f"0x{ptr:X}")

            # Read broader string search (no keyword filter)
            all_strings = find_strings_in_object(m, ptr, 0x300, None)
            attack_strings = [s for s in all_strings if any(
                kw.lower() in s[1].lower() for kw in attack_kw)]

            if attack_strings:
                print(f"      ATTACK STRINGS FOUND:")
                for str_off, s, stype in attack_strings[:5]:
                    print(f"        +0x{str_off:03X} ({stype}): '{s[:80]}'")
            else:
                # Show first few strings anyway
                for str_off, s, stype in all_strings[:3]:
                    print(f"        +0x{str_off:03X} ({stype}): '{s[:80]}'")

    # ================================================================
    # SCAN 3: Find SpellBook by looking for spell slot array pattern
    # ================================================================
    print("\n\n=== SCAN 3: Find SpellBook via spell slot pattern ===")
    # SpellBook typically has 13 spell slots (Q,W,E,R + passives + summoners)
    # Each slot is a pointer. The spell name can be found through the slot.

    hero = heroes[0]
    print(f"  Scanning {hero['name']} for SpellBook-like structures...")

    for off in range(0x2800, 0x4000, 8):
        ptr = m.u64(hero["ptr"] + off)
        if not is_heap(ptr):
            continue

        # A SpellBook has spell slots at known sub-offsets
        # Try reading a few pointers as spell slots
        # Common pattern: spellbook + 0x8*i for i in range(6) = Q,W,E,R,D,F
        # Or: spellbook + some_base + 0x8*i
        spell_names = []
        for slot_base in [0x0, 0x8, 0x10, 0x18, 0x20, 0x28, 0x30,
                          0x488, 0x490, 0x498, 0x4A0,  # pandoras had 0x488+
                          0xAE0, 0xAE8, 0xAF0, 0xAF8]:  # hacker_logs slot offset
            slot_ptr = m.u64(ptr + slot_base)
            if not is_heap(slot_ptr):
                continue
            # Spell slot -> spell info -> spell name
            for name_off in [0x28, 0x30, 0x38, 0x40, 0x128]:
                name_ptr = m.u64(slot_ptr + name_off)
                if is_heap(name_ptr):
                    name = m.string(name_ptr, 48)
                    if name and len(name) > 3 and name[0].isupper():
                        spell_names.append((slot_base, name_off, name))
                # Also try inline string
                name = m.string(slot_ptr + name_off, 48)
                if name and "Spell" in name or "Attack" in name:
                    spell_names.append((slot_base, name_off, f"(inline){name}"))

        if spell_names:
            print(f"\n  hero+0x{off:04X} -> 0x{ptr:X}: SPELL NAMES FOUND")
            for sb, no, name in spell_names[:10]:
                print(f"    slot+0x{sb:03X} -> +0x{no:02X}: '{name}'")

    # ================================================================
    # SCAN 4: Time-based detection — find fields that change during auto-attack
    # ================================================================
    print("\n\n=== SCAN 4: Monitor for auto-attack state changes ===")
    print("Taking 2 snapshots 3s apart of hero struct region 0x2800-0x3200...")

    hero = heroes[0]
    # Read a chunk of the hero struct
    snap1 = m.read(hero["ptr"] + 0x2800, 0xA00)
    time.sleep(3)
    snap2 = m.read(hero["ptr"] + 0x2800, 0xA00)

    if snap1 and snap2:
        changed_ranges = []
        for i in range(0, len(snap1) - 8, 8):
            v1 = struct.unpack("<Q", snap1[i:i+8])[0]
            v2 = struct.unpack("<Q", snap2[i:i+8])[0]
            if v1 != v2:
                off = 0x2800 + i
                changed_ranges.append((off, v1, v2))

        print(f"  {len(changed_ranges)} changed fields in 0x2800-0x3200:")
        for off, v1, v2 in changed_ranges[:30]:
            # Check if new value is a heap pointer
            marker = ""
            if is_heap(v2):
                s = m.string(v2, 32)
                if s and len(s) > 3:
                    marker = f" -> '{s[:40]}'"
                else:
                    marker = " [heap ptr]"
            elif 0 < v2 < 0xFFFFFFFF:
                f_val = struct.unpack("<f", struct.pack("<I", v2 & 0xFFFFFFFF))[0]
                if 0.01 < abs(f_val) < 100000:
                    marker = f" (f32={f_val:.2f})"
            print(f"    +0x{off:04X}: 0x{v1:016X} -> 0x{v2:016X}{marker}")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
