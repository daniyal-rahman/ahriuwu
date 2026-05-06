"""
Find oBasicAttackBase and oObjSpellBook - TARGETED version.

Skips stat descriptor region (0x2000-0x2600) and focuses on:
1. hero+0x2800-0x4000: Look for BasicAttack spell object
2. SpellBook detection via actual spell names (Q/W/E/R abilities)
3. BasicAttack detection via "BasicAttack" string in spell name field
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

def try_read_spell_name(m, spell_slot_ptr):
    """Try various patterns to read a spell name from a spell slot pointer."""
    if not is_heap(spell_slot_ptr):
        return None
    # Pattern 1: SpellSlot -> SpellInfo -> SpellData -> name
    # Try reading name at various sub-offsets as pointer-to-string
    for off in [0x28, 0x30, 0x38, 0x40, 0x48, 0x50, 0x58, 0x60, 0x68,
                0x128, 0x130, 0x138, 0x140]:
        ptr = m.u64(spell_slot_ptr + off)
        if is_heap(ptr):
            s = m.string(ptr, 64)
            if s and len(s) >= 3 and s[0].isalpha() and not all(c in '0123456789abcdef' for c in s[:8]):
                return (off, s, "ptr")
            # Try one more deref
            ptr2 = m.u64(ptr)
            if is_heap(ptr2):
                s2 = m.string(ptr2, 64)
                if s2 and len(s2) >= 3 and s2[0].isalpha():
                    return (off, s2, "ptr2")
        # Try inline string
        s = m.string(spell_slot_ptr + off, 32)
        if s and len(s) >= 4 and ("Spell" in s or "Attack" in s or "Passive" in s):
            return (off, s, "inline")
    return None

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
        heroes.append({"ptr": hp, "name": name, "idx": i})
    print(f"Heroes: {[h['name'] for h in heroes]}")

    # Known champion spell names for validation
    # Lee Sin: BlindMonkQOne, BlindMonkWOne, BlindMonkEOne, BlindMonkRKick
    # Garen: GarenQ, GarenW, GarenE, GarenR
    champ_spells = {
        "LeeSin": ["BlindMonk"],
        "Garen": ["Garen"],
        "Karthus": ["Karthus"],
        "Hwei": ["Hwei"],
        "Smolder": ["Smolder"],
        "Nami": ["Nami"],
        "Malphite": ["Malphite"],
        "Azir": ["Azir"],
        "Ashe": ["Ashe"],
        "Seraphine": ["Seraphine"],
    }

    # ================================================================
    # SCAN A: Find SpellBook by looking for consecutive spell slot ptrs
    # ================================================================
    print("\n=== SCAN A: SpellBook finder (hero+0x2800-0x3800) ===")
    hero = heroes[0]
    print(f"Testing on {hero['name']}...")

    # Read hero struct chunk
    chunk = m.read(hero["ptr"] + 0x2800, 0x1000)
    if not chunk:
        print("  Failed to read hero struct"); return

    spellbook_candidates = []

    for off in range(0, len(chunk) - 0x100, 8):
        # Look for a region with multiple consecutive heap pointers
        # SpellBook has spell slots as an array of pointers
        ptr_count = 0
        spell_names_found = []
        for slot in range(0, 0x80, 8):  # Check 16 consecutive 8-byte values
            val = struct.unpack("<Q", chunk[off+slot:off+slot+8])[0]
            if is_heap(val):
                ptr_count += 1
                # Try to read spell name from this slot
                result = try_read_spell_name(m, val)
                if result:
                    spell_names_found.append((slot, result))

        if ptr_count >= 4 and spell_names_found:
            hero_off = 0x2800 + off
            spellbook_candidates.append((hero_off, ptr_count, spell_names_found))

    # Show best candidates
    spellbook_candidates.sort(key=lambda x: len(x[2]), reverse=True)
    for hero_off, ptr_count, spell_names in spellbook_candidates[:5]:
        print(f"\n  hero+0x{hero_off:04X}: {ptr_count} heap ptrs, {len(spell_names)} spell names")
        for slot, (name_off, name, ntype) in spell_names:
            print(f"    slot+0x{slot:02X} -> name@+0x{name_off:02X} ({ntype}): '{name[:60]}'")

    # ================================================================
    # SCAN B: Direct search for "BasicAttack" string pointer in hero struct
    # ================================================================
    print("\n\n=== SCAN B: Direct 'BasicAttack' string search (hero+0x2600-0x4800) ===")

    # Read a big chunk of the hero struct
    big_chunk = m.read(hero["ptr"] + 0x2600, 0x2200)
    if not big_chunk:
        print("  Failed to read"); return

    basic_attack_hits = []
    for off in range(0, len(big_chunk) - 8, 8):
        val = struct.unpack("<Q", big_chunk[off:off+8])[0]
        if not is_heap(val):
            continue

        # Level 1: read string at val
        s = m.string(val, 64)
        if s and "attack" in s.lower():
            hero_off = 0x2600 + off
            basic_attack_hits.append((hero_off, 1, s))
            continue

        # Level 2: val -> ptr -> string
        for sub in range(0, 0x100, 8):
            ptr2 = m.u64(val + sub)
            if is_heap(ptr2):
                s2 = m.string(ptr2, 64)
                if s2 and "attack" in s2.lower():
                    hero_off = 0x2600 + off
                    basic_attack_hits.append((hero_off, 2, f"+0x{sub:02X}->{s2}"))
                    break

                # Level 3: val -> ptr -> ptr -> string
                for sub2 in range(0, 0x80, 8):
                    ptr3 = m.u64(ptr2 + sub2)
                    if is_heap(ptr3):
                        s3 = m.string(ptr3, 64)
                        if s3 and "attack" in s3.lower():
                            hero_off = 0x2600 + off
                            basic_attack_hits.append((hero_off, 3, f"+0x{sub:02X}->+0x{sub2:02X}->{s3}"))
                            break

    if basic_attack_hits:
        print(f"  Found {len(basic_attack_hits)} 'attack' string references:")
        for hero_off, depth, s in basic_attack_hits[:30]:
            print(f"    hero+0x{hero_off:04X} (depth={depth}): {s[:80]}")
    else:
        print("  No 'attack' string references found")

    # ================================================================
    # SCAN C: Cross-validate with second hero
    # ================================================================
    print("\n\n=== SCAN C: Cross-validate top hits on all heroes ===")

    # Collect unique offsets from basic_attack_hits
    interesting_offsets = sorted(set(h[0] for h in basic_attack_hits))[:20]

    if interesting_offsets:
        for off in interesting_offsets:
            results = []
            for h in heroes[:5]:
                val = m.u64(h["ptr"] + off)
                if is_heap(val):
                    s = m.string(val, 64)
                    if s:
                        results.append(f"{h['name']}='{s[:30]}'")
                    else:
                        results.append(f"{h['name']}=ptr")
                else:
                    results.append(f"{h['name']}=NULL")
            print(f"  hero+0x{off:04X}: {', '.join(results)}")

    # ================================================================
    # SCAN D: Look for SpellBook specifically at hacker_logs-like offsets
    # adjusted for our build (+0x200 to +0x400 shift expected)
    # ================================================================
    print("\n\n=== SCAN D: SpellBook shifted offset search ===")
    # hacker_logs: oObjSpellBook = 0x30E8, oBasicAttackBase = 0x2C68
    # Our build is bigger (0x207C000 vs 0x202D000), offsets may be ~0x200-0x400 higher

    for label, base_off, search_range in [
        ("BasicAttackBase", 0x2C68, 0x400),
        ("SpellBook", 0x30E8, 0x400),
    ]:
        print(f"\n  Searching for {label} near 0x{base_off:X} (+/- 0x{search_range:X}):")
        for off in range(base_off - search_range, base_off + search_range, 8):
            val = m.u64(hero["ptr"] + off)
            if not is_heap(val):
                continue

            # For BasicAttack: check for "BasicAttack" string in the object
            # For SpellBook: check for spell slot array
            s = m.string(val, 32)
            if s and len(s) > 3:
                # Direct string
                if "attack" in s.lower() or "spell" in s.lower() or "basic" in s.lower():
                    print(f"    hero+0x{off:04X} -> '{s}'")
                    continue

            # Check object for BasicAttack strings at common name offsets
            for name_off in [0x28, 0x30, 0x38, 0x128, 0x130]:
                ptr2 = m.u64(val + name_off)
                if is_heap(ptr2):
                    s2 = m.string(ptr2, 64)
                    if s2 and ("attack" in s2.lower() or "basic" in s2.lower()):
                        print(f"    hero+0x{off:04X} -> +0x{name_off:02X} -> '{s2}'")

            # Check for spell names (SpellBook detection)
            result = try_read_spell_name(m, val)
            if result:
                name_off, name, ntype = result
                if any(kw in name for kw in ["Blind", "Garen", "Karthus", hero["name"][:4]]):
                    print(f"    hero+0x{off:04X} -> spell name: '{name}' (@+0x{name_off:02X}, {ntype})")

    # ================================================================
    # SCAN E: Find "BasicAttack" by reading ALL strings in hero struct ptrs
    # ================================================================
    print("\n\n=== SCAN E: Exhaustive 'BasicAttack' deep scan (hero+0x2800-0x3400) ===")
    for off in range(0x2800, 0x3400, 8):
        val = m.u64(hero["ptr"] + off)
        if not is_heap(val):
            continue

        # Read 0x300 bytes from target and search for "BasicAttack" inline
        data = m.read(val, 0x300)
        if not data:
            continue

        # Check inline strings
        idx = data.find(b"BasicAttack")
        if idx >= 0:
            s = data[idx:idx+40].split(b'\x00')[0].decode('ascii', errors='replace')
            print(f"  hero+0x{off:04X} -> inline@+0x{idx:03X}: '{s}'")
            continue

        idx = data.find(b"Attack")
        if idx >= 0 and idx > 0 and data[idx-1:idx] not in [b'e', b'a']:  # avoid substring
            s = data[idx:idx+40].split(b'\x00')[0].decode('ascii', errors='replace')
            # Only print unique attack-related ones
            if "Basic" in s or s.startswith("Attack"):
                print(f"  hero+0x{off:04X} -> inline@+0x{idx:03X}: '{s}'")

        # Check all pointer fields for "BasicAttack" string
        for i in range(0, min(len(data), 0x200), 8):
            ptr2 = struct.unpack("<Q", data[i:i+8])[0]
            if is_heap(ptr2):
                s2 = m.string(ptr2, 64)
                if s2 and "BasicAttack" in s2:
                    print(f"  hero+0x{off:04X} -> +0x{i:03X} -> '{s2}'")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
