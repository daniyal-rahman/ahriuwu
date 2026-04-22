"""
Dump the BasicAttackBase structure at hero+0x4010 for all heroes.
Map out: spell name, target NetID, cast positions, windup, etc.

Chain discovered: hero+0x4010 -> ptr1 -> +0x00 -> ptr2 -> +0x28 -> spell_name

Also explore hero+0x4578 (had "LeeSinBasicAttack4" variant).
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
def is_map(v): return v and -500<v[0]<16000 and -500<v[1]<1000 and -500<v[2]<16000 and (v[0]!=0 or v[2]!=0)

def dump_object(m, ptr, size=0x200, label=""):
    """Dump an object's fields: pointers, strings, floats, ints, vec3s."""
    data = m.read(ptr, size)
    if not data:
        print(f"  [{label}] FAILED TO READ 0x{ptr:X}")
        return

    print(f"\n  [{label}] 0x{ptr:X} ({size} bytes):")

    # Pass 1: find all interesting values
    for off in range(0, len(data) - 8, 4):
        # Try as pointer (8 bytes)
        if off % 8 == 0 and off + 8 <= len(data):
            val64 = struct.unpack("<Q", data[off:off+8])[0]
            if is_heap(val64):
                s = m.string(val64, 64)
                if s and len(s) >= 3 and all(32 <= ord(c) < 127 for c in s[:3]):
                    print(f"    +0x{off:03X}: ptr 0x{val64:X} -> '{s[:60]}'")
                    continue
                # Try one more deref
                inner = m.u64(val64)
                if is_heap(inner):
                    s2 = m.string(inner, 64)
                    if s2 and len(s2) >= 3 and all(32 <= ord(c) < 127 for c in s2[:3]):
                        print(f"    +0x{off:03X}: ptr -> ptr -> '{s2[:60]}'")
                        continue

        # Try as float
        val32 = struct.unpack("<f", data[off:off+4])[0]
        if 0.01 < abs(val32) < 100000 and val32 != float('inf'):
            # Check if it's a reasonable game value
            ival = struct.unpack("<I", data[off:off+4])[0]
            # Skip if it looks more like an address fragment
            if ival < 0x100000:
                pass  # Skip - only print vec3/special floats below

        # Try as vec3 (12 bytes)
        if off + 12 <= len(data):
            v = struct.unpack("<fff", data[off:off+12])
            if is_map(v):
                print(f"    +0x{off:03X}: vec3 ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}) MAP POS")

        # Try as u32
        val32i = struct.unpack("<I", data[off:off+4])[0]
        if 0x40000000 <= val32i <= 0x400000FF:
            print(f"    +0x{off:03X}: u32 0x{val32i:08X} (NETID RANGE!)")

    # Pass 2: look for specific patterns
    # NetID pattern (0x400000xx)
    for off in range(0, len(data) - 4, 4):
        val = struct.unpack("<I", data[off:off+4])[0]
        if 0x40000000 <= val <= 0x400001FF:
            print(f"    +0x{off:03X}: NETID 0x{val:08X}")

    # Float timer values (0-5000 range, game time like)
    for off in range(0, len(data) - 4, 4):
        val = struct.unpack("<f", data[off:off+4])[0]
        if 1.0 < val < 5000.0:
            ival = struct.unpack("<I", data[off:off+4])[0]
            if ival < 0x10000000:  # Not a pointer fragment
                print(f"    +0x{off:03X}: f32 {val:.3f} (timer?)")


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

    print(f"Heroes: {[(h['name'], hex(h['net_id'] or 0)) for h in heroes]}")

    # ================================================================
    # 1. Dump BasicAttackBase structure for first hero
    # ================================================================
    print("\n" + "="*60)
    print("1. BasicAttackBase at hero+0x4010")
    print("="*60)

    hero = heroes[0]
    ba_ptr = m.u64(hero["ptr"] + 0x4010)
    print(f"\n  {hero['name']} hero+0x4010 = 0x{ba_ptr:X}" if is_heap(ba_ptr) else f"\n  {hero['name']} hero+0x4010 = NULL")

    if is_heap(ba_ptr):
        # Dump the BasicAttackBase object itself
        dump_object(m, ba_ptr, 0x200, "BasicAttackBase")

        # The spell name is at: ba_ptr -> +0x00 -> +0x28 -> string
        spell_info = m.u64(ba_ptr)
        if is_heap(spell_info):
            print(f"\n  BasicAttackBase -> SpellInfo at 0x{spell_info:X}")
            dump_object(m, spell_info, 0x200, "SpellInfo (BA+0x00)")

            spell_name_ptr = m.u64(spell_info + 0x28)
            if is_heap(spell_name_ptr):
                spell_name = m.string(spell_name_ptr, 64)
                print(f"\n  SpellInfo+0x28 -> spell name: '{spell_name}'")

    # ================================================================
    # 2. Cross-validate hero+0x4010 on all heroes
    # ================================================================
    print("\n" + "="*60)
    print("2. BasicAttackBase cross-validation (all heroes)")
    print("="*60)

    for h in heroes:
        ba = m.u64(h["ptr"] + 0x4010)
        if not is_heap(ba):
            print(f"  {h['name']:15s}: NULL")
            continue

        # Follow chain: ba -> +0x00 -> +0x28 -> name
        si = m.u64(ba)
        name = None
        if is_heap(si):
            np = m.u64(si + 0x28)
            if is_heap(np):
                name = m.string(np, 64)

        # Look for target NetID in the BA struct
        # Common offsets for target: check around where pandoras had it
        # hacker_logs: oBasicAttackOffset1 = 0x2C0, oBasicAttackOffset2 = 0x70
        target_netid = None
        for toff in range(0, 0x180, 4):
            val = m.u32(ba + toff)
            if val and 0x40000000 <= val <= 0x400001FF:
                target_netid = (toff, val)
                break

        # Look for position (caster/target) in BA struct
        cast_pos = None
        for poff in range(0, 0x180, 4):
            v = m.vec3(ba + poff)
            if is_map(v):
                cast_pos = (poff, v)
                break

        target_str = f"target=0x{target_netid[1]:08X}@+0x{target_netid[0]:02X}" if target_netid else "no_target"
        pos_str = f"pos@+0x{cast_pos[0]:02X}=({cast_pos[1][0]:.0f},{cast_pos[1][2]:.0f})" if cast_pos else "no_pos"
        name_str = f"'{name}'" if name else "no_name"

        print(f"  {h['name']:15s}: {name_str:35s} {target_str:30s} {pos_str}")

    # ================================================================
    # 3. Dump hero+0x4578 (had "LeeSinBasicAttack4")
    # ================================================================
    print("\n" + "="*60)
    print("3. ActiveSpell? at hero+0x4578")
    print("="*60)

    for h in heroes[:3]:
        as_ptr = m.u64(h["ptr"] + 0x4578)
        if not is_heap(as_ptr):
            print(f"  {h['name']}: NULL")
            continue

        print(f"\n  {h['name']} hero+0x4578 = 0x{as_ptr:X}")

        # Follow: +0x38 -> +0x28 -> name
        si38 = m.u64(as_ptr + 0x38)
        if is_heap(si38):
            np = m.u64(si38 + 0x28)
            if is_heap(np):
                name = m.string(np, 64)
                print(f"    +0x38 -> +0x28 -> '{name}'")

        # Also try direct +0x00 chain
        si00 = m.u64(as_ptr)
        if is_heap(si00):
            np = m.u64(si00 + 0x28)
            if is_heap(np):
                name = m.string(np, 64)
                print(f"    +0x00 -> +0x28 -> '{name}'")

        dump_object(m, as_ptr, 0x180, f"{h['name']} ActiveSpell?")

    # ================================================================
    # 4. Find SpellBook by searching wider range (0x3000-0x5000)
    # ================================================================
    print("\n" + "="*60)
    print("4. SpellBook search (hero+0x3000-0x5000)")
    print("="*60)

    hero = heroes[0]
    print(f"  Scanning {hero['name']} for objects with champion spell names...")

    for off in range(0x3000, 0x5000, 8):
        ptr = m.u64(hero["ptr"] + off)
        if not is_heap(ptr):
            continue

        # Read object and look for spell names
        # A SpellBook typically has: array of SpellSlot pointers
        # Each SpellSlot -> SpellInfo -> SpellData with name
        # Try various sub-structures
        found_spells = []

        # Check up to 13 slots at different array bases within the object
        for array_base in [0x0, 0x8, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38,
                           0x488, 0x490, 0x498, 0x4A0, 0x4A8,
                           0xAE0, 0xAE8, 0xAF0]:
            slot_ptr = m.u64(ptr + array_base)
            if not is_heap(slot_ptr):
                continue

            # Try reading spell name through: slot -> spellinfo -> name
            for name_chain in [(0x00, 0x28), (0x08, 0x28), (0x10, 0x28),
                               (0x28,), (0x30,), (0x38,),
                               (0x00, 0x00, 0x28), (0x00, 0x30)]:
                current = slot_ptr
                valid = True
                for step in name_chain[:-1]:
                    current = m.u64(current + step)
                    if not is_heap(current):
                        valid = False
                        break
                if not valid:
                    continue
                last_step = name_chain[-1]
                name_ptr = m.u64(current + last_step)
                if is_heap(name_ptr):
                    name = m.string(name_ptr, 64)
                    if name and len(name) > 3 and name[0].isalpha():
                        # Is it a champion spell?
                        if any(kw in name for kw in ["Blind", "Lee", "Monk", "Garen",
                                                      "Spell", "Flash", "Smite",
                                                      "Summon", "Ignite", "Heal",
                                                      "Barrier", "Teleport"]):
                            found_spells.append((array_base, name_chain, name))

        if found_spells:
            print(f"\n  hero+0x{off:04X} -> 0x{ptr:X}: SPELL NAMES FOUND!")
            for ab, chain, name in found_spells[:8]:
                chain_str = "->".join(f"+0x{s:02X}" for s in chain)
                print(f"    slot+0x{ab:03X} {chain_str}: '{name}'")

    # ================================================================
    # 5. Temporal analysis: take 2 snapshots of BA struct to find changing fields
    # ================================================================
    print("\n" + "="*60)
    print("5. Temporal analysis of BasicAttackBase (2 snapshots, 3s apart)")
    print("="*60)

    hero = heroes[0]
    ba_ptr = m.u64(hero["ptr"] + 0x4010)
    if is_heap(ba_ptr):
        snap1 = m.read(ba_ptr, 0x200)
        # Also read the SpellInfo
        si_ptr = m.u64(ba_ptr)
        si_snap1 = m.read(si_ptr, 0x200) if is_heap(si_ptr) else None

        time.sleep(3)

        snap2 = m.read(ba_ptr, 0x200)
        si_snap2 = m.read(si_ptr, 0x200) if is_heap(si_ptr) else None

        if snap1 and snap2:
            print(f"\n  BA struct changes (hero={hero['name']}):")
            changes = 0
            for i in range(0, len(snap1) - 4, 4):
                v1 = struct.unpack("<I", snap1[i:i+4])[0]
                v2 = struct.unpack("<I", snap2[i:i+4])[0]
                if v1 != v2:
                    changes += 1
                    f1 = struct.unpack("<f", snap1[i:i+4])[0]
                    f2 = struct.unpack("<f", snap2[i:i+4])[0]
                    marker = ""
                    if 0.01 < abs(f2) < 100000: marker = f" (f32: {f1:.3f} -> {f2:.3f})"
                    print(f"    BA+0x{i:03X}: 0x{v1:08X} -> 0x{v2:08X}{marker}")
            if changes == 0:
                print("    No changes (hero may not be auto-attacking)")

        if si_snap1 and si_snap2:
            print(f"\n  SpellInfo struct changes:")
            changes = 0
            for i in range(0, len(si_snap1) - 4, 4):
                v1 = struct.unpack("<I", si_snap1[i:i+4])[0]
                v2 = struct.unpack("<I", si_snap2[i:i+4])[0]
                if v1 != v2:
                    changes += 1
                    f1 = struct.unpack("<f", si_snap1[i:i+4])[0]
                    f2 = struct.unpack("<f", si_snap2[i:i+4])[0]
                    marker = ""
                    if 0.01 < abs(f2) < 100000: marker = f" (f32: {f1:.3f} -> {f2:.3f})"
                    print(f"    SI+0x{i:03X}: 0x{v1:08X} -> 0x{v2:08X}{marker}")
            if changes == 0:
                print("    No changes")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
