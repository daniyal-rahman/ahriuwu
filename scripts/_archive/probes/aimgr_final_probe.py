"""
Final targeted probes:
1. Follow pointer at stats_inner+0x010 (from the hero+0x4628 chain)
2. Investigate hero+0x2E58 chain that had velocity=509.8
3. Try to find movement target via the navpath approach
4. If all else fails, build velocity-based movement estimator
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
    return (100 < x < 15000 and -500 < y < 1000 and 100 < z < 15000)

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 70)
    print("Final Targeted Probes")
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

    # ================================================================
    # PROBE 1: Follow stats_inner+0x010 pointer
    # ================================================================
    print(f"\n{'='*70}")
    print("PROBE 1: Follow pointer at stats struct inner+0x010")
    print("=" * 70)

    for hero in heroes[:5]:
        ai_raw = m.u64(hero['ptr'] + 0x4628)
        if not is_heap(ai_raw): continue
        inner = m.u64(ai_raw + 0x10)
        if not is_heap(inner): continue

        mystery_ptr = m.u64(inner + 0x10)
        print(f"\n  {hero['name']}: inner+0x10 -> 0x{mystery_ptr:016X}", end="")

        if is_heap(mystery_ptr):
            print(f" (VALID HEAP)")
            # Dump 256 bytes
            data = m.read(mystery_ptr, 512)
            if data:
                # Show as hex + float interpretation
                print(f"    Dump of 0x{mystery_ptr:X}:")
                for i in range(0, min(256, len(data)), 16):
                    hex_str = ' '.join(f'{b:02X}' for b in data[i:i+16])
                    floats = []
                    for j in range(0, 16, 4):
                        if i+j+4 <= len(data):
                            f = struct.unpack("<f", data[i+j:i+j+4])[0]
                            if 100 < abs(f) < 16000:
                                floats.append(f"+0x{i+j:02X}:{f:.0f}")
                            elif 0 < abs(f) < 1000 and f != 0:
                                floats.append(f"+0x{i+j:02X}:{f:.1f}")
                    ptrs = []
                    for j in range(0, 16, 8):
                        if i+j+8 <= len(data):
                            p = struct.unpack("<Q", data[i+j:i+j+8])[0]
                            if is_heap(p):
                                ptrs.append(f"+0x{i+j:02X}:PTR")
                    extra = " ".join(ptrs + floats)
                    print(f"      +0x{i:03X}: {hex_str}  {extra}")

                # Search for map positions
                pos_hits = []
                hp = hero['pos']
                for off in range(0, len(data)-12, 4):
                    v = struct.unpack("<fff", data[off:off+12])
                    if is_map_pos(v):
                        dist = ((v[0]-hp[0])**2 + (v[2]-hp[2])**2)**0.5 if hp else 999
                        pos_hits.append((off, v, dist))
                if pos_hits:
                    print(f"    Map positions found:")
                    for off, v, dist in pos_hits[:10]:
                        print(f"      +0x{off:03X}: ({v[0]:.0f},{v[1]:.0f},{v[2]:.0f}) dist={dist:.0f}")
        else:
            print(f" (NOT HEAP)")

    # ================================================================
    # PROBE 2: hero+0x2E58 chain (had velocity=509.8)
    # ================================================================
    print(f"\n{'='*70}")
    print("PROBE 2: hero+0x2E58 chain analysis")
    print("=" * 70)

    for hero in heroes[:3]:
        ptr = m.u64(hero['ptr'] + 0x2E58)
        if not is_heap(ptr):
            print(f"  {hero['name']}: hero+0x2E58 = NULL"); continue

        print(f"\n  {hero['name']}: hero+0x2E58 -> 0x{ptr:X}")

        # Try inner deref at +0x10
        inner = m.u64(ptr + 0x10)
        if is_heap(inner):
            data = m.read(inner, 256)
            if data:
                print(f"    inner+0x10 -> 0x{inner:X}:")
                for i in range(0, 128, 16):
                    hex_str = ' '.join(f'{b:02X}' for b in data[i:i+16])
                    floats = []
                    for j in range(0, 16, 4):
                        f_val = struct.unpack("<f", data[i+j:i+j+4])[0]
                        if abs(f_val) > 0.01 and abs(f_val) < 100000:
                            floats.append(f"{f_val:.1f}")
                        else:
                            floats.append("---")
                    print(f"      +0x{i:03X}: {hex_str}  [{' | '.join(floats)}]")

    # ================================================================
    # PROBE 3: Try XOR key derivation for AiManager
    # ================================================================
    print(f"\n{'='*70}")
    print("PROBE 3: XOR key analysis for hero+0x41F0")
    print("=" * 70)

    # Read the XOR key global - look at GetAiManager function
    # At 0x292443: movzx ecx, byte [rbx+0x259]  -- reads a byte from hero+0x259
    # At 0x29244A: mov rax, [rip+0x1B2CAB7]     -- loads global table
    # The global table is at: 0x29244A + 7 + 0x1B2CAB7 = 0x1DBEF08
    # Wait, need to add base: rip = base + 0x292451 (after the 7-byte instruction)
    # global = base + 0x292451 + 0x1B2CAB7 = base + 0x1DBEF08

    global_table_rva = 0x292451 + 0x1B2CAB7  # = 0x1DBEF08
    global_table_ptr = m.u64(base + global_table_rva)
    print(f"  Global table at RVA 0x{global_table_rva:X} -> 0x{global_table_ptr:X}" if global_table_ptr else "NULL")

    for hero in heroes[:3]:
        enc_val = m.u64(hero['ptr'] + 0x41F0)
        idx_byte = m.read(hero['ptr'] + 0x259, 1)
        idx = idx_byte[0] if idx_byte else None
        print(f"\n  {hero['name']}: enc=0x{enc_val:016X}, idx_byte@+0x259={idx}")

        if global_table_ptr and is_heap(global_table_ptr) and idx is not None:
            # The function does: mov eax, [rax + rcx*4 + 0x20]
            # So it reads global_table + idx*4 + 0x20
            table_val = m.u32(global_table_ptr + idx * 4 + 0x20)
            print(f"    table[{idx}] = {table_val} (0x{table_val:X})" if table_val else "    table read failed")

    # ================================================================
    # PROBE 4: Brute-force search for AiManager via server position matching
    # ================================================================
    print(f"\n{'='*70}")
    print("PROBE 4: Find AiManager by searching heap for matching server pos")
    print("=" * 70)

    # For each hero, scan all pointers in the hero struct for a struct that contains
    # the hero's position AND a velocity float AND has changing data
    test_hero = heroes[5]  # Garen
    print(f"  Testing: {test_hero['name']} pos=({test_hero['pos'][0]:.0f},{test_hero['pos'][2]:.0f})")

    hp = test_hero['pos']
    if not hp:
        print("  No position!"); m.close(); return

    # Read entire hero struct
    hero_data = m.read(test_hero['ptr'], 0x6000)
    if not hero_data:
        print("  Failed to read hero struct"); m.close(); return

    # Scan for ALL pointers in the hero struct, follow chains up to 3 deep
    print(f"  Scanning hero struct for pointers to structs with matching position...")
    found = []

    for off in range(0, len(hero_data) - 8, 8):
        ptr1 = struct.unpack("<Q", hero_data[off:off+8])[0]
        if not is_heap(ptr1): continue

        # Check ptr1 directly
        for sub_off in [0x000, 0x034, 0x0D0, 0x330, 0x33C, 0x474]:
            v = m.vec3(ptr1 + sub_off)
            if v and is_map_pos(v):
                dist = ((v[0]-hp[0])**2 + (v[2]-hp[2])**2)**0.5
                if dist < 50:
                    # Check if there's also a velocity
                    vel = m.f32(ptr1 + 0x318)
                    vel2 = m.f32(ptr1 + sub_off + 0x318)
                    found.append((off, "direct", sub_off, v, dist, vel, vel2))

        # Check ptr1 -> deref(+0x08, +0x10, +0x18) -> check position
        for inner_off in [0x08, 0x10, 0x18]:
            ptr2 = m.u64(ptr1 + inner_off)
            if not is_heap(ptr2): continue

            for sub_off in [0x000, 0x034, 0x0D0, 0x330, 0x33C, 0x474]:
                v = m.vec3(ptr2 + sub_off)
                if v and is_map_pos(v):
                    dist = ((v[0]-hp[0])**2 + (v[2]-hp[2])**2)**0.5
                    if dist < 50:
                        vel = m.f32(ptr2 + 0x318)
                        found.append((off, f"via+0x{inner_off:X}", sub_off, v, dist, vel, None))

    # Show results
    print(f"\n  Found {len(found)} position matches (dist < 50):")
    seen_ptrs = set()
    for off, mode, sub, v, dist, vel, vel2 in found:
        key = (off, mode, sub)
        if key in seen_ptrs: continue
        seen_ptrs.add(key)
        vel_str = f"vel@0x318={vel:.0f}" if vel and 50 < vel < 1000 else ""
        print(f"    hero+0x{off:04X} [{mode}] pos@+0x{sub:03X}=({v[0]:.0f},{v[2]:.0f}) dist={dist:.0f} {vel_str}")

    # ================================================================
    # PROBE 5: ActiveSpellCast deep dive with temporal analysis
    # ================================================================
    print(f"\n{'='*70}")
    print("PROBE 5: ActiveSpellCast (hero+0x3120) temporal analysis")
    print("=" * 70)

    # Take 3 snapshots 1s apart to see spells changing
    for t in range(3):
        if t > 0: time.sleep(1.5)
        print(f"\n  Snapshot {t}:")
        for hero in heroes:
            active_ptr = m.u64(hero['ptr'] + 0x3120)
            if not is_heap(active_ptr): continue

            # Read spell name
            spell_info = m.u64(active_ptr + 0x008)
            spell_name = None
            if is_heap(spell_info):
                spell_name = m.string(spell_info + 0x28, 48)

            cast_start = m.vec3(active_ptr + 0x0D0)
            cast_target = m.vec3(active_ptr + 0x0DC)
            cast_time = m.f32(active_ptr + 0x0C0)  # try various time offsets
            cast_time2 = m.f32(active_ptr + 0x0C4)
            cast_time3 = m.f32(active_ptr + 0x0C8)

            if spell_name:
                start_str = f"({cast_start[0]:.0f},{cast_start[2]:.0f})" if cast_start and is_map_pos(cast_start) else "N/A"
                tgt_str = f"({cast_target[0]:.0f},{cast_target[2]:.0f})" if cast_target and is_map_pos(cast_target) else "N/A"

                # Try to find game time in spell struct
                time_str = ""
                spell_data = m.read(active_ptr, 0x250)
                if spell_data:
                    for toff in range(0xB0, 0xD0, 4):
                        tv = struct.unpack("<f", spell_data[toff:toff+4])[0]
                        if 10 < tv < 5000:
                            time_str += f" t@0x{toff:X}={tv:.1f}"

                    # Also check 0x1E0-0x210 range
                    for toff in range(0x1E0, 0x220, 4):
                        if toff + 4 <= len(spell_data):
                            tv = struct.unpack("<f", spell_data[toff:toff+4])[0]
                            if 10 < tv < 5000:
                                time_str += f" t@0x{toff:X}={tv:.1f}"

                    # Check for target netID at various offsets
                    for noff in [0x088, 0x090, 0x098, 0x0A0, 0x0A8, 0x0B0, 0x120, 0x128, 0x130]:
                        if noff + 4 <= len(spell_data):
                            nid = struct.unpack("<I", spell_data[noff:noff+4])[0]
                            if 0x40000000 < nid < 0x50000000:
                                # This looks like a netID
                                # Check which hero has this netID
                                for h2 in heroes:
                                    h2_netid = m.u32(h2['ptr'] + 0xCC)
                                    if h2_netid == nid:
                                        time_str += f" TARGET_NETID@0x{noff:X}={h2['name']}"
                                        break
                                else:
                                    time_str += f" netid@0x{noff:X}=0x{nid:X}"

                print(f"    {hero['name']:12s} '{spell_name}' from={start_str} to={tgt_str}{time_str}")

    # ================================================================
    # PROBE 6: Check if ActiveSpell has a target pointer/netID
    # ================================================================
    print(f"\n{'='*70}")
    print("PROBE 6: ActiveSpell target identification")
    print("=" * 70)

    for hero in heroes[:5]:
        active_ptr = m.u64(hero['ptr'] + 0x3120)
        if not is_heap(active_ptr): continue

        spell_info = m.u64(active_ptr + 0x008)
        spell_name = None
        if is_heap(spell_info):
            spell_name = m.string(spell_info + 0x28, 48)

        if not spell_name or "BasicAttack" not in spell_name:
            continue

        print(f"\n  {hero['name']}: '{spell_name}'")

        # Dump the spell struct looking for netIDs and pointers
        spell_data = m.read(active_ptr, 0x200)
        if not spell_data: continue

        # Look for all u32 values in the netID range
        print(f"    NetIDs in spell struct:")
        for off in range(0, len(spell_data) - 4, 4):
            val = struct.unpack("<I", spell_data[off:off+4])[0]
            if 0x40000000 < val < 0x50000000:
                # Look up which hero this is
                for h2 in heroes:
                    h2_netid = m.u32(h2['ptr'] + 0xCC)
                    if h2_netid == val:
                        print(f"      +0x{off:03X}: 0x{val:08X} = {h2['name']} {'(SELF)' if h2['name']==hero['name'] else '(TARGET?)'}")
                        break
                else:
                    print(f"      +0x{off:03X}: 0x{val:08X} (not a hero)")

        # Look for hero pointers in the spell struct
        print(f"    Hero pointers in spell struct:")
        for off in range(0, len(spell_data) - 8, 8):
            ptr = struct.unpack("<Q", spell_data[off:off+8])[0]
            if not is_heap(ptr): continue
            for h2 in heroes:
                if h2['ptr'] == ptr:
                    print(f"      +0x{off:03X}: 0x{ptr:X} = {h2['name']} {'(SELF)' if h2['name']==hero['name'] else '(TARGET?)'}")
                    break

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
