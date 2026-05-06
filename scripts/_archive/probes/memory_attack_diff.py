"""
Full byte-level diff of ActiveSpell struct during combat.
Captures BEFORE/AFTER when the spell name changes, dumps ALL changed fields.
This should reveal target NetID, caster NetID, positions, and timing.
"""

import ctypes
import ctypes.wintypes
import struct
import sys
import time
import urllib.request
import ssl
import json
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
ctx = ssl._create_unverified_context()

def api_post(path, data):
    try:
        req = urllib.request.Request(f"https://127.0.0.1:2999{path}",
                                     data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        return urllib.request.urlopen(req, context=ctx, timeout=3).read()
    except: return None

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

def get_spell_name(m, obj_ptr):
    if not is_heap(obj_ptr): return None
    si = m.u64(obj_ptr + 0x38)  # ActiveSpell pattern
    if is_heap(si):
        np = m.u64(si + 0x28)
        if is_heap(np):
            return m.string(np, 64)
    si2 = m.u64(obj_ptr)  # BA pattern
    if is_heap(si2):
        np2 = m.u64(si2 + 0x28)
        if is_heap(np2):
            return m.string(np2, 64)
    return None

def annotate_u32(val, netid_map):
    """Annotate a u32 value with what it might be."""
    # NetID?
    if 0x40000000 <= val <= 0x400001FF:
        name = netid_map.get(val, None)
        return f"NETID 0x{val:08X}" + (f" ({name})" if name else "")
    # Small int?
    if 0 < val < 100:
        return f"int {val}"
    # Boolean?
    if val == 0:
        return "zero"
    if val == 1:
        return "true/1"
    return None

def annotate_f32(val):
    """Annotate a float value."""
    if val != val:  # NaN
        return "NaN"
    if abs(val) < 0.001:
        return None
    if 1.0 < val < 5000.0:
        return f"timer? {val:.2f}"
    if -500 < val < 16000:
        return f"coord? {val:.1f}"
    return None

def full_diff(old_data, new_data, netid_map, m, label=""):
    """Do a full byte-level diff and annotate each changed field."""
    if not old_data or not new_data:
        return

    print(f"\n  [{label}] Changed fields:")
    for off in range(0, min(len(old_data), len(new_data)) - 4, 4):
        old_u32 = struct.unpack("<I", old_data[off:off+4])[0]
        new_u32 = struct.unpack("<I", new_data[off:off+4])[0]
        if old_u32 == new_u32:
            continue

        old_f32 = struct.unpack("<f", old_data[off:off+4])[0]
        new_f32 = struct.unpack("<f", new_data[off:off+4])[0]

        # Annotate new value
        ann_u32 = annotate_u32(new_u32, netid_map)
        ann_f32 = annotate_f32(new_f32)

        # Check if it's a pointer (8-byte aligned)
        ann_ptr = None
        if off % 8 == 0 and off + 8 <= len(new_data):
            new_u64 = struct.unpack("<Q", new_data[off:off+8])[0]
            if is_heap(new_u64):
                s = m.string(new_u64, 32)
                if s and len(s) > 2:
                    ann_ptr = f"ptr -> '{s[:40]}'"

        # Build annotation
        anns = []
        if ann_u32: anns.append(ann_u32)
        if ann_f32: anns.append(ann_f32)
        if ann_ptr: anns.append(ann_ptr)

        ann_str = f"  [{', '.join(anns)}]" if anns else ""
        print(f"    +0x{off:03X}: 0x{old_u32:08X} -> 0x{new_u32:08X}  "
              f"(f32: {old_f32:12.3f} -> {new_f32:12.3f}){ann_str}")


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
        net_id = m.u32(hp + 0xCC)
        heroes.append({"ptr": hp, "name": name, "net_id": net_id})
    print(f"Heroes: {[(h['name'], hex(h['net_id'] or 0)) for h in heroes]}")
    netid_map = {h['net_id']: h['name'] for h in heroes if h['net_id']}

    # Seek to mid-game fights
    print("\nSeeking to t=900 (15 min, mid-game teamfights)...")
    api_post("/replay/playback", {"time": 900.0})
    time.sleep(0.5)
    api_post("/replay/playback", {"paused": False, "speed": 1.0})
    time.sleep(1)

    READ_SIZE = 0x400  # Read 1024 bytes of each struct

    # Monitor 3 heroes on each team for diversity
    test_heroes = heroes[:3] + heroes[5:8]  # 3 blue + 3 red
    print(f"Monitoring: {[h['name'] for h in test_heroes]}")

    # Collect spell change events with full diffs
    print("\n=== Collecting spell change events (20s at 20Hz) ===")

    prev_snapshots = {}
    for h in test_heroes:
        as_ptr = m.u64(h["ptr"] + 0x4578)
        if is_heap(as_ptr):
            prev_snapshots[h['name']] = {
                'data': m.read(as_ptr, READ_SIZE),
                'spell': get_spell_name(m, as_ptr),
                'ptr': as_ptr,
            }
        ba_ptr = m.u64(h["ptr"] + 0x4010)
        if is_heap(ba_ptr):
            prev_snapshots[h['name'] + '_BA'] = {
                'data': m.read(ba_ptr, READ_SIZE),
                'spell': get_spell_name(m, ba_ptr),
                'ptr': ba_ptr,
            }

    diffs_collected = 0
    max_diffs = 8  # Collect up to 8 detailed diffs

    for tick in range(400):  # 20 seconds at 20Hz
        time.sleep(0.05)

        if diffs_collected >= max_diffs:
            break

        for h in test_heroes:
            # Check ActiveSpell
            as_ptr = m.u64(h["ptr"] + 0x4578)
            if not is_heap(as_ptr): continue

            cur_data = m.read(as_ptr, READ_SIZE)
            cur_spell = get_spell_name(m, as_ptr)
            key = h['name']

            if key in prev_snapshots and cur_data:
                prev = prev_snapshots[key]
                # Only report when spell NAME changes (new cast, not just timer updates)
                if cur_spell != prev['spell'] and cur_spell:
                    hero_pos = m.vec3(h["ptr"] + 0x25C)
                    pos_str = f"({hero_pos[0]:.0f},{hero_pos[2]:.0f})" if hero_pos else "?"

                    print(f"\n{'='*70}")
                    print(f"SPELL CHANGE t={tick}: {h['name']} at {pos_str}")
                    print(f"  Old spell: {prev['spell']}")
                    print(f"  New spell: {cur_spell}")

                    full_diff(prev['data'], cur_data, netid_map, m, "ActiveSpell")
                    diffs_collected += 1

                prev_snapshots[key] = {
                    'data': cur_data,
                    'spell': cur_spell,
                    'ptr': as_ptr,
                }

            # Check BA
            ba_ptr = m.u64(h["ptr"] + 0x4010)
            if not is_heap(ba_ptr): continue

            ba_data = m.read(ba_ptr, READ_SIZE)
            ba_spell = get_spell_name(m, ba_ptr)
            ba_key = h['name'] + '_BA'

            if ba_key in prev_snapshots and ba_data:
                prev_ba = prev_snapshots[ba_key]
                if ba_spell != prev_ba['spell'] and ba_spell:
                    print(f"\n{'='*70}")
                    print(f"BA CHANGE t={tick}: {h['name']}")
                    print(f"  Old: {prev_ba['spell']}")
                    print(f"  New: {ba_spell}")

                    full_diff(prev_ba['data'], ba_data, netid_map, m, "BasicAttackBase")
                    diffs_collected += 1

                prev_snapshots[ba_key] = {
                    'data': ba_data,
                    'spell': ba_spell,
                    'ptr': ba_ptr,
                }

        if tick % 100 == 0:
            print(f"\n  ... tick {tick}/400, diffs={diffs_collected}")

    # ================================================================
    # Also scan nearby hero offsets for SpellBook (wider range)
    # ================================================================
    print(f"\n\n{'='*70}")
    print("BONUS: SpellBook search hero+0x4000-0x4800 (wider)")
    hero = heroes[0]

    for off in range(0x4000, 0x4800, 8):
        ptr = m.u64(hero["ptr"] + off)
        if not is_heap(ptr): continue

        # Try to get a spell name from this object
        name = get_spell_name(m, ptr)
        if name and len(name) > 3:
            if "Attack" in name or "Spell" in name or hero['name'][:4] in name:
                print(f"  hero+0x{off:04X}: '{name}'")

        # Also try reading as a container with spell slot ptrs
        # Read first 8 pointers from the object
        slot_names = []
        for si in range(0, 0x40, 8):
            sp = m.u64(ptr + si)
            if is_heap(sp):
                sn = get_spell_name(m, sp)
                if sn and len(sn) > 3 and sn[0].isalpha():
                    slot_names.append((si, sn))

        if len(slot_names) >= 2:
            print(f"  hero+0x{off:04X} -> container with spell names:")
            for si, sn in slot_names[:6]:
                print(f"    +0x{si:02X}: '{sn}'")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
