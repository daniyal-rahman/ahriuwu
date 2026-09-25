"""
Map the SpellBook structure at hero+0x4028.
Find spell slots, spell names, cooldowns, casting state.
Then monitor for auto-attack casting during replay playback.
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
                                     headers={"Content-Type": "application/json"}, method="POST")
        return urllib.request.urlopen(req, context=ctx, timeout=3).read()
    except: return None

def api_get(path):
    try:
        return json.loads(urllib.request.urlopen(f"https://127.0.0.1:2999{path}", context=ctx, timeout=3).read())
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
    def string(s,a,n=64):
        d=s.read(a,n)
        if not d: return None
        return d.split(b'\x00')[0].decode('ascii',errors='replace') or None
    def close(s): kernel32.CloseHandle(s.h)

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF

def find_inline_string(data, offset, max_len=16):
    """Read an inline SSO string from data at offset."""
    if offset + max_len > len(data): return None
    chunk = data[offset:offset+max_len]
    # Find null terminator
    null_idx = chunk.find(b'\x00')
    if null_idx <= 0: return None
    s = chunk[:null_idx]
    try:
        result = s.decode('ascii')
        if all(32 <= ord(c) < 127 for c in result):
            return result
    except: pass
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
        net_id = m.u32(hp + 0xCC)
        heroes.append({"ptr": hp, "name": name, "net_id": net_id})

    garen = next((h for h in heroes if "Garen" in h['name']), None)
    if not garen:
        print("Garen not found!"); return
    print(f"Garen at 0x{garen['ptr']:X}")

    # ================================================================
    # Step 1: Map the SpellBook object at hero+0x4028
    # ================================================================
    print("\n=== Step 1: SpellBook structure at hero+0x4028 ===")

    sb_ptr = m.u64(garen["ptr"] + 0x4028)
    print(f"  hero+0x4028 -> SpellBook at 0x{sb_ptr:X}" if is_heap(sb_ptr) else "  NULL!")

    if not is_heap(sb_ptr):
        print("  Trying other offsets...")
        for off in [0x4028, 0x4030, 0x4090]:
            ptr = m.u64(garen["ptr"] + off)
            if is_heap(ptr):
                sb_ptr = ptr
                print(f"  hero+0x{off:04X} -> 0x{sb_ptr:X}")
                break

    if not is_heap(sb_ptr):
        print("  SpellBook not found!"); return

    # Read the SpellBook object (large)
    sb_data = m.read(sb_ptr, 0x1200)
    if not sb_data:
        print("  Failed to read SpellBook"); return

    # Find all heap pointer arrays within SpellBook
    print(f"\n  SpellBook heap pointer regions:")
    for off in range(0, len(sb_data) - 64, 8):
        # Count consecutive heap pointers
        count = 0
        for i in range(0, 128, 8):
            if off + i + 8 > len(sb_data): break
            val = struct.unpack("<Q", sb_data[off+i:off+i+8])[0]
            if is_heap(val):
                count += 1
            else:
                break
        if count >= 4:
            print(f"    SB+0x{off:04X}: {count} consecutive heap pointers")

            # Try reading first few as spell slots
            for i in range(min(count, 8)):
                slot_ptr = struct.unpack("<Q", sb_data[off+i*8:off+i*8+8])[0]
                # Read slot object
                slot_data = m.read(slot_ptr, 0x200)
                if not slot_data: continue

                # Search for inline spell names
                found_names = []
                for search_off in range(0, 0x1F0, 1):
                    s = find_inline_string(slot_data, search_off)
                    if s and len(s) >= 4 and s[0].isupper():
                        # Filter: only keep if it looks like a spell name
                        if any(kw in s for kw in ["Garen", "Summoner", "Passive",
                                                    "Attack", "Recall", "Item"]):
                            found_names.append((search_off, s))

                # Also follow ptr chains to find spell names
                for sub_off in [0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x40]:
                    if sub_off + 8 > len(slot_data): break
                    sub_ptr = struct.unpack("<Q", slot_data[sub_off:sub_off+8])[0]
                    if is_heap(sub_ptr):
                        sub_data = m.read(sub_ptr, 0x200)
                        if sub_data:
                            for search_off in range(0, 0x1F0, 1):
                                s = find_inline_string(sub_data, search_off)
                                if s and len(s) >= 4:
                                    if any(kw in s for kw in ["Garen", "Summoner",
                                                               "Attack", "Recall"]):
                                        found_names.append((sub_off * 0x1000 + search_off, f"+0x{sub_off:02X}->{s}"))

                if found_names:
                    print(f"      slot[{i}] (0x{slot_ptr:X}):")
                    for name_off, name in found_names[:5]:
                        print(f"        @0x{name_off:04X}: '{name}'")

            # Skip ahead to avoid duplicate reporting
            off += count * 8

    # ================================================================
    # Step 2: Specifically explore the chain we found
    # hero+0x4028 -> +0x28 -> +0x40 -> SpellData
    # ================================================================
    print(f"\n\n=== Step 2: Explore spell data chain ===")

    ptr1 = m.u64(sb_ptr + 0x28)
    print(f"  SB+0x28 -> 0x{ptr1:X}" if is_heap(ptr1) else "  SB+0x28 -> NULL")

    if is_heap(ptr1):
        ptr2 = m.u64(ptr1 + 0x40)
        print(f"  -> +0x40 -> SpellData at 0x{ptr2:X}" if is_heap(ptr2) else "  +0x40 -> NULL")

        if is_heap(ptr2):
            sd_data = m.read(ptr2, 0x300)
            if sd_data:
                print(f"\n  SpellData inline strings:")
                for off in range(0, 0x2F0, 1):
                    s = find_inline_string(sd_data, off)
                    if s and len(s) >= 3 and s[0].isalpha():
                        # Only print unique spell-like names
                        if any(c.isupper() for c in s[:3]):
                            print(f"    SD+0x{off:03X}: '{s}'")

    # ================================================================
    # Step 3: Look at hero+0x4168-0x4188 as spell slot array
    # ================================================================
    print(f"\n\n=== Step 3: Spell slot array at hero+0x4168-0x4188? ===")

    for hero_off in range(0x4148, 0x41C0, 8):
        ptr = m.u64(garen["ptr"] + hero_off)
        if not is_heap(ptr): continue

        # Read this potential spell slot
        slot_data = m.read(ptr, 0x100)
        if not slot_data: continue

        # Check +0x28 -> +0x40 chain for spell name
        p1 = m.u64(ptr + 0x28)
        spell_name = None
        if is_heap(p1):
            p2 = m.u64(p1 + 0x40)
            if is_heap(p2):
                sd = m.read(p2, 0x140)
                if sd:
                    for name_off in [0x050, 0x098, 0x0E0, 0x120, 0x008, 0x030, 0x060]:
                        s = find_inline_string(sd, name_off)
                        if s and len(s) >= 4 and s[0].isalpha():
                            spell_name = f"SD+0x{name_off:03X}='{s}'"
                            break

        # Also check for cooldown/timer floats
        timers = []
        for i in range(0, 0x80, 4):
            f_val = struct.unpack("<f", slot_data[i:i+4])[0]
            if 10 < f_val < 5000:
                i_val = struct.unpack("<I", slot_data[i:i+4])[0]
                if i_val < 0x10000000:
                    timers.append(f"+0x{i:02X}={f_val:.1f}")

        # Check for spell level (small int)
        levels = []
        for i in range(0, 0x80, 4):
            i_val = struct.unpack("<I", slot_data[i:i+4])[0]
            if 1 <= i_val <= 5:
                levels.append(f"+0x{i:02X}={i_val}")

        print(f"  hero+0x{hero_off:04X}: {spell_name or 'no_name'}")
        if timers: print(f"    timers: {', '.join(timers[:5])}")
        if levels: print(f"    levels: {', '.join(levels[:5])}")

    # ================================================================
    # Step 4: Cross-validate — read same offsets on Lee Sin
    # ================================================================
    print(f"\n\n=== Step 4: Cross-validate on Lee Sin ===")
    lee = next((h for h in heroes if "Lee" in h['name']), None)
    if lee:
        for hero_off in range(0x4148, 0x41C0, 8):
            ptr = m.u64(lee["ptr"] + hero_off)
            if not is_heap(ptr): continue
            p1 = m.u64(ptr + 0x28)
            if not is_heap(p1): continue
            p2 = m.u64(p1 + 0x40)
            if not is_heap(p2): continue
            sd = m.read(p2, 0x140)
            if not sd: continue
            names = []
            for name_off in [0x050, 0x098, 0x0E0, 0x120, 0x008, 0x030, 0x060]:
                s = find_inline_string(sd, name_off)
                if s and len(s) >= 4 and s[0].isalpha():
                    names.append(f"+0x{name_off:03X}='{s}'")
            if names:
                print(f"  hero+0x{hero_off:04X}: {', '.join(names[:4])}")

    # ================================================================
    # Step 5: Monitor SpellBook for casting state changes during combat
    # ================================================================
    print(f"\n\n=== Step 5: Monitor spell casting (15s) ===")
    print("Seeking to t=600 for active combat...")
    api_post("/replay/playback", {"time": 600.0})
    time.sleep(0.5)
    api_post("/replay/playback", {"paused": False, "speed": 1.0})
    time.sleep(1)

    # Read SpellBook baseline
    sb_ptr = m.u64(garen["ptr"] + 0x4028)
    if not is_heap(sb_ptr):
        print("  SpellBook NULL!"); m.close(); return

    prev_sb = m.read(sb_ptr, 0x200)

    # Also read each potential spell slot
    slot_ptrs = []
    for hero_off in range(0x4148, 0x41C0, 8):
        ptr = m.u64(garen["ptr"] + hero_off)
        if is_heap(ptr):
            slot_ptrs.append((hero_off, ptr))

    prev_slots = {}
    for hero_off, ptr in slot_ptrs:
        prev_slots[hero_off] = m.read(ptr, 0x100)

    events = []
    for tick in range(150):  # 15s at 10Hz
        time.sleep(0.1)

        # Check SpellBook changes
        cur_sb = m.read(sb_ptr, 0x200)
        if cur_sb and prev_sb and cur_sb != prev_sb:
            changed = []
            for i in range(0, min(len(cur_sb), len(prev_sb)) - 4, 4):
                old = struct.unpack("<I", prev_sb[i:i+4])[0]
                new = struct.unpack("<I", cur_sb[i:i+4])[0]
                if old != new:
                    new_f = struct.unpack("<f", cur_sb[i:i+4])[0]
                    changed.append((i, old, new, new_f))
            if changed:
                events.append(("SB", tick, changed))
            prev_sb = cur_sb

        # Check spell slot changes
        for hero_off, ptr in slot_ptrs:
            cur = m.read(ptr, 0x100)
            key = hero_off
            if cur and key in prev_slots and prev_slots[key] and cur != prev_slots[key]:
                changed = []
                for i in range(0, min(len(cur), len(prev_slots[key])) - 4, 4):
                    old = struct.unpack("<I", prev_slots[key][i:i+4])[0]
                    new = struct.unpack("<I", cur[i:i+4])[0]
                    if old != new:
                        new_f = struct.unpack("<f", cur[i:i+4])[0]
                        changed.append((i, old, new, new_f))
                if changed:
                    events.append((f"SLOT+0x{hero_off:04X}", tick, changed))
                prev_slots[key] = cur

        if tick % 50 == 0:
            print(f"  tick {tick}/150, events={len(events)}")

    print(f"\n  Total events: {len(events)}")
    for src, tick, changed in events[:20]:
        print(f"\n  [{src}] t={tick}:")
        for off, old, new, new_f in changed[:6]:
            ann = ""
            if 0x40000000 <= new <= 0x400001FF: ann = " NETID!"
            elif 10 < new_f < 5000 and new < 0x10000000: ann = f" timer={new_f:.1f}"
            print(f"    +0x{off:03X}: 0x{old:08X} -> 0x{new:08X}{ann}")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
