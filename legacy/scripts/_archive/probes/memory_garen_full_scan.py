"""
Full hero struct scan during combat.
Take 2 snapshots of Garen's ENTIRE hero struct (0x0000-0x8000) during
active combat, find ALL changing fields. This should reveal:
1. Runtime SpellBook location
2. Outgoing attack state
3. Cooldown timers
4. Target NetID for attacks
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
    def u64(s,a): d=s.read(a,8); return struct.unpack("<Q",d)[0] if d else None
    def u32(s,a): d=s.read(a,4); return struct.unpack("<I",d)[0] if d else None
    def string(s,a,n=64):
        d=s.read(a,n)
        if not d: return None
        return d.split(b'\x00')[0].decode('ascii',errors='replace') or None
    def close(s): kernel32.CloseHandle(s.h)

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF

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

    netid_map = {h['net_id']: h['name'] for h in heroes if h['net_id']}
    garen = next((h for h in heroes if "Garen" in h['name']), None)
    if not garen:
        print("Garen not found!"); return
    print(f"Garen at 0x{garen['ptr']:X}, NetID=0x{garen['net_id']:08X}")

    # Seek to active combat
    print("\nSeeking to t=600 (10 min, lane fights)...")
    api_post("/replay/playback", {"time": 600.0})
    time.sleep(0.5)
    api_post("/replay/playback", {"paused": False, "speed": 2.0})
    time.sleep(2)

    # ================================================================
    # Take 5 snapshots of entire hero struct, 1s apart
    # ================================================================
    STRUCT_SIZE = 0x8000
    print(f"\nTaking 5 snapshots of hero struct (0x{STRUCT_SIZE:X} bytes each)...")

    snapshots = []
    for i in range(5):
        data = m.read(garen["ptr"], STRUCT_SIZE)
        snapshots.append(data)
        if i < 4: time.sleep(1.0)
        print(f"  Snapshot {i}: {len(data) if data else 0} bytes")

    # ================================================================
    # Find all changing 4-byte fields between snapshots
    # ================================================================
    print(f"\n=== Changing fields between snapshot 0 and snapshot 4 ===")

    if snapshots[0] and snapshots[4]:
        s0 = snapshots[0]
        s4 = snapshots[4]

        # Group changes by hero struct region
        regions = {
            "0x0000-0x0100 (vtable/base)": (0x0000, 0x0100),
            "0x0100-0x0300 (position/netid)": (0x0100, 0x0300),
            "0x0300-0x0800 (stats?)": (0x0300, 0x0800),
            "0x0800-0x1200 (stats/health?)": (0x0800, 0x1200),
            "0x1200-0x2000 (buffs?)": (0x1200, 0x2000),
            "0x2000-0x2800 (stat descriptors)": (0x2000, 0x2800),
            "0x2800-0x3000": (0x2800, 0x3000),
            "0x3000-0x3800": (0x3000, 0x3800),
            "0x3800-0x4000 (spell slots?)": (0x3800, 0x4000),
            "0x4000-0x4200 (BA/spellbook)": (0x4000, 0x4200),
            "0x4200-0x4600 (active spell?)": (0x4200, 0x4600),
            "0x4600-0x4800 (AI manager)": (0x4600, 0x4800),
            "0x4800-0x5000": (0x4800, 0x5000),
            "0x5000-0x6000": (0x5000, 0x6000),
            "0x6000-0x7000": (0x6000, 0x7000),
            "0x7000-0x8000": (0x7000, 0x8000),
        }

        for region_name, (start, end) in regions.items():
            changes = []
            for off in range(start, min(end, len(s0), len(s4)) - 4, 4):
                v0 = struct.unpack("<I", s0[off:off+4])[0]
                v4 = struct.unpack("<I", s4[off:off+4])[0]
                if v0 != v4:
                    f0 = struct.unpack("<f", s0[off:off+4])[0]
                    f4 = struct.unpack("<f", s4[off:off+4])[0]
                    changes.append((off, v0, v4, f0, f4))

            if changes:
                print(f"\n  {region_name}: {len(changes)} changed fields")
                for off, v0, v4, f0, f4 in changes[:10]:
                    ann = ""
                    if 0x40000000 <= v4 <= 0x400001FF:
                        name = netid_map.get(v4, "?")
                        ann = f" NETID ({name})"
                    elif 10 < f4 < 5000 and v4 < 0x10000000:
                        ann = f" timer={f4:.1f}"
                    elif -500 < f4 < 16000 and abs(f4) > 1:
                        ann = f" coord={f4:.1f}"
                    print(f"    +0x{off:04X}: 0x{v0:08X} -> 0x{v4:08X} (f: {f0:.2f} -> {f4:.2f}){ann}")
                if len(changes) > 10:
                    print(f"    ... and {len(changes)-10} more")
            else:
                print(f"\n  {region_name}: no changes")

    # ================================================================
    # For the most interesting regions, do HIGH-FREQUENCY monitoring
    # ================================================================
    print(f"\n\n=== High-freq monitoring of changing regions (5s at 20Hz) ===")

    # Identify regions with changes
    interesting = []
    if snapshots[0] and snapshots[4]:
        for off in range(0, min(len(snapshots[0]), len(snapshots[4])) - 4, 4):
            v0 = struct.unpack("<I", snapshots[0][off:off+4])[0]
            v4 = struct.unpack("<I", snapshots[4][off:off+4])[0]
            if v0 != v4 and off not in [0x25C, 0x260, 0x264]:  # Skip known position
                interesting.append(off)

    # Group into contiguous regions
    if interesting:
        change_regions = []
        start = interesting[0]
        end = interesting[0]
        for off in interesting[1:]:
            if off - end <= 8:
                end = off
            else:
                change_regions.append((start, end + 4))
                start = off
                end = off
        change_regions.append((start, end + 4))

        print(f"  {len(change_regions)} contiguous changing regions:")
        for start, end in change_regions[:15]:
            size = end - start
            print(f"    +0x{start:04X} - +0x{end:04X} ({size} bytes)")

    # ================================================================
    # Now check: does hero+0x4578 ActiveSpell ptr itself change?
    # Or just the data behind it?
    # ================================================================
    print(f"\n\n=== ActiveSpell pointer stability check ===")
    ptrs = []
    for i in range(10):
        p = m.u64(garen["ptr"] + 0x4578)
        ptrs.append(p)
        time.sleep(0.2)

    unique = set(ptrs)
    if len(unique) == 1:
        print(f"  hero+0x4578 pointer STABLE: 0x{ptrs[0]:X}")
        print(f"  (The pointer stays the same, only the data behind it changes)")
    else:
        print(f"  hero+0x4578 pointer CHANGES: {[hex(p) for p in unique]}")

    # Same for BA
    ba_ptrs = []
    for i in range(10):
        p = m.u64(garen["ptr"] + 0x4010)
        ba_ptrs.append(p)
        time.sleep(0.2)

    unique_ba = set(ba_ptrs)
    if len(unique_ba) == 1:
        print(f"  hero+0x4010 BA pointer STABLE: 0x{ba_ptrs[0]:X}")
    else:
        print(f"  hero+0x4010 BA pointer CHANGES: {[hex(p) for p in unique_ba]}")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
