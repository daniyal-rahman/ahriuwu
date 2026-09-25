"""
Find Garen's OUTGOING auto-attack field.

Strategy: Seek replay to laning phase where Garen is CSing minions.
Monitor a wide range of Garen's hero struct for fields that:
1. Change frequently (every auto-attack)
2. Contain "GarenBasicAttack" spell name
3. Are NOT hero+0x4578 (which is incoming)

Also check hero+0x4010 (BA) behavior during active combat.
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
    def vec3(s,a): d=s.read(a,12); return struct.unpack("<fff",d) if d else None
    def string(s,a,n=128):
        d=s.read(a,n)
        if not d: return None
        return d.split(b'\x00')[0].decode('ascii',errors='replace') or None
    def close(s): kernel32.CloseHandle(s.h)

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF

def get_spell_name_via_chains(m, ptr):
    """Try multiple pointer chains to find a spell name."""
    if not is_heap(ptr): return None
    results = []
    # Chain A: ptr -> +0x00 -> +0x28 -> name (BA pattern)
    a = m.u64(ptr)
    if is_heap(a):
        np = m.u64(a + 0x28)
        if is_heap(np):
            s = m.string(np, 64)
            if s and len(s) > 3 and s[0].isalpha():
                results.append(("A:+0x00->+0x28", s))
    # Chain B: ptr -> +0x38 -> +0x28 -> name (ActiveSpell pattern)
    b = m.u64(ptr + 0x38)
    if is_heap(b):
        np = m.u64(b + 0x28)
        if is_heap(np):
            s = m.string(np, 64)
            if s and len(s) > 3 and s[0].isalpha():
                results.append(("B:+0x38->+0x28", s))
    # Chain C: ptr -> +0x28 -> name (direct SpellInfo)
    c = m.u64(ptr + 0x28)
    if is_heap(c):
        s = m.string(c, 64)
        if s and len(s) > 3 and s[0].isalpha():
            results.append(("C:+0x28", s))
    # Chain D: ptr -> +0x08 -> +0x28 -> name
    d = m.u64(ptr + 0x08)
    if is_heap(d):
        np = m.u64(d + 0x28)
        if is_heap(np):
            s = m.string(np, 64)
            if s and len(s) > 3 and s[0].isalpha():
                results.append(("D:+0x08->+0x28", s))
    # Chain E: ptr -> +0x10 -> +0x28 -> name
    e = m.u64(ptr + 0x10)
    if is_heap(e):
        np = m.u64(e + 0x28)
        if is_heap(np):
            s = m.string(np, 64)
            if s and len(s) > 3 and s[0].isalpha():
                results.append(("E:+0x10->+0x28", s))
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
        net_id = m.u32(hp + 0xCC)
        heroes.append({"ptr": hp, "name": name, "net_id": net_id})

    # Find Garen
    garen = None
    for h in heroes:
        if "Garen" in h['name']:
            garen = h
            break
    if not garen:
        print("Garen not found!"); return
    print(f"Garen: ptr=0x{garen['ptr']:X}, NetID=0x{garen['net_id']:08X}")

    # Seek to laning phase where Garen is CSing (around 3-5 min)
    print("\nSeeking to t=240 (4 min, Garen should be CSing)...")
    api_post("/replay/playback", {"time": 240.0})
    time.sleep(0.5)
    api_post("/replay/playback", {"paused": False, "speed": 1.0})
    time.sleep(1.5)

    pb = api_get("/replay/playback")
    if pb:
        print(f"  Game time: {pb.get('time', '?'):.1f}s")

    garen_pos = m.vec3(garen["ptr"] + 0x25C)
    print(f"  Garen position: ({garen_pos[0]:.0f}, {garen_pos[2]:.0f})" if garen_pos else "  Garen pos: ?")

    # ================================================================
    # PHASE 1: Snapshot Garen's hero struct 0x3800-0x4800 to find all
    # heap pointers that could be spell-related
    # ================================================================
    print("\n=== Phase 1: Map all heap pointers in Garen struct 0x3800-0x4800 ===")
    ptr_map = {}
    for off in range(0x3800, 0x4800, 8):
        val = m.u64(garen["ptr"] + off)
        if is_heap(val):
            names = get_spell_name_via_chains(m, val)
            ptr_map[off] = {"ptr": val, "names": names}
            if names:
                name_strs = [f"{chain}='{name}'" for chain, name in names]
                print(f"  +0x{off:04X}: {', '.join(name_strs)}")

    # ================================================================
    # PHASE 2: Monitor ALL pointer fields for 15s, detect which change
    # ================================================================
    print(f"\n=== Phase 2: Monitor changing pointers over 15s (Garen should be auto-attacking) ===")

    # Take baseline of all pointer values
    baseline = {}
    for off in range(0x3800, 0x4800, 8):
        val = m.u64(garen["ptr"] + off)
        if is_heap(val):
            baseline[off] = val

    change_log = {}  # off -> list of (tick, old_ptr, new_ptr, old_names, new_names)

    for tick in range(150):  # 15s at 10Hz
        time.sleep(0.1)
        for off in list(baseline.keys()):
            cur = m.u64(garen["ptr"] + off)
            if cur != baseline[off]:
                if off not in change_log:
                    change_log[off] = []
                # Only resolve names on first few changes to save time
                new_names = get_spell_name_via_chains(m, cur) if len(change_log[off]) < 5 else []
                change_log[off].append((tick, baseline[off], cur, new_names))
                baseline[off] = cur

        # Also check if there are NEW pointers
        if tick % 50 == 49:
            for off in range(0x3800, 0x4800, 8):
                val = m.u64(garen["ptr"] + off)
                if is_heap(val) and off not in baseline:
                    baseline[off] = val

    # Report
    print(f"\n  {len(change_log)} offsets changed during monitoring:")
    for off in sorted(change_log.keys()):
        changes = change_log[off]
        n = len(changes)
        # Get unique spell names seen
        all_names = set()
        for _, _, _, names in changes:
            for chain, name in names:
                all_names.add(name)

        garen_related = any("Garen" in name for name in all_names)
        attack_related = any("Attack" in name or "Basic" in name for name in all_names)

        marker = ""
        if garen_related: marker += " *** GAREN ***"
        if attack_related: marker += " *** ATTACK ***"
        if off == 0x4578: marker += " (KNOWN: incoming ActiveSpell)"
        if off == 0x4010: marker += " (KNOWN: BasicAttackBase)"

        print(f"\n  +0x{off:04X}: {n} changes{marker}")
        if all_names:
            print(f"    Spell names: {all_names}")
        # Show first few changes with details
        for tick, old, new, names in changes[:3]:
            name_str = ", ".join(f"{c}='{n}'" for c, n in names) if names else ""
            print(f"    t={tick:3d}: 0x{old:X} -> 0x{new:X}  {name_str}")

    # ================================================================
    # PHASE 3: Also check large struct regions for byte-level changes
    # during auto-attack (the attack might update in-place, not change ptr)
    # ================================================================
    print(f"\n\n=== Phase 3: Byte-level monitoring of BA struct during auto-attack ===")

    ba_ptr = m.u64(garen["ptr"] + 0x4010)
    if is_heap(ba_ptr):
        print(f"  Garen BA at 0x{ba_ptr:X}")
        ba_names = get_spell_name_via_chains(m, ba_ptr)
        print(f"  BA spell: {ba_names}")

        # Monitor BA struct for 10s
        prev_ba = m.read(ba_ptr, 0x400)
        ba_changes = []

        for tick in range(100):
            time.sleep(0.1)
            cur_ba = m.read(ba_ptr, 0x400)
            if cur_ba and prev_ba and cur_ba != prev_ba:
                changed_offsets = []
                for i in range(0, min(len(cur_ba), len(prev_ba)) - 4, 4):
                    old_v = struct.unpack("<I", prev_ba[i:i+4])[0]
                    new_v = struct.unpack("<I", cur_ba[i:i+4])[0]
                    if old_v != new_v:
                        new_f = struct.unpack("<f", cur_ba[i:i+4])[0]
                        changed_offsets.append((i, old_v, new_v, new_f))
                if changed_offsets:
                    ba_changes.append((tick, changed_offsets))
                prev_ba = cur_ba

        print(f"  BA struct: {len(ba_changes)} change events in 10s")
        for tick, offsets in ba_changes[:10]:
            print(f"    t={tick}:")
            for off, old, new, new_f in offsets[:8]:
                ann = ""
                if 0x40000000 <= new <= 0x400001FF: ann = " NETID!"
                elif 10 < new_f < 5000 and new < 0x10000000: ann = f" timer={new_f:.2f}"
                elif -500 < new_f < 16000 and abs(new_f) > 50: ann = f" coord={new_f:.1f}"
                print(f"      BA+0x{off:03X}: 0x{old:08X} -> 0x{new:08X}{ann}")

    # ================================================================
    # PHASE 4: Check AiManager for attack target info
    # ================================================================
    print(f"\n\n=== Phase 4: AiManager attack state ===")
    ai_raw = m.u64(garen["ptr"] + 0x4628)
    if is_heap(ai_raw):
        inner = m.u64(ai_raw + 0x10)
        if is_heap(inner):
            print(f"  AiManager inner at 0x{inner:X}")
            # Monitor for 5s
            prev_ai = m.read(inner, 0x500)
            ai_changes = []
            for tick in range(50):
                time.sleep(0.1)
                cur_ai = m.read(inner, 0x500)
                if cur_ai and prev_ai and cur_ai != prev_ai:
                    changed = []
                    for i in range(0, min(len(cur_ai), len(prev_ai)) - 4, 4):
                        old_v = struct.unpack("<I", prev_ai[i:i+4])[0]
                        new_v = struct.unpack("<I", cur_ai[i:i+4])[0]
                        if old_v != new_v:
                            new_f = struct.unpack("<f", cur_ai[i:i+4])[0]
                            changed.append((i, old_v, new_v, new_f))
                    if changed:
                        ai_changes.append((tick, changed))
                    prev_ai = cur_ai

            print(f"  AiManager: {len(ai_changes)} change events in 5s")
            for tick, offsets in ai_changes[:5]:
                print(f"    t={tick}:")
                for off, old, new, new_f in offsets[:6]:
                    ann = ""
                    if 0x40000000 <= new <= 0x400001FF: ann = " NETID!"
                    elif 10 < new_f < 5000 and new < 0x10000000: ann = f" timer={new_f:.2f}"
                    print(f"      AI+0x{off:03X}: 0x{old:08X} -> 0x{new:08X}{ann}")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
