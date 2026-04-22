"""
Live auto-attack monitoring.

1. Seek replay to a fight scene
2. Rapidly poll BasicAttackBase (hero+0x4010) and ActiveSpell (hero+0x4578)
3. Detect when auto-attacks fire by watching for changes
4. Extract: target NetID, cast position, timing
5. Also read BA+0x2C0 (hacker_logs oBasicAttackOffset1) and +0x70 (offset2)
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

def api_get(path):
    try:
        r = urllib.request.urlopen(f"https://127.0.0.1:2999{path}", context=ctx, timeout=3)
        return json.loads(r.read())
    except: return None

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
    """Get spell name from a spell-like object pointer."""
    if not is_heap(obj_ptr): return None
    # Chain: obj -> +0x00 -> SpellInfo -> +0x28 -> name string
    si = m.u64(obj_ptr)
    if is_heap(si):
        np = m.u64(si + 0x28)
        if is_heap(np):
            return m.string(np, 64)
    # Also try: obj -> +0x38 -> +0x28 -> name (ActiveSpell pattern)
    si2 = m.u64(obj_ptr + 0x38)
    if is_heap(si2):
        np2 = m.u64(si2 + 0x28)
        if is_heap(np2):
            return m.string(np2, 64)
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
    print(f"Heroes: {[(h['name'], hex(h['net_id'] or 0)) for h in heroes]}")

    netid_map = {h['net_id']: h['name'] for h in heroes if h['net_id']}

    # ================================================================
    # Step 1: Seek replay to a fight (around 600s = 10 min, laning phase fights)
    # ================================================================
    print("\nSeeking replay to t=600 (10 min)...")
    api_post("/replay/playback", {"time": 600.0})
    time.sleep(0.5)
    api_post("/replay/playback", {"paused": False, "speed": 1.0})
    time.sleep(1)

    pb = api_get("/replay/playback")
    if pb:
        print(f"  Game time: {pb.get('time', '?'):.1f}s, speed={pb.get('speed', '?')}")

    # ================================================================
    # Step 2: Read extended BA struct (0x400 bytes) to check offset 0x2C0
    # ================================================================
    print("\n=== Extended BA struct analysis ===")
    hero = heroes[0]
    ba_ptr = m.u64(hero["ptr"] + 0x4010)
    if is_heap(ba_ptr):
        ba_name = get_spell_name(m, ba_ptr)
        print(f"  {hero['name']} BA at 0x{ba_ptr:X}: '{ba_name}'")

        # Read extended
        ba_data = m.read(ba_ptr, 0x400)
        if ba_data:
            # Check hacker_logs offset 0x2C0 (oBasicAttackOffset1)
            print(f"\n  BA+0x2C0 (oBasicAttackOffset1):")
            val_2c0 = struct.unpack("<Q", ba_data[0x2C0:0x2C8])[0]
            print(f"    raw: 0x{val_2c0:016X}")
            if is_heap(val_2c0):
                s = get_spell_name(m, val_2c0)
                print(f"    -> spell name: '{s}'")
                # Check target NetID at offset2 (0x70 from this)
                tgt = m.u32(val_2c0 + 0x70)
                if tgt and 0x40000000 <= tgt <= 0x400001FF:
                    tgt_name = netid_map.get(tgt, "?")
                    print(f"    -> +0x70 target NetID: 0x{tgt:08X} ({tgt_name})")

            # Scan BA struct for ALL NetID-like values
            print(f"\n  BA struct NetID scan (0x400 bytes):")
            for off in range(0, len(ba_data) - 4, 4):
                val = struct.unpack("<I", ba_data[off:off+4])[0]
                if 0x40000000 <= val <= 0x400001FF:
                    tgt_name = netid_map.get(val, f"0x{val:08X}")
                    print(f"    BA+0x{off:03X}: 0x{val:08X} ({tgt_name})")

            # Scan for map positions in extended range
            print(f"\n  BA struct map positions (0xC0-0x180):")
            for off in range(0xC0, min(0x180, len(ba_data) - 12), 4):
                v = struct.unpack("<fff", ba_data[off:off+12])
                if is_map(v) and abs(v[0]) > 10:  # skip near-zero
                    print(f"    BA+0x{off:03X}: ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f})")

            # Scan for float timers
            print(f"\n  BA struct timer-like floats (1.0 < f < 5000.0):")
            for off in range(0, len(ba_data) - 4, 4):
                val_f = struct.unpack("<f", ba_data[off:off+4])[0]
                val_i = struct.unpack("<I", ba_data[off:off+4])[0]
                if 1.0 < val_f < 5000.0 and val_i < 0x10000000:
                    print(f"    BA+0x{off:03X}: {val_f:.3f}")

    # ================================================================
    # Step 3: Read extended ActiveSpell struct
    # ================================================================
    print("\n=== Extended ActiveSpell struct analysis ===")
    for h in heroes[:5]:
        as_ptr = m.u64(h["ptr"] + 0x4578)
        if not is_heap(as_ptr): continue

        as_name = get_spell_name(m, as_ptr)
        as_data = m.read(as_ptr, 0x200)
        if not as_data: continue

        # Find NetIDs
        netids = []
        for off in range(0, len(as_data) - 4, 4):
            val = struct.unpack("<I", as_data[off:off+4])[0]
            if 0x40000000 <= val <= 0x400001FF:
                tgt_name = netid_map.get(val, f"0x{val:08X}")
                netids.append(f"+0x{off:03X}={tgt_name}")

        # Find map positions
        positions = []
        for off in range(0, len(as_data) - 12, 4):
            v = struct.unpack("<fff", as_data[off:off+12])
            if is_map(v) and abs(v[0]) > 50:
                positions.append(f"+0x{off:03X}=({v[0]:.0f},{v[2]:.0f})")

        # Find timer floats
        timers = []
        for off in range(0, len(as_data) - 4, 4):
            val_f = struct.unpack("<f", as_data[off:off+4])[0]
            val_i = struct.unpack("<I", as_data[off:off+4])[0]
            if 10.0 < val_f < 5000.0 and val_i < 0x10000000:
                timers.append(f"+0x{off:03X}={val_f:.1f}")

        print(f"\n  {h['name']:15s} spell='{as_name}'")
        if netids: print(f"    NetIDs: {', '.join(netids[:5])}")
        if positions: print(f"    Positions: {', '.join(positions[:5])}")
        if timers: print(f"    Timers: {', '.join(timers[:5])}")

    # ================================================================
    # Step 4: Rapid polling — detect auto-attack events
    # ================================================================
    print("\n\n=== Rapid polling for auto-attack events (10s at 10Hz) ===")
    print("Monitoring BA+0x4010 and ActiveSpell+0x4578 for all heroes...")

    # Take baseline snapshots
    prev_ba = {}
    prev_as = {}
    for h in heroes:
        ba = m.u64(h["ptr"] + 0x4010)
        if is_heap(ba): prev_ba[h['name']] = m.read(ba, 0x200)
        asp = m.u64(h["ptr"] + 0x4578)
        if is_heap(asp): prev_as[h['name']] = m.read(asp, 0x200)

    events = []
    for tick in range(100):  # 10 seconds at 10Hz
        time.sleep(0.1)

        for h in heroes:
            # Check BA changes
            ba = m.u64(h["ptr"] + 0x4010)
            if is_heap(ba):
                cur = m.read(ba, 0x200)
                if cur and h['name'] in prev_ba:
                    old = prev_ba[h['name']]
                    if old and cur != old:
                        # Count changed bytes
                        n_changed = sum(1 for a, b in zip(old, cur) if a != b)
                        if n_changed > 4:  # Significant change
                            ba_name = get_spell_name(m, ba)
                            events.append({
                                "tick": tick, "hero": h['name'], "type": "BA",
                                "changed_bytes": n_changed, "spell": ba_name
                            })
                    prev_ba[h['name']] = cur

            # Check ActiveSpell changes
            asp = m.u64(h["ptr"] + 0x4578)
            if is_heap(asp):
                cur_as = m.read(asp, 0x200)
                if cur_as and h['name'] in prev_as:
                    old_as = prev_as[h['name']]
                    if old_as and cur_as != old_as:
                        n_changed = sum(1 for a, b in zip(old_as, cur_as) if a != b)
                        if n_changed > 4:
                            as_name = get_spell_name(m, asp)

                            # Find new NetIDs
                            new_netids = []
                            for off in range(0, min(len(cur_as), 0x180) - 4, 4):
                                new_val = struct.unpack("<I", cur_as[off:off+4])[0]
                                old_val = struct.unpack("<I", old_as[off:off+4])[0]
                                if new_val != old_val and 0x40000000 <= new_val <= 0x400001FF:
                                    tgt = netid_map.get(new_val, hex(new_val))
                                    new_netids.append(f"+0x{off:03X}={tgt}")

                            # Find new positions
                            new_pos = []
                            for off in range(0, min(len(cur_as), 0x180) - 12, 4):
                                v_new = struct.unpack("<fff", cur_as[off:off+12])
                                v_old = struct.unpack("<fff", old_as[off:off+12])
                                if v_new != v_old and is_map(v_new) and abs(v_new[0]) > 50:
                                    new_pos.append(f"+0x{off:03X}=({v_new[0]:.0f},{v_new[2]:.0f})")

                            events.append({
                                "tick": tick, "hero": h['name'], "type": "AS",
                                "changed_bytes": n_changed, "spell": as_name,
                                "netids": new_netids[:3], "positions": new_pos[:3]
                            })
                    prev_as[h['name']] = cur_as

        if tick % 20 == 0:
            print(f"  tick {tick}/100, events so far: {len(events)}")

    # Print events
    print(f"\n  Total events detected: {len(events)}")
    for e in events[:40]:
        parts = [f"t={e['tick']:3d}", f"{e['hero']:12s}", f"{e['type']}", f"ch={e['changed_bytes']:3d}"]
        if e.get('spell'): parts.append(f"spell={e['spell']}")
        if e.get('netids'): parts.append(f"targets={e['netids']}")
        if e.get('positions'): parts.append(f"pos={e['positions']}")
        print(f"    {'  '.join(parts)}")

    # ================================================================
    # Step 5: When we see an active attack event, dump the changed fields
    # ================================================================
    if events:
        # Find first AS event with NetIDs
        target_event = None
        for e in events:
            if e['type'] == 'AS' and e.get('netids'):
                target_event = e
                break

        if target_event:
            print(f"\n=== Detailed dump of ActiveSpell during attack ===")
            print(f"  Hero: {target_event['hero']}, Spell: {target_event['spell']}")
            print(f"  Targets: {target_event['netids']}")
            print(f"  Positions: {target_event.get('positions', [])}")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
