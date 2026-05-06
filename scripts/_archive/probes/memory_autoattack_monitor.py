"""
Auto-attack monitor for ALL heroes.

Discovered offsets:
  hero+0x4220: Target position X (f32)
  hero+0x4224: Target position Y (f32)
  hero+0x4228: Target position Z (f32)
  hero+0x422C: Target NetID (u32) - includes heroes AND minions
  hero+0x3FF8: Game time on hero struct (f32)

Monitors all 10 heroes for target changes at 20Hz.
Logs: who attacked, target NetID, target position, game time.
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
    print(f"Heroes: {[(h['name'], hex(h['net_id'] or 0)) for h in heroes]}")

    # Seek replay
    print("\nSeeking to t=300 (5 min, early lane)...")
    api_post("/replay/playback", {"time": 300.0})
    time.sleep(0.5)
    api_post("/replay/playback", {"paused": False, "speed": 2.0})
    time.sleep(1)

    pb = api_get("/replay/playback")
    gt = pb.get('time', 0) if pb else 0
    print(f"Game time: {gt:.1f}s")

    # ================================================================
    # First verify the offsets on all heroes
    # ================================================================
    print("\n=== Current attack targets for all heroes ===")
    for h in heroes:
        target_pos = m.vec3(h["ptr"] + 0x4220)
        target_netid = m.u32(h["ptr"] + 0x422C)
        hero_pos = m.vec3(h["ptr"] + 0x25C)
        local_gt = m.f32(h["ptr"] + 0x3FF8)

        target_name = netid_map.get(target_netid, None)
        is_hero_target = target_name is not None
        tgt_str = target_name if is_hero_target else (f"0x{target_netid:08X}" if target_netid else "none")
        tgt_type = "HERO" if is_hero_target else ("minion/obj" if target_netid and target_netid > 0x40000000 else "none")

        pos_str = f"({hero_pos[0]:.0f},{hero_pos[2]:.0f})" if hero_pos else "?"
        tpos_str = f"({target_pos[0]:.0f},{target_pos[2]:.0f})" if target_pos else "?"
        gt_str = f"{local_gt:.1f}" if local_gt else "?"

        print(f"  {h['name']:15s} pos={pos_str:15s} target={tgt_str:20s} [{tgt_type}] tpos={tpos_str:15s} gt={gt_str}")

    # ================================================================
    # Monitor for 20 seconds at 10Hz — detect attack target changes
    # ================================================================
    print(f"\n=== Monitoring attack targets (20s at 10Hz, speed=2x) ===")
    print(f"{'Time':>6s} {'Hero':>12s} {'Old Target':>20s} {'New Target':>20s} {'Target Pos':>20s} {'Type':>8s}")
    print("-" * 90)

    prev_targets = {}
    for h in heroes:
        target_netid = m.u32(h["ptr"] + 0x422C)
        prev_targets[h['name']] = target_netid

    events = []
    for tick in range(200):
        time.sleep(0.1)

        for h in heroes:
            target_netid = m.u32(h["ptr"] + 0x422C)
            if target_netid != prev_targets[h['name']]:
                old = prev_targets[h['name']]
                target_pos = m.vec3(h["ptr"] + 0x4220)
                local_gt = m.f32(h["ptr"] + 0x3FF8)

                old_name = netid_map.get(old, f"0x{old:08X}" if old else "none")
                new_name = netid_map.get(target_netid, f"0x{target_netid:08X}" if target_netid else "none")

                is_hero = target_netid in netid_map
                tgt_type = "HERO" if is_hero else "minion"

                tpos_str = f"({target_pos[0]:.0f},{target_pos[2]:.0f})" if target_pos else "?"
                gt_str = f"{local_gt:.1f}" if local_gt else "?"

                event = {
                    "game_time": local_gt,
                    "hero": h['name'],
                    "old_target": old,
                    "new_target": target_netid,
                    "target_pos": target_pos,
                    "is_hero_target": is_hero,
                }
                events.append(event)

                # Only print first 40
                if len(events) <= 40:
                    print(f"{gt_str:>6s} {h['name']:>12s} {old_name:>20s} {new_name:>20s} {tpos_str:>20s} {tgt_type:>8s}")

                prev_targets[h['name']] = target_netid

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n=== Summary ({len(events)} target changes in 20s) ===")

    # Count by hero
    by_hero = {}
    for e in events:
        name = e['hero']
        if name not in by_hero:
            by_hero[name] = {"total": 0, "hero_targets": 0, "minion_targets": 0}
        by_hero[name]["total"] += 1
        if e["is_hero_target"]:
            by_hero[name]["hero_targets"] += 1
        else:
            by_hero[name]["minion_targets"] += 1

    for name, counts in sorted(by_hero.items()):
        print(f"  {name:15s}: {counts['total']:3d} target changes "
              f"({counts['hero_targets']} heroes, {counts['minion_targets']} minions)")

    # Show Garen's attack timeline specifically
    garen_events = [e for e in events if e['hero'] == 'Garen']
    if garen_events:
        print(f"\n=== Garen's attack timeline ({len(garen_events)} events) ===")
        for e in garen_events[:20]:
            target = netid_map.get(e['new_target'], f"minion 0x{e['new_target']:08X}")
            pos = e['target_pos']
            pos_str = f"({pos[0]:.0f},{pos[2]:.0f})" if pos else "?"
            gt = f"{e['game_time']:.1f}" if e['game_time'] else "?"
            print(f"  t={gt}: target={target} at {pos_str}")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
