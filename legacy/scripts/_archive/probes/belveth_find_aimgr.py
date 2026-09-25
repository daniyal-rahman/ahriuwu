"""Scan all u64 pointers in Bel'Veth's hero struct looking for one where
the pointee has a Vec3 at +0x474 matching her position. That's the canonical
"AiManager has ServerPos mirror" signature documented in the UC forum.

If found: report pointer slot offset in hero, then sample TargetPos (+0x34),
PathEnd (+0x33C), Velocity (+0x318), IsMoving (+0x31C) during play.
"""
import ctypes, struct, subprocess, sys, json, time, os, ssl, urllib.request
from ctypes import wintypes
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

FOCUSED_HERO_PTR_RVA = 0x1E13490
POSITION             = 0x200
STRUCT_SIZE          = 128 * 1024
SERVERPOS_OFF        = 0x474

_k = ctypes.WinDLL("kernel32", use_last_error=True)
class ME32(ctypes.Structure):
    _fields_ = [("dwSize",wintypes.DWORD),("a",wintypes.DWORD),("pid",wintypes.DWORD),
                ("b",wintypes.DWORD),("c",wintypes.DWORD),
                ("modBase",ctypes.POINTER(ctypes.c_byte)),("modSize",wintypes.DWORD),
                ("hMod",wintypes.HMODULE),("szMod",ctypes.c_char*256),
                ("szPath",ctypes.c_char*260)]
def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
def find_base(pid):
    snap = _k.CreateToolhelp32Snapshot(0x18, pid)
    me = ME32(); me.dwSize = ctypes.sizeof(ME32)
    if _k.Module32First(snap, ctypes.byref(me)):
        while True:
            if b'league' in me.szMod.lower():
                b = ctypes.cast(me.modBase, ctypes.c_void_p).value
                _k.CloseHandle(snap); return b, me.modSize
            if not _k.Module32Next(snap, ctypes.byref(me)): break
    _k.CloseHandle(snap); return None, None
class Mem:
    def __init__(self, pid): self.h = _k.OpenProcess(0x0410, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok else b""
    def u64(self, a): d = self.read(a, 8); return struct.unpack("<Q", d)[0] if len(d)==8 else None
    def vec3(self, a): d = self.read(a, 12); return struct.unpack("<fff", d) if len(d)==12 else (0,0,0)

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def _post(obj):
    req = urllib.request.Request("https://127.0.0.1:2999/replay/playback",
            data=json.dumps(obj).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def _get():
    with urllib.request.urlopen("https://127.0.0.1:2999/replay/playback", context=_ctx, timeout=2) as r:
        return json.loads(r.read())

def main():
    pid = find_pid(); base, _ = find_base(pid); m = Mem(pid)
    # Unpause briefly if paused so position is "current"
    try: _get()
    except: pass
    time.sleep(0.2)
    hero = m.u64(base + FOCUSED_HERO_PTR_RVA)
    px, py, pz = m.vec3(hero + POSITION)
    print(f"hero=0x{hero:X} pos=({px:.0f},{py:.0f},{pz:.0f})")

    print(f"\nScanning hero struct for u64s pointing to a struct with Vec3 @ +0x{SERVERPOS_OFF:X} matching pos...")
    hero_raw = m.read(hero, STRUCT_SIZE)
    hits = []
    for slot_off in range(0, STRUCT_SIZE - 8, 8):
        val = struct.unpack_from("<Q", hero_raw, slot_off)[0]
        if not (0x10000000000 < val < 0x7FF000000000): continue
        # try reading +0x474 at that address as Vec3
        v = m.vec3(val + SERVERPOS_OFF)
        if any(c != c for c in v): continue  # NaN
        if abs(v[0] - px) < 40 and abs(v[2] - pz) < 40:
            hits.append((slot_off, val, v))
    print(f"\n{len(hits)} pointer slots in hero struct whose pointee has matching ServerPos:")
    for slot_off, val, v in hits[:30]:
        print(f"  hero+0x{slot_off:04X} -> 0x{val:X}  ServerPos=({v[0]:.0f},{v[1]:.0f},{v[2]:.0f})")
    if not hits:
        print("NONE. Replay engine does not maintain any AiManager-like struct with ServerPos matching hero.pos.")
        return

    # For each hit, also sample TargetPos / PathEnd / Velocity / IsMoving / SegCount once
    print("\n--- AiManager fields at each hit (static snapshot) ---")
    for slot_off, addr, _ in hits[:10]:
        tp = m.vec3(addr + 0x34)
        pe = m.vec3(addr + 0x33C)
        ps = m.vec3(addr + 0x330)
        mv = m.vec3(addr + 0x480)
        vel = m.vec3(addr + 0x318)
        im = m.read(addr + 0x31C, 1)
        cnt = m.read(addr + 0x350, 4)
        seg_ptr = m.u64(addr + 0x348)
        print(f"\n  slot=hero+0x{slot_off:04X} aimgr=0x{addr:X}")
        print(f"    TargetPos @+0x034  = ({tp[0]:.0f},{tp[1]:.2f},{tp[2]:.0f})")
        print(f"    PathStart @+0x330  = ({ps[0]:.0f},{ps[1]:.2f},{ps[2]:.0f})")
        print(f"    PathEnd   @+0x33C  = ({pe[0]:.0f},{pe[1]:.2f},{pe[2]:.0f})")
        print(f"    Velocity  @+0x318  = ({vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f})")
        print(f"    MoveVec3  @+0x480  = ({mv[0]:.2f},{mv[1]:.2f},{mv[2]:.2f})")
        print(f"    IsMoving  @+0x31C  = {im[0] if im else '?'}")
        print(f"    SegCount  @+0x350  = {struct.unpack('<I', cnt)[0] if len(cnt)==4 else '?'}")
        print(f"    SegPtr    @+0x348  = 0x{seg_ptr or 0:X}")

if __name__ == "__main__":
    main()
