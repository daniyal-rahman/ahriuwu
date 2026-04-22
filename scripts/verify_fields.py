"""Verify suspected-wrong offsets (position, gold_current, spellbook) live."""
import ctypes, struct, subprocess, sys, json, ssl, urllib.request, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
_k = ctypes.windll.kernel32
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE

LO, HI = 0x10000000, 0x7FFFFFFFFFFF

class Mem:
    def __init__(self, pid): self.h = _k.OpenProcess(0x0410, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u64(self, a): d=self.read(a,8); return struct.unpack('<Q',d)[0] if d else None
    def u32(self, a): d=self.read(a,4); return struct.unpack('<I',d)[0] if d else None
    def f32(self, a): d=self.read(a,4); return struct.unpack('<f',d)[0] if d else None
    def vec3(self, a): d=self.read(a,12); return struct.unpack('<fff',d) if d and len(d)==12 else None
    def string(self, a, n=80):
        d=self.read(a,n);
        if not d: return None
        try: return d.split(b'\x00')[0].decode('ascii')
        except: return None

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

def find_base(pid):
    class ME(ctypes.Structure):
        _fields_=[("dwSize",ctypes.c_ulong),("a",ctypes.c_ulong),("b",ctypes.c_ulong),
            ("c",ctypes.c_ulong),("d",ctypes.c_ulong),("modBaseAddr",ctypes.POINTER(ctypes.c_byte)),
            ("modBaseSize",ctypes.c_ulong),("hModule",ctypes.c_void_p),
            ("szModule",ctypes.c_char*256),("szExePath",ctypes.c_char*260)]
    snap=_k.CreateToolhelp32Snapshot(0x18,pid);me=ME();me.dwSize=ctypes.sizeof(ME)
    if _k.Module32First(snap,ctypes.byref(me)):
        while True:
            if b'league' in me.szModule.lower():
                _k.CloseHandle(snap)
                return ctypes.cast(me.modBaseAddr,ctypes.c_void_p).value, me.modBaseSize
            if not _k.Module32Next(snap,ctypes.byref(me)): break
    return None, None

def rpost(ep, d):
    return json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://127.0.0.1:2999{ep}", data=json.dumps(d).encode(),
        headers={"Content-Type":"application/json"}), context=_ctx, timeout=5).read())
def rget(ep):
    return json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://127.0.0.1:2999{ep}"), context=_ctx, timeout=5).read())

pid = find_pid(); base, _ = find_base(pid); m = Mem(pid)
print(f"PID={pid} base=0x{base:X}")

HERO_ARRAY_RVA = 0x1DD7128
CHAMPION_NAME  = 0x4360

# Seek to gt=300 (lane phase, should be gold accumulation)
rpost("/replay/playback", {"time": 300.0, "speed": 0.0, "paused": True})
time.sleep(1.2); rpost("/replay/playback", {"speed": 1.0, "paused": False}); time.sleep(2)
rpost("/replay/playback", {"speed": 0.0, "paused": True}); time.sleep(0.3)

arr = m.u64(base + HERO_ARRAY_RVA)
garen_hp = None
for i in range(10):
    hp = m.u64(arr + i*8)
    if hp and m.string(hp + CHAMPION_NAME, 40) == "Garen":
        garen_hp = hp; break
print(f"garen_hp = 0x{garen_hp:X}")

# Get API ground truth
pl = rget("/liveclientdata/playerlist")
api_g = next(p for p in pl if p.get("championName") == "Garen")
print(f"API garen: level={api_g.get('level')} score={api_g.get('scores')}")

# --- POSITION ---
print("\n--- Position candidates (vec3 with x in [100,16000], y in [30,300], z in [100,16000]) ---")
# Camera focus API gives camera position; Garen world position should be close to camera (since camera locked).
cam = rget("/replay/render")
print(f"camera position: {cam.get('cameraPosition')}")

for off in range(0x100, 0x400, 4):
    v = m.vec3(garen_hp + off)
    if v and 100 < v[0] < 16000 and 20 < v[1] < 300 and 100 < v[2] < 16000:
        print(f"  0x{off:X}: ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f})")

# --- GOLD_CURRENT / GOLD_TOTAL ---
print("\n--- Gold at gt=300 (lane) vs gt=600 (mid-game) ---")
def snap_gold(hp):
    out = {}
    for off in range(0x2000, 0x5000, 4):
        v = m.f32(hp + off)
        if v is not None and 0 < v < 30000:
            out[off] = round(v, 1)
    return out

s1 = snap_gold(garen_hp)
rpost("/replay/playback", {"time": 600.0, "speed": 0.0, "paused": True})
time.sleep(1.2); rpost("/replay/playback", {"speed": 1.0, "paused": False}); time.sleep(2)
rpost("/replay/playback", {"speed": 0.0, "paused": True}); time.sleep(0.3)
arr = m.u64(base + HERO_ARRAY_RVA)
garen_hp2 = None
for i in range(10):
    hp = m.u64(arr + i*8)
    if hp and m.string(hp + CHAMPION_NAME, 40) == "Garen":
        garen_hp2 = hp; break
print(f"garen_hp2 = 0x{garen_hp2:X}")
s2 = snap_gold(garen_hp2)

# Look for offsets that increased (gold_earned monotonic) and are in realistic range
print("  offsets where v2>v1+500 (earned-like):")
for off in sorted(s1.keys()):
    v1 = s1.get(off); v2 = s2.get(off)
    if v1 is not None and v2 is not None and v2 > v1 + 500 and v2 < 30000:
        print(f"    hero+0x{off:X}: {v1:.0f} → {v2:.0f}")

# --- SPELLBOOK ---
print("\n--- SpellBook probe (looking for GarenQ/W/E/R) ---")
# At gt=600 Garen has leveled spells. Find inline spellbook.
for sb_off in range(0x2F00, 0x3400, 8):
    for sa_off in range(0xA80, 0xB40, 8):
        found = set()
        for i in range(4):
            slot_ptr = m.u64(garen_hp2 + sb_off + sa_off + i*8)
            if not slot_ptr or not (LO < slot_ptr < HI): continue
            for info_off in [0x128, 0x130, 0x120, 0x138]:
                si = m.u64(slot_ptr + info_off)
                if si and LO < si < HI:
                    np = m.u64(si + 0x28)
                    if np and LO < np < HI:
                        sn = m.string(np)
                        if sn and sn.startswith("Garen") and sn[5:6] in "QWER":
                            found.add(sn)
        if len(found) >= 3:
            print(f"  hero+0x{sb_off:X}+0x{sa_off:X}: {sorted(found)}")
