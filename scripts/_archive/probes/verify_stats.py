"""Verify stats_base and stat sub-offsets via API ground truth."""
import ctypes, struct, subprocess, sys, json, ssl, urllib.request, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
_k = ctypes.windll.kernel32
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE

class Mem:
    def __init__(self, pid): self.h = _k.OpenProcess(0x0410, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u64(self, a): d=self.read(a,8); return struct.unpack('<Q',d)[0] if d else None
    def f32(self, a): d=self.read(a,4); return struct.unpack('<f',d)[0] if d else None
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

HERO_ARRAY_RVA = 0x1DD7128
CHAMPION_NAME  = 0x4360

pid = find_pid(); base, _ = find_base(pid); m = Mem(pid)

# gt=600 has Garen at level 9 with some items → stats should be non-trivial
rpost("/replay/playback", {"time": 600.0, "speed": 0.0, "paused": True})
time.sleep(1.2); rpost("/replay/playback", {"speed": 1.0, "paused": False}); time.sleep(2)
rpost("/replay/playback", {"speed": 0.0, "paused": True}); time.sleep(0.3)

arr = m.u64(base + HERO_ARRAY_RVA)
garen_hp = None
for i in range(10):
    hp = m.u64(arr + i*8)
    if hp and m.string(hp + CHAMPION_NAME, 40) == "Garen":
        garen_hp = hp; break
print(f"garen_hp = 0x{garen_hp:X}")

# Garen's base atk range = 175.0 (exact). Level-9 move speed ~340-420.
# 16.7 stats_base=0x1B88, stat_atk_range=0x618, stat_move_speed=0x5C8
# Probable new stats_base ~= 0x1B88+0x38 = 0x1BC0 (if all hero growth applies before this)
# Or still 0x1B88 (if growth is above this offset)
# Scan hero+0x1A00..0x2500 for f32 matching known stats

print("\nScanning hero+0x1A00..0x2600 for f32 in plausible stat ranges:")
hits_by_kind = {"atk_range_175":[], "move_speed":[], "small_armor":[], "atk_damage":[]}
for off in range(0x1A00, 0x2600, 4):
    v = m.f32(garen_hp + off)
    if v is None: continue
    if abs(v - 175.0) < 0.01:
        hits_by_kind["atk_range_175"].append((off, v))
    if 330 < v < 450:
        hits_by_kind["move_speed"].append((off, v))
    if 30 < v < 90:  # plausible armor level 9
        hits_by_kind["small_armor"].append((off, v))
    if 70 < v < 200:  # plausible AD level 9 + base ~65 + items
        hits_by_kind["atk_damage"].append((off, v))

for k, hits in hits_by_kind.items():
    print(f"\n  {k}: {len(hits)} hits")
    for off, v in hits:
        print(f"    hero+0x{off:X}: {v:.2f}")
