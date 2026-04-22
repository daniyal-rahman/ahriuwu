"""Find kills/assists u32 offsets via API-verified delta."""
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
    def u32(self, a): d=self.read(a,4); return struct.unpack('<I',d)[0] if d else None
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

def find_garen():
    arr = m.u64(base + HERO_ARRAY_RVA)
    for i in range(10):
        hp = m.u64(arr + i*8)
        if hp and m.string(hp + CHAMPION_NAME, 40) == "Garen":
            return hp
    return None

# Early game — minimal kills. gt=200 (just into lane)
rpost("/replay/playback", {"time": 200.0, "speed": 0.0, "paused": True})
time.sleep(1.2); rpost("/replay/playback", {"speed": 1.0, "paused": False}); time.sleep(2)
rpost("/replay/playback", {"speed": 0.0, "paused": True}); time.sleep(0.3)
garen_hp = find_garen()
api1 = next(p for p in rget("/liveclientdata/playerlist") if p.get("championName") == "Garen")
snap1 = {off: m.u32(garen_hp + off) for off in range(0x4F00, 0x5C00, 4)}

# Late game — more kills
rpost("/replay/playback", {"time": 1500.0, "speed": 0.0, "paused": True})
time.sleep(1.2); rpost("/replay/playback", {"speed": 1.0, "paused": False}); time.sleep(2)
rpost("/replay/playback", {"speed": 0.0, "paused": True}); time.sleep(0.3)
garen_hp = find_garen()
api2 = next(p for p in rget("/liveclientdata/playerlist") if p.get("championName") == "Garen")
snap2 = {off: m.u32(garen_hp + off) for off in range(0x4F00, 0x5C00, 4)}

print(f"API gt=200: K={api1.get('scores',{}).get('kills',0)} D={api1.get('scores',{}).get('deaths',0)} A={api1.get('scores',{}).get('assists',0)} ws={api1.get('scores',{}).get('wardScore',0):.1f}")
print(f"API gt=1500: K={api2.get('scores',{}).get('kills',0)} D={api2.get('scores',{}).get('deaths',0)} A={api2.get('scores',{}).get('assists',0)} ws={api2.get('scores',{}).get('wardScore',0):.1f}")
k1 = api1.get('scores',{}).get('kills',0); k2 = api2.get('scores',{}).get('kills',0)
a1 = api1.get('scores',{}).get('assists',0); a2 = api2.get('scores',{}).get('assists',0)
d1 = api1.get('scores',{}).get('deaths',0); d2 = api2.get('scores',{}).get('deaths',0)
cs1 = api1.get('scores',{}).get('creepScore',0); cs2 = api2.get('scores',{}).get('creepScore',0)

print(f"\nExpected kills: {k1}→{k2}, deaths: {d1}→{d2}, assists: {a1}→{a2}, cs: {cs1}→{cs2}")
print("\nU32 fields matching expected deltas:")
for off in sorted(snap1.keys()):
    v1, v2 = snap1.get(off), snap2.get(off)
    if v1 is None or v2 is None: continue
    if v1 > 1000 or v2 > 1000: continue  # noise filter
    label = []
    if v1 == k1 and v2 == k2 and k1 != k2: label.append(f"KILLS({k1}->{k2})")
    if v1 == d1 and v2 == d2 and d1 != d2: label.append(f"DEATHS({d1}->{d2})")
    if v1 == a1 and v2 == a2 and a1 != a2: label.append(f"ASSISTS({a1}->{a2})")
    if v1 == cs1 and v2 == cs2 and cs1 != cs2: label.append(f"CS({cs1}->{cs2})")
    if label:
        print(f"  hero+0x{off:X}: {v1} -> {v2}  [{' | '.join(label)}]")
