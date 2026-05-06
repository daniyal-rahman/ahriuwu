"""Test: is AiManager INLINE in the hero struct in replay mode?

For each candidate base B in hero+0x3800..0x4800:
  - ServerPos candidate: *(B+0x474) should ≈ hero.pos
  - Also try ServerPos at other known offsets (+0x480, +0x330, +0x33C)
  - For survivors, verify full AiManager signature:
      SegmentsCount in [0, 100], Segments ptr heap,
      Velocity in [0, 2000], IsMoving byte in {0,1}

Replay gets paused for stable sampling; position re-read at sample time.
"""
import ctypes, struct, subprocess, sys, json, time, math, ssl, urllib.request
from ctypes import wintypes
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

HERO_ARRAY_RVA = 0x1DD7128
POSITION       = 0x25C
CHAMPION_NAME  = 0x4360

# All known AiManager field offsets (inside the struct)
AI_TARGET_POS     = 0x034
AI_VELOCITY       = 0x318
AI_IS_MOVING      = 0x31C
AI_CURRENT_SEG    = 0x320
AI_PATH_START     = 0x330
AI_PATH_END       = 0x33C
AI_SEGMENTS       = 0x348
AI_SEGMENTS_COUNT = 0x350
AI_DASH_SPEED     = 0x360
AI_IS_DASHING     = 0x384
AI_SERVER_POS     = 0x474
AI_MOVE_VEC3      = 0x480

# ---- Windows plumbing (same as probe) ----
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
    def __init__(self, pid):  self.h = _k.OpenProcess(0x0410, False, pid)
    def _r(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u8 (self, a): d=self._r(a,1);  return d[0]                     if d else None
    def u32(self, a): d=self._r(a,4);  return struct.unpack("<I",d)[0] if d else None
    def i32(self, a): d=self._r(a,4);  return struct.unpack("<i",d)[0] if d else None
    def u64(self, a): d=self._r(a,8);  return struct.unpack("<Q",d)[0] if d else None
    def f32(self, a): d=self._r(a,4);  return struct.unpack("<f",d)[0] if d else None
    def vec3(self,a): d=self._r(a,12); return struct.unpack("<fff",d)  if d else None
    def string(self, a, n=64):
        d=self._r(a,n);  return d.split(b'\x00')[0].decode('ascii',errors='replace') if d else None

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF
def is_map(v):
    if v is None: return False
    x,y,z=v
    if abs(x)<0.1 and abs(z)<0.1: return False
    return -500<x<16000 and -500<y<1500 and -500<z<16000
def d2(a,b): return math.sqrt((a[0]-b[0])**2+(a[2]-b[2])**2)

_ssl = ssl.create_default_context(); _ssl.check_hostname=False; _ssl.verify_mode=ssl.CERT_NONE
def replay_playback(body):
    try:
        urllib.request.urlopen(urllib.request.Request(
            "https://127.0.0.1:2999/replay/playback", method="POST",
            data=json.dumps(body).encode(),
            headers={"Content-Type":"application/json"}), context=_ssl, timeout=3).read()
    except Exception as e:
        print(f"  playback err: {e}")

def score_inline_ai(m, hero_ptr, B, hero_pos):
    """Assume AiManager base is at hero_ptr + B, inline. Score it."""
    score = 0; d = {"B": f"0x{B:X}"}

    spos = m.vec3(hero_ptr + B + AI_SERVER_POS)
    d["server_pos"] = spos
    if spos and is_map(spos):
        dist = d2(spos, hero_pos); d["spos_dist"] = round(dist, 2)
        if   dist <   5: score += 5
        elif dist <  50: score += 3
        elif dist < 500: score += 1

    tgt = m.vec3(hero_ptr + B + AI_TARGET_POS)
    d["target_pos"] = tgt
    if tgt and is_map(tgt): score += 1

    ps = m.vec3(hero_ptr + B + AI_PATH_START)
    pe = m.vec3(hero_ptr + B + AI_PATH_END)
    d["path_start"], d["path_end"] = ps, pe
    if ps and is_map(ps): score += 1
    if pe and is_map(pe): score += 1

    vel = m.f32(hero_ptr + B + AI_VELOCITY); d["velocity"] = vel
    if vel is not None and 0 <= vel <= 2500: score += 1

    segc = m.i32(hero_ptr + B + AI_SEGMENTS_COUNT); d["seg_count"] = segc
    if segc is not None and 0 <= segc <= 100: score += 1

    seg_ptr = m.u64(hero_ptr + B + AI_SEGMENTS); d["seg_ptr"] = (f"0x{seg_ptr:X}" if seg_ptr else None)
    if seg_ptr and is_heap(seg_ptr): score += 1

    im = m.u8(hero_ptr + B + AI_IS_MOVING); d["is_moving"] = im
    if im in (0, 1): score += 1

    return score, d

def main():
    pid = find_pid();  base,_ = find_base(pid);  m = Mem(pid)
    print(f"PID={pid} base=0x{base:X}")
    arr = m.u64(base + HERO_ARRAY_RVA)
    heroes = []
    for i in range(10):
        hp = m.u64(arr + i*8)
        if not is_heap(hp): continue
        name = m.string(hp + CHAMPION_NAME, 40) or f"h{i}"
        pos  = m.vec3(hp + POSITION)
        heroes.append({"ptr": hp, "name": name, "pos": pos})

    # Pause for stability, re-read positions fresh
    print("\nPausing replay for atomic sampling...")
    replay_playback({"speed":0.0,"paused":True}); time.sleep(0.4)
    for h in heroes:
        h["pos"] = m.vec3(h["ptr"] + POSITION)

    print("Heroes (fresh pos after pause):")
    for h in heroes:
        p = h["pos"]; print(f"  {h['name']:10s} 0x{h['ptr']:X} ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})")

    # === Phase A: sweep candidate inline-AiManager base B
    #     s.t. B+0x474 (ServerPos) ≈ hero.pos ===
    # Look in 0x3800..0x5000 every 4 bytes.
    tgt = heroes[0]
    hero_ptr, hero_pos = tgt["ptr"], tgt["pos"]
    print(f"\n=== Phase A: inline-AiManager sweep for {tgt['name']} ===")

    hits = []
    for B in range(0x3800, 0x5000, 4):
        v = m.vec3(hero_ptr + B + AI_SERVER_POS)
        if v and is_map(v) and d2(v, hero_pos) < 5:
            hits.append(B)
    print(f"  Phase A.1: {len(hits)} base(s) where B+0x474 ≈ hero.pos (<5u)")
    for B in hits[:20]:
        print(f"    B=0x{B:04X}  (B+0x474 = hero+0x{B+AI_SERVER_POS:04X})")

    # Full signature score for each hit
    print(f"\n  Phase A.2: full signature score for each hit")
    scored = []
    for B in hits:
        s, d = score_inline_ai(m, hero_ptr, B, hero_pos)
        scored.append((s, B, d))
    scored.sort(key=lambda x: -x[0])
    for s, B, d in scored[:15]:
        print(f"    B=0x{B:04X} score={s}/11  "
              f"tgt={d.get('target_pos')}  vel={d.get('velocity')}  "
              f"segs={d.get('seg_count')}  seg_ptr={d.get('seg_ptr')}  mov={d.get('is_moving')}")

    # === Phase B: same sweep but across ALL heroes. Best base should work everywhere. ===
    if scored:
        print(f"\n=== Phase B: verify best base across all 10 heroes ===")
        best_B = scored[0][1]
        cross = []
        for h in heroes:
            s, d = score_inline_ai(m, h["ptr"], best_B, h["pos"])
            cross.append((h["name"], s, d))
            print(f"    {h['name']:10s} @ B=0x{best_B:04X} score={s}/11 "
                  f"spos_dist={d.get('spos_dist','-')} vel={d.get('velocity')} segs={d.get('seg_count')}")

    # === Phase C: unpause and time-sample fields to verify they change sensibly ===
    if scored and scored[0][0] >= 6:
        best_B = scored[0][1]
        print(f"\n=== Phase C: time-sample 20 frames @ 0.25s (unpaused) at B=0x{best_B:X} ===")
        replay_playback({"speed":1.0,"paused":False}); time.sleep(0.3)
        frames = []
        for t in range(20):
            f = {"t": round(t*0.25, 2), "heroes": {}}
            for h in heroes:
                pos  = m.vec3(h["ptr"] + POSITION)
                spos = m.vec3(h["ptr"] + best_B + AI_SERVER_POS)
                tgt  = m.vec3(h["ptr"] + best_B + AI_TARGET_POS)
                pe   = m.vec3(h["ptr"] + best_B + AI_PATH_END)
                vel  = m.f32 (h["ptr"] + best_B + AI_VELOCITY)
                im   = m.u8  (h["ptr"] + best_B + AI_IS_MOVING)
                seg  = m.i32 (h["ptr"] + best_B + AI_SEGMENTS_COUNT)
                f["heroes"][h["name"]] = {"pos":pos, "spos":spos, "tgt":tgt, "pe":pe,
                                           "vel":vel, "mov":im, "seg":seg}
            frames.append(f)
            time.sleep(0.25)

        # Report variability of target-like fields
        first = heroes[0]["name"]
        def uniq(key):
            vals = [f["heroes"][first].get(key) for f in frames if f["heroes"].get(first)]
            return len({tuple(v) if isinstance(v, tuple) else v for v in vals if v is not None})
        print(f"  {first}: uniq(target_pos)={uniq('tgt')}  uniq(path_end)={uniq('pe')}  uniq(mov)={uniq('mov')}  uniq(seg_count)={uniq('seg')}")
        out = {"winner_B": f"0x{best_B:X}", "frames": frames}
        with open(r"C:\Users\daniz\aimanager_inline_out.json","w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"\nWrote C:\\Users\\daniz\\aimanager_inline_out.json")
    else:
        print("\nNo inline base scored high enough for time-sampling.")

if __name__ == "__main__":
    main()
