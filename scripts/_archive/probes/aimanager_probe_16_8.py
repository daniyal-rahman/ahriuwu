"""AiManager probe for League patch 16.8.764.3737.

Combines hail-mary forum offsets with a systematic hero-struct scan,
ranks by ServerPos-matches-hero-position, then time-samples the winner.

Signal battery (10 pts):
  +5  ServerPos (+0x474) within 50u of hero position (strongest anchor)
  +2  same within 500u
  +1  TargetPosition (+0x34 or +0x33C) is map-valid
  +1  Velocity (+0x318) in [0, 2000]
  +1  SegmentsCount (+0x350) in [0, 100]
  +1  Segments ptr (+0x348) is heap
  +1  IsMoving (+0x31C) byte in {0, 1}

Output: C:\\Users\\daniz\\aimanager_probe_out.json
"""
import ctypes, struct, subprocess, sys, json, time, math
from ctypes import wintypes

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

# --- 16.8 verified hero-struct offsets (from offsets_16_8.json) ---
MOD_SIZE_EXPECTED = 0x2097000
HERO_ARRAY_RVA    = 0x1DD7128
POSITION          = 0x25C
CHAMPION_NAME     = 0x4360

# --- AiManager internal offsets (from recent UC dumps, consistent across all) ---
AI_TARGET_A        = 0x034   # most common
AI_TARGET_B        = 0x33C   # a768787747 variant (= PathStart + 0xC)
AI_VELOCITY        = 0x318
AI_IS_MOVING       = 0x31C
AI_CURRENT_SEG     = 0x320
AI_PATH_START      = 0x330
AI_PATH_END        = 0x33C
AI_SEGMENTS        = 0x348
AI_SEGMENTS_COUNT  = 0x350
AI_IS_DASHING      = 0x384
AI_SERVER_POS      = 0x474

# --- Hail-mary hero-slot candidates ---
# (offset, needs_inner_deref, source)
HAIL_MARY = [
    (0x4028, False, "a768787747 UC 2026-03-29"),
    (0x4030, False, "ibrahimcelik UC 2026-04-07"),
    (0x4038, False, "hacker_logs older"),
    (0x4060, False, "0x4028 + 0x38 (16.8 shift)"),
    (0x4068, False, "0x4030 + 0x38"),
    (0x4070, False, "0x4038 + 0x38"),
    (0x41A8, False, "sq834960394 CN"),
    (0x41E0, False, "0x41A8 + 0x38"),
    (0x41F0, True,  "trankhanhtinh1 IDA 2026-03-21 (inner+0x10)"),
    (0x4228, True,  "0x41F0 + 0x38 (inner+0x10)"),
    (0x41F0, False, "0x41F0 direct (fallback)"),
]

SCAN_LO, SCAN_HI = 0x3E00, 0x4500   # hero-struct brute-scan range

# ---------- Windows plumbing ----------
_k = ctypes.WinDLL("kernel32", use_last_error=True)
TH32_PROC = 0x2; TH32_MOD = 0x18

class PE32(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD),("a",wintypes.DWORD),("pid",wintypes.DWORD),
                ("b",ctypes.POINTER(ctypes.c_ulong)),("c",wintypes.DWORD),("d",wintypes.DWORD),
                ("e",wintypes.DWORD),("f",ctypes.c_long),("g",wintypes.DWORD),
                ("exe",ctypes.c_char*260)]
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
        if 'league' in l.lower():
            return int(l.strip('"').split('","')[1])
    return None

def find_base(pid):
    snap = _k.CreateToolhelp32Snapshot(TH32_MOD, pid)
    me = ME32(); me.dwSize = ctypes.sizeof(ME32)
    if _k.Module32First(snap, ctypes.byref(me)):
        while True:
            if b'league' in me.szMod.lower():
                b = ctypes.cast(me.modBase, ctypes.c_void_p).value
                _k.CloseHandle(snap); return b, me.modSize
            if not _k.Module32Next(snap, ctypes.byref(me)): break
    _k.CloseHandle(snap); return None, None

class Mem:
    def __init__(self, pid):
        self.h = _k.OpenProcess(0x0410, False, pid)
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
        d = self._r(a, n)
        if not d: return None
        return d.split(b'\x00')[0].decode('ascii', errors='replace')
    def close(self): _k.CloseHandle(self.h)

def is_heap(v):  return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF
def is_map(v):
    if v is None: return False
    x, y, z = v
    # Exclude (0,0,0) exact — that's "uninitialized vec3"
    if abs(x) < 0.1 and abs(z) < 0.1: return False
    return -500 < x < 16000 and -500 < y < 1500 and -500 < z < 16000

def dist2d(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[2]-b[2])**2)

# ---------- scoring ----------
def score(m, ai, hero_pos):
    s = 0; d = {}

    spos = m.vec3(ai + AI_SERVER_POS)
    d["server_pos"] = spos
    if spos and is_map(spos):
        dd = dist2d(spos, hero_pos)
        d["spos_dist"] = round(dd, 1)
        if   dd <   50: s += 5
        elif dd <  500: s += 2

    for tname, toff in [("tA_0x34", AI_TARGET_A), ("tB_0x33C", AI_TARGET_B)]:
        v = m.vec3(ai + toff); d[tname] = v
        if v and is_map(v):
            s += 1; break

    vel = m.f32(ai + AI_VELOCITY); d["velocity"] = vel
    if vel is not None and 0 <= vel <= 2000:
        s += 1

    segs = m.i32(ai + AI_SEGMENTS_COUNT); d["seg_count"] = segs
    if segs is not None and 0 <= segs <= 100:
        s += 1

    seg_ptr = m.u64(ai + AI_SEGMENTS); d["seg_ptr_heap"] = bool(seg_ptr and is_heap(seg_ptr))
    if seg_ptr and is_heap(seg_ptr):
        s += 1

    im = m.u8(ai + AI_IS_MOVING); d["is_moving"] = im
    if im in (0, 1):
        s += 1

    return s, d

# ---------- phases ----------
def phase1(m, hero, hpos):
    print("\n--- PHASE 1: hail-mary forum candidates ---")
    out = []
    for off, inner, src in HAIL_MARY:
        raw = m.u64(hero + off)
        if not raw or not is_heap(raw):
            print(f"  hero+0x{off:04X} {'inner' if inner else 'direct':6s} [{src:40s}] bad raw ptr")
            continue
        ai = m.u64(raw + 0x10) if inner else raw
        if not ai or not is_heap(ai):
            print(f"  hero+0x{off:04X} {'inner' if inner else 'direct':6s} [{src:40s}] bad inner ptr")
            continue
        s, d = score(m, ai, hpos)
        print(f"  hero+0x{off:04X} {'inner' if inner else 'direct':6s} [{src:40s}] ai=0x{ai:X} score={s}/10 "
              f"spos_dist={d.get('spos_dist','-')} vel={d.get('velocity')} segs={d.get('seg_count')}")
        out.append({"off": off, "inner": inner, "src": src, "ai_ptr": ai, "score": s, "detail": d})
    out.sort(key=lambda r: -r["score"])
    return out

def phase2(m, hero, hpos):
    print(f"\n--- PHASE 2: brute scan hero+0x{SCAN_LO:X}..0x{SCAN_HI:X} (both direct & inner) ---")
    hits = []
    for off in range(SCAN_LO, SCAN_HI, 8):
        raw = m.u64(hero + off)
        if not raw or not is_heap(raw): continue
        # direct
        s, d = score(m, raw, hpos)
        if s >= 5:
            hits.append({"off": off, "inner": False, "ai_ptr": raw, "score": s, "detail": d})
        # inner +0x10
        inner = m.u64(raw + 0x10)
        if inner and is_heap(inner):
            s2, d2 = score(m, inner, hpos)
            if s2 >= 5:
                hits.append({"off": off, "inner": True, "ai_ptr": inner, "score": s2, "detail": d2})
    hits.sort(key=lambda r: -r["score"])
    for h in hits[:15]:
        d = h["detail"]
        print(f"  hero+0x{h['off']:04X} {'inner' if h['inner'] else 'direct':6s} score={h['score']}/10 "
              f"spos_dist={d.get('spos_dist','-')} vel={d.get('velocity')} segs={d.get('seg_count')}")
    return hits

def timeseries(m, heroes, off, inner, samples=24, interval=0.25):
    print(f"\n--- PHASE 3: time-sample {samples} frames @ {interval}s using hero+0x{off:X} {'inner' if inner else 'direct'} ---")
    frames = []
    for t in range(samples):
        frame = {"t": round(t*interval, 2), "heroes": {}}
        for h in heroes:
            raw = m.u64(h["ptr"] + off)
            if not raw or not is_heap(raw):
                frame["heroes"][h["name"]] = None; continue
            ai = m.u64(raw + 0x10) if inner else raw
            if not ai or not is_heap(ai):
                frame["heroes"][h["name"]] = None; continue
            pos  = m.vec3(h["ptr"] + POSITION)
            spos = m.vec3(ai + AI_SERVER_POS)
            tA   = m.vec3(ai + AI_TARGET_A)
            tB   = m.vec3(ai + AI_TARGET_B)
            vel  = m.f32(ai + AI_VELOCITY)
            im   = m.u8(ai + AI_IS_MOVING)
            seg  = m.i32(ai + AI_SEGMENTS_COUNT)
            frame["heroes"][h["name"]] = {
                "pos": pos, "spos": spos, "tA": tA, "tB": tB,
                "vel": vel, "moving": im, "segs": seg,
            }
        frames.append(frame)
        time.sleep(interval)
    # Quick diagnostic: did target change across frames?
    first = heroes[0]["name"]
    targs_A = [f["heroes"].get(first, {}) and f["heroes"][first] and f["heroes"][first].get("tA") for f in frames]
    targs_B = [f["heroes"].get(first, {}) and f["heroes"][first] and f["heroes"][first].get("tB") for f in frames]
    mov     = [f["heroes"].get(first, {}) and f["heroes"][first] and f["heroes"][first].get("moving") for f in frames]
    uniq_A = len({t for t in targs_A if t})
    uniq_B = len({t for t in targs_B if t})
    print(f"  {first}: unique tA values over {samples} frames = {uniq_A}, tB = {uniq_B}, moving values = {set(mov)}")
    return frames

# ---------- main ----------
def main():
    pid = find_pid()
    if not pid: print("League not running"); return
    base, msize = find_base(pid)
    if not base: print("module base failed (Vanguard?)"); return
    print(f"PID={pid} base=0x{base:X} size=0x{msize:X} (expected 0x{MOD_SIZE_EXPECTED:X})")
    if msize != MOD_SIZE_EXPECTED:
        print("WARNING: module size mismatch — offsets likely stale")

    m = Mem(pid)
    arr = m.u64(base + HERO_ARRAY_RVA)
    if not is_heap(arr): print("hero_array read failed"); return

    heroes = []
    for i in range(10):
        hp = m.u64(arr + i*8)
        if not is_heap(hp): continue
        name = m.string(hp + CHAMPION_NAME, 40) or f"h{i}"
        pos  = m.vec3(hp + POSITION)
        if not pos or not is_map(pos): continue
        heroes.append({"ptr": hp, "name": name, "pos": pos})
        print(f"  [{i}] {name:15s} 0x{hp:X}  pos=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})")

    if not heroes: print("no valid heroes"); return
    tgt = heroes[0]
    print(f"\nProbing against {tgt['name']} @ 0x{tgt['ptr']:X}")

    p1 = phase1(m, tgt["ptr"], tgt["pos"])
    p2 = phase2(m, tgt["ptr"], tgt["pos"])

    combined = sorted(p1 + p2, key=lambda r: -r["score"])
    print("\n--- TOP 5 WINNERS (combined) ---")
    for w in combined[:5]:
        d = w["detail"]
        print(f"  hero+0x{w['off']:04X} {'inner' if w['inner'] else 'direct':6s} score={w['score']}/10  "
              f"spos_dist={d.get('spos_dist','-')}")

    frames = []
    if combined and combined[0]["score"] >= 6:
        w = combined[0]
        # Test winner against all 10 heroes — if it works for Garen, it should work for everyone
        frames = timeseries(m, heroes, w["off"], w["inner"])
    else:
        print("\nNo candidate scored >=6. Phase 3 reverse heap scan not implemented here.")
        print("Recommend:  1) verify replay is PLAYING (not paused), heroes moving")
        print("            2) if still nothing, implement heap-wide Vec3 match + intersect")

    out = {
        "pid": pid, "base": f"0x{base:X}", "mod_size": f"0x{msize:X}",
        "heroes": [{"name": h["name"], "ptr": f"0x{h['ptr']:X}", "pos": h["pos"]} for h in heroes],
        "phase1": [{**r, "ai_ptr": f"0x{r['ai_ptr']:X}"} for r in p1],
        "phase2_top15": [{**r, "ai_ptr": f"0x{r['ai_ptr']:X}"} for r in p2[:15]],
        "winner": ({**combined[0], "ai_ptr": f"0x{combined[0]['ai_ptr']:X}"} if combined else None),
        "frames": frames,
    }
    path = r"C:\Users\daniz\aimanager_probe_out.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {path}")

    m.close()

if __name__ == "__main__":
    main()
