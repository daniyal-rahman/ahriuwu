"""Follow documented AiManager pointer-chain conventions from the UC forum
thread (pages 651, April 2026) for patch 16.8.766.

Tests three slot conventions:
  A. hero + 0x4030  -> direct ptr to AiManager   (ibrahimcelik, 2026-04-07)
  B. hero + 0x4038  -> direct ptr to AiManager   (older convention)
  C. hero + 0x41A8  -> direct ptr to AiManager   (sq834960394 CN, 2026-04-08)
  D. hero + 0x41F0  -> LeagueObfuscation wrapper ptr; real AiManager = *(wrapper+0x10)

Also validates the new NamePlayer = 0x4328 reading vs our scanner's 0x4360
and tries position at 0x200 AND 0x25C.

For each surviving candidate:
  1. Read ServerPos @ AiMgr+0x474 — must match hero.pos within ~10u
  2. Continuously sample TargetPos, PathStart, PathEnd, Velocity, IsMoving,
     SegmentsCount over play from gt≈36 to 52. Record values.

Output: C:\\tmp\\aimgr_chain.json with per-candidate sample series.
"""
import ctypes, struct, subprocess, sys, json, time, os, ssl, urllib.request
from ctypes import wintypes
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

FOCUSED_HERO_PTR_RVA = 0x1E13490
POSITION_CANDS      = [0x200, 0x25C]
NAME_CANDS          = [0x4328, 0x4360]
AIMGR_SLOT_CANDS    = [("0x4030_direct", 0x4030, None),
                       ("0x4038_direct", 0x4038, None),
                       ("0x41A8_direct", 0x41A8, None),
                       ("0x41F0_wrap+10", 0x41F0, 0x10)]

INT_OFFSETS = {
    "TargetPos":     0x034,
    "Velocity":      0x318,
    "IsMoving_u8":   0x31C,
    "CurrSeg_u32":   0x320,
    "PathStart":     0x330,
    "PathEnd":       0x33C,
    "SegmentsPtr":   0x348,
    "SegmentsCount": 0x350,
    "DashSpeed":     0x360,
    "IsDashing_u8":  0x384,
    "ServerPos":     0x474,
    "MoveVec3":      0x480,
}

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
    def u8(self, a):  d = self.read(a, 1); return d[0] if d else None
    def u32(self, a): d = self.read(a, 4); return struct.unpack("<I", d)[0] if len(d)==4 else None
    def u64(self, a): d = self.read(a, 8); return struct.unpack("<Q", d)[0] if len(d)==8 else None
    def vec3(self, a): d = self.read(a, 12); return struct.unpack("<fff", d) if len(d)==12 else (0,0,0)
    def name(self, a): return self.read(a, 32).split(b"\x00",1)[0].decode("ascii", "replace")

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def _post(obj):
    req = urllib.request.Request("https://127.0.0.1:2999/replay/playback",
            data=json.dumps(obj).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def _get():
    with urllib.request.urlopen("https://127.0.0.1:2999/replay/playback", context=_ctx, timeout=2) as r:
        return json.loads(r.read())

def resolve_aimgr(m, hero, slot, inner):
    """Resolve an AiManager candidate. Returns (ok, aimgr_addr, note)."""
    slot_val = m.u64(hero + slot)
    if not slot_val or not (0x10000000000 < slot_val < 0x7FF000000000):
        return False, None, f"slot@+0x{slot:X} not heap (val=0x{slot_val or 0:X})"
    if inner is None:
        return True, slot_val, "direct"
    # obfuscated wrapper: real = *(slot_val + inner)
    inner_val = m.u64(slot_val + inner)
    if not inner_val or not (0x10000000000 < inner_val < 0x7FF000000000):
        return False, None, f"wrap@0x{slot_val:X}+0x{inner:X} not heap (val=0x{inner_val or 0:X})"
    return True, inner_val, f"wrap->+0x{inner:X}"

def fmt_vec3(v):
    return f"({v[0]:.0f},{v[1]:.0f},{v[2]:.0f})"

def main():
    pid = find_pid(); base, _ = find_base(pid); m = Mem(pid)
    print(f"PID={pid} base=0x{base:X}")

    # --- Seek to 36 paused, probe name + position offsets ---
    print("\nSeeking to gt=36 paused for static checks...")
    _post({"time": 36.0, "speed": 1.0, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = _get()
        if not st["seeking"] and st["paused"]: break
    _post({"speed": 1.0, "paused": False}); time.sleep(0.3); _post({"speed": 1.0, "paused": True})
    time.sleep(0.3)
    hero = m.u64(base + FOCUSED_HERO_PTR_RVA)
    print(f"hero = 0x{hero:X}")

    print("\nName offsets:")
    for o in NAME_CANDS:
        print(f"  +0x{o:X}: {m.name(hero + o)!r}")
    print("Position offsets:")
    for o in POSITION_CANDS:
        v = m.vec3(hero + o); print(f"  +0x{o:X}: {fmt_vec3(v)}")

    ref_pos = m.vec3(hero + 0x200)
    print(f"\nReference position = {fmt_vec3(ref_pos)}")

    print("\nAiManager slot candidates:")
    valid_cands = []
    for label, slot, inner in AIMGR_SLOT_CANDS:
        ok, addr, note = resolve_aimgr(m, hero, slot, inner)
        if not ok:
            print(f"  {label:>18}: SKIP  ({note})")
            continue
        spos = m.vec3(addr + INT_OFFSETS["ServerPos"])
        tpos = m.vec3(addr + INT_OFFSETS["TargetPos"])
        isM = m.u8(addr + INT_OFFSETS["IsMoving_u8"])
        cnt = m.u32(addr + INT_OFFSETS["SegmentsCount"])
        # Validate: ServerPos must be within 40u of ref_pos to count as genuine AiManager
        sp_match = abs(spos[0] - ref_pos[0]) < 40 and abs(spos[2] - ref_pos[2]) < 40
        tag = "VALID" if sp_match else "reject"
        print(f"  {label:>18}: {tag}  addr=0x{addr:X} note={note} ServerPos={fmt_vec3(spos)} TargetPos={fmt_vec3(tpos)} IsMoving={isM} SegCnt={cnt}")
        if sp_match:
            valid_cands.append((label, slot, inner, addr))

    if not valid_cands:
        print("\nNo AiManager candidates validated by ServerPos match. Aborting.")
        return

    # --- Continuous sampling ---
    print(f"\n{len(valid_cands)} valid AiManager candidates — sampling continuously from gt=34 to 52")
    _post({"time": 34.0, "speed": 1.0, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = _get()
        if not st["seeking"] and st["paused"]: break
    _post({"speed": 1.0, "paused": False})

    series = {label: [] for label, *_ in valid_cands}
    t_end = time.time() + 22.0
    step = 0.25
    last_print = time.time()
    while time.time() < t_end:
        t0 = time.time()
        try: gt = _get()["time"]
        except: gt = None
        hero_now = m.u64(base + FOCUSED_HERO_PTR_RVA)
        pos = m.vec3(hero_now + 0x200)
        for label, slot, inner, _ in valid_cands:
            ok, addr, _n = resolve_aimgr(m, hero_now, slot, inner)
            if not ok:
                series[label].append({"gt": gt, "pos": pos, "resolved": False})
                continue
            rec = {"gt": gt, "pos": pos, "resolved": True, "aimgr": hex(addr)}
            for name, off in INT_OFFSETS.items():
                if "u8" in name: rec[name] = m.u8(addr + off)
                elif "u32" in name: rec[name] = m.u32(addr + off)
                elif "Ptr" in name: rec[name] = hex(m.u64(addr + off) or 0)
                else: rec[name] = m.vec3(addr + off)
            series[label].append(rec)
        if time.time() - last_print > 2.0:
            print(f"  gt={gt:.2f} pos={fmt_vec3(pos)}")
            last_print = time.time()
        sl = step - (time.time() - t0)
        if sl > 0: time.sleep(sl)
    _post({"speed": 1.0, "paused": True})

    out = {"series": series, "hero_at_start": hex(hero)}
    with open(r"C:\tmp\aimgr_chain.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> C:\\tmp\\aimgr_chain.json")
    # summary: for each candidate, show TargetPos at key moments
    for label in series:
        print(f"\n--- {label} ---")
        for rec in series[label]:
            if not rec["resolved"]: continue
            gt = rec["gt"]
            if gt is None: continue
            if int(gt*2) != int((gt-step)*2):  # ~every 0.5s
                tp = rec.get("TargetPos")
                sp = rec.get("ServerPos")
                cnt = rec.get("SegmentsCount")
                im = rec.get("IsMoving_u8")
                tp_s = fmt_vec3(tp) if tp else "-"
                sp_s = fmt_vec3(sp) if sp else "-"
                print(f"  gt={gt:6.2f} ServerPos={sp_s} TargetPos={tp_s} IsMoving={im} SegCnt={cnt}")

if __name__ == "__main__":
    main()
