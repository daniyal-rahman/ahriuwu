"""Merged single-session pipeline: launch + mem-scrape + click-extract + record + post-process.

Replaces three separate game launches (pass1_scrape, pass2_record, garen_clicks_vtable)
with one. Key invariants:

  - NO backwards seek. Recording starts at whatever gt we're at when ready
    (~30s in, after fountain). The first ~30s of fountain footage is lost,
    but that's fine for ML and identify needs hero motion anyway.
  - One cam-lock recipe (no pre-seek + post-seek dance).
  - Click identify runs DURING recording, against the same game state we record,
    so click-dest heap pointers stay valid.
  - Cam pos is synthesized from hero pos (cam-lock invariant: cam ≈ hero + tilt offset).

Output goes to <REPLAY_OUTPUT>/<MATCH_ID>_merged/ — separate from baseline runs.

For fast iteration testing: --duration 300 caps recording at 5 min game-time.

Usage (via schtasks /IT for cam-lock keypress to reach session 1):
    python scripts/pipeline_merged.py --game-id 5547184086 \\
        --match-id NA1_5547184086 --team blue --slot 0 \\
        --champion Garen --duration 300
"""
import argparse, json, os, sys, time, glob, threading, ctypes, struct
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

# ─── Click-dest geometry ───
# Patch-dependent offsets come from `pipeline.OFFSETS` (loaded from
# scripts/offsets_<patch>.json by scan_offsets.py). Hardcoding them here was
# the previous source of "spellbook verify failed" warnings — the JSON has
# 0x30D0 / 0xB38 for 16.9.772, this file used to have 0x3120 / 0xAE0 (16.8).
def _O():  # late-binds to P.OFFSETS so post-import overrides are picked up
    return P.OFFSETS
VEC3_OFF_FROM_VPTR  = 0x14   # internal struct geometry, not patch-dependent
SLOT_CD_EXPIRE_OFF  = 0x30
SLOT_TOTAL_CD_OFF   = 0x74
SLOT_NAMES          = ["Q","W","E","R","D","F"]
DELTA_UNITS         = 50.0
POLL_MS             = 30
IDENTIFY_S          = 25

# Cam-lock invariant: when cam locked to champion, cam ≈ hero + (TILT_DX, _, TILT_DZ).
# Measured from baseline pass1 (no recording, real cam): median dx=26, dz=-1271.
# CAM_Y is constant when locked.
TILT_DX = 26.0
TILT_DZ = -1290.0
CAM_Y   = 1911.0

# ═════════════════════════════════════════════════════════════
# Win32 process memory (separate read handle from Mem class —
# we need region enumeration for vtable scan, which Mem doesn't expose)
# ═════════════════════════════════════════════════════════════
_k = ctypes.windll.kernel32

class _MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong), ("__a", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong)]
_k.VirtualQueryEx.argtypes  = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_MBI), ctypes.c_size_t]
_k.VirtualQueryEx.restype   = ctypes.c_size_t

_MEM_COMMIT  = 0x1000
_MEM_PRIVATE = 0x20000
_PAGE_RW     = 0x04 | 0x08 | 0x40   # PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE

def _regions(h, max_size=64*1024*1024):
    """Heap regions only: committed + private + RW. Skipping EXECUTE_READ /
    READONLY avoids vtable-literal false positives from code/rdata pages."""
    # Re-assert argtypes — pipeline.py's find_heroes_by_name_scan changes the
    # global VirtualQueryEx.argtypes to its local MBI; we need _MBI here.
    _k.VirtualQueryEx.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.POINTER(_MBI), ctypes.c_size_t]
    addr = 0; mbi = _MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            break
        base = mbi.BaseAddress or 0; size = mbi.RegionSize
        if (mbi.State == _MEM_COMMIT and mbi.Type == _MEM_PRIVATE
                and (mbi.Protect & _PAGE_RW) and size <= max_size):
            yield base, size
        addr = base + size
        if addr <= base: break

def _read(h, addr, n):
    buf = ctypes.create_string_buffer(n); rd = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(rd))
    return buf.raw[:rd.value] if ok else b""

def _u64(h, a):    b = _read(h, a, 8);  return struct.unpack("<Q", b)[0] if len(b)==8 else 0
def _f32(h, a):    b = _read(h, a, 4);  return struct.unpack("<f", b)[0] if len(b)==4 else None
def _vec3(h, a):
    b = _read(h, a, 12)
    return struct.unpack("<fff", b) if len(b)==12 else None

def _cstr(h, a, n=64):
    b = _read(h, a, n)
    if not b: return None
    end = b.find(b"\x00"); end = len(b) if end < 0 else end
    try: return b[:end].decode("utf-8", "replace")
    except: return None

def _valid_vec(v):
    """Click-dest Vec3 sanity: on-map x/z, terrain-height y. Tight y filter
    (45..65) is critical to drop non-click-dest objects in the heap."""
    if not v: return False
    x, y, z = v
    return 100 < x < 15000 and 45 < y < 65 and 100 < z < 15000

def _read_slot_spell_name(h, slot_ptr):
    info = _u64(h, slot_ptr + _O()["slot_spell_info"])
    if not info: return None
    nm = _u64(h, info + _O()["spell_name_ptr"])
    return _cstr(h, nm) if nm else None

def _read_active_spell_name(h, hero_ptr):
    sp = _u64(h, hero_ptr + _O()["active_spell"])
    if not sp: return None
    info = _u64(h, sp + _O()["spell_info"])
    if not info: return None
    nm = _u64(h, info + _O()["spell_name_ptr"])
    return _cstr(h, nm) if nm else None

_TRIPLE_MIRROR_B = 0x308   # SB
_TRIPLE_MIRROR_C = 0x374   # SC

def _find_hero_strict(h, champion):
    """Strict hero-pointer discovery (port of garen_clicks_vtable.find_hero_by_name).
    Scans heap for `champion + '\\0'` at +0x4368, validates via position with the
    TIGHT y-range (45..65) — pipeline.py's init_heroes uses a loose y range
    (-300..300) and can pick up a stale serialization-buffer copy of the hero
    struct whose +0x3120 spellbook is wrong (e.g. resolves to VexE for slot E)."""
    needle = champion.encode() + b"\x00"
    for base, size in _regions(h):
        data = _read(h, base, size)
        if not data: continue
        off = 0
        while True:
            j = data.find(needle, off)
            if j < 0: break
            cand = base + j - _O()["champion_name"]
            pos = _vec3(h, cand + _O()["position"])
            if pos and 100 < pos[0] < 15000 and 100 < pos[2] < 15000 and 45 < pos[1] < 65:
                nm = _read(h, cand + _O()["champion_name"], len(needle))
                if nm == needle:
                    return cand
            off = j + 1
    return 0

def _verify_hero_via_spellbook(h, hero_ptr, champion):
    """Read slot 0 (Q) spell name; live Garen → 'GarenQ', stale buffer → 'VexQ' etc."""
    slot_array_addr = hero_ptr + _O()["spellbook"] + _O()["slot_array"]
    slot_ptr = _u64(h, slot_array_addr)
    if not slot_ptr: return False
    nm = _read_slot_spell_name(h, slot_ptr)
    return bool(nm) and nm.lower().startswith(champion.lower())

def _vtable_scan(h, target_vptr):
    """Return [(vec3_addr, vec3), ...] for click-dest objects: vptr matches
    AND vec3 is on-map AND vec3 is mirrored at +0x308 / +0x374 (the click-dest
    class signature). Without the triple-mirror filter the candidate pool gets
    flooded by stale vptr literals on other heap objects."""
    target_b = struct.pack("<Q", target_vptr)
    hits = []
    for base, size in _regions(h):
        data = _read(h, base, size)
        if not data: continue
        pos = 0
        while True:
            i = data.find(target_b, pos)
            if i < 0: break
            vec3_addr = base + i + VEC3_OFF_FROM_VPTR
            v = _vec3(h, vec3_addr)
            pos = i + 8
            if not _valid_vec(v): continue
            x, y, z = v
            bb = _read(h, vec3_addr + _TRIPLE_MIRROR_B, 12)
            bc = _read(h, vec3_addr + _TRIPLE_MIRROR_C, 12)
            if len(bb) != 12 or len(bc) != 12: continue
            xb, yb, zb = struct.unpack("<fff", bb)
            xc, yc, zc = struct.unpack("<fff", bc)
            if (abs(x-xb) < 0.01 and abs(y-yb) < 0.01 and abs(z-zb) < 0.01
                    and abs(x-xc) < 0.01 and abs(z-zc) < 0.01):
                hits.append((vec3_addr, v))
    return hits

# ═════════════════════════════════════════════════════════════
# Click extract thread (identify + tight poll, combined)
# ═════════════════════════════════════════════════════════════
def click_extract_thread(h, hero_ptr, target_vptr, champion, mem_data, stop, results,
                         identify_s=IDENTIFY_S):
    """
    Phase A — IDENTIFY (first identify_s seconds):
        Periodically scan for vtable matches, accumulate candidate history,
        correlate against mem_data's hero positions, owner-filter by parent+0x68.

    Phase B — POLL (until stop_event):
        Read each watched vec3 at ~33Hz, emit click events on >50u jumps.
        Read spellbook slots for cd_expire jumps (Q/W/E/R/D/F casts).
        Read active_spell name for recall detection.

    All output goes into results['clicks'] and results['casts'] — caller waits
    until stop_event is set.
    """
    # ───── Phase A: identify ─────
    cand_hist = defaultdict(list)
    next_scan = time.time()
    t0 = time.time()
    print(f"  [click] phase A: identify ({identify_s}s, scanning every 4s, vtable=0x{target_vptr:X})...", flush=True)
    scan_idx = 0
    while time.time() - t0 < identify_s and not stop.is_set():
        now = time.time()
        if now >= next_scan:
            gt = mem_data[-1].get("gt", 0) if mem_data else 0
            scan_n = 0
            for vec3_addr, v in _vtable_scan(h, target_vptr):
                if _valid_vec(v):
                    cand_hist[vec3_addr].append((gt, v))
                    scan_n += 1
            scan_idx += 1
            print(f"  [click] scan #{scan_idx} @gt={gt:.1f}: {scan_n} valid hits, {len(cand_hist)} unique addrs", flush=True)
            next_scan = now + 2.0
        time.sleep(0.2)

    # Score candidates against the mem-thread's hero history.
    hero_hist = []
    for s in mem_data:
        h_g = (s.get("heroes") or {}).get(champion) or {}
        p = h_g.get("pos")
        if p and len(p) >= 2:
            hero_hist.append((s.get("gt", 0), p))
    scores = []
    for addr, hist in cand_hist.items():
        if len(hist) < 3 or len(set(v for _,v in hist)) < 2:
            continue
        dists, simul = [], 0
        for gt_c, v_c in hist:
            future = [p for gt,p in hero_hist if gt_c <= gt <= gt_c + 15.0]
            if future:
                dists.append(min(((p[0]-v_c[0])**2 + (p[1]-v_c[2])**2)**0.5 for p in future))
            same = [p for gt,p in hero_hist if abs(gt - gt_c) < 1.0]
            if same and ((same[0][0]-v_c[0])**2 + (same[0][1]-v_c[2])**2)**0.5 < 50:
                simul += 1
        if not dists: continue
        avg = sum(dists)/len(dists); sr = simul/len(hist)
        if sr < 0.4 and avg < 1500:
            scores.append((addr, avg, len(hist)))
    scores.sort(key=lambda r: r[1])
    print(f"  [click] cand_hist: {len(cand_hist)} addrs, scores: {len(scores)} pass-filter, hero_hist: {len(hero_hist)}", flush=True)

    # Re-verify hero_ptr: heap state shifts during the 25s identify window
    # (engine reallocates hero structs after recording starts). The ptr passed
    # in was valid at gt=~6; by now (gt=~30) it may resolve to another champion.
    try:
        if not _verify_hero_via_spellbook(h, hero_ptr, champion):
            print(f"  [click] WARN: spellbook verify failed (offsets may be stale on this patch)", flush=True)
    except Exception as e:
        print(f"  [click] WARN: spellbook verify raised {type(e).__name__}: {e}", flush=True)

    # Owner-filter (structural disambiguator): keep only candidates whose
    # parent+0x68 == hero_ptr.
    # On 16.9.772 click-dest objects are EPHEMERAL (created/freed each click),
    # so most won't appear in 3+ scans → scores filter rejects them. Bypass
    # the history filter and apply owner filter to ALL unique addrs from
    # cand_hist, then poll the survivors.
    owned = [(a, avg, n) for (a, avg, n) in scores
             if _u64(h, a + (_O()["click_owner_offset"] - VEC3_OFF_FROM_VPTR)) == hero_ptr]
    if not owned:
        # Fallback: directly owner-filter all unique cand_hist addrs
        for addr, hist in cand_hist.items():
            if _u64(h, addr + (_O()["click_owner_offset"] - VEC3_OFF_FROM_VPTR)) == hero_ptr:
                gt_, v_ = hist[-1]
                owned.append((addr, 0.0, len(hist)))
        if owned:
            print(f"  [click] owner-filter on cand_hist: {len(owned)} survivor(s)", flush=True)
        elif scores:
            print(f"  [click] WARN: no owner-matched candidate; falling back to top scored", flush=True)
            owned = scores[:1]
        else:
            # Last resort: watch ALL cand_hist addrs (may include other units' clicks).
            # Better than no clicks at all — Phase B will detect movement.
            print(f"  [click] WARN: no scored or owner-matched; watching ALL {len(cand_hist)} cand_hist addrs", flush=True)
            owned = [(addr, 0.0, len(hist)) for addr, hist in cand_hist.items()]
    watched = [a for a,*_ in owned]
    results["watched"] = [hex(a) for a in watched]
    print(f"  [click] watching {len(watched)} addr(s): {results['watched']}", flush=True)
    if not watched:
        print(f"  [click] no candidates — skipping poll phase", flush=True)
        return

    # Resolve slot spell names so D/F get tagged Flash/Ignite/TP/etc.
    slot_array_addr = hero_ptr + _O()["spellbook"] + _O()["slot_array"]
    slot_real_names = {}
    for i, name in enumerate(SLOT_NAMES):
        sp = _u64(h, slot_array_addr + i*8)
        slot_real_names[name] = _read_slot_spell_name(h, sp) if sp else None

    # ───── Phase B: tight poll ─────
    all_clicks = results["clicks"]
    all_casts  = results["casts"]
    prev_vec   = {a: _vec3(h, a) for a in watched}
    prev_cd    = {n: None for n in SLOT_NAMES}
    prev_active = None
    last_recall_t = -10.0
    print(f"  [click] phase B: tight-poll {len(watched)} addr(s) at {1000//POLL_MS}Hz", flush=True)
    while not stop.is_set():
        try:
            gt = mem_data[-1].get("gt", 0) if mem_data else 0
            hero_pos = _vec3(h, hero_ptr + _O()["position"])
            for addr in watched:
                v = _vec3(h, addr)
                if not _valid_vec(v): continue
                prev = prev_vec.get(addr)
                if prev:
                    d = ((v[0]-prev[0])**2 + (v[2]-prev[2])**2) ** 0.5
                    if d > DELTA_UNITS:
                        all_clicks.append({
                            "game_t": gt, "addr": hex(addr),
                            "x": v[0], "y": v[1], "z": v[2], "delta": round(d,1),
                            "hero_x": hero_pos[0] if hero_pos else None,
                            "hero_z": hero_pos[2] if hero_pos else None,
                        })
                prev_vec[addr] = v
            for i, name in enumerate(SLOT_NAMES):
                sp = _u64(h, slot_array_addr + i*8)
                if not sp: continue
                cd_e = _f32(h, sp + SLOT_CD_EXPIRE_OFF)
                cd_t = _f32(h, sp + SLOT_TOTAL_CD_OFF)
                if cd_e is None or cd_t is None: continue
                prev = prev_cd[name]
                if prev is not None and cd_e - prev > max(1.0, cd_t * 0.5):
                    all_casts.append({
                        "game_t": gt, "slot": name,
                        "spell_name": slot_real_names.get(name),
                        "hero_x": hero_pos[0] if hero_pos else None,
                        "hero_z": hero_pos[2] if hero_pos else None,
                        "cd_expire": cd_e, "total_cd": cd_t,
                    })
                prev_cd[name] = cd_e
            active_name = _read_active_spell_name(h, hero_ptr)
            if active_name and active_name != prev_active:
                if "recall" in active_name.lower() and gt - last_recall_t > 1.0:
                    all_casts.append({
                        "game_t": gt, "slot": "B",
                        "spell_name": active_name,
                        "hero_x": hero_pos[0] if hero_pos else None,
                        "hero_z": hero_pos[2] if hero_pos else None,
                    })
                    last_recall_t = gt
            prev_active = active_name
        except Exception:
            pass
        time.sleep(POLL_MS / 1000.0)

# ═════════════════════════════════════════════════════════════
# Main flow
# ═════════════════════════════════════════════════════════════
def run(game_id, match_id, cam_key, champion, duration, staging_dir, fps_mult=2,
        py_cores=None, speed=2.0, start_gt=None, no_clicks=False):
    print(f"\n=== MERGED PIPELINE: launch + lock + (record || mem || click) ===", flush=True)
    overall_start = time.time()
    P.CHAMPION = champion

    # 1. Launch
    P.kill_game()
    if not P.launch_replay(game_id):
        return None
    time.sleep(5)

    # 1b. Optional pre-seek
    if start_gt is not None:
        print(f"  Seeking to gt={start_gt:.1f}...")
        try:
            P.replay_post("/replay/playback", {"time": float(start_gt), "speed": 0.0, "paused": True})
            time.sleep(2.0)
            P.replay_post("/replay/playback", {"speed": 1.0, "paused": False})
            time.sleep(1.0)
            P.replay_post("/replay/playback", {"speed": 0.0, "paused": True})
            time.sleep(0.5)
            cur = P.replay_get("/replay/playback")
            print(f"  Seek complete; cur_gt={cur.get('time',0):.1f}")
        except Exception as e:
            print(f"  Seek error: {e}")

    # 2. Memory setup
    pid = P.find_league_pid()
    if not pid: print("  no PID"); P.kill_game(); return None
    base, mod_size = P.find_module_base(pid)
    if not base: print("  no module base"); P.kill_game(); return None
    if mod_size != P.EXPECTED_MOD_SIZE:
        print(f"  ABORT: module size 0x{mod_size:X} != 0x{P.EXPECTED_MOD_SIZE:X}")
        P.kill_game(); return None
    m = P.Mem(pid)
    if m.read(base, 2) != b'MZ':
        print("  RPM verify failed"); m.close(); P.kill_game(); return None
    gt_rva = P.verify_game_time(m, base)
    if not gt_rva:
        print("  GameTime RVA garbage"); m.close(); P.kill_game(); return None
    hero_ptrs = P.init_heroes(m, base)
    if champion not in hero_ptrs:
        print(f"  {champion} not found in {list(hero_ptrs.keys())}")
        m.close(); P.kill_game(); return None
    hero_ptr_int = hero_ptrs[champion]["ptr"] if isinstance(hero_ptrs[champion], dict) else hero_ptrs[champion]
    print(f"  PID={pid} base=0x{base:X}  hero=0x{hero_ptr_int:X}  heroes={list(hero_ptrs.keys())}")

    # 2b. Verify hero_ptr via spellbook — pipeline.py's init_heroes can return a
    # stale buffer (loose y-range _looks_live) whose +0x3120 spellbook is wrong.
    # Symptom: every cast spell_name comes back as another champion (e.g. VexE).
    # Fix: if slot 0 doesn't resolve to <champion>Q, re-scan with strict y-range.
    h_verify = _k.OpenProcess(0x0410, False, pid)
    try:
        if _verify_hero_via_spellbook(h_verify, hero_ptr_int, champion):
            print(f"  hero_ptr verified via spellbook ({champion}Q resolves)", flush=True)
        else:
            # On patch 16.9 spellbook offset is unverified — skip strict re-scan,
            # trust the byte-scan-validated ptr (champion_name match + valid pos).
            print(f"  WARN: spellbook verify failed (offsets may be stale on this patch); continuing with byte-scan ptr 0x{hero_ptr_int:X}", flush=True)
    except Exception as e:
        print(f"  WARN: spellbook verify raised {type(e).__name__}: {e}; continuing", flush=True)
    try: _k.CloseHandle(h_verify)
    except: pass

    # 3. Pin cores
    n_cores = os.cpu_count() or 16
    if py_cores is None: py_cores = [n_cores - 1, n_cores - 2]
    league_cores = [c for c in range(n_cores) if c not in set(py_cores)]
    P.pin_league(league_cores)
    try:
        import psutil; psutil.Process().cpu_affinity(py_cores)
    except Exception as e:
        print(f"  WARN: could not pin python: {e}")
    print(f"  pinned: League={league_cores}  Python={py_cores}")

    # 4. Open separate process handle for click extractor (needs region enum)
    h_click = _k.OpenProcess(0x0410, False, pid)

    # 5. Start recording FIRST — the /replay/recording call resets cam state
    # (selection + lock), so any earlier cam-lock recipe gets nuked. Pause the
    # game, post recording with NO backwards seek (startTime ≈ current_gt),
    # then run the cam-lock recipe last.
    P.replay_post("/replay/playback", {"paused": True})
    time.sleep(0.3)
    pb = P.replay_get("/replay/playback")
    cur_gt = pb.get("time", 0) or 0
    mem_gt = m.f32(base + gt_rva) or 0
    cur_gt = max(cur_gt, mem_gt)
    rec_start = cur_gt + 0.5
    rec_end   = cur_gt + duration + 5
    print(f"  cur_gt={cur_gt:.1f} (api={pb.get('time',0)}, mem={mem_gt:.1f})")
    os.makedirs(staging_dir, exist_ok=True)
    for f in glob.glob(os.path.join(staging_dir, "**", "*.png"), recursive=True):
        try: os.remove(f)
        except: pass
    rec = P.replay_post("/replay/recording", {
        "recording": True, "path": staging_dir.replace("\\","/"), "codec": "png",
        "framesPerSecond": P.FPS * fps_mult,
        "startTime": rec_start, "endTime": rec_end, "enforceFrameRate": True,
    })
    rec_started = time.time()
    print(f"  recording: {rec.get('width')}x{rec.get('height')}  gt={rec_start:.1f}..{rec_end:.1f}  at +{rec_started-overall_start:.1f}s")
    time.sleep(2.0)  # let recording state settle while paused

    # 6. Cam-lock recipe AFTER recording starts (recording call resets selection)
    P.replay_post("/replay/render", {"interfaceAll": True, "selectionName": champion})
    time.sleep(0.5)
    P.replay_post("/replay/playback", {"speed": speed, "paused": False})
    time.sleep(1.0)
    P.focus_game(); time.sleep(0.5)
    P.lock_camera(cam_key)
    time.sleep(0.5)
    P.focus_game(); P.lock_camera(cam_key)
    time.sleep(0.5)
    print(f"  cam locked (key={cam_key}) at {speed}x, post-recording-start")

    # 6b. Re-verify hero_ptr post cam-lock — recording + cam-lock recipe can
    # trigger heap reallocations that shift hero structs around.
    h_re = _k.OpenProcess(0x0410, False, pid)
    try:
        if not _verify_hero_via_spellbook(h_re, hero_ptr_int, champion):
            print(f"  WARN: post-lock spellbook verify failed (offsets may be stale on this patch)", flush=True)
    except Exception as e:
        print(f"  WARN: post-lock spellbook verify raised {type(e).__name__}: {e}", flush=True)
    try: _k.CloseHandle(h_re)
    except: pass

    # 7. Start mem + click + cam threads
    # Cam thread polls /replay/render in parallel with recording. Earlier note
    # ("replay_render_stale_during_recording") claimed the API goes stale, but
    # validate_cam_addr.py re-test showed api_var_rec=2938 (api moved fine).
    # We capture API alongside synth here so we can A/B compare.
    stop = threading.Event()
    mem_data = []
    cam_data_api = []
    click_results = {"clicks": [], "casts": [], "watched": []}
    target_vptr = base + _O()["click_vtable_rva"]

    mt = threading.Thread(target=P._mem_loop,
        args=(m, base, hero_ptrs, gt_rva, stop, mem_data), daemon=True)
    cam_t = threading.Thread(target=P._cam_loop,
        args=(stop, cam_data_api), daemon=True)
    ct = None
    if not no_clicks:
        ct = threading.Thread(target=click_extract_thread,
            args=(h_click, hero_ptr_int, target_vptr, champion, mem_data, stop, click_results),
            daemon=True)
    mt.start(); cam_t.start()
    if ct: ct.start()
    else:  print(f"  click thread DISABLED (--no-clicks)", flush=True)
    print(f"  threads up at +{time.time()-overall_start:.1f}s")

    # 8. Wait for recording to finish (or stall)
    max_wait = duration / max(speed, 1.0) + 120
    last_gt = 0; stall = 0
    t0 = time.time()
    while time.time() - t0 < max_wait:
        time.sleep(5)
        try:
            r = P.replay_get("/replay/recording")
            if r and not r.get("recording", False):
                print(f"  recording complete at wall={time.time()-t0:.0f}s"); break
        except:
            if P.find_league_pid() is None:
                print("  game exited"); break
        if mem_data:
            cur_gt = mem_data[-1].get("gt", 0)
            if cur_gt > 0 and abs(cur_gt - last_gt) < 2.0:
                stall += 1
                if stall >= 4:
                    print(f"  game-time stalled at gt={cur_gt:.0f}s"); break
            else: stall = 0
            last_gt = cur_gt
    else:
        print(f"  recording timeout ({max_wait:.0f}s)")

    # 9. Stop threads, collect data
    stop.set()
    mt.join(timeout=5); cam_t.join(timeout=5)
    if ct: ct.join(timeout=5)
    end_t = time.time()
    print(f"  click thread captured {len(click_results['clicks'])} clicks, {len(click_results['casts'])} casts")
    print(f"  api cam thread captured {len(cam_data_api)} samples")

    # 10. Build a synth-cam stream too (for A/B against api). Drop NaN — hero
    # pos can be NaN during respawn frames and would crash project() in post.
    import math
    cam_data_synth = []
    for s in mem_data:
        h_g = (s.get("heroes") or {}).get(champion) or {}
        p = h_g.get("pos")
        if not p or len(p) < 2: continue
        if not (math.isfinite(p[0]) and math.isfinite(p[1])): continue
        cam_data_synth.append({"wall": s["wall"],
                         "cx": round(p[0] + TILT_DX, 1), "cy": CAM_Y,
                         "cz": round(p[1] + TILT_DZ, 1)})

    # Default cam_data = api stream (real engine cam, drops smoothing-aware
    # samples). If API came back empty for some reason, fall back to synth so
    # post_process can still run — but log it loudly so we notice.
    if cam_data_api:
        cam_data = cam_data_api
        print(f"  using API cam ({len(cam_data_api)} samples) for post_process")
    else:
        cam_data = cam_data_synth
        print(f"  WARN: API cam empty — falling back to synth ({len(cam_data_synth)} samples)")

    # 11. Stats
    pngs = sorted(glob.glob(os.path.join(staging_dir, "**", "*.png"), recursive=True))
    stats = {"wall_total_s": round(end_t - overall_start, 1),
             "rec_wall_s":   round(end_t - rec_started, 1),
             "frames_recorded": len(pngs),
             "n_mem": len(mem_data), "n_cam": len(cam_data),
             "n_clicks": len(click_results["clicks"]),
             "n_casts": len(click_results["casts"])}
    if len(mem_data) >= 2:
        ws = [s["wall"] for s in mem_data]
        gts = [s.get("gt",0) for s in mem_data]
        stats["mem_hz"] = round(len(mem_data)/(ws[-1]-ws[0]), 1) if ws[-1]>ws[0] else 0
        stats["mem_max_gap"] = round(max(ws[i+1]-ws[i] for i in range(len(ws)-1)), 4)
        stats["effective_speed"] = round((gts[-1]-gts[0])/(ws[-1]-ws[0]), 2) if ws[-1]>ws[0] else 0

    print(f"\n  === stats ===")
    for k, v in stats.items(): print(f"    {k} = {v}")

    # Cleanup
    m.close()
    try: _k.CloseHandle(h_click)
    except: pass
    P.kill_game()

    return {"mem_data": mem_data, "cam_data": cam_data,
            "cam_data_api": cam_data_api, "cam_data_synth": cam_data_synth,
            "click_results": click_results,
            "stats": stats, "rec_start_gt": rec_start}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id",   required=True)
    ap.add_argument("--match-id",  required=True)
    ap.add_argument("--team",      required=True, choices=["blue","red"])
    ap.add_argument("--slot",      type=int, required=True)
    ap.add_argument("--champion",  default="Garen")
    ap.add_argument("--duration",  type=int, default=300,
                    help="game-time seconds to record (default 300 = 5 min for fast iteration)")
    ap.add_argument("--out-suffix", default="_merged")
    ap.add_argument("--fps-mult",  type=int,   default=2)
    ap.add_argument("--speed",     type=float, default=2.0)
    ap.add_argument("--py-cores",  default=None)
    ap.add_argument("--start-gt",  type=float, default=None,
                    help="Seek replay to this game-time before recording starts")
    ap.add_argument("--no-clicks", action="store_true",
                    help="Disable click extraction thread (A/B test for engine stall)")
    args = ap.parse_args()

    py_cores = [int(x) for x in args.py_cores.split(",")] if args.py_cores else None
    cam_key  = P.cam_key_for(args.team, args.slot)
    out_dir  = os.environ.get("REPLAY_OUTPUT", r"C:\tmp\replay_data")
    temp_dir = os.environ.get("REPLAY_TEMP",   r"C:\tmp\_pipeline_temp")
    test_match_id = f"{args.match_id}{args.out_suffix}"
    game_dir = os.path.join(out_dir, test_match_id)
    staging  = os.path.join(temp_dir, test_match_id)
    os.makedirs(game_dir, exist_ok=True); os.makedirs(staging, exist_ok=True)

    print(f"=== {args.match_id} {args.team} slot={args.slot} key={cam_key} duration={args.duration}s ===")
    out = run(args.game_id, test_match_id, cam_key, args.champion, args.duration, staging,
              fps_mult=args.fps_mult, py_cores=py_cores, speed=args.speed,
              start_gt=args.start_gt, no_clicks=args.no_clicks)
    if not out or not out["stats"]["frames_recorded"]:
        print("FAIL: no frames"); return 1

    # Persist raw data
    with open(os.path.join(game_dir, "raw_mem.json"), "w") as f:
        json.dump(out["mem_data"], f)
    with open(os.path.join(game_dir, "raw_cam.json"), "w") as f:
        json.dump(out["cam_data"], f)
    # Persist both streams for offline diff/inspection
    with open(os.path.join(game_dir, "raw_cam_api.json"), "w") as f:
        json.dump(out.get("cam_data_api") or [], f)
    with open(os.path.join(game_dir, "raw_cam_synth.json"), "w") as f:
        json.dump(out.get("cam_data_synth") or [], f)

    # Snapshot per-frame mtimes BEFORE post_process cleans staging.
    # Each PNG's mtime ≈ engine-emit wall-time; gives us per-frame wall-stamp
    # so we can verify whether engine renders at uniform game-time intervals.
    pngs = sorted(glob.glob(os.path.join(staging, "**", "*.png"), recursive=True))
    frame_meta = []
    for i, p in enumerate(pngs):
        try:
            st = os.stat(p)
            frame_meta.append({"frame": i, "path": os.path.basename(p),
                               "mtime": st.st_mtime, "size": st.st_size})
        except Exception as e:
            frame_meta.append({"frame": i, "path": os.path.basename(p), "error": str(e)})
    with open(os.path.join(game_dir, "frame_mtimes.json"), "w") as f:
        json.dump(frame_meta, f)
    print(f"  wrote frame_mtimes.json ({len(frame_meta)} frames)")
    with open(os.path.join(game_dir, "merged_stats.json"), "w") as f:
        json.dump(out["stats"], f, indent=2)
    click_out = {
        "champion": args.champion,
        "watched_addrs": out["click_results"]["watched"],
        "total_clicks": len(out["click_results"]["clicks"]),
        "clicks":       out["click_results"]["clicks"],
        "total_casts":  len(out["click_results"]["casts"]),
        "casts":        out["click_results"]["casts"],
    }
    with open(os.path.join(game_dir, "clicks.json"), "w") as f:
        json.dump(click_out, f, indent=2)
    print(f"  wrote raw_mem.json, raw_cam.json, clicks.json, merged_stats.json")

    # Post-process (resize + label)
    game_info = {"match_id": test_match_id, "champion": args.champion,
                 "team": args.team, "slot": args.slot, "cam_key": cam_key}
    print("\n--- Post-Processing ---")
    P.post_process(test_match_id, out["mem_data"], out["cam_data"], game_info, staging,
                   rec_start=out["rec_start_gt"])
    print(f"\nDONE → {game_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
