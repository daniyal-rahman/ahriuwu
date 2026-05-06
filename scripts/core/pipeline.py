"""
Replay Data Pipeline — 2 Pass + Overlapped Post-Processing
==========================================================
Pass 1: Memory scrape (50Hz) + camera API poll (93Hz persistent HTTPS), 2x speed
        + click/cast extraction in a third thread
        → determines real game duration from game time stall
        → cam samples timestamped via wall→gt interpolation from mem thread
Pass 2: Record video (720p PNG, 40fps enforced at 2x = 20fps game-time)
        → uses real duration from pass 1 as endTime
Post:   Resize 720→352² (cv2, capped at 2 threads), build labels.json, delete originals
        → runs in a separate process, overlapped with the next game's pass 1

Runs on Windows with:
  - League client open (for LCU replay launch)
  - Vanguard disabled (for ReadProcessMemory)
  - game.cfg: Width=1280, Height=720, WindowMode=1, EnableReplayApi=1

Usage:
    python pipeline.py --manifest <path> --champion <Champ> --batch
    python pipeline.py --game-id 5554195441 --match-id NA1_5554195441 \
                       --team blue --slot 0 --champion Garen

Output per game:  $REPLAY_OUTPUT\\{match_id}\\
    frames\\  → 352×352 PNGs at 20fps
    labels.json → per-frame labels (720p screen coords, null for unlabeled)
    clicks.json → click + cast events (game_t, world coords)
    raw_mem.json + raw_cam.json → raw scrape (overlay.py uses these)

Structured log: $REPLAY_OUTPUT\\pipeline.jsonl
Memory offsets: loaded from scripts/offsets_<patch>.json (no hardcoded fallbacks).
"""
import argparse, base64, bisect, ctypes, ctypes.wintypes as wt
import glob, http.client, json, math, multiprocessing as mp, os, shutil
import socket, struct, subprocess, sys, threading, time, traceback
import urllib.request, urllib.error, ssl
from pynput.keyboard import Controller as KbController

try:
    import psutil
except ImportError:
    psutil = None

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
SCREEN_W, SCREEN_H = 1280, 720
FRAME_SZ = 352
FPS = 20

# Camera projection (cam-locked spectator follow at 1280x720, empirically fit
# from ground-truth probe; same shape for any focused champion).
CAM_Y = 1912.0
CAM_Z_OFFSET = -1292.0
FLOOR_Y = 52.0
TILT = math.radians(56.0)
FOV_V = math.radians(40.0)
FOV_H = 2 * math.atan(math.tan(FOV_V / 2) * SCREEN_W / SCREEN_H)
TAN_H, TAN_V = math.tan(FOV_H / 2), math.tan(FOV_V / 2)
COS_T, SIN_T = math.cos(TILT), math.sin(TILT)

# ── Memory offsets ──
# Sole source of truth: scripts/offsets_<patch>.json emitted by scan_offsets.py.
# No hardcoded defaults — if a field is missing from JSON, runtime use raises a
# loud KeyError (via _StrictOffsets) telling the user to re-scan.
EXPECTED_MOD_SIZE = 0x20E0000

# Fields the rest of pipeline.py / pipeline_merged.py actively reads. The loader
# warns about any of these missing from the JSON, but does NOT silently substitute
# a default. Re-run `python scripts/scan_offsets.py --champion <Champ>` (potentially
# with --wide) to derive them.
_REQUIRED_FIELDS = {
    "hero_array", "hero_array_layout", "hero_array_stride",
    "game_time",
    "click_vtable_rva", "click_owner_offset",
    "position", "hp_current", "hp_max",
    "gold_current", "gold_earned",
    "active_spell", "champion_name", "level", "vision_score",
    "spellbook", "slot_array", "slot_spell_info",
    "spell_name_ptr",
    "spell_info", "cast_target",
}

class _StrictOffsets(dict):
    """Like dict but a missing key raises a loud, actionable KeyError instead of
    silently returning a stale hardcoded default."""
    def __missing__(self, key):
        raise KeyError(
            f"⚠ Offset {key!r} is not in offsets_<patch>.json. "
            f"Re-run scan_offsets.py (with --champion <ChampInThisReplay>) to derive it. "
            f"There are no hardcoded defaults — this read would have produced wrong values."
        )

def _load_offsets():
    """Load scripts/offsets_<patch>.json (newest mtime, matching EXPECTED_MOD_SIZE).

    No fallbacks to hardcoded defaults. If JSON is missing fields, WARN (don't
    abort) so the user can re-run the scanner to widen the search. Reading any
    missing field at runtime raises a loud KeyError.

    Returns (OFFSETS, versions, fallbacks, missing_from_scan) where:
      - OFFSETS is a _StrictOffsets dict (KeyError on miss)
      - fallbacks is the set of REQUIRED fields not present in the JSON
      - missing_from_scan is the scanner-emitted _missing list (for context)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # offsets JSONs are emitted by scan_offsets.py into scripts/; this file lives
    # in scripts/core/. Search both so either layout works.
    search_dirs = [here, os.path.dirname(here)]
    candidates = []
    seen = set()
    for d in search_dirs:
        try:
            for n in os.listdir(d):
                if n.startswith("offsets_") and n.endswith(".json"):
                    p = os.path.join(d, n)
                    if p in seen:
                        continue
                    seen.add(p)
                    candidates.append((os.path.getmtime(p), p, n))
        except OSError:
            continue
    candidates.sort(reverse=True)
    for _, path, name in candidates:
        try:
            with open(path) as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[offsets] {name}: {e}", flush=True); continue
        size = raw.get("_patch_mod_size") or raw.get("_mod_size")
        if size and size != EXPECTED_MOD_SIZE:
            print(f"[offsets] {name}: skipping (mod_size 0x{size:X} != EXPECTED 0x{EXPECTED_MOD_SIZE:X})", flush=True)
            continue
        loaded = {}
        for k, v in raw.items():
            if k.startswith("_"): continue
            if isinstance(v, dict) and "value" in v:
                loaded[k] = v["value"]
            else:
                loaded[k] = v
        versions = raw.get("_offset_versions", {})
        scanned = raw.get("_scanned_at") or "?"
        patch = raw.get("_patch") or "?"
        missing = sorted(k for k in _REQUIRED_FIELDS if k not in loaded)
        missing_from_scan = list(raw.get("_missing", []))
        print(f"[offsets] loaded {len(loaded)} fields from {name} (patch={patch}, scanned={scanned})", flush=True)
        if missing:
            print(f"[offsets] ⚠ {len(missing)} REQUIRED field(s) NOT in JSON — runtime reads will raise:", flush=True)
            for f in missing:
                print(f"[offsets]     ⚠ {f}", flush=True)
            print(f"[offsets] To fix: re-run `python scripts/scan_offsets.py --champion <ChampInReplay>` "
                  f"(scanner has been widened — current ranges should cover most patch shifts)", flush=True)
        if missing_from_scan:
            print(f"[offsets] (context) scanner reported {len(missing_from_scan)} field(s) MISSING: {missing_from_scan}", flush=True)
        return _StrictOffsets(loaded), versions, set(missing), missing_from_scan
    print("[offsets] ⚠ no scripts/offsets_*.json matched mod_size — runtime reads will all raise.", flush=True)
    print("[offsets] To fix: open a replay on this patch + run scan_offsets.py to emit a JSON.", flush=True)
    return _StrictOffsets(), {}, set(_REQUIRED_FIELDS), []

OFFSETS, OFFSET_VERSIONS, OFFSET_FALLBACKS, OFFSET_SCAN_MISSING = _load_offsets()

OUTPUT_BASE  = os.environ.get("REPLAY_OUTPUT", r"E:\replay_data")
TEMP_BASE    = os.environ.get("REPLAY_TEMP", r"E:\tmp\_pipeline")
LOCKFILE     = r"C:\Riot Games\League of Legends\lockfile"
JSONL_PATH   = os.path.join(OUTPUT_BASE, "pipeline.jsonl")

# ── Alarm thresholds ──
ALARM_MIN_SPEED    = 1.5   # effective replay speed
ALARM_MIN_REC_FPS  = 25    # effective recording fps
ALARM_MIN_MEM_HZ   = 40    # mem thread Hz
MAX_MEM_GAP        = 0.1   # 100ms — for label matching
MAX_CAM_GAP        = 0.1

# ── CPU core allocation ──
# Pass 1 leaves the last 2 cores free for the post-process worker; pass 2 takes
# all cores (PNG encoder scales linearly up to ~14 threads).
_N_CORES = os.cpu_count() or 6
PASS1_CORES = list(range(_N_CORES - 2)) if _N_CORES >= 6 else list(range(_N_CORES))
POST_CORES  = list(range(max(0, _N_CORES - 2), _N_CORES))
ALL_CORES   = list(range(_N_CORES))

_ctx = ssl.create_default_context()
_ctx.check_hostname = False; _ctx.verify_mode = ssl.CERT_NONE
_k = ctypes.windll.kernel32
_kb = KbController()

# ── Click-dest internal struct geometry (not patch-dependent) ──
# Fixed by the click-dest C++ class layout, stable across patches:
#   parent_struct
#     +0x00  vptr  → module + click_vtable_rva
#     +0x14  Vec3  click destination (DST_A)  ← we read this
#     +0x14+0x308  Vec3  mirror DST_B
#     +0x14+0x374  Vec3  mirror DST_C  (z only — y replaced)
#     +0x68  hero owner pointer (matches our hero_ptr)
# Triple-mirror filter eliminates the ~150 stale-vptr literals scattered through
# the heap. Owner pointer disambiguates 5 surviving candidates → 1 per hero.
VEC3_OFF_FROM_VPTR  = 0x14
_TRIPLE_MIRROR_B    = 0x308
_TRIPLE_MIRROR_C    = 0x374
# Spellbook slot internal layout (per slot, indexed via slot_array + i*8):
#   +0x30  cooldown_expire   ← jumps forward when slot is cast
#   +0x74  total_cooldown
#   +0x130 spell_info ptr
SLOT_CD_EXPIRE_OFF  = 0x30
SLOT_TOTAL_CD_OFF   = 0x74
SLOT_NAMES          = ["Q", "W", "E", "R", "D", "F"]

# Click-extraction tunables
CLICK_DELTA_UNITS = 50.0   # min Vec3 jump (XZ) to count as a click
CLICK_POLL_MS     = 30     # tight-poll cadence
CLICK_IDENTIFY_S  = 25     # phase-A duration (heap-scan + correlate)

# ── Champion selector (overridden via --champion) ──
CHAMPION = "Garen"    # default for back-compat with existing Garen manifests

def _is_session0():
    """Returns True if this process is running in Windows session 0 (services),
    i.e., spawned via SSH. In that case direct pynput keys can't reach session 1
    windows and we must bounce via `schtasks /IT`."""
    try:
        k = ctypes.windll.kernel32
        session_id = ctypes.c_ulong(0)
        k.ProcessIdToSessionId(k.GetCurrentProcessId(), ctypes.byref(session_id))
        return session_id.value == 0
    except Exception:
        return False

# ═══════════════════════════════════════════════════════════════
# Structured Logging
# ═══════════════════════════════════════════════════════════════
def log_event(path, **fields):
    fields["ts"] = time.time()
    fields["host"] = socket.gethostname()
    with open(path, "a") as f:
        f.write(json.dumps(fields) + "\n")

def check_alarm(label, value, threshold):
    """Print loud alarm when `value` falls below `threshold`. Returns True if alarmed."""
    if value < threshold:
        print(f"\n  *** ALARM: {label} = {value:.1f} (threshold {threshold}) ***\n", flush=True)
        return True
    return False

# ═══════════════════════════════════════════════════════════════
# API Helpers
# ═══════════════════════════════════════════════════════════════
def replay_get(ep):
    with urllib.request.urlopen(
        urllib.request.Request(f"https://127.0.0.1:2999{ep}"),
        context=_ctx, timeout=5
    ) as r:
        return json.loads(r.read())

def replay_post(ep, data):
    with urllib.request.urlopen(
        urllib.request.Request(f"https://127.0.0.1:2999{ep}",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}),
        context=_ctx, timeout=5
    ) as r:
        return json.loads(r.read())

def lcu_auth():
    parts = open(LOCKFILE).read().strip().split(":")
    return parts[2], f"Basic {base64.b64encode(f'riot:{parts[3]}'.encode()).decode()}"

def lcu_post(ep, body=None):
    port, auth = lcu_auth()
    req = urllib.request.Request(
        f"https://127.0.0.1:{port}{ep}", method="POST",
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": auth, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, context=_ctx, timeout=10) as r:
        raw = r.read()
        return json.loads(raw) if raw else None

# ═══════════════════════════════════════════════════════════════
# Memory Reader
# ═══════════════════════════════════════════════════════════════
class Mem:
    def __init__(self, pid):
        self.h = _k.OpenProcess(0x0410, False, pid)
        if not self.h: raise OSError(f"OpenProcess({pid}) failed")
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u64(self, a):
        d = self.read(a, 8); return struct.unpack('<Q', d)[0] if d else None
    def u32(self, a):
        d = self.read(a, 4); return struct.unpack('<I', d)[0] if d else None
    def f32(self, a):
        d = self.read(a, 4); return struct.unpack('<f', d)[0] if d else None
    def vec3(self, a):
        d = self.read(a, 12); return struct.unpack('<fff', d) if d else None
    def string(self, a, n=32):
        d = self.read(a, n)
        if not d: return None
        t = d.split(b'\x00')[0]
        try: return t.decode('ascii')
        except: return None
    def close(self): _k.CloseHandle(self.h)


# ═══════════════════════════════════════════════════════════════
# Click extraction
# ═══════════════════════════════════════════════════════════════
# Click-dest objects live wherever the engine's allocator dropped them — not
# at any module-static offset. We re-walk RW-private heap regions via
# VirtualQueryEx (the Mem class doesn't expose this, hence the _raw_* helpers).
class _MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong), ("__a", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong)]
_MEM_COMMIT  = 0x1000
_MEM_PRIVATE = 0x20000
_PAGE_RW     = 0x04 | 0x08 | 0x40   # PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE


def _click_regions(h, max_size=64 * 1024 * 1024):
    """Yield (base, size) for committed + private + RW heap regions only.
    Skipping EXECUTE_READ / READONLY avoids vtable-literal false positives
    from code/rdata pages."""
    _k.VirtualQueryEx.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.POINTER(_MBI), ctypes.c_size_t]
    _k.VirtualQueryEx.restype = ctypes.c_size_t
    addr = 0
    mbi = _MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            break
        base = mbi.BaseAddress or 0
        size = mbi.RegionSize
        if (mbi.State == _MEM_COMMIT and mbi.Type == _MEM_PRIVATE
                and (mbi.Protect & _PAGE_RW) and size <= max_size):
            yield base, size
        addr = base + size
        if addr <= base:
            break


def _raw_read(h, addr, n):
    buf = ctypes.create_string_buffer(n)
    rd = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(rd))
    return buf.raw[:rd.value] if ok else b""


def _raw_u64(h, a):
    b = _raw_read(h, a, 8)
    return struct.unpack("<Q", b)[0] if len(b) == 8 else 0


def _raw_f32(h, a):
    b = _raw_read(h, a, 4)
    return struct.unpack("<f", b)[0] if len(b) == 4 else None


def _raw_vec3(h, a):
    b = _raw_read(h, a, 12)
    return struct.unpack("<fff", b) if len(b) == 12 else None


def _raw_cstr(h, a, n=64):
    b = _raw_read(h, a, n)
    if not b:
        return None
    end = b.find(b"\x00")
    end = len(b) if end < 0 else end
    try:
        return b[:end].decode("utf-8", "replace")
    except Exception:
        return None


def _click_valid_vec(v):
    """Click-dest Vec3 sanity: on-map x/z, terrain-height y. Tight y filter
    (45..65) drops non-click-dest objects."""
    if not v:
        return False
    x, y, z = v
    return 100 < x < 15000 and 45 < y < 65 and 100 < z < 15000


def _vtable_scan(h, target_vptr):
    """Return [(vec3_addr, vec3), ...] for click-dest objects: vptr matches
    AND vec3 is on-map AND vec3 is mirrored at +0x308 / +0x374. The triple-
    mirror filter is critical — without it, ~150 stale vptr literals on other
    heap objects flood the candidate pool."""
    target_b = struct.pack("<Q", target_vptr)
    hits = []
    for base, size in _click_regions(h):
        data = _raw_read(h, base, size)
        if not data:
            continue
        pos = 0
        while True:
            i = data.find(target_b, pos)
            if i < 0:
                break
            vec3_addr = base + i + VEC3_OFF_FROM_VPTR
            v = _raw_vec3(h, vec3_addr)
            pos = i + 8
            if not _click_valid_vec(v):
                continue
            x, y, z = v
            bb = _raw_read(h, vec3_addr + _TRIPLE_MIRROR_B, 12)
            bc = _raw_read(h, vec3_addr + _TRIPLE_MIRROR_C, 12)
            if len(bb) != 12 or len(bc) != 12:
                continue
            xb, yb, zb = struct.unpack("<fff", bb)
            xc, yc, zc = struct.unpack("<fff", bc)
            if (abs(x - xb) < 0.01 and abs(y - yb) < 0.01 and abs(z - zb) < 0.01
                    and abs(x - xc) < 0.01 and abs(z - zc) < 0.01):
                hits.append((vec3_addr, v))
    return hits


def click_extract_thread(h, hero_ptr, target_vptr, champion, mem_data, stop, results):
    """Two-phase click + cd-jump extractor running alongside _mem_loop.

    Phase A — IDENTIFY (CLICK_IDENTIFY_S seconds):
        Periodically heap-scan for vtable matches, accumulate candidate Vec3
        addresses, then owner-filter by `parent + click_owner_offset == hero_ptr`.

    Phase B — POLL (until stop_event):
        Tight-poll watched Vec3s at CLICK_POLL_MS — emit "click" events on
        >CLICK_DELTA_UNITS jumps. Also poll each spellbook slot's cooldown_expire
        — emit "cast" events on forward jumps (catches Q dash, E channel, summoners,
        items — anything `active_spell` misses). Also detects recall via active_spell
        name change.

    Output goes into results['clicks'] and results['casts'] (caller-allocated lists).
    """
    from collections import defaultdict
    o = OFFSETS  # already loaded module-global
    target_b_ptr = target_vptr
    cand_hist = defaultdict(list)

    next_scan = time.time()
    t0 = time.time()
    print(f"  [click] phase A: identify ({CLICK_IDENTIFY_S}s, scan every 2s, "
          f"vtable=0x{target_b_ptr:X})", flush=True)
    scan_idx = 0
    while time.time() - t0 < CLICK_IDENTIFY_S and not stop.is_set():
        now = time.time()
        if now >= next_scan:
            gt = mem_data[-1].get("gt", 0) if mem_data else 0
            scan_n = 0
            for vec3_addr, v in _vtable_scan(h, target_b_ptr):
                if _click_valid_vec(v):
                    cand_hist[vec3_addr].append((gt, v))
                    scan_n += 1
            scan_idx += 1
            print(f"  [click] scan #{scan_idx} @gt={gt:.1f}: "
                  f"{scan_n} hits, {len(cand_hist)} unique addrs", flush=True)
            next_scan = now + 2.0
        time.sleep(0.2)

    if stop.is_set():
        print("  [click] stopped during phase A", flush=True)
        return

    # Owner-filter: keep only candidates whose parent+click_owner_offset == hero_ptr.
    # On 16.9.772 click-dest objects are EPHEMERAL — many appear once, never again.
    # We can't require >=3 history hits; just owner-filter every unique addr.
    owner_off = o["click_owner_offset"]
    owner_off_rel = owner_off - VEC3_OFF_FROM_VPTR  # parent+0x68 = vec3_addr+0x54
    owned = []
    for addr, hist in cand_hist.items():
        if _raw_u64(h, addr + owner_off_rel) == hero_ptr:
            owned.append(addr)
    print(f"  [click] owner-filter: {len(owned)}/{len(cand_hist)} candidates own hero_ptr=0x{hero_ptr:X}",
          flush=True)
    if not owned:
        print("  [click] WARN: no owner-matched candidates — clicks unavailable", flush=True)
        return

    watched = owned
    results["watched"] = [hex(a) for a in watched]
    # Pre-resolve slot spell names so D/F come out as Flash/Ignite/etc.
    spellbook_off = o["spellbook"]
    slot_array_off = o["slot_array"]
    slot_array_addr = hero_ptr + spellbook_off + slot_array_off
    slot_real_names = {}
    for i, name in enumerate(SLOT_NAMES):
        sp = _raw_u64(h, slot_array_addr + i * 8)
        if sp:
            info = _raw_u64(h, sp + o["slot_spell_info"])
            if info:
                np_ = _raw_u64(h, info + o["spell_name_ptr"])
                slot_real_names[name] = _raw_cstr(h, np_) if np_ else None
            else:
                slot_real_names[name] = None
        else:
            slot_real_names[name] = None
    print(f"  [click] phase B: tight-poll {len(watched)} addr(s) at "
          f"{1000 // CLICK_POLL_MS}Hz; slot_names={slot_real_names}", flush=True)

    all_clicks = results["clicks"]
    all_casts = results["casts"]
    prev_vec = {a: _raw_vec3(h, a) for a in watched}
    prev_cd = {n: None for n in SLOT_NAMES}
    prev_active = None
    last_recall_t = -10.0
    pos_off = o["position"]
    active_spell_off = o["active_spell"]
    spell_info_off = o["spell_info"]
    spell_name_ptr_off = o["spell_name_ptr"]
    while not stop.is_set():
        try:
            gt = mem_data[-1].get("gt", 0) if mem_data else 0
            hero_pos = _raw_vec3(h, hero_ptr + pos_off)
            for addr in watched:
                v = _raw_vec3(h, addr)
                if not _click_valid_vec(v):
                    continue
                prev = prev_vec.get(addr)
                if prev:
                    d = ((v[0] - prev[0])**2 + (v[2] - prev[2])**2) ** 0.5
                    if d > CLICK_DELTA_UNITS:
                        all_clicks.append({
                            "game_t": gt, "addr": hex(addr),
                            "x": v[0], "y": v[1], "z": v[2], "delta": round(d, 1),
                            "hero_x": hero_pos[0] if hero_pos else None,
                            "hero_z": hero_pos[2] if hero_pos else None,
                        })
                prev_vec[addr] = v
            for i, name in enumerate(SLOT_NAMES):
                sp = _raw_u64(h, slot_array_addr + i * 8)
                if not sp:
                    continue
                cd_e = _raw_f32(h, sp + SLOT_CD_EXPIRE_OFF)
                cd_t = _raw_f32(h, sp + SLOT_TOTAL_CD_OFF)
                if cd_e is None or cd_t is None:
                    continue
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
            # Recall via active_spell name change
            sp = _raw_u64(h, hero_ptr + active_spell_off)
            active_name = None
            if sp:
                info = _raw_u64(h, sp + spell_info_off)
                if info:
                    np_ = _raw_u64(h, info + spell_name_ptr_off)
                    if np_:
                        active_name = _raw_cstr(h, np_)
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
        time.sleep(CLICK_POLL_MS / 1000.0)


def find_league_pid():
    r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq League of Legends.exe',
                        '/FO', 'CSV', '/NH'], capture_output=True, text=True)
    for line in r.stdout.strip().split('\n'):
        if 'league' in line.lower():
            return int(line.strip('"').split('","')[1])
    return None

def find_module_base(pid):
    class ME(ctypes.Structure):
        _fields_ = [("dwSize", ctypes.c_ulong), ("th32ModuleID", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong), ("GlblcntUsage", ctypes.c_ulong),
            ("ProccntUsage", ctypes.c_ulong), ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
            ("modBaseSize", ctypes.c_ulong), ("hModule", ctypes.c_void_p),
            ("szModule", ctypes.c_char * 256), ("szExePath", ctypes.c_char * 260)]
    snap = _k.CreateToolhelp32Snapshot(0x18, pid)
    me = ME(); me.dwSize = ctypes.sizeof(ME)
    if _k.Module32First(snap, ctypes.byref(me)):
        while True:
            if b'league' in me.szModule.lower():
                base = ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value
                size = me.modBaseSize
                _k.CloseHandle(snap)
                return base, size
            if not _k.Module32Next(snap, ctypes.byref(me)): break
    _k.CloseHandle(snap)
    return None, None

def _looks_live(m, hp):
    """Quick sanity check: hp points to a struct where +0x200 is a plausible map Vec3."""
    try:
        x, y, z = m.vec3(hp + OFFSETS["position"])
    except Exception:
        return False
    return (x == x and z == z and -300 < y < 300 and 100 < x < 16000 and 100 < z < 16000)

def _live_player_names():
    """Query /liveclientdata/playerlist for the 10 champion internal names.
    Returns ordered list [(name, team)] where team is 'blue' (first 5) or 'red'."""
    try:
        with urllib.request.urlopen("https://127.0.0.1:2999/liveclientdata/playerlist",
                                     context=_ctx, timeout=3) as r:
            players = json.loads(r.read())
        out = []
        for i, p in enumerate(players):
            # champion internal names strip apostrophes: Bel'Veth → Belveth, Kai'Sa → KaiSa
            nm = p.get("championName", "").replace("'", "")
            out.append((nm, "blue" if i < 5 else "red"))
        return out
    except Exception:
        return []

def find_heroes_by_name_scan(m, base, names):
    """Byte-scan the process heap for each champion name appearing at hero+0x4360.
    For each hit we compute candidate hero_base = hit - 0x4360 and validate via position.
    Returns {name: {ptr, slot=0, team='blue'}} for all names we found."""
    import ctypes.wintypes as wt
    class MBI(ctypes.Structure):
        _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                    ("AllocationProtect", ctypes.c_ulong), ("__a", ctypes.c_ulong),
                    ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong),
                    ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong),
                    ("__b", ctypes.c_ulong)]
    _k.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
    _k.VirtualQueryEx.restype = ctypes.c_size_t
    MEM_COMMIT = 0x1000; MEM_PRIVATE = 0x20000; PAGE_RW = 0x04 | 0x08 | 0x40

    pats = {n: (n + "\x00").encode() for n in names if n}
    found = {}
    addr = 0; mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(m.h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)): break
        rbase = mbi.BaseAddress or 0; rsize = mbi.RegionSize
        if (mbi.State == MEM_COMMIT and mbi.Type == MEM_PRIVATE and (mbi.Protect & PAGE_RW) and rsize < 256*1024*1024):
            data = m.read(rbase, rsize)
            if data:
                for name, pat in pats.items():
                    if name in found: continue
                    idx = 0
                    while True:
                        j = data.find(pat, idx)
                        if j < 0: break
                        idx = j + 1
                        hp = rbase + j - OFFSETS["champion_name"]
                        if hp > 0 and _looks_live(m, hp):
                            found[name] = hp
                            break
        addr = rbase + rsize
        if addr <= rbase: break
    return found

def init_heroes(m, base):
    """Iterate hero_array using the layout/stride from offsets JSON.
    Falls through to the other layout, then byte-scan, if the primary layout is
    stale (patch updates can flip the array between deref-of-pointer and
    inline-stride layouts; both shapes have appeared on recent patches)."""
    heroes = {}
    layout = OFFSETS.get("hero_array_layout", "inline")
    stride = OFFSETS.get("hero_array_stride", 0x108)

    def _try_inline(stride_):
        for i in range(10):
            hp = m.u64(base + OFFSETS["hero_array"] + i * stride_)
            if not hp or hp < 0x10000: continue
            name = m.string(hp + OFFSETS["champion_name"])
            if name and len(name) >= 2 and name[0].isalpha() and _looks_live(m, hp):
                heroes.setdefault(name, {"ptr": hp, "slot": i % 5, "team": "blue" if i < 5 else "red"})

    def _try_deref(stride_):
        arr_ptr = m.u64(base + OFFSETS["hero_array"])
        if not arr_ptr or not (0x10000 < arr_ptr < 0x7FFFFFFFFFFF): return
        for i in range(10):
            hp = m.u64(arr_ptr + i * stride_)
            if not hp or hp < 0x10000: continue
            name = m.string(hp + OFFSETS["champion_name"])
            if name and len(name) >= 2 and name[0].isalpha() and _looks_live(m, hp):
                heroes.setdefault(name, {"ptr": hp, "slot": i % 5, "team": "blue" if i < 5 else "red"})

    # Primary layout from offsets JSON
    if layout == "inline":
        _try_inline(stride)
    else:
        _try_deref(stride)

    # Fallback to the other layout if primary returned too few
    if len(heroes) < 6:
        if layout == "inline":
            _try_deref(8)
        else:
            _try_inline(0x108)

    # Fallback: byte-scan for champ names (needed if both array layouts are stale)
    if len(heroes) < 2:
        print(f"  hero_array returned {len(heroes)} live heroes — falling back to byte-scan", flush=True)
        players = _live_player_names()
        if players:
            names = [p[0] for p in players]
            ptrs = find_heroes_by_name_scan(m, base, names)
            for i, (name, team) in enumerate(players):
                if name in ptrs:
                    heroes[name] = {"ptr": ptrs[name], "slot": i % 5, "team": team}
            print(f"  byte-scan found {len(ptrs)}/{len(names)} hero structs", flush=True)
    return heroes

def verify_game_time(m, base):
    gt_rva = OFFSETS["game_time"]
    gt = m.f32(base + gt_rva)
    if gt is not None and 0 < gt < 10000:
        return gt_rva
    return None

# ═══════════════════════════════════════════════════════════════
# Window Focus & Camera Lock
# ═══════════════════════════════════════════════════════════════
def find_game_hwnd():
    user32 = ctypes.windll.user32; hwnds = []
    def cb(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if "league of legends" in buf.value.lower(): hwnds.append(hwnd)
        return True
    user32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(cb), 0)
    return hwnds[0] if hwnds else None

def focus_game():
    user32 = ctypes.windll.user32
    hwnd = find_game_hwnd()
    if not hwnd: return False
    user32.SystemParametersInfoW(0x2001, 0, None, 0)
    fg = user32.GetForegroundWindow()
    ft = user32.GetWindowThreadProcessId(fg, None)
    ct = _k.GetCurrentThreadId()
    user32.AttachThreadInput(ct, ft, True)
    user32.keybd_event(0x12, 0, 0, 0); user32.keybd_event(0x12, 0, 2, 0)
    user32.ShowWindow(hwnd, 9); user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    user32.AttachThreadInput(ct, ft, False)
    time.sleep(0.3)
    return True

def click_center():
    user32 = ctypes.windll.user32
    hwnd = find_game_hwnd()
    if not hwnd: return
    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    cx = (rect.left + rect.right) // 2
    cy = (rect.top + rect.bottom) // 2
    user32.SetCursorPos(cx, cy)
    time.sleep(0.1)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.05)
    user32.mouse_event(0x0004, 0, 0, 0, 0)
    time.sleep(0.2)

def lock_camera(key):
    """Cam-lock via direct pynput. WORKS only when the orchestrator runs in the
    user's interactive session (session 1). When launched from SSH (session 0)
    the keypress is silently dropped — use lock_camera_via_schtasks instead."""
    click_center()
    _kb.press(key); time.sleep(0.05); _kb.release(key)
    time.sleep(0.15)
    _kb.press(key); time.sleep(0.05); _kb.release(key)
    time.sleep(0.3)

def _cam_is_locked(hero_world, tol=200.0):
    """Return True if /replay/render cameraPosition (x, z) is within `tol` units
    of `hero_world` (x, z). Locked-cam has a small tilt offset so tolerance is
    generous."""
    try:
        cp = replay_get("/replay/render").get("cameraPosition", {})
        cx = cp.get("x"); cz = cp.get("z")
        if cx is None or cz is None: return False
        return abs(cx - hero_world[0]) < tol and abs(cz - hero_world[1]) < tol + 1500  # cam offset from tilt
    except Exception:
        return False

def lock_camera_via_schtasks(key, helper_path=r"C:\Users\daniz\lock_cam_once.py",
                              verify_fn=None, max_attempts=5):
    """Cross-session cam-lock: spawns a helper script in the interactive user
    session via `schtasks /IT`. Needed when this pipeline is launched via SSH
    (session 0 can't send keys to session 1 windows).

    If `verify_fn` is provided (called with no args, returns bool), the lock
    is retried up to `max_attempts` times until verify_fn returns True."""
    subprocess.run(['schtasks', '/End', '/TN', 'LockCam'], capture_output=True)
    subprocess.run(['schtasks', '/Delete', '/TN', 'LockCam', '/F'], capture_output=True)
    subprocess.run(['schtasks', '/Create', '/TN', 'LockCam',
                    '/TR', f'cmd.exe /c python {helper_path} {key}',
                    '/SC', 'ONCE', '/ST', '23:59', '/IT', '/F'], capture_output=True)
    for attempt in range(1, max_attempts + 1):
        subprocess.run(['schtasks', '/End', '/TN', 'LockCam'], capture_output=True)
        subprocess.run(['schtasks', '/Run', '/TN', 'LockCam'], capture_output=True)
        time.sleep(2.0)
        if verify_fn is None:
            return True
        if verify_fn():
            print(f"  cam lock verified on attempt {attempt}", flush=True)
            return True
        print(f"  cam lock attempt {attempt}/{max_attempts} failed verification, retrying...", flush=True)
    print(f"  WARN: cam lock failed after {max_attempts} attempts", flush=True)
    return False

def cam_key_for(team, slot):
    if team == "blue": return str(slot + 1)
    return "qwert"[slot - 5]

# ═══════════════════════════════════════════════════════════════
# Projection
# ═══════════════════════════════════════════════════════════════
def project(wx, wz, cx, cy, cz):
    dx = wx - cx; dy = FLOOR_Y - cy; dz = wz - cz
    vy = dy * COS_T + dz * SIN_T
    vz = -dy * SIN_T + dz * COS_T
    if vz <= 10: return None
    sx = 0.5 + (dx / vz) / TAN_H * 0.5
    sy = 0.5 - (vy / vz) / TAN_V * 0.5
    px, py = int(sx * SCREEN_W), int(sy * SCREEN_H)
    return [px, py] if 0 <= px < SCREEN_W and 0 <= py < SCREEN_H else None

def classify_spell(name, champion=None):
    """Classify a champion spell name into action type.
    Order matters: BasicAttack must be checked before Q/W/E/R since
    abilities like 'GarenQAttack' contain 'Attack'."""
    if not name: return "idle"
    nl = name.lower()
    if "basicattack" in nl: return "attack"
    if "recall" in nl: return "recall"
    if nl.startswith("summoner"): return "summoner"
    # Champion abilities: <Champion>Q, <Champion>QAttack, <Champion>W, etc.
    cname = (champion or CHAMPION).lower()
    if nl.startswith(cname):
        suffix = nl[len(cname):]
        if suffix[:1] in ("q", "w", "e", "r"):
            return "ability"
    # Item actives: typically "Item####" or numeric prefix
    if nl.startswith("item"): return "item"
    return "other"

# ═══════════════════════════════════════════════════════════════
# Pipeline Utilities
# ═══════════════════════════════════════════════════════════════
def kill_game():
    os.system('taskkill /F /IM "League of Legends.exe" >nul 2>&1')
    time.sleep(3)

def pin_league(cores):
    """Set League process CPU affinity. No-op if psutil missing or process not found."""
    if psutil is None: return
    pid = find_league_pid()
    if not pid: return
    try:
        psutil.Process(pid).cpu_affinity(cores)
        print(f"  Pinned League to cores {cores}", flush=True)
    except Exception as e:
        print(f"  Could not pin League: {e}", flush=True)

def launch_replay(game_id):
    """Tell LCU to open the replay. Returns True iff the LCU POST succeeded.
    Game-readiness (process spawned + game_time advancing) is then verified by
    pass1_scrape's pid-poll + game_time>0.5 gate — the older 240s
    /liveclientdata/gamestats poll here was redundant with that."""
    print(f"  Launching replay {game_id}...", flush=True)
    try:
        lcu_post(f"/lol-replays/v1/rofls/{game_id}/watch",
                 {"componentType": "replay"})
        return True
    except Exception as e:
        print(f"  LCU launch failed: {e}", flush=True)
        return False

def _nearest(arr, keys, gt):
    i = bisect.bisect_right(keys, gt)
    best, best_gap = None, float("inf")
    for j in (i-1, i):
        if 0 <= j < len(arr):
            gap = abs(arr[j]["gt"] - gt)
            if gap < best_gap:
                best, best_gap = arr[j], gap
    return best, best_gap

# ═══════════════════════════════════════════════════════════════
# Pass 1: Memory + Camera Scrape
# ═══════════════════════════════════════════════════════════════
def _mem_loop(m, base, hero_ptrs, gt_rva, stop, results):
    o = OFFSETS
    interval = 1.0 / 50
    while not stop.is_set():
        tick = time.perf_counter()
        gt = m.f32(base + gt_rva) if gt_rva else 0
        heroes = {}
        for name, hinfo in hero_ptrs.items():
            hp = hinfo["ptr"]
            pos = m.vec3(hp + o["position"])
            if not pos: continue
            entry = {
                "pos": [round(pos[0], 1), round(pos[2], 1)],
                "hp":         round(m.f32(hp + o["hp_current"]) or 0, 1),
                "hp_max":     round(m.f32(hp + o["hp_max"]) or 0, 1),
                "gold":       round(m.f32(hp + o["gold_current"]) or 0, 1),
                "gold_total": round(m.f32(hp + o["gold_earned"]) or 0, 1),
                "level":      m.u32(hp + o["level"]) or 0,
                # kills/assists live in a separate scoreboard struct (per stats_offset_research),
                # not the hero struct — pull from /liveclientdata/playerlist instead if needed.
            }
            # Spell detection for ALL heroes (not just Garen)
            sp = m.u64(hp + o["active_spell"])
            if sp and sp > 0x10000:
                si = m.u64(sp + o["spell_info"])
                if si and si > 0x10000:
                    np_ = m.u64(si + o["spell_name_ptr"])
                    if np_ and np_ > 0x10000:
                        sn = m.string(np_)
                        if sn and len(sn) > 2:
                            entry["spell"] = sn
                            # For BasicAttacks, the active_spell.cast_target is the cast-origin
                            # (= hero pos), not the target. The real target unit's position
                            # lives on the HERO struct at attack_target_pos (16.9.772: 0x4260).
                            is_aa = "asicAttack" in sn
                            tgt = None
                            if is_aa and "attack_target_pos" in o:
                                at = m.vec3(hp + o["attack_target_pos"])
                                if at and -2000 < at[0] < 16500 and -2000 < at[2] < 16500 and -300 < at[1] < 600:
                                    tgt = at
                            if tgt is None:
                                ct = m.vec3(sp + o["cast_target"])
                                if ct and abs(ct[0]) < 20000 and abs(ct[2]) < 20000:
                                    tgt = ct
                            if tgt is not None:
                                entry["cast_target"] = [round(tgt[0], 1), round(tgt[2], 1)]
                            else:
                                entry["cast_target"] = [round(pos[0], 1), round(pos[2], 1)]
                                entry["self_cast"] = True
            heroes[name] = entry
        results.append({"wall": time.perf_counter(), "gt": round(gt, 3), "heroes": heroes})
        elapsed = time.perf_counter() - tick
        if elapsed < interval: time.sleep(interval - elapsed)

def _cam_loop(stop, results):
    conn = http.client.HTTPSConnection("127.0.0.1", 2999, context=_ctx, timeout=5)
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 50
    try:
        while not stop.is_set():
            try:
                wall = time.perf_counter()
                conn.request("GET", "/replay/render")
                data = json.loads(conn.getresponse().read())
                cp = data.get("cameraPosition", {})
                results.append({
                    "wall": wall,
                    "cx": round(cp.get("x", 0), 1),
                    "cy": round(cp.get("y", 0), 1),
                    "cz": round(cp.get("z", 0), 1),
                })
                consecutive_errors = 0
            except (http.client.RemoteDisconnected, ConnectionResetError):
                conn = http.client.HTTPSConnection("127.0.0.1", 2999, context=_ctx, timeout=5)
                consecutive_errors += 1
            except Exception:
                consecutive_errors += 1
                time.sleep(0.01)
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"  CAM LOOP: {MAX_CONSECUTIVE_ERRORS} consecutive errors, bailing", flush=True)
                break
    finally:
        conn.close()

_EMPTY_PASS1 = ([], [], {}, {"clicks": [], "casts": [], "watched": []})


def _wait_for_gt(m, base, gt_rva, threshold, timeout=30, upper=10000):
    """Block until threshold < m.f32(base+gt_rva) < upper, or timeout elapses.
    The upper bound rejects garbage reads (NaN, denormals, huge values that
    pre-init memory can return) so this also serves as the patch-validity
    check — if no sane gt appears in `timeout` seconds, the offset is wrong
    OR the game is stuck loading."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        cur = m.f32(base + gt_rva)
        if cur is not None and threshold < cur < upper:
            return cur
        time.sleep(0.05)
    return None


def pass1_scrape(game_id, cam_key, duration, force_patch=False):
    """Memory scrape (50Hz) + camera poll (93Hz) + click/cast extraction at 2x.
    Returns (mem_data, cam_data, stats, click_results).

    Sync strategy: gate the mem thread on `game_time > 0.5` so it starts at the
    fountain entry instead of ~14s in (which is when the cam-lock dance finishes
    at 2x). Cam + click threads still wait for cam-lock to finish (cam needs to
    track the focus champion; clicks need the heap scan window). Frames in the
    cam-coverage gap (~gt 0.5..7) fall back to synthetic cam = hero_pos + tilt
    in post_process — the projection holds because cam IS hero+offset when locked."""
    print("\n--- Pass 1: Memory + Camera Scrape (2x) ---", flush=True)
    scrape_start = time.time()
    kill_game()
    if not launch_replay(game_id):
        return _EMPTY_PASS1

    # No more sleep(5) — instead poll until RPM works (PID exists + module mapped)
    pid = None
    deadline = time.time() + 30
    while time.time() < deadline:
        pid = find_league_pid()
        if pid is not None:
            break
        time.sleep(0.2)
    if not pid:
        print("  No game PID after 30s", flush=True); kill_game(); return _EMPTY_PASS1
    base, mod_size = find_module_base(pid)
    if not base:
        print("  No module base", flush=True); kill_game(); return _EMPTY_PASS1
    if mod_size != EXPECTED_MOD_SIZE:
        msg = f"Module size 0x{mod_size:X} != expected 0x{EXPECTED_MOD_SIZE:X} — patch changed?"
        if force_patch:
            print(f"  WARNING: {msg} (--force-patch)", flush=True)
        else:
            print(f"  ABORT: {msg}", flush=True)
            print(f"  Update OFFSETS or pass --force-patch.", flush=True)
            kill_game(); return _EMPTY_PASS1

    m = Mem(pid)
    if m.read(base, 2) != b'MZ':
        print("  RPM verify failed", flush=True); m.close(); kill_game(); return _EMPTY_PASS1
    print(f"  Memory: PID={pid} base=0x{base:X}", flush=True)

    # Wait until game_time at OFFSETS["game_time"] reads a sane value > 0.5s.
    # This combines patch-validity check + loading-screen skip + init_heroes
    # readiness in one gate (heroes populate around gt~0.5s too).
    gt_rva = OFFSETS["game_time"]
    gt0 = _wait_for_gt(m, base, gt_rva, 0.5, timeout=45)
    if gt0 is None:
        print(f"  ABORT: gt at 0x{gt_rva:X} never reached a sane >0.5s value in 45s "
              f"(patch shift, or game stuck loading)", flush=True)
        m.close(); kill_game(); return _EMPTY_PASS1
    print(f"  GameTime: 0x{gt_rva:X} = {gt0:.2f}s (gate cleared)", flush=True)

    hero_ptrs = init_heroes(m, base)
    if CHAMPION not in hero_ptrs:
        print(f"  ABORT: champion '{CHAMPION}' not found in this game. "
              f"Heroes detected: {list(hero_ptrs.keys())}. "
              f"Pass --champion <name> matching one of those.", flush=True)
        m.close(); kill_game(); return _EMPTY_PASS1
    print(f"  Heroes: {list(hero_ptrs.keys())}", flush=True)

    pin_league(PASS1_CORES)

    # Start mem thread BEFORE cam-lock so mem coverage begins at the fountain
    # entry (~gt 0.5-1s) instead of after the lock dance (~gt 7-14s).
    stop = threading.Event()
    mem_data, cam_data = [], []
    click_results = {"clicks": [], "casts": [], "watched": []}
    mt = threading.Thread(target=_mem_loop, args=(m, base, hero_ptrs, gt_rva, stop, mem_data), daemon=True)
    mt.start()

    # Cam-lock recipe (mem thread continues sampling through this).
    # Direct pynput — requires this process to run in session 1
    # (i.e. launched via `schtasks /IT pipeline.py ...`, not plain SSH).
    print(f"  [setup] pause replay", flush=True)
    replay_post("/replay/playback", {"paused": True})
    time.sleep(0.3)
    print(f"  [setup] select {CHAMPION}", flush=True)
    replay_post("/replay/render", {"interfaceAll": True, "selectionName": CHAMPION})
    time.sleep(0.5)
    print(f"  [setup] unpause @ 2x", flush=True)
    replay_post("/replay/playback", {"speed": 2.0, "paused": False})
    time.sleep(1.0)
    print(f"  [setup] focus + lock attempt 1", flush=True)
    focus_game(); lock_camera(cam_key)
    time.sleep(0.5)
    print(f"  [setup] focus + lock attempt 2", flush=True)
    focus_game(); lock_camera(cam_key)
    print(f"  Camera locked (key={cam_key}), 2x speed", flush=True)

    # Cam + click threads start AFTER cam-lock (cam tracks focus champ,
    # click thread's heap-scan window opens here too).
    ct = threading.Thread(target=_cam_loop, args=(stop, cam_data), daemon=True)
    target_vptr = base + OFFSETS["click_vtable_rva"]
    hero_ptr = hero_ptrs[CHAMPION]["ptr"]
    kt = threading.Thread(
        target=click_extract_thread,
        args=(m.h, hero_ptr, target_vptr, CHAMPION, mem_data, stop, click_results),
        daemon=True,
    )
    ct.start(); kt.start()

    last_gt = 0; stall = 0
    while not stop.is_set():
        time.sleep(5)
        if mem_data:
            cur_gt = mem_data[-1]["gt"]
            if cur_gt > 0 and abs(cur_gt - last_gt) < 2.0:
                stall += 1
                if stall >= 3:
                    print(f"  Game ended at gt={cur_gt:.0f}s", flush=True)
                    stop.set()
            else:
                stall = 0
            last_gt = cur_gt
            if cur_gt >= duration + 30:
                print(f"  Past duration, stopping at gt={cur_gt:.0f}s", flush=True)
                stop.set()
        if find_league_pid() is None:
            print("  Game process exited", flush=True)
            stop.set()

    mt.join(timeout=5); ct.join(timeout=5); kt.join(timeout=5)
    print(f"  Clicks: {len(click_results['clicks'])}  Casts: {len(click_results['casts'])}",
          flush=True)
    scrape_end = time.time()
    wall_span = scrape_end - scrape_start

    # Compute stats
    stats = {"scrape_start": scrape_start, "scrape_end": scrape_end}
    if len(mem_data) >= 2:
        mem_walls = [s["wall"] for s in mem_data]
        mem_span = mem_walls[-1] - mem_walls[0]
        mem_hz = len(mem_data) / mem_span if mem_span > 0 else 0
        mem_gaps = [mem_walls[i+1] - mem_walls[i] for i in range(len(mem_walls)-1)]
        gt_span = mem_data[-1]["gt"] - mem_data[0]["gt"]
        effective_speed = gt_span / mem_span if mem_span > 0 else 0
        stats.update(mem_n=len(mem_data), mem_hz=round(mem_hz, 1),
                     mem_max_gap=round(max(mem_gaps), 4), gt_span=round(gt_span, 1),
                     wall_span=round(wall_span, 1), effective_speed=round(effective_speed, 2))
        check_alarm("effective_speed", effective_speed, ALARM_MIN_SPEED)
        check_alarm("mem_hz", mem_hz, ALARM_MIN_MEM_HZ)
    if len(cam_data) >= 2:
        cam_walls = [s["wall"] for s in cam_data]
        cam_span = cam_walls[-1] - cam_walls[0]
        cam_hz = len(cam_data) / cam_span if cam_span > 0 else 0
        cam_gaps = [cam_walls[i+1] - cam_walls[i] for i in range(len(cam_walls)-1)]
        stats.update(cam_n=len(cam_data), cam_hz=round(cam_hz, 1),
                     cam_max_gap=round(max(cam_gaps), 4))

    print(f"  Memory: {len(mem_data)} @ {stats.get('mem_hz',0)}Hz (max_gap={stats.get('mem_max_gap',0)*1000:.0f}ms)", flush=True)
    print(f"  Camera: {len(cam_data)} @ {stats.get('cam_hz',0)}Hz (max_gap={stats.get('cam_max_gap',0)*1000:.0f}ms)", flush=True)
    if mem_data:
        print(f"  GT: {mem_data[0]['gt']:.0f}-{mem_data[-1]['gt']:.0f}s  speed={stats.get('effective_speed',0):.2f}x", flush=True)
    m.close(); kill_game()
    stats["n_clicks"] = len(click_results["clicks"])
    stats["n_casts"] = len(click_results["casts"])
    stats["watched"] = click_results.get("watched", [])
    return mem_data, cam_data, stats, click_results

# ═══════════════════════════════════════════════════════════════
# Pass 2: Video Recording
# ═══════════════════════════════════════════════════════════════
def pass2_record(game_id, cam_key, duration, staging_dir, rec_start=1.0, rec_end=None,
                 mem_first_gt=None, mem_last_gt=None):
    """Record game as PNGs at 2x (40fps enforced = 20fps game-time).
    PNGs are always 20fps game-rate regardless of actual wall-fps.
    rec_start: game_t to start recording (seconds). Floored to mem_first_gt+0.3
               when supplied so every recorded frame has mem coverage.
    rec_end:   game_t to stop (None → mem_last_gt-0.3 when supplied, else
               max(rec_start+1, duration-2)). Caller-provided value wins.
               Aim before game-end: gameTime freezes on the post-game screen
               so the recording API can never auto-stop past that point.
    mem_first_gt / mem_last_gt: pass1's actual mem coverage. Used to clamp
               rec_start/rec_end into the labeled range so n_unlabeled → 0.
    Writes to staging_dir. Returns (frame_count, stats)."""
    print("\n--- Pass 2: Video Recording (2x, 40fps enforced) ---", flush=True)
    record_start = time.time()
    kill_game()
    if not launch_replay(game_id):
        return 0, {}
    time.sleep(5)

    # NOTE: this first selectionName POST + speed=2.0 looks redundant with the
    # pause→select→unpause sequence that follows, but removing it has caused
    # cam-lock to silently fail in past debugging — leaving it as a safety
    # primer for the second sequence. If you "clean it up" be ready for ghosts.
    # interfaceAll=False kills the in-game HUD/scoreboard/minimap so the
    # recorded PNGs are clean (training data wants no UI overlay).
    replay_post("/replay/render", {"interfaceAll": False, "selectionName": CHAMPION})
    time.sleep(1)
    replay_post("/replay/playback", {"speed": 2.0})
    time.sleep(1.0)
    # For pass 2 we don't have hero_ptrs; attach a memory reader to verify lock.
    pid = find_league_pid()
    base, _ = find_module_base(pid) if pid else (None, None)
    m2 = Mem(pid) if pid else None
    # Cam-lock recipe: pause → select → unpause @ 2x → lock → lock again.
    replay_post("/replay/playback", {"paused": True})
    time.sleep(0.3)
    replay_post("/replay/render", {"interfaceAll": False, "selectionName": CHAMPION})
    time.sleep(0.5)
    replay_post("/replay/playback", {"speed": 2.0, "paused": False})
    time.sleep(1.0)
    focus_game(); time.sleep(0.5)
    lock_camera(cam_key)
    time.sleep(0.5)
    focus_game(); lock_camera(cam_key)
    time.sleep(0.5)
    print(f"  Camera locked (key={cam_key})", flush=True)

    # Final HUD-off nudge right before recording starts (in case prior render
    # POSTs got clobbered by the cam-lock keypresses).
    replay_post("/replay/render", {"interfaceAll": False, "selectionName": CHAMPION})
    time.sleep(0.2)

    # Pass 2 gets all cores — PNG encoder saturates everything
    pin_league(ALL_CORES)
    time.sleep(0.5)

    # Prepare staging dir
    os.makedirs(staging_dir, exist_ok=True)
    for f in glob.glob(os.path.join(staging_dir, "**", "*.png"), recursive=True):
        os.remove(f)

    # Clamp rec_start / rec_end into pass1's mem coverage so every recorded
    # frame has a mem sample within MAX_MEM_GAP. The 0.3s buffer absorbs jitter.
    rec_start_eff = rec_start
    if mem_first_gt is not None:
        rec_start_eff = max(rec_start, mem_first_gt + 0.3)
    if rec_end is None:
        if mem_last_gt is not None:
            rec_end = mem_last_gt - 0.3
        else:
            rec_end = max(rec_start_eff + 1, duration - 2)
    print(f"  Recording window: gt {rec_start_eff:.2f}..{rec_end:.2f}s "
          f"(mem covers {mem_first_gt or 0:.2f}..{mem_last_gt or 0:.2f}s)", flush=True)

    rec = replay_post("/replay/recording", {
        "recording": True,
        "path": staging_dir.replace("\\", "/"),
        "codec": "png",
        "framesPerSecond": FPS * 2,
        "startTime": rec_start_eff,
        "endTime": rec_end,
        "enforceFrameRate": True,
    })
    print(f"  Recording started: {rec.get('width')}x{rec.get('height')}", flush=True)

    # Original ordering: re-lock cam a moment after recording starts
    time.sleep(1.5)
    focus_game(); time.sleep(0.3)
    lock_camera(cam_key)
    print(f"  Re-locked after recording start", flush=True)

    # Stall-aware wait loop. Mirrors pass1: if game-time freezes (replay
    # reached game-end and the player is sitting on the post-game screen),
    # force-stop recording. Without this, the API keeps writing PNGs of the
    # frozen post-game screen until the wall timeout fires.
    max_wait = duration + 60
    t0 = time.time()
    last_gt = -1.0
    last_change = time.time()
    gt_rva = OFFSETS["game_time"]
    STALL_S = 15.0
    finished = False
    while time.time() - t0 < max_wait and not finished:
        time.sleep(5)
        try:
            r = replay_get("/replay/recording")
            if not r.get("recording", False):
                print("  Recording complete (API)", flush=True); finished = True; break
        except Exception:
            if find_league_pid() is None:
                print("  Game exited", flush=True); finished = True; break
        if m2 and base:
            cur_gt = m2.f32(base + gt_rva)
            if cur_gt and cur_gt > 0:
                if cur_gt > last_gt + 0.1:
                    last_gt = cur_gt
                    last_change = time.time()
                elif time.time() - last_change > STALL_S:
                    print(f"  Game-time frozen at gt={cur_gt:.0f}s — stopping recording", flush=True)
                    try: replay_post("/replay/recording", {"recording": False})
                    except Exception: pass
                    finished = True; break
    if not finished:
        print(f"  Recording timeout ({max_wait}s)", flush=True)

    record_end = time.time()
    wall_time = record_end - record_start
    pngs = sorted(glob.glob(os.path.join(staging_dir, "**", "*.png"), recursive=True))
    n_frames = len(pngs)
    effective_fps = n_frames / wall_time if wall_time > 0 else 0
    expected_frames = int(duration * FPS)

    stats = {
        "record_start": record_start, "record_end": record_end,
        "frames_recorded": n_frames, "duration_requested": round(duration, 1),
        "wall_time": round(wall_time, 1), "effective_fps": round(effective_fps, 1),
        "expected_frames": expected_frames,
        "rec_start_eff": round(rec_start_eff, 3), "rec_end_eff": round(rec_end, 3),
    }
    check_alarm("effective_fps", effective_fps, ALARM_MIN_REC_FPS)

    print(f"  Frames: {n_frames} (expected ~{expected_frames}), {effective_fps:.1f} wall-fps, {wall_time:.0f}s wall", flush=True)
    kill_game()
    return n_frames, stats

# ═══════════════════════════════════════════════════════════════
# Post-Processing (runs in worker process)
# ═══════════════════════════════════════════════════════════════
def _resize_worker_init():
    """Pool worker initializer — pin to POST_CORES, cap cv2 at 1 thread each."""
    import cv2
    cv2.setNumThreads(1)
    if psutil is not None:
        try:
            psutil.Process().cpu_affinity(POST_CORES)
        except Exception:
            pass

def _resize_one(args):
    """Worker function: read PNG, resize to FRAME_SZ², write to dst. Returns (idx, ok)."""
    import cv2
    src, dst = args
    img = cv2.imread(src)
    if img is None:
        try:
            shutil.copy2(src, dst)
            return (dst, False)  # fallback copy
        except Exception:
            return (dst, False)
    out = cv2.resize(img, (FRAME_SZ, FRAME_SZ), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, out)
    return (dst, True)


def post_process(match_id, mem_data, cam_data, game_info, staging_dir, rec_start=1.0,
                 champion=None, click_results=None):
    """Resize frames, build labels, clean up. Returns stats dict."""
    post_start = time.time()
    # On Windows mp.Process uses spawn → globals set in main() are NOT visible
    # in worker process. Champion MUST be passed via job, not read from CHAMPION.
    if champion is None:
        champion = game_info.get("champion")
    if not champion:
        raise SystemExit("post_process: champion is required (job didn't include one)")
    if click_results is None:
        click_results = {"clicks": [], "casts": [], "watched": []}
    game_dir = os.path.join(OUTPUT_BASE, match_id)
    frames_dir = os.path.join(game_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    pngs = sorted(glob.glob(os.path.join(staging_dir, "**", "*.png"), recursive=True))
    n_frames = len(pngs)
    print(f"\n--- Post-Processing: {n_frames} frames ({match_id}) ---", flush=True)
    if n_frames == 0:
        print("  ERROR: no frames", flush=True); return None

    # ── Resize 720p → 352×352 (parallel, 2 workers pinned to POST_CORES) ──
    print(f"  Resizing to {FRAME_SZ}x{FRAME_SZ} with 2 workers on cores {POST_CORES}...", flush=True)
    jobs = [(p, os.path.join(frames_dir, f"{i:06d}.png")) for i, p in enumerate(pngs)]
    n_resize_fail = 0
    t_resize_start = time.time()
    with mp.Pool(2, initializer=_resize_worker_init) as pool:
        done = 0
        for dst, ok in pool.imap_unordered(_resize_one, jobs, chunksize=50):
            done += 1
            if not ok:
                n_resize_fail += 1
            if done % 5000 == 0:
                print(f"    {done}/{n_frames}", flush=True)
    resize_wall = time.time() - t_resize_start
    print(f"  Resize complete in {resize_wall:.1f}s ({n_frames/resize_wall:.1f} fps)", flush=True)
    if n_resize_fail > 0:
        print(f"  WARNING: {n_resize_fail} frames failed imread — raw copies used", flush=True)

    # ── Build wall→gt map from mem samples ──
    print("  Building labels...", flush=True)
    start_gt = rec_start
    mem_walls = [s["wall"] for s in mem_data]
    mem_gts   = [s["gt"]   for s in mem_data]

    def wall_to_gt(w):
        i = bisect.bisect_right(mem_walls, w)
        if i == 0 or i >= len(mem_walls): return None
        w0, w1 = mem_walls[i-1], mem_walls[i]
        g0, g1 = mem_gts[i-1], mem_gts[i]
        if w1 == w0: return g0
        t = (w - w0) / (w1 - w0)
        return g0 + t * (g1 - g0)

    # ── Assign gt to cam samples ──
    cam_with_gt = []
    n_cam_dropped = 0
    for cs in cam_data:
        gt = wall_to_gt(cs["wall"])
        if gt is None:
            n_cam_dropped += 1; continue
        cam_with_gt.append({"gt": gt, "cx": cs["cx"], "cy": cs["cy"], "cz": cs["cz"]})
    cam_with_gt.sort(key=lambda s: s["gt"])
    cam_gt_keys = [s["gt"] for s in cam_with_gt]

    if n_cam_dropped > 0:
        print(f"  Cam outside mem range: {n_cam_dropped} dropped", flush=True)

    mem_sorted = sorted(mem_data, key=lambda s: s["gt"])
    mem_gt_keys = [s["gt"] for s in mem_sorted]

    # ── Per-frame labels ──
    # Synthetic gt = start_gt + fi/FPS — assumes the engine writes PNG #0 at
    # exactly rec_start. Frame-mtime-based gt was tried but is unreliable on
    # this codepath (pass2 doesn't currently surface PNG mtimes); leaving as
    # synthetic for now since enforceFrameRate=True keeps the spacing tight.

    frames_out = []
    n_unlabeled = 0
    n_cam_fallback = 0
    mem_gaps_ms = []

    for fi in range(n_frames):
        gt = start_gt + fi / FPS
        bm, mem_gap = _nearest(mem_sorted, mem_gt_keys, gt)
        mem_gaps_ms.append(mem_gap * 1000)

        if mem_gap > MAX_MEM_GAP:
            n_unlabeled += 1
            frames_out.append({"frame": fi, "gt": round(gt, 3), "label": None})
            continue

        heroes = bm.get("heroes", {})
        garen = heroes.get(champion, {})
        gp = garen.get("pos", [0, 0])

        bc, cam_gap = _nearest(cam_with_gt, cam_gt_keys, gt)
        if bc and cam_gap <= MAX_CAM_GAP:
            cx, cy, cz = bc["cx"], bc["cy"], bc["cz"]
        else:
            n_cam_fallback += 1
            cx, cy, cz = gp[0], CAM_Y, gp[1] + CAM_Z_OFFSET

        champ_screen = project(gp[0], gp[1], cx, cy, cz)
        visible = []
        for name, hd in heroes.items():
            p = hd.get("pos", [0, 0])
            sp = project(p[0], p[1], cx, cy, cz)
            if sp:
                visible.append({
                    "name": name,
                    "screen": sp,
                    "hp": hd.get("hp", 0),
                    "hp_max": hd.get("hp_max", 0),
                    "level": hd.get("level", 0),
                })

        spell = garen.get("spell")
        cast_target = garen.get("cast_target")
        action_screen = project(cast_target[0], cast_target[1], cx, cy, cz) if cast_target else None
        waypoint = garen.get("waypoint")
        waypoint_screen = project(waypoint[0], waypoint[1], cx, cy, cz) if waypoint else None

        # Velocity / heading patched after the per-frame loop (needs next frame's pos).
        frames_out.append({
            "frame": fi, "gt": round(gt, 3),
            "_world": gp,           # temporary for velocity calc
            "_cam": (cx, cy, cz),    # temporary for projection
            "label": {
                "champion_screen": champ_screen,
                "champion_world": gp,
                "champion_stats": {
                    "hp":       garen.get("hp", 0),
                    "hp_max":   garen.get("hp_max", 0),
                    "gold":     garen.get("gold", 0),
                    "gold_total": garen.get("gold_total", 0),
                    "level":    garen.get("level", 0),
                },
                "visible_heroes": visible,
                "action": {"type": classify_spell(spell, champion), "spell": spell, "screen": action_screen},
                "waypoint": {"world": waypoint, "screen": waypoint_screen} if waypoint else None,
            },
        })

    # ── Post-loop: compute velocity and heading from position deltas ──
    # Look ahead ~10 frames (0.5s game time) to get a stable movement direction
    LOOKAHEAD = 10
    for fi in range(len(frames_out)):
        fd = frames_out[fi]
        if fd.get("label") is None:
            continue
        gp = fd.get("_world")
        if gp is None: continue
        cam = fd.get("_cam")
        # Find the next labeled frame ~LOOKAHEAD ahead
        future_gp = None
        for j in range(fi + 1, min(fi + LOOKAHEAD + 5, len(frames_out))):
            fj = frames_out[j]
            if fj.get("label") and fj.get("_world") is not None:
                if j - fi >= LOOKAHEAD:
                    future_gp = fj["_world"]
                    break
                elif future_gp is None:
                    future_gp = fj["_world"]
        if future_gp is None or gp is None:
            continue
        dx = future_gp[0] - gp[0]
        dz = future_gp[1] - gp[1]
        speed = (dx * dx + dz * dz) ** 0.5
        if speed > 5 and cam:  # moving
            # Project the future position to screen
            heading_screen = project(future_gp[0], future_gp[1], cam[0], cam[1], cam[2])
            fd["label"]["movement"] = {
                "heading_world": [round(future_gp[0], 1), round(future_gp[1], 1)],
                "heading_screen": heading_screen,
                "speed": round(speed, 1),
            }
        else:
            fd["label"]["movement"] = None

    # Clean up temporary fields
    for fd in frames_out:
        fd.pop("_world", None)
        fd.pop("_cam", None)

    # Stats
    n_labeled = sum(1 for f in frames_out if f["label"] is not None)
    act_counts = {}
    for fr in frames_out:
        if fr["label"] is None: continue
        t = fr["label"]["action"]["type"]
        act_counts[t] = act_counts.get(t, 0) + 1

    mem_gaps_ms.sort()
    p50 = mem_gaps_ms[len(mem_gaps_ms)//2] if mem_gaps_ms else 0
    p99 = mem_gaps_ms[int(len(mem_gaps_ms)*0.99)] if mem_gaps_ms else 0

    print(f"  Labeled: {n_labeled}/{n_frames} ({n_unlabeled} unlabeled, {n_cam_fallback} cam fallbacks)", flush=True)
    print(f"  Mem gap p50={p50:.1f}ms p99={p99:.1f}ms", flush=True)
    if n_unlabeled > 0:
        pct = n_unlabeled / n_frames * 100
        print(f"  WARNING: {n_unlabeled} frames ({pct:.1f}%) unlabeled", flush=True)

    labels = {
        "match_id": match_id, "champion": champion,
        "team": game_info.get("garen_team"),
        "slot": game_info.get("garen_slot"),
        "fps": FPS, "screen_resolution": [SCREEN_W, SCREEN_H],
        "frame_resolution": [FRAME_SZ, FRAME_SZ],
        "total_frames": len(frames_out),
        "projection": {"fov_v_deg": 40.0, "fov_h_deg": round(math.degrees(FOV_H), 1),
                        "tilt_deg": 56.0, "cam_y": CAM_Y, "cam_z_offset": CAM_Z_OFFSET},
        "action_distribution": act_counts,
        "frames": frames_out,
    }

    with open(os.path.join(game_dir, "labels.json"), "w") as f:
        json.dump(labels, f)
    with open(os.path.join(game_dir, "raw_mem.json"), "w") as f:
        json.dump(mem_data, f)
    if cam_data:
        with open(os.path.join(game_dir, "raw_cam.json"), "w") as f:
            json.dump(cam_data, f)
    # clicks.json — overlay consumes via --click-events.
    with open(os.path.join(game_dir, "clicks.json"), "w") as f:
        json.dump({
            "match_id": match_id, "champion": champion,
            "clicks": click_results.get("clicks", []),
            "casts": click_results.get("casts", []),
            "watched": click_results.get("watched", []),
        }, f, indent=2)

    # Delete staging PNGs
    print(f"  Cleaning staging dir...", flush=True)
    shutil.rmtree(staging_dir, ignore_errors=True)

    post_end = time.time()
    print(f"  Done: {n_labeled} frames, {act_counts}  ({post_end - post_start:.0f}s)", flush=True)

    return {
        "post_start": post_start, "post_end": post_end,
        "wall_time": round(post_end - post_start, 1),
        "n_frames": n_frames, "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled, "n_cam_fallback": n_cam_fallback,
        "mem_gap_p50": round(p50, 2), "mem_gap_p99": round(p99, 2),
        "action_distribution": act_counts,
    }

# ═══════════════════════════════════════════════════════════════
# Post-Process Worker (separate process)
# ═══════════════════════════════════════════════════════════════
def _postprocess_worker(queue, log_path):
    """Long-lived worker that consumes post-process jobs from queue."""
    while True:
        job = queue.get()
        if job is None:
            break
        try:
            stats = post_process(**job)
            if stats:
                log_event(log_path, phase="post", match_id=job["match_id"], **stats)
        except Exception:
            traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════
def process_game(game_info, force_patch=False, post_queue=None, rec_start=1.0, rec_end=None, force_rerun=False):
    match_id = game_info["match_id"]
    game_id = game_info["game_id"]
    team = game_info["garen_team"]
    slot = game_info["garen_slot"]
    duration = game_info.get("duration", 1800)
    key = cam_key_for(team, slot)

    out_dir = os.path.join(OUTPUT_BASE, match_id)
    if os.path.exists(os.path.join(out_dir, "labels.json")) and not force_rerun:
        print(f"SKIP {match_id} (already done — pass --force to overwrite)", flush=True)
        return True
    if force_rerun and os.path.exists(out_dir):
        print(f"  --force: clearing existing {out_dir}", flush=True)
        shutil.rmtree(out_dir)

    # Per-game log redirect
    log_dir = os.path.join(OUTPUT_BASE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    game_log = open(os.path.join(log_dir, f"{match_id}.log"), "w")
    old_stdout = sys.stdout
    sys.stdout = game_log

    try:
        print(f"\n{'='*60}", flush=True)
        print(f"GAME: {match_id}  team={team} slot={slot} key={key} dur={duration}s", flush=True)
        print(f"{'='*60}", flush=True)

        # Pass 1: Memory + Camera + Clicks + Casts (all in one threaded scrape)
        mem_data, cam_data, scrape_stats, click_results = pass1_scrape(
            game_id, key, duration, force_patch=force_patch)
        if len(mem_data) < 100:
            print(f"FAIL {match_id}: only {len(mem_data)} mem samples", flush=True)
            return False
        log_event(JSONL_PATH, phase="scrape", match_id=match_id, **scrape_stats)

        mem_first_gt = mem_data[0]["gt"]
        mem_last_gt = mem_data[-1]["gt"]
        real_duration = mem_last_gt
        print(f"  Real game duration: {real_duration:.0f}s ({real_duration/60:.1f}min)  "
              f"mem covers gt {mem_first_gt:.2f}..{mem_last_gt:.2f}s", flush=True)

        # Pass 2: Video recording → per-game staging dir
        staging_dir = os.path.join(TEMP_BASE, match_id)
        n_frames, record_stats = pass2_record(
            game_id, key, real_duration, staging_dir,
            rec_start=rec_start, rec_end=rec_end,
            mem_first_gt=mem_first_gt, mem_last_gt=mem_last_gt,
        )
        if n_frames < 100:
            print(f"FAIL {match_id}: only {n_frames} frames", flush=True)
            return False
        log_event(JSONL_PATH, phase="record", match_id=match_id, **record_stats)

        # Post-process: queue to worker or run inline
        # Pass post_process the EFFECTIVE rec_start (clamped into mem coverage),
        # not the user-requested one — synthetic frame gt = rec_start_eff + i/FPS
        # must match the recording's actual startTime.
        job = {
            "match_id": match_id, "mem_data": mem_data, "cam_data": cam_data,
            "game_info": game_info, "staging_dir": staging_dir,
            "rec_start": record_stats.get("rec_start_eff", rec_start),
            "champion": CHAMPION,  # spawn-process workers don't see globals
            "click_results": click_results,
        }
        if post_queue is not None:
            post_queue.put(job)
            print(f"  Post-process queued", flush=True)
        else:
            stats = post_process(**job)
            if stats:
                log_event(JSONL_PATH, phase="post", match_id=match_id, **stats)
            if not stats:
                print(f"FAIL {match_id}: post-processing failed", flush=True)
                return False

        print(f"\nOK {match_id}: {n_frames} frames recorded", flush=True)
        return True

    finally:
        sys.stdout = old_stdout
        game_log.close()
        # Echo result to terminal
        last_line = open(os.path.join(log_dir, f"{match_id}.log")).readlines()[-1].strip()
        print(f"  [{match_id}] {last_line}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Replay Data Pipeline (2-pass + post-process)")
    parser.add_argument("--manifest", help="Path to manifest JSON")
    parser.add_argument("--index", type=int, help="Process single game at index")
    parser.add_argument("--batch", action="store_true", help="Process all games")
    parser.add_argument("--start", type=int, default=0, help="Start index for batch")
    parser.add_argument("--game-id", help="Numeric game ID")
    parser.add_argument("--match-id", help="Full match ID (e.g. NA1_5528069928)")
    parser.add_argument("--team", choices=["blue", "red"])
    parser.add_argument("--slot", type=int)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--champion", default="Garen",
                        help="Internal champion name (e.g. Garen, Belveth, KaiSa — no apostrophes). "
                             "Defaults to Garen for legacy back-compat; pass explicitly for any other champ.")
    parser.add_argument("--force-patch", action="store_true")
    parser.add_argument("--rec-start", type=float, default=1.0,
                        help="Game_t (seconds) to start recording. Default 1.0.")
    parser.add_argument("--rec-end", type=float, default=None,
                        help="Game_t (seconds) to stop recording. Default: real_duration+60.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output dir instead of skipping.")
    parser.add_argument("--no-overlap", action="store_true",
                        help="Run post-process inline instead of in worker process")
    args = parser.parse_args()
    global CHAMPION
    CHAMPION = args.champion
    if CHAMPION == "Garen" and "--champion" not in sys.argv:
        print(f"[pipeline] WARN: using default --champion=Garen (pass explicitly to silence)", flush=True)
    print(f"[pipeline] champion = {CHAMPION}  session0 = {_is_session0()}", flush=True)

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    post_queue = None
    worker = None
    if not args.no_overlap:
        post_queue = mp.Queue()
        worker = mp.Process(target=_postprocess_worker, args=(post_queue, JSONL_PATH))
        worker.start()

    try:
        if args.manifest:
            manifest = json.load(open(args.manifest))
            games = manifest.get("matches", [])
            games_ok = [g for g in games if g.get("version", "").startswith("16.")]

            if args.index is not None:
                process_game(games_ok[args.index], force_patch=args.force_patch, post_queue=post_queue,
                             rec_start=args.rec_start, rec_end=args.rec_end, force_rerun=args.force)
            elif args.batch:
                ok = fail = consecutive_alarms = 0
                for i in range(args.start, len(games_ok)):
                    g = games_ok[i]
                    print(f"\n[{i+1}/{len(games_ok)}] {g['match_id']}", flush=True)
                    try:
                        if process_game(g, force_patch=args.force_patch, post_queue=post_queue,
                                        rec_start=args.rec_start, rec_end=args.rec_end, force_rerun=args.force):
                            ok += 1; consecutive_alarms = 0
                        else:
                            fail += 1; consecutive_alarms += 1
                    except Exception:
                        traceback.print_exc(); fail += 1; consecutive_alarms += 1
                        kill_game()
                    if consecutive_alarms >= 3:
                        print(f"\n*** 3 consecutive failures — stopping batch ***", flush=True)
                        break
                print(f"\nBATCH: {ok} ok, {fail} fail / {len(games_ok)} total", flush=True)
            else:
                parser.error("--manifest needs --index or --batch")

        elif args.game_id or args.match_id:
            if not args.team or args.slot is None:
                parser.error("Need --team and --slot")
            gid = args.game_id or args.match_id.split("_")[-1]
            mid = args.match_id or f"NA1_{gid}"
            process_game({
                "match_id": mid, "game_id": gid,
                "garen_team": args.team, "garen_slot": args.slot,
                "duration": args.duration,
            }, force_patch=args.force_patch, post_queue=post_queue,
               rec_start=args.rec_start, rec_end=args.rec_end, force_rerun=args.force)
        else:
            parser.error("Provide --manifest or --game-id/--match-id")

    finally:
        if post_queue is not None:
            post_queue.put(None)
        if worker is not None:
            worker.join(timeout=600)


if __name__ == "__main__":
    main()
