"""Pass 2 FINAL: Scrape with corrected offsets. Launch replay, seek to t=810, scrape 35s at ~20Hz, kill game."""
import urllib.request, ssl, json, time, os, struct, ctypes, sys, base64
import ctypes.wintypes as wt
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GAME_ID = "5528069928"
START_TIME = 810
DURATION = 35
OUT_PATH = r"C:\tmp\pass2_final.json"
LOG_PATH = r"C:\tmp\pass2_final_log.txt"

LOG = open(LOG_PATH, "w")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG.write(line+"\n"); LOG.flush(); print(line, flush=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE

def api_get(ep):
    with urllib.request.urlopen(urllib.request.Request(
        f"https://127.0.0.1:2999{ep}"), context=ctx, timeout=5) as r:
        return json.loads(r.read())

def api_post(ep, data):
    with urllib.request.urlopen(urllib.request.Request(
        f"https://127.0.0.1:2999{ep}", data=json.dumps(data).encode(),
        headers={"Content-Type":"application/json"}), context=ctx, timeout=5) as r:
        return json.loads(r.read())

def focus_game():
    user32 = ctypes.windll.user32
    rl = []
    def cb(hwnd, _):
        l = user32.GetWindowTextLengthW(hwnd)
        if l > 0:
            buf = ctypes.create_unicode_buffer(l+1)
            user32.GetWindowTextW(hwnd, buf, l+1)
            if "league of legends" in buf.value.lower(): rl.append(hwnd)
        return True
    user32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(cb), 0)
    if rl:
        hwnd = rl[0]
        fg = user32.GetForegroundWindow()
        ft = user32.GetWindowThreadProcessId(fg, None)
        ct = ctypes.windll.kernel32.GetCurrentThreadId()
        user32.AttachThreadInput(ct, ft, True)
        user32.keybd_event(0x12, 0, 0, 0); user32.keybd_event(0x12, 0, 2, 0)
        user32.ShowWindow(hwnd, 9); user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.AttachThreadInput(ct, ft, False)
        time.sleep(0.5)

def lock_camera():
    from pynput.keyboard import Controller
    kb = Controller()
    kb.press('q'); time.sleep(0.05); kb.release('q')
    time.sleep(0.15)
    kb.press('q'); time.sleep(0.05); kb.release('q')
    time.sleep(0.3)

# ======================================================================
# Memory reader — CORRECTED offsets
# ======================================================================
k = ctypes.windll.kernel32
HERO_ARRAY_RVA = 0x1DBEEE8
OFF_POS = 0x25C
OFF_NAME = 0x4328
OFF_NETID = 0xCC
# ActiveSpellCast: hero+0x3120 -> spell ptr
#   spell+0x008 -> SpellInfo -> +0x28 -> name_ptr -> string (DOUBLE DEREF)
#   spell+0x0D0 -> cast start pos (Vec3)
#   spell+0x0DC -> cast target pos (Vec3)
OFF_ACTIVE_SPELL = 0x3120
OFF_SPELLINFO = 0x008
OFF_SPELL_NAME_PTR = 0x28
OFF_CAST_START = 0x0D0
OFF_CAST_TARGET = 0x0DC

class Mem:
    def __init__(s,pid):s.h=k.OpenProcess(0x0410,False,pid)
    def read(s,a,sz):
        buf=ctypes.create_string_buffer(sz);n=ctypes.c_size_t(0)
        ok=k.ReadProcessMemory(s.h,ctypes.c_void_p(a),buf,sz,ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value==sz else None
    def u64(s,a):d=s.read(a,8);return struct.unpack('<Q',d)[0] if d else None
    def u32(s,a):d=s.read(a,4);return struct.unpack('<I',d)[0] if d else None
    def f32(s,a):d=s.read(a,4);return struct.unpack('<f',d)[0] if d else None
    def vec3(s,a):d=s.read(a,12);return struct.unpack('<fff',d) if d else None
    def string(s,a,n=64):
        d=s.read(a,n)
        if not d:return None
        t=d.split(b'\x00')[0]
        try:return t.decode('ascii')
        except:return None
    def close(s):k.CloseHandle(s.h)

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF

def find_pid():
    import subprocess
    r=subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],capture_output=True,text=True)
    for line in r.stdout.strip().split('\n'):
        if 'league' in line.lower():return int(line.strip('"').split('","')[1])
    return None

def find_base(pid):
    class ME(ctypes.Structure):
        _fields_=[("dwSize",ctypes.c_ulong),("th32ModuleID",ctypes.c_ulong),("th32ProcessID",ctypes.c_ulong),("GlblcntUsage",ctypes.c_ulong),("ProccntUsage",ctypes.c_ulong),("modBaseAddr",ctypes.POINTER(ctypes.c_byte)),("modBaseSize",ctypes.c_ulong),("hModule",ctypes.c_void_p),("szModule",ctypes.c_char*256),("szExePath",ctypes.c_char*260)]
    snap=k.CreateToolhelp32Snapshot(0x18,pid);me=ME();me.dwSize=ctypes.sizeof(ME)
    if k.Module32First(snap,ctypes.byref(me)):
        while True:
            if b'league' in me.szModule.lower():
                base=ctypes.cast(me.modBaseAddr,ctypes.c_void_p).value;k.CloseHandle(snap);return base
            if not k.Module32Next(snap,ctypes.byref(me)):break
    k.CloseHandle(snap);return None

def read_heroes(m, base):
    """Read all 10 hero positions and names."""
    arr_ptr = m.u64(base + HERO_ARRAY_RVA)
    if not arr_ptr or arr_ptr < 0x10000: return []
    heroes = []
    for i in range(10):
        hp = m.u64(arr_ptr + i * 8)
        if not is_heap(hp): continue
        name = m.string(hp + OFF_NAME)
        if not name or len(name) < 2 or not name[0].isalpha(): continue
        pos = m.vec3(hp + OFF_POS)
        if not pos or abs(pos[0]) > 20000: continue
        heroes.append({"idx": i, "ptr": hp, "name": name, "pos": pos})
    return heroes

def read_spells(m, heroes):
    """Read ActiveSpellCast for all heroes using corrected offsets.

    Chain: hero+0x3120 -> spell ptr
           spell+0x008 -> SpellInfo ptr
           SpellInfo+0x28 -> name_ptr -> string (double deref)
           spell+0x0D0 -> cast start pos
           spell+0x0DC -> cast target pos
    """
    results = {}
    hero_positions = {h["name"]: h["pos"] for h in heroes}

    for h in heroes:
        spell_ptr = m.u64(h["ptr"] + OFF_ACTIVE_SPELL)
        if not is_heap(spell_ptr): continue

        # Double-deref spell name
        spell_info = m.u64(spell_ptr + OFF_SPELLINFO)
        if not is_heap(spell_info): continue
        name_ptr = m.u64(spell_info + OFF_SPELL_NAME_PTR)
        if not is_heap(name_ptr): continue
        spell_name = m.string(name_ptr)
        if not spell_name or len(spell_name) < 3: continue

        cast_start = m.vec3(spell_ptr + OFF_CAST_START)
        cast_target = m.vec3(spell_ptr + OFF_CAST_TARGET)

        # Identify target hero by matching cast_target to hero positions
        target_hero = None
        if cast_target and 100 < cast_target[0] < 15000 and 100 < cast_target[2] < 15000:
            best_dist = 300
            for h2_name, h2_pos in hero_positions.items():
                if h2_name == h["name"]: continue
                dx = cast_target[0] - h2_pos[0]
                dz = cast_target[2] - h2_pos[2]
                dist = (dx*dx + dz*dz)**0.5
                if dist < best_dist:
                    best_dist = dist
                    target_hero = h2_name

        results[h["name"]] = {
            "spell": spell_name,
            "cast_start": [round(cast_start[0],1), round(cast_start[2],1)] if cast_start else None,
            "cast_target": [round(cast_target[0],1), round(cast_target[2],1)] if cast_target else None,
            "target_hero": target_hero,
        }
    return results

def compute_movement(prev_positions, curr_positions, dt):
    """Compute velocity and projected target from position deltas."""
    results = {}
    for name, pos in curr_positions.items():
        prev = prev_positions.get(name)
        if not prev or not pos:
            results[name] = {"speed": 0, "dir": None, "proj": None}
            continue
        dx = pos[0] - prev[0]
        dz = pos[2] - prev[2]
        dist = (dx*dx + dz*dz)**0.5
        speed = dist / dt if dt > 0 else 0
        if speed > 30:
            ndx = dx / dist; ndz = dz / dist
            proj_x = pos[0] + ndx * speed * 2.0
            proj_z = pos[2] + ndz * speed * 2.0
            results[name] = {
                "speed": round(speed, 0),
                "dir": [round(ndx, 3), round(ndz, 3)],
                "proj": [round(proj_x, 1), round(proj_z, 1)],
            }
        else:
            results[name] = {"speed": 0, "dir": None, "proj": None}
    return results

# ======================================================================
# Main
# ======================================================================
try:
    log("=== PASS 2 FINAL: Data Scraping (corrected offsets) ===")

    # Kill existing game if running
    os.system('taskkill /F /IM "League of Legends.exe" >nul 2>&1')
    time.sleep(3)

    # Launch replay
    log("Launching replay...")
    lf = open(r"C:\Riot Games\League of Legends\lockfile").read().strip().split(":")
    auth = base64.b64encode(f"riot:{lf[3]}".encode()).decode()
    urllib.request.urlopen(urllib.request.Request(
        f"https://127.0.0.1:{lf[2]}/lol-replays/v1/rofls/{GAME_ID}/watch",
        method="POST", data=json.dumps({"componentType":"replay"}).encode(),
        headers={"Authorization":f"Basic {auth}","Content-Type":"application/json"}
    ), context=ctx, timeout=10)

    # Wait for game to load
    for i in range(90):
        try: api_get("/liveclientdata/gamestats"); log(f"Game loaded ({i*3}s)"); break
        except: time.sleep(3)
    else:
        log("FAILED to load game"); sys.exit(1)

    time.sleep(5)

    # Init memory reader
    pid = find_pid()
    if not pid: log("PID not found!"); sys.exit(1)
    base = find_base(pid)
    if not base: log("Module base not found!"); sys.exit(1)
    m = Mem(pid)
    log(f"Memory: PID={pid} base=0x{base:X}")

    # Verify heroes
    heroes = read_heroes(m, base)
    log(f"Heroes: {[h['name'] for h in heroes]}")

    # Seek to start time
    focus_game()
    time.sleep(1)
    log(f"Seeking to t={START_TIME}...")
    api_post("/replay/playback", {"time": START_TIME, "speed": 0.0})
    time.sleep(5)

    # Lock camera on Garen
    api_post("/replay/render", {"selectionName": "Garen"})
    time.sleep(0.5)
    focus_game()
    time.sleep(0.5)
    api_post("/replay/playback", {"speed": 1.0})
    time.sleep(1)
    focus_game()
    time.sleep(0.3)
    lock_camera()
    log("Camera locked on Garen, playing at 1x")

    # Verify playback
    t1 = api_get("/replay/playback")["time"]
    time.sleep(1)
    t2 = api_get("/replay/playback")["time"]
    log(f"Time check: {t1:.1f} -> {t2:.1f} (delta={t2-t1:.1f}s)")

    # ================================================================
    # Main scrape loop — ~20Hz
    # ================================================================
    log(f"Scraping for {DURATION}s at ~20Hz...")
    samples = []
    prev_positions = {}
    prev_wall = time.time()
    t0 = time.time()
    api_errors = 0

    while time.time() - t0 < DURATION:
        wall = time.time()
        dt_wall = wall - prev_wall

        try:
            # Read game state from APIs
            pb = api_get("/replay/playback")
            rd = api_get("/replay/render")
            gt = pb["time"]
            cam_x = rd["cameraPosition"]["x"]
            cam_z = rd["cameraPosition"]["z"]
        except Exception as e:
            api_errors += 1
            time.sleep(0.02)
            continue

        # Read hero positions from memory
        heroes = read_heroes(m, base)
        curr_positions = {h["name"]: h["pos"] for h in heroes}

        # Compute movement from position deltas
        movement = compute_movement(prev_positions, curr_positions, dt_wall) if prev_positions else {}
        prev_positions = curr_positions
        prev_wall = wall

        # Read active spells
        spells = read_spells(m, heroes)

        # Build sample
        hero_data = {}
        for h in heroes:
            name = h["name"]
            hero_entry = {
                "pos": [round(h["pos"][0], 1), round(h["pos"][2], 1)],
            }
            # Add movement if available
            mv = movement.get(name)
            if mv and mv["speed"] > 0:
                hero_entry["speed"] = mv["speed"]
                hero_entry["dir"] = mv["dir"]
                hero_entry["proj"] = mv["proj"]

            # Add spell if active
            sp = spells.get(name)
            if sp:
                hero_entry["spell"] = sp["spell"]
                hero_entry["cast_start"] = sp["cast_start"]
                hero_entry["cast_target"] = sp["cast_target"]
                hero_entry["target_hero"] = sp["target_hero"]

            hero_data[name] = hero_entry

        samples.append({
            "gt": round(gt, 3),
            "wall": round(wall - t0, 3),
            "cam": [round(cam_x, 1), round(cam_z, 1)],
            "heroes": hero_data,
        })

        # Target ~20Hz = 50ms per iteration. API calls take ~10-20ms, RPM ~5ms.
        elapsed = time.time() - wall
        sleep_time = max(0.01, 0.045 - elapsed)
        time.sleep(sleep_time)

    # Summary
    if samples:
        gt_start = samples[0]["gt"]
        gt_end = samples[-1]["gt"]
        rate = len(samples) / DURATION
        log(f"Scraped {len(samples)} samples, gt={gt_start:.1f}-{gt_end:.1f}, rate={rate:.1f}Hz, api_errors={api_errors}")

        # Count spell detections
        spell_count = sum(1 for s in samples for h in s["heroes"].values() if "spell" in h)
        target_count = sum(1 for s in samples for h in s["heroes"].values() if h.get("target_hero"))
        log(f"Spells detected: {spell_count}, with hero targets: {target_count}")
    else:
        log("NO SAMPLES COLLECTED!")

    # Save
    os.makedirs(r"C:\tmp", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(samples, f)
    log(f"Saved {len(samples)} samples to {OUT_PATH}")

    # Kill game
    m.close()
    os.system('taskkill /F /IM "League of Legends.exe" >nul 2>&1')
    log("Game killed")
    log("PASS 2 FINAL DONE")

except Exception as e:
    import traceback
    log(f"ERROR: {e}")
    traceback.print_exc(file=LOG)
    traceback.print_exc()
finally:
    LOG.close()
