"""Extract Bel'Veth's click destinations for the remainder of a live replay.

Prereqs:
  - Replay is playing (or paused, will unpause)
  - An alloc address (found via identify_belveth_alloc.py) is known
  - DO NOT SEEK before or during this script — seeks flip ~50% of allocs

Strategy:
  - Poll alloc every POLL_MS
  - Record every Vec3 change > DELTA_UNITS (a new click)
  - Also poll mirror at +0x308 for corroboration
  - Monitor alloc health: if Vec3 becomes invalid (y out of range) for
    CONSECUTIVE_INVALID polls, alloc is dead — stop and report gap
  - Run until replay reaches END_GT or finishes

Output: C:\\tmp\\belveth_clicks_extracted.json
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, struct, json, time, ssl, urllib.request
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import builtins
_orig_print = builtins.print
def print(*a, **k):
    k.setdefault("flush", True); _orig_print(*a, **k)
builtins.print = print

ALLOC_ADDR = 0x1F28236FBE4  # from identify_belveth_alloc.py output
POLL_MS = 100
DELTA_UNITS = 50.0
CONSECUTIVE_INVALID = 30  # ~3s of nonsense before giving up
END_GT = 1800.0  # cap at 30 min
_k = ctypes.windll.kernel32
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
REPLAY = "https://127.0.0.1:2999"

def api_get(path):
    try:
        with urllib.request.urlopen(f"{REPLAY}{path}", context=_ctx, timeout=2) as r:
            return json.loads(r.read())
    except: return None

def api_post(path, body):
    try:
        req = urllib.request.Request(f"{REPLAY}{path}",
            data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=_ctx, timeout=2) as r: return r.read()
    except Exception as e: return None

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
    return None

def read_vec3(h, addr):
    buf = (ctypes.c_char * 12)(); r = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, 12, ctypes.byref(r))
    return struct.unpack("<fff", bytes(buf)) if ok and r.value == 12 else None

def valid(v):
    return v and 100 < v[0] < 15000 and 45 < v[1] < 65 and 100 < v[2] < 15000

def main():
    pid = find_pid()
    if not pid: print("ERR"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    print(f"pid={pid}  alloc=0x{ALLOC_ADDR:X}")
    pb = api_get("/replay/playback")
    if not pb: print("ERR no replay API"); return 1
    start_gt = pb.get("time", 0)
    print(f"starting at game_t={start_gt:.1f}s, playback speed=1x")
    api_post("/replay/playback", {"paused": False, "speed": 1.0})

    # Initial Vec3
    v0 = read_vec3(h, ALLOC_ADDR)
    print(f"initial Vec3 at alloc: {v0}  valid={valid(v0)}")
    if not valid(v0):
        print("ERR initial alloc reads invalid — is it the right address for current session?")
        return 1

    clicks = []
    last_v = v0
    clicks.append({"game_t": start_gt, "x": v0[0], "y": v0[1], "z": v0[2], "init": True})
    invalid_streak = 0
    t_wall0 = time.time()
    game_t_last = start_gt
    last_print_t = time.time()
    last_poll_game_t = start_gt

    while True:
        v = read_vec3(h, ALLOC_ADDR)
        pb = api_get("/replay/playback")
        if pb:
            game_t = pb.get("time", game_t_last)
            game_t_last = game_t
        else:
            game_t = game_t_last
        if not valid(v):
            invalid_streak += 1
            if invalid_streak >= CONSECUTIVE_INVALID:
                print(f"\n  ALLOC DIED at game_t={game_t:.1f}s  last valid Vec3 was {last_v}")
                break
        else:
            invalid_streak = 0
            dx = v[0] - last_v[0]; dz = v[2] - last_v[2]
            d = (dx*dx + dz*dz) ** 0.5
            if d > DELTA_UNITS:
                clicks.append({"game_t": game_t, "x": v[0], "y": v[1], "z": v[2], "delta_prev": round(d, 1)})
                last_v = v
        if game_t >= END_GT:
            print(f"\n  reached END_GT={END_GT}s"); break

        now = time.time()
        if now - last_print_t >= 15.0:
            rate = (game_t - start_gt) / max(now - t_wall0, 1)
            print(f"  game_t={game_t:.1f}s  clicks_so_far={len(clicks)}  invalid_streak={invalid_streak}  playback_rate={rate:.2f}x")
            last_print_t = now
        time.sleep(POLL_MS / 1000.0)

    out = {
        "alloc": hex(ALLOC_ADDR),
        "start_game_t": start_gt,
        "final_game_t": game_t_last,
        "total_clicks": len(clicks),
        "clicks": clicks,
    }
    with open(r"C:\tmp\belveth_clicks_extracted.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote C:\\tmp\\belveth_clicks_extracted.json  ({len(clicks)} click events, game_t {start_gt:.0f} → {game_t_last:.0f})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
