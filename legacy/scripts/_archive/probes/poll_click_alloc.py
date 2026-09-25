"""Poll the click-destination allocation during replay playback and record
every Vec3 change as a click event.

Usage:
    python poll_click_alloc.py --alloc 0x1F379DBFF04 --start-gt 0 --end-gt 65

Output:   C:\\tmp\\click_events.json  with [{game_time, x, y, z}, ...]
"""
import ctypes, ctypes.wintypes as wt
import argparse, ssl, sys, time, json, subprocess, struct, urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

_k = ctypes.windll.kernel32
_ctx = ssl.create_default_context(); _ctx.check_hostname = False; _ctx.verify_mode = ssl.CERT_NONE

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower():
            return int(l.strip('"').split('","')[1])
    return None

def open_proc(pid):
    h = _k.OpenProcess(0x0410, False, pid)
    if not h: raise RuntimeError("OpenProcess failed")
    return h

def read_vec3(h, addr):
    buf = (ctypes.c_char * 12)()
    read = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, 12, ctypes.byref(read))
    if not ok or read.value != 12:
        return None
    return struct.unpack("<fff", bytes(buf))

def api_post(ep, body):
    req = urllib.request.Request(f"https://127.0.0.1:2999{ep}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def api_get(ep):
    with urllib.request.urlopen(f"https://127.0.0.1:2999{ep}", context=_ctx, timeout=3) as r:
        return json.loads(r.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", required=True, help="hex address like 0x1F379DBFF04")
    ap.add_argument("--start-gt", type=float, default=0.0)
    ap.add_argument("--end-gt", type=float, default=65.0)
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--poll-ms", type=int, default=30)
    ap.add_argument("--change-thr", type=float, default=50.0,
                    help="min Vec3 delta in units to count as a click")
    args = ap.parse_args()

    addr = int(args.alloc, 16)
    pid = find_pid()
    if not pid:
        print("ERR: League not running"); return 1
    h = open_proc(pid)
    print(f"pid={pid} alloc=0x{addr:X}")

    # Also poll mirror B (+0x308) and mirror C (+0x374) for validation
    addr_b = addr + 0x308
    addr_c = addr + 0x374

    # Seek to start and pause
    print(f"seeking to gt={args.start_gt}, paused")
    api_post("/replay/playback", {"time": args.start_gt, "paused": True})
    for _ in range(30):
        st = api_get("/replay/playback")
        if not st["seeking"] and st["paused"]: break
        time.sleep(0.2)

    # Read initial Vec3
    init = read_vec3(h, addr)
    print(f"initial Vec3 = {init}")

    # Now play forward at `speed`
    print(f"playing at speed={args.speed}x until gt={args.end_gt}")
    api_post("/replay/playback", {"speed": args.speed, "paused": False})

    prev = init if init else (0.0, 0.0, 0.0)
    events = []
    all_samples = []
    t_wall = time.time()
    last_poll = 0
    while True:
        now = time.time()
        if now - last_poll < args.poll_ms / 1000.0:
            time.sleep(0.005); continue
        last_poll = now

        try:
            st = api_get("/replay/playback")
        except Exception:
            continue
        gt = st["time"]
        if gt >= args.end_gt: break

        v = read_vec3(h, addr)
        vb = read_vec3(h, addr_b)
        vc = read_vec3(h, addr_c)
        if v is None: continue

        # Sanity: mirrors should match. Flag when they don't.
        mirrors_match = (vb is not None and vc is not None and
                         abs(v[0] - vb[0]) < 0.5 and abs(v[0] - vc[0]) < 0.5 and
                         abs(v[2] - vb[2]) < 0.5 and abs(v[2] - vc[2]) < 0.5)

        all_samples.append({
            "game_time": round(gt, 3),
            "x": v[0], "y": v[1], "z": v[2],
            "mirrors_match": bool(mirrors_match),
        })

        # Detect Vec3 change (click event)
        dx = v[0] - prev[0]; dz = v[2] - prev[2]
        if (dx * dx + dz * dz) ** 0.5 > args.change_thr:
            events.append({
                "game_time": round(gt, 3),
                "x": v[0], "y": v[1], "z": v[2],
                "prev_x": prev[0], "prev_z": prev[2],
                "mirrors_match": bool(mirrors_match),
            })
            print(f"  gt={gt:.2f}  click ({prev[0]:.0f},{prev[2]:.0f}) -> "
                  f"({v[0]:.0f},{v[2]:.0f})  mirrors_ok={mirrors_match}")
        prev = v

    api_post("/replay/playback", {"speed": 1.0, "paused": True})
    wall = time.time() - t_wall
    print(f"\n{len(events)} click events in {wall:.1f}s wall "
          f"covering gt=[{args.start_gt},{args.end_gt}]")
    out = {
        "alloc": hex(addr),
        "start_gt": args.start_gt,
        "end_gt": args.end_gt,
        "events": events,
        "samples": all_samples,
    }
    with open(r"C:\tmp\click_events.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote C:\\tmp\\click_events.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
