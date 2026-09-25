"""Verify that the 3 candidate click-destination addresses update at each
click by playing the replay forwards from gt=34 once, without intermediate
seeks, and sampling the Vec3 at each address at 10Hz.

If the values step cleanly (3124,8122) → (3736,8358) → (4398,8444) at each
click boundary, we have our answer.
"""
import ctypes, struct, subprocess, sys, json, time
import ssl, urllib.request
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

ADDRS = [0x2075BF70114, 0x2075BF7041C, 0x2075BF70488]

_k = ctypes.windll.kernel32
def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
def read(h, a, sz):
    buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
    return buf.raw[:n.value] if ok else b""
def vec3(h, a):
    d = read(h, a, 12); return struct.unpack("<fff", d) if len(d)==12 else (0,0,0)

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def _post(obj):
    req = urllib.request.Request("https://127.0.0.1:2999/replay/playback",
            data=json.dumps(obj).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def _get():
    with urllib.request.urlopen("https://127.0.0.1:2999/replay/playback", context=_ctx, timeout=2) as r:
        return json.loads(r.read())

def main():
    pid = find_pid()
    h = _k.OpenProcess(0x0410, False, pid)
    print(f"PID={pid}")

    # Seek to 34 once, then play forward
    print("Seeking to gt=34 paused")
    _post({"time": 34.0, "speed": 1.0, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = _get()
        if not st["seeking"] and st["paused"]: break

    # Sanity check: read addresses while paused at 34
    print("\nAt gt=34 (before any click):")
    for a in ADDRS:
        v = vec3(h, a)
        print(f"  0x{a:X}  ({v[0]:.1f}, {v[1]:.2f}, {v[2]:.1f})")

    print("\nUnpausing, sampling every 0.25s for 22s")
    _post({"speed": 1.0, "paused": False})
    t_end = time.time() + 22.0
    last_print_gt = None
    while time.time() < t_end:
        t0 = time.time()
        try: gt = _get()["time"]
        except: gt = None
        vals = [vec3(h, a) for a in ADDRS]
        # Print every 0.5s of game time
        if last_print_gt is None or (gt and gt - last_print_gt > 0.45):
            line = f"gt={gt:6.2f}  " + "  ".join(f"A{i}=({v[0]:.0f},{v[2]:.0f})" for i, v in enumerate(vals))
            print(line)
            last_print_gt = gt
        sl = 0.25 - (time.time() - t0)
        if sl > 0: time.sleep(sl)
    _post({"speed": 1.0, "paused": True})

if __name__ == "__main__":
    main()
