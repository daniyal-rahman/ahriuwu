"""Orchestrator: launch Bel'Veth replay via LCU (same chain as pipeline.py),
run a 60s identify to get hero+A1+A2, pause, then deep-classify.

Single-shot, no schtasks needed (no key input — cam-lock is not required for
the heap "Belveth\\0" scan). Run over plain SSH:

    ssh windows "C:\\Python313\\python.exe -u C:\\Users\\daniz\\Repos\\ahriuwu\\scripts\\run_alloc_deep_classify.py"
"""
import sys, os, time, json, base64, ssl, urllib.request, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import builtins
_orig_print = builtins.print
def print(*a, **k):
    k.setdefault("flush", True); _orig_print(*a, **k)
builtins.print = print

GAME_ID = "5545727197"
LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE

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

def replay_get(ep):
    with urllib.request.urlopen(
        urllib.request.Request(f"https://127.0.0.1:2999{ep}"),
        context=_ctx, timeout=5) as r:
        return json.loads(r.read())

def replay_post(ep, data):
    with urllib.request.urlopen(
        urllib.request.Request(f"https://127.0.0.1:2999{ep}",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}),
        context=_ctx, timeout=5) as r:
        return json.loads(r.read())

def launch():
    print(f"[1/4] Launching replay {GAME_ID} via LCU...")
    try:
        lcu_post(f"/lol-replays/v1/rofls/{GAME_ID}/watch", {"componentType": "replay"})
    except Exception as e:
        print(f"  LCU post returned: {e}  (may be benign if game already loading)")
    for i in range(120):
        try:
            replay_get("/liveclientdata/gamestats")
            print(f"  game loaded ({i*2}s)")
            return True
        except Exception:
            time.sleep(2)
    print("  TIMEOUT waiting for game"); return False

def main():
    if not launch():
        return 1

    # Advance to gt ~30 and play at 2x for richer click history during identify
    print("[2/4] Setting playback to gt=30, speed=2x")
    try:
        replay_post("/replay/playback", {"paused": True, "time": 30.0})
        time.sleep(2)
        replay_post("/replay/playback", {"paused": False, "speed": 2.0})
    except Exception as e:
        print(f"  playback control failed: {e}")
        return 1

    # Import tightpoll helpers (find_belveth_hero, identify_top_k)
    import belveth_tightpoll as bt
    import ctypes

    pid = bt.find_pid()
    if not pid:
        print("ERR: League pid not found"); return 1
    h = ctypes._k = ctypes.windll.kernel32
    ph = h.OpenProcess(0x0410, False, pid)
    if not ph:
        print("ERR: OpenProcess failed"); return 1

    print(f"[3/4] Finding Bel'Veth hero in heap...")
    hero_ptr = None
    for attempt in range(8):
        hero_ptr = bt.find_belveth_hero(ph)
        if hero_ptr: break
        print(f"  attempt {attempt+1}: no Belveth yet, sleeping 5s")
        time.sleep(5)
    if not hero_ptr:
        print("ERR: could not find Bel'Veth hero struct"); return 1
    print(f"  hero=0x{hero_ptr:X}")

    # Run identify — 60s wall ≈ 120s game-time at 2x
    print("[3/4] Running 60s identify to find click-dest allocs...")
    top, _hero_hist = bt.identify_top_k(ph, hero_ptr, 60, k=5)
    if not top:
        print("ERR: identify produced no candidates"); return 1
    print(f"  top candidates:")
    for a, avg, sr, n in top:
        print(f"    0x{a:X}  avg_d={avg:.0f}  simul={sr:.2f}  n={n}")

    # Top two — A1 (lowest avg_d, the one we expect to be real) and A2 (sibling)
    a1 = top[0][0]
    a2 = top[1][0] if len(top) > 1 else None

    # Pause replay so allocs hold still during deep classify
    print("[4/4] Pausing replay; running deep classify")
    try:
        replay_post("/replay/playback", {"paused": True})
    except Exception as e:
        print(f"  pause failed (continuing anyway): {e}")
    time.sleep(1)

    # Stash an "anchors" sidecar so the next stage can read it without re-running
    sidecar = {
        "pid": pid,
        "hero": hex(hero_ptr),
        "top": [{"addr": hex(a), "avg_d": avg, "simul": sr, "n": n} for a, avg, sr, n in top],
        "a1": hex(a1),
        "a2": hex(a2) if a2 else None,
    }
    os.makedirs(r"C:\tmp", exist_ok=True)
    with open(r"C:\tmp\belveth_identify.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"  wrote C:\\tmp\\belveth_identify.json")

    # Now invoke alloc_deep_classify as a subprocess so it logs cleanly
    cmd = [sys.executable, "-u",
           os.path.join(os.path.dirname(__file__), "alloc_deep_classify.py"),
           "--alloc", hex(a1),
           "--hero", hex(hero_ptr),
           "--size", "0x400"]
    if a2:
        cmd.extend(["--alloc2", hex(a2)])
    print(f"  running: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    print(f"deep_classify exit code: {rc}")
    return rc

if __name__ == "__main__":
    sys.exit(main())
