"""Launch a ROFL replay via LCU and wait for game to load. Then exits.
Used before running scan_offsets.py for patch rescans.

Usage:  python launch_replay_only.py <game_id>
"""
import sys, base64, json, ssl, time, urllib.request

LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE

def lcu_auth():
    p = open(LOCKFILE).read().strip().split(":")
    return p[2], f"Basic {base64.b64encode(f'riot:{p[3]}'.encode()).decode()}"

def lcu_post(ep, body=None):
    port, auth = lcu_auth()
    req = urllib.request.Request(
        f"https://127.0.0.1:{port}{ep}", method="POST",
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": auth, "Content-Type":"application/json"})
    with urllib.request.urlopen(req, context=_ctx, timeout=10) as r:
        raw = r.read(); return json.loads(raw) if raw else None

def replay_get(ep):
    with urllib.request.urlopen(
        urllib.request.Request(f"https://127.0.0.1:2999{ep}"),
        context=_ctx, timeout=5) as r:
        return json.loads(r.read())

gid = sys.argv[1]
print(f"Launching replay {gid}...", flush=True)
lcu_post(f"/lol-replays/v1/rofls/{gid}/watch", {"componentType":"replay"})

for i in range(180):
    try:
        gs = replay_get("/liveclientdata/gamestats")
        print(f"  Loaded at gt={gs.get('gameTime',0):.1f}s ({i*2}s wait)", flush=True)
        sys.exit(0)
    except Exception:
        time.sleep(2)
print("TIMEOUT", flush=True); sys.exit(1)
