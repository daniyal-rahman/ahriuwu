"""Download a single .rofl by game_id via LCU. Run on Windows with League client open.
Usage:  python download_one_replay.py <game_id>
"""
import sys, base64, json, ssl, time, urllib.request
from urllib.error import HTTPError

LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE

p = open(LOCKFILE).read().strip().split(":")
port, token = p[2], p[3]
auth = base64.b64encode(f"riot:{token}".encode()).decode()

gid = sys.argv[1]
# Check download status first
try:
    req = urllib.request.Request(
        f"https://127.0.0.1:{port}/lol-replays/v1/metadata/{gid}",
        headers={"Authorization": f"Basic {auth}"})
    with urllib.request.urlopen(req, context=_ctx, timeout=10) as r:
        meta = json.loads(r.read())
        print(f"Metadata: state={meta.get('state')}, downloadProgress={meta.get('downloadProgress')}", flush=True)
except HTTPError as e:
    print(f"Metadata fetch: HTTP {e.code}", flush=True)

# Trigger download
req = urllib.request.Request(
    f"https://127.0.0.1:{port}/lol-replays/v1/rofls/{gid}/download",
    method="POST", data=b"{}",
    headers={"Authorization": f"Basic {auth}", "Content-Type":"application/json"})
try:
    with urllib.request.urlopen(req, context=_ctx, timeout=30) as r:
        print(f"Download triggered: HTTP {r.status}", flush=True)
        body = r.read()
        if body: print(body.decode(errors='replace')[:500], flush=True)
except HTTPError as e:
    body = e.read().decode(errors='replace') if e.fp else ''
    print(f"Download HTTP {e.code}: {body[:500]}", flush=True)

# Poll metadata until state is "watch" or timeout
print("\nWaiting for download to complete (poll /metadata)...", flush=True)
for i in range(60):
    try:
        req = urllib.request.Request(
            f"https://127.0.0.1:{port}/lol-replays/v1/metadata/{gid}",
            headers={"Authorization": f"Basic {auth}"})
        with urllib.request.urlopen(req, context=_ctx, timeout=5) as r:
            meta = json.loads(r.read())
            state = meta.get("state"); prog = meta.get("downloadProgress", 0)
            print(f"  [{i*3:3d}s] state={state} progress={prog}", flush=True)
            if state == "watch":
                print("DOWNLOADED", flush=True)
                sys.exit(0)
    except Exception as e:
        print(f"  poll err: {e}", flush=True)
    time.sleep(3)
print("TIMEOUT", flush=True); sys.exit(1)
