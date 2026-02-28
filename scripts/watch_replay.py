#!/usr/bin/env python3
"""Launch a replay in the League client for viewing."""
import base64, json, ssl, sys
from urllib.request import Request, urlopen

game_id = sys.argv[1] if len(sys.argv) > 1 else "5496610100"

with open(r"C:\Riot Games\League of Legends\lockfile") as f:
    _, _, port, token, _ = f.read().split(":")

auth = base64.b64encode(f"riot:{token}".encode()).decode()
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = f"https://127.0.0.1:{port}/lol-replays/v1/rofls/{game_id}/watch"
body = json.dumps({"componentType": "replay"}).encode()
req = Request(url, method="POST", data=body, headers={
    "Authorization": f"Basic {auth}",
    "Content-Type": "application/json",
})
urlopen(req, context=ctx)
print(f"Launched replay {game_id}!")
