#!/usr/bin/env python3
"""Lock camera to Garen, disable fog, pause at 3:00. Run on Windows while replay is playing."""
import ssl, json
from urllib.request import Request, urlopen

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def r(path, method="GET", data=None):
    body = json.dumps(data).encode() if data else None
    req = Request(f"https://127.0.0.1:2999{path}", method=method, data=body,
                  headers={"Content-Type": "application/json"})
    raw = urlopen(req, context=ctx).read()
    return json.loads(raw) if raw else None

r("/replay/render", "POST", {
    "cameraMode": "fps",
    "selectionName": "Garen",
    "cameraAttached": True,
    "fogOfWar": False,
    "interfaceAll": False,
})

r("/replay/playback", "POST", {"time": 180.0, "speed": 0.0, "paused": True})

pb = r("/replay/playback")
print(f"Paused at t={pb['time']:.1f}s, camera locked to Garen, UI hidden")
print("Take a screenshot now (Win+Shift+S), save as PNG")
print(f"Game time: {pb['time']:.1f}s")
