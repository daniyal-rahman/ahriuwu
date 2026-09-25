"""Sidecar reader for the League Live Client Data API (https://127.0.0.1:2999).

Perception escape-hatch: the v7 tokenizer does not carry scalar HUD state at
decision precision (cross-game HP probe R2~0.16, visually confirmed), so at
LIVE time the agent reads Garen's OWN exact stats straight from the client's
local API instead of pixels — the same quantities training gets from replay
labels. Enemy state has no API and stays a perception/aux-head problem.

Only available on the game machine while a game is running; every call
degrades gracefully to None off-game. The API serves a self-signed cert on
localhost -> certificate verification is disabled on purpose.

Caveat: ``currentGold`` is SPENDABLE gold (drops on purchase), not the
monotonic ``gold_total`` the solo-gold reward uses — fine for telemetry and
safety gates, wrong for reward accounting.

Probe from the game machine:  python -m ahriuwu.live.client_api
"""
from __future__ import annotations

import json
import ssl
import threading
import time
import urllib.request
from typing import Optional

BASE = "https://127.0.0.1:2999/liveclientdata"

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def _get(path: str, timeout: float = 0.25) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"{BASE}/{path}", timeout=timeout, context=_CTX) as r:
            return json.loads(r.read())
    except Exception:
        return None  # no game / client not up / transient — caller treats as absent


def read_own_state() -> Optional[dict]:
    """One-shot read: {'hp_frac','level','gold','game_time'} or None off-game."""
    ap = _get("activeplayer")
    if not ap:
        return None
    cs = ap.get("championStats") or {}
    hp, hpm = cs.get("currentHealth"), cs.get("maxHealth")
    gs = _get("gamestats")
    return {
        "hp_frac": (float(hp) / float(hpm)) if hp is not None and hpm else None,
        "level": ap.get("level"),
        "gold": ap.get("currentGold"),
        "game_time": gs.get("gameTime") if gs else None,
    }


class LiveStatePoller:
    """Background poller; ``latest`` always holds the freshest reading (or None).

    The dict is replaced atomically, so a single reader needs no lock:

        poller = LiveStatePoller(hz=10).start()
        ...
        s = poller.latest        # None until in-game
        poller.stop()
    """

    def __init__(self, hz: float = 10.0):
        self.hz = hz
        self.latest: Optional[dict] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "LiveStatePoller":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            self.latest = read_own_state()
            time.sleep(1.0 / self.hz)

    def stop(self) -> None:
        self._stop.set()


if __name__ == "__main__":
    s = read_own_state()
    print("live client state:", s if s else "NO GAME (client API not reachable)")
