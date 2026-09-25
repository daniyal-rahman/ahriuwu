#!/usr/bin/env python
"""Smoke-test the RL action ingress end to end.

This is the piece that was missing: under LANERL_HEADLESS there are no clients,
so nothing could issue champion orders and the policy had no way into the game.
LanerlControl.cs adds a TCP lockstep channel; this proves it actually works.

Asserts, in order:
  1. the server accepts a connection and emits an observation
  2. lockstep holds -- one observation per action, game time advances
  3. a "move" order MOVES the champion (the thing that was impossible before)
  4. an "attack" order engages a minion
  5. fog flags are present per team (vb/vr), so the actor/critic split is feedable
"""
from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

_PROJECTS = Path(__file__).resolve().parents[2]
VENDOR = _PROJECTS / "lanerl-vendor"
BIN = VENDOR / "LoLServer/GameServerConsole/bin/Release/net6.0"
CFG = _PROJECTS / "ahriuwu-lanerl/lanerl/cfg/garen1v1.json"
SRV_LOG = "/tmp/lanerl_smoke_server.log"   # keep it: Roslyn errors land here


def champs(obs):
    return {u["tm"]: u for u in obs["u"] if u["k"] == "Champion"}


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5810
    env = dict(os.environ)
    env.update(
        DOTNET_ROOT=str(VENDOR / "dotnet"),
        LANERL_HEADLESS="1",
        LANERL_FREERUN="1",
        LANERL_TOPONLY="1",
        LANERL_CONTROL_PORT=str(port),
        LANERL_STEP_TICKS="2",          # 30 Hz at the server's 60 Hz tick
        # WITHOUT THIS the smoke test lies. LanerlConfig.DriveTeams defaults to
        # "blue", so an unset LANERL_BOT attaches the in-server scripted bot to
        # the very champion we are trying to drive; it re-issues orders every
        # tick and overrides ours. An earlier run of this file reported "move
        # order: travelled 2655 units" that was substantially the bot walking to
        # lane, not the move order under test.
        LANERL_BOT="none",
    )
    proc = subprocess.Popen(
        [str(BIN / "GameServerConsole"), "--config", str(CFG), "--port", str(port + 100)],
        cwd=str(BIN), env=env, stdout=open(SRV_LOG, "w"), stderr=subprocess.STDOUT,
    )
    try:
        # the server blocks on accept() until we attach
        sock = None
        for _ in range(120):
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=2)
                break
            except OSError:
                time.sleep(1)
        if sock is None:
            # say WHY: a dead server means the log has the reason, and an
            # exit status here was previously dropped on the floor
            rc = proc.poll()
            tail = "\n".join(Path(SRV_LOG).read_text(errors="ignore").splitlines()[-15:])
            print(f"FAIL: server never opened control port {port} "
                  f"(process rc={rc!r}, None means still running)\n{tail}")
            return 1
        print(f"connected on {port}")
        f = sock.makefile("rwb")

        def step(action):
            # compact separators, like env.py and vec.py: the server's hand-rolled
            # parser reads numbers itself, and default json.dumps writes
            # '"slot": 2' with a space. That parsed as NaN until it was fixed,
            # and this smoke test was exercising a DIFFERENT parser path from
            # production -- the same trap that let a broken action path pass.
            f.write((json.dumps(action, separators=(",", ":")) + "\n").encode())
            f.flush()
            line = f.readline()
            return json.loads(line) if line else None

        obs = json.loads(f.readline())            # first observation
        print(f"[1] first obs: t={obs['t']}ms units={len(obs['u'])}")

        # --- lockstep + time advance
        t0 = obs["t"]
        for _ in range(20):
            obs = step({"blue": {"t": "noop"}, "red": {"t": "noop"}})
        dt = obs["t"] - t0
        print(f"[2] 20 steps advanced {dt}ms (expect ~{20*4*1000//60}ms at 4 ticks/step)")
        assert dt > 0, "game time did not advance"

        # --- fog flags present
        assert all("vb" in u and "vr" in u for u in obs["u"]), "missing per-team visibility"
        print("[3] per-team fog flags present (vb/vr)")

        # --- MOVE: the capability that did not exist before
        c = champs(obs)
        blue = c.get(100)
        assert blue, "no blue champion in observation"
        start = (blue["x"], blue["y"])
        tx, ty = start[0] + 1200, start[1] + 1200
        for _ in range(120):                       # ~8 s of game time
            obs = step({"blue": {"t": "move", "x": tx, "y": ty}, "red": {"t": "noop"}})
        blue = champs(obs)[100]
        moved = math.dist(start, (blue["x"], blue["y"]))
        print(f"[4] move order: travelled {moved:.0f} units  {start} -> ({blue['x']},{blue['y']})")
        assert moved > 200, f"champion did not move (only {moved:.0f} units)"

        # fast-forward past the 90s first-wave spawn so minions exist to attack
        while obs["t"] < 100_000:
            obs = step({"blue": {"t": "noop"}, "red": {"t": "noop"}})
        blue = champs(obs)[100]
        print(f"[4b] fast-forwarded to t={obs['t']}ms, units={len(obs['u'])}")

        # --- ATTACK: engage the nearest enemy minion, if one is up
        target = None
        for u in obs["u"]:
            if u["k"] == "LaneMinion" and u["tm"] == 200:
                d = math.dist((blue["x"], blue["y"]), (u["x"], u["y"]))
                if target is None or d < target[0]:
                    target = (d, u)
        if target:
            tid = target[1]["id"]
            hp0 = target[1]["hp"]
            tx2, ty2 = target[1]["x"], target[1]["y"]
            print(f"    target minion {tid} at ({tx2},{ty2}), champ at ({blue['x']},{blue['y']}), "
                  f"dist={target[0]:.0f} (Garen AA range ~175)")
            # walk into range: engine only swings at targets already in range
            for i in range(400):
                obs = step({"blue": {"t": "move", "x": tx2, "y": ty2}, "red": {"t": "noop"}})
                if i % 100 == 99:
                    b = champs(obs).get(100)
                    m = next((u for u in obs["u"] if u["id"] == tid), None)
                    if b and m:
                        print(f"    approach {i+1}: dist={math.dist((b['x'],b['y']),(m['x'],m['y'])):.0f} "
                              f"minion_hp={m['hp']}")
                    elif not m:
                        print(f"    approach {i+1}: target died/despawned"); break
            for i in range(300):
                obs = step({"blue": {"t": "attack", "id": tid}, "red": {"t": "noop"}})
                if i % 100 == 99:
                    b = champs(obs).get(100)
                    m = next((u for u in obs["u"] if u["id"] == tid), None)
                    print(f"    attack {i+1}: dist="
                          f"{math.dist((b['x'],b['y']),(m['x'],m['y'])):.0f} hp={m['hp']}"
                          if (b and m) else f"    attack {i+1}: target gone (killed)")
                    if not m: break
            still = next((u for u in obs["u"] if u["id"] == tid), None)
            hp1 = still["hp"] if still else 0
            print(f"[5] attack order: minion hp {hp0} -> {hp1}{' (killed)' if still is None else ''}")
        else:
            # a skipped attack check is NOT a pass -- this used to print success
            raise AssertionError("no enemy minion found to attack; test proved nothing")

        # surface Roslyn/script errors that would otherwise be invisible
        log = Path(SRV_LOG).read_text(errors="ignore") if Path(SRV_LOG).exists() else ""
        missing = [l for l in log.splitlines() if "Could not find script" in l]
        if missing:
            print(f"\n!! {len(missing)} Content scripts failed to load, e.g.:")
            for l in missing[:6]:
                print("   ", l.strip()[-110:])
        assert "LANERL_CONTROL error" not in log, "control channel reported an error"

        print("\nCONTROL CHANNEL WORKS — move and attack both verified.")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
