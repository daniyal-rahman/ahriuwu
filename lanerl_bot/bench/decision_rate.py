#!/usr/bin/env python
"""What does the decision rate actually cost in throughput?

The server ticks at exactly 60 Hz (Game.cs REFRESH_RATE = 1000/60), so a
decision period must be an integer number of ticks. 50 Hz -> 1.2 ticks, which
is why the requested 20 ms is not directly reachable. The legal rates near it:

    step_ticks=1 -> 60.0 Hz (16.7 ms)   <- closest to the 20 ms ask
    step_ticks=2 -> 30.0 Hz (33.3 ms)
    step_ticks=3 -> 20.0 Hz (50.0 ms)
    step_ticks=4 -> 15.0 Hz (66.7 ms)

Every decision costs one lockstep round trip, so halving the period roughly
doubles the round trips per game-second. This measures the real number with
the policy stubbed out (noop actions), which isolates the SERVER+transport
cost -- the floor that no amount of inference batching can get under.

Reports game-seconds simulated per wall-second, per instance.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

_PROJECTS = Path(__file__).resolve().parents[3]
VENDOR = _PROJECTS / "lanerl-vendor"
BIN = VENDOR / "LoLServer/GameServerConsole/bin/Release/net6.0"
CFG = _PROJECTS / "ahriuwu-lanerl/lanerl/cfg/garen1v1.json"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def measure(step_ticks: int, steps: int = 3000) -> dict:
    port = free_port()
    env = dict(os.environ)
    env.update(
        DOTNET_ROOT=str(VENDOR / "dotnet"),
        LANERL_HEADLESS="1", LANERL_FREERUN="1", LANERL_TOPONLY="1",
        LANERL_CONTROL_PORT=str(port), LANERL_STEP_TICKS=str(step_ticks),
        # match training conditions: LANERL_BOT defaults to "blue", which would
        # attach the scripted bot and bill its per-tick work to the transport
        LANERL_BOT="none",
    )
    log = Path(f"/tmp/lanerl_rate_{step_ticks}.log")
    proc = subprocess.Popen(
        [str(BIN / "GameServerConsole"), "--config", str(CFG), "--port", str(free_port())],
        cwd=str(BIN), env=env, stdout=log.open("w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        sock = None
        for _ in range(120):
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=2)
                break
            except OSError:
                time.sleep(1)
        if sock is None:
            rc = proc.poll()
            tail = "\n".join(log.read_text(errors="ignore").splitlines()[-12:])
            raise RuntimeError(
                f"server never opened control port {port} (rc={rc!r}):\n{tail}")
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        f = sock.makefile("rwb")

        noop = (json.dumps({"blue": {"t": "noop"}, "red": {"t": "noop"}}) + "\n").encode()
        obs = json.loads(f.readline())
        # warm up: JIT, first-wave spawn, GC settle -- otherwise step 1 dominates
        for _ in range(200):
            f.write(noop); f.flush(); f.readline()

        f.write(noop); f.flush()
        first = json.loads(f.readline())
        t_wall = time.perf_counter()
        for _ in range(steps):
            f.write(noop); f.flush()
            line = f.readline()
            if not line:
                raise RuntimeError("server closed the control channel mid-run")
        last = json.loads(line)
        wall = time.perf_counter() - t_wall

        game_s = (last["t"] - first["t"]) / 1000.0
        return {
            "step_ticks": step_ticks,
            "hz": round(60.0 / step_ticks, 2),
            "period_ms": round(1000.0 / (60.0 / step_ticks), 1),
            "steps": steps,
            "wall_s": round(wall, 2),
            "game_s": round(game_s, 2),
            "speedup_x": round(game_s / wall, 1),
            "ms_per_decision": round(1000.0 * wall / steps, 3),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    rates = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4]
    rows = []
    for st in rates:
        r = measure(st)
        rows.append(r)
        print(f"{r['hz']:>5.1f} Hz ({r['period_ms']:>4.1f} ms, {st} tick)  "
              f"{r['speedup_x']:>7.1f}x realtime   {r['ms_per_decision']:>6.3f} ms/decision  "
              f"[{r['game_s']}s game in {r['wall_s']}s wall]")
    out = Path(__file__).resolve().parent / "out/decision_rate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
