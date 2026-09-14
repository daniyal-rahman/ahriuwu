#!/usr/bin/env python
"""Drive BOTH champions from a trained checkpoint, on a server we did not spawn.

For rendered inference: the server and the League client run on the Windows
side of the dual-boot desktop (see docs/WINDOWS_RENDERED_INFERENCE.md), and the
policy runs wherever torch is -- reaching the control port through an SSH
tunnel, because ``LanerlControl`` binds ``IPAddress.Loopback`` and nothing off
that box can connect to it directly.

Why this exists rather than reusing the eval tools: ``eval_vs_bot`` and
``viz_capture`` both SPAWN their own server through ``VecLaneEnv``, which owns
the process lifecycle. Here the server is already up, already has a client
attached to it, and is blocked in ``LanerlControl``'s constructor waiting for a
trainer -- so the only thing needed is something that connects and plays.

It also drives BOTH sides, which the eval path does not: watching the agent
play itself is the point, and a scripted red would be a different thing.

The observation and action pipeline is the TRAINING one
(``LaneObservationAdapter`` / ``LaneActionEncoder``), for the reason
``eval_vs_bot`` states: a demo that builds its observation differently from
training shows the wrong policy, and that mismatch has bitten this project
twice.

Alongside the game it writes the same per-decision JSONL ``viz_capture``
produces, so ``lanerl/viz_overlay.py`` can annotate the screen recording
afterwards without replaying anything.

    # on the machine with torch, after: ssh -N -L 5200:127.0.0.1:5200 windows
    python lanerl/play_remote.py --checkpoint <ckpt> --port 5200 \
        --out lanerl/logs/play_remote.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("play_remote")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1",
                    help="use an SSH tunnel; the control channel is loopback-only")
    ap.add_argument("--port", type=int, default=5200)
    ap.add_argument("--out", default="lanerl/logs/play_remote.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-game-ms", type=int, default=900_000)
    ap.add_argument("--connect-timeout-s", type=float, default=300.0)
    ap.add_argument("--sides", default="blue,red",
                    help="which champions the policy drives. Omitting one "
                         "leaves it to whatever the server was told to do "
                         "(LANERL_BOT), which is how you get RL vs scripted.")
    args = ap.parse_args()

    import torch
    from lanerl_rl import projection
    from lanerl_rl.infer import collate_observations
    from lanerl_rl.model import LanePolicy, ModelConfig
    from lanerl_train.lane_wiring import make_lane_adapters

    sides = [s.strip() for s in args.sides.split(",") if s.strip()]

    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    net = LanePolicy(ModelConfig()).to(args.device)
    net.load_state_dict(payload["policy"] if "policy" in payload else payload)
    net.eval()

    # One adapter and one recurrent state PER SIDE. They are different
    # champions with different histories; sharing either would feed each
    # side the other's memory.
    adapters = make_lane_adapters(train_step_source=lambda: 0)
    per_side = {s: {"adapter": adapters.adapter_factory(0, s), "state": None} for s in sides}
    encoder = adapters.encoder

    # Connect, with patience: the server blocks in LanerlControl's constructor
    # until we attach, but it only reaches that constructor once the game loop
    # is running, which is after the client has been given time to join.
    #
    # An SSH tunnel makes a plain connect() useless as a readiness check: the
    # LOCAL ssh listener accepts immediately and only then dials the remote
    # side, so a connect "succeeds" against a server that is not listening
    # yet and the stream is closed a moment later. Observed exactly that --
    # the policy attached at 20:54:54, the server began listening at 20:55:04,
    # and the run ended with "0 decisions". Worse, the server then blocked in
    # AcceptTcpClient forever and the CLIENT timed out with a network error,
    # so the visible symptom was nowhere near the cause.
    #
    # So readiness is "a first observation actually arrived", not "connect
    # returned".
    # Two failure modes pull in opposite directions, and getting this wrong
    # breaks the session permanently rather than just failing.
    #
    # 1. An SSH tunnel accepts LOCALLY before dialling the remote, so a bare
    #    connect() "succeeds" against a server that is not listening yet and
    #    the stream dies a moment later. Readiness has to mean data arrived.
    # 2. But LanerlControl calls AcceptTcpClient() exactly ONCE, in its
    #    constructor. Once it has accepted us, closing the socket leaves the
    #    server holding a dead client forever -- it never accepts again, so
    #    every later retry lands in a backlog nobody reads. Retrying after an
    #    accept is worse than not retrying at all.
    #
    # So: retry only on a CLOSED connection (readline returns b""), and wait
    # patiently on a silent-but-open one. The server can be slow to speak --
    # it emits nothing until the game loop reaches a step boundary, which is
    # after the client has loaded in.
    deadline = time.time() + args.connect_timeout_s
    sock = None
    first_line = None
    attempts = 0
    while time.time() < deadline and sock is None:
        attempts += 1
        try:
            c = socket.create_connection((args.host, args.port), timeout=10)
        except OSError as exc:
            # LOUD. This path used to be silent, and a missing SSH tunnel then
            # looked identical to a slow server: the process span here for
            # minutes logging nothing while the server sat blocked in
            # AcceptTcpClient and the game client timed out at its loading
            # screen. The visible symptom was three machines away from the
            # cause.
            if attempts == 1 or attempts % 5 == 0:
                log.warning("cannot reach %s:%d (attempt %d): %s -- is the SSH "
                            "tunnel up? `ss -tlnp | grep %d` should show one.",
                            args.host, args.port, attempts, exc, args.port)
            time.sleep(2)
            continue
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        probe = c.makefile("rwb")
        while time.time() < deadline:
            c.settimeout(20)
            try:
                line = probe.readline()
            except socket.timeout:
                # Open but quiet. KEEP IT -- see (2).
                log.info("connected, waiting for the first observation...")
                continue
            except OSError:
                line = b""
            if line and line.strip():
                sock, first_line = c, line
                break
            # EOF: the peer really did go away, so a reconnect is safe.
            log.info("control channel closed before sending anything; retrying")
            try:
                c.close()
            except OSError:
                pass
            time.sleep(2)
            break
    if sock is None:
        log.error("no control channel on %s:%d after %.0fs. Is the server up, "
                  "is LANERL_CONTROL_PORT set, and is the SSH tunnel open?",
                  args.host, args.port, args.connect_timeout_s)
        return 2
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(None)
    log.info("attached at %s:%d after %d attempt(s); first observation received",
             args.host, args.port, attempts)
    f = sock.makefile("rwb")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    t0 = time.time()

    def decide(raw: dict) -> dict:
        """One wire order per driven side, plus the viz record."""
        orders: dict = {}
        rec = {"t": raw.get("t"), "sides": {}}
        for side in sides:
            st = per_side[side]
            with torch.no_grad():
                obs = st["adapter"].build(raw, side)
                batch = collate_observations([obs], device=args.device)
                if st["state"] is None:
                    st["state"] = net.initial_state(1, device=args.device)
                action, _lp, value, st["state"] = net.act(
                    batch, st["state"], deterministic=False)
                flat = {k: int(v.reshape(-1)[0]) for k, v in action.items()}
            order = encoder.encode(flat, raw, side)
            orders[side] = order

            ad = st["adapter"]
            me = ad.last_frame.champion_of_team(ad.team) if ad.last_frame else None
            if me is None:
                continue
            cam = projection.centred_on(me.x, me.y)
            slots = []
            for i, netid in enumerate(ad.last_slot_netids):
                if netid is None or not bool(ad.last_slot_valid[i]):
                    continue
                u = ad.last_frame.units.get(netid)
                if u is None:
                    continue
                sx, sy = projection.world_to_screen(cam, u.x, u.y)
                slots.append({"slot": i, "netid": int(netid), "etype": u.etype,
                              "team": int(u.team),
                              "hp_frac": 0.0 if u.mhp <= 0 else round(u.hp / u.mhp, 4),
                              "world": [round(u.x, 1), round(u.y, 1)],
                              "screen": [round(sx, 5), round(sy, 5)]})
            rec["sides"][side] = {
                "champ_world": [round(me.x, 1), round(me.y, 1)],
                "champ_hp_frac": 0.0 if me.mhp <= 0 else round(me.hp / me.mhp, 4),
                "slots": slots, "order": order,
                "value": round(float(value.reshape(-1)[0]), 4),
            }
        return orders, rec

    with out_path.open("w") as fh:
        try:
            line = first_line
            while line:
                raw = json.loads(line)
                if int(raw.get("t", 0)) > args.max_game_ms:
                    log.info("reached --max-game-ms")
                    break
                orders, rec = decide(raw)
                f.write((json.dumps(orders, separators=(",", ":")) + "\n").encode())
                f.flush()
                fh.write(json.dumps(rec) + "\n")
                n += 1
                if n % 300 == 0:
                    fh.flush()
                    champs = {u["tm"]: u for u in raw.get("u", [])
                              if u.get("k") == "Champion"}
                    cs = {t: c.get("cs") for t, c in champs.items()}
                    log.info("t=%.0fs decisions=%d cs=%s", raw.get("t", 0) / 1000.0, n, cs)
                line = f.readline()
        except KeyboardInterrupt:
            log.info("interrupted")
        finally:
            try:
                sock.close()
            except OSError:
                pass
    log.info("%d decisions in %.0fs -> %s", n, time.time() - t0, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
