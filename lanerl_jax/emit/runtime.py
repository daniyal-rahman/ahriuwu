"""Drive a real 4.20 client from the JAX lane simulation.

The C# process performs the normal login/loading bootstrap and then freezes.
This process owns every post-takeover tick and packet.  The default run is a
bounded smoke test deliberately ending before the next wave spawn; later-wave
object creation is not silently approximated.
"""
from __future__ import annotations

import argparse
import base64
import json
import socket
import struct
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.emit.packets import (
    CHL_LOW_PRIORITY, CHL_S2C, RELIABLE, UNSEQUENCED,
    basic_attack, health_replication, synch_sim_time, waypoint_list,
)
from lanerl_jax.sim.config import SimConfig
from lanerl_jax.sim.init import init_lane
from lanerl_jax.sim.orders import OrderKind, Orders
from lanerl_jax.sim.state import Kind, Team

TICK_MS = 1000.0 / 60.0
NEXT_WAVE_AFTER_DEFAULT_TAKEOVER_MS = 162_800.0


class Relay:
    def __init__(self, host: str, port: int, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=2.0)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"relay {host}:{port} unavailable") from exc
                time.sleep(0.5)
        self.sock.settimeout(None)
        self.file = self.sock.makefile("rwb", buffering=0)
        self.pending = bytearray()

    def send(self, value: dict) -> None:
        self.file.write(json.dumps(value, separators=(",", ":")).encode() + b"\n")

    def receive(self) -> dict:
        line = self.file.readline()
        if not line:
            raise EOFError("C# relay closed")
        return json.loads(line)

    def drain(self) -> list[dict]:
        self.sock.setblocking(False)
        out = []
        try:
            while True:
                line = self.file.readline()
                if not line:
                    break
                out.append(json.loads(line))
        except (BlockingIOError, OSError):
            pass
        finally:
            self.sock.setblocking(True)
        return out

    def packet(self, client_id: int, data: bytes, channel: int,
               flags: str = RELIABLE) -> None:
        value = {"cmd": "packet", "client_id": client_id,
                 "channel": channel, "flags": flags,
                 "bytes_b64": base64.b64encode(data).decode("ascii")}
        self.pending.extend(json.dumps(value, separators=(",", ":")).encode())
        self.pending.append(10)

    def flush(self) -> None:
        if self.pending:
            self.file.write(self.pending)
            self.pending.clear()


def _jax_category(kind: int) -> str | None:
    return {Kind.CHAMPION: "champion", Kind.LANE_MINION: "minion",
            Kind.TURRET: "turret"}.get(int(kind))


def _server_category(name: str) -> str | None:
    name = name.lower()
    if "champion" in name:
        return "champion"
    if "laneminion" in name:
        return "minion"
    if "turret" in name:
        return "turret"
    return None


def match_net_ids(state, objects: list[dict]) -> dict[int, int]:
    """Match frozen server objects to JAX slots without assuming NetId values."""
    kinds = np.asarray(state.kind)
    teams = np.asarray(state.team)
    alive = np.asarray(state.alive)
    seq = np.asarray(state.spawn_seq)
    x, y = np.asarray(state.x), np.asarray(state.y)
    usable = [o for o in objects if float(o["hp"]) > 0 and
              _server_category(o["kind"]) is not None]
    result: dict[int, int] = {}
    for category in ("champion", "minion", "turret"):
        for team, server_team in ((Team.BLUE, 100), (Team.RED, 200)):
            js = [i for i in range(len(kinds)) if alive[i] and teams[i] == team
                  and _jax_category(kinds[i]) == category]
            ss = [o for o in usable if int(o["team"]) == server_team
                  and _server_category(o["kind"]) == category]
            if len(js) != len(ss):
                raise RuntimeError(
                    f"bootstrap count mismatch for {category} team {server_team}: "
                    f"JAX={len(js)} server={len(ss)}")
            if category == "minion":
                # Both are creation ordered. Surviving recycled JAX slots retain
                # their object's spawn_seq; server NetIds increase on creation.
                js.sort(key=lambda i: int(seq[i]))
                ss.sort(key=lambda o: int(o["net_id"]))
                result.update((i, int(o["net_id"])) for i, o in zip(js, ss))
            else:
                # Turret NetId allocation order is not established; positions are.
                remaining = ss[:]
                for i in js:
                    o = min(remaining, key=lambda q:
                            (float(q["x"]) - x[i]) ** 2
                            + (float(q["y"]) - y[i]) ** 2)
                    remaining.remove(o)
                    result[i] = int(o["net_id"])
    return result


def _movement_view(state) -> tuple[np.ndarray, ...]:
    return (np.asarray(state.x), np.asarray(state.y),
            np.asarray(state.waypoints), np.asarray(state.n_waypoints),
            np.asarray(state.waypoint_key))


def _path(view: tuple[np.ndarray, ...], slot: int) -> list[tuple[float, float]]:
    xs, ys, waypoints, counts, keys = view
    x, y = float(xs[slot]), float(ys[slot])
    n, key = int(counts[slot]), int(keys[slot])
    points = [(x, y)]
    for p in waypoints[slot, max(0, key):n]:
        point = (float(p[0]), float(p[1]))
        if abs(point[0] - points[-1][0]) > 0.01 or abs(point[1] - points[-1][1]) > 0.01:
            points.append(point)
    return points


def _path_signature(view: tuple[np.ndarray, ...], slot: int) -> tuple:
    _xs, _ys, waypoints, counts, keys = view
    n, key = int(counts[slot]), int(keys[slot])
    remaining = tuple((round(float(p[0]), 2), round(float(p[1]), 2))
                      for p in waypoints[slot, max(0, key):n])
    # Current position changes every tick while following the SAME path and is
    # intentionally absent. The server publishes on SetWaypoints/path cursor
    # changes, not as a position stream.
    return key, n, remaining


def _replication_kind(kind: int) -> str:
    return {Kind.CHAMPION: "champion", Kind.LANE_MINION: "minion",
            Kind.TURRET: "turret"}[int(kind)]


def _decode_client_order(event: dict, center: tuple[float, float],
                         reverse_ids: dict[int, int]) -> tuple[int, float, float, int] | None:
    raw = base64.b64decode(event["bytes_b64"])
    if len(raw) < 18 or raw[0] != 0x72:  # NPC_IssueOrderReq
        return None
    order_type = raw[5]
    px, py = struct.unpack_from("<ff", raw, 6)
    target_net_id = struct.unpack_from("<I", raw, 14)[0]
    world_x, world_y = 2.0 * px + center[0], 2.0 * py + center[1]
    if order_type in (2, 7):  # MoveTo / AttackMove
        return OrderKind.MOVE, world_x, world_y, -1
    if order_type == 3 and target_net_id in reverse_ids:  # AttackTo
        return OrderKind.ATTACK, world_x, world_y, reverse_ids[target_net_id]
    if order_type == 10:  # Stop (the sim intentionally treats it as no-op)
        return OrderKind.STOP, 0.0, 0.0, -1
    return None


@dataclass
class Stats:
    paths: int = 0
    attacks: int = 0
    health: int = 0
    client_orders: int = 0


def run(args: argparse.Namespace) -> Stats:
    from lanerl_jax.sim.step import env_advance, env_apply

    if args.duration_ms and args.takeover_ms + args.duration_ms >= \
            NEXT_WAVE_AFTER_DEFAULT_TAKEOVER_MS and args.takeover_ms == 135_000:
        raise SystemExit("bounded smoke would cross the next unimplemented spawn; "
                         "use --duration-ms <= 27000")

    cfg = SimConfig.scripted().replace(step_ticks=1, name="client_relay")
    state = init_lane()
    one_tick = jax.jit(lambda s: env_advance(s, cfg))
    apply_orders = jax.jit(lambda s, o: env_apply(s, o, cfg))
    advance_many = jax.jit(lambda s, n: jax.lax.fori_loop(
        0, n, lambda _i, carry: one_tick(carry), s))
    ticks = int(np.ceil(args.takeover_ms / TICK_MS))
    print(f"preparing JAX state at {ticks} ticks ({args.takeover_ms:.0f} ms)", flush=True)
    state = advance_many(state, ticks)
    jax.block_until_ready(state.t_ms)
    # ``advance_many`` and the real-time one-tick executable are distinct JAX
    # programs. Compile both (and order ingress) before connecting: after the
    # handoff, a compile pause would freeze the visible client for tens of seconds.
    jax.block_until_ready(one_tick(state).t_ms)
    noop = Orders(jnp.zeros(2, jnp.int8), jnp.zeros(2, jnp.float32),
                  jnp.zeros(2, jnp.float32), jnp.full(2, -1, jnp.int8))
    jax.block_until_ready(apply_orders(state, noop).t_ms)

    relay = Relay(args.host, args.port, args.connect_timeout_s)
    relay.send({"cmd": "ready"})
    active = relay.receive()
    if active.get("event") != "active":
        raise RuntimeError(f"expected active bootstrap, got {active!r}")
    while float(state.t_ms) + 0.01 < float(active["t_ms"]):
        state = one_tick(state)
    ids = match_net_ids(state, active["objects"])
    reverse_ids = {v: k for k, v in ids.items()}
    center = float(active["map_center_x"]), float(active["map_center_y"])
    print(f"ACTIVE t={float(state.t_ms):.1f}ms, mapped {len(ids)} objects", flush=True)

    client_ids = tuple(int(x) for x in args.client_ids.split(","))
    stats = Stats()
    sync_id = int(float(state.t_ms))
    path_sigs: dict[int, tuple] = {}
    hp = np.asarray(state.hp).copy()
    kinds = np.asarray(state.kind)
    movement = _movement_view(state)
    for slot, net_id in ids.items():
        health = health_replication(net_id, float(hp[slot]),
                                    kind=_replication_kind(kinds[slot]),
                                    sync_id=sync_id)
        for client in client_ids:
            relay.packet(client, health, CHL_LOW_PRIORITY, UNSEQUENCED)
        stats.health += 1
        if kinds[slot] != Kind.TURRET:
            points = _path(movement, slot)
            for client in client_ids:
                relay.packet(client, waypoint_list(net_id, sync_id, points), CHL_S2C)
            path_sigs[slot] = _path_signature(movement, slot)
            stats.paths += 1
    relay.flush()

    began = time.monotonic()
    next_deadline = began
    next_sync = float(state.t_ms) + 10_000.0
    previous = state
    while not args.duration_ms or (time.monotonic() - began) * 1000 < args.duration_ms:
        for event in relay.drain():
            if event.get("event") != "client_packet":
                continue
            decoded = _decode_client_order(event, center, reverse_ids)
            if decoded is None:
                continue
            kind, ox, oy, target = decoded
            slot = int(event["client_id"])
            if not 0 <= slot < 2:
                continue
            kinds = np.zeros(2, np.int8)
            xs = np.zeros(2, np.float32)
            ys = np.zeros(2, np.float32)
            targets = np.full(2, -1, np.int8)
            kinds[slot], xs[slot], ys[slot], targets[slot] = kind, ox, oy, target
            order = Orders(jnp.asarray(kinds), jnp.asarray(xs), jnp.asarray(ys),
                           jnp.asarray(targets))
            state = apply_orders(state, order)
            stats.client_orders += 1

        previous, state = state, one_tick(state)
        jax.block_until_ready(state.t_ms)
        sync_id += 1
        kinds = np.asarray(state.kind)
        current_hp = np.asarray(state.hp)
        started = np.asarray(state.is_attacking) & ~np.asarray(previous.is_attacking)
        aa_targets = np.asarray(state.aa_target)
        aa_windups = np.asarray(state.aa_windup)
        movement = _movement_view(state)
        xs, ys = movement[:2]
        for slot, net_id in ids.items():
            signature = _path_signature(movement, slot)
            if kinds[slot] != Kind.TURRET and signature != path_sigs.get(slot):
                packet = waypoint_list(net_id, sync_id, _path(movement, slot))
                for client in client_ids:
                    relay.packet(client, packet, CHL_S2C)
                path_sigs[slot] = signature
                stats.paths += 1
            if abs(float(current_hp[slot]) - float(hp[slot])) > 0.001:
                packet = health_replication(net_id, float(current_hp[slot]),
                                            kind=_replication_kind(kinds[slot]),
                                            sync_id=sync_id)
                for client in client_ids:
                    relay.packet(client, packet, CHL_LOW_PRIORITY, UNSEQUENCED)
                hp[slot] = current_hp[slot]
                stats.health += 1
            if started[slot]:
                target = int(aa_targets[slot])
                if target >= 0 and target in ids:
                    tx, ty = float(xs[target]), float(ys[target])
                    pos = ((tx - center[0]) / 2.0, 0.0,
                           (ty - center[1]) / 2.0)
                    packet = basic_attack(net_id, ids[target],
                                          extra_time_s=-float(aa_windups[slot]),
                                          target_position=pos)
                    for client in client_ids:
                        relay.packet(client, packet, CHL_S2C)
                    stats.attacks += 1
        if float(state.t_ms) >= next_sync:
            packet = synch_sim_time(float(state.t_ms))
            for client in client_ids:
                relay.packet(client, packet, CHL_S2C)
            next_sync += 10_000.0
        relay.flush()
        next_deadline += TICK_MS / 1000.0
        delay = next_deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.25:
            raise RuntimeError(f"JAX relay fell {-delay:.3f}s behind real time")
    print(f"smoke complete: {stats}", flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5202)
    parser.add_argument("--takeover-ms", type=float, default=135_000)
    parser.add_argument("--duration-ms", type=float, default=20_000,
                        help="0 runs until interrupted")
    parser.add_argument("--client-ids", default="0")
    parser.add_argument("--connect-timeout-s", type=float, default=900)
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)


if __name__ == "__main__":
    main()
