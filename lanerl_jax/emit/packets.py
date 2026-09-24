"""Small, audited LeaguePackets-compatible encoders used by the JAX relay.

These functions only serialize state selected by the Python emitter.  They do
not contain game mechanics.  Layouts mirror the corresponding classes in the
vendor ``LeaguePackets`` project and are covered byte-for-byte by tests.
"""
from __future__ import annotations

import struct
from collections.abc import Iterable

CHL_S2C = 3
CHL_LOW_PRIORITY = 4
RELIABLE = "reliable"
UNSEQUENCED = "unsequenced"


def _game_header(packet_id: int, sender_net_id: int = 0) -> bytes:
    if not 0 <= packet_id <= 0xFFFF:
        raise ValueError(f"packet id out of range: {packet_id}")
    if packet_id > 0xFF:
        return struct.pack("<BIH", 0xFE, sender_net_id, packet_id)
    return struct.pack("<BI", packet_id, sender_net_id)


def waypoint_list(sender_net_id: int, sync_id: int,
                  points: Iterable[tuple[float, float]]) -> bytes:
    """LeaguePackets ``WaypointList`` (0xB9), with world-space Vector2s."""
    body = bytearray(struct.pack("<i", sync_id))
    count = 0
    for x, y in points:
        body.extend(struct.pack("<ff", x, y))
        count += 1
    if count == 0:
        raise ValueError("WaypointList requires at least one point")
    return _game_header(0xB9, sender_net_id) + body


def synch_sim_time(t_ms: float) -> bytes:
    """LeaguePackets ``SynchSimTimeS2C`` (0xC1)."""
    return _game_header(0xC1) + struct.pack("<f", t_ms / 1000.0)


def basic_attack(attacker_net_id: int, target_net_id: int, *,
                 extra_time_s: float, missile_net_id: int = 0,
                 attack_slot: int = 0,
                 target_position: tuple[float, float, float] = (0, 0, 0)) -> bytes:
    """LeaguePackets ``Basic_Attack`` (0x0C).

    The 4.20 wire stores extra time as a biased signed hundredth in one byte.
    """
    biased = int(extra_time_s * 100.0) + 128
    if not 0 <= biased <= 255:
        raise ValueError(f"extra attack time does not fit wire byte: {extra_time_s}")
    x, y, z = target_position
    return (_game_header(0x0C, attacker_net_id)
            + struct.pack("<IBIBfff", target_net_id, biased, missile_net_id,
                          attack_slot, x, y, z))


def _replication_float(value: float) -> bytes:
    raw = struct.pack("<f", value)
    # Replication's float stream escapes a leading marker byte.
    return (b"\xfe" if raw[0] >= 0xFE else b"") + raw


def health_replication(net_id: int, hp: float, *, kind: str,
                       sync_id: int) -> bytes:
    """A partial ``OnReplication`` containing only current health.

    ``kind`` selects the exact primary/secondary cell used by the server's
    ReplicationHero, ReplicationLaneMinion, or ReplicationAITurret class.
    """
    cells = {"champion": (3, 0), "minion": (1, 0), "turret": (3, 0)}
    try:
        primary, secondary = cells[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported replication kind: {kind}") from exc
    value = _replication_float(hp)
    data = (struct.pack("<BI", 1 << primary, net_id)
            + struct.pack("<IB", 1 << secondary, len(value)) + value)
    return _game_header(0xC4) + struct.pack("<IB", sync_id & 0xFFFFFFFF, 1) + data
