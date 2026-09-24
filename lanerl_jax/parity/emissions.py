"""Packet-emission ground truth: strict recorder I/O and scope accounting.

This module deliberately does not treat the server as an oracle for emission
rules. It only reads the plaintext delivery trace produced by
``lanerl/patch_packet_recording.py``. The JAX emitter will independently turn
``(prev_state, state, orders)`` into the same decoded event form.

Unknown packet IDs are ``unscored``, never silently ignored. That makes a new
server packet type fail the accounting gate until it is explicitly classified
as in-scope or out-of-scope. Field decoding remains LeaguePackets' job; use
``write_league_packets_input`` to make the exact JSON format consumed by the
vendored ``LeaguePacketsSerializer``.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence


Direction = Literal["in", "out"]
Scope = Literal["input", "in_scope", "out_of_scope", "unscored"]


# IDs are from LeaguePackets/GamePacketID.cs. This set is intentionally narrow:
# these are mechanics the current lane simulator models and therefore owes.
IN_SCOPE_PACKET_NAMES: dict[int, str] = {
    0x03: "Barrack_SpawnUnit",
    0x0C: "Basic_Attack",
    0x10: "UnitAddEXP",
    0x1A: "Basic_Attack_Pos",
    0x1B: "NPC_ForceDead",
    0x21: "NPC_Die_EventHistory",
    0x22: "UnitAddGold",
    0x3B: "MissileReplication",
    0x3F: "NPC_LevelUp",
    0x5A: "S2C_DestroyClientMissile",
    0x5E: "NPC_Hero_Die",
    0x61: "WaypointGroup",
    0x64: "WaypointGroupWithSpeed",
    0x68: "NPC_BuffAddGroup",
    0x6C: "S2C_ChainMissileSync",
    0x6E: "S2C_ForceCreateMissile",
    0x7B: "NPC_BuffRemove2",
    0x7C: "SpawnMinionS2C",
    0x94: "NPC_BuffRemoveGroup",
    0x9E: "NPC_Die_Broadcast",
    0xB5: "NPC_CastSpellAns",
    0xB7: "NPC_BuffAdd2",
    0xB9: "WaypointList",
    0xC4: "OnReplication",
    0xEE: "S2C_ChangeMissileTarget",
    0x10D: "S2C_ChangeMissileSpeed",
    0x110: "S2C_DestroyUnit",
    0x117: "S2C_UpdateDeathTimer",
    0x119: "S2C_UpdateBounceMissile",
    0x126: "S2C_NPC_Die_MapView",
}

# Explicitly accepted transport/setup/UI traffic. This list should grow only
# from an observed census. Gameplay-looking unknowns remain unscored.
OUT_OF_SCOPE_PACKET_NAMES: dict[int, str] = {
    0x08: "SynchSimTimeC2S",
    0x11: "S2C_EndSpawn",
    0x14: "C2S_QueryStatusReq",
    0x16: "C2S_Ping_Load_Info",
    0x28: "ServerTick",
    0x40: "S2C_MapPing",
    0x47: "C2S_UpdateGameOptions",
    0x52: "C2S_ClientReady",
    0x54: "SynchVersionS2C",
    0x56: "C2S_StatsUpdateReq",
    0x57: "C2S_MapPing",
    0x5C: "S2C_StartGame",
    0x75: "Connected",
    0x76: "SyncSimTimeFinalS2C",
    0x92: "World_SendGameNumber",
    0x95: "S2C_Ping_Load_Info",
    0xA8: "OnReplication_Acc",
    0xBD: "SynchVersionC2S",
    0xBE: "C2S_CharSelected",
    0xC1: "SynchSimTimeS2C",
    0xF0: "S2C_SetShopEnabled",
}

# Client mechanics that free-run replay must feed back to the JAX sim. They are
# not compared as server emissions, but they are first-class recorded inputs.
INPUT_PACKET_NAMES: dict[int, str] = {
    0x39: "NPC_UpgradeSpellReq",
    0x72: "NPC_IssueOrderReq",
    0x9A: "NPC_CastSpellReq",
}

GAME_PACKET_CHANNELS = frozenset({1, 2, 3, 4})
CHANNEL_NAMES = {
    0: "handshake",
    5: "chat",
    6: "quick_chat",
    7: "loading_screen",
}


def packet_id(data: bytes) -> int:
    """Return the real ID, including League's 16-bit extended packet IDs."""
    if not data:
        raise ValueError("empty packet has no ID")
    if data[0] != 0xFE:
        return data[0]
    if len(data) < 7:
        raise ValueError(f"truncated extended packet: {len(data)} bytes")
    return data[5] | data[6] << 8


@dataclass(frozen=True, slots=True)
class PacketRecord:
    sequence: int
    t_ms: float
    direction: Direction
    route: str
    client_id: int
    channel: int
    flags: int
    data: bytes

    @property
    def packet_id(self) -> int:
        return packet_id(self.data)

    @property
    def scope(self) -> Scope:
        # Packet IDs live in different namespaces on handshake/chat/loading
        # channels. A KeyCheck byte equal to 0x0C is not a Basic_Attack.
        if self.channel not in GAME_PACKET_CHANNELS:
            return "out_of_scope"
        if self.direction == "in" and self.packet_id in INPUT_PACKET_NAMES:
            return "input"
        if self.packet_id in IN_SCOPE_PACKET_NAMES:
            return "in_scope"
        if self.packet_id in OUT_OF_SCOPE_PACKET_NAMES:
            return "out_of_scope"
        return "unscored"

    @property
    def packet_name(self) -> str:
        if self.channel not in GAME_PACKET_CHANNELS:
            channel = CHANNEL_NAMES.get(self.channel, f"channel_{self.channel}")
            return f"{channel}_packet_0x{self.packet_id:X}"
        return (
            IN_SCOPE_PACKET_NAMES.get(self.packet_id)
            or OUT_OF_SCOPE_PACKET_NAMES.get(self.packet_id)
            or INPUT_PACKET_NAMES.get(self.packet_id)
            or f"UNKNOWN_0x{self.packet_id:X}"
        )


def _parse_record(raw: object, *, source: str) -> PacketRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: expected JSON object")
    required = {
        "v", "seq", "t_ms", "direction", "route", "client_id", "channel",
        "flags", "raw_packet_id", "bytes_b64",
    }
    missing = required - raw.keys()
    extra = raw.keys() - required
    if missing or extra:
        raise ValueError(f"{source}: schema mismatch missing={sorted(missing)} extra={sorted(extra)}")
    if raw["v"] != 1:
        raise ValueError(f"{source}: unsupported recorder schema v={raw['v']!r}")
    if raw["direction"] not in ("in", "out"):
        raise ValueError(f"{source}: bad direction {raw['direction']!r}")
    try:
        data = base64.b64decode(raw["bytes_b64"], validate=True)
    except (binascii.Error, TypeError) as exc:
        raise ValueError(f"{source}: invalid bytes_b64") from exc
    actual_raw_id = data[0] if data else -1
    if raw["raw_packet_id"] != actual_raw_id:
        raise ValueError(
            f"{source}: raw_packet_id={raw['raw_packet_id']} but bytes start with {actual_raw_id}"
        )
    return PacketRecord(
        sequence=int(raw["seq"]),
        t_ms=float(raw["t_ms"]),
        direction=raw["direction"],
        route=str(raw["route"]),
        client_id=int(raw["client_id"]),
        channel=int(raw["channel"]),
        flags=int(raw["flags"]),
        data=data,
    )


def load_packet_log(path: Path | str) -> list[PacketRecord]:
    """Load and fully validate a recorder JSONL file."""
    path = Path(path)
    records: list[PacketRecord] = []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            records.append(_parse_record(raw, source=f"{path}:{line_no}"))
    sequences = [record.sequence for record in records]
    expected = list(range(1, len(records) + 1))
    if sequences != expected:
        raise ValueError(
            f"{path}: seq must be contiguous from 1; first mismatch is "
            f"{next(((got, want) for got, want in zip(sequences, expected) if got != want), 'length')}"
        )
    return records


def iter_selected(
    records: Iterable[PacketRecord], *, direction: Direction = "out", client_id: int | None = None
) -> Iterator[PacketRecord]:
    for record in records:
        if record.direction == direction and (client_id is None or record.client_id == client_id):
            yield record


def census(records: Iterable[PacketRecord]) -> dict[str, object]:
    """Count every row by scope and packet type; no implicit dropping."""
    by_scope: Counter[str] = Counter()
    by_type: Counter[tuple[str, int, str]] = Counter()
    total = 0
    for record in records:
        total += 1
        by_scope[record.scope] += 1
        by_type[(record.scope, record.packet_id, record.packet_name)] += 1
    return {
        "total": total,
        "by_scope": dict(sorted(by_scope.items())),
        "by_type": [
            {"scope": scope, "packet_id": pid, "packet_name": name, "count": count}
            for (scope, pid, name), count in sorted(
                by_type.items(), key=lambda item: (item[0][0], item[0][1])
            )
        ],
    }


def write_league_packets_input(
    records: Sequence[PacketRecord], path: Path | str, *, client_id: int, direction: Direction = "out"
) -> int:
    """Write the vendored serializer's ``List<ENetPacket>`` input format."""
    selected = list(iter_selected(records, direction=direction, client_id=client_id))
    payload = [
        {
            "Time": record.t_ms,
            "Bytes": base64.b64encode(record.data).decode("ascii"),
            "Channel": record.channel,
            "Flags": record.flags,
        }
        for record in selected
    ]
    Path(path).write_text(json.dumps(payload, separators=(",", ":")))
    return len(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and census a packet recording")
    parser.add_argument("log", type=Path)
    parser.add_argument("--serializer-input", type=Path)
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--allow-unscored", action="store_true")
    args = parser.parse_args(argv)

    records = load_packet_log(args.log)
    report = census(records)
    if args.serializer_input:
        report["serializer_rows"] = write_league_packets_input(
            records, args.serializer_input, client_id=args.client_id
        )
    print(json.dumps(report, indent=2))
    return int(not args.allow_unscored and report["by_scope"].get("unscored", 0) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
