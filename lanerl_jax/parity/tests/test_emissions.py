from __future__ import annotations

import base64
import json

import pytest

from lanerl_jax.parity.emissions import (
    PacketRecord,
    census,
    load_packet_log,
    packet_id,
    write_league_packets_input,
)


def _row(seq: int, data: bytes, **updates):
    row = {
        "v": 1,
        "seq": seq,
        "t_ms": 17.066667,
        "direction": "out",
        "route": "vision",
        "client_id": 0,
        "channel": 3,
        "flags": 128,
        "raw_packet_id": data[0] if data else -1,
        "bytes_b64": base64.b64encode(data).decode(),
    }
    row.update(updates)
    return row


def test_packet_id_reads_normal_and_extended_headers():
    assert packet_id(bytes([0x0C, 0, 0, 0, 0])) == 0x0C
    assert packet_id(bytes([0xFE, 0, 0, 0, 0, 0x17, 0x01])) == 0x117
    with pytest.raises(ValueError, match="truncated extended"):
        packet_id(bytes([0xFE, 0]))


def test_log_is_strict_and_census_never_drops_unknown_types(tmp_path):
    path = tmp_path / "packets.jsonl"
    rows = [
        _row(1, bytes([0x0C, 0, 0, 0, 0])),
        _row(2, bytes([0x28, 0, 0, 0, 0])),
        _row(3, bytes([0x42, 0, 0, 0, 0])),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    records = load_packet_log(path)
    report = census(records)
    assert report["total"] == 3
    assert report["by_scope"] == {"in_scope": 1, "out_of_scope": 1, "unscored": 1}
    assert any(row["packet_name"] == "UNKNOWN_0x42" for row in report["by_type"])


def test_log_rejects_gaps_and_byte_id_disagreement(tmp_path):
    path = tmp_path / "packets.jsonl"
    path.write_text(json.dumps(_row(2, b"\x0c\0\0\0\0")) + "\n")
    with pytest.raises(ValueError, match="seq must be contiguous"):
        load_packet_log(path)

    bad = _row(1, b"\x0c\0\0\0\0", raw_packet_id=13)
    path.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="bytes start"):
        load_packet_log(path)


def test_serializer_export_filters_one_client_and_preserves_bytes(tmp_path):
    records = [
        PacketRecord(1, 1.0, "out", "send", 0, 3, 128, b"\x0c\0\0\0\0"),
        PacketRecord(2, 2.0, "out", "send", 1, 3, 128, b"\x64\0\0\0\0"),
        PacketRecord(3, 3.0, "in", "receive", 0, 1, 0, b"\x72\0\0\0\0"),
    ]
    path = tmp_path / "trace.rlp.json"
    assert write_league_packets_input(records, path, client_id=0) == 1
    payload = json.loads(path.read_text())
    assert payload == [{
        "Time": 1.0,
        "Bytes": base64.b64encode(records[0].data).decode(),
        "Channel": 3,
        "Flags": 128,
    }]
