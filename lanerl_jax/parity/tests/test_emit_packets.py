from __future__ import annotations

import base64
import struct

import numpy as np
import pytest

from lanerl_jax.emit.packets import (
    basic_attack, health_replication, synch_sim_time, waypoint_list,
)
from lanerl_jax.emit.runtime import _decode_client_order, _path_signature
from lanerl_jax.sim.orders import OrderKind


def test_waypoint_list_matches_league_packets_layout():
    got = waypoint_list(0x11223344, -7, [(1.5, -2.0), (3.25, 4.5)])
    assert got == (struct.pack("<BIi", 0xB9, 0x11223344, -7)
                   + struct.pack("<ffff", 1.5, -2.0, 3.25, 4.5))


def test_sync_and_basic_attack_match_league_packets_layout():
    assert synch_sim_time(12_500) == struct.pack("<BIf", 0xC1, 0, 12.5)
    got = basic_attack(10, 20, extra_time_s=-0.14, missile_net_id=30,
                       attack_slot=4, target_position=(1, 2, 3))
    assert got == struct.pack("<BIIBIBfff", 0x0C, 10, 20, 114, 30, 4, 1, 2, 3)


def test_health_replication_uses_type_specific_cell_and_float_escape():
    # -0.0 starts with 0x00; this also checks the compact, one-cell payload.
    got = health_replication(77, -0.0, kind="minion", sync_id=9)
    assert got == (struct.pack("<BII", 0xC4, 0, 9) + b"\x01"
                   + struct.pack("<BIIBf", 1 << 1, 77, 1, 4, -0.0))
    with pytest.raises(ValueError, match="unsupported"):
        health_replication(1, 2, kind="nexus", sync_id=3)


def test_real_client_move_order_is_translated_from_centered_coordinates():
    raw = struct.pack("<BIBffI", 0x72, 123, 2, 10.0, -20.0, 0)
    event = {"bytes_b64": base64.b64encode(raw).decode()}
    assert _decode_client_order(event, (7000.0, 7000.0), {}) == (
        OrderKind.MOVE, 7020.0, 6960.0, -1)


def test_path_signature_ignores_motion_along_an_unchanged_path():
    waypoints = np.array([[[0.0, 0.0], [50.0, 60.0]]], np.float32)
    counts, keys = np.array([2]), np.array([1])
    first = (np.array([1.0]), np.array([2.0]), waypoints, counts, keys)
    later = (np.array([9.0]), np.array([10.0]), waypoints, counts, keys)
    assert _path_signature(first, 0) == _path_signature(later, 0)
