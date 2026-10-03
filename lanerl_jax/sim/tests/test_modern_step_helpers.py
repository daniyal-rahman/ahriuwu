"""Cheap checks of ``modern_step`` helpers and the modern click decoder (no full-tick compile)."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from lanerl_jax.obs.vision import VisionGrid
from lanerl_jax.sim import modern_step as MS
from lanerl_jax.sim import modern_world_types as W


def test_in_brush_reads_flag_bit_0_and_is_false_without_fog():
    flags = np.zeros((4, 4), np.int32)
    flags[1, 2] = 1                                                   # brush cell (x=2, z=1)
    flags[2, 1] = 2                                                   # wall only
    cfg = SimpleNamespace(vision=VisionGrid(jnp.asarray(flags), 50.0, -100.0, 0.0), rift=None)
    x, y = jnp.asarray([25.0, -25.0, 25.0, 900.0]), jnp.asarray([75.0, 125.0, 125.0, 75.0])
    np.testing.assert_array_equal(MS._in_brush(cfg, x, y, jnp.int32(0)), [True, False, False, False])
    assert not np.asarray(MS._in_brush(SimpleNamespace(vision=None, rift=None), x, y, None)).any()


def test_cast_id_stride_covers_every_unit():
    from lanerl_jax.sim.modern_champions.core import KIT_ID_BASE
    from lanerl_jax.sim.modern_world import layout
    n = layout()["struct0"] + 30
    assert n <= MS.CAST_ID_STRIDE
    assert (1 << 22) * MS.CAST_ID_STRIDE <= KIT_ID_BASE              # ticks < 2^22 (~38.8 h at 30 Hz)


def test_ward_is_clickable_with_a_selection_radius():
    from lanerl_jax.train.actions import _screen_to_centred_lane
    from lanerl_jax.train.modern_actions import MODERN_BUTTON_INDEX, modern_orders_from
    n = 3                                                             # Garen, Jax, a red ward
    sx, sy = 48, 27
    ds, dn = (float(v) for v in _screen_to_centred_lane(jnp.float32((sx + 0.5) / 96), jnp.float32((sy + 0.5) / 54)))
    cx, cy = 5000.0 + ds, 5000.0 + dn                                 # click point for champion 0
    state = SimpleNamespace(
        x=jnp.asarray([5000.0, 9000.0, cx + 40.0]), y=jnp.asarray([5000.0, 9000.0, cy]),
        kind=jnp.asarray([W.KIND_CHAMPION, W.KIND_CHAMPION, W.KIND_WARD]), team=jnp.asarray([0, 1, 1]),
        alive=jnp.ones((n,), bool), targetable=jnp.ones((n,), bool), visible=jnp.ones((2, n), bool),
        radius=jnp.asarray([65.0, 65.0, 1.0]),
        champ=SimpleNamespace(inventory=SimpleNamespace(item=jnp.full((2, 7), -1, jnp.int32))))
    frame = SimpleNamespace(axis=(1.0, 0.0), normal=(0.0, 1.0))
    move = MODERN_BUTTON_INDEX["move"]
    o = modern_orders_from(([move, move], [sx, sx], [sy, sy]), state, (frame, frame))
    assert int(o.attack[0]) == 2 and not bool(o.move[0])              # 40 units off the 1-unit ward: still hit
