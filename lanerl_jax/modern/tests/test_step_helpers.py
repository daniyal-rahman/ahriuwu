"""Cheap checks of world helpers and the modern click decoder (no full-tick compile)."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from lanerl_jax.modern.core import types as W
from lanerl_jax.modern.world.phases.attack import CAST_ID_STRIDE
from lanerl_jax.modern.world.views import in_brush
from lanerl_jax.modern.rays import VisionGrid


def test_in_brush_reads_flag_bit_0_and_is_false_without_fog():
    flags = np.zeros((4, 4), np.int32)
    flags[1, 2] = 1                                                   # brush cell (x=2, z=1)
    flags[2, 1] = 2                                                   # wall only
    cfg = SimpleNamespace(vision=VisionGrid(jnp.asarray(flags), 50.0, -100.0, 0.0), rift=None)
    x, y = jnp.asarray([25.0, -25.0, 25.0, 900.0]), jnp.asarray([75.0, 125.0, 125.0, 75.0])
    np.testing.assert_array_equal(in_brush(cfg, x, y, jnp.int32(0)), [True, False, False, False])
    assert not np.asarray(in_brush(SimpleNamespace(vision=None, rift=None), x, y, None)).any()


def test_cast_id_stride_covers_every_unit():
    from lanerl_jax.modern.champions.core import KIT_ID_BASE
    from lanerl_jax.modern.world.config import Layout
    n = Layout().n_units                                          # the largest (full-map) world
    assert n <= CAST_ID_STRIDE
    assert (1 << 22) * CAST_ID_STRIDE <= KIT_ID_BASE              # ticks < 2^22 (~38.8 h at 30 Hz)


SX, SY = 48, 27                                                       # the screen centre cell


def _click_state(kind, sub, radius, dx):
    """Garen at (5000, 5000), Jax far away, one red unit ``dx`` units right of champion 0's centre click."""
    from lanerl_jax.modern.screen import screen_to_lane
    ds, dn = (float(v) for v in screen_to_lane(jnp.float32((SX + 0.5) / 96), jnp.float32((SY + 0.5) / 54)))
    cx, cy = 5000.0 + ds, 5000.0 + dn
    n = 3
    state = SimpleNamespace(
        x=jnp.asarray([5000.0, 9000.0, cx + dx]), y=jnp.asarray([5000.0, 9000.0, cy]),
        kind=jnp.asarray([W.KIND_CHAMPION, W.KIND_CHAMPION, kind]), team=jnp.asarray([0, 1, 1]),
        sub=jnp.asarray([0, 0, sub], jnp.int32), alive=jnp.ones((n,), bool), targetable=jnp.ones((n,), bool),
        visible=jnp.ones((2, n), bool), radius=jnp.asarray([65.0, 65.0, radius]),
        champ=SimpleNamespace(inventory=SimpleNamespace(item=jnp.full((2, 7), -1, jnp.int32))))
    return state


def _click(state, button):
    from lanerl_jax.modern.actions import MODERN_BUTTON_INDEX, modern_orders_from
    b = MODERN_BUTTON_INDEX[button]
    frame = SimpleNamespace(axis=(1.0, 0.0), normal=(0.0, 1.0))
    return modern_orders_from(([b, b], [SX, SX], [SY, SY]), state, (frame, frame))


def test_ward_is_clickable_with_a_selection_radius():
    state = _click_state(W.KIND_WARD, 0, 1.0, 40.0)
    o = _click(state, "move")
    assert int(o.attack[0]) == 2 and not bool(o.move[0])              # 40 units off the 1-unit ward: still hit


def test_clicks_use_selection_radii_not_hitboxes():
    for dx, hit in ((100.0, True), (130.0, False)):                    # caster minion: hitbox 48, selection 115
        state = _click_state(W.KIND_MINION, 1, 48.0, dx)
        o = _click(state, "move")
        assert (int(o.attack[0]) == 2) == hit and bool(o.move[0]) != hit, dx


def test_stop_button_issues_a_stop_order():
    state = _click_state(W.KIND_MINION, 1, 48.0, 500.0)
    o = _click(state, "stop")
    assert bool(o.stop[0]) and not bool(o.move[0]) and int(o.attack[0]) == -1
