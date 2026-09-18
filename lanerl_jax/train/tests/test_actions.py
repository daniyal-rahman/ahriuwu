import numpy as np

from lanerl_jax.obs.frame import make_lane_frame
from lanerl_jax.sim.init import TOP_OUTER_TURRET, init_lane
from lanerl_jax.sim.state import Team
from lanerl_jax.train.actions import orders_from
from lanerl_jax.sim.orders import OrderKind
from lanerl_rl import constants as C
from lanerl_rl.projection import screen_to_world_centred


BLUE_NEXUS = (1131.8, 1426.3)
RED_NEXUS = (12760.9, 13026.1)


def _frames():
    blue = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                           TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS)
    red = make_lane_frame(TOP_OUTER_TURRET[Team.RED],
                          TOP_OUTER_TURRET[Team.BLUE], RED_NEXUS)
    return blue, red


def test_jax_decoder_matches_reference_perspective_projection_for_both_sides():
    state = init_lane()
    blue, red = _frames()
    bx, by = 79, 11
    action = (np.asarray([C.BUTTON_INDEX["move"]] * 2, np.int32),
              np.asarray([bx] * 2, np.int32),
              np.asarray([by] * 2, np.int32),
              np.asarray([0, 0], np.int32))
    slots = np.full((2, C.N_SLOTS), -1, np.int32)
    orders = orders_from(action, state, slots, blue)

    sx, sy = float(C.SCREEN_X_VALUES[bx]), float(C.SCREEN_Y_VALUES[by])
    ds, dn = screen_to_world_centred(0.0, 0.0, sx, sy)
    expected = []
    for i, frame in enumerate((blue, red)):
        dx = ds * float(frame.axis[0]) + dn * float(frame.normal[0])
        dy = ds * float(frame.axis[1]) + dn * float(frame.normal[1])
        expected.append((float(state.x[i]) + dx, float(state.y[i]) + dy))
    np.testing.assert_allclose(np.asarray([orders.x, orders.y]).T,
                               np.asarray(expected), rtol=0, atol=2e-3)


def test_screen_grid_is_not_the_old_plus_minus_1800_world_square():
    state = init_lane()
    blue, _ = _frames()
    slots = np.full((2, C.N_SLOTS), -1, np.int32)
    action = (np.asarray([C.BUTTON_INDEX["move"]] * 2, np.int32),
              np.asarray([C.N_SCREEN_X - 1] * 2, np.int32),
              np.asarray([0] * 2, np.int32),
              np.asarray([0, 0], np.int32))
    orders = orders_from(action, state, slots, blue)
    dx = float(orders.x[0] - state.x[0])
    dy = float(orders.y[0] - state.y[0])
    assert not np.isclose(abs(dx), C.SCREEN_RADIUS)
    assert not np.isclose(abs(dy), C.SCREEN_RADIUS)


def test_minimap_move_cell_is_suppressed_but_targeted_attack_is_not():
    state = init_lane()
    blue, _ = _frames()
    slots = np.full((2, C.N_SLOTS), -1, np.int32)
    slots[1, 0] = 0
    action = (np.asarray([C.BUTTON_INDEX["move"],
                          C.BUTTON_INDEX["attack_move"]], np.int32),
              np.asarray([C.N_SCREEN_X - 1] * 2, np.int32),
              np.asarray([C.N_SCREEN_Y - 1] * 2, np.int32),
              np.asarray([0, 0], np.int32))
    orders = orders_from(action, state, slots, blue)
    assert int(orders.kind[0]) == OrderKind.NOOP
    assert int(orders.kind[1]) == OrderKind.ATTACK
