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


def test_head_usage_matches_what_orders_from_puts_on_the_wire():
    """`ppo.head_usage` (`PPO-14`) must say exactly which heads the decoder
    read for each SAMPLE, from the observation's pad mask -- the same
    observation that produced ``slot_unit``.

    * target used  <=>  the order is ATTACK or CAST_R on a real unit;
    * screen used  <=>  a move/attack_move that did not become an ATTACK
      (a Move, or the NOOP a minimap click is suppressed to -- the screen
      point decided that too);
    * and re-rolling the screen point leaves every order whose screen head
      was NOT used unchanged.

    Random slots, including empty ones, so attack_move's fallback MOVE and r
    on an empty slot are both exercised.
    """
    import jax.numpy as jnp

    from lanerl_jax.obs.builder import build_observation
    from lanerl_jax.sim.init import lane_params
    from lanerl_jax.train.ppo import head_usage

    state = init_lane()
    blue, red = _frames()
    params = lane_params()
    obs = [build_observation(state, i, f, params=params)
           for i, f in enumerate((blue, red))]
    slot_unit = jnp.stack([o.slot_unit for o in obs])
    pad = jnp.stack([o.entity_pad_mask for o in obs])
    np.testing.assert_array_equal(np.asarray(pad), np.asarray(slot_unit) < 0)
    assert np.asarray(pad).any() and (~np.asarray(pad)).any()

    rng = np.random.default_rng(0)
    B = C.BUTTON_INDEX
    valid_slots = [np.flatnonzero(~np.asarray(p)) for p in pad]
    seen = set()
    for _ in range(64):
        # half the slots drawn from the row's valid ones, half uniformly
        # (mostly empty), so both attack_move outcomes occur
        t = np.asarray([rng.choice(v) if rng.random() < 0.5
                        else rng.integers(0, C.N_SLOTS) for v in valid_slots],
                       np.int32)
        a = (np.asarray(rng.integers(0, len(C.BUTTONS), 2), np.int32),
             np.asarray(rng.integers(0, C.N_SCREEN_X, 2), np.int32),
             np.asarray(rng.integers(0, C.N_SCREEN_Y, 2), np.int32),
             t)
        o = orders_from(a, state, slot_unit, blue)
        us, ut = (np.asarray(u) for u in head_usage(
            jnp.asarray(a[0]), jnp.asarray(a[3]), ~pad))
        kind, target = np.asarray(o.kind), np.asarray(o.target)
        b = a[0]
        want_t = (np.isin(kind, [OrderKind.ATTACK, OrderKind.CAST_R])
                  & (target >= 0))
        want_s = np.isin(b, [B["move"], B["attack_move"]]) & (
            kind != OrderKind.ATTACK)
        np.testing.assert_array_equal(ut, want_t.astype(np.float32))
        np.testing.assert_array_equal(us, want_s.astype(np.float32))

        a2 = (a[0], (a[1] + 37) % C.N_SCREEN_X, (a[2] + 23) % C.N_SCREEN_Y, a[3])
        o2 = orders_from(a2, state, slot_unit, blue)
        same = ((np.asarray(o2.kind) == kind)
                & (np.asarray(o2.target) == target))
        assert same[us == 0].all(), "screen changed an order it does not reach"
        seen |= {(int(bb), float(s), float(t)) for bb, s, t in zip(b, us, ut)}
    am, r = B["attack_move"], B["r"]
    assert {(am, 0.0, 1.0), (am, 1.0, 0.0), (r, 0.0, 1.0), (r, 0.0, 0.0)} <= seen
