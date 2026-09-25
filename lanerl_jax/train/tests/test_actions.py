"""The model clicks coordinates; entity identity is resolved by the environment."""
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.obs.frame import make_lane_frame
from lanerl_jax.sim.init import TOP_OUTER_TURRET, init_lane
from lanerl_jax.sim.state import Team
from lanerl_jax.train.actions import orders_from
from lanerl_jax.sim.orders import OrderKind
from lanerl_rl import constants as C
from lanerl_rl.projection import screen_to_world_centred


def _frames():
    return (make_lane_frame(TOP_OUTER_TURRET[0], TOP_OUTER_TURRET[1], (1131.8, 1426.3)),
            make_lane_frame(TOP_OUTER_TURRET[1], TOP_OUTER_TURRET[0], (12760.9, 13026.1)))


def _action(button, x=54, y=27):
    return tuple(jnp.array([v, v], jnp.int32) for v in (C.BUTTON_INDEX[button], x, y))


def test_decoder_matches_reference_projection_for_both_sides():
    state = init_lane()
    blue, red = _frames()
    action = _action('move', 79, 11)
    orders = orders_from(action, state, None, blue, snap_moves=False)
    ds, dn = screen_to_world_centred(0., 0., (79.5)/96, (11.5)/54)
    expected = []
    for i, frame in enumerate((blue, red)):
        expected.append([float(state.x[i]) + ds * float(frame.axis[0]) + dn * float(frame.normal[0]),
                         float(state.y[i]) + ds * float(frame.axis[1]) + dn * float(frame.normal[1])])
    np.testing.assert_allclose(np.array([orders.x, orders.y]).T, expected, atol=2e-3)


def test_cursor_hits_entity_without_any_slot_pointer_and_miss_moves():
    state = init_lane()
    frame = _frames()[0]
    state = state.replace(x=state.x.at[0].set(1500.), y=state.y.at[0].set(12500.),
                          alive=jnp.zeros_like(state.alive).at[:2].set(True))
    cursor = orders_from(_action('move'), state, None, frame, snap_moves=False)
    state = state.replace(x=state.x.at[1].set(cursor.x[0]), y=state.y.at[1].set(cursor.y[0]))
    hit = orders_from(_action('attack_move'), state, None, frame, snap_moves=False)
    assert int(hit.kind[0]) == OrderKind.ATTACK and int(hit.target[0]) == 1
    right_click = orders_from(_action('move'), state, None, frame, snap_moves=False)
    assert int(right_click.kind[0]) == OrderKind.ATTACK and int(right_click.target[0]) == 1
    # Changing the coordinates, not a pointer, changes what is hit.
    miss = orders_from(_action('attack_move', 5, 5), state, None, frame, snap_moves=False)
    assert int(miss.kind[0]) == OrderKind.ATTACK_MOVE and int(miss.target[0]) == -1
    from lanerl_jax.sim.orders import apply_orders
    from lanerl_jax.sim.init import lane_params
    from lanerl_jax.sim.state import MoveOrder
    ordered = apply_orders(state, miss, lane_params())
    assert int(ordered.move_order[0]) == MoveOrder.ATTACK_MOVE
    # Injecting an observation-slot map cannot redirect the cursor.
    poisoned_slots = jnp.full((2, 32), 43)
    unchanged = orders_from(_action('attack_move'), state, poisoned_slots, frame, snap_moves=False)
    np.testing.assert_array_equal(unchanged.target, hit.target)
    dead = state.replace(alive=state.alive.at[1].set(False))
    assert int(orders_from(_action('attack_move'), dead, None, frame).target[0]) == -1


def test_far_tower_cannot_be_selected_via_slot_table():
    s = init_lane()
    out = orders_from(_action('attack_move'), s, jnp.full((2, 32), 43), _frames()[0])
    assert np.all(np.asarray(out.target) == -1)


def test_minimap_suppresses_attack_and_move_clicks():
    for button in ('move', 'attack_move', 'r'):
        out = orders_from(_action(button, 95, 53), init_lane(), None, _frames()[0])
        assert np.all(np.asarray(out.kind) == OrderKind.NOOP)


def test_pointer_actions_are_rejected():
    with pytest.raises(ValueError, match='entity pointers are not accepted'):
        orders_from(_action('attack_move') + (jnp.zeros(2, jnp.int32),), init_lane(), None)


def test_three_head_likelihood_tracks_coordinate_usage():
    from lanerl_jax.train.ppo import factored_log_prob, screen_head_usage
    buttons = jnp.arange(len(C.BUTTONS))
    used, target = screen_head_usage(buttons)
    np.testing.assert_array_equal(used, np.isin(np.arange(len(C.BUTTONS)),
        [C.BUTTON_INDEX[b] for b in ('move', 'attack_move', 'r')]))
    assert not np.asarray(target).any()
    logits = (jnp.zeros((8, 8)), jnp.zeros((8, 96)), jnp.zeros((8, 54)))
    actions = (buttons, jnp.zeros(8, jnp.int32), jnp.zeros(8, jnp.int32))
    np.testing.assert_allclose(factored_log_prob(logits, actions),
        -np.log(8) - np.asarray(used) * np.log(96 * 54), rtol=1e-6)


def test_ground_click_cancels_held_chase_at_ingress():
    from lanerl_jax.sim.orders import apply_orders
    from lanerl_jax.sim.init import lane_params
    state = init_lane().replace(target=init_lane().target.at[0].set(1))
    order = orders_from(_action('move'), state, None, _frames()[0])
    out = apply_orders(state, order, lane_params())
    assert int(out.target[0]) == -1
    # The historical low-level diagnostic order retains its old semantics.
    legacy = apply_orders(state, order._replace(clear_target=None), lane_params())
    assert int(legacy.target[0]) == 1
