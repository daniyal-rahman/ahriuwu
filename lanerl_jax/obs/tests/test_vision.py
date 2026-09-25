"""Brush/wall exclusions against independent segment/rectangle geometry."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.obs.vision import VisionGrid, clear_ray
from lanerl_jax.obs.fog import visible_to
from lanerl_jax.sim.state import Kind, Team


def grid(flags):
    return VisionGrid(jnp.asarray(flags, jnp.uint16), 50., 0., 0.)


@pytest.mark.parametrize('team', [0, 1])
def test_brush_hidden_state_cannot_change_policy_inputs_or_cursor_resolution(team):
    from lanerl_jax.sim.init import init_lane, lane_params
    from lanerl_jax.obs.builder import build_observation
    from lanerl_jax.parity.policy_driver import _lane_frames
    from lanerl_jax.train.server_train import screen_order
    from lanerl_jax.train.actions import orders_from
    from lanerl_jax.sim.orders import apply_orders, OrderKind
    from lanerl_rl.constants import BUTTON_INDEX, SCREEN_X_VALUES, SCREEN_Y_VALUES
    frame = _lane_frames()[team]
    own = dict(x=375., y=375., dead=False, sl=[1,1,1,1])
    choices = []
    for ix in range(len(SCREEN_X_VALUES)):
        for iy in range(len(SCREEN_Y_VALUES)):
            wire = screen_order([BUTTON_INDEX['move'], ix, iy], own, frame)
            if wire['t'] == 'click':
                choices.append(((wire['x']-575.)**2+(wire['y']-375.)**2, ix, iy, wire))
    _, ix, iy, cursor = min(choices)
    gx, gy = cursor['x'], cursor['y']
    flags = np.zeros((20,20), np.uint16)
    cx, cy = int(gx//50), int(gy//50)
    flags[cy-1:cy+2, cx-1:cx+2] = 1
    vision = grid(flags)
    s = init_lane()
    s = s.replace(alive=jnp.zeros_like(s.alive).at[:2].set(True),
        x=s.x.at[team].set(375.).at[1-team].set(gx),
        y=s.y.at[team].set(375.).at[1-team].set(gy),
        spell_level=s.spell_level.at[team].set(1))
    changed = s.replace(hp=s.hp.at[1-team].set(1.),
        gold=s.gold.at[1-team].set(9000.), cs=s.cs.at[1-team].set(50),
        level=s.level.at[1-team].set(18), spell_level=s.spell_level.at[1-team].set(3),
        spell_cooldown=s.spell_cooldown.at[1-team].set(9.))
    captured = []
    class CapturePolicy:
        def apply(self, params, *actor_inputs):
            assert len(actor_inputs) == 4
            jax.debug.callback(lambda *a: captured.append(tuple(np.array(x) for x in a)),
                               *actor_inputs, ordered=True)
            return jnp.int32(0)
    policy, params = CapturePolicy(), lane_params()
    @jax.jit
    def run(state, button, grid_):
        obs = build_observation(state, team, frame, params=params, vision=grid_)
        token = policy.apply(None, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
        action = (jnp.full(2, BUTTON_INDEX['noop'], jnp.int32).at[team].set(button+token),
                  jnp.full(2, ix, jnp.int32), jnp.full(2, iy, jnp.int32))
        order = orders_from(action, state, None, _lane_frames()[0],
                            snap_moves=False, params=params, vision=grid_)
        next_state = apply_orders(state, order, params, vision=grid_)
        return order, next_state.r_cast_ms
    for button in ('move', 'attack_move', 'r'):
        inputs = []
        for state in (s, changed):
            order, r_cast = run(state, jnp.int32(BUTTON_INDEX[button]), vision)
            jax.effects_barrier()
            inputs.append(captured[-1])
            assert int(order.target[team]) == -1
            if button != 'r':
                assert int(order.kind[team]) == (OrderKind.MOVE if button == 'move' else OrderKind.ATTACK_MOVE)
            assert float(r_cast[team]) == 0.
        for a, b in zip(*inputs):
            np.testing.assert_array_equal(a, b)
        assert bool(inputs[0][1][0])
    # Without brush the same reachable cursor really hits this enemy.
    shown, _ = run(s, jnp.int32(BUTTON_INDEX['move']), grid(np.zeros_like(flags)))
    assert int(shown.target[team]) == 1-team


def test_brush_entry_exit_and_separate_brushes():
    flags = np.zeros((4, 10), np.uint16)
    flags[1, 2:5] = 1
    flags[1, 7:9] = 1
    g = grid(flags)
    # An outside observer cannot see into brush; same brush can see within it
    # or out of it. Separate patches do not give each other vision.
    rays = [(75, 75, 125, 75), (125, 75, 175, 75),
            (125, 75, 75, 75), (125, 75, 375, 75)]
    result = jax.jit(lambda r: clear_ray(g, *r.T))(jnp.asarray(rays, jnp.float32))
    np.testing.assert_array_equal(result, [False, True, True, False])


def test_wall_see_through_corner_and_bound():
    flags = np.zeros((4, 100), np.uint16)
    flags[1, 2] = 2
    assert not bool(clear_ray(grid(flags), 75., 75., 175., 75.))
    flags[1, 2] |= 0x40
    assert bool(clear_ray(grid(flags), 75., 75., 175., 75.))
    # The segment touches a blocked side cell exactly at a grid corner.
    flags[1, 2] = 2
    assert not bool(clear_ray(grid(flags), 75., 75., 175., 175.))
    assert not bool(clear_ray(grid(flags), -1., 25., 25., 25.))
    assert not bool(clear_ray(grid(flags), 25., 25., 4500., 25.))


def oracle(flags, ray):
    """Intersect every grid rectangle with the segment; no DDA recurrence."""
    x0, y0, x1, y1 = np.asarray(ray, np.float64)/50
    start_grass = bool(flags[int(y0), int(x0)] & 1)
    end_grass = bool(flags[int(y1), int(x1)] & 1)
    for y in range(flags.shape[0]):
        for x in range(flags.shape[1]):
            low, high = 0., 1.
            for a, delta, lo in ((x0, x1-x0, x), (y0, y1-y0, y)):
                if delta == 0:
                    if not lo <= a <= lo+1:
                        high = -1.
                else:
                    ends = sorted(((lo-a)/delta, (lo+1-a)/delta))
                    low, high = max(low, ends[0]), min(high, ends[1])
            if low > high:
                continue
            f = int(flags[y, x]); grass = bool(f & 1)
            transparent = not f & 2 or bool(f & (0x40 | 0x100))
            brush_ok = (not end_grass or grass) if start_grass else not grass
            if not transparent or not brush_ok:
                return False
    return True


def test_random_rays_against_independent_geometry():
    rng = np.random.default_rng(824)
    flags = rng.choice([0, 0, 0, 1, 2, 0x42], size=(10, 10)).astype(np.uint16)
    rays = rng.uniform(1, 499, size=(256, 4)).astype(np.float32)
    result = jax.jit(lambda r: clear_ray(grid(flags), *r.T))(jnp.asarray(rays))
    expected = np.asarray([oracle(flags, r) for r in rays])
    # Conservative corner handling may hide an extra sliver, never reveal one.
    assert not np.any(np.asarray(result) & ~expected)
    assert np.count_nonzero(np.asarray(result) != expected) <= 1


def test_brush_enemy_is_not_visible_until_an_ally_sees_inside():
    flags = np.zeros((4, 10), np.uint16); flags[1, 2:5] = 1
    g = grid(flags)
    x, y = jnp.array([75., 175., 225.]), jnp.array([75.]*3)
    kind = jnp.array([Kind.CHAMPION, Kind.CHAMPION, Kind.LANE_MINION])
    teams = jnp.array([Team.BLUE, Team.RED, Team.BLUE])
    visible = visible_to(Team.BLUE, x, y, kind, teams, jnp.array([True, True, False]), g)
    assert not bool(visible[1])
    visible = visible_to(Team.BLUE, x, y, kind, teams, jnp.ones(3, bool), g)
    assert bool(visible[1])


def test_viewport_and_brush_both_gate_witnessed_cast_memory():
    from lanerl_jax.sim.orders import _record_observed_enemy_casts
    from lanerl_jax.sim.state import empty_state
    from lanerl_jax.obs.frame import make_lane_frame
    from lanerl_jax.sim.init import TOP_OUTER_TURRET
    from lanerl_rl.projection import target_on_screen
    s = empty_state()
    s = s.replace(kind=s.kind.at[:2].set(Kind.CHAMPION),
                  team=s.team.at[0].set(Team.BLUE).at[1].set(Team.RED),
                  alive=s.alive.at[:2].set(True),
                  x=s.x.at[0].set(75.).at[1].set(175.),
                  y=s.y.at[:2].set(75.))
    casts = jnp.zeros((2, 4), bool).at[1, 0].set(True)
    flags = np.zeros((4, 10), np.uint16); flags[1, 2:5] = 1
    hidden = _record_observed_enemy_casts(s, casts, grid(flags))
    assert float(hidden[0, 0]) == -1.
    shown = _record_observed_enemy_casts(s, casts, grid(np.zeros_like(flags)))
    assert float(shown[0, 0]) == 0.
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], (1131.8, 1426.3))
    # Within champion sight radius, but beyond the lower camera boundary.
    assert not target_on_screen(0., -1000.)
    s = s.replace(x=s.x.at[1].set(s.x[0]-1000*frame.normal[0]),
                  y=s.y.at[1].set(s.y[0]-1000*frame.normal[1]))
    offscreen = _record_observed_enemy_casts(s, casts)
    assert float(offscreen[0, 0]) == -1.


def test_production_configs_record_and_hash_vision_grid():
    from lanerl_jax.sim.config import SimConfig
    production = SimConfig.training(route_artifact=None)
    assert production.vision is not None
    assert SimConfig.gate(route_artifact=None).vision is not None
    assert SimConfig.scripted().vision is not None
    assert SimConfig.unit_test().vision is None
    assert production.describe()["vision"] == "map-grid-supercover-v1"
    changed = production.replace(vision=production.vision._replace(
        flags=production.vision.flags.at[0, 0].set(production.vision.flags[0, 0] ^ 1)))
    assert production.fingerprint() != changed.fingerprint()
    assert production.differing_fields(changed) == {"vision"}


def test_masked_long_rays_do_not_change_enabled_results_under_vmap():
    g = grid(np.zeros((4, 100), np.uint16))
    rays = jnp.array([[25., 25., 75., 25.], [25., 25., 4500., 25.]])
    enabled = jnp.array([[True, False], [True, True]])
    result = jax.jit(jax.vmap(lambda mask: clear_ray(g, *rays.T, enabled=mask)))(enabled)
    np.testing.assert_array_equal(result, [[True, False], [True, False]])


def test_brush_filtered_before_actor_slots_are_built():
    from lanerl_jax.sim.init import init_lane, lane_params, TOP_OUTER_TURRET
    from lanerl_jax.obs.builder import build_observation
    from lanerl_jax.obs.frame import make_lane_frame
    s = init_lane()
    s = s.replace(alive=jnp.zeros_like(s.alive).at[:2].set(True),
                  x=s.x.at[0].set(75.).at[1].set(175.), y=s.y.at[:2].set(75.))
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], (1131.8, 1426.3))
    flags = np.zeros((4, 10), np.uint16); flags[1, 2:5] = 1
    hidden = build_observation(s, 0, frame, params=lane_params(), vision=grid(flags))
    assert 1 not in np.asarray(hidden.slot_unit)
    shown = build_observation(s, 0, frame, params=lane_params(), vision=grid(np.zeros_like(flags)))
    assert 1 in np.asarray(shown.slot_unit)
