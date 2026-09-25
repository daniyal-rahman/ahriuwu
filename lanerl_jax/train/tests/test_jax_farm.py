"""Task boundaries, reset isolation and visibility requirements for JAX farming."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_rl.constants import BUTTON_INDEX, SCREEN_X_VALUES, SCREEN_Y_VALUES
from lanerl_jax.sim.config import SimConfig
from lanerl_jax.sim.init import init_lane
from lanerl_jax.sim.step import env_step
from lanerl_jax.train.jax_farm import JaxFarmCollector, validate_wave_start, WAVE_START_POS
from lanerl_jax.train.server_train import screen_order


def test_wave_start_rejects_death_even_if_champion_has_respawned():
    s = init_lane().replace(t_ms=jnp.float32(120000.))
    s = s.replace(x=s.x.at[0].set(WAVE_START_POS[0]),
                  y=s.y.at[0].set(WAVE_START_POS[1]))
    validate_wave_start(s)
    with pytest.raises(RuntimeError, match='died or farmed'):
        validate_wave_start(s.replace(deaths=s.deaths.at[0].set(1)))
    with pytest.raises(RuntimeError, match='died or farmed'):
        validate_wave_start(s.replace(cs=s.cs.at[0].set(1)))
    with pytest.raises(RuntimeError, match='missed'):
        validate_wave_start(s.replace(x=s.x.at[0].add(101.)))
    with pytest.raises(RuntimeError, match='before 120'):
        validate_wave_start(s.replace(t_ms=jnp.float32(119000.)))


def test_no_radius_only_or_waveless_collector(tmp_path):
    with pytest.raises(ValueError, match='map visibility and top-lane waves'):
        JaxFarmCollector(1, tmp_path, sim_config=SimConfig.unit_test())
    with pytest.raises(ValueError, match='after 120'):
        JaxFarmCollector(1, tmp_path, episode_s=120., start_near_wave=True)


@pytest.fixture(scope='module')
def collector(tmp_path_factory):
    # Explicit no-route CONTROL: tests reset/actor boundaries, not navigation.
    c = JaxFarmCollector(2, tmp_path_factory.mktemp('jax-farm'), episode_s=.05,
                         sim_config=SimConfig.training(route_artifact=None))
    yield c
    c.close()


def test_blue_only_observation_and_selective_reset(collector):
    c = collector
    obs, stats = c.observe()
    assert obs.entities.shape[:2] == (2, 32)
    assert stats.shape == (2, 3)
    np.testing.assert_array_equal(c.spell_ranks(), [[0, 0, 1, 0]]*2)
    initial_hp = np.asarray(c.states.max_hp[:, 0]).copy()
    initial_red = np.asarray(c.states.x[:, 1]).copy()
    a = np.zeros((2, 3), np.int32); a[:, 0] = BUTTON_INDEX['noop']
    assert c.step(a).all()
    before_t = np.asarray(c.states.t_ms).copy()
    before_cs = np.asarray(c.states.cs).copy()
    c.restart_done([True, False])
    assert float(c.states.t_ms[0]) < before_t[0]
    assert float(c.states.t_ms[1]) == before_t[1]
    np.testing.assert_array_equal(c.states.cs[1], before_cs[1])
    np.testing.assert_allclose(c.states.max_hp[:, 0], initial_hp)
    np.testing.assert_allclose(c.states.x[:, 1], initial_red)
    assert c.episodes == [1, 0]
    assert json.loads((c.out / 'collector.json').read_text())['control'] is True


def test_entity_pointer_and_invalid_screen_actions_rejected(collector):
    with pytest.raises(ValueError, match='button/x/y'):
        collector.step(np.zeros((2, 4), np.int32))
    bad = np.zeros((2, 3), np.int32); bad[0, 1] = 99999
    with pytest.raises(ValueError, match='outside'):
        collector.step(bad)


def test_ground_clicks_reach_physics_without_decoder_terrain_snap(collector, monkeypatch):
    c = collector
    from lanerl_jax.sim.terrain_jax import map1_terrain
    terrain = map1_terrain()
    walkable = np.asarray(terrain.walkable)
    points = {}
    champion = {'x': float(c.states.x[0, 0]), 'y': float(c.states.y[0, 0])}
    for ix in range(len(SCREEN_X_VALUES)):
        for iy in range(len(SCREEN_Y_VALUES)):
            wire = screen_order([BUTTON_INDEX['move'], ix, iy], champion, c.frame)
            if wire['t'] != 'click':
                continue
            cx = int(np.floor((wire['x'] - terrain.min_x) / terrain.cell_size))
            cy = int(np.floor((wire['y'] - terrain.min_y) / terrain.cell_size))
            offgrid = not (0 <= cx < walkable.shape[1] and 0 <= cy < walkable.shape[0])
            label = 'offgrid' if offgrid else ('blocked' if not walkable[cy, cx] else None)
            if label:
                points.setdefault(label, (ix, iy, wire))
    assert set(points) == {'offgrid', 'blocked'}
    captured = []
    def capture(state, orders, enabled):
        captured.append(orders)
        return state
    monkeypatch.setattr(c, '_step_states', capture)
    for button in ('move', 'attack_move'):
        for ix, iy, wire in points.values():
            action = np.zeros((c.n, 3), np.int32)
            action[:, 0] = BUTTON_INDEX['noop']
            action[0] = [BUTTON_INDEX[button], ix, iy]
            c.step(action)
            got = captured[-1]
            from lanerl_jax.sim.orders import OrderKind
            assert int(got.kind[0, 0]) == (
                OrderKind.MOVE if button == 'move' else OrderKind.ATTACK_MOVE)
            np.testing.assert_allclose([got.x[0, 0], got.y[0, 0]],
                                       [wire['x'], wire['y']], rtol=0, atol=.002)


def test_rank_command_time_only_advances_environment_that_gained_rank(collector):
    c = collector
    c.states = c.states.replace(xp=c.states.xp.at[0, 0].set(
        c.sim.params["xp_to_reach_level"][2]))
    before = np.asarray(c.states.t_ms).copy()
    action = np.zeros((2, 3), np.int32); action[:, 0] = BUTTON_INDEX['noop']
    c.step(action)
    decision_ms = c.sim.step_ticks * c.sim.delta_ms
    np.testing.assert_allclose(np.asarray(c.states.t_ms) - before,
                               [2*decision_ms, decision_ms], atol=.01)
    assert int(c.states.spell_level[0, 0].sum()) == 2


def test_cpu_batch_matches_scalar_physics_and_preserves_paused_peer(collector):
    c = collector
    if jax.default_backend() == 'cpu':
        assert c.batch_mode == 'map'
    # Exercise a wave-spawn tick in one environment with a paused early-game
    # peer, rather than comparing identical empty-lane states.
    before = c.states.replace(t_ms=c.states.t_ms.at[0].set(90000.))
    actual = c._step_states(before, c._noop_orders, [True, False])
    scalar_state = jax.tree.map(lambda a: a[0], before)
    scalar_orders = jax.tree.map(lambda a: a[0], c._noop_orders)
    # Independent scalar env_step: catches semantic changes from batch control
    # flow without constructing the expected result with the collector kernel.
    expected = jax.jit(lambda s, o: env_step(s, o, c.sim))(scalar_state, scalar_orders)
    for result, one, original in zip(jax.tree.leaves(actual),
                                     jax.tree.leaves(expected),
                                     jax.tree.leaves(before)):
        if jax.dtypes.issubdtype(result.dtype, jax.dtypes.prng_key):
            result, one, original = map(jax.random.key_data, (result, one, original))
        np.testing.assert_allclose(result[0], one, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(result[1], original[1])
