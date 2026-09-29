"""Reset contract: waves behind towers, coherent routes, swapped HP roles."""
from types import SimpleNamespace
import numpy as np
from lanerl_jax.sim.init import lane_params, TOP_LANE_PATH, TOP_OUTER_TURRET
from lanerl_jax.sim.state import Kind
from lanerl_jax.train.wave_scenario import raw_state, health_pair, point_on_path, turret_distance, START_MS


def test_scenario_starts_and_health_roles():
    s=raw_state(SimpleNamespace(params=lane_params()),0)
    assert float(s.t_ms)==START_MS
    assert float(s.next_spawn_ms)-float(s.t_ms)==30000
    for team in (0,1):
        path=np.asarray(TOP_LANE_PATH)[::1 if team==0 else -1]
        champion,_=point_on_path(path,turret_distance(path,TOP_OUTER_TURRET[team])+100)
        np.testing.assert_allclose([s.x[team],s.y[team]],champion,atol=.001)
        assert np.linalg.norm(champion-TOP_OUTER_TURRET[team])<550
        units=np.flatnonzero(np.asarray((s.kind==Kind.LANE_MINION)&(s.team==team)&s.alive))
        assert len(units)==6
        for k,u in enumerate(units):
            expected,vertex=point_on_path(path,turret_distance(path,TOP_OUTER_TURRET[team])+100-250-260*k)
            np.testing.assert_allclose([s.x[u],s.y[u]],expected,atol=.001)
            assert int(s.lane_waypoint_key[u])==vertex
            assert s.target[u]==-1
    a,b=health_pair(s)
    np.testing.assert_allclose(a.hp[:2]/a.max_hp[:2],[.7,1.],atol=1e-6)
    np.testing.assert_allclose(b.hp[:2]/b.max_hp[:2],[1.,.7],atol=1e-6)
    np.testing.assert_array_equal(a.x,b.x)
    np.testing.assert_array_equal(a.spell_cooldown,b.spell_cooldown)


def test_scenario_alive_spell_counter_keeps_team_axis():
    from lanerl_jax.train.wave_scenario_train import alive_spell_count
    buttons=np.array([[5,1],[3,4],[2,5]])
    obs=np.zeros((3,2,16));obs[1,0,14]=1
    np.testing.assert_array_equal(alive_spell_count(buttons,obs),[1,2])
