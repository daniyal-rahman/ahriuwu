from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from lanerl_jax.data.modern_map import ModernMapGrid
from lanerl_jax.data.modern_routes import build_routes,load_routes,clear_segment
from lanerl_jax.sim.modern_pathing import route_next,segment_clear


def test_routes_go_around_wall_and_are_jittable(tmp_path):
    flags=np.zeros((12,12),np.uint16);flags[2:10,6]=2
    g=ModernMapGrid(flags,np.zeros((12,12,4),np.uint8),np.zeros((13,13),np.float32),
                    (25,25),50,(0,-1,0),(600,1,600))
    m=build_routes(g,tmp_path/'routes',spacing=50,radius=10)
    routes,_=load_routes(tmp_path/'routes',g)
    src=jnp.array([225.,225.]);goal=jnp.array([425.,225.])
    assert not clear_segment(g,np.asarray(src),np.asarray(goal),10)
    f=jax.jit(lambda p:route_next(p,goal,10.,routes,g.as_jax()))
    point=src
    for _ in range(30):
        nxt,ok=f(point)
        assert bool(ok)
        assert clear_segment(g,np.asarray(point),np.asarray(nxt),10)
        if np.linalg.norm(np.asarray(nxt-goal))<.01:break
        assert np.linalg.norm(np.asarray(nxt-point))>.01
        point=nxt
    else:raise AssertionError('route failed to reach goal')
    assert not bool(segment_clear(src,goal,10,g.as_jax()))


def test_route_artifact_rejects_different_terrain(tmp_path):
    from dataclasses import replace
    import pytest
    flags=np.zeros((4,4),np.uint16)
    g=ModernMapGrid(flags,np.zeros((4,4,4),np.uint8),np.zeros((5,5),np.float32),
                    (25,25),50,(0,-1,0),(200,1,200))
    build_routes(g,tmp_path/'routes',spacing=50,radius=10)
    f=flags.copy();f[1,1]=2
    with pytest.raises(ValueError,match='terrain mismatch'):
        load_routes(tmp_path/'routes',replace(g,flags=f))
