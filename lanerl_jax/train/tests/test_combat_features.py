"""Combat input migration and learned-feature actor/learner contract."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lanerl_jax.obs.combat_features import COMBAT_INTERFACE, COMBAT_ENTITY_DIM, COMBAT_SELF_DIM
from lanerl_jax.train.policy import LanePolicy, merge_combat_params
from lanerl_jax.train.tests.test_visible_history import small_config


@pytest.fixture(autouse=True)
def production_matmul_precision():
    # The scenario worker explicitly uses highest, while CUDA's default may
    # choose different reduced-precision kernels for sliced vs packed inputs.
    with jax.default_matmul_precision('highest'):
        yield


def config():
    return small_config()._replace(observation_interface=COMBAT_INTERFACE,
        entity_dim=COMBAT_ENTITY_DIM, self_dim=COMBAT_SELF_DIM)


def test_combat_migration_preserves_outputs_and_both_projections_learn():
    old, new = LanePolicy(small_config()), LanePolicy(config())
    e = jax.random.normal(jax.random.key(1),(2,32,COMBAT_ENTITY_DIM),dtype=jnp.float32)
    sv = jax.random.normal(jax.random.key(2),(2,COMBAT_SELF_DIM),dtype=jnp.float32)
    gv = jnp.ones((2,6),jnp.float32);mask=jnp.zeros((2,32),bool)
    carry=old.initial_carry((2,))
    p=old.init(jax.random.key(3),e[...,:16],mask,sv[...,:16],gv,carry)
    q=merge_combat_params(new.init(jax.random.key(3),e,mask,sv,gv,carry),p)
    ref=jax.jit(old.apply)(p,e[...,:16],mask,sv[...,:16],gv,carry)
    got=jax.jit(new.apply)(q,e,mask,sv,gv,carry)
    for a,b in zip(jax.tree.leaves(ref),jax.tree.leaves(got)):
        np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-6)
    for name in p['params']:
        for a,b in zip(jax.tree.leaves(p['params'][name]),jax.tree.leaves(q['params'][name])):
            np.testing.assert_array_equal(a,b)
    grad=jax.grad(lambda w:new.apply(w,e,mask,sv,gv,carry)[0].button[:,2].sum())(q)
    for name in ('combat_entities','combat_self'):
        assert float(jnp.abs(grad['params'][name]['kernel']).sum()) > 0


def test_combat_nonzero_features_collector_learner_and_update():
    from lanerl_jax.sim.config import SimConfig
    from lanerl_jax.sim.init import lane_params
    from lanerl_jax.train.wave_scenario import raw_state
    from lanerl_jax.train.vec_train import VecConfig, make_vec_train
    from lanerl_jax.train.ppo import factored_log_prob
    cfg=VecConfig(n_envs=2,rollout_steps=4,n_minibatches=2,opponent='afk',policy=config(),
                  episode_s=120.3,stagger_initial=False)
    states=[raw_state(SimpleNamespace(params=lane_params()),seed) for seed in (0,1)]
    bank=jax.tree.map(lambda a,b:jnp.stack([a,b]),*states)
    built=make_vec_train(cfg,SimConfig.training().replace(step_ticks=6),bank)
    r=built['initial_runner'](jax.random.key(0))
    for i,name in enumerate(('combat_entities','combat_self')):
        k=r.params['params'][name]['kernel']
        r.params['params'][name]['kernel']=.02*jax.random.normal(jax.random.key(7+i),k.shape,dtype=k.dtype)
    r2,tr,batch=jax.jit(built['rollout'])(r)
    assert np.asarray(tr.obs_entities[...,16:]).any()
    assert np.asarray(tr.obs_self[...,16:]).any()
    assert np.asarray(tr.done).any(), 'exercise episode reset'
    lg=built['loss'].forward(r.params,batch)
    lp=factored_log_prob((lg.button,lg.screen_x,lg.screen_y),batch['action'],batch['uses_screen'])
    np.testing.assert_allclose(lp,batch['log_prob'],atol=1e-5,rtol=1e-4)
    trained,metrics=jax.jit(built['run_chunk'],static_argnums=1)(r2,1)
    assert np.isfinite(np.asarray(metrics['policy_loss'])).all()
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(trained.params))
