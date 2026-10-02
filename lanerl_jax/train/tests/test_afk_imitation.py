"""Imitation contracts: recurrent history, used action heads and frozen actor."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from lanerl_jax.train.afk_imitation import (
    actor_unchanged, discounted_returns, episode_valid, imitation_loss,
    label_batch, make_update, replay_prefix, sequence_forward)
from lanerl_jax.train.policy import ActionLogits, LanePolicy, PolicyConfig


def fixture():
    policy = LanePolicy(PolicyConfig(core='gru', core_norm=True, core_residual=True,
        detach_critic=True, d_model=16, n_layers=1, n_heads=2, ffn_dim=16,
        ctx_dim=16, core_dim=16, mlp_hidden=16, mlp_layers=1))
    rng = np.random.default_rng(7)
    data = dict(entities=jnp.asarray(rng.normal(0,.1,(2,8,32,16)),jnp.float32),
        mask=jnp.zeros((2,8,32),bool), self=jnp.asarray(rng.normal(0,.1,(2,8,16)),jnp.float32),
        global_=jnp.asarray(rng.normal(0,.1,(2,8,6)),jnp.float32),
        reset=jnp.zeros((2,8),bool).at[1,3].set(True),
        valid=jnp.ones((2,8)), action=jnp.zeros((2,8,3),jnp.int32),
        returns=jnp.ones((2,8)))
    data['global'] = data.pop('global_')
    params = policy.init(jax.random.key(3), *(data[k][:,0] for k in
        ('entities','mask','self','global')), policy.initial_carry((2,)))
    return policy, params, data


def test_full_prefix_replay_matches_online_gru_across_terminal():
    policy, params, data = fixture()
    _, full = sequence_forward(policy, params, policy.initial_carry((2,)), data)
    prefix = jax.jit(lambda p,d,t: replay_prefix(policy,p,d,t))(params,data,jnp.int32(4))
    _, suffix = sequence_forward(policy,params,prefix,{k:v[:,4:] for k,v in data.items()})
    for a,b in zip(jax.tree.leaves(full),jax.tree.leaves(suffix)):
        np.testing.assert_allclose(a[:,4:],b,atol=2e-6,rtol=2e-5)
    assert np.any(np.asarray(prefix[0]) != 0), 'must not silently zero each recurrent window'


def test_critic_fitting_preserves_actor_and_bc_can_update_it():
    policy, params, data = fixture()
    tx = optax.chain(optax.clip_by_global_norm(.5),optax.adam(1e-3))
    step = make_update(policy,tx,4,4.,critic_only=True)
    after,_,metrics = step(params,tx.init(params),data,jnp.int32(4))
    assert np.isfinite(float(metrics['loss']))
    assert actor_unchanged(params,after)
    assert not np.array_equal(params['params']['value_head']['bias'],after['params']['value_head']['bias'])
    bc = make_update(policy,tx,4,4.)
    after,_,metrics = bc(params,tx.init(params),data,jnp.int32(4))
    assert np.isfinite(float(metrics['loss']))
    assert not actor_unchanged(params,after)


def test_first_episode_and_returns_do_not_leak_next_reset():
    done=np.array([[False,False],[True,False],[False,True],[True,False]])
    valid=episode_valid(done)
    np.testing.assert_array_equal(valid,[[1,1],[1,1],[0,1],[0,0]])
    reward=jnp.array([[0.,1.,99.,99.],[0.,0.,2.,99.]])
    out=discounted_returns(reward,jnp.asarray(done.T),jnp.asarray(valid.T),.5)
    np.testing.assert_allclose(out,[[.5,1.,0.,0.],[.5,1.,2.,0.]])


def test_unused_click_coordinates_have_no_supervision_loss():
    logits=ActionLogits(jnp.zeros((1,2,8)),jnp.zeros((1,2,96)),
        jnp.zeros((1,2,54)),jnp.zeros((1,2)))
    a=jnp.array([[[0,0,0],[3,0,0]]])
    b=a.at[...,1].set(80).at[...,2].set(40)
    np.testing.assert_equal(imitation_loss(logits,a,jnp.ones((1,2)),4.)[0],
        imitation_loss(logits,b,jnp.ones((1,2)),4.)[0])
    changed=logits._replace(screen_x=logits.screen_x.at[...,0].set(10.))
    np.testing.assert_equal(imitation_loss(logits,a,jnp.ones((1,2)),4.)[0],
        imitation_loss(changed,a,jnp.ones((1,2)),4.)[0])


def test_teacher_collection_executes_labels_and_refuses_ppo():
    from lanerl_jax.train.tests.test_vec_train import _bank, _sim
    from lanerl_jax.train.scripted_policy import scripted_act
    from lanerl_jax.train.vec_train import VecConfig, make_vec_train
    pcfg=PolicyConfig(core='gru',core_norm=True,core_residual=True,
        d_model=16,n_layers=1,n_heads=2,ffn_dim=16,ctx_dim=16,core_dim=16,mlp_hidden=16,mlp_layers=1)
    cfg=VecConfig(n_envs=2,rollout_steps=3,n_minibatches=2,stagger_initial=False,
        opponent='afk',cs_only=True,policy=pcfg)
    built=make_vec_train(cfg,_sim(),_bank(),blue_actor=scripted_act)
    runner=built['initial_runner'](jax.random.key(4))
    _,tr,_=jax.jit(built['collect'])(runner)
    labels=label_batch(tr.obs_entities[:,:,0],tr.obs_mask[:,:,0],tr.obs_self[:,:,0],tr.obs_global[:,:,0])
    np.testing.assert_array_equal(np.stack([a[:,:,0] for a in tr.action],-1),labels)
    assert np.all(np.asarray(tr.action[0][:,:,1])==0)
    with pytest.raises(ValueError,match='teacher-overridden'):
        built['learn'](None,None,None)
