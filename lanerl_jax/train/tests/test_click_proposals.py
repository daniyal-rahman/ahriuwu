"""Exact mixture law, observation boundaries, migration and real PPO agreement."""
import math

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.train.click_proposals import proposal_cells, mixture_click_logits
from lanerl_jax.train.policy import LanePolicy, PolicyConfig, JointActionLogits, merge_click_proposal_params
from lanerl_jax.train.ppo import factored_log_prob, factored_entropy, PPOConfig
from lanerl_jax.train.trainer import _sample, greedy_action


def test_proposal_mixture_duplicates_mask_and_sampling():
    x, y = jnp.log(jnp.array([.3, .7])), jnp.log(jnp.array([.2, .3, .5]))
    scores = jnp.log(jnp.array([.2, .3, .5]))
    cells, valid = jnp.array([1, 1, 4]), jnp.ones(3, bool)
    gate = jnp.float32(math.log(.2/.8))
    joint = mixture_click_logits(x, y, scores, gate, cells, valid)
    expected = .8*np.outer(np.exp(x), np.exp(y)).ravel()
    expected[[1, 4]] += .1
    np.testing.assert_allclose(jax.nn.softmax(joint), expected, rtol=1e-6)
    button = jnp.full(8, -100.).at[2].set(0.)
    logits = JointActionLogits(button, x, y, jnp.float32(0), joint)
    action, lp, _ = jax.jit(jax.vmap(lambda key: _sample(logits, key, valid)))(jax.random.split(jax.random.key(0), 10000))
    flat = np.asarray(action[1]*3+action[2])
    np.testing.assert_allclose(np.bincount(flat, minlength=6)/len(flat), expected, atol=.015)
    np.testing.assert_allclose(lp, np.log(expected[flat]), atol=1e-6)
    entropy = factored_entropy((button, x, y), click_logits=joint)
    np.testing.assert_allclose(entropy, -(expected*np.log(expected)).sum(), atol=1e-6)
    mask = jnp.ones((2, 3), bool).at[0, 1].set(False)
    masked = expected.copy(); masked[1] = 0; masked /= masked.sum()
    got = factored_log_prob((button,x,y), (jnp.int32(2),jnp.int32(1),jnp.int32(1)), click_mask=mask, click_logits=joint)
    np.testing.assert_allclose(got, np.log(masked[4]), atol=1e-6)
    assert int(greedy_action(logits, mask)[1]*3+greedy_action(logits, mask)[2]) == int(masked.argmax())
    # Permuting candidates, including duplicate locations, changes no distribution.
    order = jnp.array([2,0,1])
    np.testing.assert_allclose(joint, mixture_click_logits(x,y,scores[order],gate,cells[order],valid), atol=1e-6)


def test_proposal_visibility_fallback_and_finite_gradients():
    entities = jnp.zeros((4,16)).at[:,0].set(1).at[:,5].set(1).at[:,11].set(1)
    entities = entities.at[1,11].set(0).at[2,1].set(100)
    mask = jnp.array([False,False,False,True])
    cells, valid = proposal_cells(entities, mask)
    np.testing.assert_array_equal(valid, [True,False,False,False])
    np.testing.assert_array_equal(cells[0], 48*54+27)
    x, y = jnp.array([.2,-.1]), jnp.array([.1,-.2,.3])
    def objective(scores, gate, valid):
        joint = mixture_click_logits(x,y,scores,gate,jnp.array([1,1,4,5]),valid)
        p = jax.nn.softmax(joint)
        return -(p*jax.nn.log_softmax(joint)).sum()
    for valid in (jnp.zeros(4,bool),jnp.array([True,False,True,False])):
        # Even very negative valid scores must not lose mass to invalid slots.
        score = jnp.array([-20000.,3.,-20001.,8.])
        val, grad = jax.value_and_grad(objective, argnums=(0,1))(score,jnp.float32(0.),valid)
        assert np.isfinite(val) and all(np.isfinite(a).all() for a in jax.tree.leaves(grad))
        joint = mixture_click_logits(x,y,score,jnp.float32(0.),jnp.array([1,1,4,5]),valid)
        np.testing.assert_allclose(jnp.exp(joint).sum(),1.,atol=1e-6)
        if not bool(valid.any()):
            np.testing.assert_allclose(jax.nn.softmax(joint), np.outer(jax.nn.softmax(x),jax.nn.softmax(y)).ravel(),atol=1e-6)


def small_config():
    return PolicyConfig(core='gru', core_norm=True, core_residual=True, d_model=16,
        n_heads=4,n_layers=1,ffn_dim=32,ctx_dim=32,core_dim=32,mlp_hidden=32,mlp_layers=1)


def test_proposal_migration_preserves_old_weights_and_new_heads_learn():
    old = LanePolicy(small_config()); new = LanePolicy(small_config()._replace(click_proposals=True))
    e = jnp.zeros((1,32,16)).at[0,:2,0].set(1).at[0,:2,5].set(1).at[0,:2,11].set(1)
    e = e.at[0,0,1].set(.02).at[0,1,1].set(-.02).at[0,0,3].set(.1).at[0,1,3].set(.8)
    mask = jnp.arange(32)[None,:] >= 2
    sv, gv, carry = jnp.ones((1,16)), jnp.ones((1,6)), old.initial_carry((1,))
    args = (e,mask,sv,gv,carry)
    p = old.init(jax.random.key(1),*args)
    q = merge_click_proposal_params(new.init(jax.random.key(2),*args),p)
    before, bc = old.apply(p,*args); after, ac = new.apply(q,*args)
    for a,b in zip(before,after[:4]):np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(bc,ac)
    assert set(q['params'])-set(p['params']) == {'click_proposal_query','click_proposal_gate'}
    cells,_ = proposal_cells(e,mask)
    def loss(weights):
        lg,_ = new.apply(weights,*args)
        return -jax.nn.log_softmax(lg.click_logits)[0,cells[0,0]]
    grad = jax.grad(loss)(q)
    for head in ('click_proposal_query','click_proposal_gate'):
        assert sum(float(jnp.abs(a).sum()) for a in jax.tree.leaves(grad['params'][head])) > 0


def test_proposal_real_gru_collector_and_update():
    from lanerl_jax.sim.init import init_lane,TOP_OUTER_TURRET
    from lanerl_jax.sim.config import SimConfig
    from lanerl_jax.train.vec_train import VecConfig,make_vec_train
    from lanerl_jax.parity.policy_driver import _lane_frames
    f = _lane_frames()[0]; tower = TOP_OUTER_TURRET[1]
    states=[]
    for seed in range(2):
        s=init_lane(seed=seed)
        states.append(s.replace(x=s.x.at[0].set(tower[0]-100*f.axis[0]),
                                y=s.y.at[0].set(tower[1]-100*f.axis[1])))
    bank=jax.tree.map(lambda a,b:jnp.stack([a,b]),*states)
    cfg=VecConfig(n_envs=2,rollout_steps=4,n_minibatches=2,opponent='afk',
        policy=small_config()._replace(click_proposals=True),ppo=PPOConfig.standard(epochs=1))
    built=make_vec_train(cfg,SimConfig.training().replace(step_ticks=6),bank)
    runner=built['initial_runner'](jax.random.key(0))
    _,tr,batch=jax.jit(built['rollout'])(runner)
    assert np.asarray(proposal_cells(tr.obs_entities,tr.obs_mask)[1]).any()
    lg=built['loss'].forward(runner.params,batch)
    lp=factored_log_prob((lg.button,lg.screen_x,lg.screen_y),batch['action'],batch['uses_screen'],click_logits=lg.click_logits)
    np.testing.assert_allclose(lp,batch['log_prob'],rtol=1e-4,atol=1e-5)
    updated,metrics=jax.jit(built['run_chunk'],static_argnums=1)(runner,1)
    assert all(np.isfinite(a).all() for a in jax.tree.leaves(updated.params))
    assert float(metrics['loss_nonfinite'][0]) == 0
