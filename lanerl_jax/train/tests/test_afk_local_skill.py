"""Integrated E83 reset and diagnostic contracts; no standalone benchmark."""
import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.sim.config import SimConfig
from lanerl_jax.sim.orders import Orders, OrderKind
from lanerl_jax.sim.state import Kind, MI_SLICE
from lanerl_jax.train.afk_local_skill import prepare_local_bank, diagnostic_counts, make_diagnostic_step
from lanerl_jax.train.afk_imitation import setup_collector
from lanerl_jax.train.policy import PolicyConfig
from lanerl_jax.train.vec_train import VecConfig, make_vec_train
from lanerl_jax.train.wave_scenario import START_MS


def test_local_resets_are_fresh_balanced_and_heldout(tmp_path):
    sim=SimConfig.training().replace(step_ticks=6)
    a,rows=prepare_local_bank(sim,tmp_path/'a',73,4)
    b,_=prepare_local_bank(sim,tmp_path/'b',9073,4)
    enemies=np.asarray((a.kind==Kind.LANE_MINION)&(a.team==1)&a.alive)
    np.testing.assert_array_equal(enemies.sum(-1),[1,3,1,3])
    np.testing.assert_array_equal(a.collision_present[:,MI_SLICE],a.alive[:,MI_SLICE])
    np.testing.assert_array_equal(a.aa_cooldown,0)
    np.testing.assert_array_equal(a.aa_windup,0)
    np.testing.assert_array_equal(a.spell_cooldown[:,:2],0)
    np.testing.assert_array_equal(a.level[:,:2],3)
    np.testing.assert_array_equal(a.cs[:,:2],0)
    np.testing.assert_array_equal(a.t_ms,START_MS)
    np.testing.assert_array_equal(a.next_spawn_ms,START_MS+30000)
    ratio=np.asarray(a.hp/a.max_hp)[enemies]
    assert np.all((ratio>=.10)&(ratio<=.65))
    assert not np.array_equal(a.hp,b.hp)
    assert not np.array_equal(a.x,b.x)
    assert [r['enemy_minions'] for r in rows]==[1,3,1,3]


def test_diagnostic_separates_command_target_and_readiness():
    from types import SimpleNamespace
    state=SimpleNamespace(aa_cooldown=jnp.zeros(2),aa_windup=jnp.zeros(2))
    attack=Orders(jnp.array([OrderKind.ATTACK,0]),jnp.zeros(2),jnp.zeros(2),jnp.array([5,-1]))
    wrong=attack._replace(target=jnp.array([7,-1])) if hasattr(attack,'_replace') else attack.replace(target=jnp.array([7,-1]))
    move=Orders(jnp.array([OrderKind.MOVE,0]),jnp.zeros(2),jnp.zeros(2),jnp.array([-1,-1]))
    np.testing.assert_array_equal(diagnostic_counts(state,attack,attack),[1,1,0,0,0])
    np.testing.assert_array_equal(diagnostic_counts(state,wrong,attack),[1,0,1,0,0])
    np.testing.assert_array_equal(diagnostic_counts(state,move,attack),[1,0,0,1,0])
    state.aa_cooldown=jnp.ones(2)
    np.testing.assert_array_equal(diagnostic_counts(state,attack,attack),[0,0,0,0,0])


def test_diagnostic_preserves_physical_rollout_and_terminal_cs(tmp_path):
    sim=SimConfig.training().replace(step_ticks=6)
    bank,_=prepare_local_bank(sim,tmp_path/'bank',73,4)
    pcfg=PolicyConfig(core='gru',core_norm=True,core_residual=True,
        d_model=16,n_layers=1,n_heads=2,ffn_dim=16,ctx_dim=16,core_dim=16,mlp_hidden=16,mlp_layers=1)
    cfg=VecConfig(n_envs=4,rollout_steps=4,n_minibatches=4,bank_size=4,
        episode_s=START_MS/1000+.3,stagger_initial=False,opponent='afk',cs_only=True,policy=pcfg)
    params=make_vec_train(cfg,sim,bank)['init_params'](jax.random.key(0))
    _,r,collect=setup_collector(cfg,sim,bank,params,2007)
    expected,tr,_=jax.block_until_ready(collect(r))
    built,actual,_=setup_collector(cfg._replace(rollout_steps=1),sim,bank,params,2007)
    step=make_diagnostic_step(built,sim)
    cs=[];done=[]
    for _ in range(4):
        actual,counts,c,d=jax.block_until_ready(step(actual));cs.append(c);done.append(d)
        assert counts.shape==(4,5)
    for x,y in zip(jax.tree.leaves(expected),jax.tree.leaves(actual)):
        # Typed PRNG keys cannot convert directly to NumPy.
        if jax.dtypes.issubdtype(x.dtype,jax.dtypes.prng_key):
            x,y=jax.random.key_data(x),jax.random.key_data(y)
        np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-5)
    np.testing.assert_array_equal(np.stack(cs),tr.cs_delta[:,:,0])
    np.testing.assert_array_equal(np.stack(done),tr.done_full[:,:,0])
    assert np.asarray(done).any()
    np.testing.assert_array_equal(tr.reward,tr.cs_delta)
