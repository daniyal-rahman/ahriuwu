"""E60: own-state visibility, behavior-preserving migration, learner agreement."""
import jax
import jax.numpy as jnp
import numpy as np
from lanerl_jax.obs.builder import build_observation
from lanerl_jax.sim.init import init_lane,lane_params
from lanerl_jax.sim.config import SimConfig
from lanerl_jax.parity.policy_driver import _lane_frames
from lanerl_jax.train.policy import LanePolicy,PolicyConfig,OWN_ACTION_INTERFACE,expand_own_action_inputs
from lanerl_jax.train.vec_train import VecConfig,make_vec_train
from lanerl_jax.train.ppo import factored_log_prob


def test_own_action_only_exposes_own_timers():
 s=init_lane()
 s=s.replace(aa_cooldown=s.aa_cooldown.at[0].set(1.5),aa_windup=s.aa_windup.at[0].set(.2),is_attacking=s.is_attacking.at[0].set(True))
 def obs(x,own):return build_observation(x,0,_lane_frames()[0],params=lane_params(),own_action_state=own)
 old,new=obs(s,False),obs(s,True)
 assert old.self_vec.shape==(16,) and new.self_vec.shape==(20,)
 np.testing.assert_array_equal(old.self_vec,new.self_vec[:16])
 np.testing.assert_allclose(new.self_vec[16:],[.75,.2,1,0])
 enemy=s.replace(aa_cooldown=s.aa_cooldown.at[1].set(1.9),aa_windup=s.aa_windup.at[1].set(.8))
 np.testing.assert_array_equal(new.self_vec,obs(enemy,True).self_vec)


def small_config():
 return PolicyConfig(core='gru',core_norm=True,core_residual=True,d_model=16,n_heads=4,n_layers=1,ffn_dim=32,ctx_dim=32,core_dim=32,mlp_hidden=32,mlp_layers=1)


def test_own_action_migration_preserves_outputs_and_learns_new_rows():
 cfg=small_config();old=LanePolicy(cfg);new=LanePolicy(cfg._replace(self_dim=20,observation_interface=OWN_ACTION_INTERFACE))
 e=jax.random.normal(jax.random.key(1),(2,32,16));mask=jnp.zeros((2,32),bool)
 sv=jax.random.normal(jax.random.key(2),(2,16));gv=jax.random.normal(jax.random.key(3),(2,6));carry=old.initial_carry((2,))
 p=old.init(jax.random.key(4),e,mask,sv,gv,carry);q=expand_own_action_inputs(p)
 own=jnp.array([[.7,.2,1.,1.],[.1,0.,0.,0.]])
 args=(e,mask,jnp.concatenate([sv,own],-1),gv,carry)
 ref=old.apply(p,e,mask,sv,gv,carry);got=new.apply(q,*args)
 for a,b in zip(jax.tree.leaves(ref),jax.tree.leaves(got)):np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-5)
 grad=jax.grad(lambda w:new.apply(w,*args)[0].button[:,2].sum())(q)
 assert float(jnp.abs(grad['params']['Dense_1']['kernel'][16:20]).sum())>0
 assert p['params']['Dense_1']['kernel'].shape[0]==22
 assert q['params']['Dense_1']['kernel'].shape[0]==26


def test_own_action_actor_learner_agreement():
 pc=small_config()._replace(self_dim=20,observation_interface=OWN_ACTION_INTERFACE)
 cfg=VecConfig(n_envs=2,rollout_steps=4,n_minibatches=2,opponent='afk',policy=pc)
 bank=jax.tree.map(lambda a,b:jnp.stack([a,b]),init_lane(seed=0),init_lane(seed=1))
 built=make_vec_train(cfg,SimConfig.training().replace(step_ticks=6),bank)
 r=built['initial_runner'](jax.random.key(0))
 _,tr,batch=jax.jit(built['rollout'])(r)
 lg=built['loss'].forward(r.params,batch)
 lp=factored_log_prob((lg.button,lg.screen_x,lg.screen_y),batch['action'],batch['uses_screen'])
 np.testing.assert_allclose(lp,batch['log_prob'],atol=1e-5,rtol=1e-4)
 assert tr.obs_self.shape[-1]==20
