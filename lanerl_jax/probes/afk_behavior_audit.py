"""DIAG: frozen AFK behavior, observation and reward positive controls (E58)."""
import json,os,shutil,sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from lanerl_jax.sim.config import SimConfig,DEFAULT_ROUTE_ARTIFACT
from lanerl_jax.sim.step import env_step
from lanerl_jax.sim.state import Kind
from lanerl_jax.sim.orders import OrderKind
from lanerl_jax.sim.combat import growth_sum
from lanerl_jax.obs.builder import build_observation
from lanerl_jax.parity.policy_driver import load_params,_lane_frames
from lanerl_jax.train.wave_scenario import prepare_scenario_bank,park_afk_opponent
from lanerl_jax.train.vec_train import VecConfig,_relative_reward
from lanerl_jax.train.trainer import _sample
from lanerl_jax.train.actions import orders_from
from lanerl_jax.train.scripted_policy import scripted_act
from lanerl_jax.train.run_manifest import git_provenance,file_sha256


def main():
 spec=json.loads(Path('experiments',sys.argv[1]+'.json').read_text())
 out=Path('/mnt/nfs/shared')/spec['id'];out.mkdir(exist_ok=False)
 scratch=Path('/scratch')/(spec['id']+'-'+os.environ['SLURM_JOB_ID']);scratch.mkdir()
 shutil.copytree(DEFAULT_ROUTE_ARTIFACT,scratch/'routes')
 jax.config.update('jax_default_matmul_precision','highest')
 jax.config.update('jax_compilation_cache_dir','/scratch/lanerl-jax-compilation-cache')
 sim=SimConfig.training(route_artifact=scratch/'routes').replace(step_ticks=6)
 bank=park_afk_opponent(prepare_scenario_bank(sim,out/'bank',[-45,-5,35,75],1007))
 frame=_lane_frames()[0];cfg=VecConfig(xp_scale=0.,health_loss_gold=100.,tower_damage_gold=900.,tower_damage_personal=True)
 observe=lambda s:build_observation(s,0,frame,params=sim.params,horizon_s=600.,vision=sim.vision)
 report={'spec':spec,'source':git_provenance(),'policies':{},'limitations':'Frozen diagnostic seeds; opportunities are instantaneous geometric/HP candidates, not proof a last hit is achievable before allies kill it. Scripted control is observation-only, not optimal. Monte Carlo value targets follow one sampled continuation, high variance. No training.'}
 # Explicit alias check: current inputs do not expose own AA timer/windup.
 s=jax.tree.map(lambda x:x[0],bank);o=observe(s)
 altered=s.replace(aa_cooldown=s.aa_cooldown.at[0].set(1.),aa_windup=s.aa_windup.at[0].set(.2))
 a=observe(altered)
 report['own_attack_timer_observation_maxdelta']=max(float(np.max(np.abs(np.asarray(x,dtype=float)-np.asarray(y,dtype=float)))) for x,y in zip((o.entities,o.self_vec,o.global_vec),(a.entities,a.self_vec,a.global_vec)))
 for entry in spec['policies']:
  name=entry['name'];source=Path(entry['checkpoint']);stage=scratch/name;stage.mkdir()
  shutil.copyfile(source,stage/'checkpoint.msgpack');shutil.copyfile(source.parent/'manifest.json',stage/'manifest.json')
  policy,params,_=load_params(str(stage/'checkpoint.msgpack'))
  scripted=entry.get('scripted',False)
  def one(s,c,k):
   k,ak=jax.random.split(k);o=observe(s)
   lg,nc=policy.apply(params,o.entities,o.entity_pad_mask,o.self_vec,o.global_vec,c)
   act,_,_=_sample(lg,ak,~o.entity_pad_mask)
   if scripted:act=scripted_act(o,ak)
   action=tuple(jnp.stack([x,jnp.zeros_like(x)]) for x in act)
   order=orders_from(action,s,None,frame,snap_moves=False,params=sim.params,vision=sim.vision,drop_unwalkable_moves=True)
   nxt=env_step(s,order,sim);reward,terms=_relative_reward(s,nxt,cfg)
   dist=jnp.hypot(s.x-s.x[0],s.y-s.y[0]);enemy=s.alive&(s.kind==Kind.LANE_MINION)&(s.team==1)
   # slot_unit is used ONLY by diagnostic, never policy.
   visible=jnp.zeros_like(enemy).at[jnp.maximum(o.slot_unit,0)].max(~o.entity_pad_mask)
   p=sim.params;ad=p['attack_damage'][s.model[0]]+p['ad_per_level'][s.model[0]]*growth_sum(s.level[0])
   dmg=ad*100/(100+jnp.maximum(p['armor'][s.model],0.))
   killable=enemy&visible&(s.hp<=dmg)
   reach=p['attack_range'][s.model[0]]+p['collision_radius'][s.model]
   near=killable&(dist<=reach);approach=killable&(dist<=500)
   target=jnp.maximum(order.target[0],0)
   aimed=(order.kind[0]==OrderKind.ATTACK)&(order.target[0]>=0)&killable[target]
   button=act[0];is_spell=(button>=3)&(button<=6)
   locked=is_spell&(o.self_vec[6+jnp.clip(button-3,0,3)]>0)
   data=dict(reward=reward[0],gold=terms['cs'][0],health=terms['health'][0],tower=terms['tower'][0],value=lg.value,
    cs=nxt.cs[0]-s.cs[0],hp=nxt.hp[0]/nxt.max_hp[0],death=nxt.deaths[0]-s.deaths[0],button=button,
    opportunity=near.any(),approach=approach.any(),ready=(s.aa_cooldown[0]<=0)&(s.aa_windup[0]<=0)&~s.buffs.e.active[0],
    targeted_killable=aimed,decoded_noop=order.kind[0]==OrderKind.NOOP,locked_spell=locked,e_active=s.buffs.e.active[0],
    aa_windup=s.aa_windup[0],aa_cooldown=s.aa_cooldown[0],enemy_deaths=(enemy&~nxt.alive).sum(),
    attack_probability=jax.nn.softmax(lg.button)[2])
   return nxt,nc,k,data
  vm=jax.vmap(one)
  @jax.jit
  def chunk(state,carry,key):
   def step(r,_):
    ns,nc,nk,d=vm(*r);return (ns,nc,nk),d
   return jax.lax.scan(step,(state,carry,key),None,length=100)
  n=spec['games'];ids=jnp.arange(n)%len(bank.t_ms)
  state=jax.tree.map(lambda x:x[ids],bank);carry=policy.initial_carry((n,));key=jax.random.split(jax.random.key(88001),n)
  chunks=[]
  for i in range(12):
   (state,carry,key),d=jax.block_until_ready(chunk(state,carry,key));chunks.append(jax.tree.map(np.asarray,d))
   print('FROZEN AUDIT',name,(i+1)*10,'seconds',flush=True)
  data={k:np.concatenate([d[k] for d in chunks]) for k in chunks[0]};np.savez_compressed(out/(name+'.npz'),**data)
  returns=np.zeros_like(data['reward']);acc=np.zeros(n)
  for t in range(len(returns)-1,-1,-1):acc=data['reward'][t]+.99*acc;returns[t]=acc
  opp=data['opportunity'];ready=opp&data['ready'];err=returns-data['value']
  def conditional(a,m):return float(a[m].mean()) if m.any() else None
  result=dict(checkpoint_sha256=file_sha256(source),scripted=scripted,games=n,
   cs_mean=float(data['cs'].sum(0).mean()),cs_per_game=data['cs'].sum(0).tolist(),deaths=float(data['death'].sum(0).mean()),
   reward_terms={k:float(data[k].sum(0).mean()) for k in ('reward','gold','health','tower')},
   visible_onehit_inrange_frames=int(opp.sum()),ready_onehit_frames=int(ready.sum()),
   targeted_killable_given_ready=conditional(data['targeted_killable'],ready),
   attack_probability_given_ready=None if scripted else conditional(data['attack_probability'],ready),
   locked_spell_fraction=float(data['locked_spell'].mean()),decoded_noop_fraction=float(data['decoded_noop'].mean()),
   buttons=np.bincount(data['button'].ravel(),minlength=8).tolist(),
   initial_value_mean=None if scripted else float(data['value'][0].mean()),
   discounted_return_initial=float(returns[0].mean()),
   mc_value_mse=None if scripted else float((err**2).mean()),
   mc_value_explained_variance=None if scripted else float(1-err.var()/max(returns.var(),1e-12)))
  report['policies'][name]=result;(out/'result.json').write_text(json.dumps(report,indent=2));print('RESULT',name,result,flush=True)
 print('PROFILE COMPLETE',flush=True)

if __name__=='__main__':main()
