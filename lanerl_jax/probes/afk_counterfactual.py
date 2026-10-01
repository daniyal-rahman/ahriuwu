"""DIAG E59: matched natural versus directed-caster continuations, no training."""
import json,os,shutil,sys
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from lanerl_jax.sim.config import SimConfig,DEFAULT_ROUTE_ARTIFACT
from lanerl_jax.sim.step import env_step
from lanerl_jax.obs.builder import build_observation,NORM_DIST,NORM_AD,HP_BAR_STEPS
from lanerl_jax.parity.policy_driver import load_params,_lane_frames
from lanerl_jax.train.wave_scenario import prepare_scenario_bank,park_afk_opponent
from lanerl_jax.train.vec_train import VecConfig,_relative_reward
from lanerl_jax.train.trainer import _sample
from lanerl_jax.train.actions import orders_from
from lanerl_jax.train.scripted_policy import cell_for_offset
from lanerl_jax.train.replay_audit import serialize_replay_state
from lanerl_jax.train.run_manifest import git_provenance,file_sha256


def main():
 spec=json.loads(Path('experiments',sys.argv[1]+'.json').read_text())
 out=Path('/mnt/nfs/shared')/spec['id'];out.mkdir(exist_ok=False)
 scratch=Path('/scratch')/(spec['id']+'-'+os.environ['SLURM_JOB_ID']);scratch.mkdir()
 shutil.copytree(DEFAULT_ROUTE_ARTIFACT,scratch/'routes')
 source=Path(spec['checkpoint']);shutil.copyfile(source,scratch/'checkpoint.msgpack');shutil.copyfile(source.parent/'manifest.json',scratch/'manifest.json')
 jax.config.update('jax_default_matmul_precision','highest');jax.config.update('jax_compilation_cache_dir','/scratch/lanerl-jax-compilation-cache')
 sim=SimConfig.training(route_artifact=scratch/'routes').replace(step_ticks=6)
 bank=park_afk_opponent(prepare_scenario_bank(sim,out/'bank',[-45,-5,35,75],1007))
 policy,params,_=load_params(str(scratch/'checkpoint.msgpack'));frame=_lane_frames()[0]
 cfg=VecConfig(xp_scale=0.,health_loss_gold=100.,tower_damage_gold=900.,tower_damage_personal=True)
 def observe(s):return build_observation(s,0,frame,params=sim.params,horizon_s=600.,vision=sim.vision)
 def act(s,c,k):
  o=observe(s);lg,nc=policy.apply(params,o.entities,o.entity_pad_mask,o.self_vec,o.global_vec,c)
  nk,ak=jax.random.split(k);a,_,_=_sample(lg,ak,~o.entity_pad_mask)
  return a,nc,nk,lg,o
 def decode(a,s):
  return orders_from(tuple(jnp.stack([x,jnp.zeros_like(x)]) for x in a),s,None,frame,snap_moves=False,params=sim.params,vision=sim.vision,drop_unwalkable_moves=True)
 def choose(mask,a,b):
  return jax.tree.map(lambda x,y:jnp.where(mask.reshape(mask.shape+(1,)*(x.ndim-mask.ndim)),x,y),a,b)
 def capture_one(s,c,k):
  a,nc,nk,lg,o=act(s,c,k);e=o.entities;dist=jnp.hypot(e[:,1],e[:,2])*NORM_DIST
  # Observation-only robust one-hit caster threshold; target id only for diagnostic forks.
  caster=(jnp.arange(len(e))>=13)&(jnp.arange(len(e))<25)&~o.entity_pad_mask&(e[:,14]>.5)
  killable=(e[:,3]+.5/HP_BAR_STEPS)*290 <= o.self_vec[10]*NORM_AD
  ok=caster&killable&(dist<=500)&(dist>=30)
  slot=jnp.argmin(jnp.where(ok,dist,jnp.inf));target=o.slot_unit[slot]
  eligible=ok.any()&~s.buffs.e.active[0]&(s.aa_windup[0]<=0)&(s.target[0]!=target)&(s.t_ms>=128000)&(s.t_ms<=210000)
  nxt=env_step(s,decode(a,s),sim)
  return nxt,nc,nk,eligible,target
 @jax.jit
 def capture_chunk(s,c,k,cs,cc,ck,ct,found):
  def step(r,_):
   s,c,k,cs,cc,ck,ct,found=r
   ns,nc,nk,eligible,target=jax.vmap(capture_one)(s,c,k);take=eligible&~found
   return (ns,nc,nk,choose(take,s,cs),choose(take,c,cc),choose(take,k,ck),jnp.where(take,target,ct),found|take),None
  return jax.lax.scan(step,(s,c,k,cs,cc,ck,ct,found),None,length=100)[0]
 n=32;ids=jnp.arange(n)%len(bank.t_ms);s=jax.tree.map(lambda x:x[ids],bank);c=policy.initial_carry((n,));k=jax.random.split(jax.random.key(99001),n)
 r=(s,c,k,s,c,k,jnp.full(n,-1,jnp.int32),jnp.zeros(n,bool))
 for i in range(12):r=jax.block_until_ready(capture_chunk(*r));print('CAPTURE',i+1,int(r[-1].sum()),flush=True)
 _,_,_,cs,cc,ck,ct,found=r;ids=np.flatnonzero(np.asarray(found))[:spec['max_cases']]
 if not len(ids):raise RuntimeError('No eligible caster cases; do not infer absence of all opportunities')
 cs=jax.tree.map(lambda x:x[ids],cs);cc=cc[ids];ck=ck[ids];ct=ct[ids]
 (out/'cases.msgpack').write_bytes(serialize_replay_state(dict(state=cs,carry=cc,key=ck,target=ct)))
 # Four independent continuations per case, shared random numbers across branches.
 reps=spec['seeds'];idx=jnp.repeat(jnp.arange(len(ids)),reps*2)
 state=jax.tree.map(lambda x:x[idx],cs);carry=cc[idx];target=ct[idx]
 fork=jnp.tile(jnp.arange(2),len(ids)*reps);seedids=jnp.repeat(jnp.arange(len(ids)*reps),2)
 keys=jax.vmap(lambda z:jax.random.fold_in(jax.random.key(99002),z))(seedids)
 spawn=state.spawn_seq[jnp.arange(len(idx)),target]
 def branch_one(s,c,k,target,spawn,forced,t):
  a,nc,nk,lg,o=act(s,c,k)
  live=s.t_ms<240000
  visible=jnp.any((o.slot_unit==target)&~o.entity_pad_mask)
  dx=s.x[target]-s.x[0];dy=s.y[target]-s.y[0]
  sx,sy=cell_for_offset(dx*frame.axis[0]+dy*frame.axis[1],dx*frame.normal[0]+dy*frame.normal[1])
  force=(forced==1)&(t<20)&s.alive[target]&(s.spawn_seq[target]==spawn)&visible
  a=tuple(jnp.where(force,x,y) for x,y in zip((jnp.int32(2),sx,sy),a))
  order=decode(a,s);nxt=env_step(s,order,sim);reward,terms=_relative_reward(s,nxt,cfg)
  d=dict(reward=jnp.where(live,reward[0],0.),gold=jnp.where(live,terms['cs'][0],0.),health=jnp.where(live,terms['health'][0],0.),tower=jnp.where(live,terms['tower'][0],0.),
   value=jnp.where(live,lg.value,0.),done=~live|(nxt.t_ms>=240000),cs=jnp.where(live,nxt.cs[0]-s.cs[0],0),
   forced=force&live,decoded_target=(order.target[0]==target)&force&live)
  return jax.tree.map(lambda x,y:jnp.where(live,x,y),nxt,s),jnp.where(live,nc,c),nk,d
 @jax.jit
 def fork_chunk(state,carry,keys,t0):
  def step(r,t):
   ns,nc,nk,d=jax.vmap(branch_one,in_axes=(0,0,0,0,0,0,None))(*r,target,spawn,fork,t)
   return (ns,nc,nk),d
  return jax.lax.scan(step,(state,carry,keys),jnp.arange(100)+t0)
 chunks=[]
 for i in range(12):
  (state,carry,keys),d=jax.block_until_ready(fork_chunk(state,carry,keys,i*100));chunks.append(jax.tree.map(np.asarray,d))
  print('FORK',i+1,flush=True)
 data={key:np.concatenate([d[key] for d in chunks]) for key in chunks[0]};np.savez_compressed(out/'forks.npz',**data)
 assert np.isfinite(data['reward']).all() and np.isfinite(data['value']).all()
 summaries={}
 for health_scale in (1.,.25,0.):
  reward=data['reward']+(health_scale-1)*data['health'];entry={}
  for seconds in (2,8,120):
   length=min(int(seconds*10),len(reward));disc=(.99**np.arange(length))[:,None]
   ret=(reward[:length]*disc).sum(0).reshape(len(ids),reps,2)
   cs_result=data['cs'][:length].sum(0).reshape(len(ids),reps,2)
   entry[str(seconds)]={'mean_return':ret.mean((0,1)).tolist(),'per_case_return_difference':(ret[:,:,1]-ret[:,:,0]).mean(1).tolist(),'mean_cs':cs_result.mean((0,1)).tolist()}
  summaries[str(health_scale)]=entry
 advantage=np.zeros_like(data['reward']);acc=np.zeros(len(idx))
 for t in range(len(advantage)-1,-1,-1):
  nv=data['value'][t+1] if t+1<len(advantage) else np.zeros(len(idx));alive=1-data['done'][t]
  acc=data['reward'][t]+.99*nv*alive-data['value'][t]+.99*.95*alive*acc;advantage[t]=acc
 report=dict(spec=spec,source=git_provenance(),checkpoint_sha256=file_sha256(source),cases=len(ids),found=int(np.asarray(found).sum()),
  case_seconds=((np.asarray(cs.t_ms)-120000)/1000).tolist(),initial_value=data['value'][0].reshape(len(ids),reps,2)[:,:,0].mean(1).tolist(),
  first_step_gae=advantage[0].reshape(len(ids),reps,2).mean(1).tolist(),results=summaries,
  forced_decisions=int(data['forced'].sum()),forced_decoded_target_fraction=float(data['decoded_target'].sum()/max(data['forced'].sum(),1)),
  limitations='Directed option can override up to2s of actions, not a single action advantage. Frozen cases selected by observation-based caster opportunity and own spin/windup exclusion. Common random numbers,4 continuations/case; correlated cases and limited samples, not global policy eval. Health coefficient comparisons are reward rescoring only. GAE uses frozen critic predictions along forced paths and is diagnostic, never used for training.')
 (out/'result.json').write_text(json.dumps(report,indent=2));print('RESULT',json.dumps(report),flush=True);print('PROFILE COMPLETE',flush=True)

if __name__=='__main__':main()
