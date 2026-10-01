"""DIAG: real AFK reward -> GAE -> PPO probability changes; no exported weights."""
import json,os,shutil,sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import msgpack_restore,from_state_dict
from lanerl_jax.sim.config import SimConfig,DEFAULT_ROUTE_ARTIFACT
from lanerl_jax.train.run_manifest import git_provenance,file_sha256
from lanerl_jax.train.wave_scenario import prepare_scenario_bank,park_afk_opponent
from lanerl_jax.train.policy import PolicyConfig
from lanerl_jax.train.ppo import PPOConfig,gae,factored_log_prob
from lanerl_jax.train.vec_train import VecConfig,make_vec_train
from lanerl_jax.train.replay_audit import restore_replay_state
from lanerl_jax.obs.builder import build_observation
from lanerl_jax.parity.policy_driver import _lane_frames


def branch_audit(built,cfg,sim,bank,before,after,tr,batch,hits,production_params):
 """Same observations/advantages/Adam state/RNG; vary only update settings."""
 old=built['loss'].forward(before.params,batch)
 def measure(params):
  new=built['loss'].forward(params,batch);out={}
  for i,name in enumerate(('button','screen_x','screen_y')):
   a=jax.nn.log_softmax(getattr(old,name));b=jax.nn.log_softmax(getattr(new,name))
   action=batch['action'][i][...,None]
   delta=np.asarray(jnp.take_along_axis(b-a,action,-1)[...,0])
   kl=np.asarray((jnp.exp(a)*(a-b)).sum(-1))
   mask=hits if i==0 else hits & np.asarray(batch['uses_screen']).astype(bool)
   out[name]={'hit_count':int(mask.sum()),'hit_mean_delta_logprob':float(delta[mask].mean()),
    'hit_probability_increased':float((delta[mask]>0).mean()),'all_mean_exact_kl':float(kl.mean())}
  return out
 result={}
 for label,changes in [('standard',{}),('lr3e5',{'lr':3e-5}),('one_epoch',{'epochs':1}),('no_entropy',{'entropy_coef':0.})]:
  pc=cfg.ppo._replace(**changes)
  variant=make_vec_train(cfg._replace(ppo=pc),sim,bank)
  result_runner,metrics=jax.block_until_ready(jax.jit(variant['learn'])(after,tr,before.carry))
  params=result_runner.params
  result[label]={'heads':measure(params),'metrics':{k:float(v) for k,v in metrics.items()}}
  if label=='standard':
   err=max(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(production_params)))
   assert err<1e-6,err
   result[label]['production_param_max_error']=err
 return result

def main():
 spec=json.loads(Path('experiments',sys.argv[1]+'.json').read_text())
 out=Path('/mnt/nfs/shared')/spec['id'];out.mkdir(exist_ok=False)
 scratch=Path('/scratch')/(spec['id']+'-'+os.environ['SLURM_JOB_ID']);scratch.mkdir()
 shutil.copytree(DEFAULT_ROUTE_ARTIFACT,scratch/'routes')
 jax.config.update('jax_default_matmul_precision','highest')
 jax.config.update('jax_compilation_cache_dir','/scratch/lanerl-jax-compilation-cache')
 sim=SimConfig.training(route_artifact=scratch/'routes').replace(step_ticks=6)
 bank=park_afk_opponent(prepare_scenario_bank(sim,out/'bank',[-60,-20,20,60],0))
 cfg=VecConfig(n_envs=128,rollout_steps=128,n_updates=128,n_minibatches=4,
  episode_s=240.,observation_horizon_s=600.,stagger_initial=False,bank_size=8,
  opponent='afk',xp_scale=0.,health_loss_gold=100.,tower_damage_gold=900.,tower_damage_personal=True,
  policy=PolicyConfig(core='gru',core_norm=True,core_residual=True,detach_critic=True),
  ppo=PPOConfig.standard(lr=1e-4,entropy_coef=.001))
 built=make_vec_train(cfg,sim,bank);roll=jax.jit(built['rollout']);learn=jax.jit(built['learn'])
 frame=_lane_frames()[0]
 @jax.jit
 def end_value(r):
  def one(s,c):
   o=build_observation(s,0,frame,params=sim.params,horizon_s=600.,vision=sim.vision)
   lg,_=built['policy'].apply(r.params,o.entities,o.entity_pad_mask,o.self_vec,o.global_vec,c[0]);return lg.value
  return jax.vmap(one)(r.env_state,r.carry)
 @jax.jit
 def forward(p,b):
  lg=built['loss'].forward(p,b)
  lp=factored_log_prob((lg.button,lg.screen_x,lg.screen_y),b['action'],b['uses_screen'])
  return lp,lg.value
 def rows(x):return np.asarray(x)[:,:,0].T
 report={'spec':spec,'source':git_provenance(),'checkpoint_sha256':{},'stages':{},'limitations':'Diagnostic updates on isolated copies, no saved policy. Same recorded observations/carry for probability comparisons. Cohorts are correlations, not counterfactual action causality. Future-hit windows limited to1s within each128-step rollout; no crossing episode ends. Normalized advantages average actual4 epoch minibatch normalizations. Anchor compares fixed old recurrent histories, not new-policy state coverage.'}
 for stage in spec['stages']:
  name=stage['name'];dest=out/name;dest.mkdir();source=Path(stage['checkpoint'])
  shutil.copyfile(source,scratch/(name+'.msgpack'));payload=msgpack_restore((scratch/(name+'.msgpack')).read_bytes())
  report['checkpoint_sha256'][name]=file_sha256(scratch/(name+'.msgpack'))
  params=from_state_dict(built['init_params'](jax.random.key(0)),payload['params'])
  runner=built['initial_runner'](jax.random.key(0),params)
  if stage['restore_runner']:
   template={k:getattr(runner,k) for k in ('env_state','carry','rng','deadline_ms')}
   restored=jax.tree.map(jnp.asarray,restore_replay_state(template,payload['rollout_state']))
   runner=runner._replace(params=params,opt_state=from_state_dict(runner.opt_state,payload['opt_state']),step=jnp.asarray(payload['step']),**restored)
  records=[];anchor=None;branches_done=False
  for u in range(stage['updates']):
   before=runner;after,tr,batch=jax.block_until_ready(roll(before));last=end_value(after)
   av,rt=gae(tr.reward[:,:,0],tr.value[:,:,0],tr.done[:,:,0],last,cfg.ppo.gamma,cfg.ppo.gae_lambda)
   adv=np.asarray(av).T;returns=np.asarray(rt).T;value=rows(tr.value);reward=rows(tr.reward);done=rows(tr.done)
   # Independent NumPy GAE recurrence, matching actual post-action done semantics.
   manual=np.zeros_like(adv);running=np.zeros(cfg.n_envs,np.float32);nv=np.asarray(last)
   for t in range(cfg.rollout_steps-1,-1,-1):
    live=1-done[:,t];delta=reward[:,t]+cfg.ppo.gamma*nv*live-value[:,t]
    running=delta+cfg.ppo.gamma*cfg.ppo.gae_lambda*live*running;manual[:,t]=running;nv=value[:,t]
   gae_error=float(np.max(np.abs(manual-adv)));assert gae_error<1e-4,gae_error
   batch=dict(batch,adv=jnp.asarray(adv),returns=jnp.asarray(returns))
   lp,v=forward(before.params,batch);lp=np.asarray(lp)
   lp_error=float(np.max(np.abs(lp-np.asarray(batch['log_prob']))));v_error=float(np.max(np.abs(np.asarray(v)-value)))
   assert max(lp_error,v_error)<1e-4,(lp_error,v_error)
   normalized=np.zeros_like(adv);key=after.rng
   for _ in range(cfg.ppo.epochs):
    key,pk=jax.random.split(key);perm=np.asarray(jax.random.permutation(pk,cfg.n_envs))
    for ids in perm.reshape(cfg.n_minibatches,-1):
     a=adv[ids];normalized[ids]+=(a-a.mean())/(a.std()+1e-8)/cfg.ppo.epochs
   runner,metrics=jax.block_until_ready(learn(after,tr,before.carry))
   assert not bool(metrics["loss_nonfinite"]), "nonfinite diagnostic update"
   assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(runner.params))
   new_lp,_=forward(runner.params,batch);delta_lp=np.asarray(new_lp)-lp
   hits=rows(tr.cs_delta)>0;gold=rows(tr.reward_terms['cs']);buttons=rows(tr.action[0])
   future=np.zeros_like(hits);eligible=np.ones_like(hits)
   for lag in range(1,11):
    eligible[:,:-lag]&=~done[:,lag-1:128-1]
    future[:,:-lag]|=hits[:,lag:]&eligible[:,:-lag]
   def summarize(mask):
    n=int(mask.sum())
    if not n:return {'count':0}
    return dict(count=n,reward=float(reward[mask].mean()),gold=float(gold[mask].mean()),
      advantage=float(adv[mask].mean()),positive_adv=float((adv[mask]>0).mean()),
      normalized_adv=float(normalized[mask].mean()),positive_normalized=float((normalized[mask]>0).mean()),
      delta_logprob=float(delta_lp[mask].mean()),probability_increased=float((delta_lp[mask]>0).mean()),
      button_counts=np.bincount(buttons[mask].astype(int),minlength=8).tolist())
   if spec.get('branch_audit') and not branches_done and hits.sum()>=16:
    report.setdefault('branches',{})[name]=branch_audit(built,cfg,sim,bank,before,after,tr,batch,hits,runner.params)
    branches_done=True
    print('BRANCH AUDIT',name,report['branches'][name],flush=True)
   record=dict(update=int(runner.step)//16384,cs=int(np.asarray(tr.cs_delta)[:,:,0].sum()),
    actor_learner_logprob_error=lp_error,actor_learner_value_error=v_error,gae_error=gae_error,
    cohorts={'cs_event':summarize(hits),'preceding_1s':summarize(future&~hits),
     'cs_positive_normalized':summarize(hits&(normalized>0)),
     'cs_negative_normalized':summarize(hits&(normalized<0)),
     'no_cs_positive_gold':summarize((~hits)&(gold>0)),
     'all':summarize(np.ones_like(hits))},metrics={k:float(v) for k,v in metrics.items()})
   if anchor is None and hits.sum()>=16:
    anchor=(jax.tree.map(lambda x:x,batch),lp.copy(),hits.copy(),future.copy())
   if anchor:
    ab,oldlp,ah,af=anchor;alp,_=forward(runner.params,ab);dl=np.asarray(alp)-oldlp
    record['fixed_anchor']=dict(cs_delta_logprob=float(dl[ah].mean()),prehit_delta_logprob=float(dl[af].mean()) if af.any() else None)
   np.savez_compressed(dest/f'update_{record["update"]:04d}.npz',cs=rows(tr.cs_delta),reward=reward,gold=gold,
     value=value,adv=adv,normalized_adv=normalized,logprob=lp,delta_logprob=delta_lp,button=buttons,done=done)
   records.append(record);report['stages'][name]=records
   (out/'result.json').write_text(json.dumps(report,indent=2))
   print('AUDIT',name,record['update'],'CS',record['cs'],record['cohorts']['cs_event'],flush=True)
  del anchor
 print('PROFILE COMPLETE',flush=True)

if __name__=='__main__':main()
