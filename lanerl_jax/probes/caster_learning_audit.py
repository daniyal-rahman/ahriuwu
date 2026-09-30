"""LEARN-PAIR-09: bounded caster input, Monte Carlo and on-policy update audit.
Called by wave_replay inside its registered CPU Slurm job. Never saves live weights.
"""
import json
import numpy as np
import jax
import jax.numpy as jnp
from . import wave_replay as wr
from lanerl_jax.train.ppo import PPOConfig, gae
from lanerl_jax.train.learner import make_learner, make_update


def run(policy, initial, sim, base, carry0, dest, target, step, decode, reward, target_action):
    frames=wr._lane_frames(); report={}
    def save(): (dest/'learning_audit.json').write_text(json.dumps(report,indent=2))
    @jax.jit
    def observe(s):
        obs=[wr.build_observation(s,t,frames[t],params=sim.params,horizon_s=600.,vision=sim.vision) for t in (0,1)]
        return jax.tree.map(lambda a,b:jnp.stack([a,b]),*obs)
    @jax.jit
    def forward(p,o,c): return policy.apply(p,o.entities,o.entity_pad_mask,o.self_vec,o.global_vec,c)
    @jax.jit
    def act(p,s,c,k,temp):
        o=observe(s);lg,nc=forward(p,o,c)
        old,oc=forward(initial,o,c)
        lg=jax.tree.map(lambda a,b:a.at[1].set(b[1]),lg,old);nc=nc.at[1].set(oc[1])
        scaled=lg._replace(button=lg.button/temp,screen_x=lg.screen_x/temp,screen_y=lg.screen_y/temp)
        a,lp,usage=wr._sample(scaled,k,~o.entity_pad_mask)
        return a,nc,lg,o,lp,usage
    gs,gn,ok=wr.screen_grid();ax,nm=np.asarray(frames[0].axis),np.asarray(frames[0].normal)
    def inspect(s,c,p=initial):
        lg,_=forward(p,observe(s),c);bp=np.asarray(jax.nn.softmax(lg.button[0]));xy=np.asarray(jax.nn.softmax(lg.screen_y[0]))[:,None]*np.asarray(jax.nn.softmax(lg.screen_x[0]))[None,:]
        dx=float(s.x[target]-s.x[0]);dy=float(s.y[target]-s.y[0])
        near=((gs*ax[0]+gn*nm[0]-dx)**2+(gs*ax[1]+gn*nm[1]-dy)**2<125**2)&ok
        return dict(attack=float(bp[2]),move=float(bp[1]),cursor_mass=float(xy[near].sum()),direct_attack_mass=float(bp[2]*xy[near].sum()),value=float(lg.value[0]))
    report['baseline']=inspect(base,carry0);sens=[]
    for hp in (20.,83.,160.,300.):
        s=base._replace(hp=base.hp.at[target].set(hp));sens.append(dict(change='caster_hp',setting=hp,**inspect(s,carry0)))
    for hp in (1.,200.,479.,800.):
        s=base._replace(hp=base.hp.at[1].set(hp));sens.append(dict(change='enemy_hp',setting=hp,**inspect(s,carry0)))
    for dist in (150.,300.,450.,639.):
        dx=base.x[target]-base.x[0];dy=base.y[target]-base.y[0];length=jnp.hypot(dx,dy)
        s=base._replace(x=base.x.at[target].set(base.x[0]+dx/length*dist),y=base.y.at[target].set(base.y[0]+dy/length*dist))
        sens.append(dict(change='caster_distance',setting=dist,**inspect(s,carry0)))
    report['input_interventions']=sens;save();print('INPUT AUDIT DONE',flush=True)
    def rollout(p,start,c0,seed,n=80,temp=1.,collect=False,greedy=False):
        s=start;c=c0;k=jax.random.key(seed);ret=0.;rows=[];ncs0=int(s.cs[0]);deaths0=int(s.deaths[0])
        for i in range(n):
            k,ak=jax.random.split(k);a,nc,lg,o,lp,usage=act(p,s,c,ak,temp)
            if greedy: a=tuple(jnp.asarray(jnp.argmax(z,axis=-1),jnp.int32) for z in (lg.button,lg.screen_x,lg.screen_y))
            nxt=jax.block_until_ready(step(s,decode(a,s)));r=float(reward(s,nxt)[0]);ret+=.99**i*r
            if collect:
                rows.append(dict(entities=np.asarray(o.entities[0]),mask=np.asarray(o.entity_pad_mask[0]),self=np.asarray(o.self_vec[0]),global_=np.asarray(o.global_vec[0]),action=tuple(np.asarray(x[0]) for x in a),log_prob=float(lp[0]),uses_screen=float(usage[0][0]),value=float(lg.value[0]),reward=r,done=False))
            s=nxt;c=nc
        lg,_=forward(p,observe(s),c)
        result=dict(seed=seed,cs=int(s.cs[0])-ncs0,deaths=int(s.deaths[0])-deaths0,discounted_return=ret,final_value=float(lg.value[0]))
        return result,rows,float(lg.value[0])
    # Original scenario ends at195s. Full remaining horizon, no critic bootstrap.
    horizon=int(np.ceil((195000-float(base.t_ms))/100.))
    mc=[]
    for seed in range(16):
        r,_,_=rollout(initial,base,carry0,10000+seed,n=horizon);mc.append(r)
        print('MONTE CARLO',seed,r,flush=True)
    report['monte_carlo']=dict(horizon_steps=horizon,predicted=report['baseline']['value'],samples=mc,mean=float(np.mean([r['discounted_return'] for r in mc])),standard_error=float(np.std([r['discounted_return'] for r in mc],ddof=1)/4));save()
    temps=[]
    for temp in (.5,1.,2.):
        for seed in range(4):
            r,_,_=rollout(initial,base,carry0,20000+seed,temp=temp);temps.append(dict(temperature=temp,**r))
    r,_,_=rollout(initial,base,carry0,20000,greedy=True);temps.append(dict(temperature='argmax',**r))
    report['temperature']=temps;save();print('TEMPERATURE DONE',flush=True)
    # Real on-policy PPO: warm-start a curriculum state closer to the caster;
    # no forced actions are ever presented as on-policy training samples.
    s=base;c=carry0;k=jax.random.key(30000)
    for i in range(10):
        k,ak=jax.random.split(k);a,nc,_,_,_,_=act(initial,s,c,ak,1.)
        forced=target_action(s,target);a=tuple(v.at[0].set(f) for v,f in zip(a,forced));s=jax.block_until_ready(step(s,decode(a,s)));c=nc
    near,near_carry=s,c
    report['curriculum']=dict(distance=float(jnp.hypot(s.x[target]-s.x[0],s.y[target]-s.y[0])),target_hp=float(s.hp[target]),note='State and recurrent history after1s directed approach. PPO samples below are unforced; fixed old red opponent.')
    before=[]
    for name,ss,cc in [('original',base,carry0),('near',near,near_carry)]:
        for seed in range(4):
            r,_,_=rollout(initial,ss,cc,40000+seed);before.append(dict(state=name,**r))
    report['before']=before;save()
    cfg=PPOConfig.standard(lr=3e-5,entropy_coef=.001,epochs=4,n_minibatches=2)
    tx,loss=make_learner(policy,cfg);update=make_update(tx,loss,cfg);p=initial;opt=tx.init(p);updates=[]
    for u in range(2):
        seqs=[];ends=[];scores=[]
        for seed in range(8):
            r,rows,end=rollout(p,near,near_carry,50000+u*100+seed,n=32,collect=True);seqs.append(rows);ends.append(end);scores.append(r)
        batch={}
        for key in ('entities','mask','self','global_','log_prob','uses_screen','value','done'):
            batch['global' if key=='global_' else key]=jnp.asarray([[r[key] for r in seq] for seq in seqs])
        batch['action']=tuple(jnp.asarray([[r['action'][h] for r in seq] for seq in seqs]) for h in range(3))
        rewards=jnp.asarray([[r['reward'] for r in seq] for seq in seqs]);adv,returns=gae(rewards.T,batch['value'].T,batch['done'].T,jnp.asarray(ends),.99,.95)
        batch.update(adv=adv.T,returns=returns.T,carry0=jnp.repeat(near_carry[0:1],8,axis=0))
        lg=loss.forward(p,batch);err=float(jnp.max(jnp.abs(lg.value-batch['value'])));assert err<1e-4,err
        p,opt,_,metrics=update(p,opt,batch,jax.random.key(60000+u));jax.block_until_ready(p)
        updates.append(dict(update=u,scores=scores,actor_learner_value_error=err,metrics={k:float(v) for k,v in metrics.items()},original=inspect(base,carry0,p),near=inspect(near,near_carry,p)))
        report['updates']=updates;save();print('LOCAL PPO',u,updates[-1],flush=True)
    after=[]
    for name,ss,cc in [('original',base,carry0),('near',near,near_carry)]:
        for seed in range(4):
            r,_,_=rollout(p,ss,cc,40000+seed);after.append(dict(state=name,**r))
    report['after']=after;report['limitations']='One checkpoint/state; input interventions hold old GRU fixed and can be off-distribution. MC16 at original state only. Local PPO512 decisions, fresh optimizer, curriculum reset, no weights exported; not a training recommendation or aggregate evaluation.';save();print('LEARNING AUDIT COMPLETE',flush=True)
