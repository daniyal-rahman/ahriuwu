"""E37: restore a verified replay state, then override a few blue decisions.

All branches retain frozen policies and full recurrent histories. Red responds
normally. This is a one-state causal diagnostic, not a trained-policy result.
"""
import json,os,shutil,sys,hashlib
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import to_bytes
from lanerl_jax.parity.policy_driver import load_params
from lanerl_jax.sim.config import SimConfig,DEFAULT_ROUTE_ARTIFACT
from lanerl_jax.train.jax_farm import JaxFarmCollector
from lanerl_jax.train.trainer import _sample
from lanerl_jax.train.scripted_policy import cell_for_offset
from lanerl_jax.train.vec_train import _relative_reward,VecConfig
from lanerl_jax.train.run_manifest import git_provenance,file_sha256
from lanerl_jax.train.replay_audit import serialize_replay_state,restore_replay_state


def main():
    spec=json.loads(Path('experiments',sys.argv[1]+'.json').read_text())
    out=Path('/mnt/nfs/shared')/spec['id'];out.mkdir(exist_ok=False)
    source=Path(spec['replay'])
    shutil.copyfile(source/'input/checkpoint.msgpack',out/'checkpoint.msgpack')
    shutil.copyfile(source/'input/manifest.json',out/'manifest.json')
    shutil.copyfile(Path(__file__),out/'probe_source.py')
    with np.load(source/'trace.npz') as z:
        trace={k:z[k] for k in ['t_ms','x','y','hp','cs','deaths','order_x','order_y']}
    recorded=np.array([[r['blue'],r['red']] for r in map(json.loads,(source/'actions.jsonl').read_text().splitlines())],np.int32)
    index=int(np.searchsorted(trace['t_ms'],spec['start_seconds']*1000))
    sim=SimConfig.training(route_artifact=DEFAULT_ROUTE_ARTIFACT).replace(step_ticks=6)
    c=JaxFarmCollector(1,out/'setup',episode_s=600,start_near_wave=True,step_ticks=6,
        seed=7,sim_config=sim,teams=(0,1),drop_unwalkable_moves=True)
    policy,params,_=load_params(str(out/'checkpoint.msgpack'))
    params_hash=hashlib.sha256(to_bytes(params)).hexdigest()
    @jax.jit
    def act(obs,key,carry):
        logits,carry=policy.apply(params,obs.entities,obs.entity_pad_mask,obs.self_vec,obs.global_vec,carry)
        a,_,_=_sample(logits,key,~obs.entity_pad_mask)
        return jnp.stack(a,-1),carry
    reward=jax.jit(lambda before,after:_relative_reward(before,after,VecConfig())[0])
    def scalar(s):return jax.tree.map(lambda x:x[0],s)
    key=jax.random.key(7);carry=policy.initial_carry((2,))
    action_mismatches=0;max_error=0.
    # Drive with recorded commands: observation/history restoration is exact
    # even if compiler fusion changes a near-boundary sampled action.
    for i in range(index):
        obs,_=c.observe();key,ak=jax.random.split(key);a,carry=act(obs,ak,carry)
        action_mismatches+=int(not np.array_equal(np.asarray(a),recorded[i]))
        if i%250==0:
            err=max(float(np.max(np.abs(np.asarray(getattr(c.states,k))[0]-trace[k][i]))) for k in ['x','y','hp'])
            max_error=max(max_error,err)
            if err>1e-4:raise RuntimeError(f'reconstruction diverged at{i}: {err}')
        c.step(recorded[i])
    base=c.states;base_carry=carry;base_key=key
    for k in ['x','y','hp','cs','deaths']:
        np.testing.assert_allclose(np.asarray(getattr(base,k))[0],trace[k][index],rtol=0,atol=1e-4)
    np.testing.assert_allclose(np.asarray(base.t_ms)[0],trace['t_ms'][index],rtol=0,atol=1e-4)
    saved=dict(state=base,carry=carry,key=key)
    payload=serialize_replay_state(saved)
    restored=restore_replay_state(saved,payload)
    assert serialize_replay_state(restored)==payload
    (out/'restored_state.msgpack').write_bytes(payload)
    provenance=dict(spec=spec,source=git_provenance(),checkpoint_sha256=file_sha256(out/'checkpoint.msgpack'),
        state_sha256=file_sha256(out/'restored_state.msgpack'),sim=sim.describe(),sim_fingerprint=sim.fingerprint(),
        restored_index=index,start_ms=float(base.t_ms[0]),prefix_max_state_error=max_error,
        prefix_sample_mismatches=action_mismatches)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('RESTORED',index,'state exact; sampled mismatches',action_mismatches,flush=True)
    origin=np.array([float(base.x[0,0]),float(base.y[0,0])])
    direction=np.array([trace['order_x'][index,0],trace['order_y'][index,0]])-origin
    direction/=np.linalg.norm(direction)
    vectors={'reverse':-direction,'left':np.array([-direction[1],direction[0]]),
             'right':np.array([direction[1],-direction[0]])}
    goals={k:origin+v*500 for k,v in vectors.items()}
    @jax.jit
    def click(s,goal):
        dx=goal[0]-s.x[0,0];dy=goal[1]-s.y[0,0]
        sx,sy=cell_for_offset(dx*c.frame.axis[0]+dy*c.frame.axis[1],dx*c.frame.normal[0]+dy*c.frame.normal[1])
        return jnp.stack([jnp.int32(1),sx,sy])
    results=[]
    for name in spec['branches']:
        c.states=base;carry=base_carry;key=base_key;rows=[];control_error=0.
        while float(c.states.t_ms[0])-float(base.t_ms[0])<spec['horizon_seconds']*1000:
            i=len(rows);before=c.states;elapsed=(float(before.t_ms[0])-float(base.t_ms[0]))/1000
            obs,_=c.observe();key,ak=jax.random.split(key);action,carry=act(obs,ak,carry)
            action=np.asarray(action).copy()
            if name=='recorded_control':action=recorded[index+i].copy()
            elif name in ('w','e','q') and i==0:action[0]=[dict(w=4,e=5,q=3)[name],0,0]
            elif name=='w_then_e' and i<2:action[0]=[4 if i==0 else 5,0,0]
            elif name in goals and elapsed<1.:action[0]=np.asarray(click(before,jnp.asarray(goals[name])))
            elif name=='e_then_reverse':
                if i==0:action[0]=[5,0,0]
                elif elapsed<1.1:action[0]=np.asarray(click(before,jnp.asarray(goals['reverse'])))
            c.step(action);s=c.states
            rew=np.asarray(reward(scalar(before),scalar(s)))
            row=dict(elapsed_s=(float(s.t_ms[0])-float(base.t_ms[0]))/1000,
                hp=np.asarray(s.hp[0,:2]).tolist(),alive=np.asarray(s.alive[0,:2]).tolist(),
                deaths=np.asarray(s.deaths[0,:2]).tolist(),cs=np.asarray(s.cs[0,:2]).tolist(),
                gold=np.asarray(s.gold[0,:2]).tolist(),xp=np.asarray(s.xp[0,:2]).tolist(),
                xy=np.stack([np.asarray(s.x[0,:2]),np.asarray(s.y[0,:2])],-1).tolist(),
                reward=rew.tolist(),action=action.tolist(),order=np.asarray(c.last_orders.kind[0]).tolist(),
                e_active=bool(s.buffs.e.active[0,0]),w_active=bool(s.buffs.w.active[0,0]))
            rows.append(row)
            if name=='recorded_control':
                for k in ['x','y','hp','cs','deaths']:
                    error=float(np.max(np.abs(np.asarray(getattr(s,k))[0]-trace[k][index+i+1])))
                    control_error=max(control_error,error)
                    if error>1e-4:raise RuntimeError(f'control diverged {k} {i} {error}')
        start=scalar(base);end=scalar(c.states)
        summary=dict(branch=name,steps=len(rows),horizon_s=rows[-1]['elapsed_s'],
            deaths_delta=int(end.deaths[0]-start.deaths[0]),cs_delta=int(end.cs[0]-start.cs[0]),
            gold_delta=float(end.gold[0]-start.gold[0]),
            relative_gold_delta=float((end.gold[0]-end.gold[1])-(start.gold[0]-start.gold[1])),
            first_death_s=next((r['elapsed_s'] for r in rows if not r['alive'][0]),None),
            reward_sum=sum(r['reward'][0] for r in rows),
            discounted_reward=sum(.99**i*r['reward'][0] for i,r in enumerate(rows)),
            reward_60s_discount=sum((1-1/600)**i*r['reward'][0] for i,r in enumerate(rows)),
            control_max_error=control_error if name=='recorded_control' else None)
        for t in [1,5,10,20,40]:
            if t>spec['horizon_seconds']:continue
            r=min(rows,key=lambda r:abs(r['elapsed_s']-t))
            summary[f'at_{t}s']=dict(hp=r['hp'][0],alive=r['alive'][0],cs=r['cs'][0]-int(start.cs[0]),
                displacement=float(np.linalg.norm(np.array(r['xy'][0])-origin)))
        (out/(name+'.json')).write_text(json.dumps(dict(summary=summary,frames=rows)))
        results.append(summary);(out/'summary.json').write_text(json.dumps(results,indent=2))
        print('BRANCH',json.dumps(summary),flush=True)
    assert hashlib.sha256(to_bytes(params)).hexdigest()==params_hash
    c.close();print('PROFILE COMPLETE',flush=True)

if __name__=='__main__':main()
