"""Frozen E39/E40 75-second scenario recording; LEARN-PAIR-07, no training."""
import json, os, shutil, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
from lanerl_jax.sim.config import SimConfig, DEFAULT_ROUTE_ARTIFACT
from lanerl_jax.sim.step import env_step
from lanerl_jax.parity.policy_driver import load_params, _lane_frames
from lanerl_jax.obs.builder import build_observation
from lanerl_jax.train.wave_scenario import prepare_scenario_bank, START_MS
from lanerl_jax.train.actions import orders_from, _screen_to_centred_lane
from lanerl_jax.train.trainer import _sample
from lanerl_jax.train.run_manifest import file_sha256, git_provenance
from lanerl_jax.replay import state_snapshot, render_metadata, summarize
from lanerl_rl.constants import N_SCREEN_X, N_SCREEN_Y
from lanerl_jax.train.scripted_policy import cell_for_offset, screen_grid
from lanerl_jax.sim.combat import growth_sum
from lanerl_jax.sim.profiles import PROFILES
from lanerl_jax.obs.fog import visible_to
from lanerl_jax.train.vec_train import _relative_reward, VecConfig
from lanerl_jax.train.replay_audit import serialize_replay_state, restore_replay_state


def main():
    spec=json.loads(Path('experiments',sys.argv[1]+'.json').read_text())
    out=Path('/mnt/nfs/shared')/spec['id'];out.mkdir(exist_ok=False)
    stage=Path('/mnt/nfs/shared')/(spec['id']+'-staged-'+os.environ['SLURM_JOB_ID'])
    stage.mkdir();shutil.copytree(DEFAULT_ROUTE_ARTIFACT,stage/'routes')
    source=Path(spec['checkpoint'])
    shutil.copyfile(source,stage/'checkpoint.msgpack')
    shutil.copyfile(source.parent/'manifest.json',stage/'manifest.json')
    sim=SimConfig.training(route_artifact=stage/'routes').replace(step_ticks=6)
    policy,params,_=load_params(str(stage/'checkpoint.msgpack'))
    assert not policy.cfg.click_mask, 'This probe mirrors the unmasked E40 policy only'
    bank=prepare_scenario_bank(sim,out/'bank',[spec['offset']],1007)
    old_params=None
    if spec.get('opponent_checkpoint'):
        old_source=Path(spec['opponent_checkpoint']); old_stage=stage/'opponent';old_stage.mkdir()
        shutil.copyfile(old_source,old_stage/'checkpoint.msgpack')
        shutil.copyfile(old_source.parent/'manifest.json',old_stage/'manifest.json')
        _,old_params,_=load_params(str(old_stage/'checkpoint.msgpack'))
    frames=_lane_frames()
    @jax.jit
    def act(state,carry,key):
        input_carry=carry
        ob=[build_observation(state,t,frames[t],params=sim.params,horizon_s=600.,vision=sim.vision) for t in (0,1)]
        obs=jax.tree.map(lambda a,b:jnp.stack([a,b]),*ob)
        lg,carry=policy.apply(params,obs.entities,obs.entity_pad_mask,obs.self_vec,obs.global_vec,carry)
        if old_params is not None:
            old_lg,old_carry=policy.apply(old_params,obs.entities,obs.entity_pad_mask,obs.self_vec,obs.global_vec,input_carry)
            lg=jax.tree.map(lambda a,b:a.at[1].set(b[1]),lg,old_lg)
            carry=carry.at[1].set(old_carry[1])
        action,_,_=_sample(lg,key,~obs.entity_pad_mask)
        return action,carry,lg
    decode=jax.jit(lambda a,s:orders_from(a,s,None,frames[0],snap_moves=False,
        params=sim.params,vision=sim.vision,drop_unwalkable_moves=True))
    step=jax.jit(lambda s,o:env_step(s,o,sim))
    @jax.jit
    def snapshot(s,o,a):
        ds,dn=_screen_to_centred_lane((a[1]+.5)/N_SCREEN_X,(a[2]+.5)/N_SCREEN_Y)
        side=jnp.where(s.team[:2]==0,1.,-1.)
        axis,normal=frames[0].axis,frames[0].normal
        cursor=SimpleNamespace(x=s.x[:2]+side*ds*axis[0]+dn*normal[0],y=s.y[:2]+side*ds*axis[1]+dn*normal[1])
        return state_snapshot(s,o,a,cursor)
    seen=jax.jit(lambda s:visible_to(0,s.x,s.y,s.kind,s.team,s.alive,sim.vision))
    reward=jax.jit(lambda a,b:_relative_reward(a,b,VecConfig())[0])
    grid_s,grid_n,grid_ok=screen_grid()
    axis,normal=np.asarray(frames[0].axis),np.asarray(frames[0].normal)
    grid_dx=grid_s*axis[0]+grid_n*normal[0];grid_dy=grid_s*axis[1]+grid_n*normal[1]
    @jax.jit
    def target_action(state,target):
        dx=state.x[target]-state.x[0];dy=state.y[target]-state.y[0]
        sx,sy=cell_for_offset(dx*frames[0].axis[0]+dy*frames[0].axis[1],dx*frames[0].normal[0]+dy*frames[0].normal[1])
        return jnp.int32(2),sx,sy
    for low in spec.get("low_teams",[0,1]):
        dest=out/f'low_team_{low}';dest.mkdir()
        state=jax.tree.map(lambda x:x[low],bank)
        carry=policy.initial_carry((2,));key=jax.random.key(spec['seed']);rows=[];audit=[];cases=[];case_keys=set()
        assert np.allclose(np.asarray(state.hp[:2]/state.max_hp[:2]),[.7,1.] if low==0 else [1.,.7])
        if spec.get('restore_case'):
            source_case=Path(spec['restore_case'])
            restored=restore_replay_state(dict(state=state,carry=carry,key=key),source_case.read_bytes())
            restored=jax.tree.map(jnp.asarray,restored)
            candidates=json.loads((source_case.parent/'opportunities.json').read_text())
            case=next(r for r in candidates if f"{r['category']}_{r['index']}"==source_case.stem)
            cases=[(case,restored['state'],restored['carry'],restored['key'])]
            audit=candidates
        while not spec.get('restore_case') and float(state.t_ms)<START_MS+spec['seconds']*1000:
            before_carry,before_key=carry,key
            key,ak=jax.random.split(key)
            action,carry,lg=act(state,carry,ak);orders=decode(action,state)
            if spec.get('audit'):
                model=np.asarray(state.model);p=sim.params
                ad=float(p['attack_damage'][model[0]]+p['ad_per_level'][model[0]]*growth_sum(float(state.level[0])))
                visible=np.asarray(seen(state));dist=np.hypot(np.asarray(state.x)-float(state.x[0]),np.asarray(state.y)-float(state.y[0]))
                bp=np.asarray(jax.nn.softmax(lg.button[0]));xy=np.asarray(jax.nn.softmax(lg.screen_y[0]))[:,None]*np.asarray(jax.nn.softmax(lg.screen_x[0]))[None,:]
                for target in np.flatnonzero(visible & np.asarray(state.alive) & (np.asarray(state.team)==1) & (dist<=650) & ((np.asarray(state.kind)==2)|(np.arange(len(dist))==1))):
                    subtype=int(PROFILES[model[target]][1]);armor=float(p['armor'][model[target]])
                    damage=ad*100/(100+max(armor,0));killable=float(state.hp[target])<=damage
                    category='champion' if target==1 else 'caster' if subtype==1 else 'melee' if subtype==0 else 'other'
                    if target!=1 and not killable:continue
                    cursor_near=((grid_dx+float(state.x[0])-float(state.x[target]))**2+(grid_dy+float(state.y[0])-float(state.y[target]))**2<=125**2)&grid_ok
                    reach=float(p['attack_range'][model[0]]+p['collision_radius'][model[target]])
                    row=dict(index=len(rows),seconds=(float(state.t_ms)-START_MS)/1000,target=int(target),spawn_seq=int(state.spawn_seq[target]),category=category,hp=float(state.hp[target]),estimated_plain_aa_damage=damage,distance=float(dist[target]),attack_range=reach,aa_cd=float(state.aa_cooldown[0]),aa_windup=float(state.aa_windup[0]),e_active=bool(state.buffs.e.active[0]),e_elapsed=float(state.buffs.e.elapsed_s[0]),own_hp=float(state.hp[0]),enemy_hp=float(state.hp[1]),button_probs=bp.tolist(),cursor_within125_mass=float(xy[cursor_near].sum()),value=float(lg.value[0]),selected_button=int(action[0][0]))
                    audit.append(row)
                    typ=category+('_spin' if row['e_active'] else '_free')
                    if category in ('caster','melee','champion') and typ not in case_keys and len(cases)<6 and row['seconds']<60:
                        cases.append((row,state,before_carry,before_key));case_keys.add(typ)
            rows.append(jax.tree.map(np.asarray,snapshot(state,orders,action)))
            state=jax.block_until_ready(step(state,orders))
            if len(rows)%100==0:print('RECORD',low,len(rows),float(state.t_ms),flush=True)
        if spec.get('restore_case'):
            with np.load(Path(spec['restore_case']).parent/'trace.npz') as ref:
                fields=[k for k in ref.files if k not in ('walkable','metadata')]
                reference_arrays={k:ref[k] for k in fields}
                rows=[{k:reference_arrays[k][i] for k in fields} for i in range(len(reference_arrays['t_ms']))]
        else:
            rows.append(jax.tree.map(np.asarray,snapshot(state,orders,tuple(jnp.zeros(2,jnp.int32) for _ in range(3)))))
        data={k:np.stack([r[k] for r in rows]) for k in rows[0]}
        meta=dict(label=spec.get('label',f'E40 u{spec["update"]} | 75s mirror | low HP team {low}'),
            controller='current blue vs old red' if old_params is not None else 'frozen policy on both champions',environment='jax-wave-scenario',opponent_checkpoint=spec.get('opponent_checkpoint'),
            checkpoint=str(source),checkpoint_sha256=file_sha256(stage/'checkpoint.msgpack'),
            source=git_provenance(),seed=spec['seed'],hz=10,seconds=spec['seconds'],start_seconds=120.,
            observation_horizon_s=600.,view='omniscient diagnostic; not actor input',
            action_alignment='pre-action; terminal frame has unsent NOOP placeholder',
            terrain=dict(min_x=float(sim.terrain.min_x),min_y=float(sim.terrain.min_y),cell_size=float(sim.terrain.cell_size)),
            **render_metadata())
        if spec.get('restore_case'): meta['copied_reference_trace']=str(Path(spec['restore_case']).parent/'trace.npz')
        if spec.get('reference_root') and not spec.get('restore_case'):
            with np.load(Path(spec['reference_root'])/f'low_team_{low}'/'trace.npz') as reference:
                error=max(float(np.max(np.abs(data[k]-reference[k]))) for k in ('x','y','hp','cs','deaths'))
            meta['reference_max_error']=error
            assert error<1e-4, f'Original replay reproduction failed: {error}'
            print('ORIGINAL REPLAY EXACT',low,error,flush=True)
        meta['summary']=summarize(data)
        np.savez_compressed(dest/'trace.npz',**data,walkable=np.asarray(sim.terrain.walkable),metadata=np.asarray(json.dumps(meta)))
        (dest/'trace.json').write_text(json.dumps(meta,indent=2))
        print('REPLAY READY',str(dest),flush=True)
        if spec.get('audit'):
            (dest/'opportunities.json').write_text(json.dumps(audit,indent=2))
            outcomes=[]
            for row,base,base_carry,base_key in cases:
                if spec.get('case_category') and row['category'] != spec['case_category']: continue
                if spec.get('learning_audit'):
                    from .caster_learning_audit import run
                    run(policy,params,sim,base,base_carry,dest,row['target'],step,decode,reward,target_action)
                tag=f"{row['category']}_{row['index']}"
                (dest/(tag+'.msgpack')).write_bytes(serialize_replay_state(dict(state=base,carry=base_carry,key=base_key)))
                for branch in ('control','attack3s','cancel_e_attack3s','e_trade') + tuple(f'sample_{i}' for i in range(spec.get('sample_branches',0))):
                    if branch=='e_trade' and row['category']!='champion':continue
                    if branch=='cancel_e_attack3s' and not row['e_active']:continue
                    bs,bc,bk=base,base_carry,base_key;ret=0.;first_cs=None;target=row['target'];seq=row['spawn_seq'];timeline=[];control_error=0.
                    branch_rows=[];decisions=[];discounted=0.
                    if branch.startswith('sample_'): bk=jax.random.fold_in(bk,int(branch.split('_')[1])+1)
                    for tick in range(80):
                        bk,ak=jax.random.split(bk);a,bc,lg=act(bs,bc,ak)
                        same=bool(bs.alive[target]) and int(bs.spawn_seq[target])==seq
                        if branch in ('attack3s','cancel_e_attack3s','e_trade') and tick<30 and same:
                            forced=target_action(bs,target)
                            if (branch=='cancel_e_attack3s' and bool(bs.buffs.e.active[0])) or (branch=='e_trade' and tick==0):
                                forced=(jnp.int32(5),jnp.int32(0),jnp.int32(0))
                            a=tuple(v.at[0].set(f) for v,f in zip(a,forced))
                        orders_branch=decode(a,bs)
                        if spec.get('save_branches'): branch_rows.append(jax.tree.map(np.asarray,snapshot(bs,orders_branch,a)))
                        nxt=jax.block_until_ready(step(bs,orders_branch));r=float(reward(bs,nxt)[0]);ret+=r;discounted+=(.99**tick)*r
                        decisions.append(dict(seconds=tick/10,value=float(lg.value[0]),button=int(a[0][0]),button_probs=np.asarray(jax.nn.softmax(lg.button[0])).tolist(),reward=r,target_distance=float(jnp.hypot(bs.x[target]-bs.x[0],bs.y[target]-bs.y[0]))))
                        if first_cs is None and int(nxt.cs[0])>int(base.cs[0]):first_cs=(tick+1)/10
                        bs=nxt
                        if branch=='control' and row['index']+tick+1<len(rows):
                            ref=rows[row['index']+tick+1]
                            control_error=max(control_error,max(float(np.max(np.abs(np.asarray(getattr(bs,k))-ref[k]))) for k in ('x','y','hp','cs','deaths')))
                        if tick in (9,19,29,49,79):timeline.append(dict(seconds=(tick+1)/10,cs=int(bs.cs[0]-base.cs[0]),hp=np.asarray(bs.hp[:2]).tolist(),target_alive=bool(bs.alive[target]) and int(bs.spawn_seq[target])==seq))
                    result=dict(case=row,branch=branch,cs_gain=int(bs.cs[0]-base.cs[0]),deaths=np.asarray(bs.deaths[:2]-base.deaths[:2]).tolist(),hp_change=np.asarray(bs.hp[:2]-base.hp[:2]).tolist(),reward=ret,first_cs_s=first_cs,control_error=control_error,timeline=timeline)
                    if branch=='control':assert control_error<1e-4,control_error
                    _,_,end_lg=act(bs,bc,jax.random.key(0))
                    result.update(discounted_reward=discounted,bootstrap_value=float(end_lg.value[0]),bootstrapped_return=discounted+.99**80*float(end_lg.value[0]),decisions=decisions)
                    if spec.get('save_branches'):
                        bd=dest/(tag+'_'+branch);bd.mkdir()
                        branch_rows.append(jax.tree.map(np.asarray,snapshot(bs,orders_branch,tuple(jnp.zeros(2,jnp.int32) for _ in range(3)))))
                        arrays={k:np.stack([r[k] for r in branch_rows]) for k in branch_rows[0]}
                        bm=dict(meta,label=f'{tag} | {branch} | frozen u{spec["update"]}',branch=branch,seconds=8,start_seconds=float(base.t_ms)/1000)
                        np.savez_compressed(bd/'trace.npz',**arrays,walkable=np.asarray(sim.terrain.walkable),metadata=np.asarray(json.dumps(bm)))
                    outcomes.append(result);print('BRANCH',tag,branch,result['cs_gain'],result['hp_change'],flush=True)
            (dest/'counterfactuals.json').write_text(json.dumps(outcomes,indent=2))
    print('PROFILE COMPLETE',flush=True)

if __name__=='__main__':main()
