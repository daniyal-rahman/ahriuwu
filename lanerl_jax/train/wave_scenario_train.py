"""Bounded E39 curriculum run with frozen role-separated evaluations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_state_dict, msgpack_restore
from .policy import PolicyConfig
from .ppo import PPOConfig
from .run_manifest import RunDir, file_sha256
from .vec_train import VecConfig, make_vec_train
from .wave_scenario import prepare_scenario_bank, START_MS, turret_distance, point_on_path
from ..sim.config import SimConfig, DEFAULT_ROUTE_ARTIFACT
from ..sim.init import TOP_LANE_PATH, TOP_OUTER_TURRET
from ..sim.state import Kind
from ..sim.orders import Orders, OrderKind
from ..sim.step import env_step


def calibrate(bank,sim,out):
    """Check wave travel with stationary champions; record HP at plausible contact."""
    state=jax.tree.map(lambda x:x[0],bank)
    threshold=int(state.next_spawn_seq)
    path=np.asarray(TOP_LANE_PATH)
    centre,_=point_on_path(path,sum(turret_distance(path,TOP_OUTER_TURRET[t]) for t in (0,1))/2)
    noop=Orders(jnp.zeros(2,jnp.int8),jnp.zeros(2),jnp.zeros(2),jnp.full(2,-1,jnp.int8))
    advance=jax.jit(lambda s:jax.lax.fori_loop(0,10,lambda i,v:env_step(v,noop,sim),s))
    first=[None,None];rows=[]
    for _ in range(80):
        state=jax.block_until_ready(advance(state)); elapsed=(float(state.t_ms)-START_MS)/1000
        xy=np.stack([state.x,state.y],-1)
        for team in (0,1):
            candidate=np.asarray(state.alive & (state.kind==Kind.LANE_MINION)&(state.team==team)&(state.spawn_seq>=threshold))
            if first[team] is None and np.any(candidate & (np.linalg.norm(xy-centre,axis=1)<900)):
                first[team]=elapsed
        rows.append(dict(elapsed_s=elapsed,hp_fraction=np.asarray(state.hp[:2]/state.max_hp[:2]).tolist(),
                         minions=int(np.asarray(state.alive&(state.kind==Kind.LANE_MINION)).sum())))
    result=dict(next_wave_spawn_s=30.,next_wave_central_band_s=first,
                definition='First later-wave minion within900 units of initial neutral meeting point, idle champions',
                trace=rows)
    (out/'calibration.json').write_text(json.dumps(result,indent=2))
    if any(t is None or t>75 for t in first):
        raise RuntimeError(f'next-wave timing outside scenario budget: {first}')
    print('SCENARIO CALIBRATION',json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)


def alive_spell_count(buttons, self_obs):
    """Count per-team selections over time; self_obs is [time, team, features]."""
    alive = np.asarray(self_obs)[..., 14] < .5
    buttons = np.asarray(buttons)
    if buttons.shape != alive.shape:
        raise ValueError(f"action/observation shape mismatch: {buttons.shape} vs {alive.shape}")
    return ((buttons >= 3) & (buttons <= 6) & alive).sum(axis=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment');parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    spec=json.loads(Path('experiments',args.experiment+'.json').read_text())
    if args.resume:
        previous=json.loads((args.resume.parent/'manifest.json').read_text())
        if previous['config']['scenario'] != spec:
            raise ValueError('Resume requires the identical experiment specification')
    start=time.monotonic();stopping=[]
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        signal.signal(sig,lambda signum,frame:stopping.append(signum))
    def stop():return bool(stopping) or time.monotonic()-start>=spec['max_seconds']
    jax.config.update('jax_default_matmul_precision','highest')
    jax.config.update('jax_compilation_cache_dir','/scratch/lanerl-jax-compilation-cache')
    scratch=Path('/scratch')/(spec['id']+'-'+os.environ['SLURM_JOB_ID']);scratch.mkdir(parents=True)
    shutil.copytree(DEFAULT_ROUTE_ARTIFACT,scratch/'routes')
    source=Path(spec['init_from']);shutil.copyfile(source,scratch/'initial.msgpack')
    opponent_source=Path(spec.get('eval_opponent_from',spec['init_from']))
    shutil.copyfile(opponent_source,scratch/'eval_opponent.msgpack')
    out=Path('/mnt/nfs/checkpoints/lanerl-jax')/spec['id'];out.mkdir(parents=True,exist_ok=True)
    sim=SimConfig.training(route_artifact=scratch/'routes').replace(step_ticks=6)
    bank=prepare_scenario_bank(sim,out/'train_bank',spec['train_offsets'],0)
    eval_bank=prepare_scenario_bank(sim,out/'eval_bank',spec['eval_offsets'],1007)
    calibrate(bank,sim,out)
    pcfg=PolicyConfig(core='gru',core_norm=True,core_residual=True,detach_critic=True)
    cfg=VecConfig(n_envs=128,rollout_steps=128,n_updates=spec['updates'],n_minibatches=4,
        episode_s=START_MS/1000+spec['duration_s'],observation_horizon_s=600.,stagger_initial=False,
        bank_size=len(bank.t_ms),lr_anneal=True,policy=pcfg,
        ppo=PPOConfig.standard(lr=spec['lr'],entropy_coef=spec['entropy_coef']))
    built=make_vec_train(cfg,sim,bank)
    params=from_state_dict(built['init_params'](jax.random.key(0)),msgpack_restore((scratch/'initial.msgpack').read_bytes())['params'])
    opponent_params=from_state_dict(params,msgpack_restore((scratch/'eval_opponent.msgpack').read_bytes())['params'])
    runner=built['initial_runner'](jax.random.key(0),params)
    update=0
    if args.resume:
        payload=msgpack_restore(args.resume.read_bytes())
        runner=runner._replace(params=from_state_dict(runner.params,payload['params']),
            opt_state=from_state_dict(runner.opt_state,payload['opt_state']),
            step=jnp.asarray(payload['step']))
        update=int(runner.step)//(cfg.n_envs*cfg.learn_agents*cfg.rollout_steps)
        assert int(runner.step)==update*cfg.n_envs*cfg.learn_agents*cfg.rollout_steps
        if 'rollout_state' in payload:
            from .replay_audit import restore_replay_state
            template={k:getattr(runner,k) for k in ('env_state','carry','rng','deadline_ms')}
            runner=runner._replace(**restore_replay_state(template,payload['rollout_state']))
            continuity='full runner restored'
        else:
            runner=runner._replace(rng=jax.random.fold_in(runner.rng,update))
            continuity='legacy checkpoint: new episodes/carries/RNG; optimizer and schedule preserved'
        print('RESUME',update,continuity,flush=True)
    run=RunDir(out,'vec-s0',dict(train={'policy':pcfg._asdict()},ppo=cfg.ppo._asdict(),
        vec={k:v for k,v in cfg._asdict().items() if k not in ('policy','ppo')},
        collector=dict(episode_s=cfg.episode_s,step_ticks=6,unwalkable_click='noop'),
        scenario=spec,environment='jax-vectorised',opponent='mirror-self-play',
        initialization=('same experiment continuation; optimizer/schedule retained' if args.resume else 'init_from checkpoint parameters only; fresh optimizer and schedule'),
        continuation=dict(checkpoint=str(args.resume),sha256=file_sha256(args.resume),start_update=update,continuity=continuity) if args.resume else None,
        init_source_sha256=file_sha256(source),eval_opponent_source=str(opponent_source),
        eval_opponent_sha256=file_sha256(opponent_source),sim=sim.describe(),sim_fingerprint=sim.fingerprint()),
        notes='Finite75s tower-wave task; not a ten-minute lane score. Frozen evals by initial HP role.')
    run.keep_checkpoints=0;failed=False
    def save():
        from .replay_audit import serialize_replay_state
        rollout={k:getattr(runner,k) for k in ('env_state','carry','rng','deadline_ms')}
        run.save(int(runner.step),update,dict(params=runner.params,opt_state=runner.opt_state,
            step=runner.step,rollout_state=serialize_replay_state(rollout)))
        (out/'study.json').write_text(json.dumps(dict(job=os.environ['SLURM_JOB_ID'],path=str(run.path),
            update=update,status='failed' if failed else 'complete' if update==cfg.n_updates else 'running',
            elapsed_s=time.monotonic()-start),indent=2))
    save()
    # Validate real checkpoint on the actual scenario before expensive learner compilation.
    obs_builder=built['policy']; from ..obs.builder import build_observation
    from ..parity.policy_driver import _lane_frames
    sample=jax.tree.map(lambda x:x[0],bank)
    obs=build_observation(sample,0,_lane_frames()[0],params=sim.params,horizon_s=600.,vision=sim.vision)
    logits,_=obs_builder.apply(params,obs.entities[None],obs.entity_pad_mask[None],obs.self_vec[None],obs.global_vec[None],obs_builder.initial_carry((1,)))
    assert np.isfinite(np.asarray(logits.button)).all()
    print('SCENARIO READY: actual checkpoint, level3,12minions, both HP roles, next-wave timing verified',flush=True)
    print('COMPILE training',flush=True)
    update_fn=jax.jit(lambda r:built['run_chunk'](r,1)).lower(runner).compile()
    evaluations={}
    for mode in ('mirror','frozen'):
        ecfg=cfg._replace(n_envs=64,opponent=mode,bank_size=len(eval_bank.t_ms))
        eb=make_vec_train(ecfg,sim,eval_bank,opponent_params=opponent_params if mode=='frozen' else None)
        er=eb['initial_runner'](jax.random.key(2007),params)
        indices=jnp.arange(ecfg.n_envs)%len(eval_bank.t_ms)
        er=er._replace(env_state=jax.tree.map(lambda b:b[indices],eval_bank))
        print('COMPILE eval',mode,flush=True)
        evaluations[mode]=(er,jax.jit(eb['collect']).lower(er).compile())
    # Regression gate on the compiled path: endpoint metrics precede reset.
    probe, probe_fn = evaluations["mirror"]
    st = probe.env_state
    st = st.replace(t_ms=jnp.full_like(st.t_ms, cfg.episode_s*1000-.01),
                    hp=st.hp.at[:,:2].set(st.max_hp[:,:2]*.42),
                    kills=st.kills.at[:,:2].set(3))
    _, terminal, _ = jax.block_until_ready(probe_fn(probe._replace(env_state=st)))
    assert np.all(np.asarray(terminal.done_full[0]))
    assert np.all(np.asarray(terminal.hp_at_end[0]) < .5), "endpoint HP read after reset"
    assert np.all(np.asarray(terminal.kills_at_end[0]) == 3), "endpoint kills read after reset"
    print("ENDPOINT CANARY PASSED: HP/kills recorded before reset",flush=True)
    def evaluate():
        for mode,(template,fn) in evaluations.items():
            r=template._replace(params=runner.params);seen=np.zeros(64,bool);rows=[]
            low=np.asarray(template.env_state.hp[:,:2]/template.env_state.max_hp[:,:2])<.85
            first_seen_hp=np.full((64,2),np.nan)
            deaths=np.zeros((64,2));returns=np.zeros((64,2));alive_spells=np.zeros((64,2))
            for _ in range(int(np.ceil(spec['duration_s']*10/128))+2):
                if stop():return
                r,tr,_=jax.block_until_ready(fn(r))
                done=np.asarray(tr.done_full[:,:,0])
                for env in np.flatnonzero(~seen):
                    hits=np.flatnonzero(done[:,env]);end=int(hits[0])+1 if len(hits) else len(done)
                    for team in (0,1):
                        visible=np.flatnonzero(np.asarray(tr.obs_global[:end,env,team,1])>.5)
                        if np.isnan(first_seen_hp[env,team]) and len(visible):
                            first_seen_hp[env,team]=float(tr.obs_self[visible[0],env,team,2])
                    deaths[env]+=np.asarray(tr.deaths[:end,env]).sum(0)
                    returns[env]+=np.asarray(tr.reward[:end,env]).sum(0)
                    alive_spells[env]+=alive_spell_count(tr.action[0][:end,env],tr.obs_self[:end,env])
                    if len(hits):
                        t=end-1;seen[env]=True
                        for team in (0,1):
                            rows.append(dict(env=int(env),team=team,low_hp=bool(low[env,team]),
                                cs=float(tr.cs[t,env,team]),gold=float(tr.gold[t,env,team]),
                                gold_diff=float(tr.gold[t,env,team]-tr.gold[t,env,1-team]),
                                first_enemy_seen_hp=float(first_seen_hp[env,team]) if np.isfinite(first_seen_hp[env,team]) else None,
                                kills=float(tr.kills_at_end[t,env,team]),deaths=float(deaths[env,team]),
                                hp_fraction=float(tr.hp_at_end[t,env,team]),reward=float(returns[env,team]),
                                spell_selections=float(alive_spells[env,team])))
                if seen.all():break
            if not seen.all():raise RuntimeError('incomplete scenario eval')
            summaries={}
            for team in (0,1):
                for disadvantaged in (True,False):
                    cohort=[x for x in rows if x['team']==team and x['low_hp']==disadvantaged]
                    summaries[f'{team}_{"low" if disadvantaged else "full"}']={k:float(np.mean([x[k] for x in cohort]))
                        for k in ('cs','gold_diff','kills','deaths','hp_fraction','reward','spell_selections')}
                    contact=[x['first_enemy_seen_hp'] for x in cohort if x['first_enemy_seen_hp'] is not None]
                    summaries[f'{team}_{"low" if disadvantaged else "full"}']['first_enemy_seen_hp']=float(np.mean(contact)) if contact else None
            result=dict(update=update,frozen=True,opponent=mode,duration_s=spec['duration_s'],games=64,
                        summaries=summaries,episodes=rows)
            with (run.path/'evaluations.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            print('FROZEN',mode,update,json.dumps(summaries),flush=True)
    try:
        evaluate()
        while update<cfg.n_updates and not stop():
            t=time.monotonic();runner,m=jax.block_until_ready(update_fn(runner));update+=1
            row={k:float(np.asarray(v).reshape(-1)[0]) for k,v in m.items()}
            row['scenario_cs_train']=row.pop('cs_at_10min')
            row['scenario_gold_train']=row.pop('gold_at_10min');row['scenario_xp_train']=row.pop('xp_at_10min')
            row['deaths_per_episode']*=spec['duration_s']/cfg.episode_s
            row.update(update=update,step=int(runner.step),update_s=time.monotonic()-t);run.log(row)
            if row['loss_nonfinite'] or not np.isfinite(row['policy_loss']):
                failed=True;raise RuntimeError('nonfinite training')
            if update%25==0:print('TRAIN',json.dumps(row),flush=True)
            if update%50==0:save()
            if update in spec['eval_updates']:save();evaluate()
    except Exception:
        failed=True
        raise
    finally:
        save();run.set_results(status='failed' if failed else 'finished' if update==cfg.n_updates else 'interrupted',updates=update)
        run.close()
        study=json.loads((out/"study.json").read_text())
        study["status"]="failed" if failed else "complete" if update==cfg.n_updates else "interrupted"
        (out/"study.json").write_text(json.dumps(study,indent=2))
    print('PROFILE COMPLETE',flush=True)

if __name__=='__main__':main()
