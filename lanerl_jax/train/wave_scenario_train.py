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
from .policy import (PolicyConfig, OWN_ACTION_INTERFACE, expand_own_action_inputs,
                     merge_click_proposal_params, merge_visible_history_params, merge_combat_params)
from ..obs.visible_history import VISIBLE_HISTORY_INTERFACE, HISTORY_ENTITY_DIM
from ..obs.combat_features import COMBAT_INTERFACE, COMBAT_ENTITY_DIM, COMBAT_SELF_DIM
from .wave_evaluation import alive_spell_count, evaluate_frozen
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



def warmup_staggered_runner(built, runner, cfg, stop=lambda: False):
    """Discard the shortened first episodes; learn only full-length games.

    Existing collector staggering randomizes the first deadline. Collect with
    fixed parameters until every such deadline has passed, retaining real
    environment/carry/RNG state but no training transitions or optimizer steps.
    Subsequent resets all use the normal full episode deadline.
    """
    remaining_s = cfg.episode_s - float(np.asarray(runner.env_state.t_ms).min()) / 1000.
    chunks = int(np.ceil(remaining_s * cfg.decision_hz / cfg.rollout_steps)) + 1
    advance = jax.jit(lambda r: built['collect'](r)[0])
    initial_step = int(runner.step)
    for _ in range(chunks):
        if stop():
            raise InterruptedError('stopped during stagger warmup; no learning performed')
        runner = jax.block_until_ready(advance(runner))
    np.testing.assert_array_equal(np.asarray(runner.deadline_ms), cfg.episode_s * 1000.)
    assert int(runner.step) == initial_step, 'warmup must not increment learner decisions'
    clocks = np.asarray(runner.env_state.t_ms)
    if cfg.n_envs > 1 and np.ptp(clocks) <= 1000. / cfg.decision_hz:
        raise RuntimeError('stagger warmup did not diversify training phases')
    return runner, dict(rollouts=chunks,
        untrained_environment_decisions=chunks*cfg.rollout_steps*cfg.n_envs,
        clock_min_ms=float(clocks.min()), clock_max_ms=float(clocks.max()),
        all_deadlines_full=True, discarded_shortened_episodes=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment');parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    spec=json.loads(Path('experiments',args.experiment+'.json').read_text())
    if spec.get('init_handoff'):
        handoff=json.loads(Path(spec['init_handoff']).read_text())
        if handoff['status'] != 'complete' or handoff['checkpoint'] != spec['init_from']:
            raise RuntimeError('initial imitation handoff is incomplete or identifies a different checkpoint')
        if handoff['evaluation_update'] != spec['initial_eval_reference']['update']:
            raise RuntimeError('initial imitation evaluation does not match the planned endpoint')
        spec['init_sha256']=handoff['checkpoint_sha256']
    if spec.get('preflight_gate'):
        audit=json.loads(Path(spec['preflight_gate']).read_text())
        if not audit['passed']:
            raise RuntimeError('required feature preflight did not pass')
    train_seed = int(spec.get('train_seed', 0))
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
    if spec.get('init_sha256') and file_sha256(scratch/'initial.msgpack') != spec['init_sha256']:
        raise RuntimeError('initial checkpoint SHA does not match experiment specification')
    opponent_source=Path(spec.get('eval_opponent_from',spec['init_from']))
    shutil.copyfile(opponent_source,scratch/'eval_opponent.msgpack')
    out=Path('/mnt/nfs/checkpoints/lanerl-jax')/spec['id'];out.mkdir(parents=True,exist_ok=True)
    sim=SimConfig.training(route_artifact=scratch/'routes').replace(step_ticks=6)
    bank=prepare_scenario_bank(sim,out/'train_bank',spec['train_offsets'],0)
    eval_bank=prepare_scenario_bank(sim,out/'eval_bank',spec['eval_offsets'],1007)
    if spec.get("opponent") == "afk":
        from .wave_scenario import park_afk_opponent
        bank=park_afk_opponent(bank);eval_bank=park_afk_opponent(eval_bank)
        for name,actual in (("train_bank",bank),("eval_bank",eval_bank)):
            setup=out/name/'setup.json'
            rows=json.loads(setup.read_text())
            for i,row in enumerate(rows):
                row.update(afk_red=True,hp=np.asarray(actual.hp[i,:2]).tolist(),
                    xy=np.stack([actual.x[i,:2],actual.y[i,:2]],-1).tolist(),
                    blue_initial_hp_fraction=float(actual.hp[i,0]/actual.max_hp[i,0]))
                row.pop('low_hp_team',None)
            setup.write_text(json.dumps(rows,indent=2))
    calibrate(bank,sim,out)
    pcfg=PolicyConfig(core='gru',core_norm=True,core_residual=True,detach_critic=spec.get('detach_critic',True))
    pcfg=pcfg._replace(click_mask=spec.get('click_mask',False),
                      action_mask=spec.get('action_mask',False))
    if spec.get('click_proposals', False):
        if spec.get('opponent') != 'afk':
            raise ValueError('click-proposal experiment currently supports AFK evaluation only')
        pcfg=pcfg._replace(click_proposals=True)
    own_action = spec.get('own_action_state', False)
    if own_action:
        pcfg=pcfg._replace(observation_interface=OWN_ACTION_INTERFACE,self_dim=20)
    history = spec.get('visible_history', False)
    if history:
        if own_action or pcfg.click_proposals or spec.get('opponent') != 'afk':
            raise ValueError('visible-history experiment is AFK-only and separate from own-action/proposal experiments')
        pcfg=pcfg._replace(observation_interface=VISIBLE_HISTORY_INTERFACE, entity_dim=HISTORY_ENTITY_DIM)
    combat = spec.get('combat_features', False)
    if combat:
        if history or own_action or pcfg.click_proposals or spec.get('opponent') != 'afk':
            raise ValueError('combat feature experiment requires a separate AFK arm')
        pcfg=pcfg._replace(observation_interface=COMBAT_INTERFACE,
                          entity_dim=COMBAT_ENTITY_DIM, self_dim=COMBAT_SELF_DIM)
    cfg=VecConfig(n_envs=128,rollout_steps=128,n_updates=spec['updates'],n_minibatches=4,
        episode_s=START_MS/1000+spec['duration_s'],observation_horizon_s=600.,
        stagger_initial=spec.get('stagger_initial',False),
        bank_size=len(bank.t_ms),lr_anneal=spec.get("lr_anneal",True),policy=pcfg,
        xp_scale=spec.get("xp_scale",0.008),
        cs_only=spec.get("cs_only",False),
        opponent=spec.get("opponent","mirror"),health_loss_gold=spec.get("health_loss_gold",0.),
        death_loss_gold=spec.get("death_loss_gold",0.),
        tower_damage_gold=spec.get("tower_damage_gold",0.),
        tower_damage_personal=spec.get("tower_damage_personal",False),
        ppo=PPOConfig.standard(lr=spec['lr'],entropy_coef=spec['entropy_coef'],
            epochs=spec.get('ppo_epochs',4),discount=spec.get('discount',0.99)))
    built=make_vec_train(cfg,sim,bank)
    initial_params=msgpack_restore((scratch/'initial.msgpack').read_bytes())['params']
    other_params=msgpack_restore((scratch/'eval_opponent.msgpack').read_bytes())['params']
    if own_action:
        initial_params=expand_own_action_inputs(initial_params)
        other_params=expand_own_action_inputs(other_params)
    initialized=built['init_params'](jax.random.key(0))
    if combat:
        if 'combat_entities' not in initial_params['params']:
            initial_params=merge_combat_params(initialized, initial_params)
        if 'combat_entities' not in other_params['params']:
            other_params=merge_combat_params(initialized, other_params)
    if pcfg.click_proposals:
        initial_params=merge_click_proposal_params(initialized,initial_params)
        other_params=merge_click_proposal_params(initialized,other_params)
    if history:
        if 'visible_history' not in initial_params['params']:
            initial_params=merge_visible_history_params(initialized, initial_params)
        if 'visible_history' not in other_params['params']:
            other_params=merge_visible_history_params(initialized, other_params)
    params=from_state_dict(initialized,initial_params)
    opponent_params=from_state_dict(params,other_params)
    runner=built['initial_runner'](jax.random.key(train_seed),params)
    rollout_fields=('env_state','carry','rng','deadline_ms') + (('visible_history',) if history else ())
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
            template={k:getattr(runner,k) for k in rollout_fields}
            runner=runner._replace(**restore_replay_state(template,payload['rollout_state']))
            continuity='full runner restored'
        else:
            runner=runner._replace(rng=jax.random.fold_in(runner.rng,update))
            continuity='legacy checkpoint: new episodes/carries/RNG; optimizer and schedule preserved'
        print('RESUME',update,continuity,flush=True)
    warmup = None
    if cfg.stagger_initial and not args.resume:
        print('STAGGER WARMUP: fixed policy; shortened episodes discarded',flush=True)
        runner,warmup=warmup_staggered_runner(built,runner,cfg,stop)
        print('STAGGER READY',json.dumps(warmup),flush=True)
    run=RunDir(out,f'vec-s{train_seed}',dict(train={'policy':pcfg._asdict()},ppo=cfg.ppo._asdict(),
        vec={k:v for k,v in cfg._asdict().items() if k not in ('policy','ppo')},
        collector=dict(episode_s=cfg.episode_s,step_ticks=6,unwalkable_click='noop'),
        scenario=spec,environment='jax-vectorised',opponent=spec.get("opponent","mirror-self-play"),
        initialization=('same experiment continuation; optimizer/schedule retained' if args.resume else 'init_from checkpoint parameters only; fresh optimizer and schedule'),
        continuation=dict(checkpoint=str(args.resume),sha256=file_sha256(args.resume),start_update=update,continuity=continuity) if args.resume else None,
        stagger_warmup=warmup,
        init_source_sha256=file_sha256(source),eval_opponent_source=str(opponent_source),
        eval_opponent_sha256=file_sha256(opponent_source),sim=sim.describe(),sim_fingerprint=sim.fingerprint()),
        notes='Finite tower-wave task; not a ten-minute lane score. Frozen evals by initial HP role; AFK parks red at fountain.')
    run.keep_checkpoints=0;failed=False
    def save():
        from .replay_audit import serialize_replay_state
        rollout={k:getattr(runner,k) for k in rollout_fields}
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
    obs=jax.tree.map(lambda x:x[0],built['observe'](sample)[0])
    logits,_=obs_builder.apply(params,obs.entities[None],obs.entity_pad_mask[None],obs.self_vec[None],obs.global_vec[None],obs_builder.initial_carry((1,)))
    assert np.isfinite(np.asarray(logits.button)).all()
    print('SCENARIO READY: actual checkpoint, level3,12minions, both HP roles, next-wave timing verified',flush=True)
    print('COMPILE training',flush=True)
    update_fn=jax.jit(lambda r:built['run_chunk'](r,1)).lower(runner).compile()
    evaluations={}
    for mode in (('afk',) if spec.get('opponent')=='afk' else ('mirror','frozen')):
        ecfg=cfg._replace(n_envs=64,opponent=mode,bank_size=len(eval_bank.t_ms),stagger_initial=False)
        eb=make_vec_train(ecfg,sim,eval_bank,opponent_params=opponent_params if mode=='frozen' else None)
        er=eb['initial_runner'](jax.random.key(int(spec.get('eval_seed',2007))),params)
        indices=jnp.arange(ecfg.n_envs)%len(eval_bank.t_ms)
        er=er._replace(env_state=jax.tree.map(lambda b:b[indices],eval_bank))
        print('COMPILE eval',mode,flush=True)
        evaluations[mode]=(er,jax.jit(eb['collect']).lower(er).compile())
    # Regression gate on the compiled path: endpoint metrics precede reset.
    probe, probe_fn = next(iter(evaluations.values()))
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
        return evaluate_frozen(runner, evaluations, cfg, spec, update, run, stop)
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
