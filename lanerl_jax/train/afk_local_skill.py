"""Opt-in short farming study: matched-start PPO and scripted imitation.

Only reset conditions differ. Native observations, clicks, GRU, timing, spell
availability and simulator physics are preserved. Diagnostics see state; actors do not.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.serialization import from_state_dict, msgpack_restore

from ..sim.config import DEFAULT_ROUTE_ARTIFACT, SimConfig
from ..sim.init import init_lane, spawn_minion, TOP_LANE_PATH, TOP_OUTER_TURRET
from ..sim.orders import Orders, OrderKind
from ..sim.profiles import profile_id
from ..sim.state import Kind, MI_SLICE
from ..sim.step import env_step
from ..parity.policy_driver import _lane_frames
from .actions import orders_from
from .afk_imitation import capture, label_batch, make_update, setup_collector
from .policy import PolicyConfig
from .ppo import PPOConfig
from .run_manifest import RunDir, file_sha256
from .vec_train import VecConfig, make_vec_train
from .wave_evaluation import evaluate_frozen
from .wave_scenario import START_MS, park_afk_opponent, point_on_path, prepare_scenario_bank, turret_distance


def prepare_local_bank(sim, out, seed, count=64):
    """Fresh level3 starts: one or three enemy minions, with allied competition.

    Units have normal ready AA/spell clocks. No mid-episode hidden-state splice,
    projectiles, imposed cooldowns or forced holds. RNG changes HP and geometry.
    """
    if count % 2:
        raise ValueError('balanced one/three-minion bank requires an even size')
    out.mkdir(parents=True, exist_ok=True)
    s = init_lane(seed=seed).replace(t_ms=jnp.float32(START_MS),
        next_spawn_ms=jnp.float32(START_MS+30000))
    s = s.replace(xp=s.xp.at[:2].set(sim.params['xp_to_reach_level'][3]))
    noop = Orders(jnp.zeros(2,jnp.int8),jnp.zeros(2),jnp.zeros(2),jnp.full(2,-1,jnp.int8))
    s = jax.block_until_ready(jax.jit(lambda x: env_step(x,noop,sim))(s))
    s = s.replace(t_ms=jnp.float32(START_MS))
    rng = np.random.default_rng(seed)
    path = np.asarray(TOP_LANE_PATH, float)
    length = np.linalg.norm(np.diff(path,axis=0),axis=1).sum()
    middle = (turret_distance(path,TOP_OUTER_TURRET[0])+turret_distance(path,TOP_OUTER_TURRET[1]))/2
    rows, champs, positions, vertices, health, enabled = [], [], [], [], [], []
    for i in range(count):
        n = 1 if i % 2 == 0 else 3
        centre = middle + rng.uniform(-150,150)
        distance = rng.uniform(100,300)
        champ,_ = point_on_path(path,centre-distance)
        xy, keys, hp, live = [], [], [], []
        for k in range(3):
            for team in (1,0):
                d = centre+70*k if team else centre-100-90*k
                p,vertex = point_on_path(path if team==0 else path[::-1],d if team==0 else length-d)
                xy.append(p); keys.append(vertex)
                hp.append(rng.uniform(.10,.65) if team else 1.)
                live.append(k<n)
        champs.append(champ); positions.append(xy); vertices.append(keys)
        health.append(hp); enabled.append(live)
        rows.append(dict(index=i,enemy_minions=n,champion_distance=distance,
            champion_xy=champ.tolist(),minion_xy=np.asarray(xy).tolist(),hp_fraction=hp))
    # Build one constant six-slot template, then vary only legitimate reset fields.
    for k in range(3):
        for team in (1,0):
            model=profile_id(Kind.LANE_MINION,0 if k==0 else 1,team)
            s=spawn_minion(s,team,model,sim.params['max_hp'][model],
                jnp.asarray(path if team==0 else path[::-1]),spawn_xy=positions[0][2*k+(1-team)])
    units=jnp.arange(MI_SLICE.start,MI_SLICE.start+6)
    def vary(champ,xy,vertex,hp,live):
        return s.replace(x=s.x.at[0].set(champ[0]).at[units].set(xy[:,0]),
            y=s.y.at[0].set(champ[1]).at[units].set(xy[:,1]),
            collision_x=s.collision_x.at[0].set(champ[0]).at[units].set(xy[:,0]),
            collision_y=s.collision_y.at[0].set(champ[1]).at[units].set(xy[:,1]),
            collision_present=s.collision_present.at[units].set(live),
            waypoints=s.waypoints.at[0,0].set(champ).at[units,0].set(xy),
            n_waypoints=s.n_waypoints.at[0].set(1),
            waypoint_key=s.waypoint_key.at[0].set(1),
            lane_waypoint_key=s.lane_waypoint_key.at[units].set(vertex),
            hp=s.hp.at[units].set(s.max_hp[units]*hp),alive=s.alive.at[units].set(live))
    bank=park_afk_opponent(jax.jit(jax.vmap(vary))(
        jnp.asarray(champs,jnp.float32),jnp.asarray(positions,jnp.float32),
        jnp.asarray(vertices,jnp.int32),jnp.asarray(health,jnp.float32),jnp.asarray(enabled,bool)))
    (out/'setup.json').write_text(json.dumps(rows,indent=2))
    return bank, rows


def diagnostic_counts(state, actual, teacher):
    """Command comparisons, NOT counterfactual proof that a choice loses CS.

    MOVE clicks can decode to ATTACK, so compare decoded targets, not buttons.
    Script's conservative killable decision is the reference, not an oracle.
    """
    ta=teacher.kind[0]==OrderKind.ATTACK
    pa=actual.kind[0]==OrderKind.ATTACK
    ready=(state.aa_cooldown[0]<=0)&(state.aa_windup[0]<=0)
    opportunity=ta&ready
    same=pa&(actual.target[0]==teacher.target[0])
    return jnp.array([opportunity,opportunity&same,opportunity&pa&~same,
        opportunity&~pa,pa&~ta],jnp.int32)


def make_diagnostic_step(built, sim):
    frame=_lane_frames()[0]
    def decode(state, action):
        return orders_from(tuple(action),state,None,frame,snap_moves=False,
            params=sim.params,vision=sim.vision,drop_unwalkable_moves=True)
    @jax.jit
    def step(runner):
        after,tr,_=built['collect'](runner)
        labels=label_batch(tr.obs_entities[:,:,0],tr.obs_mask[:,:,0],tr.obs_self[:,:,0],tr.obs_global[:,:,0])[0]
        teacher=jnp.zeros((labels.shape[0],3,2),jnp.int32).at[:,:,0].set(labels)
        actual=jnp.stack([a[0] for a in tr.action],axis=1)
        metrics=jax.vmap(diagnostic_counts)(runner.env_state,
            jax.vmap(decode)(runner.env_state,actual),jax.vmap(decode)(runner.env_state,teacher))
        return after,metrics,tr.cs_delta[0,:,0],tr.done_full[0,:,0]
    return step


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('experiment')
    args=parser.parse_args();spec=json.loads(Path('experiments',args.experiment+'.json').read_text())
    started=time.monotonic();signals=[]
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        signal.signal(sig,lambda signum,frame:signals.append(signum))
    def stop():return bool(signals) or time.monotonic()-started>=spec['max_seconds']
    def check():
        if stop():raise InterruptedError('study time/signal bound reached')
    jax.config.update('jax_default_matmul_precision','highest')
    jax.config.update('jax_compilation_cache_dir','/scratch/lanerl-jax-compilation-cache')
    scratch=Path('/scratch')/(spec['id']+'-'+os.environ['SLURM_JOB_ID']);scratch.mkdir(parents=True)
    shutil.copytree(DEFAULT_ROUTE_ARTIFACT,scratch/'routes')
    shutil.copyfile(spec['init_from'],scratch/'initial.msgpack')
    if file_sha256(scratch/'initial.msgpack')!=spec['init_sha256']:raise RuntimeError('source SHA mismatch')
    out=Path('/mnt/nfs/checkpoints/lanerl-jax')/spec['id'];out.mkdir(parents=True,exist_ok=True)
    sim=SimConfig.training(route_artifact=scratch/'routes').replace(step_ticks=6)
    bank,_=prepare_local_bank(sim,out/'train_bank',spec['bank_train_seed'])
    held,cohort=prepare_local_bank(sim,out/'held_bank',spec['bank_eval_seed'])
    full=park_afk_opponent(prepare_scenario_bank(sim,out/'full_bank',spec['eval_offsets'],1007))
    pcfg=PolicyConfig(core='gru',core_norm=True,core_residual=True,detach_critic=True)
    cfg=VecConfig(n_envs=128,rollout_steps=128,n_updates=spec['updates'],n_minibatches=4,
        episode_s=START_MS/1000+spec['duration_s'],observation_horizon_s=600.,
        stagger_initial=False,bank_size=len(bank.t_ms),policy=pcfg,opponent='afk',
        cs_only=True,xp_scale=0.,tower_damage_personal=True,lr_anneal=False,
        ppo=PPOConfig.standard(lr=spec['lr'],entropy_coef=.001,epochs=4,discount=.99))
    built=make_vec_train(cfg,sim,bank)
    initial=from_state_dict(built['init_params'](jax.random.key(0)),
        msgpack_restore((scratch/'initial.msgpack').read_bytes())['params'])
    ecfg=cfg._replace(n_envs=64)
    _,er,efn=setup_collector(ecfg,sim,held,initial,2007)
    fullcfg=ecfg._replace(episode_s=START_MS/1000+120,bank_size=len(full.t_ms))
    _,fr,ffn=setup_collector(fullcfg,sim,full,initial,2007)
    db,dr,_=setup_collector(ecfg._replace(rollout_steps=1),sim,held,initial,2007)
    dstep=make_diagnostic_step(db,sim)
    run=RunDir(out,'local-s0',dict(train={'policy':pcfg._asdict()},scenario=spec,
        ppo=cfg.ppo._asdict(),init_source_sha256=spec['init_sha256'],sim=sim.describe()),
        notes='Matched initial E78 parameters; synthetic short starts; fixed BC/DAgger and PPO endpoints.')
    run.keep_checkpoints=0
    phase='initial';status='running';completed=[]
    params=initial;optimizer=built['initial_runner'](jax.random.key(0),initial).opt_state;steps=0
    def save():
        ck=run.save(steps,len(completed),dict(params=params,opt_state=optimizer,step=steps))
        (out/'study.json').write_text(json.dumps(dict(job=os.environ['SLURM_JOB_ID'],
            path=str(run.path),phase=phase,status=status,completed=completed,checkpoint=str(ck),
            elapsed_s=time.monotonic()-started),indent=2))
        return ck
    def evaluate(name,p,teacher=False,reference=False):
        check();dest=run.path/name;dest.mkdir(exist_ok=True)
        local_er,local_fn=er,efn
        if teacher:
            _,local_er,local_fn=setup_collector(ecfg,sim,held,p,2007,teacher=True)
        local_spec={'duration_s':spec['duration_s']}
        local=evaluate_frozen(SimpleNamespace(params=p),{'afk':(local_er,local_fn)},ecfg,
            local_spec,0,SimpleNamespace(path=dest),stop)
        if local is None:raise InterruptedError('local eval interrupted')
        short=sorted((r for r in local[0]['episodes'] if r['team']==0),key=lambda r:r['env'])
        summaries={}
        for n in (1,3):
            subset=[r for r in short if cohort[r['env']]['enemy_minions']==n]
            summaries[str(n)]=dict(mean_cs=float(np.mean([r['cs'] for r in subset])),
                fraction_collected=float(np.mean([r['cs']/n for r in subset])))
        if not teacher:
            rr=dr._replace(params=p);seen=np.zeros(64,bool);counts=np.zeros((64,5),int);cs=np.zeros(64)
            for _ in range(int(spec['duration_s']*10)+3):
                check();rr,c,d,done=jax.block_until_ready(dstep(rr))
                counts+=np.asarray(c)*(~seen)[:,None];cs+=np.asarray(d)*(~seen)
                seen|=np.asarray(done)
                if seen.all():break
            assert seen.all();np.testing.assert_array_equal(cs,[r['cs'] for r in short])
            (dest/'command_diagnostics.json').write_text(json.dumps(dict(
                fields=['ready_teacher_attack','same_target_attack','different_target_attack',
                    'no_direct_attack','attack_when_teacher_positions'],
                per_episode=counts.tolist(),sum=counts.sum(0).tolist(),
                caveat='Decoded command agreement, not lost-CS attribution; spells and held orders remain legal.'),indent=2))
            fs={'duration_s':120}
            if reference:fs['initial_eval_reference']=spec['initial_eval_reference']
            result=evaluate_frozen(SimpleNamespace(params=p),{'afk':(fr,ffn)},fullcfg,
                fs,0,SimpleNamespace(path=dest),stop)
            if result is None:raise InterruptedError('transfer eval interrupted')
            summaries['full_120s_cs']=float(np.mean([r['cs'] for r in result[0]['episodes'] if r['team']==0]))
        (dest/'scores.json').write_text(json.dumps(summaries,indent=2))
        print('LOCAL_FROZEN',name,json.dumps(summaries),flush=True)
        return summaries
    try:
        save();evaluate('initial',initial,reference=True);evaluate('teacher',initial,teacher=True)
        # Supervised arm, independent of PPO: fixed teacher round then own-state relabelling.
        phase='teaching';tx=optax.chain(optax.clip_by_global_norm(.5),optax.adam(spec['lr']))
        optimizer=tx.init(initial);update=make_update(built['policy'],tx,128,4.)
        datasets=[];rng=np.random.default_rng(spec['train_seed'])
        for round_idx,epochs in enumerate(spec['epochs_per_round']):
            _,cr,cf=setup_collector(ecfg,sim,bank,params,3107+round_idx,teacher=round_idx==0)
            data=capture(cr,cf,spec['duration_s'],stop);data.pop('behavior');datasets.append(data)
            np.savez_compressed(scratch/f'round{round_idx}.npz',**data)
            shutil.copyfile(scratch/f'round{round_idx}.npz',run.path/f'round{round_idx}.npz')
            merged=jax.tree.map(jnp.asarray,{k:np.concatenate([d[k] for d in datasets]) for k in data})
            n,length=merged['valid'].shape
            for epoch in range(epochs):
                losses=[]
                for ids in rng.permutation(n).reshape(-1,8):
                    for t in range(0,length,128):
                        check();params,optimizer,m=jax.block_until_ready(update(params,optimizer,
                            jax.tree.map(lambda a:a[ids],merged),jnp.int32(t)))
                        if not all(np.isfinite(float(v)) for v in m.values()):raise RuntimeError('nonfinite teaching')
                        losses.append(float(m['loss']));steps+=1024
                run.log(dict(phase=phase,round=round_idx,epoch=epoch+1,loss=float(np.mean(losses))))
                if (epoch+1)%10==0:save()
        save();shutil.copyfile(run.path/'ckpt_latest.msgpack',run.path/'teaching_final.msgpack')
        evaluate('teaching_final',params);completed.append('teaching')
        phase='ppo';runner=built['initial_runner'](jax.random.key(spec['train_seed']),initial)
        params=initial;optimizer=runner.opt_state
        update_ppo=jax.jit(lambda r:built['run_chunk'](r,1))
        for u in range(1,spec['updates']+1):
            check();runner,m=jax.block_until_ready(update_ppo(runner))
            params=runner.params;optimizer=runner.opt_state;steps+=cfg.n_envs*cfg.rollout_steps
            metrics={k:float(np.asarray(v).reshape(-1)[0]) for k,v in m.items()}
            if metrics['loss_nonfinite'] or not np.isfinite(metrics['policy_loss']):raise RuntimeError('nonfinite PPO')
            run.log(dict(phase=phase,update=u,**metrics))
            if u%64==0:save();print('LOCAL_PPO',u,flush=True)
        save();shutil.copyfile(run.path/'ckpt_latest.msgpack',run.path/'ppo_final.msgpack')
        evaluate('ppo_final',params);completed.append('ppo');status='complete'
    except InterruptedError:
        status='interrupted';raise
    except Exception:
        status='failed';raise
    finally:
        save();run.set_results(status=status,completed=completed);run.close()
    print('PROFILE COMPLETE',flush=True)


if __name__=='__main__':main()
