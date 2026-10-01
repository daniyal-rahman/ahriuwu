"""PROBE: isolated C# modern-champion startup, resource and cast checks."""
import json
import os
from pathlib import Path
import sys
from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv
from lanerl_train.ports import PortAllocator
from lanerl_jax.parity.script_health import assert_all_scripts_loaded


def run(server,out):
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    config=json.loads((root/'lanerl/cfg/modern_garen_jax_26_19.json').read_text())
    config['gameInfo']['CONTENT_PATH']=str(Path(server).parents[1]/'Content')
    cfg=out/'game.json';cfg.write_text(json.dumps(config))
    spec=ServerLaunchSpec(config_path=cfg,server_dir=Path(server),dotnet_root=Path('/mnt/nfs/projects/lanerl-vendor/dotnet'),
                         step_ticks=6,bot_teams='none',extra_env={'LANERL_AUTOBUY':'0'})
    env=VecLaneEnv(1,spec=spec,ports=PortAllocator(base=27100).allocate(1),log_dir=out/'server',auto_restart=False)
    rows=[]
    def step(action=None):
        res=env.step([action]);assert all(res.alive),res.died
        obs=res.obs[0];rows.append(obs);return {u['tm']:u for u in obs['u'] if u['k']=='Champion'}
    try:
        env.start()
        for h in env.handles:
            os.sched_setaffinity(h.proc.pid,{min(os.sched_getaffinity(0))})
        assert_all_scripts_loaded(out/'server/instance000.log')
        for _ in range(25): champs=step()
        assert champs[100]['mhp']==690,champs[100]
        assert champs[200]['mhp']==650,champs[200]
        assert abs(champs[100]['ad']-69)<.01,champs[100]
        assert abs(champs[200]['ad']-68)<.01,champs[200]
        assert champs[200]['modern']['maxMana']==339
        print('CANARY PASSED: all C# scripts loaded, both modern stat profiles verified',flush=True)
        before=champs[200]['modern']['mana']
        champs=step({'blue':{'t':'cast','slot':2},'red':{'t':'cast','slot':2}})
        active=[]
        for _ in range(35):
            active.append([champs[t]['modern'] for t in (100,200)])
            champs=step()
        assert any(a[0]['e']>0 for a in active),active[:3]
        assert any(a[1]['e']>0 for a in active),active[:3]
        assert min(a[1]['mana'] for a in active)<before-45,active[:3]
        assert max(a[0]['spinCount'] for a in active)==7,active
        assert champs[100]['modern']['e']==0 and champs[200]['modern']['e']==0
        step({'cmd':'reset'})
        for _ in range(25):champs=step()
        for t in (100,200):
            assert champs[t]['modern']['e']==0
            assert champs[t]['modern']['passiveStacks']==0
        assert champs[200]['modern']['mana']==339
        # Fixed scripted lane smoke; this is a mechanics diagnostic, not a
        # frozen-policy evaluation or a matchup-balance estimate.
        routes={100:[(1950.,12350.)],200:[(11000.,13600.),(7500.,13700.),(4500.,13600.),(2431.,12741.)]}
        legs={100:0,200:0}
        for tick in range(1800):
            action={}
            for team,key in ((100,'blue'),(200,'red')):
                ch=champs[team]
                if ch['dead']:continue
                distance=lambda u:(ch['x']-u['x'])**2+(ch['y']-u['y'])**2
                if tick<1100:
                    route=routes[team];leg=legs[team];goal=route[leg]
                    if (ch['x']-goal[0])**2+(ch['y']-goal[1])**2<200**2 and leg<len(route)-1:legs[team]+=1;goal=route[leg+1]
                    action[key]={'t':'move','x':goal[0],'y':goal[1]};continue
                enemies=[u for u in rows[-1]['u'] if u['tm']!=team and u['k'] in ('Champion','LaneMinion') and u['hp']>0 and distance(u)<800**2]
                if not enemies:continue
                target=min(enemies,key=distance);dist=distance(target)**.5
                chosen=None
                for slot in (3,2,1,0):
                    if ch['sl'][slot]<=0 or ch.get('cd'+str(slot),0)>0:continue
                    if slot==3 and (target['k']!='Champion' or dist>375):continue
                    if slot==2 and dist>350:continue
                    if slot==1 and (dist>240 or ch['modern']['w']>0):continue
                    if slot==0 and dist>(225 if team==100 else 700):continue
                    if team==200 and ch['modern']['mana']<[50,30,40+10*ch['sl'][2],100][slot]:continue
                    chosen={'t':'cast','slot':slot,'id':target['id']};break
                action[key]=chosen or {'t':'attack','id':target['id']}
            champs=step(action)
            assert all(c['modern']['mana']>=-.001 for c in champs.values())
        assert ' ERROR ' not in (out/'server/instance000.log').read_text(), 'Server logged an exception during scripted combat'
        for team in (100,200):
            assert any(u['tm']==team and u['k']=='Champion' and u['y']>12000 and u['x']<3000 for row in rows for u in row['u']), 'Champion failed to reach top lane'
        print('SCRIPTED LANE COMPLETE: 180 simulated seconds, both reached lane, no server errors',flush=True)
        (out/'result.json').write_text(json.dumps({'wire_and_scripted_passed':True,'frames':len(rows),'final':champs},indent=2))
    finally:
        (out/'trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        env.close()
    import subprocess
    selftest_env=spec.environment(env.ports[0]);selftest_env['LANERL_MODERN_SELFTEST']='1'
    log=out/'combat-selftest.log'
    with log.open('w') as fh:
        result=subprocess.run(spec.command(env.ports[0].game),cwd=spec.resolved_server_dir(),env=selftest_env,stdout=fh,stderr=subprocess.STDOUT,timeout=180)
    assert result.returncode==0,log.read_text()[-3000:]
    assert_all_scripts_loaded(log)
    assert 'MODERN COMBAT SELFTEST PASSED' in log.read_text(),log.read_text()[-3000:]
    result=json.loads((out/'result.json').read_text());result['passed']=True;result['combat_selftest_passed']=True
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print('MODERN COMBAT SELFTEST PASSED',flush=True)
    collector_smoke(server, out)
    print('MODERN VALIDATION COMPLETE',flush=True)

def collector_smoke(server, out):
    """Real policy -> collector -> wire -> observation, resets and workers."""
    import jax
    import numpy as np
    from flax.serialization import to_bytes
    from lanerl_jax.train.server_train import (ServerCollector, MultiProcessCollector,
        evaluate_frozen, load_checkpoint_policy)
    from lanerl_jax.train.policy import LanePolicy, PolicyConfig
    from lanerl_jax.train.run_manifest import RunDir
    results = []
    # Small networks exercise the normal forward/sampling/checkpoint contract;
    # this is an integration check, not a policy-quality evaluation.
    for workers, pair, core in ((1, ('Garen','Jax'), 'mlp'), (2, ('Jax','Garen'), 'gru')):
        cfg = PolicyConfig(self_dim=28, core=core, d_model=16, n_heads=2,
                           n_layers=1, ffn_dim=32, ctx_dim=16, core_dim=16,
                           mlp_hidden=32, mlp_layers=1)
        run = RunDir(out/'collector', f'workers{workers}', {
            'train': {'policy': cfg._asdict()}, 'collector': {'modern_champions': pair}})
        kwargs = dict(n=workers, out=run.path, port_base=27500+200*workers,
                      episode_s=4, step_ticks=6, server_dir=Path(server), teams=(0,1),
                      modern_champions=pair)
        collector = (ServerCollector(**kwargs) if workers==1 else
                     MultiProcessCollector(**kwargs, workers=workers))
        try:
            obs,_ = collector.observe()
            assert obs.self_vec.shape==(workers*2,28)
            expected = [[1,0,0,1],[0,1,1,0]] if pair[0]=='Garen' else [[0,1,1,0],[1,0,0,1]]
            np.testing.assert_array_equal(np.asarray(obs.self_vec)[:,16:20],expected*workers)
            policy = LanePolicy(cfg)
            args=(obs.entities,obs.entity_pad_mask,obs.self_vec,obs.global_vec)
            if core=='gru':args += (policy.initial_carry((workers*2,)),)
            params=policy.init(jax.random.key(7),*args)
            checkpoint=run.path/'probe.msgpack';checkpoint.write_bytes(to_bytes({'params':params}))
            loaded, restored=load_checkpoint_policy(checkpoint)
            for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(restored)):
                np.testing.assert_array_equal(a,b)
            episodes=evaluate_frozen(collector,loaded,restored,run,seed=7,
                                     episodes_per_env=2,diag_steps=False)
            assert len(episodes)==workers*4
            assert all(e==2 for e in collector.episodes)
            results.append(dict(workers=workers,pair=pair,core=core,episodes=len(episodes)))
        finally:
            collector.close()
    (out/'collector-result.json').write_text(json.dumps(results,indent=2))
    print('MODERN COLLECTOR PASSED: MLP/GRU, checkpoint reload, both sides, workers and resets',flush=True)


if __name__=='__main__':run(sys.argv[1],sys.argv[2])
