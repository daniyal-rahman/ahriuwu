"""Frozen wave evaluator shared by PPO and imitation studies; identical cohort contract."""
import json
from pathlib import Path
import jax
import numpy as np

def alive_spell_count(buttons, self_obs):
    """Count per-team selections over time; self_obs is [time, team, features]."""
    alive = np.asarray(self_obs)[..., 14] < .5
    buttons = np.asarray(buttons)
    if buttons.shape != alive.shape:
        raise ValueError(f"action/observation shape mismatch: {buttons.shape} vs {alive.shape}")
    return ((buttons >= 3) & (buttons <= 6) & alive).sum(axis=0)


def evaluate_frozen(runner, evaluations, cfg, spec, update, run, stop=lambda: False):
    results = []
    for mode,(template,fn) in evaluations.items():
        r=template._replace(params=runner.params);seen=np.zeros(64,bool);rows=[]
        low=np.asarray(template.env_state.hp[:,:2]/template.env_state.max_hp[:,:2])<.85
        first_seen_hp=np.full((64,2),np.nan)
        deaths=np.zeros((64,2));returns=np.zeros((64,2));alive_spells=np.zeros((64,2))
        reward_terms={}
        for _ in range(int(np.ceil(spec['duration_s']*10/128))+2):
            if stop():return
            r,tr,_=jax.block_until_ready(fn(r))
            done=np.asarray(tr.done_full[:,:,0])
            chunk_terms={k:np.asarray(v) for k,v in tr.reward_terms.items()}
            for k in chunk_terms:
                if k not in reward_terms:reward_terms[k]=np.zeros((64,2))
            for env in np.flatnonzero(~seen):
                hits=np.flatnonzero(done[:,env]);end=int(hits[0])+1 if len(hits) else len(done)
                for team in (0,1):
                    visible=np.flatnonzero(np.asarray(tr.obs_global[:end,env,team,1])>.5)
                    if np.isnan(first_seen_hp[env,team]) and len(visible):
                        first_seen_hp[env,team]=float(tr.obs_self[visible[0],env,team,2])
                deaths[env]+=np.asarray(tr.deaths[:end,env]).sum(0)
                returns[env]+=np.asarray(tr.reward[:end,env]).sum(0)
                for k,v in chunk_terms.items():reward_terms[k][env]+=v[:end,env].sum(0)
                alive_spells[env]+=alive_spell_count(tr.action[0][:end,env],tr.obs_self[:end,env])
                if len(hits):
                    t=end-1;seen[env]=True
                    for team in (0,1):
                        rows.append(dict(env=int(env),team=team,low_hp=bool(low[env,team]),
                            cs=float(tr.cs[t,env,team]),gold=float(tr.gold[t,env,team]),
                            gold_diff=float(tr.gold[t,env,team]-tr.gold[t,env,1-team]),
                            first_enemy_seen_hp=float(first_seen_hp[env,team]) if np.isfinite(first_seen_hp[env,team]) else None,
                            tower_damage=float(tr.tower_damage_at_end[t,env,team]),
                            kills=float(tr.kills_at_end[t,env,team]),deaths=float(deaths[env,team]),
                            hp_fraction=float(tr.hp_at_end[t,env,team]),reward=float(returns[env,team]),
                            reward_terms={k:float(v[env,team]) for k,v in reward_terms.items()},
                            spell_selections=float(alive_spells[env,team])))
            if seen.all():break
        if not seen.all():raise RuntimeError('incomplete scenario eval')
        np.testing.assert_allclose(sum(reward_terms.values()),returns,rtol=1e-5,atol=1e-4)
        if cfg.cs_only:
            for row in rows:
                np.testing.assert_equal(row['reward'], row['cs'])
                np.testing.assert_equal(row['reward_terms']['cs'], row['cs'])
                assert all(value == 0. for key,value in row['reward_terms'].items() if key != 'cs')
        summaries={}
        for team in (0,1):
            for disadvantaged in (True,False):
                cohort=[x for x in rows if x['team']==team and x['low_hp']==disadvantaged]
                if not cohort: continue
                summaries[f'{team}_{"low" if disadvantaged else "full"}']={k:float(np.mean([x[k] for x in cohort]))
                    for k in ('cs','gold_diff','kills','deaths','hp_fraction','reward','spell_selections','tower_damage')}
                contact=[x['first_enemy_seen_hp'] for x in cohort if x['first_enemy_seen_hp'] is not None]
                summaries[f'{team}_{"low" if disadvantaged else "full"}']['first_enemy_seen_hp']=float(np.mean(contact)) if contact else None
                summaries[f'{team}_{"low" if disadvantaged else "full"}']['reward_terms']={
                    k:float(np.mean([x['reward_terms'][k] for x in cohort])) for k in reward_terms}
        result=dict(update=update,frozen=True,opponent=mode,duration_s=spec['duration_s'],games=64,
                    summaries=summaries,episodes=rows)
        if update == 0 and spec.get('initial_eval_reference'):
            reference=spec['initial_eval_reference']
            saved=[json.loads(line) for line in Path(reference['path']).read_text().splitlines()]
            matches=[r for r in saved if r['update']==reference['update'] and r['opponent']==mode]
            if len(matches)!=1:
                raise RuntimeError('initial frozen reference must identify exactly one evaluation')
            prior={(r['env'],r['team']):r for r in matches[0]['episodes']}
            assert len(prior)==len(rows)
            for row in rows:
                previous=prior[(row['env'],row['team'])]
                for field in ('cs','deaths','kills','spell_selections','low_hp'):
                    if row[field]!=previous[field]:
                        raise RuntimeError(f'initial frozen trajectory changed: env{row["env"]}/team{row["team"]}/{field}')
                fields=('gold','gold_diff','tower_damage','hp_fraction')
                if reference.get('compare_reward', True):
                    fields += ('reward',)
                for field in fields:
                    np.testing.assert_allclose(row[field],previous[field],rtol=1e-5,atol=1e-4,
                        err_msg=f'initial frozen reference: {field}')
            print('INITIAL FROZEN REFERENCE PASSED: all64games, bothteams, E67source',flush=True)
        with (run.path/'evaluations.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        print('FROZEN',mode,update,json.dumps(summaries),flush=True)
        results.append(result)
    return results
