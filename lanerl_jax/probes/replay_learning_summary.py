"""Summarize E36 recorded decisions; windows are descriptive, not causal tests."""
import argparse,json,statistics
from pathlib import Path
import numpy as np


def summarize(path):
    with np.load(path/'trace.npz') as archive:
        d={k:archive[k] for k in archive.files if k not in ('walkable','metadata')}
    n=len(d['t_ms'])-1
    actions=np.array([[r['blue'],r['red']] for r in map(json.loads,(path/'actions.jsonl').read_text().splitlines())])
    assert n==len(actions)
    audit_path=path/'learning_audit.jsonl'
    audit=[json.loads(x) for x in audit_path.read_text().splitlines()] if audit_path.exists() else []
    out={'path':str(path),'sides':[]}
    for side in (0,1):
        alive=d['alive'][:n,side]; move=np.isin(actions[:,side,0],[1,2])
        kinds=d['order_kind'][:n,side]; target=d['order_target'][:n,side]; safe=np.maximum(target,0)
        rows=np.arange(n); direct=kinds==2
        minion=(d['kind'][:n]==2)&(d['team'][:n]!=side)&d['alive'][:n]
        dist=np.hypot(d['x'][:n]-d['x'][:n,side,None],d['y'][:n]-d['y'][:n,side,None])
        near=np.min(np.where(minion,dist,np.inf),axis=1)
        champdist=np.hypot(d['x'][:n,0]-d['x'][:n,1],d['y'][:n,0]-d['y'][:n,1])
        close=alive&d['alive'][:n,1-side]&(champdist<200)
        attacking_champ=direct&(target==1-side)
        row=dict(side=side,cs=int(d['cs'][-1,side]),deaths=int(d['deaths'][-1,side]),
            alive_decisions=int(alive.sum()),movement_clicks=int((move&alive).sum()),
            dropped_movement_clicks=int((move&alive&(kinds==0)).sum()),
            close_champion_decisions=int(close.sum()),close_champion_attack_orders=int((close&attacking_champ).sum()),
            near_minion_alive_fraction=float((alive&(near<250)).sum()/max(alive.sum(),1)),
            enemy_minion_attack_orders=int((direct&(target>=0)&minion[rows,safe]).sum()))
        if audit:
            teacher=np.array([r['teacher'][side] for r in audit]); opportunity=alive&(teacher[:,0]==2)
            row.update(teacher_attack_opportunities=int(opportunity.sum()),
                actual_attack_button_on_teacher_opportunity=int((opportunity&(actions[:,side,0]==2)).sum()))
            for name in ['initial','final']:
                tp=np.array([r[name]['button_prob'][side][2] for r in audit]); lp=np.array([r[name]['teacher_logp'][side] for r in audit])
                row[name+'_attack_prob_on_teacher_opportunity']=float(tp[opportunity].mean()) if opportunity.any() else None
                row[name+'_teacher_nll_on_opportunity']=float(-lp[opportunity].mean()) if opportunity.any() else None
        # Largest five-second HP drops, excluding deaths/respawns; disjoint starts.
        hp=d['hp'][:,side]; windows=[]; occupied=[]
        candidates=[]
        for i in range(n-50):
            j=i+50
            if not d['alive'][i:j+1,side].all() or d['deaths'][j,side]!=d['deaths'][i,side]:continue
            candidates.append((float(hp[i]-hp[j]),i,j))
        for loss,i,j in sorted(candidates,reverse=True):
            if loss<=0 or any(abs(i-k)<100 for k in occupied):continue
            occupied.append(i)
            focus=(slice(i,j),side)
            w=dict(start_s=float(d['t_ms'][i]/1000),end_s=float(d['t_ms'][j]/1000),hp_loss=loss,
                cs_gained=int(d['cs'][j,side]-d['cs'][i,side]),
                displacement=float(np.hypot(d['x'][j,side]-d['x'][i,side],d['y'][j,side]-d['y'][i,side])),
                buttons=np.bincount(actions[i:j,side,0],minlength=8).tolist(),
                orders={str(k):int((kinds[i:j]==k).sum()) for k in np.unique(kinds[i:j])},
                minion_attackers_mean=float(((d['target'][i:j]==side)&minion[i:j]).sum(axis=1).mean()),
                windup_frames=int(((d['aa_windup'][focus]>0)&d['is_attacking'][focus]).sum()))
            if audit:
                w['teacher_buttons']=np.bincount(teacher[i:j,0],minlength=8).tolist()
                w['reward_sum']=float(sum(r['reward'][side] for r in audit[i:j]))
            windows.append(w)
            if len(windows)==3:break
        row['damage_windows']=windows;out['sides'].append(row)
    return out

def training_summary(path):
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    # E33/E34 each collect128 envs x128 decisions x2 champions per update.
    return dict(path=str(path), decisions=len(rows)*32768,
        button_counts={k:round(sum(r[k]*32768 for r in rows))
                       for k in rows[0] if k.startswith('button_')},
        windows=[dict(first=part[0]['update'],last=part[-1]['update'],
            explained_variance_median=statistics.median(r['explained_variance'] for r in part),
            entropy_median=statistics.median(r['entropy'] for r in part))
            for part in [rows[:100],rows[-100:]]])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',type=Path,nargs='*')
    p.add_argument('--metrics',type=Path,nargs='*',default=[]);a=p.parse_args()
    print(json.dumps(dict(replays=[summarize(x) for x in a.paths],
        training=[training_summary(x) for x in a.metrics]),indent=2))
