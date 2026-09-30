"""LEARN-PAIR-08: summarize opportunity windows and same-state branches."""
import json,sys
from pathlib import Path
import numpy as np

def main():
    root=Path(sys.argv[1]);rows=json.loads((root/'opportunities.json').read_text());report={}
    for cat in ('caster','melee','champion'):
        cohort=[r for r in rows if r['category']==cat];windows=0;last={}
        for r in cohort:
            ident=r['spawn_seq'];windows+=int(ident not in last or r['index']>last[ident]+1);last[ident]=r['index']
        frames={r['index'] for r in cohort}
        report[cat]=dict(unique_entities=len(last),windows=windows,decision_frames=len(frames),seconds=len(frames)/10,
            within_aa_frames=len({r['index'] for r in cohort if r['distance']<=r['attack_range']}),
            within_aa_ready_frames=len({r['index'] for r in cohort if r['distance']<=r['attack_range'] and not r['e_active'] and r['aa_cd']<=0}),
            spinning_frames=len({r['index'] for r in cohort if r['e_active']}))
    with np.load(root/'trace.npz') as d:
        report['spell_attempts_and_activations']={}
        for side in (0,1):
            report['spell_attempts_and_activations'][str(side)]={}
            for name,button in [('q',3),('w',4),('e',5)]:
                active=d[name+'_active'][:,side]
                report['spell_attempts_and_activations'][str(side)][name]=dict(attempts=int((d['button'][:-1,side]==button).sum()),activations=int((active[1:]&~active[:-1]).sum()),active_s=float(active[:-1].sum()/10))
    cf=root/'counterfactuals.json'
    if cf.exists():
        report['branches']=[]
        for r in json.loads(cf.read_text()):
            c=r['case'];p=c['button_probs']
            report['branches'].append(dict(time=c['seconds'],category=c['category'],e_active=c['e_active'],hp=c['hp'],distance=c['distance'],aa_cd=c['aa_cd'],p_attack_move=p[2],p_move=p[1],p_e=p[5],cursor_near125_mass=c['cursor_within125_mass'],branch=r['branch'],cs_gain=r['cs_gain'],hp_change=r['hp_change'],reward=r['reward'],first_cs_s=r['first_cs_s'],control_error=r['control_error']))
            if 'decisions' in r:
                ds=r['decisions'];gae=0.
                for i in range(len(ds)-1,-1,-1):
                    vn=ds[i+1]['value'] if i+1<len(ds) else r['bootstrap_value']
                    gae=ds[i]['reward']+.99*vn-ds[i]['value']+.99*.95*gae
                report['branches'][-1].update(initial_value=ds[0]['value'],
                    discounted_reward=r['discounted_reward'],bootstrap_value=r['bootstrap_value'],
                    bootstrapped_return=r['bootstrapped_return'],diagnostic_initial_gae=gae)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
