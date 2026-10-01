"""TOOL: frozen AFK comparison; --group debug plots the current debugging arms."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--group',choices=['lr','debug'],default='lr')
group=parser.parse_args().group
experiments=[('E50_afk_personal_lr3e5','LR 0.00003'),('E51_afk_personal_lr1e4','LR 0.0001'),('E52_afk_entropy','LR 0.0001 + entropy 0.01'),('E53_afk_shared_critic','LR 0.0001 + shared critic')]
if group=='debug':
 experiments=[('E60_afk_own_action','E60: own combat inputs'),('E61_afk_long_control','E61: unchanged control'),('E62_afk_own_death_cost','E62: own inputs + death cost'),('E63_afk_death_control','E63: death cost only'),('E66_afk_click_proposals','E66: death cost + click proposals')]
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for exp,label in experiments:
 study_path=Path('/mnt/nfs/checkpoints/lanerl-jax',exp,'study.json')
 if not study_path.exists():
  print('No results yet:',exp);continue
 study=json.loads(study_path.read_text())
 evaluation_path=Path(study['path'])/'evaluations.jsonl'
 if not evaluation_path.exists():
  print('No frozen evaluations yet:',exp);continue
 rows=[json.loads(l) for l in evaluation_path.read_text().splitlines()]
 assert all(r['frozen'] and r['games']==64 and r['duration_s']==120 for r in rows)
 for ax,key,title in zip(axes,['cs','deaths','tower_damage'],['CS / game','Deaths / game','Garen tower damage / game (HP)']):
  ax.plot([r['update']*16384/1e6 for r in rows],[np.mean([e[key] for e in r['episodes'] if e['team']==0]) for r in rows],'-o',label=label)
  ax.set(title=title,xlabel='Additional decisions (millions)');ax.grid(alpha=.2)
axes[0].legend(fontsize=7)
for ax in axes:ax.set_ylim(bottom=0)
if group=='lr':axes[0].set_ylim(0,12)
fig.suptitle(('E46 checkpoint weights; E66 changes the initial click distribution' if group=='debug' else
             'Same E46 start; zero XP reward, personal tower shaping')+
            '\nFrozen: 64 games per point, 120s games; one training seed per arm')
p=ROOT/'docs/figures'/('E50_E51_comparison.png' if group=='lr' else 'afk_debug_comparison.png')
fig.savefig(p,dpi=150);print(p)
