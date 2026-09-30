"""TOOL: frozen E50/E51 matched learning-rate comparison, one training seed."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for exp,label in [('E50_afk_personal_lr3e5','LR 0.00003'),('E51_afk_personal_lr1e4','LR 0.0001')]:
 study=json.loads(Path('/mnt/nfs/checkpoints/lanerl-jax',exp,'study.json').read_text())
 rows=[json.loads(l) for l in (Path(study['path'])/'evaluations.jsonl').read_text().splitlines()]
 for ax,key,title in zip(axes,['cs','deaths','tower_damage'],['CS / game','Deaths / game','Garen tower damage / game (HP)']):
  ax.plot([r['update']*16384/1e6 for r in rows],[np.mean([e[key] for e in r['episodes'] if e['team']==0]) for r in rows],'-o',label=label)
  ax.set(title=title,xlabel='Additional decisions (millions)');ax.grid(alpha=.2)
axes[0].legend();axes[0].set_ylim(0,12)
fig.suptitle('Same E46 start; zero XP reward, personal tower shaping\nFrozen: 64 games per point, 120s games; one training seed per arm')
p=ROOT/'docs/figures/E50_E51_comparison.png';fig.savefig(p,dpi=150);print(p)
