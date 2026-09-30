"""TOOL: E46 frozen learning curves and explicitly labeled training diagnostics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
RUN=Path('/mnt/nfs/checkpoints/lanerl-jax/E46_afk_farm/vec-s0-20260930-214903-32499e76')
OUT=ROOT/'docs/figures/E46_training_curve.png'
e=[json.loads(l) for l in (RUN/'evaluations.jsonl').read_text().splitlines()]
m=[json.loads(l) for l in (RUN/'metrics.jsonl').read_text().splitlines()]
x=np.array([r['update']*16384/1e6 for r in e]);tx=np.array([r['step']/1e6 for r in m])
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for role,label,color in [('0_low','Start at 70% HP','#d97706'),('0_full','Start at 100% HP','#2563eb')]:
 axes[0,0].plot(x,[r['summaries'][role]['cs'] for r in e],'-o',label=label,color=color)
 axes[0,1].plot(x,[r['summaries'][role]['deaths'] for r in e],'-o',label=label,color=color)
cs=[np.mean([z['cs'] for z in r['episodes'] if z['team']==0]) for r in e]
axes[0,0].plot(x,cs,'--',color='black',alpha=.5,label='Combined mean')
for xx,yy in zip(x,cs):axes[0,0].annotate(f'{yy:.2f}',(xx,yy),xytext=(0,-17),textcoords='offset points',ha='center',fontsize=9)
axes[0,0].set(title='Frozen CS per 120-second game',ylabel='Mean CS',ylim=(0,12));axes[0,0].legend(fontsize=9)
axes[0,1].set(title='Frozen deaths per 120-second game',ylabel='Mean deaths',ylim=(-.04,1.1))
lr=3e-5*(1-(np.array([r['update'] for r in m])-1)/612)
axes[1,0].plot(tx,lr*1e5,color='#7c3aed');axes[1,0].set(title='Configured learning rate (linear decay)',ylabel='Learning rate × 100,000')
for key,label,color in [('reward_cs','Gold','#2563eb'),('reward_xp','XP','#059669'),('reward_tower','Tower HP loss (all sources)','#d97706'),('reward_health','Health penalty','#dc2626')]:
 y=np.array([r.get(key,0.) for r in m]);axes[1,1].plot(tx[24:],np.convolve(y,np.ones(25)/25,mode='valid'),label=label,color=color)
axes[1,1].set(title='TRAIN reward components (25-update mean)',ylabel='Reward per learner decision');axes[1,1].legend(fontsize=8)
for a in axes.flat:a.set_xlabel('Additional learner decisions (millions)');a.grid(alpha=.2)
fig.suptitle('E46: AFK farming from trained E40 checkpoint\nFrozen: 64 games/checkpoint, 32 per HP role · one training seed · 5 evaluation points',fontsize=14)
OUT.parent.mkdir(exist_ok=True);fig.savefig(OUT,dpi=160);print(OUT)
for lo,hi in [(1,64),(65,306),(307,612)]:
 rows=[r for r in m if lo<=r['update']<=hi]
 print(lo,hi,{k:float(np.mean([r[k] for r in rows])) for k in ['approx_kl','clip_frac','post_kl','reward_cs','reward_xp','reward_tower','reward_health']})
