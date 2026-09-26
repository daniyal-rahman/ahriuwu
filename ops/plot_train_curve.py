"""Training curve for a server_train run root (episode CS, rolling mean, frozen evals).
    .venv-jax/bin/python ops/plot_train_curve.py lanerl_jax/runs/server_train/mirror-wave-s0 out.png [extra_root...]"""
import json, glob, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
COLORS = ['#2a78d6', '#d9822b', '#3a9d5d', '#8b5cf6']
def load(root):
    """Segments are run dirs in time order; a later segment that resumed from
    update R supersedes the earlier one past R (a crash/collapse rerun)."""
    segs = []
    for d in sorted(glob.glob(root + '/*-s[0-9]*/')):
        eps, ups = [], []
        for l in open(d + 'metrics.jsonl'):
            r = json.loads(l)
            (ups if 'entropy' in r else eps if 'episode' in r and 'cs' in r else []).append(r)
        if ups: segs.append((eps, ups))
    eps, ups = [], []
    for i, (e, u) in enumerate(segs):
        cut = segs[i + 1][1][0]['update'] if i + 1 < len(segs) else float('inf')
        eps += [r for r in e if r['update'] < cut]; ups += [r for r in u if r['update'] < cut]
    per = {u['update']: u['steps'] // u['update'] for u in ups if u['update']}
    return eps, ups, (list(per.values()) or [1])[-1]
roots = [sys.argv[1]] + sys.argv[3:]
fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
for i, root in enumerate(roots):
    eps, ups, per = load(root)
    if not eps: continue
    x = np.array([e['update'] * per for e in eps]) / 1e6; y = np.array([e['cs'] for e in eps])
    ax.scatter(x, y, s=6, color=COLORS[i], alpha=0.25, linewidths=0)
    if len(y) >= 24:
        k = 24; ax.plot(x[k-1:], np.convolve(y, np.ones(k)/k, mode='valid'), color=COLORS[i], lw=2, label=f'{root.split("/")[-2] if root.endswith("seed0") else root.split("/")[-1]} (24-episode mean)')
try:
    evals = [json.loads(l) for l in open('lanerl_jax/runs/EVAL/summary.jsonl')]
    key = '/'.join(roots[0].rstrip('/').split('/')[-2:])
    evals = [v for v in evals if v.get('summary') and v.get('run', '').rstrip('/').endswith(key)]
    invalid = {920}   # E06 u920: OPS-004
    evals = [v for v in evals if v['update'] not in invalid]
    _, _, per0 = load(roots[0])
    ex = [v['update'] * per0 / 1e6 for v in evals]
    ax.scatter(ex, [v['summary']['0']['mean_cs'] for v in evals], marker='D', s=60, color='#1a1a19', zorder=5, label='frozen eval, blue mean (4-5 ep)')
    ax.scatter(ex, [v['summary']['1']['mean_cs'] for v in evals], marker='s', s=60, facecolors='none', edgecolors='#1a1a19', zorder=5, label='frozen eval, red mean (4-5 ep)')
except FileNotFoundError:
    pass
ax.axhline(30, color='#c3c2b7', lw=1, ls='--'); ax.text(0.01, 30.6, '30-CS gate', color='#555', fontsize=9)
ax.axhline(8, color='#c3c2b7', lw=1, ls=':'); ax.text(0.01, 8.6, 'untrained policy, median 8', color='#555', fontsize=9)
ax.set_xlabel('agent decisions (millions)'); ax.set_ylabel('CS per 600 s episode (train, per agent)')
ax.set_title(roots[0].split('/')[-2] + ': mirror self-play on the C# server (near-wave start, 10 Hz)')
ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', color='#eee'); ax.legend(loc='upper left', fontsize=8, frameon=False); ax.set_ylim(0, 40)
plt.tight_layout(); plt.savefig(sys.argv[2]); print('saved', sys.argv[2])
