#!/usr/bin/env python3
"""Progress of a lanerl_jax.train.server_train run: learner stats, episode CS.

    ops/server_train_status.py lanerl_jax/runs/server_train/mirror-s0 [--last N]
"""
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); last = int(sys.argv[sys.argv.index('--last') + 1]) if '--last' in sys.argv else 5
run = sorted(p for p in root.glob('server-farm-s*') if p.is_dir())[-1]
rows = [json.loads(l) for l in (run / 'metrics.jsonl').open()]
ups = [r for r in rows if 'update' in r and 'entropy' in r]
eps = [r for r in rows if 'episode' in r and 'cs' in r and 'entropy' not in r]
print(run.name, f'{len(ups)} updates, {len(eps)} completed episodes')
for r in ups[-last:]:
    print(f"u{r['update']:5d} steps {r['steps']:8d} wall {r['wall_s']:7.0f}s ent {r['entropy']:.3f} kl {r['approx_kl']:.1e} "
          f"clip {r['clip_frac']:.3f} vl {r['value_loss']:.3f} r {r['mean_reward']:.4f} "
          f"btn {r['sampled_buttons']}")
if len(ups) > 1:
    a, b = ups[max(0, len(ups) - 20)], ups[-1]
    print(f"recent {(b['wall_s']-a['wall_s'])/(b['update']-a['update']):.2f} s/update, "
          f"{(b['steps']-a['steps'])/(b['wall_s']-a['wall_s']):.0f} decisions/s")
if eps:
    import statistics as st
    by = {}
    for e in eps:
        by.setdefault(e.get('team', 0), []).append(e)
    for team, es in sorted(by.items()):
        cs = [e['cs'] for e in es]
        chunks = [cs[i:i + 12] for i in range(0, len(cs), 12)]
        print(f"team {team}: {len(cs)} episodes; CS by 12-episode chunk (mean):",
              [round(st.mean(c), 1) for c in chunks][-15:], 'last', cs[-12:])
