import torch, json, numpy as np, sys
OUT = {}

def load(p):
    return torch.load(p, map_location='cpu', weights_only=False)

d = load('/srv/nfs/projects/ahriuwu/data/cache_baseline.pt')
md = d['match_data']
OUT['meta_baseline'] = {k: (v if not isinstance(v, dict) else v) for k, v in d['meta'].items() if k != 'reward_config'}
OUT['n_matches'] = len(md)

rep = 0; tot = 0
runs = []            # hold-run lengths (frames a held value persists)
ev_tot = 0; frames_tot = 0
sent_tot = 0         # pre-first-click sentinel frames
prefirst_lens = []
steps = []
per_game_rep = []
for name, m in md.items():
    mv = m['movement'].float().numpy()
    ev = m['movement_event'].numpy().astype(bool)
    T = mv.shape[0]
    frames_tot += T
    ev_tot += int(ev.sum())
    same = np.all(mv[1:] == mv[:-1], axis=1)
    rep += int(same.sum()); tot += int(same.size)
    per_game_rep.append(float(same.mean()))
    steps.append(np.abs(mv[1:] - mv[:-1]).max(axis=1))
    # hold-run lengths: number of consecutive identical rows
    chg = np.flatnonzero(~same) + 1
    bounds = np.concatenate(([0], chg, [T]))
    runs.append(np.diff(bounds))
    # pre-first-click sentinel window
    first = np.flatnonzero(ev)
    f0 = int(first[0]) if first.size else T
    prefirst_lens.append(f0)
    sent_tot += f0

runs = np.concatenate(runs)
steps = np.concatenate(steps)
OUT['movement'] = dict(
    exact_repeat_frac=rep / tot, n_pairs=int(tot), frames=int(frames_tot),
    event_frac=ev_tot / frames_tot, n_events=int(ev_tot),
    per_game_repeat_mean=float(np.mean(per_game_rep)),
    per_game_repeat_min=float(np.min(per_game_rep)),
    per_game_repeat_max=float(np.max(per_game_rep)),
    step_median=float(np.median(steps)), step_p90=float(np.percentile(steps, 90)),
    run_median=float(np.median(runs)), run_mean=float(runs.mean()),
    run_p90=float(np.percentile(runs, 90)), run_p99=float(np.percentile(runs, 99)),
    run_max=int(runs.max()), n_runs=int(runs.size),
)
# hold-run histogram (capped)
h, edges = np.histogram(np.clip(runs, 1, 200), bins=np.arange(1, 202))
OUT['run_hist'] = dict(counts=h.tolist(), edges=edges[:-1].tolist())
OUT['run_lengths_sample'] = runs[:0].tolist()  # placeholder
np.save('/srv/nfs/projects/ahriuwu/reports/figdata/hold_runs.npy', runs.astype(np.int32))

# per-frame dropout survival: P(shortcut still available) = 1 - E_j[ p^(j+1) ]
# For a frame at position i within a run of length L, j = i (frames since last click).
# The crutch is hidden only if ALL j+1 replicas are dropped.
ps = np.linspace(0.0, 1.0, 101)
# build the empirical distribution of j over ALL frames
js = np.concatenate([np.arange(L) for L in runs])
surv = []
for p in ps:
    surv.append(float(1.0 - np.mean(p ** (js + 1))))
OUT['dropout_curve'] = dict(p=ps.tolist(), available=surv)
OUT['dropout_at'] = {str(round(p,2)): float(1.0 - np.mean(p ** (js + 1))) for p in (0.15, 0.5, 0.95, 0.99)}
OUT['prefirst'] = dict(
    total_frames=int(sent_tot), frac=sent_tot / frames_tot,
    median_frames=float(np.median(prefirst_lens)),
    median_seconds=float(np.median(prefirst_lens)) / 20.0,
    min_frames=int(np.min(prefirst_lens)), max_frames=int(np.max(prefirst_lens)),
    n_games=len(prefirst_lens),
)
np.save('/srv/nfs/projects/ahriuwu/reports/figdata/prefirst_lens.npy', np.array(prefirst_lens))

# reward decomposition
rew = np.concatenate([md[k]['rewards'].float().numpy() for k in md])
gold = rew / 1e-3
bands = [('zero', -1e-9, 1e-9), ('passive <2.1g', 1e-9, 2.1), ('2.1-10g', 2.1, 10),
         ('10-25g melee', 10, 25), ('25-45g caster', 25, 45), ('45-100g cannon/plate', 45, 100),
         ('>100g kill/turret', 100, 1e9)]
dec = []
pos = gold[gold > 0].sum()
for nm, lo, hi in bands:
    sel = (gold > lo) & (gold <= hi) if lo > -1e-9 else (np.abs(gold) < 1e-9)
    dec.append(dict(band=nm, n=int(sel.sum()), frac_frames=float(sel.mean()),
                    share_pos_reward=float(gold[sel].sum() / pos) if pos else 0.0))
neg = gold < -1e-9
dec.append(dict(band='deaths (negative)', n=int(neg.sum()), frac_frames=float(neg.mean()),
                share_pos_reward=float(gold[neg].sum())))
OUT['reward_decomp'] = dec
OUT['reward_n'] = int(rew.size)

# state-label coverage: which of the 4 aux targets are actually present
sm = np.concatenate([md[k]['state_mask'].float().numpy() for k in md], axis=0)
OUT['state_mask_coverage'] = sm.mean(axis=0).tolist()

# abilities press rates
ab = {}
for name, m in md.items():
    for k, v in m['abilities'].items():
        ab.setdefault(k, []).append(v.float().numpy())
OUT['ability_press_rate'] = {k: float(np.concatenate(v).mean()) for k, v in ab.items()}

json.dump(OUT, open('/srv/nfs/projects/ahriuwu/reports/figdata/corpus_stats.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in OUT.items() if k not in ('run_hist', 'dropout_curve', 'run_lengths_sample')}, indent=1)[:4000])
