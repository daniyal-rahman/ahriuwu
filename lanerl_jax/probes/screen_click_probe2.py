"""Right-click on a hostile: does the server set a target? Raw clicks at the
minion's exact position (no grid), for blue and for red, recording tgt and mo."""
import json, sys, math
from pathlib import Path
import numpy as np
from lanerl_jax.train.server_train import ServerCollector, screen_order, TEAM_KEY, TEAM_WIRE
from lanerl_rl.projection import world_to_screen, centred_on
from lanerl_rl.constants import BUTTONS, N_SCREEN_X, N_SCREEN_Y
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
SD = Path('/mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0')
c = ServerCollector(1, out, 50500, 600., True, 6, SD, teams=(0, 1))
cam = centred_on(0., 0.)
def step(blue, red): c.env.step([{'blue': blue, 'red': red}]); c._rank()
def units(): return c.env.last_obs[0]['u']
def enemies(team): return [u for u in units() if u.get('k') == 'LaneMinion' and u['tm'] != TEAM_WIRE[team] and u['hp'] > 0 and u.get('vb' if team == 0 else 'vr')]
def inverse(team, ch, wx, wy):
    f = c.frames[team]; ax, nm = np.asarray(f.axis, float), np.asarray(f.normal, float)
    dx, dy = wx - ch['x'], wy - ch['y']
    sx, sy = world_to_screen(cam, dx*ax[0]+dy*ax[1], dx*nm[0]+dy*nm[1])
    return int(np.clip(round(sx*N_SCREEN_X-0.5), 0, N_SCREEN_X-1)), int(np.clip(round(sy*N_SCREEN_Y-0.5), 0, N_SCREEN_Y-1))
for _ in range(200):
    if len(enemies(0)) >= 3 and len(enemies(1)) >= 3: break
    step({'t': 'noop'}, {'t': 'noop'})
res = []
for team in (0, 1):
    ch = c.champion(0, team); ms = sorted(enemies(team), key=lambda u: math.hypot(u['x']-ch['x'], u['y']-ch['y']))[:2]
    for u in ms:
        for mode in ('raw-move', 'raw-attack_move', 'grid-move', 'grid-attack_move', 'raw-attack-id'):
            ch = c.champion(0, team)
            u2 = next((v for v in units() if v['id'] == u['id']), None)
            if u2 is None: continue
            if mode == 'raw-attack-id': order = {'t': 'attack', 'id': u2['id']}
            elif mode.startswith('raw'): order = {'t': 'click', 'button': mode.split('-')[1], 'x': float(u2['x']), 'y': float(u2['y'])}
            else:
                ix, iy = inverse(team, ch, u2['x'], u2['y']); order = screen_order((BUTTONS.index(mode.split('-')[1]), ix, iy), ch, c.frames[team])
            other = {'t': 'noop'}
            step(order if team == 0 else other, order if team == 1 else other)
            after = [(c.champion(0, team).get('tgt'), c.champion(0, team).get('mo'))]
            for _ in range(4): step({'t': 'noop'}, {'t': 'noop'}); after.append((c.champion(0, team).get('tgt'), c.champion(0, team).get('mo')))
            r = {'team': team, 'mode': mode, 'minion': u2['id'], 'order': order, 'dist': round(math.hypot(u2['x']-ch['x'], u2['y']-ch['y'])), 'tgt_mo_after': after, 'hit': any(t == u2['id'] for t, _ in after)}
            res.append(r); print(json.dumps(r), flush=True)
            step({'t': 'noop'}, {'t': 'noop'})
(out/'results.json').write_text(json.dumps(res, indent=1)); c.close()
