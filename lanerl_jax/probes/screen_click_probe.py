"""Does the screen-click interface land where we think, on the live C# server?

For the blue champion at the near-wave start: (1) how many world units one
click cell covers at screen centre; (2) invert the camera projection to find
the click cell whose ground point lies on a chosen visible enemy minion, send
`attack_move` and `move` clicks there, and read back the server's target
(`tgt`) and the champion's movement; (3) click ground points at known lane
offsets and check the champion walks to the projected point.
"""
import json, sys, time, math
from pathlib import Path
import numpy as np
from lanerl_jax.train.server_train import ServerCollector, screen_order, WAVE_START_MS
from lanerl_rl.projection import (world_to_screen, centred_on, screen_to_world_centred,
                                  units_per_pixel_at_centre, DEFAULT_RESOLUTION)
from lanerl_rl.constants import BUTTONS, SCREEN_X_VALUES, SCREEN_Y_VALUES, N_SCREEN_X, N_SCREEN_Y

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
SD = Path('/mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0')
c = ServerCollector(1, out, 50400, 600., True, 6, SD, teams=(0,))
frame = c.frames[0]; ax, nm = np.asarray(frame.axis, float), np.asarray(frame.normal, float)
cam = centred_on(0., 0.)
upx, upz = units_per_pixel_at_centre(cam)
cell_w = upx * DEFAULT_RESOLUTION[0] / N_SCREEN_X; cell_h = upz * DEFAULT_RESOLUTION[1] / N_SCREEN_Y
print(json.dumps({'units_per_pixel_centre': [upx, upz], 'cell_units_at_centre': [cell_w, cell_h],
                  'grid': [N_SCREEN_X, N_SCREEN_Y]}), flush=True)

def inverse_click(champ, wx, wy):
    """World point -> (ix, iy) grid cell, via the lane frame and the camera."""
    dx, dy = wx - champ['x'], wy - champ['y']
    ds, dn = dx * ax[0] + dy * ax[1], dx * nm[0] + dy * nm[1]
    sx, sy = world_to_screen(cam, ds, dn)
    ix = int(np.clip(round(sx * N_SCREEN_X - 0.5), 0, N_SCREEN_X - 1))
    iy = int(np.clip(round(sy * N_SCREEN_Y - 0.5), 0, N_SCREEN_Y - 1))
    return ix, iy, (sx, sy)

def frame_units(): return c.env.last_obs[0]['u']
def me(): return c.champion(0, 0)
def step(order):
    c.env.step([{'blue': order, 'red': {'t': 'noop'}}]); c._rank()
def enemy_minions():
    return [u for u in frame_units() if u.get('k') == 'LaneMinion' and u['tm'] == 200 and u['hp'] > 0 and u.get('vb')]

results = []
# wait for the wave to arrive (up to 15 game-s)
for _ in range(150):
    if len(enemy_minions()) >= 3: break
    step({'t': 'noop'})
ch = me(); mins = sorted(enemy_minions(), key=lambda u: math.hypot(u['x']-ch['x'], u['y']-ch['y']))
print('champion', ch['x'], ch['y'], 'visible enemy minions', [(u['id'], round(u['x']), round(u['y'])) for u in mins[:5]], flush=True)

# (2) click ON each of the 3 nearest minions with attack_move and with move
for u in mins[:3]:
    for button in ('attack_move', 'move'):
        ch = me(); ix, iy, s = inverse_click(ch, u['x'], u['y'])
        order = screen_order((BUTTONS.index(button), ix, iy), ch, frame)
        err = math.hypot(order['x'] - u['x'], order['y'] - u['y'])
        step(order)
        tg = [me().get('tgt') for _ in range(1)]
        for _ in range(5): step({'t': 'noop'}); tg.append(me().get('tgt'))
        rec = {'button': button, 'minion': u['id'], 'minion_xy': [u['x'], u['y']], 'cell': [ix, iy], 'screen': s,
               'click_world': [order['x'], order['y']], 'click_err_units': err, 'tgt_after': tg,
               'hit': u['id'] in tg}
        results.append(rec); print(json.dumps(rec), flush=True)
        step({'t': 'noop'})
# (3) ground clicks at known lane offsets: walk and check arrival
for ds, dn in [(300., 0.), (-300., 0.), (0., 200.), (0., -200.), (500., 300.)]:
    ch = me(); wx, wy = ch['x'] + ds*ax[0] + dn*nm[0], ch['y'] + ds*ax[1] + dn*nm[1]
    ix, iy, s = inverse_click(ch, wx, wy)
    order = screen_order((BUTTONS.index('move'), ix, iy), ch, frame)
    step(order); start = (me()['x'], me()['y'])
    for _ in range(40): step({'t': 'noop'})
    end = me()
    rec = {'ground': [ds, dn], 'intended': [wx, wy], 'click_world': [order['x'], order['y']],
           'quantisation_err': math.hypot(order['x']-wx, order['y']-wy),
           'arrival_err': math.hypot(end['x']-order['x'], end['y']-order['y']), 'moved': math.hypot(end['x']-start[0], end['y']-start[1])}
    results.append(rec); print(json.dumps(rec), flush=True)
(out / 'results.json').write_text(json.dumps(results, indent=1))
c.close()
