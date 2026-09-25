"""screen-click-v3 check: a click past the map edge must move the champion to the
closest reachable point instead of walking it into the wall. Also checks a
right-click into a wall inside the map."""
import json, sys, math
from pathlib import Path
from lanerl_train.ports import PortAllocator
from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv
SD = Path(sys.argv[2]); out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
env = VecLaneEnv(1, spec=ServerLaunchSpec(bot_teams='none', step_ticks=6, toponly=True, server_dir=SD, extra_env={'LANERL_AUTOBUY': '0'}),
                 log_dir=out, ports=PortAllocator(base=23100).allocate(1), auto_restart=False)
env.start()
def champ(): return next(u for u in env.last_obs[0]['u'] if u.get('k') == 'Champion' and u['tm'] == 100)
def run(label, order, steps=120):
    env.step([{'blue': order, 'red': {'t': 'noop'}}]); start = (champ()['x'], champ()['y'])
    for _ in range(steps): env.step([{'blue': {'t': 'noop'}, 'red': {'t': 'noop'}}])
    c = champ(); print(json.dumps({'case': label, 'order': order, 'start': start, 'end': [c['x'], c['y']], 'moved': round(math.hypot(c['x']-start[0], c['y']-start[1]))}), flush=True)
# walk blue up to the lane corner first (legs)
for leg in [(1200., 6000.), (1100., 10500.), (1950., 12350.)]:
    run('setup', {'t': 'move', 'x': leg[0], 'y': leg[1]}, steps=300)
run('off-map above the top edge', {'t': 'click', 'button': 'move', 'x': 2500., 'y': 16000.})
run('off-map left of the map', {'t': 'click', 'button': 'move', 'x': -800., 'y': 12500.})
run('wall inside the map (top-left corner mass)', {'t': 'click', 'button': 'attack_move', 'x': 600., 'y': 14300.})
run('plain reachable point', {'t': 'click', 'button': 'move', 'x': 2600., 'y': 12900.})
env.close()
