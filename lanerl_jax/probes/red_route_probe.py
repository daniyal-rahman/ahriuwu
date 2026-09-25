"""Why does red stop at (12058,12979) on a move to the top-lane corner? Direct vs legged moves."""
import json, sys, time
from pathlib import Path
from lanerl_train.ports import PortAllocator
from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv
SD = Path('/mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0')
def champ(o, tm): return next(u for u in o['u'] if u.get('k') == 'Champion' and u['tm'] == tm)
def run(label, legs, port):
    env = VecLaneEnv(1, spec=ServerLaunchSpec(bot_teams='none', step_ticks=6, toponly=True, server_dir=SD, extra_env={'LANERL_AUTOBUY': '0'}),
                     log_dir=Path(sys.argv[1]) / label, ports=PortAllocator(base=port).allocate(1), auto_restart=False)
    env.start(); o = env.last_obs[0]; leg = 0; last = None; t_last = 0
    print(label, 'start', champ(o, 200)['x'], champ(o, 200)['y'], flush=True)
    for step in range(1500):   # 150 game-s at 10 Hz
        r = champ(o, 200); pos = (r['x'], r['y'])
        cmd = {'t': 'noop'}
        if leg < len(legs) and (step == 0 or (pos == last and step - t_last > 10)):
            if step > 0: leg += 1
            if leg < len(legs):
                cmd = {'t': 'move', 'x': legs[leg][0], 'y': legs[leg][1]}; t_last = step
                print(f'  t={o["t"]} at {pos} -> leg {leg} {legs[leg]}', flush=True)
        last = pos
        env.step([{'blue': {'t': 'noop'}, 'red': cmd}]); o = env.last_obs[0]
        if step % 100 == 0: print(f'  t={o["t"]} red={pos}', flush=True)
    print(label, 'end', champ(o, 200)['x'], champ(o, 200)['y'], flush=True)
    env.close()
run('direct', [(2431., 12741.)], 50300)
run('legs', [(11000., 13600.), (7500., 13700.), (4500., 13600.), (2431., 12741.)], 50310)
