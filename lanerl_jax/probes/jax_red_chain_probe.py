import sys, numpy as np, jax, jax.numpy as jnp
from pathlib import Path
from lanerl_jax.train.jax_farm import JaxFarmCollector
from lanerl_jax.sim.orders import OrderKind, Orders
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
c = JaxFarmCollector(1, out, 600., False, 6, seed=0, teams=(0, 1))
def move(tx, ty):
    return Orders(jnp.asarray([[OrderKind.NOOP, OrderKind.MOVE]], jnp.int8), jnp.asarray([[0., tx]], jnp.float32),
                  jnp.asarray([[0., ty]], jnp.float32), jnp.full((1, 2), -1, jnp.int8), clear_target=jnp.zeros((1, 2), bool))
chains = {'A': [(12500., 13300.), (11500., 13700.), (11000., 13600.), (7500., 13700.), (4500., 13600.), (2431., 12741.)],
          'B': [(12500., 13300.), (11800., 13850.), (9500., 13850.), (7000., 13800.), (4500., 13600.), (2431., 12741.)]}
for label, legs in chains.items():
    c.states = jax.tree.map(lambda a, b: a.at[0].set(b), c.states, c._fresh(0)); c._initialize(np.ones(1, bool))
    leg = 0; c.states = c._step_states(c.states, move(*legs[0]), np.ones(1, bool)); last = None; still = 0
    for k in range(1200):
        c.states = c._step_states(c.states, c._noop_orders, np.ones(1, bool))
        rx, ry = float(c.states.x[0, 1]), float(c.states.y[0, 1])
        still = still + 1 if (rx, ry) == last else 0; last = (rx, ry)
        if leg + 1 < len(legs) and (np.hypot(rx - legs[leg][0], ry - legs[leg][1]) <= 150. or still >= 20):
            print(f'  {label} t={float(c.states.t_ms[0]):.0f} at ({rx:.0f},{ry:.0f}) still={still} -> leg {leg+1} {legs[leg+1]}', flush=True)
            leg += 1; still = 0
            c.states = c._step_states(c.states, move(*legs[leg]), np.ones(1, bool))
        if k % 100 == 99: print(f'  {label} t={float(c.states.t_ms[0]):.0f} red=({rx:.0f},{ry:.0f})', flush=True)
    print(label, 'end', last, 'dist to goal', np.hypot(last[0]-2431, last[1]-12741), flush=True)
