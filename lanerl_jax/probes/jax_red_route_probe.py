"""Where does a JAX red champion stall on a Move toward the top lane?"""
import sys, numpy as np, jax, jax.numpy as jnp
from pathlib import Path
from lanerl_jax.train.jax_farm import JaxFarmCollector
from lanerl_jax.sim.orders import OrderKind, Orders
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
c = JaxFarmCollector(1, out, 600., False, 6, seed=0, teams=(0, 1))
def move(tx, ty):
    return Orders(jnp.asarray([[OrderKind.NOOP, OrderKind.MOVE]], jnp.int8), jnp.asarray([[0., tx]], jnp.float32),
                  jnp.asarray([[0., ty]], jnp.float32), jnp.full((1, 2), -1, jnp.int8), clear_target=jnp.zeros((1, 2), bool))
for label, (tx, ty) in [('leg0', (11000., 13600.)), ('direct', (2431., 12741.)), ('short', (13000., 13800.)), ('mid', (12500., 13300.))]:
    c.states = jax.tree.map(lambda a, b: a.at[0].set(b), c.states, c._fresh(0))
    c._initialize(np.ones(1, bool))
    c.states = c._step_states(c.states, move(tx, ty), np.ones(1, bool))
    print(label, 'target', (tx, ty), 'start', float(c.states.x[0, 1]), float(c.states.y[0, 1]), flush=True)
    for k in range(400):
        c.states = c._step_states(c.states, c._noop_orders, np.ones(1, bool))
        if k % 50 == 49:
            s = c.states
            extra = {f: np.asarray(getattr(s, f)[0, 1]).tolist() for f in ('route_status',) if hasattr(s, f)}
            print(f'  t={float(s.t_ms[0]):.0f} red=({float(s.x[0,1]):.0f},{float(s.y[0,1]):.0f}) {extra}', flush=True)
