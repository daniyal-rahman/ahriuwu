"""Probe (one-off): does the JAX sim execute SHORT move clicks? Through the
production collector path (JaxFarmCollector, near-wave start, screen-click
actions): after control starts, click a point d units behind the champion
along the lane (d in 60, 120, 250, 500) and measure the displacement after
1 s and 3 s. The server executes a 120-u click (server trace: small moves
every second); the JAX champion stood still for 6 s while clicking ~120 u
behind (probes/click_vs_move.py)."""
import numpy as np, jax, jax.numpy as jnp
from lanerl_jax.train.jax_farm import JaxFarmCollector
from lanerl_jax.train.scripted_policy import cell_for_offset
from lanerl_jax.train.reward import LANE_AXIS
from lanerl_rl.constants import BUTTON_INDEX
a = {k: float(np.asarray(v)[0]) for k, v in LANE_AXIS.items()}
def s_of(x, y): return (x - a["origin_x"]) * a["axis_x"] + (y - a["origin_y"]) * a["axis_y"]
for d in (60., 120., 250., 500.):
    c = JaxFarmCollector(1, "/tmp/short_move_probe", episode_s=600., start_near_wave=True, step_ticks=6, seed=0, teams=(0,))
    obs, _ = c.observe()
    p0 = c.positions()[0]; s0 = s_of(*p0)
    sx, sy = cell_for_offset(jnp.float32(-d), jnp.float32(0.))
    act = np.array([[BUTTON_INDEX["move"], int(sx), int(sy)]], np.int32)
    noop = np.array([[BUTTON_INDEX["noop"], 0, 0]], np.int32)
    for t in range(30):
        c.step(act if t == 0 else noop)          # ONE click, then no-ops: a fixed world goal
        if t == 9: p1 = c.positions()[0]
    p3 = c.positions()[0]
    print(f"click {d:4.0f} u behind (cell {int(sx)},{int(sy)}): moved after 1 s {s_of(*p1) - s0:7.1f} u, after 3 s {s_of(*p3) - s0:7.1f} u")
    c.close()
