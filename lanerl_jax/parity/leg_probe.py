"""Is the 545u detour on lane leg wp0->wp1 the ROUTE TABLE or the follower?

The sim walks 2875u on a leg whose straight line is 2329.8u and is fully
walkable (0 of 121 sample points blocked at pathfinding_radius 35); the server
walks it at -4.4u versus straight. `gate3_route_inputs(table_disabled=True)`
makes the driver use the raw `[position, destination]` path instead of the
local route artifact, which splits the two candidates cleanly:

  detour gone with the table off  -> the route table is wrong for this leg
  detour survives                 -> it is in movement/waypoint following
"""
import math, sys
from lanerl_jax.parity.last_hit_drive import run_oracle_in_sim, gate3_route_inputs
from lanerl_jax.sim.init import TOP_LANE_PATH

WP0, WP1 = tuple(map(float, TOP_LANE_PATH[0])), tuple(map(float, TOP_LANE_PATH[1]))
STRAIGHT = math.dist(WP0, WP1)

def measure(table_disabled: bool):
    pts = []
    def on_dec(ev):
        st = ev["state"]
        import numpy as np
        pts.append((float(np.asarray(st.x)[0]), float(np.asarray(st.y)[0])))
    kw = {}
    if table_disabled:
        rt, terr = gate3_route_inputs(table_disabled=True)
        kw = dict(route_table=rt, terrain=terr, table_disabled=True)
    run_oracle_in_sim(decisions=700, on_decision=on_dec, **kw)
    start = next((i for i, p in enumerate(pts) if math.dist(p, WP0) < 120), None)
    end = next((i for i, p in enumerate(pts) if start is not None and i > start
                and math.dist(p, WP1) < 120), None)
    if start is None or end is None:
        return None
    L = sum(math.dist(pts[i-1], pts[i]) for i in range(start+1, end+1))
    west = min(p[0] for p in pts[start:end+1])
    return end - start, L, L - STRAIGHT, west

print(f"LEG straight-line {STRAIGHT:.1f}u; server walks it at -4.4u")
for name, dis in (("route table ON ", False), ("route table OFF", True)):
    r = measure(dis)
    if r is None:
        print(f"RESULT {name}: never reached wp1 within 700 decisions")
        continue
    n, L, exc, west = r
    print(f"RESULT {name}: {n:>3} decisions, walked {L:8.1f}u, "
          f"excess {exc:+7.1f}u, westmost x {west:7.1f}")
