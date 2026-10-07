"""Replay one tick's route steering of a mover natively and in JAX (``map.pathing``) on the same inputs.

    python -m ops.native.debug_route --tick 13289 --unit 6
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "native" / "python"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick", type=int, required=True)
    ap.add_argument("--unit", type=int, nargs="+", required=True)
    args = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp

    from lanerl_jax.modern import world as MS
    from lanerl_jax.modern.jax_cache import enable_compile_cache
    from lanerl_jax.modern.map import pathing as P
    from lanerl_jax.modern.mechanics import team_terrain
    from ops.modern.golden import build
    import lanesim as LS
    enable_compile_cache()
    cfg = build("top")
    world = LS.NativeWorld(cfg)
    step = jax.jit(lambda s: MS.step(s, MS.no_orders(), cfg)[0])
    run = jax.jit(lambda s: jax.lax.fori_loop(0, 600, lambda i, s: MS.step(s, MS.no_orders(), cfg)[0], s))
    s = MS.init_state(cfg)
    t = 0
    while t + 600 <= args.tick:
        s, t = run(s), t + 600
    while t < args.tick:
        s, t = step(s), t + 1
    s1 = step(s)
    env = LS.env_from_state(world, s)
    m = cfg.layout.ward0
    dbg = np.full((m, 24), np.nan, np.float32)
    LS.lib().ls_debug_route(dbg.ctypes.data)
    world.step(env)
    LS.lib().ls_debug_route(None)
    # Collision replay in JAX on the native move-step output.
    from lanerl_jax.modern import collision as UC
    from lanerl_jax.modern.core import types as W
    from lanerl_jax.modern.lane import ai as LA
    from lanerl_jax.modern.world import units as U
    kind, alive = np.asarray(s.kind)[:m], np.asarray(s.alive)[:m]
    now = jnp.float32(float(s.t) + cfg.dt)
    ghost = np.asarray(LA.minion_ghosted(s.lane_ai, U.units_view(s), now))[:m]
    collide = alive & (kind != W.KIND_WARD) & ~np.asarray(W.is_structure(kind))
    solid = collide & ~ghost
    team = np.clip(np.asarray(s.team)[:m], 0, 1)
    rad = np.asarray(UC.pathing_radius(kind, np.asarray(s.sub)[:m], np.asarray(s.radius)[:m]))
    clr = np.minimum(np.asarray(s.radius)[:m], cfg.routes.radius).astype(np.float32)
    x0, y0 = np.asarray(s.x)[:m], np.asarray(s.y)[:m]
    x1, y1, act = dbg[:, 17], dbg[:, 18], dbg[:, 6] > 0
    gx_, gy_ = dbg[:, 2], dbg[:, 3]
    ax, ay = UC.avoid(*(jnp.asarray(v) for v in (x0, y0, x1, y1, rad)), jnp.asarray(solid), jnp.asarray(solid & act),
                      jnp.asarray(gx_), jnp.asarray(gy_), jnp.asarray(team), jnp.asarray(clr), cfg.terrain, cfg.dt)
    fx, fy = UC.separate(ax, ay, jnp.asarray(rad), jnp.asarray(solid), jnp.asarray(act), jnp.asarray(team),
                         jnp.asarray(clr), cfg.terrain)
    print(json.dumps({"solid_native_vs_jax": int(np.sum((dbg[:, 23] > 0) != solid)),
                      "avoid_max_diff": float(np.max(np.abs(np.asarray(ax) - dbg[:, 19]))),
                      "avoid_units_diff": np.flatnonzero(np.asarray(ax) != dbg[:, 19]).tolist(),
                      "sep_units_diff": np.flatnonzero(np.asarray(fx) != dbg[:, 21]).tolist()}))
    # The real JAX tick's collision inputs: one eager step with resolve / move_step spied.
    import lanerl_jax.modern.collision as UCm
    import lanerl_jax.modern.mechanics as Mm
    cap = {}
    r0, m0 = UCm.resolve, Mm.move_step

    def spy_resolve(x0_, y0_, x1_, y1_, **kw):
        cap.update(x1=np.asarray(x1_), y1=np.asarray(y1_), **{k: np.asarray(v) for k, v in kw.items()
                                                             if k in ("collide", "ghosted", "moving", "goal_x", "goal_y")})
        return r0(x0_, y0_, x1_, y1_, **kw)

    def spy_move(*a, **kw):
        out = m0(*a, **kw)
        cap.update(mv_gx=np.asarray(a[2]), mv_gy=np.asarray(a[3]), mv_speed=np.asarray(a[4]), mv_active=np.asarray(a[5]))
        return out
    UCm.resolve, Mm.move_step = spy_resolve, spy_move
    with jax.disable_jit():
        MS.step(s, MS.no_orders(), cfg)
    UCm.resolve, Mm.move_step = r0, m0
    for name, nat in (("x1", dbg[:, 17]), ("y1", dbg[:, 18]), ("goal_x", dbg[:, 2]), ("goal_y", dbg[:, 3]),
                      ("moving", dbg[:, 6] > 0), ("mv_speed", dbg[:, 16])):
        got = cap[name][:m]
        bad = np.flatnonzero(got != nat)
        print(json.dumps({"field": name, "units": bad.tolist()[:10],
                          "jax": got[bad][:5].tolist(), "native": nat[bad][:5].tolist()}))
    for u in args.unit:
        print(json.dumps({"unit": u, "move": dbg[u, 16:19].tolist(), "native_avoid": dbg[u, 19:21].tolist(),
                          "jax_avoid": [float(ax[u]), float(ay[u])], "native_final": dbg[u, 21:23].tolist(),
                          "jax_final_on_native_inputs": [float(fx[u]), float(fy[u])]}))
        x, y, gx, gy, rr, anchor = (float(v) for v in dbg[u, :6])
        tm = int(np.clip(np.asarray(s.team)[u], 0, 1))
        ter = team_terrain(cfg.terrain, jnp.int32(tm))
        pos, goal = jnp.asarray([x, y], jnp.float32), jnp.asarray([gx, gy], jnp.float32)
        f = P.route_follow(pos, goal, jnp.float32(rr), jnp.int32(int(anchor)), cfg.routes, ter)
        r = P.route_replan(pos, goal, jnp.float32(rr), cfg.routes, ter)
        print(json.dumps({
            "unit": u, "inputs": dbg[u, :7].tolist(),
            "native_follow": dbg[u, 7:12].tolist(), "jax_follow": [*np.asarray(f[0]).tolist(), bool(f[1]), int(f[2]), bool(f[3])],
            "native_replan": dbg[u, 12:16].tolist(), "jax_replan": [*np.asarray(r[0]).tolist(), bool(r[1]), int(r[2])],
            "native_next": [float(env["x"][u]), float(env["y"][u])],
            "jax_next": [float(np.asarray(s1.x)[u]), float(np.asarray(s1.y)[u])],
            "jax_anchor_next": int(np.asarray(s1.route_anchor)[u]), "native_anchor_next": int(env["route_anchor"][u])}))


if __name__ == "__main__":
    main()
