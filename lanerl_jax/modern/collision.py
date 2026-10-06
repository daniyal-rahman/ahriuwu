"""Unit-vs-unit collision (docs/modern/COLLISION.md), two vectorized O(N^2) phases per tick.

Live champions, lane minions and monsters collide with each other through their pathing radius; ghosted units,
wards and structures do not (structures block through navgrid pads).

1. Avoidance: each mover's route step is retried at headings rotated by ``AVOID_ANGLES_DEG`` (away from the nearest
   blocker first), keeping per side the smallest clear turn else the latest first contact over ``AVOID_HORIZON_S``;
   the better walkable pick wins. A pick that still meets an obstacle stops at contact (>= ``CONTACT_MIN_FRAC``).
   Obstacles on or past the mover's goal are ignored.
2. Separation: ``SEPARATION_ITERS`` Jacobi rounds push overlapping pairs apart by mobility (stationary units
   ``STATIONARY_MOBILITY``), at most ``MAX_PUSH`` per round; a push into terrain is refused and pins that unit.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .core import types as W
from .data import PATCH_DIR
from .jungle.camps import CHARACTERS
from .map.terrain import is_walkable, team_view

CHAMPION_PATHING_RADIUS = 35.0                               # client pathfindingCollisionRadius (Garen, Jax)
MINION_PATHING_RADIUS = (35.7437, 35.7437, 55.7437, 55.5208)  # melee, caster, siege, super (MINIONS §1.1)
_JUNGLE = PATCH_DIR / "jungle_client.json"

# Tuning, INFERRED (COLLISION §4).
AVOID_ANGLES_DEG = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)     # heading offsets, both sides
AVOID_HORIZON_S = 0.3                                        # = client ai_PostAvoidanceFilterDuration
AVOID_MAX_STEP = 60.0                                        # longer steps are blinks: no avoidance
CONTACT_MIN_FRAC = 0.25                                      # contact stop never below this share (no deadlock)
SEPARATION_ITERS = 3
STATIONARY_MOBILITY = 0.25                                   # share of a push a non-mover takes vs a mover
MAX_PUSH = 20.0                                              # per round: deep overlaps take several ticks
_GOLDEN = 2.399963229728653                                  # coincident-pair fallback direction step (rad)


@lru_cache(maxsize=1)
def monster_pathing_radii() -> tuple:
    """Client pathing radius per jungle ``Monster`` type."""
    data = json.loads(_JUNGLE.read_text())["monsters"]
    return tuple(float(data[c]["pathing_radius"]) for c in CHARACTERS)


def pathing_radius(kind: Any, sub: Any, gameplay_radius: Any) -> Any:
    """(N,) collision radius: champions 35, minions by type, camp monsters by record; epic monsters and others keep
    their gameplay radius (Baron's client pathing radius is 0, yet it blocks; INFERRED L)."""
    kind, sub = jnp.asarray(kind), jnp.asarray(sub, jnp.int32)
    r = jnp.asarray(gameplay_radius, jnp.float32)
    mon = jnp.asarray(monster_pathing_radii(), jnp.float32)
    mtab = jnp.asarray(MINION_PATHING_RADIUS, jnp.float32)
    is_camp = (kind == W.KIND_MONSTER) & (sub >= 0) & (sub < mon.shape[0])
    r = jnp.where(is_camp, mon[jnp.clip(sub, 0, mon.shape[0] - 1)], r)
    r = jnp.where(kind == W.KIND_MINION, mtab[jnp.clip(sub, 0, 3)], r)
    return jnp.where(kind == W.KIND_CHAMPION, CHAMPION_PATHING_RADIUS, r).astype(jnp.float32)


def contact_distance(ri, rj):
    """Minimum centre distance of a pair: a centre cannot enter the other's pathing radius (wiki Unit collision)."""
    return jnp.maximum(ri, rj)


def _walkable(px, py, team, clearance, terrain):
    tv = lambda tm: team_view(terrain, jnp.where(tm == 1, 1, 0))                     # noqa: E731
    return jax.vmap(lambda u, v, tm, r: is_walkable(u, v, r, tv(tm)))(px, py, team, clearance)


def _rot(ux, uy, a):
    c, s = jnp.cos(a), jnp.sin(a)
    return ux * c - uy * s, ux * s + uy * c


def avoid(x0, y0, x1, y1, radius, obstacle, mover, goal_x, goal_y, team, clearance, terrain, dt):
    """Phase 1: steer each ``mover`` (N,) around ``obstacle`` units; returns the new ``(x1, y1)``."""
    n = x0.shape[0]
    vx, vy = x1 - x0, y1 - y0
    step = jnp.sqrt(vx * vx + vy * vy)
    act = mover & (step > 1e-3) & (step <= AVOID_MAX_STEP)
    ux, uy = vx / jnp.maximum(step, 1e-6), vy / jnp.maximum(step, 1e-6)
    h = AVOID_HORIZON_S / dt                                                        # horizon in ticks
    rx, ry = x0[None, :] - x0[:, None], y0[None, :] - y0[:, None]                   # (i, j): j relative to i
    dist = jnp.sqrt(rx * rx + ry * ry)
    rsum = contact_distance(radius[:, None], radius[None, :])
    gdist = jnp.sqrt((goal_x - x0) ** 2 + (goal_y - y0) ** 2)
    on_goal = jnp.sqrt((x0[None, :] - goal_x[:, None]) ** 2 + (y0[None, :] - goal_y[:, None]) ** 2) < rsum
    rel = obstacle[None, :] & ~jnp.eye(n, dtype=bool) & ~on_goal & (dist - rsum < gdist[:, None]) \
        & (dist < rsum + (step[:, None] + AVOID_MAX_STEP) * h)
    walks = obstacle & (step <= AVOID_MAX_STEP)                                     # obstacle's own step
    ovx, ovy = jnp.where(walks, vx, 0.0), jnp.where(walks, vy, 0.0)

    def contact_time(cx, cy):
        """(N,) heading -> earliest contact over the horizon in ticks (inf = clear), and the hit mask."""
        wx, wy = cx[:, None] * step[:, None] - ovx[None, :], cy[:, None] * step[:, None] - ovy[None, :]
        a = rx * wx + ry * wy
        ww = jnp.maximum(wx * wx + wy * wy, 1e-9)
        c = dist * dist - (rsum - 0.5) ** 2
        disc = a * a - ww * c
        t_hit = jnp.where(c <= 0.0, 0.0, (a - jnp.sqrt(jnp.maximum(disc, 0.0))) / ww)
        hit = rel & (a > 0.0) & (disc > 0.0) & (t_hit <= h)
        return jnp.min(jnp.where(hit, t_hit, jnp.inf), axis=1), hit

    t0, b0 = contact_time(ux, uy)
    # Turn away from the nearest blocker of the straight heading (left of the path -> turn right).
    near = jnp.argmin(jnp.where(b0, dist, jnp.inf), axis=1)
    cross = ux * ry[jnp.arange(n), near] - uy * rx[jnp.arange(n), near]
    # Dead-ahead ties: mirror the handedness between the teams, like the map.
    side = jnp.where(cross > 0.0, -1.0, jnp.where(cross < 0.0, 1.0, jnp.where(team == 1, -1.0, 1.0)))
    angles = np.deg2rad(np.asarray(AVOID_ANGLES_DEG))
    picks = []
    for sgn in (1.0, -1.0):
        best = None
        for j, a in enumerate(angles):
            cx, cy = _rot(ux, uy, side * sgn * a)
            ttc, _ = contact_time(cx, cy) if j else (t0, None)
            score = jnp.where(jnp.isinf(ttc), 1e6 - j, ttc)                         # clear > blocked; small turns
            if best is None:
                best = (score, cx, cy, jnp.full((n,), j))
            else:
                take = score > best[0]
                best = tuple(jnp.where(take, v, b) for v, b in zip((score, cx, cy, jnp.full((n,), j)), best))
        picks.append(best)
    (sa, ax, ay, ka), (sb, bx, by, kb) = picks
    pa = (ax * step + x0, ay * step + y0)
    pb = (bx * step + x0, by * step + y0)
    wa = (ka == 0) | _walkable(pa[0], pa[1], team, clearance, terrain)
    wb = (kb == 0) | _walkable(pb[0], pb[1], team, clearance, terrain)
    use_b = wb & (~wa | (sb > sa))
    use_a = wa & ~use_b
    nx = jnp.where(use_a, pa[0], jnp.where(use_b, pb[0], x1))
    ny = jnp.where(use_a, pa[1], jnp.where(use_b, pb[1], y1))
    k = jnp.where(use_a, ka, jnp.where(use_b, kb, 0))
    ok = act & (k > 0)
    nx, ny = jnp.where(ok, nx, x1), jnp.where(ok, ny, y1)
    # Every heading was blocked this tick: stop at contact (wiki "collide upon meeting").
    sc = jnp.where(use_a, sa, jnp.where(use_b, sb, jnp.where(jnp.isinf(t0), 1e6, t0)))
    frac = jnp.where(act & (sc < 1.0), jnp.maximum(sc, CONTACT_MIN_FRAC), 1.0)
    return x0 + (nx - x0) * frac, y0 + (ny - y0) * frac


def separate(x, y, radius, collide, moving, team, clearance, terrain, iters: int = SEPARATION_ITERS):
    """Phase 2: soft Jacobi separation of overlapping ``collide`` units; returns ``(x, y)``."""
    n = x.shape[0]
    idx = jnp.arange(n)
    pair = collide[:, None] & collide[None, :] & ~jnp.eye(n, dtype=bool)
    rsum = contact_distance(radius[:, None], radius[None, :])
    lo = jnp.minimum(idx[:, None], idx[None, :]).astype(jnp.float32)
    hi = jnp.maximum(idx[:, None], idx[None, :]).astype(jnp.float32)
    ang = _GOLDEN * (lo * 7.0 + hi)
    sgn = jnp.where(idx[:, None] < idx[None, :], 1.0, -1.0)
    fx, fy = sgn * jnp.cos(ang), sgn * jnp.sin(ang)                                 # antisymmetric fallback
    start_ok = _walkable(x, y, team, clearance, terrain)
    mob = jnp.where(moving, 1.0, STATIONARY_MOBILITY)
    for _ in range(iters):
        dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]                   # i away from j
        d = jnp.sqrt(dx * dx + dy * dy)
        over = jnp.where(pair, jnp.maximum(rsum - d, 0.0), 0.0)
        tiny = d < 1e-3
        nx = jnp.where(tiny, fx, dx / jnp.maximum(d, 1e-6))
        ny = jnp.where(tiny, fy, dy / jnp.maximum(d, 1e-6))
        share = mob[:, None] / jnp.maximum(mob[:, None] + mob[None, :], 1e-9)
        px, py = jnp.sum(share * over * nx, 1), jnp.sum(share * over * ny, 1)
        mag = jnp.sqrt(px * px + py * py)
        f = jnp.minimum(1.0, MAX_PUSH / jnp.maximum(mag, 1e-6))
        cx, cy = x + px * f, y + py * f
        ok = _walkable(cx, cy, team, clearance, terrain) | ~start_ok
        pushed = mag > 1e-4
        x, y = jnp.where(ok, cx, x), jnp.where(ok, cy, y)
        mob = jnp.where(pushed & ~ok, 1e-3, mob)
    return x, y


def resolve(x0, y0, x1, y1, *, radius, collide, ghosted, moving, goal_x, goal_y, team, clearance, terrain, dt,
            movers: int | None = None):
    """One tick of collision: new ``(x, y)`` (N,) from tick-start ``x0, y0`` and post-movement ``x1, y1``.

    ``collide``: units taking part; ``ghosted``: excluded; ``moving``: walked a route step (avoid, yield more);
    ``goal_*``: each mover's goal (chase target); ``team``: terrain mask; ``clearance``: terrain-check radius.
    ``movers`` (static): only slots ``[0, movers)`` take part.
    """
    if movers is not None and movers < x1.shape[0]:
        m = movers
        sl = lambda v: v[:m]                                                       # noqa: E731
        mx, my = resolve(sl(x0), sl(y0), sl(x1), sl(y1), radius=sl(radius), collide=sl(collide),
                         ghosted=sl(ghosted), moving=sl(moving), goal_x=sl(goal_x), goal_y=sl(goal_y),
                         team=sl(team), clearance=sl(clearance), terrain=terrain, dt=dt)
        return jnp.concatenate([mx, x1[m:]]), jnp.concatenate([my, y1[m:]])
    f32 = lambda v: jnp.asarray(v, jnp.float32)                                     # noqa: E731
    x0, y0, x1, y1, radius, goal_x, goal_y, clearance = map(f32, (x0, y0, x1, y1, radius, goal_x, goal_y, clearance))
    team = jnp.asarray(team, jnp.int32)
    solid = jnp.asarray(collide, bool) & ~jnp.asarray(ghosted, bool)
    moving = jnp.asarray(moving, bool)
    x, y = avoid(x0, y0, x1, y1, radius, solid, solid & moving, goal_x, goal_y, team, clearance, terrain, dt)
    return separate(x, y, radius, solid, moving, team, clearance, terrain)

