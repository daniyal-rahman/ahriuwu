"""Unit-vs-unit collision of the 26.19 modern world (docs/modern/COLLISION.md).

Replaces the legacy C# port (``collision.resolve_collisions``: sequential spawn-order escape
teleports, no avoidance) in ``world.tick._move``. Current League (COLLISION §1):

* every live champion, lane minion and monster collides with every other one, allies and enemies
  alike, through its **pathing radius** (client ``pathfindingCollisionRadius``: Garen/Jax 35,
  melee/caster 35.74, siege 55.74, super 55.52, camps per record). The gameplay radius (48/65)
  stays the hitbox / attack-range edge and is not used here;
* ghosted units (Ghost, Garen E, dashes, first-wave minions) neither block nor are blocked; wards
  and structures are not unit obstacles (structures block through their navgrid pads);
* movers steer around blockers (avoidance) rather than shoving them; what still overlaps is
  separated softly, the stationary side yielding less, never into terrain.

Two vectorized phases, fixed shapes, no per-unit sequential loop (O(N^2) pair math):

1. **Avoidance** (velocity level). Each mover's route step ``p1 - p0`` is tried at headings rotated
   by ``AVOID_ANGLES_DEG``, preferring the side away from the nearest blocker. Per heading, the first
   contact time against every obstacle over ``AVOID_HORIZON_S`` (relative motion, moving obstacles at
   their own step) is solved. Per side, the pick is the smallest clear turn, else the latest contact.
   Both picks are terrain-checked (end disk at movement clearance on the team mask) and the better
   walkable one is taken at the route step's length (else the route step). A pick that still meets
   an obstacle this tick stops at contact (``CONTACT_STOP``; at least ``CONTACT_MIN_FRAC`` of the
   step). Obstacles on the mover's goal (its chase target) or past it are ignored.
2. **Separation** (position level). ``SEPARATION_ITERS`` Jacobi rounds: each overlapping pair is
   pushed apart along its centre line, split by mobility (movers ``1``, stationary units
   ``STATIONARY_MOBILITY``), at most ``MAX_PUSH`` per round. A push whose end disk is not walkable
   is refused and that unit is pinned (mobility ~0) for the remaining rounds so its partner takes
   the whole push. Units that start in terrain are not held by the check (DTR.eject handles them).

Deterministic: Jacobi sums are order-independent and exactly coincident pairs separate along an
index-derived direction.
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

# ---- radii (CLIENT H, 26.19 character records) -------------------------------------------------
CHAMPION_PATHING_RADIUS = 35.0                               # garen.bin / jax.bin pathfindingCollisionRadius
MINION_PATHING_RADIUS = (35.7437, 35.7437, 55.7437, 55.5208)  # melee, caster, siege, super (MINIONS §1.1)
_JUNGLE = PATCH_DIR / "jungle_client.json"

# ---- tuning (INFERRED; client ai_PostAvoidance* semantics unknown, COLLISION §4) ----------------
AVOID_ANGLES_DEG = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)     # candidate heading offsets, both sides
AVOID_HORIZON_S = 0.3                                        # look-ahead (= ai_PostAvoidanceFilterDuration)
AVOID_MAX_STEP = 60.0                                        # longer "steps" are blinks/teleports: no avoidance
CONTACT_STOP = True                                          # wiki "collide upon meeting": step ends at contact
CONTACT_MIN_FRAC = 0.25                                      # ... but never below this share (no deadlock)
SEPARATION_ITERS = 3
STATIONARY_MOBILITY = 0.25                                   # share of a push a non-mover takes vs a mover
MAX_PUSH = 20.0                                              # per separation round (soft: deep overlaps take ticks)
# Contact distance of a pair. Wiki Unit_collision: "the center of a unit ... cannot enter another unit's
# pathing radius" -> centres stay max(r_i, r_j) apart ("max"); "sum" is the disk-disk reading.
PAIR_RULE = "max"
_GOLDEN = 2.399963229728653                                  # coincident-pair fallback direction step (rad)


@lru_cache(maxsize=1)
def monster_pathing_radii() -> tuple:
    """Pathing radius per jungle ``Monster`` type (``jungle.camps.CHARACTERS`` order), client records."""
    data = json.loads(_JUNGLE.read_text())["monsters"]
    return tuple(float(data[c]["pathing_radius"]) for c in CHARACTERS)


def pathing_radius(kind: Any, sub: Any, gameplay_radius: Any) -> Any:
    """(N,) unit-collision radius: champions 35, minions by type, camp monsters by record.

    Epic monsters (``sub >= jungle.camps.EPIC_SUB_BASE``) and anything else keep their gameplay
    radius (INFERRED L: Baron's client pathing radius is 0, yet it always blocks)."""
    kind, sub = jnp.asarray(kind), jnp.asarray(sub, jnp.int32)
    r = jnp.asarray(gameplay_radius, jnp.float32)
    mon = jnp.asarray(monster_pathing_radii(), jnp.float32)
    mtab = jnp.asarray(MINION_PATHING_RADIUS, jnp.float32)
    is_camp = (kind == W.KIND_MONSTER) & (sub >= 0) & (sub < mon.shape[0])
    r = jnp.where(is_camp, mon[jnp.clip(sub, 0, mon.shape[0] - 1)], r)
    r = jnp.where(kind == W.KIND_MINION, mtab[jnp.clip(sub, 0, 3)], r)
    return jnp.where(kind == W.KIND_CHAMPION, CHAMPION_PATHING_RADIUS, r).astype(jnp.float32)


def contact_distance(ri, rj):
    """Minimum centre distance of a colliding pair (``PAIR_RULE``)."""
    return jnp.maximum(ri, rj) if PAIR_RULE == "max" else ri + rj


def _walkable(px, py, team, clearance, terrain):
    tv = lambda tm: team_view(terrain, jnp.where(tm == 1, 1, 0))                     # noqa: E731
    return jax.vmap(lambda u, v, tm, r: is_walkable(u, v, r, tv(tm)))(px, py, team, clearance)


def _rot(ux, uy, a):
    c, s = jnp.cos(a), jnp.sin(a)
    return ux * c - uy * s, ux * s + uy * c


def avoid(x0, y0, x1, y1, radius, obstacle, mover, goal_x, goal_y, team, clearance, terrain, dt):
    """Phase 1: steer each ``mover`` (N,) around ``obstacle`` units. Returns new ``(x1, y1)``."""
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

    def contact_time(cx, cy):                                                       # (N,) heading -> (N,) ticks
        """Earliest first-contact time over the horizon against any relevant obstacle (inf = clear)."""
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
    # Dead-ahead ties: the map is mirror-symmetric between the teams (handedness flips), so mirror it.
    side = jnp.where(cross > 0.0, -1.0, jnp.where(cross < 0.0, 1.0, jnp.where(team == 1, -1.0, 1.0)))
    angles = np.deg2rad(np.asarray(AVOID_ANGLES_DEG))                                # 0 first, both sides
    # Per side: the smallest-turn clear heading, else the one whose first contact comes latest. Score
    # = clear first, then smaller turn; uncleared ranked by contact time. Then terrain-check both sides'
    # picks (two disk checks per mover) and take the better walkable one (the route step if neither).
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
    if CONTACT_STOP:
        # The chosen heading still meets an obstacle this tick (every heading was blocked): stop at contact.
        sc = jnp.where(use_a, sa, jnp.where(use_b, sb, jnp.where(jnp.isinf(t0), 1e6, t0)))
        frac = jnp.where(act & (sc < 1.0), jnp.maximum(sc, CONTACT_MIN_FRAC), 1.0)
        nx, ny = x0 + (nx - x0) * frac, y0 + (ny - y0) * frac
    return nx, ny


def separate(x, y, radius, collide, moving, team, clearance, terrain, iters: int = SEPARATION_ITERS):
    """Phase 2: soft Jacobi separation of overlapping ``collide`` units. Returns ``(x, y)``."""
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
    """Modern unit collision for one tick. Returns new ``(x, y)`` (N,).

    ``x0, y0``: positions at the start of the tick; ``x1, y1``: after route movement, dashes and
    blinks. ``radius``: pathing radius (``pathing_radius``). ``collide``: live units that take part
    (no wards/structures). ``ghosted``: excluded from both roles. ``moving``: units that walked a
    route step this tick (avoidance applies to them; they also yield more in separation).
    ``goal_x, goal_y``: each mover's goal (its chase target's position when chasing). ``team``:
    terrain mask per unit (0/1). ``clearance``: (N,) movement clearance for the terrain checks.
    ``movers`` (static): only slots ``[0, movers)`` take part; later slots are returned unchanged.
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


__all__ = ["CHAMPION_PATHING_RADIUS", "MINION_PATHING_RADIUS", "monster_pathing_radii", "pathing_radius", "avoid",
           "separate", "resolve"]
