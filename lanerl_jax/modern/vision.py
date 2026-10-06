"""Fog of war (docs/modern/VISION.md): which units each team sees, recomputed every tick.

Sight radius belongs to the viewer, centre to centre (champions, super minions, turrets, Nexus 1350; other
minions 1200; wards 900, Farsight 500; inhibitors 0). Walls block except transparent/always-visible cells
(``fog="rays"``; ``"fast"`` ignores walls); brush hides units from viewers outside it. Structures and the
own team are always visible. A hidden champion that starts an attack or unit-targeted cast reveals a 300-unit
circle for 2 s. Stealthed units need enemy true sight (turrets 1100); exposed units are seen through fog.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from .rays import VisionGrid, clear_pairs, clear_ray, with_bush_ids
from .core import types as W

CHAMPION_SIGHT = 1350.0
MINION_SIGHT = 1200.0
SUPER_MINION_SIGHT = 1350.0
TURRET_SIGHT = 1350.0
NEXUS_SIGHT = 1350.0
INHIBITOR_SIGHT = 0.0
REVEAL_RADIUS = 300.0
REVEAL_DURATION = 2.0
SUPER = 3                                  # lane.minions.MinionType.SUPER
WARD_SIGHT = 900.0                         # client YellowTrinket/JammerDevice perceptionBubbleRadius
FARSIGHT_SIGHT = 500.0                     # client BlueTrinket perceptionBubbleRadius
FARSIGHT_SUB = 2                           # wards.WardType.FARSIGHT
TURRET_TRUE_SIGHT = 1100.0                 # wiki Turret "True Sight ... within 1100 range"


class Reveal(NamedTuple):
    """One attack-reveal circle per champion (C,)."""
    x: Any
    y: Any
    until: Any              # game time the circle ends (-inf = none)


def init_reveal(c: int) -> Reveal:
    z = jnp.zeros((c,), jnp.float32)
    return Reveal(z, z, jnp.full((c,), -jnp.inf, jnp.float32))


def vision_grid(grid, *, rays: bool = False) -> Any:
    """``VisionGrid`` of a navgrid; ``rays=False`` labels brush patches for the fast (lookup) fog."""
    import numpy as np
    g = VisionGrid(jnp.asarray(np.asarray(grid.flags, np.int32)), float(grid.cell_size),
                   float(grid.min_bounds[0]), float(grid.min_bounds[2]))
    return g if rays else with_bush_ids(g)


def sight_radius(kind, sub, alive) -> Any:
    r = jnp.where(kind == W.KIND_CHAMPION, CHAMPION_SIGHT,
                  jnp.where(kind == W.KIND_MINION, jnp.where(sub == SUPER, SUPER_MINION_SIGHT, MINION_SIGHT),
                            jnp.where(kind == W.KIND_TURRET, TURRET_SIGHT,
                                      jnp.where(kind == W.KIND_NEXUS, NEXUS_SIGHT,
                                                jnp.where(kind == W.KIND_INHIBITOR, INHIBITOR_SIGHT,
                                                          jnp.where(kind == W.KIND_WARD,
                                                                    jnp.where(sub == FARSIGHT_SUB, FARSIGHT_SIGHT,
                                                                              WARD_SIGHT), 0.0))))))
    return jnp.where(alive, r, 0.0).astype(jnp.float32)


def visibility(x, y, kind, sub, team, alive, reveal: Reveal, now, grid, *, n_fogged: int,
               radius=None, stealthed=None, true_sight=None, unobstructed=None, exposed=None, sources=None,
               ray_capacity=None):
    """``(visible (2, N), sight (N, N), dropped)``: team t sees unit j; unit i itself sees j (rune "own sight");
    ``dropped`` sight rays past ``ray_capacity`` (``rays.clear_pairs``; must stay 0).

    Units from ``n_fogged`` on are structures (never fogged). Optional (N,) inputs, None = off: ``radius``
    overrides the sight radius, ``stealthed`` units need enemy ``true_sight``, ``unobstructed`` viewers ignore
    walls and brush, ``exposed`` units are seen through fog. ``sources = (x, y, radius, team)`` are extra
    sight points that are not units (Scuttle shrines)."""
    n = x.shape[0]
    live = alive & (kind != W.KIND_NONE)
    r = sight_radius(kind, sub, live) if radius is None else jnp.where(live, radius, 0.0).astype(jnp.float32)
    tx, ty = x[:n_fogged], y[:n_fogged]
    d2 = (x[:, None] - tx[None, :]) ** 2 + (y[:, None] - ty[None, :]) ** 2
    in_range = (d2 <= (r ** 2)[:, None]) & (r > 0)[:, None] & live[:n_fogged][None, :]
    # Rays only where they can change the answer: an enemy viewer in range of a live unit.
    enemy = team[:, None] != team[None, :n_fogged]
    clear, dropped = clear_pairs(grid, x, y, tx, ty, in_range & enemy, ray_capacity)
    if unobstructed is not None:
        clear = clear | unobstructed[:, None]
    self_pair = jnp.arange(n)[:, None] == jnp.arange(n_fogged)[None, :]
    sight_f = in_range & (clear | ~enemy) | self_pair & live[:n_fogged][None, :]
    hidden_t = jnp.zeros((n_fogged,), bool)
    if stealthed is not None:
        hidden_t = stealthed[:n_fogged] & live[:n_fogged]
        ts = jnp.where(kind == W.KIND_TURRET, TURRET_TRUE_SIGHT, 0.0)
        if true_sight is not None:
            ts = jnp.maximum(ts, true_sight)
        ts = jnp.where(live, ts, 0.0)
        ts_in = (d2 <= (ts ** 2)[:, None]) & (ts > 0)[:, None]
        sight_f = jnp.where(hidden_t[None, :] & enemy, ts_in & live[:n_fogged][None, :], sight_f)
    structure = (kind == W.KIND_TURRET) | (kind == W.KIND_INHIBITOR) | (kind == W.KIND_NEXUS)
    sight = jnp.concatenate([sight_f, jnp.broadcast_to((structure & live)[None, n_fogged:], (n, n - n_fogged))
                             & ((x[:, None] - x[None, n_fogged:]) ** 2 + (y[:, None] - y[None, n_fogged:]) ** 2
                                <= (r ** 2)[:, None])], axis=1)
    teams = jnp.arange(2)
    seen = jnp.any((team[None, :, None] == teams[:, None, None]) & live[None, :, None] & sight_f[None], axis=1)
    # Attack reveal circles: champion c's circle is shown to the team that is not c's.
    c = reveal.x.shape[0]
    active = now < reveal.until
    in_circle = ((x[None, :n_fogged] - reveal.x[:, None]) ** 2 + (y[None, :n_fogged] - reveal.y[:, None]) ** 2
                 <= REVEAL_RADIUS ** 2) & active[:, None] & ~hidden_t[None, :]
    shown_to = teams[:, None] != team[None, :c]                                            # (2, C)
    seen = seen | jnp.any(shown_to[:, :, None] & in_circle[None], axis=1)
    if exposed is not None:
        seen = seen | (exposed[None, :n_fogged] & (team[None, :n_fogged] != teams[:, None]))
    if sources is not None:
        px, py, pr, pt = (jnp.asarray(v) for v in sources)
        d2p = (px[:, None] - tx[None, :]) ** 2 + (py[:, None] - ty[None, :]) ** 2
        in_p = (d2p <= (pr ** 2)[:, None]) & (pr > 0)[:, None] & live[:n_fogged][None, :]
        clear_p = clear_ray(grid, px[:, None], py[:, None], tx[None, :], ty[None, :], enabled=in_p)
        seen_p = in_p & clear_p & ~hidden_t[None, :]
        seen = seen | jnp.any((pt[None, :, None] == teams[:, None, None]) & seen_p[None], axis=1)
    visible = jnp.concatenate([seen, jnp.ones((2, n - n_fogged), bool)], axis=1)
    visible = visible | (team[None, :] == teams[:, None])
    return visible & live[None, :], sight, dropped


def reveal_step(reveal: Reveal, hidden, triggered, x, y, now) -> Reveal:
    """Open a circle where a hidden champion attacked or cast at a unit this tick."""
    start = triggered & hidden
    return Reveal(jnp.where(start, x, reveal.x), jnp.where(start, y, reveal.y),
                  jnp.where(start, now + REVEAL_DURATION, reveal.until))
