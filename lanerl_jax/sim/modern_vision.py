"""Fog of war for the 26.19 modern world (MODERN-009 vision slice).

``visibility`` answers, every tick, which units each team can see. Rules and
evidence (docs/modern/VISION.md):

* **Sight radius** belongs to the viewer and is measured centre to centre.
  The values are 1350 for champions, super minions, turrets and the Nexus,
  and 1200 for melee, caster and siege minions. Sources: the wiki Sight page
  (2026-10) and the client ``perceptionBubbleRadius`` (1200 in the
  ``SRU_*Minion{Melee,Ranged,Siege}`` records, 1350 in ``Nexus``). The
  champion, turret and super-minion records carry no override, so their
  values come from the wiki. Inhibitors grant no sight (no record value, not
  listed; INFERRED-L, irrelevant to the top lane). Dead units grant none.
* **Walls** block sight, except transparent walls (navgrid 0x40) and
  always-visible cells (0x100) -- the default ``fog="rays"``. The optional
  "fast" fog ignores walls (VIS-FAST-M).
* **Brush** (navgrid bit 1) hides a unit from viewers outside it. From inside
  a brush a unit sees out and sees its own brush patch. With ``fog="rays"`` a
  ray passing through a brush is blocked too; the fast fog does not model that.
* **Structures** are not affected by fog: turrets, inhibitors and Nexuses
  are always visible to both teams.
* **Own team**: a team always sees its own units.
* **Attack reveal**: a champion that is hidden from the enemy team and starts
  a basic attack or a unit-targeted ability reveals a 300-unit circle around
  where it stood. The circle lasts 2 s, ignores walls and brush, and is shown
  to the enemy team (wiki Sight/Brush). The client map constants
  ``ca_RevealAttackerRange`` 400 / ``ca_RevealAttackerTimeOut`` 4.5 s exist
  but their trigger is undocumented; they are kept as named alternatives and
  not used (U-VIS-1).

* **Wards** (``KIND_WARD``, ``sub`` = ``modern_wards.WardType``) are ordinary
  viewers with their own radius (900 Totem/Control, 500 Farsight; WARDS.md);
  a ward in a brush sees that brush like any unit. Disabled wards get radius 0
  through the ``radius`` override.
* **Stealth / true sight** (optional, unit-agnostic): a ``stealthed`` unit is
  hidden from the enemy team unless it lies within ``true_sight`` radius of a
  live enemy unit (centre to centre, ignoring walls and brush, WARDS U-W-3);
  turrets always carry 1100 true sight (wiki Turret). Attack-reveal circles
  are standard sight and do not show stealthed units.
* **Exposed** units (Sixth Sense / Oracle-hit reveals, a Control Ward that is
  revealing a stealthed ward) are visible to the enemy team regardless of fog.
* **Unobstructed** viewers (Farsight Ward) ignore walls and brush.

Both modes reuse ``obs.vision.clear_ray`` over the modern navgrid (same flag
bits as the legacy grid): the fast mode is its brush-id lookup (the legacy
VIS-FAST lane approximation), the ray mode its supercover caster (fused CUDA
kernel on GPU, reference loop on CPU). Faelights, nearsight and dynamic
terrain are deferred.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from . import modern_world_types as W

__all__ = ["CHAMPION_SIGHT", "MINION_SIGHT", "SUPER_MINION_SIGHT", "TURRET_SIGHT", "NEXUS_SIGHT",
           "INHIBITOR_SIGHT", "REVEAL_RADIUS", "REVEAL_DURATION", "CLIENT_REVEAL_ATTACKER_RANGE",
           "CLIENT_REVEAL_ATTACKER_TIMEOUT", "Reveal", "init_reveal", "vision_grid", "sight_radius",
           "visibility", "reveal_step", "WARD_SIGHT", "FARSIGHT_SIGHT", "TURRET_TRUE_SIGHT"]

CHAMPION_SIGHT = 1350.0
MINION_SIGHT = 1200.0
SUPER_MINION_SIGHT = 1350.0
TURRET_SIGHT = 1350.0
NEXUS_SIGHT = 1350.0
INHIBITOR_SIGHT = 0.0
REVEAL_RADIUS = 300.0
REVEAL_DURATION = 2.0
CLIENT_REVEAL_ATTACKER_RANGE = 400.0      # map11 ca_RevealAttackerRange (unused, U-VIS-1)
CLIENT_REVEAL_ATTACKER_TIMEOUT = 4.5      # map11 ca_RevealAttackerTimeOut (unused, U-VIS-1)
SUPER = 3                                  # modern_minions.MinionType.SUPER
WARD_SIGHT = 900.0                         # client YellowTrinket/JammerDevice perceptionBubbleRadius
FARSIGHT_SIGHT = 500.0                     # client BlueTrinket perceptionBubbleRadius
FARSIGHT_SUB = 2                           # modern_wards.WardType.FARSIGHT
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
    """``obs.vision.VisionGrid`` over a ``data.modern_map.ModernMapGrid`` (arrays [z, x]).

    ``rays=False`` ("fast" fog): brush patches are labelled once and
    visibility is a position lookup (target outside any brush, or in the
    viewer's brush); walls do not block sight. ``rays=True``: the full
    supercover ray rule (walls and brush along the ray).
    """
    import numpy as np
    from ..obs.vision import VisionGrid, with_bush_ids
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
               radius=None, stealthed=None, true_sight=None, unobstructed=None, exposed=None, sources=None):
    """Returns ``(visible (2, N), sight (N, N))``.

    ``visible[t, j]``: team t sees unit j. ``sight[i, j]``: unit i itself has
    j within its sight radius on a clear ray (the rune "own sight"). Units
    ``n_fogged`` and above are structures (never fogged), so rays are only
    cast to the first ``n_fogged`` slots (champions, minions, monsters, wards).

    Optional (N,) inputs (None = rule off): ``radius`` overrides the sight
    radius (wards: disabled = 0, Farsight 800/500); ``stealthed`` hides a unit
    from enemies outside their ``true_sight`` radius (turrets add 1100);
    ``unobstructed`` viewers ignore walls/brush; ``exposed`` units are shown to
    the enemy team through fog. ``sources`` = ``(x, y, radius, team)`` (K,) extra
    sight points that are not units (Scuttle Speed Shrine); they see like a unit
    standing there (walls/brush rays, no true sight).
    """
    from ..obs.vision import clear_ray
    n = x.shape[0]
    live = alive & (kind != W.KIND_NONE)
    r = sight_radius(kind, sub, live) if radius is None else jnp.where(live, radius, 0.0).astype(jnp.float32)
    tx, ty = x[:n_fogged], y[:n_fogged]
    d2 = (x[:, None] - tx[None, :]) ** 2 + (y[:, None] - ty[None, :]) ** 2
    in_range = (d2 <= (r ** 2)[:, None]) & (r > 0)[:, None] & live[:n_fogged][None, :]
    # Rays only where they can change the answer: an enemy viewer in range of a live unit.
    enemy = team[:, None] != team[None, :n_fogged]
    clear = clear_ray(grid, x[:, None], y[:, None], tx[None, :], ty[None, :], enabled=in_range & enemy)
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
    return visible & live[None, :], sight


def reveal_step(reveal: Reveal, hidden, triggered, x, y, now) -> Reveal:
    """Open a circle where a hidden champion attacked or cast at a unit this tick."""
    start = triggered & hidden
    return Reveal(jnp.where(start, x, reveal.x), jnp.where(start, y, reveal.y),
                  jnp.where(start, now + REVEAL_DURATION, reveal.until))
