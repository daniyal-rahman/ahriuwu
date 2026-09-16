"""Unit collision: units push each other apart.

``AttackableUnit.OnCollision`` (non-terrain branch)::

    exit = Extensions.GetCircleEscapePoint(Position, PathfindingRadius + 1,
                                           collider.Position,
                                           collider.PathfindingRadius);
    if (!IsWalkable(exit, PathfindingRadius))
        exit = GetClosestTerrainExit(exit, PathfindingRadius + 1);
    SetPosition(exit, false);

and ``GetCircleEscapePoint(p1, r1, p2, r2)`` unwinds to a clean formula::

    edgepoint1 = p1 + u*r1            u = normalize(p2 - p1)
    edgepoint2 = p2 - u*r2 = p1 + u*(d - r2)
    exit       = p1 + (edgepoint2 - edgepoint1) = p1 + u*(d - r1 - r2)

So when the two overlap (``d < r1 + r2``) the term is negative and the unit
slides *away* from the collider by exactly the overlap. Not a spring, not a
velocity change -- a teleport to touching.

Who collides
------------
``IsCollisionAffected`` excludes ``LevelProp``, ``Particle``, ``ObjBuilding``
and **``BaseTurret``**, so turrets neither push nor are pushed. Missiles,
sectors and owned regions are skipped inside ``OnCollision`` itself. Ghosted
units (Garen's E sets ``StatusFlags.Ghosted``) pass through everything, and so
do units under ``MovementParameters`` (dashes).

Why it matters here
-------------------
Without collision, casters are never pushed into melee reach. Measured: the
sim's live-minion mix ran to 59.9% casters against the server's 53.6%, and the
steady-state population sat +10% high after the minion-modifier bug was fixed --
both consistent with the back line never being forced into the fight.

**Booked approximation:** the server resolves collisions pair-by-pair as its
handler iterates, so one unit can be pushed several times in a single update and
later pushes see earlier ones. This applies **one push per unit per tick**,
simultaneously for all units. Same fixed point in a settled formation;
different transient in a crush.

Measured cost of this approximation, and what "lowest-index" actually is
--------------------------------------------------------------------------
`docs/TIER1_POST_REORDER.md` (gate-1 task 3) localises essentially all of the
one-step position residual to here: units with zero colliding neighbours are
93-94% exact to the gate's own ≤1/16 criterion, falling to ~70-75% at one
neighbour and ~45-50% at three or more.

The "lowest-index overlapping neighbour, matching the server's iteration
order" choice below is **not correct as stated, and was checked against
source rather than left alone once that was flagged.** The server's
`CollisionHandler._objects` genuinely does have a stable order (creation
order -- `AddObject`/`GameObject.OnAdded` insert once, and `List<T>.Remove`
preserves every survivor's relative position), but this array's SLOT index
does not track it: a minion slot is recycled on death, so a low slot index
means "whichever free slot the last spawn picked", not "created early", and
turrets (created at map load, first) sit in the LAST slice of this array
while the server creates them before any minion -- slot and creation order
disagree even before recycling. Measured directly: a from-scratch NumPy
port of the server's real per-tick algorithm (sequential, one escape per
overlapping neighbour, each an immediate in-place update later checks in
the same pass can see) run in SLOT order came out WORSE than this module's
one-push simultaneous approximation at every crowding level; the SAME port
run in a reconstructed CREATION order came out BETTER than it at every
level, including a clean test (one colliding neighbour, where the OTHER
open question below cannot matter) that isolates creation order alone. See
that doc for the numbers, the reconstruction method, and what is not yet
resolved.

Separately: the claim that lowest-index-wins mirrors the server's own
tie-break rule is real, but it is transplanted from the wrong subsystem.
`ObjAIBase.cs:1295-1320` (TARGETING, not collision) breaks distance ties by
`ObjectManager.GetObjects()` iteration order -- that is where "ties go to
whoever was added first" is an actual server rule. Nothing in
`CollisionHandler` picks "the" overlapping neighbour by any index at all:
`UpdateCollision` pushes against EVERY neighbour `GetNearestObjects` returns
(a quadtree query, not an index scan), one escape per neighbour, in
sequence. That traversal's own order is a second, separate open question
(`docs/TIER1_POST_REORDER.md` again) -- not resolved here, and not
something "lowest index" was ever a reconstruction of.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from .state import Kind

__all__ = ["resolve_collisions"]


def resolve_collisions(x: jax.Array, y: jax.Array, kind: jax.Array,
                       alive: jax.Array, pathfinding_radius: jax.Array,
                       ghosted: jax.Array | None = None
                       ) -> Tuple[jax.Array, jax.Array]:
    """One round of push-apart. Returns new ``(x, y)``."""
    n = x.shape[0]
    # turrets and buildings are not collision-affected
    collides = alive & (kind != Kind.TURRET) & (kind != Kind.NONE)
    if ghosted is not None:
        collides = collides & ~ghosted

    r1 = pathfinding_radius + 1.0            # `PathfindingRadius + 1`
    r2 = pathfinding_radius
    dx = x[None, :] - x[:, None]             # row = me, col = collider
    dy = y[None, :] - y[:, None]
    d = jnp.sqrt(dx * dx + dy * dy)
    touching = r1[:, None] + r2[None, :]
    overlap = (
        collides[:, None] & collides[None, :]
        & ~jnp.eye(n, dtype=bool)
        & (d < touching) & (d > 0)
    )

    # lowest-index overlapping neighbour -- an arbitrary, deterministic
    # tie-break for "the one push per tick" approximation, NOT a
    # reconstruction of the server's true iteration order (see the module
    # docstring: that order is real, is creation order, is not this array's
    # slot index, and is not read here at all yet).
    first = jnp.argmax(overlap, axis=1)
    has = jnp.any(overlap, axis=1)
    j = jnp.clip(first, 0, n - 1)

    dj = d[jnp.arange(n), j]
    safe = jnp.where(dj > 0, dj, 1.0)
    ux = dx[jnp.arange(n), j] / safe
    uy = dy[jnp.arange(n), j] / safe
    push = dj - r1 - r2[j]                   # negative while overlapping
    return (jnp.where(has, x + ux * push, x),
            jnp.where(has, y + uy * push, y))
