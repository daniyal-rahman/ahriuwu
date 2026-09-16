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
velocity change -- a teleport to touching. Terrain re-projection
(``GetClosestTerrainExit``) is not modelled here; nothing in this lane's
corpus has put a unit inside terrain via a collision push.

Two different radii, two different jobs
----------------------------------------
``CollisionRadius`` and ``PathfindingRadius`` are DIFFERENT fields on
``GameObject`` (`GameObject.cs:53-60`), set independently, and this module
uses both:

* ``IsCollidingWith`` -- **the trigger**, whether a push happens at all --
  sums ``CollisionRadius`` (`GameObject.cs:226-229`, squared-distance
  compare, no ``+1``): ``d < CollisionRadius[i] + CollisionRadius[j]``.
* ``GetCircleEscapePoint`` -- **the resolution**, how far the push goes --
  sums ``PathfindingRadius`` (`AttackableUnit.cs:312`, ``+1`` on the pusher's
  own side only): ``exit = p1 + u*(d - (PathfindingRadius[i] + 1) -
  PathfindingRadius[j])``.

A unit's own ``PathfindingRadius`` always defers to
``CharData.PathfindingCollisionRadius`` (else 40) -- `ObjAIBase.cs:119-125` --
so `data.patch.UnitStats.pathfinding_radius` is already correct as loaded.
``CollisionRadius`` is different: `ObjAIBase`'s ctor
(`ObjAIBase.cs:105-116`) only reads ``CharData.GameplayCollisionRadius`` when
the constructor's own ``collisionRadius`` argument is non-positive, and
``Minion.cs:57`` / ``Champion.cs:52`` pass ``40`` / ``30`` there
UNCONDITIONALLY -- Content is never consulted for a minion or a champion.
Only ``BaseTurret`` passes nothing (defers to CharData for real). See
``sim/profiles.py`` for where the 40/30 override actually lives; this module
just consumes the resulting ``collision_radius`` / ``pathfinding_radius``
columns without knowing why they differ.

Who collides: two masks, not one
---------------------------------
``CollisionHandler`` asks two different questions per object
(`CollisionHandler.cs:33-56`), and they disagree specifically about turrets:

* ``IsCollisionObject`` -- can this be COLLIDED WITH (an obstacle)? Excludes
  only ``LevelProp``/``Particle``/``SpellMissile``/``Region``. Turrets and
  buildings ARE collision objects.
* ``IsCollisionAffected`` -- can this BE PUSHED? Excludes those same four
  PLUS ``ObjBuilding`` and ``BaseTurret``. Turrets are never pushed.

So a turret blocks everyone else's movement but never moves itself --
`BaseTurret` isn't `ObjBuilding` (`AttackableUnit.OnCollision`'s
``collider is ObjBuilding`` early-return does not catch it), so a champion or
minion that overlaps one really does get teleported off its edge. A single
symmetric mask (this module's previous version) makes turrets invisible to
collision entirely, and units pass straight through their hitboxes.
``Ghosted`` (Garen's E) and a live ``MovementParameters`` (a dash, not
modelled in this slice) drop a unit from BOTH roles: `OnCollision` early-
returns whenever either side of the pair is ghosted/dashing
(`AttackableUnit.cs:300-305`), which is equivalent to that unit never being a
valid obstacle and never being affected.

Gauss-Seidel, creation order, multiple pushes per unit per tick
-----------------------------------------------------------------
``CollisionHandler.Update`` (`:121-134`) loops ``_objects`` calling
``UpdateCollision(obj)`` on each; that in turn loops **every** object
``GetNearestObjects(obj)`` returns, calling ``obj.OnCollision(obj2)`` -- which
ends in an immediate ``SetPosition`` -- for each one that currently overlaps
(`:133-153`). Two consequences neither approximated here:

1. **Gauss-Seidel, not Jacobi.** A later unit in `_objects`' order escapes
   from an EARLIER unit's already-moved position, because `SetPosition` is
   immediate, not buffered to end-of-tick.
2. **One escape per overlapping neighbour, not one per unit.** A unit
   wedged between two others gets pushed off BOTH, in sequence, in the same
   tick; each push can change whether the NEXT neighbour still overlaps.

``_objects`` is genuine, permanent creation order: `GameObject.OnAdded` calls
`AddObject` exactly once, and `List<T>.Remove` (on death) shifts survivors
down without reordering them (`GameObject.cs:150-155`). It is NOT this
project's slot index -- a minion slot is recycled on death, and our
``[champions | minions | turrets]`` layout disagrees with the server's
turrets-at-map-load-first order even before recycling -- so a dedicated
``spawn_seq`` field carries it (see ``sim/state.py``).

There is no separate "inner" (quadtree) order to also reconstruct.
`CollisionHandler`'s quadtree (`:26-31`) is constructed with its top/left
arguments swapped (`top: MinGridPosition.X, left: MaxGridPosition.Z`), and
``Circle.ContainedBy`` (`QuadTree.cs:33-41`) compares ``Position.X`` on its
fourth (Y) condition -- so every child rectangle needs
``Position.Y >= MaxGridPosition.Z + Radius``, which is never true on-map.
No child quadrant is ever created (verified by reading `Quadrant.Insert`:
`child` stays null every time, so every node lands in the root's own
circularly-linked list), and that list's own construction+traversal
(`QuadTree.cs`'s `QuadNode`/`GetIntersectingNodes`) yields insertion order,
not "random" order as its own comment claims. So ``GetNearestObjects``
returns creation order too, and one ``spawn_seq``-sorted traversal, reused
for both loops, is not an approximation -- it is what the source does.

Measured (before this fix; ``docs/TIER1_POST_REORDER.md`` gate-1 task 3):
this module's prior one-push Jacobi approximation localises essentially all
of the one-step position residual to crowding -- units with zero colliding
neighbours are 93-94% exact to the gate's own <=1/16 criterion, falling to
~70-75% at one neighbour and ~45-50% at three or more. A from-scratch NumPy
port of the algorithm above (`lanerl_jax.parity.tier1_collision_sequential`),
run in SLOT order, measured WORSE than Jacobi at every crowding level; the
SAME port run in reconstructed CREATION order measured BETTER than Jacobi at
every level, including the n=1 case (one colliding neighbour -- exactly one
escape regardless of any traversal order) that isolates the outer order
alone. See that module and ``docs/TIER1_POST_REORDER.md`` for the full
numbers and the reconstruction method used there (this module now reads the
REAL ``spawn_seq`` instead of reconstructing it).

Implementation: nested ``lax.scan``, not a host callback
-----------------------------------------------------------
A per-tick Python loop over units is off the table by explicit decision (it
would break out of JIT and out of ``vmap``-ability over environments). The
outer scan (over units, in creation order) is inherently sequential --
that IS the Gauss-Seidel property being reproduced -- and carries the whole
``(x, y)`` array so each unit's turn sees every earlier unit's already-moved
position. The inner scan (over candidate obstacles, same creation order)
threads only that one unit's own ``(x, y)`` scalar pair through up to
``n`` escapes. Both loop bounds are the static unit count, so this compiles
to one fixed-trip-count XLA program regardless of how many units are alive
this tick -- masking, not a variable-length loop, is what makes dead slots
and non-participants free of behavioural effect. See
``docs/JAX_REWRITE_PLAN.md`` §1.12 (throughput) and §1.13-adjacent gate 5
(compile time) for why that property matters and
``lanerl_jax/sim/tests/test_collision.py`` / the tier-1 re-measurement for
what this costs and buys.
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .state import Kind

__all__ = ["resolve_collisions"]


def resolve_collisions(x: jax.Array, y: jax.Array, kind: jax.Array,
                       alive: jax.Array, spawn_seq: jax.Array,
                       collision_radius: jax.Array,
                       pathfinding_radius: jax.Array,
                       ghosted: Optional[jax.Array] = None
                       ) -> Tuple[jax.Array, jax.Array]:
    """The server's collision pass for one tick. Returns new ``(x, y)``.

    Args:
      spawn_seq: (N,) creation rank, lower = created earlier. Ties (e.g.
        never-spawned slots, which never participate -- see below) break
        arbitrarily; only relative order among real, live units matters.
      collision_radius: (N,) the ``IsCollidingWith`` TRIGGER radius.
      pathfinding_radius: (N,) the ``GetCircleEscapePoint`` RESOLUTION radius.
      ghosted: (N,) bool, optional. A ghosted unit is dropped from both being
        pushed and being an obstacle -- see the module docstring.
    """
    # `IsCollisionObject`: can be collided WITH. Turrets count; a never-
    # spawned or dead slot does not.
    obstacle = alive & (kind != Kind.NONE)
    # `IsCollisionAffected`: can BE PUSHED. The one place this differs from
    # `obstacle` -- turrets are obstacles but are never affected.
    affected = obstacle & (kind != Kind.TURRET)
    if ghosted is not None:
        obstacle = obstacle & ~ghosted
        affected = affected & ~ghosted

    # True creation order, both loops -- see the module docstring for why
    # there is no separate "inner" order to also reconstruct.
    order = jnp.argsort(spawn_seq)

    def outer_body(carry, i):
        cx, cy = carry
        xi0, yi0 = cx[i], cy[i]
        i_affected = affected[i]
        ri1 = pathfinding_radius[i] + 1.0        # `PathfindingRadius + 1`
        ci = collision_radius[i]

        def inner_body(carry2, j):
            xi, yi = carry2
            dx = cx[j] - xi
            dy = cy[j] - yi
            d = jnp.sqrt(dx * dx + dy * dy)
            trigger = (i_affected & obstacle[j] & (j != i) & (d > 0)
                      & (d < ci + collision_radius[j]))
            safe = jnp.where(d > 0, d, jnp.ones_like(d))
            push = d - ri1 - pathfinding_radius[j]     # negative: overlapping
            xi_next = jnp.where(trigger, xi + dx / safe * push, xi)
            yi_next = jnp.where(trigger, yi + dy / safe * push, yi)
            return (xi_next, yi_next), None

        (xi_f, yi_f), _ = jax.lax.scan(inner_body, (xi0, yi0), order)
        cx = cx.at[i].set(xi_f)
        cy = cy.at[i].set(yi_f)
        return (cx, cy), None

    (x, y), _ = jax.lax.scan(outer_body, (x, y), order)
    return x, y
