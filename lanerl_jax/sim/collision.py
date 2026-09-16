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
outer scan (over units, in creation order) is inherently sequential -- that
IS the Gauss-Seidel property being reproduced -- and carries the whole
``(x, y)`` array so each unit's turn sees every earlier unit's already-moved
position.

Why the inner loop is BOUNDED rounds, not a length-``n`` scan
---------------------------------------------------------------
A first version made the inner loop a second `lax.scan` over all ``n``
candidates, in creation order, one at a time -- a direct transcription of
`UpdateCollision`'s `foreach`. It is correct (this is how it was verified),
and it is far too slow: nested at ``n = 66``, that is 4,356 genuinely
SEQUENTIAL scan steps per tick, and a sequential scan's step count does not
shrink under `vmap` the way elementwise work does -- every environment in
the batch pays for all 4,356 steps every tick. Measured (RTX 5080, real
policy in the loop, `lanerl_jax/train/benchmark.py`): throughput fell from
the previously-measured 164x baseline to 30x at best (2,048 envs), well under
gate 4's 50x floor.

The fix exploits a fact the crowding-bucket measurements already established:
almost every unit has zero or one colliding neighbour, and the "3+" bucket is
rare. So instead of visiting all ``n`` candidates one index at a time, each
round of a MUCH shorter, bounded scan does the whole candidate scan
VECTORISED (one O(n) elementwise computation, not n sequential steps) and
extracts just "the earliest-created still-open candidate that currently
overlaps, if any" via `argmin` over a masked key. That candidate's escape is
applied (if there is one), and every candidate at-or-before it in creation
order is marked settled -- correct because the server's own single,
fixed-order pass would have visited those exact candidates first, with this
exact pre-escape position, and (by construction, since none of them was the
argmin) found none of them overlapping either. A round where nothing
overlaps settles every remaining candidate in that same step, so once a
unit's true collisions are exhausted, further rounds are free no-ops rather
than continued sequential cost. This is mathematically the SAME single pass
`UpdateCollision` performs -- not a different, faster-but-approximate
algorithm -- re-expressed so its genuinely sequential part is bounded by how
many escapes a unit actually needs (``MAX_ESCAPES_PER_UNIT``), not by how
many candidates exist to check.

``MAX_ESCAPES_PER_UNIT`` is a **correctness** bound in exactly the sense
``movement_jax.MAX_STEPS_PER_TICK`` is: too small silently truncates a unit's
escapes mid-tick, and that reads as a position bug, not a performance
symptom. 8 is provisional, chosen from precedent (the same number
``sim/movement.py``/``sim/movement_jax.py`` already use for an analogous
"bound an unbounded per-tick loop" problem) rather than a corpus measurement
of the true maximum simultaneous-overlap count this lane ever produces;
:func:`max_escapes_used` exists to make that measurement possible, the same
way ``movement_jax.max_steps_used`` does for waypoints.

Both loop bounds (the outer unit count and the inner round count) are static,
so this still compiles to one fixed-trip-count XLA program regardless of how
many units are alive or how crowded this tick is -- masking, not a variable-
length loop, is what makes dead slots and non-participants free of
behavioural effect. See ``docs/JAX_REWRITE_PLAN.md`` §1.12 (throughput) and
gate 5 (compile time) for why that property matters, and
``lanerl_jax/sim/tests/test_collision.py`` for both the hand-derived
correctness cases and the bounded-vs-exhaustive equivalence check.
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .state import Kind

__all__ = ["resolve_collisions", "MAX_ESCAPES_PER_UNIT", "max_escapes_used"]

#: Bound on how many escapes one unit may apply in a single tick's collision
#: pass. See the module docstring's "why the inner loop is BOUNDED rounds"
#: section -- this is a correctness bound (silent truncation risk), not a
#: performance knob, and 8 is provisional pending a corpus measurement via
#: :func:`max_escapes_used`.
MAX_ESCAPES_PER_UNIT = 8


def _masks(kind, alive, ghosted):
    """``(obstacle, affected)`` -- see the module docstring's "who collides"
    section. Shared by :func:`resolve_collisions` and :func:`max_escapes_used`
    so the two can never quietly disagree about who participates.
    """
    obstacle = alive & (kind != Kind.NONE)
    affected = obstacle & (kind != Kind.TURRET)
    if ghosted is not None:
        obstacle = obstacle & ~ghosted
        affected = affected & ~ghosted
    return obstacle, affected


def _escape_rounds(i, x, y, affected, obstacle, seq_key,
                   collision_radius, pathfinding_radius, rounds: int):
    """Resolve unit ``i``'s own escapes against the CURRENT ``(x, y)``
    (everyone else's positions are read-only here -- only ``i``'s own scalar
    position advances). Returns ``(xi, yi, n_escapes)``.

    One round = one escape, vectorised: see :func:`resolve_collisions`'s
    module docstring for why this is the same single pass
    `UpdateCollision`/`OnCollision` performs, not a different approximation
    of it, and why ``rounds`` is a correctness bound.
    """
    n = x.shape[0]
    idx = jnp.arange(n)
    BIG = jnp.iinfo(jnp.int32).max
    xi0, yi0 = x[i], y[i]
    i_affected = affected[i]
    ri1 = pathfinding_radius[i] + 1.0            # `PathfindingRadius + 1`
    ci = collision_radius[i]
    # Candidates this unit could ever need to escape from this tick --
    # everyone else that is a valid obstacle -- start "not yet settled";
    # everything else (dead slots, itself) is trivially already settled, so a
    # round's vectorised argmin only ever considers real candidates.
    done0 = ~(obstacle & (idx != i))

    def round_body(carry, _):
        xi, yi, done, count = carry
        dx = x - xi
        dy = y - yi
        d = jnp.sqrt(dx * dx + dy * dy)
        trigger = i_affected & ~done & (d > 0) & (d < ci + collision_radius)
        key = jnp.where(trigger, seq_key, BIG)
        m = jnp.argmin(key)
        has = trigger[m]
        # Everyone at or before `m` in creation order is settled this round:
        # those strictly before `m` were just checked (same `d`, this SAME
        # pre-escape position) and did NOT trigger -- exactly what the
        # server's single, fixed left-to-right pass would also find, so they
        # are correctly never revisited. `m` itself is settled by applying
        # its escape.
        cutoff = jnp.where(has, seq_key[m], BIG)
        done = done | (~done & (seq_key <= cutoff))
        safe = jnp.where(d[m] > 0, d[m], jnp.ones_like(d[m]))
        push = d[m] - ri1 - pathfinding_radius[m]    # negative: overlapping
        xi_next = jnp.where(has, xi + dx[m] / safe * push, xi)
        yi_next = jnp.where(has, yi + dy[m] / safe * push, yi)
        return (xi_next, yi_next, done, count + has.astype(jnp.int32)), None

    (xi_f, yi_f, _, n_escapes), _ = jax.lax.scan(
        round_body, (xi0, yi0, done0, jnp.int32(0)), None, length=rounds)
    return xi_f, yi_f, n_escapes


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
    obstacle, affected = _masks(kind, alive, ghosted)

    # True creation order, both loops -- see the module docstring for why
    # there is no separate "inner" order to also reconstruct.
    order = jnp.argsort(spawn_seq)

    n = x.shape[0]
    idx = jnp.arange(n)
    # A combined, TIE-PROOF sort key: `spawn_seq` first, array index as an
    # arbitrary but deterministic tie-break (real units never legitimately
    # share a `spawn_seq`, but nothing here should silently misbehave if two
    # ever do -- e.g. two never-spawned slots, both already excluded via
    # `obstacle`, or a hand-built test state). `n` comfortably bounds the
    # tie-break term below `spawn_seq`'s own stride.
    seq_key = spawn_seq.astype(jnp.int32) * jnp.int32(n + 1) + idx.astype(jnp.int32)

    def outer_body(carry, i):
        cx, cy = carry
        xi_f, yi_f, _ = _escape_rounds(
            i, cx, cy, affected, obstacle, seq_key,
            collision_radius, pathfinding_radius, MAX_ESCAPES_PER_UNIT)
        cx = cx.at[i].set(xi_f)
        cy = cy.at[i].set(yi_f)
        return (cx, cy), None

    (x, y), _ = jax.lax.scan(outer_body, (x, y), order)
    return x, y


def max_escapes_used(x: jax.Array, y: jax.Array, kind: jax.Array,
                     alive: jax.Array, spawn_seq: jax.Array,
                     collision_radius: jax.Array, pathfinding_radius: jax.Array,
                     ghosted: Optional[jax.Array] = None, probe: int = 32
                     ) -> jax.Array:
    """How many escapes a tick would actually apply to each unit, on the
    SAME pre-tick snapshot :func:`resolve_collisions` would use (not a
    multi-tick simulation). Mirrors ``movement_jax.max_steps_used``: use this
    over a real corpus to set :data:`MAX_ESCAPES_PER_UNIT` from data instead
    of precedent. Returns ``(n,)`` counts; if any equals ``probe`` the probe
    itself was too small and the answer is a lower bound.

    Deliberately does not just call :func:`resolve_collisions` with a larger
    bound and diff a position: two DIFFERENT unresolved overlaps can produce
    the same net displacement by coincidence, which would undercount.
    """
    obstacle, affected = _masks(kind, alive, ghosted)
    n = x.shape[0]
    idx = jnp.arange(n)
    seq_key = spawn_seq.astype(jnp.int32) * jnp.int32(n + 1) + idx.astype(jnp.int32)

    def one(i):
        _, _, count = _escape_rounds(
            i, x, y, affected, obstacle, seq_key,
            collision_radius, pathfinding_radius, probe)
        return count

    return jax.vmap(one)(idx)
