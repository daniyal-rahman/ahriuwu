"""JAX navigation on patch-pinned static geometry; all fallbacks fail closed."""
import math
import numbers
from typing import NamedTuple
import jax
import jax.numpy as jnp
from .terrain import is_walkable, StaticTerrain


class FlowRoutes(NamedTuple):
    points: object
    cells: object
    next_node: object  # [goal, source] predecessor of source in goal-rooted tree
    spacing: float
    min_x: float
    min_z: float
    radius: float


def _window_cells(max_radius, cell_size, default=3):
    """Static ``is_walkable`` cell window for query radii up to ``max_radius`` (``default`` if not static)."""
    if isinstance(max_radius, numbers.Real) and isinstance(cell_size, numbers.Real):
        return max(1, math.ceil(max_radius / cell_size))
    return default


def segment_clear(start, end, radius, terrain, *, samples=65, max_length=600., max_radius=None):
    """Swept-capsule check. ``max_radius`` (static) bounds ``radius``; it sizes the cell window of
    each disk test (exact: inflated radii beyond it only occur past ``max_length``, which fails)."""
    start,end=jnp.asarray(start),jnp.asarray(end)
    length=jnp.linalg.norm(end-start)
    cells=3 if max_radius is None else _window_cells(max_radius+max_length/(samples-1)/2,terrain.cell_size)
    # Half-step inflated disks cover the entire swept capsule; failure may be
    # conservative near walls, but sampling cannot tunnel through thin walls.
    p=start[None,:]+jnp.linspace(0.,1.,samples)[:,None]*(end-start)[None,:]
    open_=jax.vmap(lambda q:is_walkable(q[0],q[1],radius+length/(samples-1)/2,terrain,max_radius_cells=cells))(p)
    return (length<=max_length)&jnp.all(open_)


def nearest_node(position, routes: FlowRoutes, terrain, *, check_connection):
    x=jnp.floor((position[0]-routes.min_x)/routes.spacing).astype(jnp.int32)
    z=jnp.floor((position[1]-routes.min_z)/routes.spacing).astype(jnp.int32)
    offsets=jnp.arange(-2,3)
    xx=jnp.broadcast_to(x+offsets[None,:],(5,5)).reshape(-1)
    zz=jnp.broadcast_to(z+offsets[:,None],(5,5)).reshape(-1)
    valid=(xx>=0)&(zz>=0)&(xx<routes.cells.shape[1])&(zz<routes.cells.shape[0])
    ids=routes.cells[jnp.clip(zz,0,routes.cells.shape[0]-1),jnp.clip(xx,0,routes.cells.shape[1]-1)]
    p=routes.points[jnp.maximum(ids,0)]
    if check_connection:
        valid &= jax.vmap(lambda q:segment_clear(position,q,routes.radius,terrain,samples=33,max_length=350.,
                                               max_radius=routes.radius))(p)
    distance=jnp.sum((p-position)**2,axis=1)
    valid &= ids>=0
    choice=jnp.argmin(jnp.where(valid,distance,jnp.inf))
    return ids[choice],jnp.any(valid)


def _steer(position, goal, radius, source, source_ok, routes, terrain):
    """Steering point from a ``source`` node: its successor toward the goal's node when that is
    in sight, else the source itself. Returns ``(point, valid, anchor)``; ``anchor`` is the node
    steered to (the next tick's source)."""
    dest,dest_ok=nearest_node(goal,routes,terrain,check_connection=False)
    nxt=routes.next_node[jnp.maximum(dest,0),jnp.maximum(source,0)]
    source_point=routes.points[jnp.maximum(source,0)]
    next_point=routes.points[jnp.maximum(nxt,0)]
    next_clear=(nxt>=0)&segment_clear(position,next_point,radius,terrain,max_radius=routes.radius)
    point=jnp.where(next_clear,next_point,source_point)
    valid=source_ok&dest_ok&((nxt>=0)|(source==dest))&(radius<=routes.radius)
    return point,valid,jnp.where(next_clear,nxt,source)


def _direct(position, goal, radius, routes, terrain):
    return segment_clear(position,goal,radius,terrain,max_radius=routes.radius)&(radius<=routes.radius)


def route_next(position, goal, radius, routes, terrain):
    """Return one safe local steering point and success; no client parity claim.

    Recomputed from current position as units move. Nonwalkable goals project
    to a graph node; the exact selected route is a simulation approximation.
    ``radius`` above ``routes.radius`` (the baked clearance) fails closed.
    """
    direct=_direct(position,goal,radius,routes,terrain)
    source,source_ok=nearest_node(position,routes,terrain,check_connection=True)
    point,valid,_=_steer(position,goal,radius,source,source_ok,routes,terrain)
    point=jnp.where(direct,goal,jnp.where(valid,point,position))
    return point,direct|valid


def route_follow(position, goal, radius, anchor, routes, terrain):
    """``route_next`` from a cached ``anchor`` node instead of a fresh nearest-node search.

    The anchor (the node the unit last steered to) is kept while it stays in sight, so a tick
    costs three segment checks instead of the 25-candidate connection search. Returns
    ``(point, ok, anchor, replan)``; ``replan``: the anchor is lost (none yet, or out of sight
    after a blink/teleport/push), or reached with no successor in sight, and ``route_replan``
    must pick a new one. Without a direct line
    and an anchor the unit holds position (fail closed), like an invalid ``route_next``.
    """
    direct=_direct(position,goal,radius,routes,terrain)
    a=jnp.maximum(anchor,0)
    seen=(anchor>=0)&segment_clear(position,routes.points[a],radius,terrain,max_radius=routes.radius)
    point,valid,nxt=_steer(position,goal,radius,a,seen,routes,terrain)
    # Standing on the anchor with no successor in sight: this anchor leads nowhere from here.
    stuck=seen&(nxt==a)&(jnp.sum((position-routes.points[a])**2)<1.0)
    point=jnp.where(direct,goal,jnp.where(valid,point,position))
    return point,direct|valid,jnp.where(seen,nxt,-1),~direct&(~seen|stuck)


def route_replan(position, goal, radius, routes, terrain):
    """Full search for a unit without a usable anchor: ``(point, ok, anchor)``. Point and ok equal
    ``route_next`` for a unit with no direct line (the case ``route_follow`` asks for)."""
    source,source_ok=nearest_node(position,routes,terrain,check_connection=True)
    point,valid,nxt=_steer(position,goal,radius,source,source_ok,routes,terrain)
    return jnp.where(valid,point,position),valid,jnp.where(source_ok,nxt,-1)
