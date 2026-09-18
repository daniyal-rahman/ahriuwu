"""The server's navigation grid, and a faithful port of its pathfinder.

Why port the A* at all, when the JAX sim will not run one
---------------------------------------------------------
Because the JAX sim needs a *table*, and a table is only as good as the thing it
is baked from.  The plan's design is to precompute next-hop directions over the
lane corridor so pathing on device is a gather rather than a search.  Bake that
table from the wrong search and every champion in the JAX sim walks a slightly
different line from the one the server walks, which shows up as a position
divergence nobody can attribute.

So this module is the reference: an exact-as-practical port of
``NavigationGrid.GetPath``, run on the host, used to (a) bake the table and
(b) measure how far a cheaper table is from the truth.

What the server's pathfinder actually is
----------------------------------------
Read off ``GameServerLib/Content/Navigation/NavigationGrid.cs`` and then checked
against the Map1 grid file itself:

* **Step cost is a flat ``+1`` per cell** (``GetPath``: ``cost = currentCost + 1
  + ArrivalCost + AdditionalCost``).  All three of ``ArrivalCost``,
  ``AdditionalCost`` and ``Heuristic`` are **zero for all 86,142 cells** of
  Map1 -- measured, not assumed -- so the metric is plain step count.  A
  diagonal costs the same as an orthogonal step: this is **Chebyshev distance,
  not Euclidean**, and a shortest-path table built with ``sqrt(2)`` diagonals
  would be a different pathfinder.
* **Priority is ``cost + euclidean(neighbour_centre, goal)``** in nav-grid (cell)
  units -- an admissible-looking heuristic, but see the next point.
* **Cells are closed when *enqueued*, not when expanded.**  ``GetPath`` calls
  ``closedList.Add(neighborCell.ID)`` immediately after enqueueing.  Every cell
  is therefore reached exactly once, by whichever path got there first, and the
  result is **not guaranteed to be a shortest path**.  This is the single most
  important fact in this module: the server does not compute optimal paths, so
  "optimal" is the wrong thing to bake.
* **Neighbours are all 9 cells** (``dirY`` outer, ``dirX`` inner, each -1..1,
  including the cell itself, which is already closed and so skipped).
* **Each edge is gated by ``CastCircle``**, a swept-circle walkability test at
  the unit's pathfinding radius (Garen: ``PathfindingCollisionRadius`` 35).
  Passability is therefore **radius-dependent**: a gap two cells wide is open
  to a point and closed to a champion.
* ``IsWalkable`` excludes **both** ``NOT_PASSABLE`` *and* ``SEE_THROUGH``.
  Missing the second costs 516 cells on Map1 (53,651 vs 53,135) -- small, and
  exactly the kind of small that produces an unexplained divergence months later.
* The returned waypoint list is ``[from] + cell centres + [to]`` after
  ``SmoothPath`` collapses runs of cells that have clear line of sight.

Known deviation
---------------
.NET's ``PriorityQueue<TElement,TPriority>`` does not specify a tie-break, and
its heap order is an implementation detail we cannot read from here.  This port
breaks ties by insertion order.  Where two frontier cells have exactly equal
priority the two implementations may expand in different orders and return
different (equal-cost, or under closed-on-enqueue possibly unequal-cost) paths.
That is a measurement to take against recorded server waypoints, not something
to assume away -- see ``docs/JAX_REWRITE_PLAN.md`` risk R4.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .dotnet_pq import DotNetPriorityQueue

_f32 = np.float32


def _dist32(ax: float, ay: float, bx: float, by: float) -> float:
    """``Vector2.Distance`` in **32-bit float**, as the server computes it.

    This matters more than it looks. ``GetPath`` is closed-on-enqueue, so the
    frontier order decides which cells are ever considered; a priority that
    differs in the last bit can reorder two expansions and send the search down
    a different corridor. Python floats are doubles, so computing the heuristic
    in double precision is a *different pathfinder* from the server's, not a
    more accurate one.

    ``System.Numerics.Vector2`` accumulates in float32 throughout:
    ``MathF.Sqrt(dx*dx + dy*dy)`` with every intermediate a float.
    """
    dx = _f32(ax) - _f32(bx)
    dy = _f32(ay) - _f32(by)
    return float(np.sqrt(_f32(dx * dx + dy * dy)))

__all__ = [
    "NavigationGridCellFlags",
    "NavGrid",
    "GridPath",
    "DEFAULT_NGRID",
    "GAREN_PATHFINDING_RADIUS",
]

from .paths import ngrid_path

#: Resolved from this file, not typed -- see ``paths.py``.
DEFAULT_NGRID = ngrid_path(1)

#: ``Garen.json`` -> ``PathfindingCollisionRadius``.
GAREN_PATHFINDING_RADIUS = 35.0

_CELL_BYTES = 56
_FLAGS_OFFSET = 50


class NavigationGridCellFlags:
    HAS_GRASS = 0x1
    NOT_PASSABLE = 0x2
    SEE_THROUGH = 0x40
    HAS_GLOBAL_VISION = 0x100


@dataclass(frozen=True)
class GridPath:
    """The unsmoothed cell itinerary produced by ``GetPath``.

    This is deliberately separate from :meth:`NavGrid.get_path`: a route-table
    bake needs the adjacent cell hops *before* ``SmoothPath`` collapses them,
    while normal reference callers want the server-visible world waypoints.
    ``goal`` is already the server's terrain-clamped destination.
    """

    source: Tuple[float, float]
    goal: Tuple[float, float]
    cells: Tuple[Tuple[int, int], ...]


def _short(v: float) -> int:
    """C#'s ``(short)`` cast: truncation toward zero, not ``floor``.

    These differ for negatives -- ``(short)(-0.5) == 0`` but ``floor(-0.5) ==
    -1`` -- and ``GetAllCellsInRange`` casts ``origin - radius``, which goes
    negative near the map edge. Using ``floor`` there would enumerate a row of
    cells the server never looks at.
    """
    return int(v)


@dataclass(slots=True)
class NavGrid:
    flags: np.ndarray          # (cy, cx) uint16
    cell_size: float
    min_grid: Tuple[float, float, float]
    max_grid: Tuple[float, float, float]

    # ---------------------------------------------------------------- load --
    @classmethod
    def load(cls, path: Path | str = DEFAULT_NGRID) -> "NavGrid":
        d = Path(path).read_bytes()
        off = 0
        major = d[off]
        off += 1
        if major != 2:
            off += 2
        if major not in (2, 3, 5):
            raise NotImplementedError(
                f"navgrid version {major}: only the ReadVersion5 layout is ported"
            )
        min_grid = struct.unpack_from("<3f", d, off); off += 12
        max_grid = struct.unpack_from("<3f", d, off); off += 12
        cell_size = struct.unpack_from("<f", d, off)[0]; off += 4
        cx, cy = struct.unpack_from("<II", d, off); off += 8
        raw = np.frombuffer(d, np.uint8, count=cx * cy * _CELL_BYTES, offset=off)
        raw = raw.reshape(cx * cy, _CELL_BYTES)
        flags = raw[:, _FLAGS_OFFSET:_FLAGS_OFFSET + 2].copy().view(np.uint16)
        return cls(flags=flags.reshape(cy, cx), cell_size=cell_size,
                   min_grid=min_grid, max_grid=max_grid)

    @property
    def cell_count_x(self) -> int:
        return self.flags.shape[1]

    @property
    def cell_count_y(self) -> int:
        return self.flags.shape[0]

    # ----------------------------------------------------------- translate --
    def to_nav(self, x: float, y: float) -> Tuple[float, float]:
        """World -> cell-space (``TranslateToNavGrid``). ``MinGridPosition.Z``
        is the y origin: the server stores the grid as XZ."""
        return ((x - self.min_grid[0]) / self.cell_size,
                (y - self.min_grid[2]) / self.cell_size)

    def from_nav(self, cx: float, cy: float) -> Tuple[float, float]:
        return (cx * self.cell_size + self.min_grid[0],
                cy * self.cell_size + self.min_grid[2])

    def cell_center_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """``TranslateFromNavGrid(locator)`` -- the centre, not the corner."""
        wx, wy = self.from_nav(ix, iy)
        return wx + 0.5 * self.cell_size, wy + 0.5 * self.cell_size

    # ------------------------------------------------------------ topology --
    def in_bounds(self, ix: int, iy: int) -> bool:
        """``GetCell(short,short)``'s own bounds test, quirk included.

        The server tests ``x > CellCountX`` (not ``>=``), so ``x ==
        CellCountX`` falls through to ``index = y*CellCountX + x``, which is the
        first cell of the *next row* -- a real off-by-one that silently returns
        a wrong cell at the right-hand map edge. It is caught only when
        ``index >= Cells.Length``. Reproduced rather than fixed: this module is
        a reference for what the server does, and a "corrected" reference is
        not a reference.
        """
        if ix < 0 or ix > self.cell_count_x or iy < 0 or iy > self.cell_count_y:
            return False
        return iy * self.cell_count_x + ix < self.flags.size

    def _flags_at(self, ix: int, iy: int) -> Optional[int]:
        if not self.in_bounds(ix, iy):
            return None
        idx = iy * self.cell_count_x + ix
        return int(self.flags.reshape(-1)[idx])

    def is_walkable_cell(self, ix: int, iy: int) -> bool:
        """``IsWalkable(NavigationGridCell)`` -- NOT_PASSABLE *and* SEE_THROUGH."""
        f = self._flags_at(ix, iy)
        if f is None:
            return False
        return not (f & NavigationGridCellFlags.NOT_PASSABLE) and \
               not (f & NavigationGridCellFlags.SEE_THROUGH)

    def walkable_mask(self) -> np.ndarray:
        """(cy, cx) bool, the pathing mask. Static; safe to share on device."""
        return ((self.flags & NavigationGridCellFlags.NOT_PASSABLE) == 0) & \
               ((self.flags & NavigationGridCellFlags.SEE_THROUGH) == 0)

    # -------------------------------------------------------- enumerations --
    def cells_in_range(self, ox: float, oy: float, radius: float) -> Iterator[Tuple[int, int]]:
        """``GetAllCellsInRange`` in cell space (``translate=False``).

        ``radius`` arrives in world units and is divided by ``CellSize``, and
        the rectangle test is distance-to-the-*cell-square*, not to its centre.
        """
        r = radius / self.cell_size
        fx, lx = _short(ox - r), _short(ox + r)
        fy, ly = _short(oy - r), _short(oy + r)
        r2 = r * r
        for ix in range(fx, lx + 1):
            for iy in range(fy, ly + 1):
                # DistanceSquaredToRectangle(centre=(ix+.5, iy+.5), 1x1, origin)
                dx = max(abs(ox - (ix + 0.5)) - 0.5, 0.0)
                dy = max(abs(oy - (iy + 0.5)) - 0.5, 0.0)
                if dx * dx + dy * dy <= r2 and self.in_bounds(ix, iy):
                    yield ix, iy

    def cells_in_line(self, x0: float, y0: float, x1: float, y1: float
                      ) -> Iterator[Tuple[int, int]]:
        """``GetAllCellsInLine`` -- the server's integer-error line walk.

        Two things this got wrong before, found by rereading the C# source
        (``NavigationGrid.cs``) rather than the obvious Bresenham sketch:

        * **``error == 0`` is a third branch, not folded into the ``else``.**
          The server's loop is ``if (error > 0) ... else if (error < 0) ...
          else { yield BOTH (x+x_inc, y) and (x, y+y_inc); step x AND y;
          n-- an extra time }``. Landing exactly on a lattice corner is not a
          rare event here: ``CastCircle`` always calls this on a pair of
          parallel offset lines, and the offset itself is often small and
          fraction-of-a-cell in size, so the walk revisits corner geometry a
          lot. Collapsing the tie into "advance x only" silently drops the
          diagonal neighbour the server also checks, which can turn a real
          obstruction into a false "clear" (or vice versa).
        * **Cells are not bounds-filtered here.** ``GetAllCellsInLine`` calls
          ``GetCell`` with no null check and ``CastCircle`` immediately tests
          ``IsWalkable(cell)``, which is ``false`` for a null cell -- so
          stepping off the edge of the grid *blocks* the cast. Filtering
          out-of-bounds cells here (matching ``GetAllCellsInRange``, which
          DOES null-check) silently turns that into "nothing in the way".
          ``is_walkable_cell`` already returns ``False`` off-grid, so not
          filtering reproduces the null-blocks behaviour for free -- see
          ``cast_circle``, which is the only caller.
        """
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        ix, iy = int(math.floor(x0)), int(math.floor(y0))
        n = 1
        if dx == 0:
            x_inc, error = 0, math.inf
        elif x1 > x0:
            x_inc = 1
            n += int(math.floor(x1)) - ix
            error = (math.floor(x0) + 1 - x0) * dy
        else:
            x_inc = -1
            n += ix - int(math.floor(x1))
            error = (x0 - math.floor(x0)) * dy
        if dy == 0:
            y_inc, error = 0, -math.inf
        elif y1 > y0:
            y_inc = 1
            n += int(math.floor(y1)) - iy
            error -= (math.floor(y0) + 1 - y0) * dx
        else:
            y_inc = -1
            n += iy - int(math.floor(y1))
            error -= (y0 - math.floor(y0)) * dx
        while n > 0:
            yield ix, iy
            if error > 0:
                iy += y_inc
                error -= dx
            elif error < 0:
                ix += x_inc
                error += dy
            else:
                yield ix + x_inc, iy
                yield ix, iy + y_inc
                ix += x_inc
                iy += y_inc
                error += dy - dx
                n -= 1
            n -= 1

    # ------------------------------------------------------------ casting ---
    def cast_circle(self, ox: float, oy: float, dx_: float, dy_: float,
                    radius: float) -> bool:
        """``CastCircle`` in cell space. **True means BLOCKED.**

        Note the polarity: the server uses this as "is something in the way",
        so ``GetPath`` closes a neighbour when it returns *true*.
        """
        tradius = radius / self.cell_size
        vx, vy = dx_ - ox, dy_ - oy
        L = math.hypot(vx, vy)
        if L == 0.0:
            px = py = 0.0
        else:
            # Normalized().Perpendicular() * tradius
            px, py = -(vy / L) * tradius, (vx / L) * tradius

        cells: List[Tuple[int, int]] = []
        cells += list(self.cells_in_range(ox, oy, radius))
        cells += list(self.cells_in_range(dx_, dy_, radius))
        cells += list(self.cells_in_line(ox + px, oy + py, dx_ + px, dy_ + py))
        cells += list(self.cells_in_line(ox - px, oy - py, dx_ - px, dy_ - py))

        min_y = int(min(oy, dy_) - tradius) - 1
        max_y = int(max(oy, dy_) + tradius) + 1
        count_y = max_y - min_y + 1
        lo = [0] * count_y
        hi = [0] * count_y
        seen = [False] * count_y

        for ix, iy in cells:
            if not self.is_walkable_cell(ix, iy):
                return True
            k = iy - min_y
            if not (0 <= k < count_y):
                continue
            if not seen[k]:
                seen[k] = True
                lo[k] = hi[k] = ix
            else:
                lo[k] = min(lo[k], ix)
                hi[k] = max(hi[k], ix)

        for k in range(count_y):
            for ix in range(lo[k] + 1, hi[k]):
                if not self.is_walkable_cell(ix, min_y + k):
                    return True
        return False

    def closest_terrain_exit(self, x: float, y: float, radius: float = 0.0,
                             max_iter: int = 4096) -> Tuple[float, float]:
        """``GetClosestTerrainExit`` -- a **cumulative drifting spiral**.

        Worth reading carefully, because the obvious implementation is wrong and
        I wrote the obvious one first::

            double angle = Math.PI / 4;
            for (int r = 1; !IsWalkable(location, distanceThreshold); r++) {
                location.X += r * (float)Math.Cos(angle);
                location.Y += r * (float)Math.Sin(angle);
                angle += Math.PI / 4;
            }

        The step is **added to ``location``**, not sampled around the original
        point, so the search point walks away in a widening spiral and the
        answer depends on the whole trajectory. Steps are in **world units**
        (1, 2, 3, ... not cell multiples) and the angle advances by a fixed
        45 degrees.

        A fixed-centre polar search returns a *nearer* exit, which sounds
        better and is wrong: measured against the server, goals that landed in
        terrain then resolved somewhere else entirely and champion trajectories
        diverged by up to 317 units while agreeing perfectly whenever the goal
        happened to be walkable. Reproducing the drift is the fix.

        ``max_iter`` bounds a loop the server leaves unbounded (it would hang on
        a fully unwalkable map); reaching it raises rather than returning a
        point that is not actually an exit.
        """
        angle = math.pi / 4.0
        r = 1
        while not self.is_walkable_world(x, y, radius):
            x += r * math.cos(angle)
            y += r * math.sin(angle)
            angle += math.pi / 4.0
            r += 1
            if r > max_iter:
                raise RuntimeError(
                    f"GetClosestTerrainExit did not converge in {max_iter} steps "
                    f"from ({x:.1f}, {y:.1f}) at radius {radius}"
                )
        return x, y

    def is_walkable_world(self, x: float, y: float, radius: float = 0.0) -> bool:
        cx, cy = self.to_nav(x, y)
        if radius == 0.0:
            return self.is_walkable_cell(_short(cx), _short(cy))
        return all(self.is_walkable_cell(ix, iy)
                   for ix, iy in self.cells_in_range(cx, cy, radius))

    # -------------------------------------------------------------- A* -----
    def get_cell_path(self, from_xy: Tuple[float, float], to_xy: Tuple[float, float],
                      radius: float = 0.0, max_expansions: int = 200_000
                      ) -> Optional[GridPath]:
        """Port of ``GetPath`` through A*, before its ``SmoothPath`` pass.

        The returned cells include source and destination.  This low-level
        form is for deterministic offline route-table bakes; gameplay callers
        should use :meth:`get_path`, whose output is the server's smoothed
        world-waypoint sequence.
        """
        if from_xy == to_xy:
            return None
        fnx, fny = self.to_nav(*from_xy)
        cell_from = (_short(fnx), _short(fny))
        to_xy = self.closest_terrain_exit(to_xy[0], to_xy[1], radius)
        tnx, tny = self.to_nav(*to_xy)
        cell_to = (_short(tnx), _short(tny))
        if not self.in_bounds(*cell_from) or not self.in_bounds(*cell_to):
            return None
        if cell_from == cell_to:
            return GridPath(from_xy, to_xy, (cell_from, cell_to))

        # closed-on-ENQUEUE, exactly as the server does it, and on .NET's own
        # 4-ary heap -- with this pathfinder the frontier order decides not only
        # WHICH path is found but WHETHER one is (see dotnet_pq).
        closed = {cell_from}
        pq: DotNetPriorityQueue = DotNetPriorityQueue()
        pq.enqueue((_f32(0.0), [cell_from]), _dist32(fnx, fny, tnx, tny))

        path: Optional[List[Tuple[int, int]]] = None
        expansions = 0
        while True:
            item = pq.try_dequeue()
            if item is None:
                return None                      # "no solution", as the server does
            (cost, path), _ = item
            expansions += 1
            if expansions > max_expansions:
                raise RuntimeError(
                    f"A* exceeded {max_expansions} expansions from {from_xy} to "
                    f"{to_xy}; the server's loop is unbounded, so a real path may "
                    "exist -- raise the bound rather than treating this as no-path"
                )
            cell = path[-1]
            if cell == cell_to:
                break
            # GetCellNeighbors: dirY outer, dirX inner, both -1..1, self included
            for diry in (-1, 0, 1):
                for dirx in (-1, 0, 1):
                    nb = (cell[0] + dirx, cell[1] + diry)
                    if not self.in_bounds(*nb) or nb in closed:
                        continue
                    if nb != cell_to:
                        nbx, nby = nb[0] + 0.5, nb[1] + 0.5
                        if cell == cell_from:
                            cx_, cy_ = fnx, fny
                        else:
                            cx_, cy_ = cell[0] + 0.5, cell[1] + 0.5
                        if self.cast_circle(cx_, cy_, nbx, nby, radius):
                            closed.add(nb)
                            continue
                        gx, gy = nbx, nby
                    else:
                        gx, gy = tnx, tny
                    # ArrivalCost/AdditionalCost are 0 for all Map1 cells, and
                    # every term here is a C# float on the server.
                    ncost = _f32(cost + _f32(1.0))
                    pq.enqueue((ncost, path + [nb]),
                               float(_f32(ncost + _f32(_dist32(gx, gy, tnx, tny)))))
                    closed.add(nb)

        if path is None:
            return None
        return GridPath(from_xy, to_xy, tuple(path))

    def get_path(self, from_xy: Tuple[float, float], to_xy: Tuple[float, float],
                 radius: float = 0.0, max_expansions: int = 200_000
                 ) -> Optional[List[Tuple[float, float]]]:
        """Port of ``NavigationGrid.GetPath``. World in, world waypoints out.

        Returns ``None`` for the server's no-solution cases, including
        ``from == to``.  ``get_cell_path`` owns the A* implementation so an
        offline bake can consume its unsmoothed adjacent hops without
        accidentally substituting a different search.
        """
        route = self.get_cell_path(from_xy, to_xy, radius, max_expansions)
        if route is None:
            return None
        path = list(route.cells)
        self._smooth(path, radius)
        out = [route.source]
        for ix, iy in path[1:-1]:
            out.append(self.cell_center_world(ix, iy))
        out.append(route.goal)
        return out

    def _smooth(self, path: List[Tuple[int, int]], radius: float) -> None:
        """``SmoothPath``, in place: drop cells the previous kept one can see."""
        if len(path) < 3:
            return
        j = 0
        for i in range(2, len(path)):
            ax, ay = path[j][0] + 0.5, path[j][1] + 0.5
            bx, by = path[i][0] + 0.5, path[i][1] + 0.5
            if self.cast_circle(ax, ay, bx, by, radius):
                j += 1
                path[j] = path[i - 1]
        j += 1
        path[j] = path[-1]
        del path[j + 1:]
