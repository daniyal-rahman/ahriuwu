"""A faithful port of .NET's ``PriorityQueue<TElement, TPriority>``.

Why a *specific* heap matters, when any heap would "work"
---------------------------------------------------------
``NavigationGrid.GetPath`` closes cells when it **enqueues** them, not when it
expands them.  Under that rule the frontier order does not merely pick between
equal-cost paths -- it decides which cells ever get considered at all, and
therefore whether a path is found.  Measured on the real server: my first port,
using Python's ``heapq`` with an insertion-order tie-break, returned *no path*
on 4 of 20 lane move orders that the server pathed without trouble, and the
straight-line fallback then sent the champion up to 950 units astray.

So "use any priority queue" is not a free choice here.  Matching the server's
paths requires matching .NET's heap, exactly.

What .NET actually does
-----------------------
``System.Collections.Generic.PriorityQueue`` is a **4-ary** (not binary) min-heap
over a flat array:

* ``Arity = 4``; parent of ``i`` is ``(i - 1) >> 2``; first child of ``i`` is
  ``(i << 2) + 1``.
* ``Enqueue`` appends at ``_size`` and sifts **up**, stopping as soon as the
  parent is **not strictly greater** -- so equal priorities never swap, and the
  earlier-inserted element stays nearer the root.
* ``Dequeue`` takes index 0, then moves the **last** element to the root and
  sifts **down**.  That last-element promotion is what makes the resulting order
  depend on the whole operation history rather than only on priorities.
* ``MoveDown`` picks the minimal child with a **strict** ``<`` comparison
  scanning children in index order, so the *first* minimum wins ties.

None of this is exotic; it is just not what ``heapq`` does (binary, and with a
different tie-break), and the difference is observable through the pathfinder.

Scope
-----
Only what ``GetPath`` uses: ``Enqueue`` and ``TryDequeue``.  Priorities are
floats compared with ``Comparer<float>.Default``.  ``float.NaN`` would compare
inconsistently in both languages; a NaN priority raises here rather than
silently corrupting heap order.
"""
from __future__ import annotations

import math
from typing import Generic, List, Optional, Tuple, TypeVar

__all__ = ["DotNetPriorityQueue"]

T = TypeVar("T")
_ARITY = 4


class DotNetPriorityQueue(Generic[T]):
    """Min-heap with .NET's exact sift semantics. Not a drop-in for ``heapq``."""

    __slots__ = ("_nodes",)

    def __init__(self) -> None:
        self._nodes: List[Tuple[T, float]] = []

    def __len__(self) -> int:
        return len(self._nodes)

    def enqueue(self, element: T, priority: float) -> None:
        if math.isnan(priority):
            raise ValueError(
                "NaN priority: Comparer<float>.Default orders NaN before every "
                "other value in .NET but Python's comparisons make it unordered, "
                "so the two heaps would diverge silently from here on."
            )
        nodes = self._nodes
        nodes.append((element, priority))
        self._move_up(element, priority, len(nodes) - 1)

    def try_dequeue(self) -> Optional[Tuple[T, float]]:
        nodes = self._nodes
        if not nodes:
            return None
        root = nodes[0]
        last = nodes.pop()              # RemoveRootNode: _size-- first
        if nodes:                       # anything left -> promote `last` to the root
            self._move_down(last[0], last[1], 0)
        return root

    # ``MoveUp``: stop at the first parent that is not strictly greater, so equal
    # priorities never swap and the earlier insertion stays nearer the root.
    def _move_up(self, element: T, priority: float, node_index: int) -> None:
        nodes = self._nodes
        while node_index > 0:
            parent_index = (node_index - 1) >> 2
            parent = nodes[parent_index]
            if priority < parent[1]:
                nodes[node_index] = parent
                node_index = parent_index
            else:
                break
        nodes[node_index] = (element, priority)

    # ``MoveDown``: minimal child by strict ``<`` scanning in index order, so a
    # tie keeps the lowest index.
    def _move_down(self, element: T, priority: float, node_index: int) -> None:
        nodes = self._nodes
        size = len(nodes)
        while True:
            first_child = node_index * _ARITY + 1
            if first_child >= size:
                break
            min_index = first_child
            min_priority = nodes[first_child][1]
            last_child = min(first_child + _ARITY, size)
            for i in range(first_child + 1, last_child):
                if nodes[i][1] < min_priority:
                    min_index = i
                    min_priority = nodes[i][1]
            if min_priority < priority:
                nodes[node_index] = nodes[min_index]
                node_index = min_index
            else:
                break
        nodes[node_index] = (element, priority)
