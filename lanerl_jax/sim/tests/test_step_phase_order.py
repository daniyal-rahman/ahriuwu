"""`STRUCT-006`: ``step.py``'s header documents the phases of ``tick()``, and
the ``# ---- N.`` markers in ``tick()`` appear in exactly that order.

The header once said buffs, regen, casts, missiles, waves and gold/XP were
"not yet wired in" long after all of them were, and its implied order was
never checked. `AA-001` was an order bug; a documented order nobody compares
to the code is how the next one hides. This is a source-text test on purpose:
it runs in milliseconds and needs no JAX.
"""
from __future__ import annotations

import inspect
import re

from lanerl_jax.sim import step

_DOC_ITEM = re.compile(r"^\s{2,4}(\d+)\. (.+?)\s*$")
_MARKER = re.compile(r"^    # ---- (\d+)\. (.+?) -*\s*$")


def _documented():
    doc = step.__doc__
    section = doc.split("The phases of ``tick()``", 1)[1]
    section = section.split("Known order deviations", 1)[0]
    return [(int(m.group(1)), m.group(2))
            for m in map(_DOC_ITEM.match, section.splitlines()) if m]


def _in_code():
    src = inspect.getsource(step.tick)
    return [(int(m.group(1)), m.group(2))
            for m in map(_MARKER.match, src.splitlines()) if m]


def test_the_documented_phases_are_numbered_one_to_n():
    doc = _documented()
    assert doc, "the phase list is missing from step.py's docstring"
    assert [n for n, _ in doc] == list(range(1, len(doc) + 1))


def test_every_phase_marker_in_tick_is_numbered():
    """A ``# ---- `` marker without a number is a phase the list cannot
    name, which is how the header fell behind the code last time."""
    src = inspect.getsource(step.tick)
    bare = [ln for ln in src.splitlines()
            if ln.startswith("    # ---- ") and not _MARKER.match(ln)]
    assert not bare, f"unnumbered phase markers in tick(): {bare}"


def test_tick_runs_the_phases_in_the_documented_order():
    assert _in_code() == _documented()
