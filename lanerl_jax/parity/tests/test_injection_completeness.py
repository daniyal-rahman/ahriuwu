"""
Every field the server publishes must be consumed, or explicitly waived.

WHY THIS TEST EXISTS
--------------------
On 2026-09-22 three separate gate-1 residuals turned out to have the same
cause, and it was not a defect in the simulator:

    field        published   parsed by trace.py   consumed by inject.py
    aihad            yes            yes                   NO
    aacdbits         yes            yes                   NO
    aawindup*        yes            yes                   NO

`aihad` is `LaneMinionAI.hadTarget`, a one-tick latch. `step.py` substituted
`target >= 0` for it, which disagrees on 556 of 395,486 scored LaneMinion
unit-ticks. Injecting the published value moved three residual families at
once: `target` 522 -> 393, `move_order` 354 -> 267, `waypoints` 317 -> 247.

`aacdbits` is the UNCLAMPED auto-attack cooldown. `aacd` publishes
`Q(Math.Max(0f, remaining))`, so a still-positive cooldown within a rounding
step of the gate reads as a flat 0 and the simulator swings a tick early. Of
the 122 `sim fires early` rows in one window, 122 sat on that clamp and 121
had a strictly positive exact value. The count of rows where the PORT was
actually wrong was zero.

That is the point. A field the server publishes and the injector drops is not
a port defect -- it is the harness feeding the simulator a state the server
never had, and then scoring the difference against the simulator. It is
invisible in every residual count by construction, because the residual is
where you look for port bugs and this is not one.

So: this is a completeness check, not a behaviour check. It fails when
somebody adds a dump field and forgets to consume it, which is exactly how
all three of the above happened.

WAIVERS
-------
Not every published field is state. Diagnostics (`aagate`, `status`), derived
duplicates (`aacd` beside `aacdbits`) and identity/debug fields are waived
BY NAME with a reason, so adding a waiver is a visible decision rather than
a silent omission.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_TRACE = _ROOT / "lanerl_jax/parity/trace.py"
_INJECT = _ROOT / "lanerl_jax/parity/inject.py"

#: dump key -> why it does not have to reach `inject.py`.
_WAIVED = {
    "aacd": "superseded by `aacdbits`, which is the same quantity unclamped",
    "aadelay": "accumulator; the port needs the REMAINDER, which is aawindupbits",
    "aacast": "cast-time accumulator; not part of the injected state",
    "aacdraw": "quantised duplicate of aacdbits, kept for historical corpora",
    "aagate": "AA-005 diagnostic: the gate CONDITIONS, read but never injected",
    "status": "raw StatusFlags word, a diagnostic beside aagate",
    "aathreshbits": "windup threshold; only needed if the remainder were derived",
    "aadelaybits": "as aadelay, exact",
    "aacastbits": "as aacast, exact",
    "aahelp": "call-for-help map, injected via `help` under a different name",
    "id": "identity, not state",
    "kind": "identity, not state",
    "team": "identity, not state",
    "owner": "missile identity, not unit state",
    "coll": "collision-cache diagnostic",
    "collbits": "collision-cache diagnostic",
    "cr": "static profile constant, not per-tick state",
    "pr": "static profile constant, not per-tick state",
    "h": "hash field of the canonical row",
    "n": "entity count of the canonical row",
    "speed": "read from the profile table, not injected per unit",
    "damage": "read from the profile table, not injected per unit",
    "x": "position, injected via xbits where available",
    "y": "position, injected via ybits where available",
    "xbits": "position bits, consumed through the position path",
    "ybits": "position bits, consumed through the position path",
    "wps": "waypoint list, consumed through `waypoints`",
    "wpkey": "waypoint cursor, consumed through `waypoints`",
    "aiwp": "lane waypoint cursor, consumed via lane_waypoint_key",
    "aibuffs": "buff phase, consumed via buffs_phase",
    "aiignore": "ignore list, consumed via `ignored`",
    "aihelp": "call-for-help map, consumed via `help`",
}


def _parsed_fields() -> dict[str, str]:
    """dump key -> the `AIInternal` attribute `trace.py` stores it in.

    Parsed with `ast`, not a regex. The first version of this scrape was
    line-bounded and therefore MISSED every field whose parse spans two
    lines -- including `aacdbits`, one of the two bugs this test exists to
    catch. It passed anyway, because a completeness check that cannot see a
    field reports it as complete. The negative control below is what caught
    that, and it is the reason this function is not a regex.
    """
    tree = ast.parse(_TRACE.read_text())
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg is None:
            continue
        for sub in ast.walk(node.value):
            key = None
            # values["x"]
            if (isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name) and sub.value.id == "values"
                    and isinstance(sub.slice, ast.Constant)
                    and isinstance(sub.slice.value, str)):
                key = sub.slice.value
            # values.get("x", ...)
            elif (isinstance(sub, ast.Call)
                  and isinstance(sub.func, ast.Attribute) and sub.func.attr == "get"
                  and isinstance(sub.func.value, ast.Name)
                  and sub.func.value.id == "values"
                  and sub.args and isinstance(sub.args[0], ast.Constant)
                  and isinstance(sub.args[0].value, str)):
                key = sub.args[0].value
            if key is not None:
                out.setdefault(key, node.arg)
    return out


def test_every_parsed_field_is_injected_or_waived():
    parsed = _parsed_fields()
    assert parsed, "parsed nothing -- the trace.py scrape broke, not the invariant"
    inject_src = _INJECT.read_text()

    orphans = []
    for key, attr in sorted(parsed.items()):
        if key in _WAIVED:
            continue
        if re.search(rf"internal\.{attr}\b|iv\.{attr}\b", inject_src):
            continue
        orphans.append(f"  {key:<16} -> AIInternal.{attr}")

    assert not orphans, (
        "These fields are published by the server and parsed by trace.py, but "
        "never reach inject.py:\n" + "\n".join(orphans) + "\n\n"
        "The simulator is therefore started from a state the server never had, "
        "and the difference is scored against the simulator. This is how "
        "`aihad` and `aacdbits` each hid a multi-hundred-row residual. Either "
        "consume the field in inject.py, or add it to _WAIVED with the reason "
        "it is not state."
    )


def test_waivers_are_live():
    """A waiver for a field that no longer exists is stale documentation."""
    parsed = _parsed_fields()
    known = set(parsed) | {
        # canonical-row keys, not part of the AIInternal scrape
        "h", "n", "cr", "pr", "aacdraw", "coll", "collbits", "wps", "wpkey",
        "status", "aagate", "aathreshbits", "aadelaybits", "aacastbits",
        "aahelp", "aibuffs",
    }
    stale = sorted(k for k in _WAIVED if k not in known)
    assert not stale, (
        f"waivers for fields that are no longer parsed: {stale}. "
        "Remove them so the waiver list stays a statement about the present."
    )


def test_scrape_would_catch_the_bugs_it_was_written_for():
    """Negative control. A completeness check that cannot fail is decoration.

    Rebuild the PRE-FIX injector by deleting the two consumption sites added
    on 2026-09-22, and assert both fields show up as orphans. The first
    version of `_parsed_fields` failed exactly this and passed everything
    else.
    """
    parsed = _parsed_fields()
    for key in ("aihad", "aacdbits"):
        assert key in parsed, f"{key} is not even parsed -- the scrape is blind"
    pre_fix = (_INJECT.read_text()
               .replace("internal.had_target", "_gone_")
               .replace("internal.aa_cooldown_bits", "_gone_"))
    orphans = {
        key for key, attr in parsed.items()
        if key not in _WAIVED
        and not re.search(rf"internal\.{attr}\b|iv\.{attr}\b", pre_fix)
    }
    assert {"aihad", "aacdbits"} <= orphans, (
        f"the scrape would NOT have caught {sorted({'aihad','aacdbits'} - orphans)}; "
        "this test cannot detect the class of bug it exists for"
    )
