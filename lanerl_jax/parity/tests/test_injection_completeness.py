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

#: dump key -> why it is deliberately NOT PARSED by `trace.py`.
#:
#: A separate list from `_WAIVED`, because the chain has two stages and they
#: need different exemptions: published->parsed, then parsed->injected. Putting
#: an unparsed field in `_WAIVED` made the staleness check fire on it, which was
#: correct -- a waiver naming a field the scrape never sees is stale
#: documentation for the stage it claims to cover.
_NOT_PARSED = {
    "aacdraw": "quantised duplicate of aacdbits; kept in the dump so older "
               "corpora stay readable, but aacdbits is what anything new parses",
}

#: dump key -> why it does not have to reach `inject.py`.
#:
#: PRUNED 2026-09-23. This list had 32 entries of which 15 named fields that
#: `inject.py` ACTUALLY CONSUMES -- aacd, aibuffs, aihelp, aiignore, aiwp,
#: damage, id, owner, speed, wpkey, wps, x, xbits, y, ybits. A waiver on a
#: consumed field protects nothing today and silently permits its removal
#: tomorrow, which is the opposite of the point. The dangerous one was `aacd`,
#: waived as "superseded by aacdbits" while being the live FALLBACK for
#: recordings older than 2026-09-21: delete that fallback as dead code and this
#: test would still have passed while every historical corpus injected
#: aa_cooldown = 0. Those 15 are gone, so their consumption is now enforced.
#: `aahelp` is also gone -- it was a typo for `aihelp` and was never a dump key,
#: so the staleness test below could never have flagged it.
_WAIVED = {
    "aadelay": "accumulator; the port needs the REMAINDER, which is aawindupbits",
    "aacast": "cast-time accumulator; not part of the injected state",
    "aagate": "AA-005 diagnostic: the gate CONDITIONS, read but never injected",
    "status": "raw StatusFlags word, a diagnostic beside aagate",
    "aathreshbits": "windup threshold; only needed if the remainder were derived",
    "aadelaybits": "as aadelay, exact",
    "aacastbits": "as aacast, exact",
    "kind": "identity, not state",
    "team": "identity, not state",
    "coll": "collision-cache diagnostic",
    "collbits": "collision-cache diagnostic",
    "cr": "static profile constant, not per-tick state",
    "pr": "static profile constant, not per-tick state",
    "h": "hash field of the canonical row",
    "n": "entity count of the canonical row",
    #: Parsed 2026-09-23 for DIAGNOSTIC reads, not for injection.
    #: `hp`/`mhp`/`mo` duplicate quantities the canonical row already injects;
    #: they are parsed off the per-unit stream so `server_vs_server.py` and
    #: `floor_pool.py` can measure their order-dependence floor -- `hp` and
    #: `move_order` were the only two scored families without one, and so the
    #: only two that could not be classified under `GATE1-004`. `adbits` pinned
    #: `STAT-003`'s exact level-1 AD (78.13500213623047) where the observation
    #: stream's 2 dp rounding could only bound it to [78.135, 78.145).
    "hp": "canonical row already injects it; parsed for the order-floor measure",
    "mhp": "as hp",
    "mo": "canonical row already injects move_order; parsed for the floor measure",
    "adbits": "diagnostic: pinned STAT-003's exact level-1 AD, not injected",
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
    """A waiver for a field that is not parsed, or that IS consumed, is wrong.

    The previous version unioned in a hardcoded list of 16 names that happened
    to contain every phantom, so it could not fire on any of them -- including
    `aahelp`, which was never a dump key at all.

    Two failure modes, both real and both previously invisible:
      * a waiver whose field is not parsed is stale documentation;
      * a waiver whose field IS consumed masks a live site (see `_WAIVED`).
    """
    parsed = _parsed_fields()
    inject_src = _INJECT.read_text()

    # Canonical-row keys are not `AIInternal` fields and are never parsed by
    # the scrape; they are named explicitly rather than folded into a blanket
    # allow-list, so adding one is a visible decision.
    CANONICAL_ROW = {"h", "n", "cr", "pr", "coll", "collbits"}

    stale = sorted(k for k in _WAIVED if k not in parsed and k not in CANONICAL_ROW)
    assert not stale, (
        f"waivers for fields that are not parsed at all: {stale}. Remove them, "
        "or add them to CANONICAL_ROW if they are canonical-row keys.")

    redundant = sorted(
        k for k, a in parsed.items()
        if k in _WAIVED and re.search(rf"internal\.{a}\b|iv\.{a}\b", inject_src))
    assert not redundant, (
        f"these fields are WAIVED but actually consumed: {redundant}. A waiver "
        "on a consumed field protects nothing now and permits its silent "
        "removal later -- drop the waiver so the consumption is enforced.")


def test_published_but_unparsed_is_reported():
    """`trace.py -> inject.py` is only the second half of the chain.

    The title of this module promises "every field the server PUBLISHES must be
    consumed", and the scrape only sees what `trace.py` parses. A field added to
    `LanerlStateDump.cs` and never parsed is invisible to every check above --
    the same bug class as `aihad`, one layer earlier. `aacdraw` is the live
    example: published since 2026-09-19, parsed by nothing.

    The vendored server is not in this repository (see `REPRODUCING_GATES.md`),
    so this SKIPS rather than fails when it is absent, and says which.
    """
    dump = (_ROOT.parent / "lanerl-vendor" / "LoLServer" / "GameServerLib"
            / "Lanerl" / "LanerlStateDump.cs")
    if not dump.exists():
        pytest.skip(f"vendored server absent ({dump}); published-field check "
                    "cannot run from a bare clone")
    published = set(re.findall(r'" (\w+)=" \+', dump.read_text()))
    assert published, "scraped no published fields -- the emit idiom moved"
    parsed = set(_parsed_fields())
    unparsed = sorted(published - parsed - _NOT_PARSED.keys()
                      - {"h", "n", "cr", "pr", "coll", "collbits"})
    assert not unparsed, (
        f"published by the server but parsed by nothing: {unparsed}. Either "
        "parse them in trace.py or waive them with a reason -- an unparsed "
        "field cannot be consumed, and no other check in this file can see it.")
