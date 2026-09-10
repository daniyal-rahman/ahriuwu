"""Leak audit: prove the actor observation is deployable from a screenshot.

The whole point of the observation design is that a policy trained against the
server can be dropped onto a screen-reading client without a distribution
shift.  That only holds if *nothing* in the actor observation depends on state
a screenshot cannot produce: enemy gold, enemy experience, enemy exact HP,
enemy cooldowns, or anything currently behind fog of war.

This module checks that three ways, and fails loudly:

**Schema completeness.**  The other two checks can only guard against fields
they know exist.  The original design was reactive -- a closed allowlist of
attribute names that had already leaked, plus hand-written probes for specific
fields -- so a server field nobody had thought of was identical in both probe
frames, ``np.array_equal`` was True, and the audit reported ok *vacuously*.
:data:`lanerl_rl.frame.WIRE_FIELDS` inverts that: every key
``LanerlControl.BuildObservation`` can emit must be classified as consumed (and
audited) or explicitly ignored (with a reason).  Three checks hold the registry
to three independent sources of truth -- the C# emitter's own source, an
instrumented run of ``decode_frame``, and a differential probe *generated from
the registry* -- and an unrecognised key fails all of them.

**Static (AST).**  ``obs.py`` declares which of its functions are on the actor
path (``ACTOR_PATH_FUNCTIONS``) and which are privileged
(``PRIVILEGED_PATH_FUNCTIONS``).  We parse the module and assert that no actor
function reads a server-only attribute, and that no actor function calls a
privileged one.  We also scan the field-name registries in ``constants.py`` for
names that describe server-only quantities.

**Differential (dynamic).**  Far stronger than any name check: build the same
observation twice from frames that differ *only* in server-only quantities, and
assert the actor arrays are bit-identical while the privileged arrays are not.
The probes:

* ``enemy gold / xp / cs`` set to absurd values,
* ``enemy HP`` jittered below health-bar resolution,
* a fogged unit teleported across the map,
* no *valid* slot ever aliases a fogged unit's position,
* the **enemy's remaining cooldowns rewritten while they are fogged** -- the
  agent may only ever learn about an enemy ability from a cast it witnessed,
* the enemy's cooldowns jittered *below the cast-detection threshold* while
  they are in plain sight -- seeing a champion is not reading their cooldown
  numbers off the server.

The last two exist because ``obs.py`` now carries an enemy ability book
(``EnemyAbilityIntel``).  That book is the most plausible new leak in the
package, so it gets the sharpest differential probe: the only thing that may
move the actor observation is the *event* of a witnessed cast, never the value
of the enemy's cooldown.

If any of those moves a single float in ``entities`` / ``self_vec`` /
``global_vec``, the actor is reading something it must not.

Run as ``python -m lanerl_rl.audit``; exit status 1 on any finding.
"""

from __future__ import annotations

import ast
import copy
import inspect
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import constants as C
from . import obs as obs_module
from .frame import (
    WIRE_DYNAMIC_FAMILIES,
    WIRE_FIELDS,
    Frame,
    WireField,
    decode_frame,
    record_keys,
    unit_keys,
)
from .obs import ACTOR_PATH_FUNCTIONS, PRIVILEGED_PATH_FUNCTIONS, ObservationBuilder
from .scenarios import encode_frame, make_frame, top_lane_scenario, unit

__all__ = ["AuditFinding", "run_audit", "main", "control_source_path"]


# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------

#: Attribute names that carry information a screenshot cannot recover about
#: another unit.
SERVER_ONLY_ATTRS: Set[str] = {
    "gold",
    "xp",
    "experience",
    "cs",
    "lvl",
    "level",
    "cooldowns",
    "spell_levels",
    "mana",
    "visible_to",
    # Own recall channel is HUD; the *enemy's* is privileged -- seeing that they
    # are recalling through a wall is exactly the kind of leak this list catches.
    "recalling",
    # Added to the C# observation (LanerlControl.BuildObservation) for debugging
    # an order that would not stick. NOTHING consumes them yet, and they are
    # listed here BEFORE anything does.
    #
    # `tgt` is the nastiest field on the wire: it is the enemy's current target
    # netid -- server truth about enemy INTENT, invisible on any screenshot, and
    # it does not disappear under fog. If it is ever wanted, it must be gated on
    # vb/vr like everything else, or routed to priv_vec for the critic only.
    "tgt",
    "target",
    "target_netid",
    "atk",
    "is_attacking",
    "mo",
    "move_order",
}

#: Attributes that are only legal on specific receivers.  ``.units`` is the raw
#: server unit table on a :class:`~lanerl_rl.frame.Frame`; on the actor path it
#: may only ever be reached through the agent's own fog-gated memory.
RECEIVER_RESTRICTED_ATTRS: Dict[str, Set[str]] = {
    "units": {"self.memory"},
}

#: Per-function exemptions, with the reason each one is legitimate.
ATTR_EXEMPTIONS: Dict[str, Dict[str, str]] = {
    # The agent's own HUD shows its own gold, level, CS and XP bar.
    "_build_self_vec": {
        "gold": "own gold is on the agent's own HUD",
        "xp": "own XP bar is on the agent's own HUD",
        "lvl": "own level is on the agent's own HUD",
        "cs": "own creep score is on the agent's own HUD",
    },
    # AbilityBook.sync is only ever called with the agent's OWN champion row;
    # its own ability bar (ranks and cooldown sweeps) is drawn on its own HUD.
    # The ENEMY's ability bar never comes through here -- it goes through
    # frame.EnemyAbilityIntel, which records witnessed casts only.
    "sync": {
        "cooldowns": "own cooldown sweep is on the agent's own HUD",
        "spell_levels": "own ability ranks are on the agent's own HUD",
    },
}

#: Field-name exemptions.  The name check is a cheap tripwire, not a proof, and
#: it fires on a couple of names that are legitimately about the enemy.  Each
#: entry has to justify itself, and each is additionally covered by a
#: *differential* check, which is the thing that actually proves the property.
FIELD_NAME_EXEMPTIONS: Dict[str, str] = {
    **{
        f"enemy_ability_{s.lower()}_cd_estimate": (
            "estimated from a WITNESSED cast (frame.EnemyAbilityIntel) using the "
            "rank-1 base cooldown table; the server's cooldown value is never "
            "stored, and check_enemy_cooldown_leak proves it cannot move the "
            "actor observation"
        )
        for s in C.SPELL_SLOTS
    },
}

#: Substrings that make a *field name* suspect when combined with "enemy".
FORBIDDEN_ENEMY_NOUNS = (
    "gold",
    "xp",
    "exp",
    "cooldown",
    "_cd",
    "mana",
    "exact",
    "true",
    "abs",
    "cs_",
)

#: Substrings that make a field name suspect on their own.
FORBIDDEN_ANY = ("priv", "unfog", "server_", "_true", "ground_truth")


@dataclass
class AuditFinding:
    check: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"[{self.check}] {self.message}"


# --------------------------------------------------------------------------
# Static checks
# --------------------------------------------------------------------------


def _module_functions(path: Path) -> Dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: Dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = node
    return out


def check_actor_function_names_are_unambiguous() -> List[AuditFinding]:
    """Two functions with the same name would make the static check skip one.

    ``ACTOR_PATH_FUNCTIONS`` addresses functions by bare name, and
    :func:`_module_functions` keys on that name, so a second definition of e.g.
    ``sync`` anywhere in ``obs.py`` would silently shadow the audited one and
    the leak check would pass while inspecting the wrong body.
    """
    path = Path(inspect.getsourcefile(obs_module))
    tree = ast.parse(path.read_text(), filename=str(path))
    seen: Dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seen[node.name] = seen.get(node.name, 0) + 1
    watched = set(ACTOR_PATH_FUNCTIONS) | set(PRIVILEGED_PATH_FUNCTIONS)
    return [
        AuditFinding(
            "static/names",
            f"{name!r} is defined {n} times in {path.name}; the static leak check "
            f"addresses it by bare name and would inspect only one of them",
        )
        for name, n in seen.items()
        if n > 1 and name in watched
    ]


def _receiver_repr(node: ast.AST) -> str:
    """Dotted source-level name of an attribute's receiver, e.g. ``self.memory``."""
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        parts.append("<expr>")
    return ".".join(reversed(parts))


def check_actor_path_attrs() -> List[AuditFinding]:
    """No actor-path function may read a server-only attribute."""
    findings: List[AuditFinding] = []
    path = Path(inspect.getsourcefile(obs_module))
    funcs = _module_functions(path)
    for name in ACTOR_PATH_FUNCTIONS:
        fn = funcs.get(name)
        if fn is None:
            findings.append(AuditFinding("static/attrs", f"actor function {name!r} not found in {path}"))
            continue
        exempt = ATTR_EXEMPTIONS.get(name, {})
        for node in ast.walk(fn):
            if not isinstance(node, ast.Attribute):
                continue
            if node.attr in SERVER_ONLY_ATTRS and node.attr not in exempt:
                findings.append(
                    AuditFinding(
                        "static/attrs",
                        f"{name}() reads server-only attribute '.{node.attr}' "
                        f"at line {node.lineno}",
                    )
                )
            allowed = RECEIVER_RESTRICTED_ATTRS.get(node.attr)
            if allowed is not None:
                recv = _receiver_repr(node.value)
                if recv not in allowed:
                    findings.append(
                        AuditFinding(
                            "static/attrs",
                            f"{name}() reads '{recv}.{node.attr}' at line {node.lineno}; "
                            f"'.{node.attr}' is only allowed on {sorted(allowed)} "
                            f"(the actor must read fog-gated memory, not the raw frame)",
                        )
                    )
    return findings


def check_actor_does_not_call_privileged() -> List[AuditFinding]:
    findings: List[AuditFinding] = []
    path = Path(inspect.getsourcefile(obs_module))
    funcs = _module_functions(path)
    banned = set(PRIVILEGED_PATH_FUNCTIONS)
    for name in ACTOR_PATH_FUNCTIONS:
        fn = funcs.get(name)
        if fn is None:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                target = node.func
                called = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
                if called in banned:
                    findings.append(
                        AuditFinding(
                            "static/calls",
                            f"{name}() calls privileged {called}() at line {node.lineno}",
                        )
                    )
    return findings


def check_field_names() -> List[AuditFinding]:
    """Scan the declared actor field layouts for server-only sounding fields."""
    findings: List[AuditFinding] = []
    registries = {
        "entity": C.ENTITY_FIELD_NAMES,
        "self": C.SELF_FIELD_NAMES,
        "global": C.GLOBAL_FIELD_NAMES,
    }
    for where, names in registries.items():
        for n in names:
            low = n.lower()
            if n in FIELD_NAME_EXEMPTIONS:
                continue
            for bad in FORBIDDEN_ANY:
                if bad in low:
                    findings.append(
                        AuditFinding("static/names", f"{where} field {n!r} contains forbidden token {bad!r}")
                    )
            if "enemy" in low:
                for noun in FORBIDDEN_ENEMY_NOUNS:
                    if noun in low:
                        findings.append(
                            AuditFinding(
                                "static/names",
                                f"{where} field {n!r} looks like enemy-private information ({noun!r})",
                            )
                        )
    return findings


def check_field_name_exemptions_are_used() -> List[AuditFinding]:
    """An exemption for a field that no longer exists is a stale excuse."""
    known = set(C.ENTITY_FIELD_NAMES) | set(C.SELF_FIELD_NAMES) | set(C.GLOBAL_FIELD_NAMES)
    return [
        AuditFinding(
            "static/names",
            f"field-name exemption {name!r} does not match any declared actor field; "
            f"delete it rather than leaving a standing excuse",
        )
        for name in FIELD_NAME_EXEMPTIONS
        if name not in known
    ]


# --------------------------------------------------------------------------
# Schema completeness: the wire, the decoder and the registry must agree
# --------------------------------------------------------------------------

#: Keys we insist the C# parse finds.  If it finds fewer, the *parser* is
#: broken and every "no new field" conclusion drawn from it is worthless --
#: which is the one way a completeness check can go vacuous.
_EMITTER_SENTINEL_KEYS = frozenset({"t", "u", "id", "k", "tm", "x", "y", "hp", "mhp"})

_LITERAL_GAP = "<EXPR>"
#: Text between two adjacent literals that is *only* method-call plumbing, so
#: the two literals concatenate rather than sandwiching a value.
_PLUMBING = re.compile(r"^[\s.)(]*(?:Append|AppendFormat|ToString|CultureInfo\.InvariantCulture|,)*[\s.)(]*$")
_KEY_RE = re.compile(r'"([A-Za-z0-9_<>]*)":')


def control_source_path() -> Optional[Path]:
    """Where ``LanerlControl.cs`` lives, or ``None`` if it cannot be found.

    ``LANERL_CONTROL_CS`` overrides.  Otherwise it is resolved from
    ``__file__`` -- never hardcoded -- because the shared export is mounted at
    a different absolute path on each node.
    """
    env = os.environ.get("LANERL_CONTROL_CS")
    if env:
        p = Path(env)
        return p if p.exists() else None
    repo = Path(__file__).resolve().parents[1]
    p = repo.parent / "lanerl-vendor/LoLServer/GameServerLib/Lanerl/LanerlControl.cs"
    return p if p.exists() else None


def _method_body(source: str, signature: str) -> Optional[str]:
    """The brace-matched body of the method whose declaration contains ``signature``."""
    at = source.find(signature)
    if at < 0:
        return None
    start = source.find("{", at)
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(source)):
        ch = source[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    return None


def _literal_skeleton(body: str) -> str:
    """C# body -> the JSON text it emits, with ``<EXPR>`` where a value goes.

    Only string and char literals survive; anything between two of them that is
    not pure ``.Append(`` plumbing becomes a gap.  So
    ``Append(",\\"cd").Append(sl).Append("\\":")`` collapses to ``,"cd<EXPR>":``
    and a looped key is still visible as a key.

    Comments are stripped in the same pass, and that is not a nicety: an
    apostrophe in an English comment ("the observation's feature") opens a char
    literal that swallows the rest of the method, and the parse then silently
    reports fewer keys than the server emits.  ``_EMITTER_SENTINEL_KEYS`` is the
    backstop for exactly this class of parser bug.
    """
    pieces: List[Tuple[int, int, str]] = []
    # A copy with comments blanked out, so the plumbing test between two
    # literals is not confused by prose.
    clean = list(body)
    i, n = 0, len(body)
    while i < n:
        two = body[i : i + 2]
        if two == "//":
            j = body.find("\n", i)
            j = n if j < 0 else j
            clean[i:j] = " " * (j - i)
            i = j
            continue
        if two == "/*":
            j = body.find("*/", i + 2)
            j = n if j < 0 else j + 2
            clean[i:j] = " " * (j - i)
            i = j
            continue
        ch = body[i]
        if ch in "\"'":
            quote = ch
            j = i + 1
            buf: List[str] = []
            while j < n:
                if body[j] == "\\" and j + 1 < n:
                    esc = body[j + 1]
                    buf.append({"n": "\n", "t": "\t"}.get(esc, esc))
                    j += 2
                    continue
                if body[j] == quote:
                    break
                buf.append(body[j])
                j += 1
            pieces.append((i, j + 1, "".join(buf)))
            i = j + 1
            continue
        i += 1
    cleaned = "".join(clean)
    out: List[str] = []
    prev_end: Optional[int] = None
    for start, end, text in pieces:
        if prev_end is not None and not _PLUMBING.match(cleaned[prev_end:start]):
            out.append(_LITERAL_GAP)
        out.append(text)
        prev_end = end
    return "".join(out)


def emitted_wire_keys(source: Optional[str] = None) -> Tuple[Set[str], Set[str]]:
    """``(resolved keys, unresolvable dynamic patterns)`` from the C# emitter."""
    if source is None:
        path = control_source_path()
        source = "" if path is None else path.read_text()
    body = _method_body(source, "string BuildObservation(")
    if body is None:
        return set(), set()
    found = set(_KEY_RE.findall(_literal_skeleton(body)))
    keys: Set[str] = set()
    unresolved: Set[str] = set()
    for name in found:
        if not name:
            continue
        if _LITERAL_GAP in name:
            family = WIRE_DYNAMIC_FAMILIES.get(name)
            if family is None:
                unresolved.add(name)
            else:
                keys.update(family)
        else:
            keys.add(name)
    return keys, unresolved


def check_wire_schema_covers_the_emitter() -> List[AuditFinding]:
    """Every key ``BuildObservation`` writes must be classified in ``WIRE_FIELDS``.

    This is the structural replacement for the old reactive allowlist: a new
    server field fails the audit *because it is new*, not because somebody
    remembered to add a probe for it.
    """
    findings: List[AuditFinding] = []
    path = control_source_path()
    if path is None:
        return [
            AuditFinding(
                "schema/emitter",
                "cannot find LanerlControl.cs, so the claim 'no unclassified server "
                "field exists' is unproven. Point LANERL_CONTROL_CS at it. Reporting "
                "this as a finding rather than skipping: a completeness check that "
                "quietly does nothing is exactly the failure this replaces.",
            )
        ]
    keys, unresolved = emitted_wire_keys(path.read_text())
    if not keys:
        return [
            AuditFinding(
                "schema/emitter",
                f"parsed {path} but found no emitted keys at all. BuildObservation was "
                f"renamed or restructured; the parser, not the server, is what needs "
                f"fixing.",
            )
        ]
    missing_sentinels = sorted(_EMITTER_SENTINEL_KEYS - keys)
    if missing_sentinels:
        findings.append(
            AuditFinding(
                "schema/emitter",
                f"the emitter parse missed keys that are certainly there "
                f"({missing_sentinels}); it cannot be trusted to have found a NEW one "
                f"either, so treat this as a broken check rather than a clean bill",
            )
        )
    for pattern in sorted(unresolved):
        findings.append(
            AuditFinding(
                "schema/emitter",
                f"the emitter builds key {pattern!r} from an expression. Add it to "
                f"frame.WIRE_DYNAMIC_FAMILIES with the keys it expands to; a key this "
                f"parser cannot resolve is a key the audit cannot classify.",
            )
        )
    for key in sorted(keys - set(WIRE_FIELDS)):
        findings.append(
            AuditFinding(
                "schema/emitter",
                f"LanerlControl.BuildObservation emits {key!r}, which is not in "
                f"frame.WIRE_FIELDS. Classify it (actor / critic / internal / "
                f"unconsumed) with a reason before anything reads it. If it is "
                f"server-only state, it must never reach the actor observation.",
            )
        )
    for key in sorted(k for k, f in WIRE_FIELDS.items() if f.emitted and k not in keys):
        findings.append(
            AuditFinding(
                "schema/emitter",
                f"WIRE_FIELDS says {key!r} is emitted by the current server, but the "
                f"emitter does not write it. Either the field was removed (mark it "
                f"emitted=False with a reason, or delete it) or the parse is wrong.",
            )
        )
    for key in sorted(k for k, f in WIRE_FIELDS.items() if not f.emitted and k in keys):
        findings.append(
            AuditFinding(
                "schema/emitter",
                f"WIRE_FIELDS marks {key!r} as legacy/not-emitted, but the current "
                f"emitter writes it. A stale 'legacy' flag hides a live field from "
                f"every reader of this table.",
            )
        )
    return findings


class _KeyRecorder(dict):
    """A dict that remembers which keys were asked for.

    Used to enumerate what ``decode_frame`` *actually* consumes.  An AST scan
    would have been fooled by ``ru.get(f"cd{i}")``; running the decoder cannot
    be.
    """

    def __init__(self, data, seen: Set[str]):
        super().__init__(data)
        self._seen = seen

    def __getitem__(self, key):
        self._seen.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._seen.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self._seen.add(key)
        return super().__contains__(key)


def _recorded_decode(raw: dict) -> Tuple[Set[str], Set[str]]:
    """Decode ``raw`` and report ``(record keys read, unit keys read)``."""
    top_seen: Set[str] = set()
    unit_seen: Set[str] = set()
    wrapped = dict(raw)
    wrapped["u"] = [_KeyRecorder(ru, unit_seen) for ru in raw["u"]]
    decode_frame(_KeyRecorder(wrapped, top_seen))
    return top_seen, unit_seen


def _schema_record_variants() -> List[dict]:
    """Wire records that between them exercise every decode branch."""
    f = top_lane_scenario()
    return [
        encode_frame(f),
        encode_frame(f, legacy_visibility=True, legacy_cooldowns=True),
        encode_frame(f, with_optional=True),
    ]


def check_wire_schema_matches_the_decoder() -> List[AuditFinding]:
    """``decoded=True`` must mean what it says, in both directions.

    ``decode_frame`` is run under an instrumented mapping, so this compares the
    registry against the decoder's real behaviour rather than against a comment.
    A key the decoder reads but nobody registered is a leak surface with no
    probe; a key registered as ``unconsumed`` that the decoder in fact reads is
    a *false* safety claim, which is worse.
    """
    findings: List[AuditFinding] = []
    top_seen: Set[str] = set()
    unit_seen: Set[str] = set()
    for raw in _schema_record_variants():
        a, b = _recorded_decode(raw)
        top_seen |= a
        unit_seen |= b
    if not top_seen or not unit_seen:
        return [
            AuditFinding(
                "schema/decoder",
                "the key recorder observed no reads at all, so it proves nothing "
                "about what decode_frame consumes",
            )
        ]
    for seen, known, scope in ((top_seen, record_keys(), "record"), (unit_seen, unit_keys(), "unit")):
        for key in sorted(seen - known):
            findings.append(
                AuditFinding(
                    "schema/decoder",
                    f"decode_frame reads unregistered {scope} key {key!r}; it is "
                    f"consumed by the pipeline and nothing classifies it",
                )
            )
        for key in sorted(k for k in known if WIRE_FIELDS[k].decoded and k not in seen):
            findings.append(
                AuditFinding(
                    "schema/decoder",
                    f"WIRE_FIELDS claims {scope} key {key!r} is decoded, but "
                    f"decode_frame never reads it. Mark it unconsumed (which is a "
                    f"stronger safety statement) or fix the decoder.",
                )
            )
        for key in sorted(k for k in known if not WIRE_FIELDS[k].decoded and k in seen):
            findings.append(
                AuditFinding(
                    "schema/decoder",
                    f"WIRE_FIELDS claims {scope} key {key!r} is NOT consumed, but "
                    f"decode_frame reads it. That claim is the whole reason it has no "
                    f"differential probe.",
                )
            )
    return findings


# --------------------------------------------------------------------------
# Differential checks
# --------------------------------------------------------------------------


def _build_pair(frame_a: Frame, frame_b: Frame, team: int = C.TEAM_BLUE):
    ba = ObservationBuilder(team, fog_model=_quiet_fog())
    bb = ObservationBuilder(team, fog_model=_quiet_fog())
    return ba.build(frame_a), bb.build(frame_b)


def _quiet_fog():
    from .frame import ApproxFogModel

    return ApproxFogModel(warn=False)


def _actor_arrays(o) -> Dict[str, np.ndarray]:
    return {
        "entities": o.entities,
        "entity_pad_mask": o.entity_pad_mask,
        "self_vec": o.self_vec,
        "global_vec": o.global_vec,
    }


def _diff_report(a, b, label: str, check: str) -> List[AuditFinding]:
    findings = []
    for key, arr_a in _actor_arrays(a).items():
        arr_b = _actor_arrays(b)[key]
        if not np.array_equal(arr_a, arr_b):
            where = np.argwhere(arr_a != arr_b)
            first = where[0].tolist() if len(where) else None
            names = None
            if key == "entities" and first is not None:
                names = C.ENTITY_FIELD_NAMES[first[1]]
            elif key == "self_vec" and first is not None:
                names = C.SELF_FIELD_NAMES[first[0]]
            elif key == "global_vec" and first is not None:
                names = C.GLOBAL_FIELD_NAMES[first[0]]
            findings.append(
                AuditFinding(
                    check,
                    f"{label}: actor array {key!r} changed at index {first} "
                    f"(field {names!r}); {len(where)} element(s) differ",
                )
            )
    return findings


def check_enemy_economy_leak() -> List[AuditFinding]:
    """Enemy gold / xp / cs / level must not touch the actor observation."""
    base = top_lane_scenario()
    poisoned = top_lane_scenario(red_gold=999_999.0, red_lvl=18)
    for u in poisoned.units.values():
        if u.etype == "champion" and u.team == C.TEAM_RED:
            u.xp = 999_999.0
            u.cs = 999
            u.cooldowns = (0.0, 0.0, 0.0, 0.0)
            u.spell_levels = (5, 5, 5, 5)
    a, b = _build_pair(base, poisoned)
    findings = _diff_report(a, b, "enemy gold/xp/cs/level poisoned", "dynamic/economy")
    if np.array_equal(a.priv_vec, b.priv_vec):
        findings.append(
            AuditFinding(
                "dynamic/economy",
                "privileged vector did NOT change when enemy economy was poisoned -- "
                "the critic is not actually receiving privileged information",
            )
        )
    return findings


def check_hp_quantisation() -> List[AuditFinding]:
    """Sub-health-bar HP precision must be quantised away."""
    step = 1.0 / C.HP_BAR_STEPS
    base = top_lane_scenario(red_hp_frac=0.5)
    # Nudge by a third of a bar segment: invisible on screen, visible to the server.
    nudged = top_lane_scenario(red_hp_frac=0.5)
    for u in nudged.units.values():
        if u.etype == "champion" and u.team == C.TEAM_RED:
            u.hp = u.mhp * (0.5 + step / 3.0)
    a, b = _build_pair(base, nudged)
    return _diff_report(a, b, "enemy HP nudged by 1/3 of a health-bar segment", "dynamic/hp_quantisation")


def check_fog_leak() -> List[AuditFinding]:
    """A fogged unit's live position must not reach the actor observation."""
    findings: List[AuditFinding] = []
    # Blue champion alone at its own turret; red champion parked deep in its own
    # half, far outside anybody's vision radius.  No minions, so nothing else
    # can grant vision.
    a_pos = C.TOP_OUTER_TURRET[C.TEAM_BLUE]
    far = (12000, 12000)
    other = (11000, 11000)

    def build(red_xy) -> Frame:
        return make_frame(
            120_000,
            [
                unit(1001, "champion", C.TEAM_BLUE, a_pos[0], a_pos[1], hp=671, mhp=671, gold=500.0, xp=0.0, lvl=1),
                unit(1002, "champion", C.TEAM_RED, red_xy[0], red_xy[1], hp=671, mhp=671, gold=500.0, xp=0.0, lvl=1),
                unit(4001, "turret", C.TEAM_BLUE, a_pos[0], a_pos[1], hp=1550, mhp=1550),
            ],
        )

    f1, f2 = build(far), build(other)
    a, b = _build_pair(f1, f2)
    findings += _diff_report(a, b, "fogged enemy teleported", "dynamic/fog")

    if a.entities[0, C.E_VALID] != 0.0:
        findings.append(
            AuditFinding("dynamic/fog", "fogged enemy champion has valid=1 in the enemy-champion slot")
        )
    if a.global_vec[C.G_ENEMY_VISIBLE] != 0.0:
        findings.append(AuditFinding("dynamic/fog", "global enemy_visible is set for a fogged enemy"))
    return findings


def check_all_slots_never_alias_fogged() -> List[AuditFinding]:
    """No *valid* slot may carry a fogged unit's position."""
    from .frame import ApproxFogModel

    findings: List[AuditFinding] = []
    frames = [top_lane_scenario(t_ms=90_000 + 100 * i, blue_s=0.30, red_s=0.30 + 0.02 * i) for i in range(20)]
    builder = ObservationBuilder(C.TEAM_BLUE, fog_model=ApproxFogModel(warn=False))
    fog = ApproxFogModel(warn=False)
    for f in frames:
        o = builder.build(f)
        visible = fog.visible_ids(f, C.TEAM_BLUE)
        ax, ay = builder.transform.point(f.champion_of_team(C.TEAM_BLUE).x, f.champion_of_team(C.TEAM_BLUE).y)
        for uid, u in f.units.items():
            if uid in visible:
                continue
            cx, cy = builder.transform.point(u.x, u.y)
            ds, dn = (cx - ax) / C.NORM_XY, (cy - ay) / C.NORM_XY
            for slot in range(C.N_SLOTS):
                if o.entities[slot, C.E_VALID] < 0.5:
                    continue
                if (
                    abs(float(o.entities[slot, C.E_DS]) - ds) < 1e-6
                    and abs(float(o.entities[slot, C.E_DN]) - dn) < 1e-6
                ):
                    findings.append(
                        AuditFinding(
                            "dynamic/fog",
                            f"t={f.t_ms}: valid slot {slot} carries the live position of "
                            f"fogged unit {uid}",
                        )
                    )
    return findings


# -- the enemy ability book --------------------------------------------------
#
# The one genuinely new leak surface in the package.  Two probes, one for each
# half of the property: while the enemy is FOGGED nothing about their abilities
# may reach the actor at all, and while they are VISIBLE only a witnessed *cast*
# may -- never the cooldown value itself.


def _cooldown_scenario(t_ms: int, red_s: float, cds) -> Frame:
    f = top_lane_scenario(t_ms=t_ms, blue_s=0.35, red_s=red_s, n_minions=0)
    for u in f.units.values():
        if u.etype == "champion" and u.team == C.TEAM_RED:
            u.cooldowns = cds
    return f


def _run_sequence(frames: Sequence[Frame]):
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=_quiet_fog())
    out = None
    for f in frames:
        out = b.build(f)
    return out


def check_enemy_cooldown_leak() -> List[AuditFinding]:
    """A fogged enemy's cooldowns must not reach the actor observation at all."""
    findings: List[AuditFinding] = []
    # RED parked at the far end of the lane: outside blue's 1100 vision radius,
    # and no minions to grant vision.
    seq_a = [_cooldown_scenario(100_000 + 100 * i, 0.98, (8.0, 24.0, 9.0, 160.0)) for i in range(6)]
    seq_b = [_cooldown_scenario(100_000 + 100 * i, 0.98, (0.0, 0.0, 0.0, 0.0)) for i in range(6)]
    a, b = _run_sequence(seq_a), _run_sequence(seq_b)
    findings += _diff_report(a, b, "fogged enemy cooldowns rewritten", "dynamic/cooldowns")

    # And a fogged enemy going ON cooldown mid-sequence must be invisible too:
    # a cast we did not witness is a cast we do not know about.
    seq_c = [
        _cooldown_scenario(100_000 + 100 * i, 0.98, (0.0,) * 4 if i < 3 else (8.0, 24.0, 9.0, 160.0))
        for i in range(6)
    ]
    c = _run_sequence(seq_c)
    findings += _diff_report(a, c, "fogged enemy cast mid-sequence", "dynamic/cooldowns")

    if a.priv_vec[C.P_E_COOLDOWN_TRUE].sum() == b.priv_vec[C.P_E_COOLDOWN_TRUE].sum():
        findings.append(
            AuditFinding(
                "dynamic/cooldowns",
                "privileged vector did NOT change when the enemy's cooldowns were "
                "rewritten -- the critic is not receiving them",
            )
        )
    return findings


def check_visible_enemy_cooldown_value_leak() -> List[AuditFinding]:
    """While the enemy is in plain sight, only a CAST may move the observation.

    The cooldown *value* must not: a player watching an enemy champion sees the
    cast animation, not the number of seconds left on the enemy's Q.  So a
    sub-threshold jitter of the reported cooldown -- large enough that a value
    read would notice, far smaller than the rise a real cast produces -- has to
    leave the actor arrays bit-identical.
    """
    findings: List[AuditFinding] = []
    jitter = C.CAST_DETECT_RISE_S / 4.0
    seq_a = [
        _cooldown_scenario(100_000 + 100 * i, 0.37, (4.0 - 0.05 * i,) * 4) for i in range(8)
    ]
    seq_b = [
        _cooldown_scenario(100_000 + 100 * i, 0.37, (4.0 - 0.05 * i + jitter,) * 4) for i in range(8)
    ]
    a, b = _run_sequence(seq_a), _run_sequence(seq_b)
    if a.global_vec[C.G_ENEMY_VISIBLE] != 1.0:
        findings.append(
            AuditFinding(
                "dynamic/cooldowns",
                "the visible-enemy cooldown probe did not actually have the enemy in "
                "vision, so it proves nothing; fix the scenario",
            )
        )
    findings += _diff_report(
        a, b, "visible enemy cooldowns jittered below the cast threshold", "dynamic/cooldowns"
    )

    # Positive control: a real cast (a full-cooldown rise) MUST be noticed, or
    # the feature is dead and the probe above is vacuous.
    seq_c = [
        _cooldown_scenario(100_000 + 100 * i, 0.37, (0.0,) * 4 if i < 4 else (8.0,) * 4)
        for i in range(8)
    ]
    c = _run_sequence(seq_c)
    if np.array_equal(a.global_vec, c.global_vec):
        findings.append(
            AuditFinding(
                "dynamic/cooldowns",
                "a witnessed enemy cast did NOT move the actor observation -- the "
                "enemy ability book is dead, so the leak probe above is vacuous",
            )
        )
    return findings


# -- the registry-driven probe ----------------------------------------------
#
# Everything above this line is a probe somebody wrote for a field they had
# thought of.  This one is generated from `WIRE_FIELDS`, so a new server field
# gets a differential probe the moment it is classified -- and the audit
# refuses to run without it being classified.


def _poisoned_records(
    field: WireField, scenario_kwargs: dict, n: int = 4
) -> Tuple[List[dict], List[dict]]:
    """``(clean, poisoned)`` wire records, differing only in ``field`` on RED."""
    kw = dict(legacy_cooldowns=field.key == "cd", with_optional=field.key in ("cs", "sl"))
    clean = [
        encode_frame(top_lane_scenario(t_ms=100_000 + 100 * i, **scenario_kwargs), **kw)
        for i in range(n)
    ]
    poisoned = copy.deepcopy(clean)
    for rec in poisoned:
        if field.scope == "record":
            rec[field.key] = field.poison
            continue
        for row in rec["u"]:
            if row["k"] == "Champion" and row["tm"] == C.TEAM_RED:
                row[field.key] = copy.deepcopy(field.poison)
    return clean, poisoned


def _run_records(records: Sequence[dict], team: int = C.TEAM_BLUE):
    b = ObservationBuilder(team, fog_model=_quiet_fog())
    out = None
    for rec in records:
        out = b.build(decode_frame(rec))
    return out


def check_wire_fields_do_not_leak_to_the_actor() -> List[AuditFinding]:
    """Every ``actor_invariant`` wire field, poisoned on RED, generated from the registry.

    Two situations per field, because they fail differently: the enemy in plain
    sight (where the actor legitimately reads *some* of their record, so a
    value read is easy to hide) and the enemy fogged (where the actor must read
    none of it).  Each situation carries its own positive control, so a probe
    that has stopped being able to fire reports itself instead of passing.
    """
    findings: List[AuditFinding] = []
    situations = {
        # RED at s=0.37 against BLUE at 0.35: inside the 1100-unit champion
        # vision radius, and no minions, so nothing else grants vision.
        "enemy visible": (dict(blue_s=0.35, red_s=0.37, n_minions=0), True),
        "enemy fogged": (dict(blue_s=0.30, red_s=0.98, n_minions=0), False),
    }

    for label, (kwargs, want_visible) in situations.items():
        control = _run_records(
            [encode_frame(top_lane_scenario(t_ms=100_000, **kwargs))]
        )
        got_visible = bool(control.global_vec[C.G_ENEMY_VISIBLE])
        if got_visible != want_visible:
            findings.append(
                AuditFinding(
                    "dynamic/wire_schema",
                    f"the '{label}' situation has enemy_visible={got_visible}, not "
                    f"{want_visible}; every probe run in it proves nothing. Fix the "
                    f"scenario.",
                )
            )
            continue

        for key, field in sorted(WIRE_FIELDS.items()):
            if not field.actor_invariant:
                continue
            clean, poisoned = _poisoned_records(field, kwargs)
            a, b = _run_records(clean), _run_records(poisoned)
            findings += _diff_report(
                a,
                b,
                f"{label}: enemy wire field {key!r} rewritten to {field.poison!r}",
                "dynamic/wire_schema",
            )
            # Positive control, per field and per situation: the poison has to
            # be *reaching* the pipeline, or "the actor did not move" is a
            # statement about nothing.  A decoded field must change the decoded
            # frame; an unconsumed one must not (that IS its safety argument).
            changed = decode_frame(clean[-1]) != decode_frame(poisoned[-1])
            if field.decoded and not changed:
                findings.append(
                    AuditFinding(
                        "dynamic/wire_schema",
                        f"{label}: poisoning {key!r} did not change the decoded frame, "
                        f"so the leak probe for it is vacuous. Either the poison value "
                        f"is a no-op or the field is not really decoded.",
                    )
                )
            if not field.decoded and changed:
                findings.append(
                    AuditFinding(
                        "dynamic/wire_schema",
                        f"{label}: {key!r} is registered as unconsumed, but poisoning "
                        f"it changed the decoded frame. Its safety argument -- that "
                        f"nothing can read what nothing decodes -- is now false.",
                    )
                )
    return findings


def check_actor_visible_wire_fields_are_declared() -> List[AuditFinding]:
    """A field that is *not* actor-invariant has to be one the agent may see.

    The inverse of the probe above, and the reason the registry cannot be
    silenced by flipping a flag: ``actor_invariant=False`` is only legitimate
    for a field the actor is allowed to read (``disposition="actor"``) or for
    the fog gate and the identity keys, which change *which* unit a record
    describes rather than revealing a hidden property of it.
    """
    allowed_non_invariant = {"vb", "vr", "vis", "id", "u"}
    return [
        AuditFinding(
            "schema/registry",
            f"wire field {key!r} is declared disposition={f.disposition!r} but "
            f"actor_invariant=False, i.e. it is allowed to move the actor "
            f"observation while not being actor-visible. Either it is actually "
            f"actor-visible (say so) or the exemption is a hole.",
        )
        for key, f in sorted(WIRE_FIELDS.items())
        if not f.actor_invariant and f.disposition != "actor" and key not in allowed_non_invariant
    ]


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

CHECKS = (
    ("schema: every emitted server field is classified", check_wire_schema_covers_the_emitter),
    ("schema: the registry matches what decode_frame reads", check_wire_schema_matches_the_decoder),
    ("schema: no undeclared actor-visible wire field", check_actor_visible_wire_fields_are_declared),
    ("static: actor path reads no server-only attribute", check_actor_path_attrs),
    ("static: actor path calls no privileged builder", check_actor_does_not_call_privileged),
    ("static: declared field names", check_field_names),
    ("static: no stale field-name exemptions", check_field_name_exemptions_are_used),
    ("static: audited function names are unambiguous", check_actor_function_names_are_unambiguous),
    ("differential: enemy economy", check_enemy_economy_leak),
    ("differential: HP quantisation", check_hp_quantisation),
    ("differential: fog", check_fog_leak),
    ("differential: no fogged aliasing in valid slots", check_all_slots_never_alias_fogged),
    ("differential: fogged enemy cooldowns", check_enemy_cooldown_leak),
    ("differential: visible enemy cooldown values", check_visible_enemy_cooldown_value_leak),
    ("differential: every wire field, generated from the registry", check_wire_fields_do_not_leak_to_the_actor),
)


def run_audit(verbose: bool = True) -> List[AuditFinding]:
    findings: List[AuditFinding] = []
    for label, fn in CHECKS:
        got = fn()
        findings.extend(got)
        if verbose:
            status = "FAIL" if got else "ok"
            print(f"  [{status:>4}] {label}" + (f"  ({len(got)} finding(s))" if got else ""))
    return findings


def main(argv: Optional[Sequence[str]] = None) -> int:
    print("lanerl_rl observation audit")
    print("=" * 70)
    findings = run_audit(verbose=True)
    print("=" * 70)
    if findings:
        print(f"\n!!! OBSERVATION AUDIT FAILED: {len(findings)} finding(s) !!!\n")
        for f in findings:
            print(f"  {f}")
        print(
            "\nThe actor observation is reading state a screenshot cannot produce.\n"
            "A policy trained on it will not transfer to a deployed client.\n"
        )
        return 1
    print("\nAUDIT PASSED: the actor observation is free of server-only fields.\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
