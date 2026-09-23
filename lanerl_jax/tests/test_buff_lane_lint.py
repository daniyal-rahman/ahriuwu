"""`STRUCT-001` gate: an AST lint for the two index spaces of Garen's kit.

The kit had two small-integer index spaces that read alike and are not:

* ``Slot.Q/W/E/R`` = 0..3 index the SPELL arrays ``spell_level`` and
  ``spell_cooldown`` (``CharData`` ``Spell1``-``Spell4``), and
  ``spells.Status``'s ``cast_locked``/``can_cast``;
* the buff table's LANES (``E_BUFF_SLOT = 0`` ... ``E_TICK_BUFF_SLOT = 6``) of
  the ``(N, 8)`` arrays ``buff_id`` / ``buff_elapsed`` / ``buff_duration`` /
  ``buff_power``.

``Slot.E`` is 2 and ``E_BUFF_SLOT`` was 0, so ``buff_id[:, Slot.E]`` read lane
2 (the W passive) and compared identically False against ``BuffId.GAREN_E``
-- ``SPELL-005``: Garen never ghosted while spinning (``step.py``, and the same
line in ``parity/archive/truncation_bound_probe.py``). ``STRUCT-001`` counted
the collision at three sites, and nothing failed on any of them.

The rewrite removed the lane table: buffs are now one typed record per kind
(``LaneState.buffs``, :class:`lanerl_jax.sim.state.Buffs`), indexed by UNIT
only. The intent of this lint is unchanged -- nothing indexes a buff with a
spell slot or a spell array with a buff -- and it now also keeps the lane
table from coming back and pins the ``spell_cooldown`` writer set. Over
``lanerl_jax/**/*.py`` and ``tools/**/*.py`` it flags:

1. a ``Slot.<X>`` (or a parameter defaulting to one) anywhere in the index of
   a BUFF RECORD array (``<...>.buffs.<rec>.<field>[...]``,
   ``<...>.buffs.w_passive[...]``, including ``.at[...]`` updates) -- a buff
   field is ``(N,)``, so its only index is a unit;
2. a buff-record expression, or a retired ``*_BUFF_SLOT`` lane constant, in
   the index of ``spell_level`` / ``spell_cooldown`` / ``cast_locked`` /
   ``can_cast``;
3. the RETIRED lane table anywhere in live code (``parity/archive/`` is dead
   code and exempt): the names ``buff_id``, ``buff_elapsed``,
   ``buff_duration``, ``buff_power``, ``BuffId``, ``MAX_BUFFS`` or any
   ``*_BUFF_SLOT``, as a name, an attribute, an import or a keyword;
4. an ``.at[...]`` write to ``spell_cooldown`` in the simulator
   (``lanerl_jax/sim``, tests excluded) outside the functions listed in
   ``spells.COOLDOWN_WRITERS`` in ``spells.py`` -- the one documented writer
   set.

The record and field names are read from ``state.py`` and the writer set from
``spells.py`` by parsing them, not importing them, so this file needs no JAX
and runs under any interpreter.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
SPELLS = ROOT / "lanerl_jax" / "sim" / "spells.py"
STATE = ROOT / "lanerl_jax" / "sim" / "state.py"

SPELL_ARRAY = re.compile(r"^(spell_level|spell_cooldown|cast_locked|can_cast)(_\w+)?$")
COOLDOWN_ARRAY = re.compile(r"^spell_cooldown(_\w+)?$")
RETIRED = {"buff_id", "buff_elapsed", "buff_duration", "buff_power", "BuffId",
           "MAX_BUFFS"}
RETIRED_LANE = re.compile(r"^[A-Z_]*_BUFF_SLOT$")
#: dead code, not importable against the current state (`STRUCT-001`)
ARCHIVE = "lanerl_jax/parity/archive/"


# ---------------------------------------------------------------- constants --
def _class_ints(tree: ast.Module, name: str) -> Dict[str, int]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            out: Dict[str, int] = {}
            for st in node.body:
                if isinstance(st, ast.Assign):
                    tgts, val = st.targets, st.value
                    if (len(tgts) == 1 and isinstance(tgts[0], ast.Tuple)
                            and isinstance(val, ast.Tuple)):
                        for t, v in zip(tgts[0].elts, val.elts):
                            out[t.id] = ast.literal_eval(v)
                    elif isinstance(tgts[0], ast.Name):
                        out[tgts[0].id] = ast.literal_eval(val)
            return out
    raise AssertionError(f"class {name} not found in {SPELLS}")


def _class_fields(tree: ast.Module, name: str) -> Dict[str, str]:
    """``field -> annotation source`` of a dataclass body."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return {st.target.id: ast.unparse(st.annotation)
                    for st in node.body
                    if isinstance(st, ast.AnnAssign)
                    and isinstance(st.target, ast.Name)}
    raise AssertionError(f"class {name} not found in {STATE}")


def _load_constants():
    spells = ast.parse(SPELLS.read_text())
    slot = _class_ints(spells, "Slot")
    writers: Tuple[str, ...] = ()
    functions: Set[str] = set()
    for node in spells.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "COOLDOWN_WRITERS"):
            writers = tuple(ast.literal_eval(node.value))
        if isinstance(node, ast.FunctionDef):
            functions.add(node.name)
    state = ast.parse(STATE.read_text())
    buffs = _class_fields(state, "Buffs")
    records = {name: set(_class_fields(state, ann))
               for name, ann in buffs.items() if ann != "jax.Array"}
    flags = {name for name, ann in buffs.items() if ann == "jax.Array"}
    return slot, writers, functions, records, flags


SLOT, WRITERS, SPELL_FUNCTIONS, RECORDS, FLAGS = _load_constants()
RECORD_FIELDS = set().union(*RECORDS.values())


def test_the_buff_struct_and_writer_set_are_what_the_rules_assume():
    """Guard the guard: the struct parsed into the shape the rules match, and
    every declared cooldown writer is a real function in `spells.py`."""
    assert set(SLOT) == {"Q", "W", "E", "R"}
    assert set(RECORDS) == {"e", "q", "q_haste", "w", "r_pending"}, RECORDS
    assert FLAGS == {"w_passive"}, FLAGS
    assert {"active", "elapsed_s"} <= RECORD_FIELDS
    assert WRITERS, "COOLDOWN_WRITERS not found in spells.py"
    assert set(WRITERS) <= SPELL_FUNCTIONS, set(WRITERS) - SPELL_FUNCTIONS


# --------------------------------------------------------------------- lint --
def _chain(node: ast.AST) -> List[str]:
    """``state.buffs.e.active`` -> ``["state", "buffs", "e", "active"]``;
    ``x.at`` is looked through (the ``.at[...]`` update form)."""
    out: List[str] = []
    while True:
        if isinstance(node, ast.Attribute):
            if node.attr != "at":
                out.append(node.attr)
            node = node.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Name):
            out.append(node.id)
            break
        else:
            break
    return out[::-1]


def _is_buff_record_access(node: ast.AST) -> bool:
    """A ``buffs`` record field, or the ``w_passive`` flag, reached through a
    ``.buffs`` attribute (``state.buffs.e.active``, ``bs.buffs.w_passive``) or
    from a bare ``buffs`` name (``buffs.q.skip_next``)."""
    c = _chain(node)
    if "buffs" not in c:
        return False
    rest = c[c.index("buffs") + 1:]
    if len(rest) >= 1 and rest[0] in FLAGS:
        return True
    return len(rest) >= 2 and rest[0] in RECORDS and rest[1] in RECORDS[rest[0]]


def _base_name(node: ast.AST) -> Optional[str]:
    """``state.spell_cooldown`` / ``spell_cooldown`` / ``x.at`` -> the name."""
    if isinstance(node, ast.Attribute) and node.attr == "at":
        return _base_name(node.value)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_slot_attr(node: ast.AST) -> Optional[str]:
    """``Slot.E`` or ``<anything>.Slot.E`` -> ``"E"``."""
    if (isinstance(node, ast.Attribute) and node.attr in SLOT
            and _base_name(node.value) == "Slot"):
        return node.attr
    return None


def _subscript_parts(sub: ast.Subscript):
    """``(value, all_index_exprs)`` of a subscript."""
    sl = sub.slice
    if isinstance(sl, ast.Index):          # py<3.9 AST
        sl = sl.value                       # pragma: no cover
    elts = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
    return sub.value, elts


class _Linter(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.findings: List[str] = []
        #: rule-4 writes that ARE in the writer set (a positive control)
        self.allowed_cooldown_writes = 0
        self.slot_alias: List[Dict[str, str]] = [{}]
        self.functions: List[str] = []
        self.archive = path.startswith(ARCHIVE)
        self.in_sim = (path.startswith("lanerl_jax/sim/")
                       and "/tests/" not in path)
        self.is_spells = path == "lanerl_jax/sim/spells.py"

    # -- scopes --
    def _visit_function(self, node):
        slots: Dict[str, str] = {}
        args = node.args
        pos = args.posonlyargs + args.args
        defaults = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)
        pairs = list(zip(pos, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
        for a, d in pairs:
            if d is None:
                continue
            s = _is_slot_attr(d)
            if s is not None:
                slots[a.arg] = s
        self.slot_alias.append({**self.slot_alias[-1], **slots})
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()
        self.slot_alias.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # -- helpers --
    def _where(self, node) -> str:
        return f"{self.path}:{node.lineno}"

    def _slot_in(self, expr: ast.AST) -> Optional[str]:
        for n in ast.walk(expr):
            s = _is_slot_attr(n)
            if s is not None:
                return f"Slot.{s}"
            if isinstance(n, ast.Name) and n.id in self.slot_alias[-1]:
                return f"{n.id}(=Slot.{self.slot_alias[-1][n.id]})"
        return None

    def _buff_in(self, expr: ast.AST) -> Optional[str]:
        for n in ast.walk(expr):
            if isinstance(n, (ast.Attribute, ast.Name)):
                name = n.attr if isinstance(n, ast.Attribute) else n.id
                if RETIRED_LANE.match(name):
                    return name
                if isinstance(n, ast.Attribute) and _is_buff_record_access(n):
                    return ".".join(_chain(n))
        return None

    def _retired(self, name: str, node) -> None:
        if self.archive:
            return
        if name in RETIRED or RETIRED_LANE.match(name):
            self.findings.append(
                f"{self._where(node)}: rule 3: `{name}` is the retired buff "
                f"lane table -- use LaneState.buffs (sim.state.Buffs)")

    # -- rules --
    def visit_Subscript(self, node: ast.Subscript):
        value, elts = _subscript_parts(node)
        if _is_buff_record_access(value):
            for e in elts:
                s = self._slot_in(e)
                if s is not None:
                    self.findings.append(
                        f"{self._where(node)}: rule 1: spell slot {s} used to "
                        f"index buff record `{'.'.join(_chain(value))}` -- a "
                        f"buff field is (N,), indexed by UNIT")
                    break
        name = _base_name(value)
        if name is not None and SPELL_ARRAY.match(name):
            for e in elts:
                b = self._buff_in(e)
                if b is not None:
                    self.findings.append(
                        f"{self._where(node)}: rule 2: buff `{b}` used to "
                        f"index spell array `{name}` -- use Slot.<X>")
                    break
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        self._retired(node.id, node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        self._retired(node.attr, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for a in node.names:
            self._retired(a.name, node)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword):
        if node.arg is not None:
            self._retired(node.arg, node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # rule 4: spell_cooldown.at[...].<op>(...) in the simulator
        f = node.func
        if (self.in_sim and isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Subscript)
                and isinstance(f.value.value, ast.Attribute)
                and f.value.value.attr == "at"):
            name = _base_name(f.value.value.value)
            if name is not None and COOLDOWN_ARRAY.match(name):
                fn = self.functions[-1] if self.functions else "<module>"
                if self.is_spells and fn in WRITERS:
                    self.allowed_cooldown_writes += 1
                else:
                    self.findings.append(
                        f"{self._where(node)}: rule 4: `{name}` written in "
                        f"`{fn}`, which is not in spells.COOLDOWN_WRITERS "
                        f"{WRITERS}")
        self.generic_visit(node)


def lint_source(src: str, path: str = "<string>") -> List[str]:
    linter = _Linter(path)
    linter.visit(ast.parse(src))
    return linter.findings


def _repo_files() -> List[Path]:
    files = sorted((ROOT / "lanerl_jax").rglob("*.py"))
    tools = ROOT / "tools"
    if tools.exists():
        files += sorted(tools.rglob("*.py"))
    return [f for f in files if "__pycache__" not in f.parts]


# -------------------------------------------------------------------- tests --
def test_the_repository_has_no_slot_buff_misuse():
    files = _repo_files()
    assert len(files) > 20, f"lint scanned suspiciously few files: {len(files)}"
    findings: List[str] = []
    for f in files:
        findings += lint_source(f.read_text(), str(f.relative_to(ROOT)))
    assert not findings, "slot/buff misuse:\n" + "\n".join(findings)


def test_the_lint_scans_the_files_that_carried_spell_005():
    """The lint must actually be looking where the bug was."""
    rel = {str(f.relative_to(ROOT)) for f in _repo_files()}
    for must in ("lanerl_jax/sim/step.py", "lanerl_jax/sim/spells.py",
                 "lanerl_jax/sim/orders.py", "lanerl_jax/obs/builder.py"):
        assert must in rel, must


def test_the_lint_sees_the_correct_sites_it_would_otherwise_miss():
    """Positive controls on the REAL sources: each rule's pattern exists in
    the code in its correct form, and mutating it into the bug fires."""
    step = (ROOT / "lanerl_jax" / "sim" / "step.py").read_text()
    spells = SPELLS.read_text()
    # rule 3: step.py's ghost read, reverted to the SPELL-005 line
    assert "pre_ghosted = status_of(state).ghosted" in step
    bad = step.replace("pre_ghosted = status_of(state).ghosted",
                       "pre_ghosted = (state.buff_id[:, Slot.E] == BuffId.GAREN_E)")
    assert any("rule 3" in f for f in lint_source(bad, "lanerl_jax/sim/step.py"))
    # rule 1: a real per-unit buff index, swapped for a spell slot
    site = "already_casting = buffs.r_pending.active[mirror_idx]"
    assert site in spells
    bad = spells.replace(site, "already_casting = buffs.r_pending.active[Slot.R]")
    assert any("rule 1" in f for f in lint_source(bad, "lanerl_jax/sim/spells.py"))
    # rule 4: the writer set really is where the writes are ...
    linter = _Linter("lanerl_jax/sim/spells.py")
    linter.visit(ast.parse(spells))
    assert not linter.findings, linter.findings
    assert linter.allowed_cooldown_writes >= len(WRITERS) - 1, (
        "the writer set's `.at` writes were not seen -- rule 4 is matching "
        "nothing", linter.allowed_cooldown_writes)
    # ... and a write from outside it (here: `consume_q_skip`) fires
    site = "def consume_q_skip(buffs: Buffs, consumed):\n"
    assert site in spells
    bad = spells.replace(site, site + "    spell_cooldown = spell_cooldown"
                         ".at[:, Slot.Q].set(0.0)\n")
    assert any("rule 4" in f for f in lint_source(bad, "lanerl_jax/sim/spells.py"))


# --- synthetic snippets: each rule must FIRE on the known bug shapes --------
#: `SPELL-005` as it was written against the lane table.
SPELL_005_STEP = """
from .spells import BuffId, Slot
pre_ghosted = (state.buff_id[:, Slot.E] == BuffId.GAREN_E) & state.alive
"""

#: the same mistake in the new shape: a spell slot where a UNIT goes.
SPELL_005_VARIANTS = [
    "x = state.buffs.e.active[Slot.E]",
    "x = bs.buffs.q.elapsed_s[..., Slot.Q]",
    "x = s.buffs.r_pending.rank[spells.Slot.R]",
    "x = st.buffs.w_passive[Slot.W]",
    "x = buffs.e.active.at[Slot.E].set(True)",
    "x = buffs.q_haste.active[b, lanerl_jax.sim.spells.Slot.Q]",
    "def f(buffs, slot=Slot.E):\n    return buffs.e.tick_acc_ms[slot]\n",
]


def test_rule3_fires_on_the_spell_005_site():
    found = lint_source(SPELL_005_STEP, "lanerl_jax/sim/step.py")
    assert any("rule 3" in f and "buff_id" in f for f in found), found
    assert any("rule 3" in f and "BuffId" in f for f in found), found


def test_rule1_fires_on_every_index_shape():
    for snippet in SPELL_005_VARIANTS:
        found = lint_source(snippet)
        assert any("rule 1" in f for f in found), (snippet, found)


def test_rule2_fires_on_a_buff_indexing_a_spell_array():
    for snippet in [
        "cd = state.spell_cooldown[:, state.buffs.q_haste.rank]",
        "r = state.spell_level[me, buffs.w.rank]",
        "c = st.cast_locked[:, E_BUFF_SLOT]",
        "c = can_cast[me, spells.Q_BUFF_SLOT]",
    ]:
        found = lint_source(snippet)
        assert any("rule 2" in f for f in found), (snippet, found)


def test_rule3_fires_on_the_retired_lane_table():
    for snippet in [
        "x = state.buff_elapsed[:, 0]",
        "b = buff_id.at[:, 0].set(1)",
        "from .spells import E_TICK_BUFF_SLOT",
        "s = s.replace(buff_power=p)",
        "from .state import MAX_BUFFS",
        "k = W_PASSIVE_BUFF_SLOT",
    ]:
        found = lint_source(snippet, "lanerl_jax/sim/step.py")
        assert any("rule 3" in f for f in found), (snippet, found)
    # dead code in the archive is exempt
    assert lint_source("x = state.buff_id[:, E_BUFF_SLOT]",
                       ARCHIVE + "old.py") == []


def test_rule4_fires_on_a_cooldown_write_outside_the_writer_set():
    snippet = ("def cast_x(spell_cooldown):\n"
               "    return spell_cooldown.at[:, Slot.E].set(0.0)\n")
    for path in ("lanerl_jax/sim/step.py", "lanerl_jax/sim/spells.py"):
        found = lint_source(snippet, path)
        assert any("rule 4" in f for f in found), (path, found)
    # a writer, in spells.py, is allowed; tests and non-sim code set up states
    ok = ("def end_e(buffs, spell_cooldown, ended, rank):\n"
          "    return spell_cooldown.at[:, Slot.E].set(13.0)\n")
    assert lint_source(ok, "lanerl_jax/sim/spells.py") == []
    assert lint_source(snippet, "lanerl_jax/sim/tests/test_x.py") == []
    assert lint_source(snippet, "lanerl_jax/parity/inject.py") == []


def test_the_lint_accepts_the_correct_forms():
    ok = """
from .spells import Slot, status_of
a = state.buffs.e.active[me]
b = state.spell_cooldown[:, Slot.E] <= 0
c = state.spell_level[me, Slot.Q]
d = status_of(state).cast_locked[me, Slot.E]
e = buffs.r_pending.active[mirror_idx]
f = st.can_cast[:, Slot.R]
g = bs.buffs.q_haste.active
def cast_e(buffs, spell_cooldown, slot=Slot.E):
    return buffs.e.active & (spell_cooldown[:, slot] <= 0)
"""
    assert lint_source(ok, "lanerl_jax/sim/orders.py") == []
