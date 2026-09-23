"""`STRUCT-001` gate: an AST lint for the two index spaces of Garen's kit.

The kit has two small-integer index spaces that read alike and are not:

* ``Slot.Q/W/E/R`` = 0..3 index the SPELL arrays ``spell_level`` and
  ``spell_cooldown`` (``CharData`` ``Spell1``-``Spell4``);
* ``*_BUFF_SLOT`` = 0..6 index the LANES of the ``(N, 8)`` buff table
  ``buff_id`` / ``buff_elapsed`` / ``buff_duration`` / ``buff_power``.

``Slot.E`` is 2 and ``E_BUFF_SLOT`` is 0, so ``buff_id[:, Slot.E]`` reads lane 2
(``W_PASSIVE_BUFF_SLOT``) and compares identically False against
``BuffId.GAREN_E`` -- ``SPELL-005``: Garen never ghosted while spinning
(``step.py``, and the same line in ``parity/archive/truncation_bound_probe.py``);
``STRUCT-001`` counts the ``Slot.E``/``E_BUFF_SLOT`` collision at three sites,
and nothing failed on any of them. This lint flags, over ``lanerl_jax/**/*.py`` and
``tools/**/*.py``:

1. a ``Slot.<X>`` in the LANE index (the last index) of a buff-table subscript,
   including ``.at[...]`` updates;
2. a ``<X>_BUFF_SLOT`` (or a function parameter whose default is one, e.g.
   ``cast_e(..., slot=E_BUFF_SLOT)``) in the index of ``spell_level`` /
   ``spell_cooldown``;
3. ``buff_id[..., L] == BuffId.<X>`` (either side, ``==`` or ``!=``), and a
   ``BuffId.<X>`` written by ``buff_id.at[..., L].set(...)``, where ``<X>``
   cannot live in lane ``L``.

The constants are read from ``spells.py`` by parsing it, not importing it, so
this file needs no JAX and runs under any interpreter.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[2]
SPELLS = ROOT / "lanerl_jax" / "sim" / "spells.py"

BUFF_ARRAY = re.compile(r"^(buff_id|buff_elapsed|buff_duration|buff_power)(_\w+)?$")
SPELL_ARRAY = re.compile(r"^(spell_level|spell_cooldown)(_\w+)?$")
BUFF_ID_ARRAY = re.compile(r"^buff_id(_\w+)?$")


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


def _load_constants():
    tree = ast.parse(SPELLS.read_text())
    slot = _class_ints(tree, "Slot")
    buff = _class_ints(tree, "BuffId")
    lanes: Dict[str, int] = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.endswith("_BUFF_SLOT")):
            lanes[node.targets[0].id] = ast.literal_eval(node.value)
    return slot, buff, lanes


SLOT, BUFF_ID, LANES = _load_constants()

#: Which buff ids may live in which lane. ``NONE`` is legal everywhere.
#: Lane 6 (``E_TICK_BUFF_SLOT``) is E's millisecond accumulator: scratch, never
#: a buff, so only ``NONE``.
ALLOWED_IN_LANE: Dict[int, Set[str]] = {
    LANES["E_BUFF_SLOT"]: {"GAREN_E"},
    LANES["W_BUFF_SLOT"]: {"GAREN_W"},
    LANES["W_PASSIVE_BUFF_SLOT"]: {"GAREN_W_PASSIVE"},
    LANES["Q_BUFF_SLOT"]: {"GAREN_Q"},
    LANES["Q_HASTE_BUFF_SLOT"]: {"GAREN_Q_HASTE"},
    LANES["R_PENDING_BUFF_SLOT"]: {"GAREN_R_PENDING"},
    LANES["E_TICK_BUFF_SLOT"]: set(),
}


def test_the_lane_table_covers_every_lane_and_every_buff_id():
    """Guard the guard: a new lane or buff id must be placed in the table."""
    assert set(ALLOWED_IN_LANE) == set(LANES.values()), (LANES, ALLOWED_IN_LANE)
    placed = set().union(*ALLOWED_IN_LANE.values())
    assert placed == set(BUFF_ID) - {"NONE"}, (placed, BUFF_ID)
    assert set(SLOT) == {"Q", "W", "E", "R"}


# --------------------------------------------------------------------- lint --
def _base_name(node: ast.AST) -> Optional[str]:
    """``state.buff_id`` / ``bs.buff_id`` / ``buff_id`` -> ``"buff_id"``.
    ``x.at`` -> the name of ``x`` (the ``.at[...]`` update form)."""
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


def _lane_const_name(node: ast.AST) -> Optional[str]:
    """``E_BUFF_SLOT`` / ``spells.E_BUFF_SLOT`` -> ``"E_BUFF_SLOT"``."""
    if isinstance(node, ast.Name) and node.id in LANES:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in LANES:
        return node.attr
    return None


def _buff_id_attr(node: ast.AST) -> Optional[str]:
    """``BuffId.GAREN_E`` / ``spells.BuffId.GAREN_E`` -> ``"GAREN_E"``."""
    if (isinstance(node, ast.Attribute) and node.attr in BUFF_ID
            and _base_name(node.value) == "BuffId"):
        return node.attr
    return None


def _subscript_parts(sub: ast.Subscript):
    """``(array_name, lane_index_expr, all_index_exprs)`` of a subscript.

    ``x[:, L]`` / ``x[i, L]`` / ``x[..., L]`` / ``x[b, i, L]``: the lane is the
    LAST index. ``x[i][L]``: an outer subscript on a row of a named array, the
    lane is the outer index. ``x[L]`` on a bare name: treated the same (a row
    variable, or a unit index that happens to be a lane/slot constant -- a bug
    either way).
    """
    sl = sub.slice
    if isinstance(sl, ast.Index):          # py<3.9 AST
        sl = sl.value                       # pragma: no cover
    value = sub.value
    name = _base_name(value.value if isinstance(value, ast.Subscript) else value)
    elts = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
    return name, elts[-1], elts


class _Linter(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.findings: List[str] = []
        # Per-function aliases: a parameter whose default is a lane constant
        # (`slot=E_BUFF_SLOT`) or a spell slot (`slot=Slot.E`).
        self.lane_alias: List[Dict[str, int]] = [{}]
        self.slot_alias: List[Dict[str, str]] = [{}]

    # -- scopes --
    def _visit_function(self, node):
        lanes: Dict[str, int] = {}
        slots: Dict[str, str] = {}
        args = node.args
        pos = args.posonlyargs + args.args
        defaults = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)
        pairs = list(zip(pos, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
        for a, d in pairs:
            if d is None:
                continue
            c = _lane_const_name(d)
            if c is not None:
                lanes[a.arg] = LANES[c]
            s = _is_slot_attr(d)
            if s is not None:
                slots[a.arg] = s
        self.lane_alias.append({**self.lane_alias[-1], **lanes})
        self.slot_alias.append({**self.slot_alias[-1], **slots})
        self.generic_visit(node)
        self.lane_alias.pop()
        self.slot_alias.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # -- helpers --
    def _where(self, node) -> str:
        return f"{self.path}:{node.lineno}"

    def _lane_value(self, expr: ast.AST) -> Optional[int]:
        c = _lane_const_name(expr)
        if c is not None:
            return LANES[c]
        if isinstance(expr, ast.Name) and expr.id in self.lane_alias[-1]:
            return self.lane_alias[-1][expr.id]
        if isinstance(expr, ast.Constant) and isinstance(expr.value, int) \
                and not isinstance(expr.value, bool):
            return expr.value
        return None

    def _slot_in(self, expr: ast.AST) -> Optional[str]:
        for n in ast.walk(expr):
            s = _is_slot_attr(n)
            if s is not None:
                return f"Slot.{s}"
            if isinstance(n, ast.Name) and n.id in self.slot_alias[-1]:
                return f"{n.id}(=Slot.{self.slot_alias[-1][n.id]})"
        return None

    def _lane_in(self, expr: ast.AST) -> Optional[str]:
        for n in ast.walk(expr):
            c = _lane_const_name(n)
            if c is not None:
                return c
            if isinstance(n, ast.Name) and n.id in self.lane_alias[-1]:
                return f"{n.id}(=lane {self.lane_alias[-1][n.id]})"
        return None

    def _buff_id_lane(self, node: ast.AST) -> Optional[int]:
        """If ``node`` is ``buff_id[..., L]`` with a resolvable ``L``, return L."""
        if not isinstance(node, ast.Subscript):
            return None
        name, lane, _ = _subscript_parts(node)
        if name is None or not BUFF_ID_ARRAY.match(name):
            return None
        return self._lane_value(lane)

    # -- rules --
    def visit_Subscript(self, node: ast.Subscript):
        name, lane, elts = _subscript_parts(node)
        if name is not None and BUFF_ARRAY.match(name):
            s = self._slot_in(lane)
            if s is not None:
                self.findings.append(
                    f"{self._where(node)}: rule 1: spell slot {s} used as the "
                    f"LANE index of buff array `{name}` -- use a *_BUFF_SLOT")
        if name is not None and SPELL_ARRAY.match(name):
            for e in elts:
                c = self._lane_in(e)
                if c is not None:
                    self.findings.append(
                        f"{self._where(node)}: rule 2: buff lane {c} used to "
                        f"index spell array `{name}` -- use Slot.<X>")
                    break
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        operands = [node.left] + list(node.comparators)
        for op, a, b in zip(node.ops, operands, operands[1:]):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            for arr, const in ((a, b), (b, a)):
                lane = self._buff_id_lane(arr)
                bid = _buff_id_attr(const)
                if lane is None or bid is None or bid == "NONE":
                    continue
                if bid not in ALLOWED_IN_LANE.get(lane, set()):
                    self.findings.append(
                        f"{self._where(node)}: rule 3: BuffId.{bid} compared "
                        f"against buff lane {lane}, which only ever holds "
                        f"{sorted(ALLOWED_IN_LANE.get(lane, set())) or ['NONE']}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # buff_id.at[..., L].set(<expr containing BuffId.X>)
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == "set"
                and isinstance(f.value, ast.Subscript)
                and isinstance(f.value.value, ast.Attribute)
                and f.value.value.attr == "at"):
            name = _base_name(f.value.value.value)
            if name is not None and BUFF_ID_ARRAY.match(name):
                _, lane_expr, _ = _subscript_parts(f.value)
                lane = self._lane_value(lane_expr)
                if lane is not None:
                    for arg in list(node.args) + [k.value for k in node.keywords]:
                        for n in ast.walk(arg):
                            bid = _buff_id_attr(n)
                            if bid is None or bid == "NONE":
                                continue
                            if bid not in ALLOWED_IN_LANE.get(lane, set()):
                                self.findings.append(
                                    f"{self._where(node)}: rule 3: BuffId.{bid} "
                                    f"written into buff lane {lane}")
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
def test_the_repository_has_no_slot_lane_misuse():
    files = _repo_files()
    assert len(files) > 20, f"lint scanned suspiciously few files: {len(files)}"
    findings: List[str] = []
    for f in files:
        findings += lint_source(f.read_text(), str(f.relative_to(ROOT)))
    assert not findings, "slot/lane misuse:\n" + "\n".join(findings)


def test_the_lint_scans_the_files_that_carried_spell_005():
    """The lint must actually be looking where the bug was."""
    rel = {str(f.relative_to(ROOT)) for f in _repo_files()}
    for must in ("lanerl_jax/sim/step.py", "lanerl_jax/sim/spells.py",
                 "lanerl_jax/sim/orders.py", "lanerl_jax/obs/builder.py"):
        assert must in rel, must


def test_the_lint_sees_the_correct_sites_it_would_otherwise_miss():
    """Positive control: the real, correct sites parse into the shapes the
    rules look for, so a clean run is not just a lint that matches nothing."""
    src = (ROOT / "lanerl_jax" / "sim" / "step.py").read_text()
    assert "state.buff_id[:, E_BUFF_SLOT] == BuffId.GAREN_E" in src
    # If the constant were swapped for Slot.E the lint must fire (rule 1 AND 3).
    bad = src.replace("state.buff_id[:, E_BUFF_SLOT] == BuffId.GAREN_E",
                      "state.buff_id[:, Slot.E] == BuffId.GAREN_E")
    found = lint_source(bad, "step.py(mutated)")
    assert any("rule 1" in f for f in found), found


# --- synthetic snippets: each rule must FIRE on the known bug shapes --------
SPELL_005_STEP = """
from .spells import BuffId, Slot
pre_ghosted = (state.buff_id[:, Slot.E] == BuffId.GAREN_E) & state.alive
"""

SPELL_005_VARIANTS = [
    "x = bs.buff_id[i, Slot.E]",
    "x = buff_elapsed[..., Slot.E]",
    "x = s.buff_duration[:, spells.Slot.E]",
    "x = s.buff_power[me][Slot.Q]",
    "x = buff_id.at[:, Slot.E].set(0)",
    "x = buff_id_out[b, i, lanerl_jax.sim.spells.Slot.R]",
]


def test_rule1_fires_on_the_spell_005_site():
    found = lint_source(SPELL_005_STEP)
    assert any("rule 1" in f and "Slot.E" in f for f in found), found


def test_rule1_fires_on_every_index_shape():
    for snippet in SPELL_005_VARIANTS:
        found = lint_source(snippet)
        assert any("rule 1" in f for f in found), (snippet, found)


def test_rule2_fires_on_a_lane_constant_indexing_a_spell_array():
    for snippet in [
        "cd = state.spell_cooldown[:, E_BUFF_SLOT]",
        "cd = spell_cooldown.at[:, spells.Q_BUFF_SLOT].set(8.0)",
        "r = state.spell_level[me, W_BUFF_SLOT]",
        # a function parameter defaulting to a lane is a lane
        "def cast_e(buff_id, spell_cooldown, slot=E_BUFF_SLOT):\n"
        "    return spell_cooldown[:, slot]\n",
    ]:
        found = lint_source(snippet)
        assert any("rule 2" in f for f in found), (snippet, found)


def test_rule3_fires_on_a_buff_id_that_cannot_live_in_that_lane():
    for snippet in [
        # Slot.E resolved as a lane would be lane 2; here written with the lane
        # constant of the WRONG lane, which only rule 3 can see.
        "a = state.buff_id[:, W_PASSIVE_BUFF_SLOT] == BuffId.GAREN_E",
        "a = BuffId.GAREN_Q != buff_id[me, E_BUFF_SLOT]",
        "a = buff_id[..., 2] == BuffId.GAREN_E",
        "a = buff_id[:, E_TICK_BUFF_SLOT] == BuffId.GAREN_E",
        "def cast_q(buff_id, slot=Q_BUFF_SLOT):\n"
        "    return buff_id[:, slot] == BuffId.GAREN_E\n",
        "b = buff_id.at[:, Q_HASTE_BUFF_SLOT].set(jnp.where(m, jnp.int8(BuffId.GAREN_Q), 0))",
    ]:
        found = lint_source(snippet)
        assert any("rule 3" in f for f in found), (snippet, found)


def test_the_lint_accepts_the_correct_forms():
    ok = """
from .spells import BuffId, Slot, E_BUFF_SLOT, Q_BUFF_SLOT
a = state.buff_id[:, E_BUFF_SLOT] == BuffId.GAREN_E
b = state.spell_cooldown[:, Slot.E] <= 0
c = state.spell_level[me, Slot.Q]
d = buff_id.at[:, Q_BUFF_SLOT].set(jnp.where(r, jnp.int8(BuffId.GAREN_Q), BuffId.NONE))
e = buff_id[:, E_BUFF_SLOT] == BuffId.NONE
def cast_e(buff_id, spell_cooldown, slot=E_BUFF_SLOT):
    return (buff_id[:, slot] == BuffId.GAREN_E) & (spell_cooldown[:, Slot.E] <= 0)
"""
    assert lint_source(ok) == []
