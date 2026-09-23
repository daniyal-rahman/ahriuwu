"""`RL-004` class gate: every field of every trainer config is READ.

`critic_lr` and `target_kl` were declared in `PPOConfig`, written into every
run manifest and read by nothing; so were `PolicyConfig.frame_stack`, the
unported reward flags and the policy's input widths. A recorded-but-unread
field makes the manifest lie in the direction of looking correct.

The rule, AST-only (no JAX, any interpreter), over ``lanerl_jax/**/*.py``
minus tests: for each config class in ``lanerl_jax/train/``, every field name
must appear as an attribute READ (``x.field``, or ``getattr(x, "field")``)
somewhere OUTSIDE the class body. A read inside one of the class's own
methods or properties counts only if that method is itself read from
outside (so ``PPOConfig.horizon_s`` counts through ``cfg.ppo.gamma``, while
the removed ``RewardConfig.alpha()`` would not have made the anneal fields
count had nothing called it).

Limitation, stated rather than hidden: it matches attribute NAMES, not
types. A field whose name is also an attribute of something else (``reward``
is also ``Transition.reward``) can be satisfied by the other object. It
catches the class that has actually happened here -- a field nobody reads
under any receiver -- and not a field that is read somewhere under the wrong
receiver.

Every NamedTuple/dataclass in ``lanerl_jax/train/`` must be classified below
as a config (checked) or not (state, results, logits), so a new config class
cannot slip past by name.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "lanerl_jax"
TRAIN = PKG / "train"

CONFIGS = {"PPOConfig", "PolicyConfig", "TrainConfig", "RewardConfig",
           "RewardWeights"}
NOT_CONFIGS = {"ActionLogits", "RunnerState", "Transition", "RewardState",
               "BenchResult", "ResetBenchResult"}


def _is_record(cls: ast.ClassDef) -> bool:
    bases = {getattr(b, "id", getattr(b, "attr", None)) for b in cls.bases}
    decos = {getattr(d, "id", getattr(getattr(d, "func", None), "id", None))
             for d in cls.decorator_list}
    return "NamedTuple" in bases or "dataclass" in decos


def _fields(cls: ast.ClassDef) -> List[str]:
    return [st.target.id for st in cls.body
            if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)]


def _methods(cls: ast.ClassDef) -> Dict[str, ast.FunctionDef]:
    return {st.name: st for st in cls.body if isinstance(st, ast.FunctionDef)}


def _node_read(n: ast.AST) -> str | None:
    """The attribute name ``n`` itself reads, if any."""
    if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
        return n.attr
    if (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "getattr"
            and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant)
            and isinstance(n.args[1].value, str)):
        return n.args[1].value
    return None


def _reads(node: ast.AST, skip: ast.AST | None = None) -> Set[str]:
    """Attribute reads anywhere under ``node``, not descending into ``skip``."""
    out: Set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n is skip:
            continue
        name = _node_read(n)
        if name is not None:
            out.add(name)
        stack.extend(ast.iter_child_nodes(n))
    return out


def unread_fields(config_trees: Iterable[Tuple[str, ast.Module]],
                  reader_trees: Iterable[Tuple[str, ast.Module]],
                  names: Set[str]) -> Dict[str, List[str]]:
    """``{class: [unread field, ...]}`` for the config classes in ``names``.

    ``reader_trees`` must include the files the configs are defined in (a
    read elsewhere in the same module counts; the class body does not).
    """
    reader_trees = list(reader_trees)
    out: Dict[str, List[str]] = {}
    for _, tree in config_trees:
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            if cls.name not in names:
                continue
            outside: Set[str] = set()
            for _, rt in reader_trees:
                outside |= _reads(rt, skip=cls)
            methods = _methods(cls)
            live = {m for m in methods if m in outside}
            read = set(outside)
            changed = True
            while changed:                      # methods reading methods
                changed = False
                for m in list(live):
                    inner = _reads(methods[m])
                    read |= inner
                    new = {k for k in methods if k in inner} - live
                    if new:
                        live |= new
                        changed = True
            missing = [f for f in _fields(cls) if f not in read]
            if missing:
                out[cls.name] = missing
    return out


def _parse(paths: Iterable[Path]) -> List[Tuple[str, ast.Module]]:
    return [(str(p.relative_to(ROOT)), ast.parse(p.read_text(), str(p)))
            for p in paths]


def _package_sources() -> List[Path]:
    return [p for p in sorted(PKG.rglob("*.py"))
            if "tests" not in p.relative_to(PKG).parts
            and "__pycache__" not in p.parts]


def test_every_record_class_in_train_is_classified():
    found = set()
    for _, tree in _parse(sorted(TRAIN.glob("*.py"))):
        found |= {n.name for n in ast.walk(tree)
                  if isinstance(n, ast.ClassDef) and _is_record(n)}
    unclassified = found - CONFIGS - NOT_CONFIGS
    assert not unclassified, (
        f"new NamedTuple/dataclass in lanerl_jax/train/: {sorted(unclassified)}. "
        "Add it to CONFIGS (fields must be read) or NOT_CONFIGS here.")
    assert CONFIGS <= found, f"config classes gone: {sorted(CONFIGS - found)}"


def test_every_trainer_config_field_is_read():
    readers = _parse(_package_sources())
    configs = [t for t in readers if t[0].startswith("lanerl_jax/train/")]
    bad = unread_fields(configs, readers, CONFIGS)
    assert not bad, (
        f"config fields declared and read by nothing (the RL-004 class): {bad}. "
        "Delete the field, or wire it in and record it in the manifest.")


# ---- the lint on a synthetic case: it must FAIL on an unread field -------
_SYNTH_CONFIG = '''
from typing import NamedTuple
class FooConfig(NamedTuple):
    used: int = 1
    via_property: int = 2
    via_dead_method: int = 3
    unread: int = 4
    @property
    def derived(self):
        return self.via_property * 2
    def dead(self):
        return self.via_dead_method
    def uses_unread_internally(self):
        return self.unread
'''
_SYNTH_READER = '''
def f(cfg):
    return cfg.used + cfg.derived + getattr(cfg, "nothing_else")
'''


def test_the_lint_flags_a_synthetic_unread_field():
    conf = ("synth/config.py", ast.parse(_SYNTH_CONFIG))
    reader = ("synth/reader.py", ast.parse(_SYNTH_READER))
    bad = unread_fields([conf], [conf, reader], {"FooConfig"})
    # `unread` is read only inside a method nobody calls, `via_dead_method`
    # only inside another uncalled method; `via_property` counts through
    # the property that IS read.
    assert bad == {"FooConfig": ["via_dead_method", "unread"]}, bad

    reader2 = ("synth/reader2.py", ast.parse(
        "def g(c):\n    return c.via_dead_method + c.uses_unread_internally()\n"))
    assert unread_fields([conf], [conf, reader, reader2], {"FooConfig"}) == {}
