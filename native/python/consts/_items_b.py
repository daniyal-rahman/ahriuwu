"""Shared builder of the item-module constant tables (items_fighter/defense/mage/marksman)."""
from __future__ import annotations

import importlib
import inspect
import re

_DV = re.compile(r'\bdv\((\w+),\s*"(\w+)"\)')


def consts() -> dict:
    return {}


def module_consts(name: str, names: tuple[str, ...], extra_dv: tuple = ()) -> dict:
    """``items.<name>.dv.<item id>.<data value>`` for every ``dv(ITEM, "Name")`` in the JAX module source, plus
    ``extra_dv`` (id, name) pairs and ``items.<name>.<NAME>`` for the listed module-level constants."""
    from lanerl_jax.modern.items.effects.core import dv
    mod = importlib.import_module(f"lanerl_jax.modern.items.effects.{name}")
    out = {}
    pairs = [(getattr(mod, sym), value) for sym, value in _DV.findall(inspect.getsource(mod)) if hasattr(mod, sym)]
    for iid, value in [*pairs, *extra_dv]:
        out[f"items.{name}.dv.{iid}.{value}"] = dv(iid, value)
    for n in names:
        v = getattr(mod, n)
        out[f"items.{name}.{n}"] = list(v) if isinstance(v, (tuple, list)) else float(v)
    return out
