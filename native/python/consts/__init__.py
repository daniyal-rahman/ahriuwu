"""Constants the native modules read by name (``lanesim::data``), one module per native area.

Each submodule defines ``consts() -> dict[str, float | sequence]`` computed with the same Python helpers the JAX
code calls (catalog ``dv``, rune ``ea``, champion ``values``/``cooldowns``, economy tables), so the float32 values
match what JAX bakes into its graph. Keys are ``"<area>.<name>"``.
"""
from __future__ import annotations

import importlib
import pkgutil

import numpy as np


def all_consts() -> dict[str, np.ndarray]:
    out = {}
    for info in pkgutil.iter_modules(__path__):
        mod = importlib.import_module(f"{__name__}.{info.name}")
        for k, v in mod.consts().items():
            if k in out:
                raise KeyError(f"constant {k} defined twice")
            out[k] = np.ascontiguousarray(np.asarray(v, np.float64).astype(np.float32).ravel())
    return out
