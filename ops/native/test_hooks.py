"""Differential tests of individual native functions against their JAX originals (registered with LANESIM_TEST).

Each check calls the JAX function and the native one on the same pytree arguments and compares the flattened
outputs leaf by leaf (bit-exact, or reports the largest difference).

    python -m ops.native.test_hooks [names...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "native" / "python"))

ROUNDING = 1e-5          # float leaves within this relative difference count as rounding (XLA fuses FMAs)


def compare(name: str, want, got: list) -> bool:
    """``want``: JAX output pytree; ``got``: native flat leaves."""
    import jax
    from lanesim import _flat
    w = [np.zeros(0) if v is None else np.asarray(v) for v in _flat(want)]
    if len(w) != len(got):
        print(f"FAIL {name}: {len(got)} native leaves vs {len(w)} JAX")
        return False
    bad, rounding = [], [0]
    for i, (a, b) in enumerate(zip(w, got)):
        a = a.ravel()
        if a.shape != b.shape:
            bad.append((i, f"shape {a.shape} vs {b.shape}"))
            continue
        if a.dtype.kind == "f":
            a32 = a.astype(np.float32)
            eq = (a32 == b) | (np.isnan(a32) & np.isnan(b))
            close = eq | (np.abs(a32 - b) <= ROUNDING * np.maximum(np.maximum(np.abs(a32), np.abs(b)), 1.0))
            if not close.all():
                bad.append((i, f"max diff {np.max(np.abs(a32 - b)):.3g} at {np.flatnonzero(~close)[:4]}"))
            elif not eq.all():
                rounding[0] += 1
        elif not np.array_equal(a.astype(b.dtype), b):
            bad.append((i, f"values differ at {np.flatnonzero(a.astype(b.dtype) != b)[:4]}"))
    paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(
        want, is_leaf=lambda v: v is None)[0]]
    for i, msg in bad[:12]:
        print(f"  {name}{paths[i] if i < len(paths) else i}: {msg}")
    print(("ok  " if not bad else "FAIL") + f" {name}" + (f" ({len(bad)} leaves differ)" if bad else "")
          + (f" [{rounding[0]} leaves at rounding level]" if rounding[0] else ""))
    return not bad


def test_stats():
    import jax
    import jax.numpy as jnp

    from lanerl_jax.modern.core.stat_pipeline import compose
    from lanerl_jax.modern.items.catalog import ItemStats, combine_stats
    from ops.modern.golden import build
    import lanesim as LS
    cfg = build("top")
    rng = np.random.default_rng(0)
    ok = True
    for trial in range(20):
        bonus = ItemStats(*(jnp.asarray(rng.uniform(0, 1 if k in (13, 14, 21, 23, 6, 9) else 80, 2), jnp.float32)
                            for k in range(len(ItemStats._fields))))
        bonus2 = ItemStats(*(jnp.asarray(rng.uniform(0, 0.5, 2), jnp.float32) for _ in ItemStats._fields))
        level = jnp.asarray(rng.integers(1, 19, 2), jnp.int32)
        phys = jnp.asarray(rng.integers(0, 2, 2).astype(bool))
        slow = jnp.asarray(rng.uniform(0, 0.6, 2), jnp.float32)
        want = jax.jit(lambda b, l, s, p, sl: compose(b, l, s, adaptive_physical=p, slow=sl))(
            cfg.champion_base, level, bonus, phys, slow)
        ok &= compare("stats.compose", want, LS.call("stats.compose", cfg.champion_base, level, bonus, phys, slow))
        want = jax.jit(combine_stats)(bonus, bonus2)
        ok &= compare("stats.combine", want, LS.call("stats.combine", bonus, bonus2))
    return ok


CAPTURES = Path(os.environ.get("LANESIM_CAPTURES", "/mnt/nfs/shared/THROWAWAY-native001/captures"))


def replay(name: str, limit: int | None = None) -> bool:
    """Replay the captured calls of ``name`` (``ops/native/capture.py``) against the native function of the same
    name: positional arguments, then keyword arguments in call order."""
    import pickle
    import lanesim as LS
    calls = pickle.load(open(CAPTURES / f"{name}.pkl", "rb"))[:limit]
    ok = True
    for a, kw, res in calls:
        ok &= compare(name, res, LS.call(name, *a, *kw.values()))
    return ok


def test_captured(prefixes=()):
    """Every captured function that has a native registration (``prefixes`` filter names)."""
    import lanesim as LS
    native = set(LS.test_names())
    names = sorted(p.stem for p in CAPTURES.glob("*.pkl"))
    todo = [n for n in names if n in native and (not prefixes or any(n.startswith(p) for p in prefixes))]
    missing = [n for n in names if n not in native and (not prefixes or any(n.startswith(p) for p in prefixes))]
    if missing:
        print("not ported yet:", ", ".join(missing))
    return all([replay(n) for n in todo])


TESTS = {"stats": test_stats, "captured": test_captured}


def main() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    names = sys.argv[1:] or ["stats", "captured"]
    if names[0] == "captured":                       # captured [prefix ...]
        ok = test_captured(tuple(names[1:]))
    else:
        ok = all([TESTS[n]() for n in names])
    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
