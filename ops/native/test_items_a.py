"""Randomized differential test of the item effect modules A (consumables, starters, spellblade, hydra, boots,
actives) against their JAX originals.

The captured calls (``ops/native/capture.py``) rarely reach item effects, so this takes each captured call and
perturbs it: holders own random allow-listed items, module timers land around ``now``, item actives are requested,
attacks land on random units, casts start, kills happen, CC is applied. JAX runs eagerly on the perturbed
arguments and the native function must match (``test_hooks.compare``).

    python -m ops.native.test_items_a [module ...] [--rounds N]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

from ops.native.test_hooks import CAPTURES, compare

MODULES = ("consumables", "starters", "spellblade", "hydra", "boots", "actives")
# Discrete state fields drawn from their meaningful values; every other float leaf of a state is a timer/counter.
CHOICES = {
    "elixir": [0, 2138, 2139, 2140], "cast_item": [0, 3077, 6631], "consume_row": [-1], "drank": [0],
    "skill_points": [0, 1, 2],
}

FOCUS = {"consumables": (2003, 2031, 2138, 2139, 2140, 2010, 2150, 2151, 2152), "starters": (1054,),
         "spellblade": (3057, 3078, 2510, 3044), "hydra": (3077, 6631),
         "boots": (3006, 3009, 3047, 3111, 3170, 3172, 3173, 3174), "actives": (3157, 2420, 3142, 3146)}


def _perturb_state(st, now: float, n: int, rng):
    out = {}
    for f in st._fields:
        v = np.asarray(getattr(st, f))
        if f in CHOICES:
            out[f] = rng.choice(np.asarray(CHOICES[f], v.dtype), v.shape)
        elif f in ("dd_extra_target", "aim_unit"):
            out[f] = rng.integers(-1, min(n, 4), v.shape).astype(v.dtype)
        elif v.dtype == bool:
            out[f] = rng.random(v.shape) < 0.3
        elif v.dtype.kind == "f":
            keep = rng.random(v.shape) < 0.3
            timer = now + rng.uniform(-3.0, 3.0, v.shape)
            small = rng.integers(0, 4, v.shape).astype(np.float64)
            out[f] = np.where(keep, v, np.where(rng.random(v.shape) < 0.6, timer, small)).astype(v.dtype)
        else:
            out[f] = v
    import jax.numpy as jnp
    return type(st)(**{k: jnp.asarray(v) for k, v in out.items()})


def _perturb(module: str, hook: str, args: list, items, rng):
    from lanerl_jax.modern.items.catalog import catalog
    from lanerl_jax.modern.items.effects.core import Owned
    cat = catalog()
    args = list(args)
    ctx = args[2] if len(args) > 2 else None
    now = float(np.asarray(ctx.now)) if ctx is not None else 0.0
    n = np.asarray(args[3].x).shape[0] if len(args) > 3 else 88
    args[0] = _perturb_state(args[0], now, n, rng)
    own = args[1]
    counts = np.zeros_like(np.asarray(own.counts))
    for c in range(counts.shape[0]):
        pool = [i for i in items[c] if own.allowed[c, cat.row(i)]]
        focus = [i for i in pool if i in FOCUS.get(module, ())]
        pick = list(rng.choice(pool, rng.integers(0, 4), replace=True)) if pool else []
        pick += list(rng.choice(focus, rng.integers(0, 3), replace=True)) if focus else []
        for i in pick:
            counts[c, cat.row(int(i))] += 1
    args[1] = Owned(counts.astype(np.int32), own.allowed)
    if ctx is not None:
        c = np.asarray(ctx.alive).shape[0]
        hp = (np.asarray(ctx.max_hp) * rng.uniform(0.05, 1.0, c)).astype(np.float32)
        args[2] = ctx._replace(alive=rng.random(c) < 0.9, hp=hp, in_shop=rng.random(c) < 0.5,
                               in_combat=rng.random(c) < 0.5,
                               attack_windup=rng.uniform(0.0, 0.4, c).astype(np.float32))
    if len(args) > 3:
        u = args[3]
        alive = np.asarray(u.alive)
        args[3] = u._replace(hp=(np.asarray(u.max_hp) * rng.uniform(0.0, 1.0, n)).astype(np.float32))
        live = np.flatnonzero(alive)
    if len(args) > 4:
        ev = args[4]
        name = type(ev).__name__
        c = 2
        if name == "Attack":
            tgt = rng.choice(np.concatenate([live, [-1, 0, 1]]), c).astype(np.int32)
            args[4] = ev._replace(launched=rng.random(c) < 0.7, hit=rng.random(c) < 0.8, target=tgt,
                                  raw=rng.uniform(0, 200, c).astype(np.float32))
        elif name == "Cast":
            args[4] = ev._replace(started=rng.random(c) < 0.7, slot=rng.integers(0, 4, c).astype(np.int32))
        elif name == "Kills":
            args[4] = ev._replace(champion_kill=rng.integers(0, 2, c).astype(np.float32),
                                  champion_assist=rng.integers(0, 2, c).astype(np.float32),
                                  minion_kill=rng.integers(0, 3, c).astype(np.float32),
                                  holder_died=rng.random(c) < 0.2)
        elif name == "CC":
            m = np.asarray(ev.slowed).shape
            args[4] = ev._replace(slowed=rng.random(m) < 0.05, immobilized=rng.random(m) < 0.05)
        elif name == "Report":
            pk, rs = ev.packets, ev.resolved
            m = np.asarray(pk.valid).shape[0]
            if m:
                units = np.concatenate([[0, 1, 0, 1], live]).astype(np.int32)
                flags = np.asarray([0, 1, 2, 16, 8 | 65536, 32 | 256, 16 | 1, 1312], np.int32)
                pk = pk._replace(valid=rng.random(m) < 0.6, src=rng.choice(units, m), dst=rng.choice(units, m),
                                 dtype=rng.integers(0, 3, m).astype(np.int32), flags=rng.choice(flags, m),
                                 item=rng.choice(np.asarray([0, 0, 2139, 1054], np.int32), m))
                rs = rs._replace(final=np.where(rng.random(m) < 0.8, rng.uniform(0, 300, m), 0.0).astype(np.float32))
                args[4] = ev._replace(packets=pk, resolved=rs)
        elif hook == "active":
            own_items = {"consumables": [2003, 2031, 2138, 2139, 2140, 2010, 2150, 2151, 2152],
                         "hydra": [3077, 6631], "actives": [3157, 2420, 3142, 3146]}.get(module, [])
            pool = own_items * 3 + sorted({i for row in items for i in row}) + [0, 0, 0]
            args[4] = rng.choice(np.asarray(pool, np.int32), c)
    return args


def _activity(res) -> int:
    """1 if the result shows an item effect (valid packet, heal, mana, gold, shield, slow, active used)."""
    from lanerl_jax.modern.items.effects.core import ActiveOut, Effects
    parts = res if isinstance(res, tuple) and not hasattr(res, "_fields") else (res,)
    for r in parts:
        if isinstance(r, Effects):
            vals = [r.packets.valid, r.heal, r.heal_plain, r.mana, r.gold, r.shields.amount, r.slow]
            if any(np.asarray(v).any() for v in vals):
                return 1
        if isinstance(r, ActiveOut) and np.asarray(r.used).any():
            return 1
    return 0


def run(module: str, rounds: int, rng) -> bool:
    import importlib

    import lanesim as LS
    from lanerl_jax.modern.data.loadouts import allowed_items
    from lanerl_jax.modern.items.loadout import acquirable_rows
    from lanerl_jax.modern.items.catalog import catalog
    cat = catalog()
    items = [[cat.ids[r] for r in np.flatnonzero(acquirable_rows(allowed_items(ch)))] for ch in ("Garen", "Jax")]
    mod = importlib.import_module(f"lanerl_jax.modern.items.effects.{module}")
    ok = True
    for path in sorted(CAPTURES.glob(f"items.{module}.*.pkl")):
        name = path.stem
        hook = name.rsplit(".", 1)[1]
        calls = pickle.load(open(path, "rb"))
        fn = getattr(mod, hook)
        good, active = 0, 0
        for r in range(rounds):
            a, kw, _ = calls[r % len(calls)]
            args = _perturb(module, hook, list(a) + list(kw.values()), items, rng)
            want = fn(*args)
            active += _activity(want)
            if compare(name, want, LS.call(name, *args)):
                good += 1
            else:
                ok = False
        print(f"== {name}: {good}/{rounds} perturbed calls match ({active} with item effects)")
    return ok


def main() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="*", default=list(MODULES))
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    ok = all([run(m, args.rounds, rng) for m in args.modules])
    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
