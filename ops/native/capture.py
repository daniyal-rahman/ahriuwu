"""Capture real (arguments, result) pairs of JAX module functions from a running world, for native replay tests.

The listed functions are wrapped so that every call also emits a ``jax.debug.callback`` with its arguments and
result; the world runs jitted (Garen vs Jax top lane, allow-lists, chaos orders) and every ``--stride``-th call
of each function, up to ``--per``, is pickled to ``<out>/<name>.pkl`` as a list of ``(args, result)`` numpy
pytrees. ``ops/native/test_hooks.py captured`` replays them against the registered native functions.

    python -m ops.native.capture --ticks 6000 --out /mnt/nfs/shared/THROWAWAY-native001/captures
"""
from __future__ import annotations

import argparse
import collections
import importlib
import os
import pickle
from pathlib import Path

# name -> (module, function): the module attribute is replaced, so callers that look it up at call time see it.
ITEM_HOOKS = ("stats", "defense", "status", "debuffs", "dealt_amp", "packet_amp", "attack_mods", "on_attack",
              "on_hit", "on_cast", "on_cc", "on_damage", "on_takedown", "periodic", "active", "on_shop")
RUNE_HOOKS = ("stats", "debuffs", "packet_amp", "packet_block", "heal_mult", "on_cast", "on_attack", "on_hit",
              "on_cc", "periodic", "on_damage", "on_takedown", "post_tick", "outputs")
KIT_HOOKS = ("cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats", "defense",
             "attack_mods", "debuffs", "ghosted", "dodging")


def targets() -> dict:
    from lanerl_jax.modern.items import effects as IE
    from lanerl_jax.modern.runes import effects as RE
    out = {}
    for m in IE.MODULES:
        name = m.__name__.rsplit(".", 1)[-1]
        out.update({f"items.{name}.{h}": (m, h) for h in ITEM_HOOKS if hasattr(m, h)})
    for m in RE.MODULES:
        name = m.__name__.rsplit(".", 1)[-1]
        out.update({f"runes.{name}.{h}": (m, h) for h in RUNE_HOOKS if hasattr(m, h)})
    for mod in ("garen", "jax"):
        m = importlib.import_module(f"lanerl_jax.modern.champions.{mod}")
        out.update({f"kits.{mod}.{h}": (m, h) for h in KIT_HOOKS if hasattr(m, h)})
    extra = {"summoners.step": ("lanerl_jax.modern.champions.summoners", "step"),
             "economy.economy_step": ("lanerl_jax.modern.economy", "economy_step"),
             "wards.ward_step": ("lanerl_jax.modern.wards", "ward_step"),
             "wards.ward_view": ("lanerl_jax.modern.wards", "ward_view"),
             "combat.combat_tick": ("lanerl_jax.modern.combat", "combat_tick"),
             "damage.resolve": ("lanerl_jax.modern.core.damage", "resolve")}
    for name, (mod, fn) in extra.items():
        out[name] = (importlib.import_module(mod), fn)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticks", type=int, default=6000)
    ap.add_argument("--stride", type=int, default=37, help="keep every stride-th call of each function")
    ap.add_argument("--per", type=int, default=60, help="calls kept per function")
    ap.add_argument("--only", nargs="*", default=None, help="name prefixes to capture (default all)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import numpy as np

    from lanerl_jax.modern import world as MS
    from lanerl_jax.modern.jax_cache import enable_compile_cache
    from ops.modern.golden import chaos_orders
    enable_compile_cache()
    seen = collections.Counter()
    kept = collections.defaultdict(list)
    to_np = lambda tree: jax.tree.map(lambda v: np.asarray(v), tree)      # noqa: E731

    def wrap(name, fn):
        def record(call):
            seen[name] += 1
            if seen[name] % args.stride == 1 and len(kept[name]) < args.per:
                kept[name].append(to_np(call))

        def spied(*a, **kw):
            res = fn(*a, **kw)
            jax.debug.callback(record, (a, kw, res))
            return res
        return spied

    for name, (mod, fn) in targets().items():
        if args.only and not any(name.startswith(p) for p in args.only):
            continue
        setattr(mod, fn, wrap(name, getattr(mod, fn)))
    from ops.modern.bench import build_world
    cfg, _ = build_world(argparse.Namespace(fog="rays", no_jungle=True, no_objectives=True, lanes=[2],
                                            packet_capacity=0, allowlist=True))
    lane_mid = cfg.lane_path[cfg.lane_path.shape[0] // 2]
    key = jax.random.PRNGKey(7)
    jnp = jax.numpy
    from lanerl_jax.modern.data.loadouts import allowed_items
    pool = [sorted(allowed_items(n)) for n in ("Garen", "Jax")]
    width = max(len(p) for p in pool)
    pool = jnp.asarray([p + p[:width - len(p)] for p in pool], jnp.int32)          # (C, width) item ids

    def orders(s, k):
        """Golden chaos orders, buying and pressing actives of random allow-listed items (all reachable items
        get exercised; the champions start rich)."""
        o = chaos_orders(s, k, lane_mid)
        k1, k2, k3 = jax.random.split(jax.random.fold_in(k, 99), 3)
        pick = lambda kk: pool[jnp.arange(2), jax.random.randint(kk, (2,), 0, width)]     # noqa: E731
        buy = jnp.where(jax.random.uniform(k1, (2,)) < 0.02, pick(k1), 0)
        act = jnp.where(jax.random.uniform(k2, (2,)) < 0.03, pick(k2), 0)
        return o._replace(buy=jnp.where(o.buy > 0, o.buy, buy), item_active=jnp.where(o.item_active > 0,
                                                                                   o.item_active, act),
                          recall=o.recall | (jax.random.uniform(k3, (2,)) < 0.002))

    @jax.jit
    def run(s, t0):
        def body(s, i):
            return MS.step(s, orders(s, jax.random.fold_in(key, t0 + i)), cfg)[0], None
        return jax.lax.scan(body, s, jax.numpy.arange(300))[0]

    s = MS.init_state(cfg)
    s = s._replace(econ=s.econ._replace(gold=s.econ.gold + 30000.0))
    for t in range(0, args.ticks, 300):
        s = run(s, t)
        jax.block_until_ready(s.t)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, calls in kept.items():
        with open(args.out / f"{name}.pkl", "wb") as f:
            pickle.dump(calls, f)
    print({k: (seen[k], len(kept[k])) for k in sorted(kept)})


if __name__ == "__main__":
    main()
