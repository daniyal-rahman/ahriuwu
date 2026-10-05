"""Bitwise trajectory fingerprint of the modern world tick, for behaviour-preserving refactors.

TOOL (MODERN-024). Runs the Garen-vs-Jax world for ``--ticks`` ticks on CPU under
"chaos" orders (a fixed PRNG stream, independent of the world's own key: moves,
attacks, attack-moves, casts at the nearest enemy or a point, summoners, item
actives, buys, wards, recalls, stops) and records a sha256 per state leaf at
every checkpoint. ``--compare`` checks a new run against a saved file and lists
the first leaves that differ.

    python -m ops.modern.golden --out before.json [--ticks 3600] [--every 600]
    python -m ops.modern.golden --compare before.json

Two worlds are fingerprinted: the full map, and top lane only without jungle and
objectives. Run both sides on the same backend (CPU by default) and JAX version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

WORLDS = {"full": dict(lanes=(0, 1, 2), jungle=True, objectives=True),
          "top": dict(lanes=(2,), jungle=False, objectives=False)}
ITEMS = (1055, 2003, 1036, 1001, 3044, 3071, 3078, 2055, 3340)


def chaos_orders(s, k, lane_mid):
    import jax
    import jax.numpy as jnp
    from lanerl_jax.modern import world as MS
    from lanerl_jax.modern.core import types as W
    c = 2
    ks = jax.random.split(k, 12)
    d = jnp.sqrt((s.x[None, :] - s.x[:c, None]) ** 2 + (s.y[None, :] - s.y[:c, None]) ** 2)
    enemy = (s.team[None, :] != s.team[:c, None]) & s.alive[None, :] & (s.kind[None, :] != W.KIND_NONE)
    near = jnp.argmin(jnp.where(enemy, d, jnp.inf), axis=1).astype(jnp.int32)
    kind = jax.random.randint(ks[0], (c,), 0, 12)
    pt = lane_mid[None, :] + jax.random.normal(ks[1], (c, 2)) * 1500.0
    tick = lambda p: jax.random.uniform(ks[2], (c,)) < p
    o = MS.no_orders()
    return o._replace(
        move=kind <= 2, move_x=pt[:, 0], move_y=pt[:, 1],
        attack=jnp.where((kind >= 3) & (kind <= 5), near, -1),
        attack_move=kind == 6,
        stop=(kind == 7) & tick(0.2),
        cast_slot=jnp.where((kind >= 8) & tick(0.5), jax.random.randint(ks[3], (c,), 0, 4), -1),
        cast_target=near, cast_x=pt[:, 0], cast_y=pt[:, 1],
        summoner_slot=jnp.where((kind == 9) & tick(0.05), jax.random.randint(ks[4], (c,), 0, 2), -1),
        summoner_target=near, summoner_x=pt[:, 0], summoner_y=pt[:, 1],
        item_active=jnp.where((kind == 10) & tick(0.1), jnp.asarray(ITEMS)[jax.random.randint(ks[5], (c,), 0, 9)], 0),
        buy=jnp.where((kind == 11) & tick(0.05), jnp.asarray(ITEMS)[jax.random.randint(ks[6], (c,), 0, 9)], 0),
        recall=(kind == 11) & tick(0.01),
        ward_kind=jnp.where((kind == 10) & tick(0.02), jax.random.randint(ks[7], (c,), 0, 2), -1),
        ward_x=pt[:, 0], ward_y=pt[:, 1])


def fingerprint(world: str, ticks: int, every: int) -> list[dict]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from lanerl_jax.modern.runes import catalog as RD
    from lanerl_jax.modern import world as MS
    from lanerl_jax.modern.world import config as MW
    lo = (MW.Loadout("Garen", items=(1055, 2003), rune_page=RD.GAREN_DEFAULT_PAGE),
          MW.Loadout("Jax", items=(1055, 2003), rune_page=RD.RunePage(
              RD.PRECISION, 8010, (9111, 9104, 8299), RD.RESOLVE, (8444, 8242), (5005, 5008, 5001))))
    cfg = MW.build_config(lo, **WORLDS[world])
    lane_mid = cfg.lane_path[cfg.lane_path.shape[0] // 2]
    key = jax.random.PRNGKey(1234)

    @jax.jit
    def run(s, t0):
        def body(s, i):
            s, _ = MS.step(s, chaos_orders(s, jax.random.fold_in(key, t0 + i), lane_mid), cfg)
            return s, None
        return jax.lax.scan(body, s, jnp.arange(every))[0]

    s = MS.init_state(cfg)
    names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(s)[0]]
    out = []
    for t in range(0, ticks, every):
        s = run(s, t)
        leaves = jax.tree_util.tree_leaves(s)
        out.append({"world": world, "tick": t + every,
                    "leaves": {n: hashlib.sha256(np.asarray(v).tobytes()).hexdigest()[:16]
                               for n, v in zip(names, leaves)}})
        print(json.dumps({"world": world, "tick": t + every, "game_s": float(s.t),
                          "alive": int(jnp.sum(s.alive & (s.kind != 0)))}), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticks", type=int, default=3600)
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--worlds", nargs="+", default=list(WORLDS))
    ap.add_argument("--out")
    ap.add_argument("--compare")
    args = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from lanerl_jax.jax_cache import enable_compile_cache
    enable_compile_cache()
    prints = [cp for w in args.worlds for cp in fingerprint(w, args.ticks, args.every)]
    if args.out:
        with open(args.out, "w") as f:
            json.dump(prints, f)
    if args.compare:
        ref = {(cp["world"], cp["tick"]): cp["leaves"] for cp in json.load(open(args.compare))}
        bad = 0
        for cp in prints:
            old = ref.get((cp["world"], cp["tick"]))
            if old is None:
                continue
            diff = sorted(k for k in set(old) | set(cp["leaves"]) if old.get(k) != cp["leaves"].get(k))
            if diff:
                bad += 1
                print(json.dumps({"world": cp["world"], "tick": cp["tick"], "differs": diff[:12],
                                  "n_differ": len(diff)}))
        print(json.dumps({"verdict": "identical" if bad == 0 else "differs", "checkpoints_differing": bad}))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
