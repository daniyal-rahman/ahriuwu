"""Bitwise trajectory fingerprint of the modern world tick, for behaviour-preserving refactors.

TOOL (MODERN-024). Runs the Garen-vs-Jax world for ``--ticks`` ticks on CPU under
"chaos" orders (a fixed PRNG stream, independent of the world's own key: moves,
attacks, attack-moves, casts at the nearest enemy or a point, summoners, item
actives, buys, wards, recalls, stops) and records a sha256 per state leaf at
every checkpoint, plus a game-level ``summary`` (champions, minion counts and HP,
structure HP) that does not depend on the unit layout. ``--compare`` checks a new run
against a saved file: identical leaves (same layout), else the summary differences
(e.g. a resized world).

    python -m ops.modern.golden --out before.json [--ticks 3600] [--every 600]
    python -m ops.modern.golden --compare before.json
    python -m ops.modern.golden --diff before.json after.json

Two worlds are fingerprinted: the full map, and top lane only without jungle and
objectives. Run both sides on the same backend (CPU by default) and JAX version.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

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


def build(world: str, allowed: bool = False):
    """The Garen-vs-Jax ``WorldConfig`` of ``WORLDS[world]``; ``allowed`` restricts both champions to the items the
    chaos orders buy (behaviour must not change)."""
    from lanerl_jax.modern.runes import catalog as RD
    from lanerl_jax.modern.world import config as MW
    shop = ITEMS if allowed else ()
    lo = (MW.Loadout("Garen", items=(1055, 2003), rune_page=RD.GAREN_DEFAULT_PAGE, allowed_items=shop),
          MW.Loadout("Jax", items=(1055, 2003), rune_page=RD.RunePage(
              RD.PRECISION, 8010, (9111, 9104, 8299), RD.RESOLVE, (8444, 8242), (5005, 5008, 5001)),
              allowed_items=shop))
    return MW.build_config(lo, **WORLDS[world])


def summary(s) -> dict:
    """Layout-independent game state: champions, lane minions per team, structures (the last 30 slots)."""
    import numpy as np

    from lanerl_jax.modern.core import types as W
    a = lambda v: np.asarray(v, np.float64)                          # noqa: E731
    minion = (a(s.kind) == W.KIND_MINION) & a(s.alive).astype(bool)
    team = a(s.team)
    return {"t": float(s.t), "x": a(s.x[:2]).tolist(), "y": a(s.y[:2]).tolist(), "hp": a(s.hp[:2]).tolist(),
            "mana": a(s.champ.mana).tolist(), "gold": a(s.econ.gold).tolist(), "xp": a(s.econ.xp).tolist(),
            "cs": a(s.champ.cs).tolist(),
            "minions": [int((minion & (team == t)).sum()) for t in (0, 1)],
            "minion_hp": [float(a(s.hp)[minion & (team == t)].sum()) for t in (0, 1)],
            "structure_hp": a(s.hp[-30:]).tolist()}


def fingerprint(world: str, ticks: int, every: int, allowed: bool = False) -> list[dict]:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from lanerl_jax.modern import world as MS
    cfg = build(world, allowed)
    lane_mid = cfg.lane_path[cfg.lane_path.shape[0] // 2]
    key = jax.random.PRNGKey(1234)

    @jax.jit
    def run(s, t0):
        def body(s, i):
            s, e = MS.step(s, chaos_orders(s, jax.random.fold_in(key, t0 + i), lane_mid), cfg)
            return s, jnp.stack([jnp.sum(e.report.packets.valid), jnp.sum(e.follow_up.packets.valid),
                                 e.packet_overflow, e.missile_overflow, e.ray_overflow]).astype(jnp.int32)
        s, use = jax.lax.scan(body, s, jnp.arange(every))
        return s, jnp.max(use, axis=0)

    s = MS.init_state(cfg)
    names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(s)[0]]
    out = []
    for t in range(0, ticks, every):
        s, use = run(s, t)
        leaves = jax.tree_util.tree_leaves(s)
        peak = dict(zip(("packets_max", "follow_up_max", "packet_overflow", "missile_overflow", "ray_overflow"),
                        map(int, np.asarray(use))))
        out.append({"world": world, "tick": t + every, "summary": summary(s), "peak": peak,
                    "leaves": {n: hashlib.sha256(np.asarray(v).tobytes()).hexdigest()[:16]
                               for n, v in zip(names, leaves)}})
        print(json.dumps({"world": world, "tick": t + every, "game_s": float(s.t),
                          "alive": int(jnp.sum(s.alive & (s.kind != 0))), **peak}), flush=True)
    return out


def compare(ref_prints: list[dict], prints: list[dict]) -> int:
    """Print the differences of ``prints`` against ``ref_prints``; returns the number of differing checkpoints.

    Same leaf names: leaf-by-leaf. Otherwise (renamed, regrouped or removed leaves) every new leaf value must
    occur in the old state; leaves only in the old state are listed as dropped."""
    ref = {(cp["world"], cp["tick"]): cp for cp in ref_prints}
    bad = 0
    for cp in prints:
        old = ref.get((cp["world"], cp["tick"]))
        if old is None:
            continue
        a, b = old["leaves"], cp["leaves"]
        if set(a) == set(b):
            diff = sorted(k for k in a if a[k] != b[k])
        else:
            left = collections.Counter(a.values())
            left.subtract(b.values())
            diff = sorted(k for k, h in b.items() if left[h] < 0)
            count = collections.Counter(a.values())
            gone = sum(max(v, 0) for v in left.values())
            named = sorted(k for k, h in a.items() if left[h] > 0 and count[h] == 1)   # unambiguous only
            print(json.dumps({"world": cp["world"], "tick": cp["tick"], "leaves_dropped": gone,
                              "dropped_unique": named[:12]}))
        if diff:
            bad += 1
            gap = {k: float(np.max(np.abs(np.asarray(v, float) - np.asarray(old["summary"][k], float))))
                   for k, v in cp["summary"].items() if k in old.get("summary", {})}
            print(json.dumps({"world": cp["world"], "tick": cp["tick"], "differs": diff[:12],
                              "n_differ": len(diff), "summary_max_abs_diff": gap}))
    print(json.dumps({"verdict": "identical" if bad == 0 else "differs", "checkpoints_differing": bad}))
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticks", type=int, default=3600)
    ap.add_argument("--every", type=int, default=600)
    ap.add_argument("--worlds", nargs="+", default=list(WORLDS))
    ap.add_argument("--allowed", action="store_true", help="restrict the shop to the items the chaos orders buy")
    ap.add_argument("--out")
    ap.add_argument("--compare", help="saved run to check this run against")
    ap.add_argument("--diff", nargs=2, metavar=("REF", "NEW"), help="compare two saved runs (no simulation)")
    args = ap.parse_args()
    if args.diff:
        sys.exit(1 if compare(json.load(open(args.diff[0])), json.load(open(args.diff[1]))) else 0)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from lanerl_jax.modern.jax_cache import enable_compile_cache
    enable_compile_cache()
    prints = [cp for w in args.worlds for cp in fingerprint(w, args.ticks, args.every, args.allowed)]
    if args.out:
        with open(args.out, "w") as f:
            json.dump(prints, f)
    if args.compare:
        sys.exit(1 if compare(json.load(open(args.compare)), prints) else 0)


if __name__ == "__main__":
    main()
