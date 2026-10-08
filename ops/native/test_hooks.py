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
    from lanesim import _flat, _prepare
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
        _prepare(want), is_leaf=lambda v: v is None)[0]]
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


def test_actives_world():
    """items.effects.actives functions the world phases call outside the hook dispatch (no captures)."""
    import jax
    import jax.numpy as jnp

    from lanerl_jax.modern.items.effects import actives as A
    import lanesim as LS
    rng = np.random.default_rng(1)
    ok = True
    ids = np.asarray([0, *A.ACTIVE_ITEMS, 3077, 2003], np.int32)
    for trial in range(20):
        s = A.init(2, 88)
        s = s._replace(stasis_until=jnp.asarray(rng.uniform(-5, 15, 2), jnp.float32),
                       actualizer_until=jnp.asarray(rng.uniform(-5, 15, 2), jnp.float32),
                       cleanse_now=jnp.asarray(rng.integers(0, 2, 2).astype(bool)),
                       dash_now=jnp.asarray(rng.integers(0, 2, 2).astype(bool)),
                       dash_x=jnp.asarray(rng.uniform(0, 15000, 2), jnp.float32),
                       dash_y=jnp.asarray(rng.uniform(0, 15000, 2), jnp.float32),
                       shatter_now=jnp.asarray(rng.integers(0, 2, 2).astype(bool)))
        now = jnp.float32(rng.uniform(0, 10))
        ok &= compare("items.actives.world", jax.jit(A.world)(s, now), LS.call("items.actives.world", s, now))
        unit = jnp.asarray(rng.integers(-1, 88, 2), jnp.int32)
        x, y = (jnp.asarray(rng.uniform(0, 15000, 2), jnp.float32) for _ in range(2))
        ok &= compare("items.actives.with_aim", jax.jit(A.with_aim)(s, unit, x, y),
                      LS.call("items.actives.with_aim", s, unit, x, y))
        ok &= compare("items.actives.with_aim", A.with_aim(s, unit), LS.call("items.actives.with_aim", s, unit,
                                                                               None, None))
        req = jnp.asarray(rng.choice(ids, 2), jnp.int32)
        dis, stas = (jnp.asarray(rng.integers(0, 2, 2).astype(bool)) for _ in range(2))
        ok &= compare("items.actives.request_allowed",
                      jax.jit(lambda r, d, i: A.request_allowed(r, disabled=d, in_stasis=i))(req, dis, stas),
                      LS.call("items.actives.request_allowed", req, dis, stas))
        ok &= compare("items.actives.request_allowed", A.request_allowed(req, disabled=dis),
                      LS.call("items.actives.request_allowed", req, dis, None))
    return ok


def test_econ():
    """Direct JAX-vs-native checks of the economy/quest/ward/inventory/region helpers the world phases call outside
    the captured economy_step / ward_step / ward_view (random inputs, jitted JAX)."""
    import jax
    import jax.numpy as jnp

    from lanerl_jax.modern import economy as E
    from lanerl_jax.modern import role_quest as Q
    from lanerl_jax.modern import wards as WD
    from lanerl_jax.modern.core import types as W
    from lanerl_jax.modern.items import inventory as I
    from lanerl_jax.modern.items.catalog import catalog
    from lanerl_jax.modern.map import regions as REG
    from lanerl_jax.modern.map.lanes import LANE_PATHS
    from ops.modern.golden import build
    import lanesim as LS
    cfg = build("top")
    cat = catalog()
    a = cat.arrays
    n_rows = len(cat.ids)
    rng = np.random.default_rng(1)
    ok = True
    f32, i32 = np.float32, np.int32

    # economy helpers
    k = 256
    xp = rng.uniform(0, 30000, k).astype(f32)
    xp[:40] = np.asarray(E._tables()["need"])[rng.integers(0, 22, 40)]          # exact level boundaries
    cap = rng.choice([18, 20], k).astype(i32)
    ok &= compare("economy.decimal_level", jax.jit(E.decimal_level)(xp, cap), LS.call("economy.decimal_level", xp, cap))
    ok &= compare("economy.level_for_xp", jax.jit(E.level_for_xp)(xp, cap), LS.call("economy.level_for_xp", xp, cap))
    lv = rng.integers(0, 22, k).astype(i32)
    ok &= compare("economy.ranks", jax.jit(lambda l: (E.skill_points(l), E.max_rank(l), E.max_rank(l, ultimate=True)))(lv),
                  LS.call("economy.ranks", lv))
    fx = np.asarray(cfg.fountain)[rng.integers(0, 2, k)]
    px = (fx[:, 0] + rng.normal(0, 900, k)).astype(f32)
    py = (fx[:, 1] + rng.normal(0, 900, k)).astype(f32)
    ok &= compare("economy.in_fountain", jax.jit(E.in_fountain)(px, py, fx[:, 0].astype(f32), fx[:, 1].astype(f32)),
                  LS.call("economy.in_fountain", px, py, fx[:, 0].astype(f32), fx[:, 1].astype(f32)))
    for trial in range(10):
        mh = rng.uniform(500, 3000, k).astype(f32)
        hp = (mh * rng.uniform(0, 1, k)).astype(f32)
        mm = rng.uniform(0, 1500, k).astype(f32)
        mana = (mm * rng.uniform(0, 1, k)).astype(f32)
        inf_ = rng.integers(0, 2, k).astype(bool)
        hg = rng.integers(0, 2, k).astype(bool)
        t0 = f32(rng.uniform(0, 1500))
        t1 = f32(t0 + (1 / 30 if trial < 8 else rng.uniform(0, 3)))
        want = jax.jit(lambda *v: E.fountain_regen(*v[:7], homeguard=v[7]))(hp, mh, mana, mm, inf_, t0, t1, hg)
        ok &= compare("economy.fountain_regen", want, LS.call("economy.fountain_regen", hp, mh, mana, mm, inf_, t0, t1, hg))
    ok &= compare("economy.starting_gold", np.float32(E.starting_gold()), LS.call("economy.starting_gold", np.zeros(1)))
    roles = np.asarray([Q.ROLE_TOP, Q.ROLE_TOP], i32)
    ok &= compare("economy.init_economy", E.init_economy(2, cfg.n_units, roles),
                  LS.call("economy.init_economy", i32(2), i32(cfg.n_units), roles))

    # role quest
    qs_jit = jax.jit(lambda st, ev, now, dt, il, al, lv, rc: Q.quest_step(st, ev, now=now, dt=dt, in_lane=il, alive=al,
                                                                          level=lv, recalled=rc))
    for trial in range(40):
        st = Q.QuestState(np.asarray(rng.choice([0, 1, 1], 2), i32), rng.uniform(0, 1300, 2).astype(f32),
                          rng.integers(0, 2, 2).astype(bool), rng.uniform(0, 70, 2).astype(f32),
                          rng.uniform(0, 200, 2).astype(f32))
        ev = Q.QuestEvents(*(rng.integers(0, 3, 2).astype(f32) for _ in range(8)))
        args = (st, ev, f32(rng.uniform(0, 300)), f32(1 / 30), rng.integers(0, 2, 2).astype(bool),
                rng.integers(0, 2, 2).astype(bool), rng.integers(1, 19, 2).astype(i32), rng.integers(0, 2, 2).astype(bool))
        ok &= compare("role_quest.quest_step", qs_jit(*args), LS.call("role_quest.quest_step", *args))
        mil = rng.integers(0, 2, 88).astype(bool)
        ok &= compare("role_quest.minion_penalty", jax.jit(Q.minion_penalty)(st, args[6], mil),
                      LS.call("role_quest.minion_penalty", st, args[6], mil))

    # inventory / shop
    ids = np.asarray(a.item_id)
    stealth = cat.row(3340)
    recipes = [r for r in range(n_rows) if (np.asarray(a.node_item)[r] >= 0).sum() > 1]

    def random_inv():
        item = np.full(7, -1, i32)
        stack = np.zeros(7, i32)
        n = rng.integers(0, 7)
        rows = rng.integers(0, n_rows, n)
        if rng.uniform() < 0.5 and recipes:                      # components of a recipe
            r = recipes[rng.integers(len(recipes))]
            comp = [c for c in np.asarray(a.node_item)[r][1:] if c >= 0]
            rows = np.asarray(list(rng.permutation(comp))[:6] + list(rows))[:6]
        for s_, r in enumerate(rows[:6]):
            item[s_], stack[s_] = r, rng.integers(1, max(int(a.max_stack[r]), 1) + 1)
        if rng.uniform() < 0.9:
            item[6], stack[6] = stealth, 1
        return I.Inventory(item, stack), (recipes[rng.integers(len(recipes))] if rng.uniform() < 0.5
                                          else rng.integers(0, n_rows))
    buy_jit = jax.jit(lambda inv, g, r, cs, lv, rg, now, gcd: I.buy(inv, g, r, can_shop=cs, level=lv, is_ranged=rg,
                                                                     now=now, group_cd_until=gcd))
    sell_jit = jax.jit(lambda inv, g, sl, cs: I.sell(inv, g, sl, can_shop=cs))
    codes = set()
    for trial in range(300):
        inv, row = random_inv()
        gold = f32(rng.choice([0.0, rng.uniform(0, 4000), 1e5]))
        gcd = np.where(rng.uniform(size=a.group_max.shape[0]) < 0.3, rng.uniform(0, 200, a.group_max.shape[0]),
                       0).astype(f32)
        args = (inv, gold, i32(row), np.bool_(rng.uniform() < 0.9), i32(rng.integers(1, 19)), np.bool_(rng.uniform() < 0.3),
                f32(rng.uniform(0, 300)), gcd)
        want = buy_jit(*args)
        codes.add(int(want.code))
        ok &= compare("inventory.buy", want, LS.call("inventory.buy", *args, None))
        slot = i32(rng.integers(0, 7))
        args = (inv, gold, slot, np.bool_(rng.uniform() < 0.9))
        ok &= compare("inventory.sell", sell_jit(*args), LS.call("inventory.sell", *args))
        frm, to = i32(rng.choice([inv.item[rng.integers(7)], rng.integers(0, n_rows)])), i32(rng.integers(0, n_rows))
        en = np.bool_(rng.uniform() < 0.8)
        ok &= compare("inventory.replace_item", jax.jit(I.replace_item)(inv, frm, to, en),
                      LS.call("inventory.replace_item", inv, frm, to, en))
        ok &= compare("inventory.consume_one", jax.jit(I.consume_one)(inv, slot, en),
                      LS.call("inventory.consume_one", inv, slot, en))
    print("     buy result codes exercised:", sorted(codes))
    for trial in range(20):
        invs = [random_inv()[0] for _ in range(2)]
        inv2 = I.Inventory(np.stack([v.item for v in invs]), np.stack([v.stack for v in invs]))
        ok &= compare("inventory.owned_counts", jax.jit(I.owned_counts)(inv2), LS.call("inventory.owned_counts", inv2))
        ok &= compare("inventory.inventory_stats", jax.jit(I.inventory_stats)(inv2),
                      LS.call("inventory.inventory_stats", inv2))
    team = rng.integers(0, 2, k).astype(i32)
    sx = np.asarray(I.SHOP_CENTER, f32)[team]
    x = (sx[:, 0] + rng.normal(0, 800, k)).astype(f32)
    z = (sx[:, 1] + rng.normal(0, 800, k)).astype(f32)
    dead = rng.uniform(size=k) < 0.1
    ok &= compare("inventory.in_shop_area", jax.jit(I.in_shop_area)(x, z, team, dead),
                  LS.call("inventory.in_shop_area", x, z, team, dead))
    for lo in cfg.loadouts:
        ids_ = [list(lo.items), list(lo.items)[:2]]
        width = max(len(v) for v in ids_)
        pad = np.asarray([v + [0] * (width - len(v)) for v in ids_], i32)
        for tr in (True, False):
            ok &= compare("inventory.inventory_from_ids", I.inventory_from_ids(ids_, trinket=tr),
                          LS.call("inventory.inventory_from_ids", pad, i32(width), np.bool_(tr)))

    # wards: init and vision_kwargs (ward_step / ward_view are captured)
    for tids in ([3340, 3340], [3364, 3363], [3363, 1001]):
        ok &= compare("wards.init_wards", WD.init_wards(2, tids), LS.call("wards.init_wards", i32(2), np.asarray(tids, i32)))
    s0 = __import__("lanerl_jax.modern.world", fromlist=["init_state"]).init_state(cfg)
    lay = cfg.layout
    for trial in range(10):
        w = s0.wards
        sl = w.slots
        S = sl.x.shape[0]
        sl = sl._replace(alive=rng.uniform(size=S) < 0.6, type=rng.integers(0, 3, S).astype(i32),
                         x=rng.uniform(0, 15000, S).astype(f32), y=rng.uniform(0, 15000, S).astype(f32),
                         placed_at=rng.uniform(0, 100, S).astype(f32),
                         triggered_at=np.where(rng.uniform(size=S) < 0.3, rng.uniform(0, 100, S), np.inf).astype(f32),
                         revealed_until=rng.uniform(0, 120, S).astype(f32),
                         disabled_until=rng.uniform(0, 120, S).astype(f32), tracked=rng.uniform(size=S) < 0.3)
        w = w._replace(slots=sl, trinket=w.trinket._replace(oracle_until=rng.uniform(80, 120, 2).astype(f32)))
        now = f32(100.0)
        cx, cy = rng.uniform(0, 15000, 2).astype(f32), rng.uniform(0, 15000, 2).astype(f32)
        args = dict(now=now, x=cx, y=cy, team=np.asarray([0, 1], i32), alive=np.asarray([True, rng.uniform() < .8]),
                    level=rng.integers(1, 19, 2).astype(i32))
        view, oracle = jax.jit(lambda w, a_: WD.ward_view(w, **a_))(w, args)
        ok &= compare("wards.ward_view(direct)", (view, oracle), LS.call("wards.ward_view", w, *(args[k_] for k_ in sorted(args))))
        kind = np.asarray(s0.kind).copy()
        kind[lay.ward0:lay.ward0 + S] = np.where(view.alive, W.KIND_WARD, W.KIND_NONE)
        sub = np.asarray(s0.sub).copy()
        sub[lay.ward0:lay.ward0 + S] = view.sub
        alive = np.asarray(s0.alive).copy() | (rng.uniform(size=kind.shape[0]) < 0.5)
        want = jax.jit(lambda *v: WD.vision_kwargs(*v, ward_start=lay.ward0))(view, oracle, kind, sub, alive)
        ok &= compare("wards.vision_kwargs", tuple(want[k_] for k_ in ("radius", "stealthed", "true_sight", "unobstructed",
                                                                       "exposed")),
                      LS.call("wards.vision_kwargs", view, oracle, kind, sub, alive, i32(lay.ward0)))

    # map regions: point queries, lane progress, Homeguard flags
    paths = np.asarray(LANE_PATHS, f32)
    pts = paths[rng.integers(0, 2, 2000), rng.integers(0, 3, 2000), rng.integers(0, paths.shape[2], 2000)]
    px = np.concatenate([(pts[:, 0] + rng.normal(0, 700, 2000)), rng.uniform(-500, 15500, 500)]).astype(f32)
    py = np.concatenate([(pts[:, 1] + rng.normal(0, 700, 2000)), rng.uniform(-500, 15500, 500)]).astype(f32)
    px[:3], py[:3] = [np.nan, np.inf, 5000.0], [100.0, 5000.0, -np.inf]
    reg = cfg.regions
    want = jax.jit(lambda x_, y_: (REG.region_of(x_, y_, reg), REG.lane_of(x_, y_, reg), REG.in_quest_lane(x_, y_, 2, reg),
                                   REG.in_jungle(x_, y_, reg), REG.in_river(x_, y_, reg)))(px, py)
    ok &= compare("regions.queries", want, LS.call("regions.queries", px, py))
    tm, ln = rng.integers(0, 2, px.shape[0]).astype(i32), rng.integers(0, 3, px.shape[0]).astype(i32)
    fin = np.isfinite(px) & np.isfinite(py)
    ok &= compare("regions.lane_progress", jax.jit(REG.lane_progress)(px[fin], py[fin], tm[fin], ln[fin]),
                  LS.call("regions.lane_progress", px[fin], py[fin], tm[fin], ln[fin]))
    units0 = __import__("lanerl_jax.modern.world.units", fromlist=["units_view"]).units_view(s0)
    n = cfg.n_units
    hf_jit = jax.jit(lambda x_, y_, t_, now, u, sl_, ml: REG.homeguard_flags(x_, y_, t_, now, u, sl_, ml, reg))
    for trial in range(60):
        kind = np.asarray(units0.kind).copy()
        mslots = np.arange(n)[kind == W.KIND_NONE]
        kind[mslots] = np.where(rng.uniform(size=mslots.size) < 0.7, W.KIND_MINION, W.KIND_NONE)
        lane_pts = paths[rng.integers(0, 2, n), 2, rng.integers(0, paths.shape[2], n)]
        ux = np.where(kind == W.KIND_MINION, lane_pts[:, 0] + rng.normal(0, 300, n), units0.x).astype(f32)
        uy = np.where(kind == W.KIND_MINION, lane_pts[:, 1] + rng.normal(0, 300, n), units0.y).astype(f32)
        u = units0._replace(kind=kind.astype(i32), x=ux, y=uy,
                            alive=np.asarray(units0.alive) | (rng.uniform(size=n) < 0.8) & (rng.uniform(size=n) < 0.9),
                            team=np.where(kind == W.KIND_MINION, rng.integers(0, 2, n), units0.team).astype(i32))
        minion_lane = np.where(kind == W.KIND_MINION, rng.choice([2, 2, 2, 1], n), -1).astype(i32)
        cp = paths[[0, 1], 2, rng.integers(0, paths.shape[2], 2)]
        cxy = (cp + rng.normal(0, 500, (2, 2))).astype(f32)
        args = (cxy[:, 0], cxy[:, 1], np.asarray([0, 1], i32), f32(rng.choice([100.0, 900.0])), u,
                np.asarray(cfg.unit_lane, i32), minion_lane)
        ok &= compare("regions.homeguard_flags", hf_jit(*args), LS.call("regions.homeguard_flags", *args))
    return ok


CAPTURES = Path(os.environ.get("LANESIM_CAPTURES", "/mnt/nfs/shared/THROWAWAY-native001/captures"))


# Natives taking keyword arguments in the JAX signature's order (captures store them sorted by name).
SIGNATURE_ORDER = {"combat.combat_tick": "lanerl_jax.modern.combat:combat_tick"}


def replay(name: str, limit: int | None = None) -> bool:
    """Replay the captured calls of ``name`` (``ops/native/capture.py``) against the native function of the same
    name: positional arguments, then keyword arguments (sorted, or in signature order: SIGNATURE_ORDER)."""
    import importlib
    import inspect
    import pickle
    import lanesim as LS
    calls = pickle.load(open(CAPTURES / f"{name}.pkl", "rb"))[:limit]
    order = None
    if name in SIGNATURE_ORDER:
        mod, fn = SIGNATURE_ORDER[name].split(":")
        order = list(inspect.signature(getattr(importlib.import_module(mod), fn)).parameters)
    ok = True
    for a, kw, res in calls:
        if order:
            kw = {k: kw[k] for k in order if k in kw}
        try:
            ok &= compare(name, res, LS.call(name, *a, *kw.values()))
        except Exception as exc:                                         # noqa: BLE001
            print(f"FAIL {name}: {exc}")
            return False
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


TESTS = {"stats": test_stats, "econ": test_econ, "actives_world": test_actives_world, "captured": test_captured}


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
