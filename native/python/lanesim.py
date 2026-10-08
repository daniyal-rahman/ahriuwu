"""ctypes binding of liblanesim (the native lane-slice tick) and its mapping to the JAX world.

``NativeWorld(cfg)`` builds the C++ ``World`` from a JAX ``WorldConfig`` (layout, terrain gap tables, route graph,
vision grid, lane paths, and the few float32 constants XLA evaluates: avoidance rotations, separation fallback
directions, eject rings). The native env has one field per ``ModernState`` leaf (``ops/native/gen_state.py``) plus
native caches; ``env_from_state`` / ``state_from_env`` convert, ``orders_from`` converts ``ModernOrders``.
``step`` advances an env held in numpy arrays; ``Batch`` runs many envs the library owns.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

LIB = Path(os.environ.get("LANESIM_LIB", Path(__file__).resolve().parents[1] / "build-clang" / "liblanesim.so"))
_CTYPES = {"float": np.float32, "int32_t": np.int32, "uint8_t": np.uint8, "uint32_t": np.uint32}


def _lib():
    lib = ctypes.CDLL(str(LIB))
    P, I, L, D, S = ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_double, ctypes.c_char_p
    sig = {"ls_env_fields": ([], S), "ls_order_fields": ([], S), "ls_world_env_counts": ([P, P, L], None),
           "ls_world_new": ([], P),
           "ls_world_free": ([P], None), "ls_world_int": ([P, S, L], I), "ls_world_float": ([P, S, D], I),
           "ls_world_array": ([P, S, P, L], I), "ls_world_finish": ([P], None),
           "ls_step": ([P, P, P, P], None), "ls_step_full": ([P, P, P, P], None), "ls_batch_new": ([P, I, P, I], P), "ls_batch_free": ([P], None),
           "ls_batch_run": ([P, I, I, P], None), "ls_batch_get": ([P, I, P], None),
           "ls_batch_env_bytes": ([P], L), "ls_profile": ([P, I], None), "ls_debug_route": ([P], None),
           "ls_test_names": ([], S), "ls_test_signature": ([S], S), "ls_test_call": ([S, P, P], P),
           "ls_result_leaves": ([P], L), "ls_result_code": ([P, L], ctypes.c_char), "ls_result_count": ([P, L], L),
           "ls_result_data": ([P, L], P), "ls_result_free": ([P], None),
           "ls_data_put": ([S, P, L], None), "ls_world_table": ([P, S, P, L], None), "ls_env_index": ([S], L)}
    for name, (args, res) in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, res
    return lib


_L = None


def lib():
    """The library, with the named constants (native/python/consts) loaded on first use."""
    global _L
    if _L is None:
        _L = _lib()
        import consts
        for k, v in consts.all_consts().items():
            _L.ls_data_put(k.encode(), v.ctypes.data, v.size)
    return _L


def _fields(text: bytes) -> list[tuple[str, str]]:
    return [tuple(item.split()[::-1]) for item in text.decode().split(";") if item]


def env_fields() -> list[tuple[str, str]]:
    """``(name, ctype)`` per native env field, in struct order."""
    return _fields(lib().ls_env_fields())


def order_fields() -> list[tuple[str, str]]:
    return _fields(lib().ls_order_fields())


def _leaves(tree) -> list[tuple[str, np.ndarray]]:
    import jax
    return [(jax.tree_util.keystr(p).lstrip(".").replace(".", "_"), np.asarray(v))
            for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]]


def xla_constants(cfg) -> dict:
    """The float32 constants the JAX tick evaluates with XLA's cos/sin, computed the same way."""
    import jax.numpy as jnp

    from lanerl_jax.modern import collision as UC
    from lanerl_jax.modern.map import dynamic_terrain as DTR
    ang = np.deg2rad(np.asarray(UC.AVOID_ANGLES_DEG))
    signs = jnp.asarray([1.0, -1.0], jnp.float32)
    cos = np.stack([np.asarray(jnp.cos(signs[k] * a)) for a in ang for k in range(2)]).reshape(6, 2)
    sin = np.stack([np.asarray(jnp.sin(signs[k] * a)) for a in ang for k in range(2)]).reshape(6, 2)
    if not np.array_equal(cos[:, 0], cos[:, 1]):
        raise ValueError("XLA cos is not even on the avoidance angles")
    m = cfg.layout.ward0
    idx = jnp.arange(m)
    lo = jnp.minimum(idx[:, None], idx[None, :]).astype(jnp.float32)
    hi = jnp.maximum(idx[:, None], idx[None, :]).astype(jnp.float32)
    a = UC._GOLDEN * (lo * 7.0 + hi)
    sgn = jnp.where(idx[:, None] < idx[None, :], 1.0, -1.0)
    d_ang = jnp.arange(DTR._EJECT_DIRS) * (2 * jnp.pi / DTR._EJECT_DIRS)
    rings = jnp.asarray(DTR._EJECT_RINGS, jnp.float32)
    return dict(avoid_cos=np.ascontiguousarray(cos[:, 0], np.float32),
                avoid_sin=np.ascontiguousarray(sin.T, np.float32),           # [turn sign][angle]
                sep_fx=np.asarray(sgn * jnp.cos(a), np.float32).ravel(), sep_fy=np.asarray(sgn * jnp.sin(a), np.float32).ravel(),
                eject_dx=np.asarray((rings[:, None] * jnp.cos(d_ang)[None, :]).reshape(-1), np.float32),
                eject_dy=np.asarray((rings[:, None] * jnp.sin(d_ang)[None, :]).reshape(-1), np.float32),
                eject_r=np.asarray(jnp.repeat(rings, DTR._EJECT_DIRS), np.float32))


def champion_tables(cfg) -> dict:
    """Per-world champion data for ``World::tables`` (float32 tables; ints exact)."""
    from lanerl_jax.modern import champions as K
    from lanerl_jax.modern.core.stat_pipeline import ChampionBase
    from lanerl_jax.modern.items.catalog import ItemStats
    from lanerl_jax.modern.items.loadout import stat_shard_stats
    from lanerl_jax.modern.runes import catalog as R
    from lanerl_jax.modern.world import views as V
    c = len(cfg.loadouts)
    order = np.zeros((c, 20), np.int64)
    for i in range(c):
        so = list(V.skill_order(cfg, i))
        order[i, :len(so)] = so[:20]
    shard_const, shard_hs = [], []
    for lo in cfg.loadouts:
        shards = lo.rune_page.shards
        st = stat_shard_stats(shards, level=1, adaptive_to_ad=None, xp=np)
        hs = R.SHARD_HEALTH_SCALING in shards
        # Health scaling is the only level-dependent shard (slot 3, so the only health shard when present): its
        # health is lin(start, end, level), added per tick; everything else is constant.
        shard_const.append([0.0 if (hs and f == "health") else float(getattr(st, f)) for f in ItemStats._fields])
        start, end = (R.ea(R.SHARD_HEALTH_SCALING, "StatGainMin"), R.ea(R.SHARD_HEALTH_SCALING, "StatGainMax"))
        shard_hs.append([float(hs), start, end - start])
    out = {
        "champion_ids": np.asarray(cfg.champion_ids), "rune_pages": np.asarray(cfg.rune_pages),
        "champion_base": np.stack([np.asarray(getattr(cfg.champion_base, f)) for f in ChampionBase._fields]),
        "adaptive_physical": np.asarray(cfg.adaptive_physical), "uses_energy": np.asarray(cfg.uses_energy),
        "fountain": np.asarray(cfg.fountain), "skill_order": order,
        "auto_skill": np.asarray([lo.auto_skill for lo in cfg.loadouts]),
        "unit_target_ranges": np.asarray(K.unit_target_ranges(cfg.champion_ids)),
        "shard_const": np.asarray(shard_const), "shard_hs": np.asarray(shard_hs),
    }
    if cfg.item_allowed is not None:
        out["item_allowed"] = np.asarray(cfg.item_allowed)
    return out


class NativeWorld:
    """The C++ ``World`` of a JAX ``WorldConfig`` (keeps the borrowed tables alive)."""

    def __init__(self, cfg):
        from lanerl_jax.modern import collision as UC
        from lanerl_jax.modern.map.lanes import BARRACKS, LANE_PATH_LEN, LANE_PATHS
        from lanerl_jax.modern.map.terrain import row_gaps
        L = lib()
        self.cfg = cfg
        self.ptr = L.ls_world_new()
        self._keep = []
        lay = cfg.layout
        ints = dict(n=cfg.n_units, minion0=lay.minion0, monster0=lay.monster0, epic0=lay.epic0, ward0=lay.ward0,
                    struct0=lay.struct0, missiles=64, packet_capacity=lay.packet_capacity,
                    ray_capacity=lay.ray_capacity, fog=int(cfg.vision is not None), path_cap=LANE_PATHS.shape[2])
        t0 = cfg.terrain[0]
        walk = np.asarray(t0.walkable)
        ints.update(terrain_height=walk.shape[0], terrain_width=walk.shape[1])
        rt = cfg.routes
        ints.update(route_points=np.asarray(rt.points).shape[0], route_cells_h=np.asarray(rt.cells).shape[0],
                    route_cells_w=np.asarray(rt.cells).shape[1])
        floats = dict(dt=cfg.dt, avoid_horizon_ticks=UC.AVOID_HORIZON_S / cfg.dt, terrain_cell_size=t0.cell_size,
                      terrain_min_x=t0.min_x, terrain_min_z=t0.min_z, terrain_max_x=t0.max_x, terrain_max_z=t0.max_z,
                      route_spacing=rt.spacing, route_min_x=rt.min_x, route_min_z=rt.min_z, route_radius=rt.radius)
        sl = lay.ai_slots
        arrays = dict(lanes=np.asarray(lay.lanes, np.int32), unit_lane=np.asarray(cfg.unit_lane, np.int32),
                      rows_m=np.asarray(sl.minions, np.int32), rows_s=np.asarray(sl.structures, np.int32),
                      cols=np.asarray(sl.cols, np.int32), lane_paths=np.asarray(LANE_PATHS, np.float32).ravel(),
                      lane_len=np.asarray(LANE_PATH_LEN, np.int32), barracks=np.asarray(BARRACKS, np.float32).ravel(),
                      route_points_xy=np.asarray(rt.points, np.float32), route_cells=np.asarray(rt.cells, np.int32),
                      route_next=np.asarray(rt.next_node),
                      lane_mid=np.asarray(cfg.lane_path, np.float32)[np.asarray(cfg.lane_path).shape[0] // 2])
        if arrays["route_next"].dtype != np.int16:
            raise ValueError("route next-hop table must be int16")
        for team in (0, 1):
            t = cfg.terrain[team]
            g = np.asarray(row_gaps(t.walkable) if t.gaps is None else t.gaps, np.int8)
            assert g.shape == (walk.shape[1], walk.shape[0] + 16, 2)
            arrays[f"gaps{team}"] = g
        if cfg.vision is not None:
            v = cfg.vision
            fl = np.asarray(v.flags, np.int32)
            arrays["vision_flags"] = fl
            ints.update(vision_height=fl.shape[0], vision_width=fl.shape[1])
            floats.update(vision_cell_size=v.cell_size, vision_min_x=v.min_x, vision_min_y=v.min_y)
        arrays.update(xla_constants(cfg))
        for k, v in ints.items():
            assert L.ls_world_int(self.ptr, k.encode(), int(v)) == 0, k
        for k, v in floats.items():
            assert L.ls_world_float(self.ptr, k.encode(), float(v)) == 0, k
        for k, v in arrays.items():
            v = np.ascontiguousarray(v)
            self._keep.append(v)
            assert L.ls_world_array(self.ptr, k.encode(), v.ctypes.data, v.size) == 0, k
        for k, v in champion_tables(cfg).items():
            v = np.ascontiguousarray(np.asarray(v, np.float64).astype(np.float32).ravel())
            self._keep.append(v)
            L.ls_world_table(self.ptr, k.encode(), v.ctypes.data, v.size)
        L.ls_world_finish(self.ptr)
        from lanerl_jax.modern import world as MS
        self.fields = env_fields()
        shapes = {name: v.shape for name, v in _leaves(MS.init_state(cfg))}
        n, k, p = cfg.n_units, len(sl.cols), lay.packet_capacity
        shapes.update(memo_route=(n, 4), memo_anchor=(n,), ev_n=(1,), ev_src=(p,), ev_dst=(p,), rec_n=(1,),
                      rec=(k * k,))
        if set(shapes) != {name for name, _ in self.fields}:
            raise ValueError("native env fields differ from the JAX state: rerun ops/native/gen_state.py")
        self.shapes = shapes
        self.counts = {name: int(np.prod(shapes[name], dtype=np.int64)) for name, _ in self.fields}
        counts = np.asarray([self.counts[name] for name, _ in self.fields], np.int64)
        L.ls_world_env_counts(self.ptr, counts.ctypes.data, len(counts))
        self.order_fields = order_fields()

    def __del__(self):
        if getattr(self, "ptr", None) and _L is not None:
            _L.ls_world_free(self.ptr)

    def empty_env(self) -> dict:
        return {name: np.zeros(self.counts[name], _CTYPES[t]) for name, t in self.fields}

    def pointers(self, env: dict):
        arr = (ctypes.c_void_p * len(self.fields))()
        for k, (name, t) in enumerate(self.fields):
            a = env[name]
            assert a.dtype == _CTYPES[t] and a.flags.c_contiguous and a.size == self.counts[name], name
            arr[k] = a.ctypes.data
        return arr

    def order_pointers(self, orders: dict):
        arr = (ctypes.c_void_p * len(self.order_fields))()
        for k, (name, t) in enumerate(self.order_fields):
            a = orders[name]
            assert a.dtype == _CTYPES[t] and a.flags.c_contiguous and a.size == 2, name
            arr[k] = a.ctypes.data
        return arr

    def step(self, env: dict, orders: dict | None = None, full: bool = False) -> np.ndarray:
        """One tick in place (``full``: with champions); returns (packet, missile, ray overflow, packets, rays)."""
        orders = no_orders() if orders is None else {k: v.copy() for k, v in orders.items()}
        stats = np.zeros(5, np.int32)
        fn = lib().ls_step_full if full else lib().ls_step
        fn(self.ptr, self.pointers(env), self.order_pointers(orders), stats.ctypes.data)
        return stats


def orders_from(o) -> dict:
    """Native order fields of a ``ModernOrders``."""
    return {name: np.ascontiguousarray(np.broadcast_to(v, (2,)).astype(_CTYPES[t]))
            for (name, v), (_, t) in zip(_leaves(o), order_fields())}


def no_orders() -> dict:
    from lanerl_jax.modern import world as MS
    return orders_from(MS.no_orders())


def env_from_state(world: NativeWorld, s, env: dict | None = None) -> dict:
    """Native env fields of a ``ModernState`` (native caches rebuilt: empty route memo, events and recent attacks
    from the damage matrix and last_attack)."""
    env = world.empty_env() if env is None else env
    names = dict(world.fields)
    for name, v in _leaves(s):
        env[name][...] = v.astype(_CTYPES[names[name]]).ravel()
    env["memo_route"][...] = np.nan
    env["memo_anchor"][...] = -3
    n = world.cfg.n_units
    src, dst = np.nonzero(env["prev_damage_matrix"].reshape(n, n))
    env["ev_n"][0] = len(src)
    env["ev_src"][:len(src)], env["ev_dst"][:len(dst)] = src, dst
    rec = np.flatnonzero(np.isfinite(env["lane_ai_last_attack"]))   # pruned to the attack memory next tick
    env["rec_n"][0] = len(rec)
    env["rec"][:len(rec)] = rec
    return env


def state_from_env(world: NativeWorld, env: dict, like):
    """The ``ModernState`` (structure and dtypes of ``like``) holding the native env's values."""
    import jax
    import jax.numpy as jnp
    leaves, tree = jax.tree_util.tree_flatten(like)
    names = [name for name, _ in _leaves(like)]
    return jax.tree_util.tree_unflatten(tree, [jnp.asarray(env[name].reshape(np.shape(v)).astype(np.asarray(v).dtype))
                                               for name, v in zip(names, leaves)])


_CODES = {"f": np.float32, "i": np.int32, "u": np.uint32, "b": np.uint8}


def test_names() -> list[str]:
    return [s for s in lib().ls_test_names().decode().split(";") if s]


def _empty_report():
    from lanerl_jax.modern.core import damage as D
    from lanerl_jax.modern.items.effects.core import Report
    z = lambda k: [np.zeros(0)] * k                                                    # noqa: E731
    return Report(D.Packets(*z(10)), D.Resolved(*z(7), D.Shields(*z(6)), *z(4)), np.zeros(0), np.zeros(0))


def _prepare(tree):
    """Native argument conventions: RuneEvents always carries a Report (empty outside on_damage)."""
    import jax
    from lanerl_jax.modern.runes.effects.core import RuneEvents

    def fix(v):
        return v._replace(report=_empty_report()) if isinstance(v, RuneEvents) and v.report is None else v
    return jax.tree_util.tree_map(fix, tree, is_leaf=lambda v: isinstance(v, RuneEvents))


def _flat(tree):
    """Leaves of a pytree in JAX flatten order, ``None`` fields as empty arrays (the native optional)."""
    import jax
    return jax.tree_util.tree_leaves(_prepare(tree), is_leaf=lambda v: v is None)


def call(name: str, *args) -> list[np.ndarray]:
    """Call the registered native function ``name`` on JAX arguments (pytrees); returns its flat output leaves.
    Fields the native side does not take (``RuneEvents.report``) must be None in ``args``... and are dropped."""
    sig = lib().ls_test_signature(name.encode())
    if sig is None:
        raise KeyError(name)
    codes, counts = sig.decode().split("|")
    per_arg = [int(k) for k in counts.split(",") if k]
    leaves = []
    for a, k in zip(args, per_arg):
        flat = [np.zeros(0) if v is None else np.asarray(v) for v in _flat(a)]
        if len(flat) != k:
            raise ValueError(f"{name}: argument has {len(flat)} leaves, native expects {k}")
        leaves += flat
    arrs = [np.ascontiguousarray(np.asarray(v).astype(_CODES[c]).ravel()) for v, c in zip(leaves, codes)]
    ptrs = (ctypes.c_void_p * len(arrs))(*[a.ctypes.data for a in arrs])
    cnts = np.asarray([a.size for a in arrs], np.int64)
    h = lib().ls_test_call(name.encode(), ptrs, cnts.ctypes.data)
    out = []
    for i in range(lib().ls_result_leaves(h)):
        code, n = lib().ls_result_code(h, i).decode(), lib().ls_result_count(h, i)
        dt = _CODES[code]
        buf = (ctypes.c_char * (n * np.dtype(dt).itemsize)).from_address(lib().ls_result_data(h, i)) if n else b""
        out.append(np.frombuffer(bytes(buf), dt).copy())
    lib().ls_result_free(h)
    return out


PHASES = ("spawn", "turret", "select", "move_prep", "route", "collide", "attack", "damage", "death", "timers", "fog",
          "select.reset", "select.victims", "select.minions_a", "select.minions_b", "select.turrets", "champ")


def profile(reset: bool = True) -> dict:
    """Per-phase seconds of this thread's native ticks since the last reset."""
    out = np.zeros(len(PHASES))
    lib().ls_profile(out.ctypes.data, int(reset))
    return dict(zip(PHASES, (out * 1e-9).tolist()))


class Batch:
    """``n_envs`` copies of an env, stepped by the library's threads."""

    def __init__(self, world: NativeWorld, env: dict, n_envs: int, order_mode: int = 0):
        """``order_mode``: bit 0 the JAX bench's scripted walk-and-attack orders (else none), bit 1 the full tick
        with champions (else the lane slice)."""
        self.world, self.n = world, n_envs
        self.ptr = lib().ls_batch_new(world.ptr, n_envs, world.pointers(env), order_mode)

    def __del__(self):
        if getattr(self, "ptr", None):
            lib().ls_batch_free(self.ptr)

    def run(self, ticks: int, threads: int = 0) -> np.ndarray:
        stats = np.zeros(3, np.int32)
        lib().ls_batch_run(self.ptr, ticks, threads, stats.ctypes.data)
        return stats

    def get(self, k: int) -> dict:
        env = self.world.empty_env()
        lib().ls_batch_get(self.ptr, k, self.world.pointers(env))
        return env
