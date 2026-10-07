"""ctypes binding of liblanesim (the native lane-slice tick) and its mapping to the JAX world.

``NativeWorld(cfg)`` builds the C++ ``World`` from a JAX ``WorldConfig`` (layout, terrain gap tables, route graph,
vision grid, lane paths, and the few float32 constants XLA evaluates: avoidance rotations, separation fallback
directions, eject rings). ``env_from_state`` / ``state_fields`` map a ``ModernState`` to the native env fields;
``step`` advances an env held in numpy arrays; ``Batch`` runs many envs the library owns.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

LIB = Path(os.environ.get("LANESIM_LIB", Path(__file__).resolve().parents[1] / "build" / "liblanesim.so"))
_CTYPES = {"float": np.float32, "int32_t": np.int32, "uint8_t": np.uint8}


def _lib():
    lib = ctypes.CDLL(str(LIB))
    P, I, L, D, S = ctypes.c_void_p, ctypes.c_int, ctypes.c_long, ctypes.c_double, ctypes.c_char_p
    sig = {"ls_env_fields": ([], S), "ls_field_count": ([P, S], L), "ls_world_new": ([], P),
           "ls_world_free": ([P], None), "ls_world_int": ([P, S, L], I), "ls_world_float": ([P, S, D], I),
           "ls_world_array": ([P, S, P, L], I), "ls_world_finish": ([P], None),
           "ls_step": ([P, P, P], None), "ls_batch_new": ([P, I, P], P), "ls_batch_free": ([P], None),
           "ls_batch_run": ([P, I, I, P], None), "ls_batch_get": ([P, I, P], None),
           "ls_batch_env_bytes": ([P], L), "ls_profile": ([P, I], None)}
    for name, (args, res) in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, res
    return lib


_L = None


def lib():
    global _L
    if _L is None:
        _L = _lib()
    return _L


def env_fields() -> list[tuple[str, str, str]]:
    """``(name, ctype, size class)`` per native env field, in struct order."""
    out = []
    for item in lib().ls_env_fields().decode().split(";"):
        if item:
            t, name, size = item.split()
            out.append((name, t, size))
    return out


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
                      route_next=np.asarray(rt.next_node))
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
        L.ls_world_finish(self.ptr)
        self.fields = env_fields()
        self.counts = {name: L.ls_field_count(self.ptr, size.encode()) for name, _, size in self.fields}

    def __del__(self):
        if getattr(self, "ptr", None):
            lib().ls_world_free(self.ptr)

    def empty_env(self) -> dict:
        return {name: np.zeros(self.counts[name], _CTYPES[t]) for name, t, _ in self.fields}

    def pointers(self, env: dict):
        arr = (ctypes.c_void_p * len(self.fields))()
        for k, (name, t, _) in enumerate(self.fields):
            a = env[name]
            assert a.dtype == _CTYPES[t] and a.flags.c_contiguous and a.size == self.counts[name], name
            arr[k] = a.ctypes.data
        return arr

    def step(self, env: dict) -> np.ndarray:
        """One tick in place; returns (packet, missile, ray overflow, packets, rays)."""
        stats = np.zeros(5, np.int32)
        lib().ls_step(self.ptr, self.pointers(env), stats.ctypes.data)
        return stats


# ModernState -> native env field: (path in the state, dtype).
STATE_FIELDS = {
    "t": "t", "tick": "tick", "next_seq": "next_seq", "game_over": "game_over", "winner": "winner",
    **{k: k for k in ("kind", "sub", "team", "spawn_seq", "bounty_level", "alive", "targetable", "x", "y", "hp",
                      "max_hp", "radius", "armor", "windup", "spawn_time", "missile_speed", "bounty_gold",
                      "bounty_xp")},
    "mr": "magic_resist", "ad": "attack_damage", "range": "attack_range", "aspd": "attack_speed", "ms": "move_speed",
    "wave": "spawn.wave", "unit": "spawn.unit", "supers": "spawn.supers",
    "att_target": "att.target", "att_seq": "att.target_seq", "windup_left": "att.windup_left",
    "cooldown_left": "att.cooldown_left",
    **{f"m_{k}": f"missiles.{v}" for k, v in dict(alive="alive", src="src", dst="dst", dst_seq="dst_seq", x="x", y="y",
                                                  speed="speed", raw="raw", dtype="dtype", flags="flags",
                                                  cast="cast_id", crit="crit").items()},
    **{k: f"cc.{k}" for k in ("stun_until", "root_until", "silence_until", "knockup_until", "slow", "slow_until",
                              "champion_cc_until")},
    "level": "econ.level",
    **{f"ai_{k}": f"lane_ai.{v}" for k, v in dict(seq="seq", target="target", target_seq="target_seq",
                                                  priority="target_priority", sweep="sweep_timer",
                                                  since="since_attack", lane="lane", waypoint="waypoint",
                                                  first_wave="first_wave", engaged="engaged",
                                                  champion_aggro="champion_aggro", warm_stacks="warm_stacks",
                                                  warm_until="warm_until").items()},
    "ignore_until": "lane_ai.ignore_until", "last_attack": "lane_ai.last_attack",
    **{f"tw_{k}": f"towers.turret.{v}" for k, v in dict(hp="hp", max_hp="max_hp", tier="tier", respawn_at="respawn_at",
                                                        plates="plates", bulwark="bulwark_until",
                                                        backdoor="backdoor_until", growth_since="growth_since",
                                                        growth_active="growth_active", warm_stacks="warm_stacks",
                                                        warm_until="warm_until").items()},
    **{f"tw_{k}": f"towers.{v}" for k, v in dict(is_structure="is_structure", team="team", lane="lane",
                                                 prereq="prereq", targetable="targetable",
                                                 first_turret="first_turret_taken").items()},
    "damage_matrix": "prev.damage_matrix", "visible": "visible",
    "reveal_x": "reveal.x", "reveal_y": "reveal.y", "reveal_until": "reveal.until", "route_anchor": "route_anchor",
}


def _get(s, path):
    for part in path.split("."):
        s = getattr(s, part)
    return np.asarray(s)


def env_from_state(world: NativeWorld, s, env: dict | None = None) -> dict:
    """Native env fields of a ``ModernState`` (the route memo starts empty)."""
    env = world.empty_env() if env is None else env
    for name, t, _ in world.fields:
        if name in STATE_FIELDS:
            env[name][...] = _get(s, STATE_FIELDS[name]).astype(_CTYPES[t]).ravel()
    env["memo_route"][...] = np.nan
    env["memo_anchor"][...] = -3
    n = world.cfg.n_units
    src, dst = np.nonzero(env["damage_matrix"].reshape(n, n))
    env["ev_n"][0] = len(src)
    env["ev_src"][:len(src)], env["ev_dst"][:len(dst)] = src, dst
    return env


PHASES = ("spawn", "turret", "select", "move_prep", "route", "collide", "attack", "damage", "death", "timers", "fog",
          "select.reset", "select.victims", "select.minions_a", "select.minions_b", "select.turrets")


def profile(reset: bool = True) -> dict:
    """Per-phase seconds of this thread's native ticks since the last reset."""
    out = np.zeros(len(PHASES))
    lib().ls_profile(out.ctypes.data, int(reset))
    return dict(zip(PHASES, (out * 1e-9).tolist()))


class Batch:
    """``n_envs`` copies of an env, stepped by the library's threads."""

    def __init__(self, world: NativeWorld, env: dict, n_envs: int):
        self.world, self.n = world, n_envs
        self.ptr = lib().ls_batch_new(world.ptr, n_envs, world.pointers(env))

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
