"""Generic per-tick mechanics of the modern world (DAMAGE_AND_STATS §8–10).

Pure, fixed-shape helpers used by ``modern_step``:

* ``attack_step``   the shared basic-attack machine (windup -> launch -> period,
                    cancel resets the timer, edge-to-edge range, §8.2–8.3);
* ``Missiles``      fixed-capacity projectile buffer for ranged attacks;
* ``CCTimers``      per-unit crowd-control end times and the capability flags
                    of §10.3, tenacity applied at application;
* ``move_step``     route-steered movement on the Map11 route graph with the
                    terrain clamp (``modern_pathing``), plus dashes/blinks.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from . import modern_world_types as W
from .modern_pathing import route_next, segment_clear
from .modern_stat_pipeline import cc_duration

BIG = 1e9


# ---- basic attacks -------------------------------------------------------------

def in_attack_range(units: W.WorldUnits, target: Any) -> Any:
    """(N,) edge-to-edge range check of each unit against its ``target``."""
    t = jnp.clip(target, 0, units.x.shape[0] - 1)
    d = jnp.sqrt((units.x - units.x[t]) ** 2 + (units.y - units.y[t]) ** 2)
    return (target >= 0) & (d <= units.attack_range + units.radius + units.radius[t])


def att_target_ok(att: W.AttackState, units: W.WorldUnits) -> Any:
    """(N,) the current windup's target is still alive, targetable, hostile and the same unit."""
    n = units.x.shape[0]
    t = jnp.clip(att.target, 0, n - 1)
    return (att.target >= 0) & units.alive[t] & units.targetable[t] & (units.team[t] != units.team) \
        & (units.spawn_seq[t] == att.target_seq)


def attack_step(att: W.AttackState, units: W.WorldUnits, desired: Any, *, can_attack: Any, windup: Any,
                dt: Any, reset: Any = None, period: Any = None, uncancellable: Any = None) -> tuple[W.AttackState, Any]:
    """Advance the attack machine one tick. Returns ``(state, launched)``.

    ``desired`` (N,) is the target each unit wants. Switching target, losing
    the target, leaving range or losing ``can_attack`` during the windup
    cancels it and resets the timer to 0 (U-08 default). ``reset`` (N,) is an
    attack reset (Garen Q, Jax W, Titanic): cooldown 0 and any windup cancelled.
    ``period`` (N,) overrides the attack's total time when > 0 (Garen Q);
    ``uncancellable`` (N,) keeps a started windup running through range loss
    and a new order; it still ends if the target dies or becomes invalid.
    """
    n = units.x.shape[0]
    t = jnp.clip(desired, 0, n - 1)
    valid = (desired >= 0) & units.alive[t] & units.targetable[t] & (units.team[t] != units.team)
    same = (desired == att.target) & (units.spawn_seq[t] == att.target_seq)
    ready = valid & in_attack_range(units, desired) & can_attack & units.alive
    winding = att.windup_left > 0.0
    cancel = winding & ~(same & ready)
    if uncancellable is not None:
        cancel = cancel & ~(uncancellable & att_target_ok(att, units))
    if reset is not None:
        cancel = cancel | (reset & winding)
    cooldown = jnp.maximum(att.cooldown_left - dt, 0.0)
    cooldown = jnp.where(cancel, 0.0, cooldown)
    if reset is not None:
        cooldown = jnp.where(reset, 0.0, cooldown)
    left = jnp.where(cancel, 0.0, att.windup_left)
    start = ready & ~(winding & ~cancel) & (cooldown <= 0.0)
    stat_period = 1.0 / jnp.maximum(units.attack_speed, 1e-3)
    period = stat_period if period is None else jnp.where(period > 0, period, stat_period)
    left = jnp.where(start, windup, left)
    cooldown = jnp.where(start, period, cooldown)
    # Fires on the first tick whose accumulated time reaches the windup (tick rounding, §1.3).
    held = ready if uncancellable is None else ready | (uncancellable & att_target_ok(att, units) & (left > 0.0))
    launched = (left > 0.0) & (left - dt <= 1e-5) & held
    left = jnp.where(launched | (left <= 0.0), 0.0, jnp.maximum(left - dt, 0.0))
    target = jnp.where(valid, desired, -1)
    seq = jnp.where(valid, units.spawn_seq[t], att.target_seq)
    if uncancellable is not None:
        keep = uncancellable & winding & ~cancel
        target = jnp.where(keep, att.target, target)
        seq = jnp.where(keep, att.target_seq, seq)
    return W.AttackState(target.astype(jnp.int32), seq.astype(jnp.int32), left, cooldown), launched


# ---- missiles ----------------------------------------------------------------------

class Missiles(NamedTuple):
    """Ranged basic-attack projectiles, shape (M,)."""
    alive: Any
    src: Any
    dst: Any
    dst_seq: Any
    x: Any
    y: Any
    speed: Any
    raw: Any
    dtype: Any
    flags: Any
    cast_id: Any
    crit: Any


def init_missiles(m: int = 64) -> Missiles:
    z = jnp.zeros((m,), jnp.float32)
    zi = jnp.zeros((m,), jnp.int32)
    return Missiles(jnp.zeros((m,), bool), zi, zi, zi, z, z, z, z, zi, zi, zi, jnp.zeros((m,), bool))


def spawn_missiles(ms: Missiles, launch: Any, units: W.WorldUnits, target: Any, raw: Any, dtype: Any,
                   flags: Any, speed: Any, cast_id: Any, crit: Any) -> tuple[Missiles, Any]:
    """Add one missile per launching unit (N,) into free slots; returns overflow count."""
    m = ms.alive.shape[0]
    free_rank = jnp.cumsum(~ms.alive) - 1                                   # rank of each free slot
    want_rank = jnp.cumsum(launch) - 1                                      # rank of each launcher
    take = (~ms.alive)[None, :] & (free_rank[None, :] == want_rank[:, None]) & launch[:, None]   # (N, M)
    got = jnp.any(take, axis=0)
    pick = lambda v: jnp.sum(jnp.where(take, v[:, None], 0), axis=0)
    t = jnp.clip(target, 0, units.x.shape[0] - 1)
    new = Missiles(
        ms.alive | got, jnp.where(got, pick(jnp.arange(launch.shape[0])), ms.src).astype(jnp.int32),
        jnp.where(got, pick(target), ms.dst).astype(jnp.int32),
        jnp.where(got, pick(units.spawn_seq[t]), ms.dst_seq).astype(jnp.int32),
        jnp.where(got, pick(units.x), ms.x), jnp.where(got, pick(units.y), ms.y),
        jnp.where(got, pick(speed), ms.speed), jnp.where(got, pick(raw), ms.raw),
        jnp.where(got, pick(dtype), ms.dtype).astype(jnp.int32), jnp.where(got, pick(flags), ms.flags).astype(jnp.int32),
        jnp.where(got, pick(cast_id), ms.cast_id).astype(jnp.int32),
        jnp.where(got, jnp.any(take & crit[:, None], axis=0), ms.crit))
    return new, jnp.sum(launch) - jnp.sum(got)


def advance_missiles(ms: Missiles, units: W.WorldUnits, dt: Any) -> tuple[Missiles, Any]:
    """Home on targets; arrivals (contact with the target hitbox) return a hit mask (M,).

    A missile whose target died or whose slot was reused fizzles.
    """
    t = jnp.clip(ms.dst, 0, units.x.shape[0] - 1)
    gone = ~units.alive[t] | (units.spawn_seq[t] != ms.dst_seq)
    dx, dy = units.x[t] - ms.x, units.y[t] - ms.y
    d = jnp.sqrt(dx * dx + dy * dy)
    step = ms.speed * dt
    arrive = ms.alive & ~gone & (d - units.radius[t] <= step)
    f = jnp.where(d > 0, jnp.minimum(step / jnp.maximum(d, 1e-6), 1.0), 1.0)
    moved = ms._replace(x=ms.x + dx * f, y=ms.y + dy * f, alive=ms.alive & ~gone & ~arrive)
    return moved, arrive


# ---- crowd control -----------------------------------------------------------------

class CCTimers(NamedTuple):
    """Per-unit CC end times (absolute seconds) and the strongest slow, (N,)."""
    stun_until: Any
    root_until: Any
    silence_until: Any
    knockup_until: Any
    slow: Any
    slow_until: Any
    champion_cc_until: Any      # end of any CC applied by an enemy champion (Unflinching)


def init_cc(n: int) -> CCTimers:
    z = jnp.zeros((n,), jnp.float32)
    return CCTimers(z, z, z, z, z, z, z)


def apply_cc(cc: CCTimers, out: W.CCOut, tenacity: Any, slow_resist: Any, now: Any, *,
             source_is_champion: Any, cleansed: Any = None) -> CCTimers:
    """Add this tick's CC (C, N) with tenacity at application (§10.2).

    Knock-ups ignore tenacity; slow strength is reduced by slow resist at use
    (``move_speed``) and the strongest active slow wins (§9.1).
    """
    reduce = lambda d: cc_duration(d, tenacity[None, :])
    stun = jnp.max(jnp.where(out.stun > 0, reduce(out.stun), 0.0), axis=0)
    root = jnp.max(jnp.where(out.root > 0, reduce(out.root), 0.0), axis=0)
    sil = jnp.max(jnp.where(out.silence > 0, reduce(out.silence), 0.0), axis=0)
    up = jnp.max(out.knockup, axis=0)
    k = jnp.argmax(out.slow, axis=0)
    strength = jnp.max(out.slow, axis=0)
    sdur = jnp.where(strength > 0, cc_duration(jnp.take_along_axis(out.slow_duration, k[None, :], axis=0)[0],
                                               tenacity), 0.0)
    active_slow = jnp.where(cc.slow_until > now, cc.slow, 0.0)
    take = (strength > 0) & (strength >= active_slow)
    champ = jnp.any(source_is_champion[:, None] & ((out.stun > 0) | (out.root > 0) | (out.silence > 0)
                                                   | (out.knockup > 0) | (out.slow > 0)), axis=0)
    longest = jnp.maximum(jnp.maximum(stun, root), jnp.maximum(jnp.maximum(sil, up), sdur))
    new = CCTimers(jnp.maximum(cc.stun_until, jnp.where(stun > 0, now + stun, 0.0)),
                   jnp.maximum(cc.root_until, jnp.where(root > 0, now + root, 0.0)),
                   jnp.maximum(cc.silence_until, jnp.where(sil > 0, now + sil, 0.0)),
                   jnp.maximum(cc.knockup_until, jnp.where(up > 0, now + up, 0.0)),
                   jnp.where(take, strength, cc.slow),
                   jnp.where(take, jnp.maximum(now + sdur, jnp.where(strength == active_slow, cc.slow_until, 0.0)),
                             cc.slow_until),
                   jnp.maximum(cc.champion_cc_until, jnp.where(champ, now + longest, 0.0)))
    if cleansed is not None:
        new = cleanse(new, cleansed, now)
    return new


def cleanse(cc: CCTimers, mask: Any, now: Any) -> CCTimers:
    """Remove stun/root/silence/slow (not knock-ups) from ``mask`` units (Cleanse)."""
    f = lambda v: jnp.where(mask, jnp.minimum(v, now), v)
    return cc._replace(stun_until=f(cc.stun_until), root_until=f(cc.root_until), silence_until=f(cc.silence_until),
                       slow_until=f(cc.slow_until))


def capabilities(cc: CCTimers, now: Any) -> dict:
    """§10.3 capability flags (N,) from active CC."""
    stun, root = cc.stun_until > now, cc.root_until > now
    sil, up = cc.silence_until > now, cc.knockup_until > now
    return {"can_move": ~(stun | root | up), "can_attack": ~(stun | up), "can_cast": ~(stun | up | sil),
            "can_summoner": ~(stun | up), "stunned": stun | up, "silenced": sil,
            "slow": jnp.where(cc.slow_until > now, cc.slow, 0.0), "impaired": stun | root | sil | up
            | (cc.slow_until > now), "movement_impaired": stun | root | up | (cc.slow_until > now)}


# ---- movement ----------------------------------------------------------------------

def team_terrain(terrain: tuple, team: Any):
    """Team 0's mask for team 0, team 1's otherwise (a layer view: no per-unit grid copy)."""
    from .modern_terrain import team_view
    return team_view(terrain, jnp.where(team == 0, 0, 1))


def move_step(x: Any, y: Any, goal_x: Any, goal_y: Any, speed: Any, active: Any, team: Any, radius: Any,
              routes, terrain: tuple, dt: Any) -> tuple[Any, Any, Any]:
    """Steer each active unit toward its goal along the route graph, one tick.

    Returns ``(x, y, route_ok)``. Steps are clamped by a swept terrain check;
    a blocked step leaves the unit in place (fail closed).
    """
    def one(px, py, gx, gy, sp, act, tm, r):
        ter = team_terrain(terrain, tm)
        pos, goal = jnp.stack([px, py]), jnp.stack([gx, gy])
        nxt, ok = route_next(pos, goal, jnp.minimum(r, routes.radius), routes, ter)
        d = nxt - pos
        dist = jnp.linalg.norm(d)
        step = jnp.minimum(sp * dt, dist)
        new = pos + jnp.where(dist > 1e-6, d / jnp.maximum(dist, 1e-6) * step, 0.0)
        clear = segment_clear(pos, new, jnp.minimum(r, routes.radius), ter, samples=5, max_length=200.,
                              max_radius=routes.radius)
        new = jnp.where(act & clear, new, pos)
        return new[0], new[1], ok | ~act
    return jax.vmap(one)(x, y, goal_x, goal_y, speed, active, team, radius)


def blink_point(x0: Any, y0: Any, x1: Any, y1: Any, max_range: Any, team: Any, radius: Any, terrain: tuple,
                samples: int = 16) -> tuple[Any, Any]:
    """Flash landing: clamp to ``max_range``, then the farthest walkable point
    on the segment (scanning back from the target; Flash over walls lands on
    the far side when that side is walkable, SUMMONER_SPELLS §2)."""
    dx, dy = x1 - x0, y1 - y0
    d = jnp.sqrt(dx * dx + dy * dy)
    s = jnp.minimum(1.0, max_range / jnp.maximum(d, 1e-6))
    tx, ty = x0 + dx * s, y0 + dy * s
    from .modern_terrain import is_walkable

    def one(ax, ay, bx, by, tm, r):
        ter = team_terrain(terrain, tm)
        f = jnp.linspace(1.0, 0.0, samples)
        px, py = ax + (bx - ax) * f, ay + (by - ay) * f
        ok = jax.vmap(lambda u, v: is_walkable(u, v, jnp.minimum(r, 50.0), ter))(px, py)
        i = jnp.argmax(ok)
        return jnp.where(jnp.any(ok), px[i], ax), jnp.where(jnp.any(ok), py[i], ay)
    return jax.vmap(one)(x0, y0, tx, ty, team, radius)
