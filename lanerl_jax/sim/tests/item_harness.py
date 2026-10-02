"""Small fixed-shape world for item-effect unit tests.

Unit 0 and 1 are the two champions (holders 0 and 1, teams 0 and 1); further
units are placed by the caller. Everything is plain JAX so tests can also
``jax.jit`` the hooks.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from lanerl_jax.sim import modern_damage as D
from lanerl_jax.sim import modern_inventory as inv_mod
from lanerl_jax.sim.modern_item_effects.core import Attack, Cast, Ctx, Kills, Report, Units


def units(rows):
    """rows: list of dicts with x, y, team, cls and optional hp/max_hp/radius/bonus_hp/siege."""
    f = lambda k, d: jnp.asarray([r.get(k, d) for r in rows], jnp.float32)
    return Units(
        x=f("x", 0.0), y=f("y", 0.0), team=jnp.asarray([r["team"] for r in rows], jnp.int32),
        cls=jnp.asarray([r.get("cls", D.CLASS_MINION) for r in rows], jnp.int32),
        alive=jnp.asarray([r.get("alive", True) for r in rows]), hp=f("hp", 1000.0),
        max_hp=f("max_hp", 1000.0), radius=f("radius", 48.0),
        targetable=jnp.asarray([r.get("targetable", True) for r in rows]),
        is_siege_or_super=jnp.asarray([r.get("siege", False) for r in rows]),
        bonus_hp=f("bonus_hp", 0.0), armor=f("armor", 0.0), magic_resist=f("mr", 0.0))


def champions(**kw):
    """Two champion unit rows: holder 0 at (0,0) team 0, holder 1 at (x1,0) team 1."""
    x1 = kw.pop("x1", 300.0)
    base = dict(cls=D.CLASS_CHAMPION, radius=65.0)
    return [dict(base, x=0.0, y=0.0, team=0, **kw), dict(base, x=x1, y=0.0, team=1, **kw)]


def ctx(n=2, *, now=0.0, dt=1 / 30, level=1, ranged=False, base_ad=60.0, bonus_ad=0.0, ap=0.0,
        base_hp=600.0, max_hp=None, hp=None, base_armor=30.0, bonus_armor=0.0, base_mr=30.0,
        bonus_mr=0.0, mana=300.0, max_mana=300.0, base_ms=340.0, move_speed=None, crit_chance=0.0,
        crit_damage=2.0, life_steal=0.0, bonus_as=0.0, ability_haste=0.0, lethality=0.0, hsp=0.0, windup=0.2, in_combat=False, in_shop=False, x=None, y=None,
        facing=(1.0, 0.0), moved=0.0, unit=None, team=None, alive=True):
    v = lambda val: jnp.broadcast_to(jnp.asarray(val, jnp.float32), (n,))
    max_hp = base_hp if max_hp is None else max_hp
    hp = max_hp if hp is None else hp
    return Ctx(
        now=jnp.float32(now), dt=jnp.float32(dt),
        unit=jnp.arange(n, dtype=jnp.int32) if unit is None else jnp.asarray(unit, jnp.int32),
        team=jnp.arange(n, dtype=jnp.int32) if team is None else jnp.asarray(team, jnp.int32),
        alive=jnp.broadcast_to(jnp.asarray(alive), (n,)), level=v(level),
        is_ranged=jnp.broadcast_to(jnp.asarray(ranged), (n,)),
        x=v(0.0) if x is None else v(x), y=v(0.0) if y is None else v(y),
        facing_x=v(facing[0]), facing_y=v(facing[1]), moved=v(moved),
        base_ad=v(base_ad), bonus_ad=v(bonus_ad), ap=v(ap), base_hp=v(base_hp), max_hp=v(max_hp),
        hp=v(hp), base_armor=v(base_armor), bonus_armor=v(bonus_armor), base_mr=v(base_mr),
        bonus_mr=v(bonus_mr), mana=v(mana), max_mana=v(max_mana), base_ms=v(base_ms),
        move_speed=v(base_ms if move_speed is None else move_speed), crit_chance=v(crit_chance),
        crit_damage=v(crit_damage), life_steal=v(life_steal), bonus_attack_speed=v(bonus_as),
        ability_haste=v(ability_haste), lethality=v(lethality), heal_shield_power=v(hsp), attack_windup=v(windup),
        in_combat=jnp.broadcast_to(jnp.asarray(in_combat), (n,)),
        in_shop=jnp.broadcast_to(jnp.asarray(in_shop), (n,)))


def own(*loadouts):
    """(C, I) owned counts from per-holder item id lists."""
    return inv_mod.owned_counts(inv_mod.inventory_from_ids([list(l) for l in loadouts]))


def attack(hit=(True, False), target=(1, 0), raw=(100.0, 0.0), crit=(False, False), launched=None):
    hit = jnp.asarray(hit)
    return Attack(hit if launched is None else jnp.asarray(launched), hit,
                  jnp.asarray(target, jnp.int32), jnp.asarray(raw, jnp.float32), jnp.asarray(crit))


def cast(started=(True, False), slot=(0, 0), target=(-1, -1)):
    return Cast(jnp.asarray(started), jnp.asarray(slot, jnp.int32), jnp.asarray(target, jnp.int32))


def kills(n_units, champion_kill=(0, 0), champion_assist=(0, 0), minion_kill=(0, 0), died=(False, False),
          killed_units=None):
    ku = jnp.zeros((2, n_units), bool) if killed_units is None else jnp.asarray(killed_units)
    return Kills(jnp.asarray(champion_kill, jnp.float32), jnp.asarray(champion_assist, jnp.float32),
                 jnp.asarray(minion_kill, jnp.float32), jnp.asarray(died), ku)


def resolve(p, u: Units, *, armor=0.0, mr=0.0, shields=None, now=0.0, defense=None, offense=None,
            life_steal=None, omnivamp=None):
    """Resolve packets against ``u`` and return (Report, Resolved)."""
    n = u.x.shape[0]
    dfn = defense or D.default_defense(n, armor=armor, magic_resist=mr)._replace(unit_class=u.cls)
    off = offense or D.default_offense(n)._replace(unit_class=u.cls)
    sh = shields if shields is not None else D.init_shields(n)
    res = D.resolve(p, off, dfn, u.hp, u.max_hp, sh, jnp.float32(now))
    ls = jnp.zeros((n,)) if life_steal is None else jnp.asarray(life_steal, jnp.float32)
    ov = jnp.zeros((n,)) if omnivamp is None else jnp.asarray(omnivamp, jnp.float32)
    heal = D.vamp_heal(p, res, D.Vamp(ls, ov), u.cls)
    return Report(p, res, heal), res


def packet_total(p, *, dst=None, item=None, src=None):
    """Sum of valid raw packet damage, optionally filtered."""
    sel = np.asarray(p.valid)
    if dst is not None:
        sel = sel & (np.asarray(p.dst) == dst)
    if item is not None:
        sel = sel & (np.asarray(p.item) == item)
    if src is not None:
        sel = sel & (np.asarray(p.src) == src)
    return float(np.sum(np.where(sel, np.asarray(p.raw), 0.0)))


def packet_targets(p, *, item=None):
    sel = np.asarray(p.valid)
    if item is not None:
        sel = sel & (np.asarray(p.item) == item)
    return sorted(set(np.asarray(p.dst)[sel].tolist()))
