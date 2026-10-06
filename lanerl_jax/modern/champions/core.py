"""Contract shared by the 26.19 champion kits (docs/modern/CHAMPIONS.md).

Kits are pure fixed-shape functions over the holder context ``KitCtx`` (C,) and ``core.types.WorldUnits`` (N,);
they never write world arrays. Damage is raw pre-mitigation ``core.damage`` packets, CC is ``CCOut`` with
pre-tenacity durations, stats are ``ItemStats``. ``now`` is fixed across one tick's hooks; ``cast`` runs before
``periodic``. A timer ending at ``until`` expires on the tick whose end ``now + dt`` reaches it.

Cast ids: ``KIT_ID_BASE + (tick * C + holder) * ID_STRIDE + code``, ``code`` the slot 0..3 or a ``CODE_*`` for a
non-cast instance (one Garen E spin, one Jax R passive proc). ``KIT_ID_BASE`` keeps them clear of attack ids.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..core import damage as D
from ..core.types import KIND_NONE, KIND_WARD, STRUCTURE_KINDS, CCOut, Dash, merge_cc, no_cc, no_dash
from ..data.champions import cooldowns, spell, values
from ..items.effects.core import ShieldGrant

Q, W, E, R = range(4)

KIT_ID_BASE = 1 << 30
ID_STRIDE = 8
CODE_GAREN_E_TICK = 4
CODE_JAX_R_PASSIVE = 5
EPS = 1e-6
NEVER = 1e9


class KitCtx(NamedTuple):
    """Per champion holder (C,); holder ``c`` is world unit ``unit[c]``."""
    unit: Any
    champion_id: Any
    team: Any
    alive: Any
    level: Any
    ranks: Any                       # (C, 4) int32 Q/W/E/R
    x: Any
    y: Any
    mana: Any
    max_mana: Any
    hp: Any
    max_hp: Any
    base_ad: Any
    bonus_ad: Any
    ap: Any
    bonus_hp: Any
    armor: Any
    magic_resist: Any
    bonus_attack_speed: Any          # bonus AS ratio (growth + items)
    crit_chance: Any
    crit_damage: Any
    ability_haste: Any               # basic ability haste
    ultimate_haste: Any
    cooldowns: Any                   # (C, 4) remaining seconds
    now: Any                         # () seconds
    dt: Any                          # () seconds
    silenced: Any
    stunned: Any
    in_combat_ms_since_damaged: Any  # seconds since the holder last took damage (unused by 26.19 kits)
    # Optional world inputs (defaults keep the kits self-contained):
    attack_target_kind: Any = None   # (C,) int32 world kind of the unit the holder attacks (KIND_NONE if none)
    rooted: Any = None               # (C,) bool (Jax Q cantCastWhileRooted)

    @property
    def total_ad(self):
        return self.base_ad + self.bonus_ad


class KitOut(NamedTuple):
    packets: D.Packets               # every damage instance; cast_id on each; item=0
    cc: CCOut                        # (C, N), pre-tenacity durations
    dash: Dash                       # (C,)
    heal: Any                        # (C,) self heal (Garen passive regen)
    shield: ShieldGrant              # (C, S)
    mana_cost: Any                   # (C,)
    cooldown_start: Any              # (C, 4) bool: start that slot's cooldown now
    base_cooldown: Any               # (C, 4) base cooldown before haste
    attack_reset: Any                # (C,) bool
    cast_started: Any                # (C,) bool
    cast_slot: Any                   # (C,) int32, -1 none
    cast_id: Any                     # (C,) int32, 0 none
    cast_lockout: Any                # (C,) seconds the champion can't move/attack
    cleanse_slow: Any = None         # (C,) bool: remove the holder's slows (Garen Q); None = never
    attack_target: Any = None        # (C,) int32: issue a basic-attack order on this unit (-1 none; Jax Q
                                     # on an enemy champion); None = never


class KitDefense(NamedTuple):
    received_mult: Any               # (C,) DMG.60 product (Garen W DR)
    dodge_basic: Any                 # (C,) bool (Jax E)
    aoe_received_mult: Any           # (C,) (Jax E 0.75)
    tenacity_bonus: Any              # (C,) multiplicative tenacity source (Garen W 0.6)


class KitAttackMods(NamedTuple):
    extra_range: Any                 # (C,) attack range bonus
    attack_reset: Any                # (C,) bool: a reset was requested this tick
    cannot_attack: Any               # (C,) bool
    cannot_crit: Any = None          # (C,) bool: the next attack cannot crit (no 26.19 kit sets it)
    windup: Any = None               # (C,) seconds: override the next attack's windup (> 0), else the stat windup
    period: Any = None               # (C,) seconds: override the next attack's total attack time (> 0)
    uncancellable: Any = None        # (C,) bool: the next attack's windup cannot be cancelled


def f32(x: Any) -> Any:
    return jnp.asarray(x, jnp.float32)


def ranked(name: str, slot: str, key: str, rank: Any) -> Any:
    """JSON value at ``rank`` (index 0 is the rank-0 entry, 1..5 ranks)."""
    return jnp.asarray(values(name, slot, key), jnp.float32)[jnp.clip(rank, 0, 6)]


def scalar(name: str, slot: str, key: str) -> float:
    return float(values(name, slot, key)[1])


def cooldown_row(name: str, ranks: Any) -> Any:
    """(C, 4) base cooldowns at the holder's ranks (``modern.cooldown_table``)."""
    cols = []
    for i, slot in enumerate("QWER"):
        table = jnp.asarray(cooldowns(name, slot), jnp.float32)
        cols.append(table[jnp.clip(ranks[:, i] - 1, 0, table.shape[0] - 1)])
    return jnp.stack(cols, -1)


def mana_row(name: str, ranks: Any) -> Any:
    """(C, 4) mana costs at the holder's ranks (0 for manaless kits)."""
    cols = []
    for i, slot in enumerate("QWER"):
        mana = spell(name, slot).get("mana")
        if mana is None:
            cols.append(jnp.zeros(ranks.shape[:1], jnp.float32))
            continue
        table = jnp.asarray(mana["values"], jnp.float32)
        cols.append(table[jnp.clip(ranks[:, i] - 1, 0, table.shape[0] - 1)])
    return jnp.stack(cols, -1)


def tick_index(kctx: KitCtx) -> Any:
    return jnp.round(jnp.asarray(kctx.now) / jnp.maximum(jnp.asarray(kctx.dt), EPS)).astype(jnp.int32)


def make_cast_id(kctx: KitCtx, code: Any) -> Any:
    c = kctx.unit.shape[0]
    holder = jnp.arange(c, dtype=jnp.int32)
    return (KIT_ID_BASE + (tick_index(kctx) * c + holder) * ID_STRIDE + jnp.asarray(code, jnp.int32)).astype(jnp.int32)


def due(kctx: KitCtx, until: Any) -> Any:
    """The timer ending at ``until`` expires during this tick."""
    return kctx.now + kctx.dt >= until - EPS


def later(kctx: KitCtx, seconds: Any) -> Any:
    """``until`` for a timer started this tick that expires ``seconds`` later."""
    return f32(kctx.now + seconds)


def later_after_tick(kctx: KitCtx, seconds: Any) -> Any:
    """``until`` for a timer that starts counting next tick (on-hit stacks, post-advance buffs)."""
    return f32(kctx.now + kctx.dt + seconds)


def is_structure(kind: Any) -> Any:
    kind = jnp.asarray(kind)
    out = jnp.zeros(kind.shape, bool)
    for k in STRUCTURE_KINDS:
        out = out | (kind == k)
    return out


def gather(arr: Any, idx: Any) -> Any:
    return arr[jnp.clip(idx, 0, arr.shape[0] - 1)]


def enemies(kctx: KitCtx, units) -> Any:
    """(C, N) living, targetable enemy non-structure, non-ward units (spell AoE targets;
    wards only take basic-attack hits)."""
    return (units.team[None, :] != kctx.team[:, None]) & units.alive[None, :] & units.targetable[None, :] \
        & (units.kind[None, :] != KIND_NONE) & (units.kind[None, :] != KIND_WARD) & ~is_structure(units.kind)[None, :]


def center_dist(kctx: KitCtx, units) -> Any:
    return jnp.sqrt((units.x[None, :] - kctx.x[:, None]) ** 2 + (units.y[None, :] - kctx.y[:, None]) ** 2)


def within_edge(kctx: KitCtx, units, radius: float) -> Any:
    """(C, N) units whose hitbox touches the circle of ``radius`` around the holder."""
    return center_dist(kctx, units) <= radius + units.radius[None, :]


def target_dist(kctx: KitCtx, units, target: Any) -> Any:
    t = jnp.clip(target, 0, units.x.shape[0] - 1)
    return jnp.sqrt((units.x[t] - kctx.x) ** 2 + (units.y[t] - kctx.y) ** 2)


def holder_rows(x: Any, kctx: KitCtx) -> Any:
    """Per-holder rows of an (N,) or (C,) array (champions are units 0..C-1)."""
    x = jnp.asarray(x)
    return x if x.shape[0] == kctx.unit.shape[0] else x[kctx.unit]


def no_out(c: int, n: int) -> KitOut:
    zc = jnp.zeros((c,), jnp.float32)
    zs = jnp.zeros((c, 0), jnp.float32)
    return KitOut(D.empty_packets(0), no_cc(c, n), no_dash(c), zc,
                  ShieldGrant(zs, zs.astype(jnp.int32), zs, zs), zc,
                  jnp.zeros((c, 4), bool), jnp.zeros((c, 4), jnp.float32), jnp.zeros((c,), bool),
                  jnp.zeros((c,), bool), jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), jnp.int32), zc,
                  jnp.zeros((c,), bool), jnp.full((c,), -1, jnp.int32))


def out(c: int, n: int, **fields) -> KitOut:
    return no_out(c, n)._replace(**fields)


def merge_out(parts: list[KitOut], c: int, n: int) -> KitOut:
    acc = no_out(c, n)
    for p in parts:
        cs = p.cleanse_slow if p.cleanse_slow is not None else jnp.zeros((c,), bool)
        at = p.attack_target if p.attack_target is not None else jnp.full((c,), -1, jnp.int32)
        acc = KitOut(
            D.concat_packets(acc.packets, p.packets), merge_cc(acc.cc, p.cc),
            Dash(*(jnp.where(p.dash.active, b, a) for a, b in zip(acc.dash, p.dash))),
            acc.heal + p.heal,
            ShieldGrant(*(jnp.concatenate([a, b], axis=1) for a, b in zip(acc.shield, p.shield))),
            acc.mana_cost + p.mana_cost, acc.cooldown_start | p.cooldown_start,
            jnp.where(p.base_cooldown > 0, p.base_cooldown, acc.base_cooldown),
            acc.attack_reset | p.attack_reset, acc.cast_started | p.cast_started,
            jnp.where(p.cast_started, p.cast_slot, acc.cast_slot),
            jnp.where(p.cast_started, p.cast_id, acc.cast_id),
            jnp.maximum(acc.cast_lockout, p.cast_lockout), acc.cleanse_slow | cs,
            jnp.where(at >= 0, at, acc.attack_target).astype(jnp.int32))
    return acc


def cc_matrix(mask: Any, duration: Any, cast_id: Any) -> tuple[Any, Any]:
    """(C, N) durations and cast ids for a CC applied where ``mask``."""
    dur = jnp.where(mask, jnp.broadcast_to(f32(duration), mask.shape), 0.0).astype(jnp.float32)
    cid = jnp.where(mask, jnp.broadcast_to(jnp.asarray(cast_id, jnp.int32), mask.shape), 0).astype(jnp.int32)
    return dur, cid


def neutral_defense(c: int) -> KitDefense:
    o, z = jnp.ones((c,), jnp.float32), jnp.zeros((c,), jnp.float32)
    return KitDefense(o, jnp.zeros((c,), bool), o, z)


def neutral_attack_mods(c: int) -> KitAttackMods:
    f = jnp.zeros((c,), bool)
    z = jnp.zeros((c,), jnp.float32)
    return KitAttackMods(z, f, f, f, z, z, f)


def combine_attack_mods(a: KitAttackMods, b: KitAttackMods) -> KitAttackMods:
    """Merge two holders' kit mods (kits are gated per holder, so at most one is non-neutral)."""
    o = lambda x, d: d if x is None else x          # noqa: E731
    c = a.extra_range.shape[0]
    f, z = jnp.zeros((c,), bool), jnp.zeros((c,), jnp.float32)
    return KitAttackMods(a.extra_range + b.extra_range, a.attack_reset | b.attack_reset,
                         a.cannot_attack | b.cannot_attack, o(a.cannot_crit, f) | o(b.cannot_crit, f),
                         jnp.maximum(o(a.windup, z), o(b.windup, z)), jnp.maximum(o(a.period, z), o(b.period, z)),
                         o(a.uncancellable, f) | o(b.uncancellable, f))


def pulses(kctx: KitCtx, period: float) -> Any:
    """Number of ``period`` grid boundaries crossed by this tick (world HP-regen convention)."""
    now = jnp.asarray(kctx.now, jnp.float32)
    return jnp.floor(now / period + 1e-6) - jnp.floor((now - kctx.dt) / period + 1e-6)
