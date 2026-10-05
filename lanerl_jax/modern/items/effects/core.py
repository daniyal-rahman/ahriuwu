"""Shared contract for 26.19 item passives and actives (ITEMS.md §10–11).

Every effect module is pure and fixed-shape. Arrays with a leading ``C`` axis
are per champion holder; ``N`` axes index world units. A champion holder ``c``
is world unit ``ctx.unit[c]``. Modules never mutate world state: they return
``Effects`` (damage packets, heals, shields, debuffs, gold) and contribute to
stat, defense and offense profiles which the integrator folds into the
damage pipeline (core.damage) in the canonical order of
docs/modern/DAMAGE_AND_STATS.md §2.

Module protocol (all optional except ``init``/``COVERAGE``):

    COVERAGE: dict[int, str]                      item id -> what is implemented
    State: NamedTuple; init(n_champions, n_units) -> State
    stats(state, own, ctx) -> ItemStats           STAT.50 dynamic bonus stats
    defense(state, own, ctx) -> HolderDefense     holder-side DMG.10–80 inputs
    status(state, own, ctx) -> StatusFlags        collision/movement flags
    debuffs(state, own, ctx, units) -> Debuffs    target-side reductions/amps (N,)
    dealt_amp(state, own, ctx, units) -> (C, N)   DMG.40 additive amp per target (all packets)
    packet_amp(state, own, ctx, units, packets) -> (P,)  DMG.40 amp filtered by packet tags/type
    on_cc(state, own, ctx, units, cc) -> (state, Effects)    holder slowed/immobilized units
    attack_mods(state, own, ctx, units, target) -> AttackMods   at launch
    on_attack(state, own, ctx, units, attack) -> (state, Effects)   launch
    on_hit(state, own, ctx, units, attack) -> (state, Effects)      land
    on_cast(state, own, ctx, units, cast) -> (state, Effects)
    on_damage(state, own, ctx, units, report) -> (state, Effects)   after resolve
    periodic(state, own, ctx, units) -> (state, Effects)            every tick
    on_takedown(state, own, ctx, units, kills) -> (state, Effects)
    active(state, own, ctx, units, request) -> (state, Effects, ActiveOut)
                                                  called EVERY tick (request 0 = none) so
                                                  pending cast times resolve
    on_shop(state, own, ctx) -> state                               in shop area

``own`` is the (C, I) owned-count matrix from ``items.inventory.owned_counts``.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import (CLASS_CHAMPION, CLASS_MINION, CLASS_MONSTER, CLASS_STRUCTURE, SHIELD_ALL,
                            Packets, Resolved, concat_packets, empty_packets, has)
from ..catalog import catalog

BIG = 1e9


def row(item_id: int) -> int:
    return catalog().row(item_id)


def holds(own: Any, item_id: int) -> Any:
    """(C,) bool: holder owns ``item_id``."""
    return own[:, row(item_id)] > 0


def holds_any(own: Any, item_ids) -> Any:
    out = jnp.zeros(own.shape[:1], bool)
    for iid in item_ids:
        out = out | holds(own, iid)
    return out


def dv(item_id: int, name: str, default: float | None = None) -> float:
    return catalog().dv(item_id, name, default)


class Ctx(NamedTuple):
    """Per-holder context for one tick, shape (C,) unless noted.

    Stats are the *pre-dynamic* values: champion base/growth plus static
    item, shard and rune stats, before any module's ``stats`` contribution
    (ITEMS.md §2.1 dependent-stat order).
    """
    now: Any                # () seconds
    dt: Any                 # () seconds
    unit: Any               # int32 world unit index
    team: Any
    alive: Any
    level: Any
    is_ranged: Any
    x: Any
    y: Any
    facing_x: Any
    facing_y: Any
    moved: Any              # distance moved this tick
    base_ad: Any
    bonus_ad: Any
    ap: Any
    base_hp: Any
    max_hp: Any
    hp: Any
    base_armor: Any
    bonus_armor: Any
    base_mr: Any
    bonus_mr: Any
    mana: Any
    max_mana: Any
    base_ms: Any
    move_speed: Any
    crit_chance: Any
    crit_damage: Any        # total crit multiplier (2.0 base + items)
    life_steal: Any
    bonus_attack_speed: Any  # bonus AS ratio (growth + items), e.g. 0.45
    ability_haste: Any
    lethality: Any
    heal_shield_power: Any
    attack_windup: Any      # current basic-attack windup (s)
    in_combat: Any          # dealt/took damage with an enemy in the last 5 s
    in_shop: Any
    base_mana: Any = 0.0      # champion base + growth mana ("bonus mana" = max_mana - base_mana)
    attack_range: Any = 125.0  # holder basic-attack range (center to edge of target radius)

    @property
    def total_ad(self):
        return self.base_ad + self.bonus_ad

    @property
    def bonus_hp(self):
        return self.max_hp - self.base_hp

    @property
    def armor(self):
        return self.base_armor + self.bonus_armor

    @property
    def magic_resist(self):
        return self.base_mr + self.bonus_mr

    @property
    def moving(self):
        return self.moved > 0.0


class Units(NamedTuple):
    """World units visible to item effects, shape (N,)."""
    x: Any
    y: Any
    team: Any
    cls: Any                # CLASS_* (core.damage)
    alive: Any
    hp: Any
    max_hp: Any
    radius: Any             # gameplay radius
    targetable: Any
    is_siege_or_super: Any  # lane minion subtype for Hullbreaker
    bonus_hp: Any           # LDR Giant Slayer reads target bonus HP
    armor: Any
    magic_resist: Any


class Attack(NamedTuple):
    """At most one basic attack per holder per tick (C,)."""
    launched: Any           # windup completed this tick (on-attack)
    hit: Any                # attack landed this tick (on-hit)
    target: Any             # int32 unit index
    raw: Any                # base attack pre-mitigation damage after crit
    is_crit: Any
    natural_crit: Any = None  # crit from the crit-chance roll (not forced by attack_mods); None = is_crit


class Cast(NamedTuple):
    started: Any            # (C,) bool: an ability cast started this tick
    slot: Any               # (C,) int32 0..3 = Q/W/E/R
    target: Any             # (C,) int32 unit index or -1

    @property
    def is_ultimate(self):
        return self.slot == 3


class CC(NamedTuple):
    """Crowd control applied by holder c to unit n this tick, (C, N) bool."""
    slowed: Any
    immobilized: Any


class Kills(NamedTuple):
    champion_kill: Any      # (C,) count of champion kills credited this tick
    champion_assist: Any    # (C,) assists
    minion_kill: Any        # (C,) minions last-hit by the holder
    holder_died: Any        # (C,) bool
    killed_units: Any       # (C, N) bool: holder had takedown on unit n this tick


class Report(NamedTuple):
    """Resolved packets of the current tick (TICK.70)."""
    packets: Packets
    resolved: Resolved
    life_steal_heal: Any    # (N,) life-steal heal per source unit (before HEAL.20/30)
    omnivamp_heal: Any = None  # (N,) omnivamp heal per source unit (before HEAL.20/30)


class AttackMods(NamedTuple):
    force_crit: Any         # (C,) bool
    crit_scale: Any         # (C,) crit-bonus multiplier (Sundered Sky 0.8)


class HolderDefense(NamedTuple):
    """Holder-side DMG inputs (C,), folded into core.damage.Defense."""
    received_mult: Any
    basic_attack_mult: Any
    crit_taken_mult: Any
    champion_attack_block: Any
    postmit_flat: Any
    store_fraction: Any
    lifeline_ready: Any
    lifeline_magic_only: Any
    lifeline_shield: Any
    lifeline_shield_kind: Any
    lifeline_duration: Any
    lifeline_decay_hold: Any
    lifeline_bonus_health: Any
    spell_shield: Any       # Annul ready (Banshee's/Verdant Barrier/Edge of Night)
    champion_received_mult: Any = None  # reduction on damage from enemy champions only (Celestial)


def neutral_defense(n: int) -> HolderDefense:
    o, z = jnp.ones((n,), jnp.float32), jnp.zeros((n,), jnp.float32)
    f = jnp.zeros((n,), bool)
    return HolderDefense(o, o, o, z, z, z, f, f, z, jnp.full((n,), SHIELD_ALL, jnp.int32), z,
                         jnp.full((n,), jnp.inf), z, f, o)


def combine_defense(parts: list[HolderDefense], n: int) -> HolderDefense:
    out = neutral_defense(n)
    for p in parts:
        ready = p.lifeline_ready
        out = HolderDefense(
            out.received_mult * p.received_mult, out.basic_attack_mult * p.basic_attack_mult,
            out.crit_taken_mult * p.crit_taken_mult, out.champion_attack_block + p.champion_attack_block,
            out.postmit_flat + p.postmit_flat, out.store_fraction + p.store_fraction,
            out.lifeline_ready | ready, jnp.where(ready, p.lifeline_magic_only, out.lifeline_magic_only),
            jnp.where(ready, p.lifeline_shield, out.lifeline_shield),
            jnp.where(ready, p.lifeline_shield_kind, out.lifeline_shield_kind),
            jnp.where(ready, p.lifeline_duration, out.lifeline_duration),
            jnp.where(ready, p.lifeline_decay_hold, out.lifeline_decay_hold),
            jnp.where(ready, p.lifeline_bonus_health, out.lifeline_bonus_health),
            out.spell_shield | p.spell_shield,
            out.champion_received_mult * (1.0 if p.champion_received_mult is None else p.champion_received_mult))
    return out


class Debuffs(NamedTuple):
    """Target-side modifiers from holders' effects, shape (N,)."""
    percent_armor_reduction: Any    # combined 1 - prod(1 - p)
    flat_armor_reduction: Any
    percent_mr_reduction: Any
    flat_mr_reduction: Any
    received_amp: Any               # additive vulnerability, all damage types
    magic_received_amp: Any         # additive, magic only (Abyssal Mask)
    attack_speed_cripple: Any       # strongest-only AS reduction (Frozen Heart)


def neutral_debuffs(n: int) -> Debuffs:
    z = jnp.zeros((n,), jnp.float32)
    return Debuffs(z, z, z, z, z, z, z)


def combine_debuffs(parts: list[Debuffs], n: int) -> Debuffs:
    out = neutral_debuffs(n)
    for p in parts:
        out = Debuffs(1 - (1 - out.percent_armor_reduction) * (1 - p.percent_armor_reduction),
                      out.flat_armor_reduction + p.flat_armor_reduction,
                      1 - (1 - out.percent_mr_reduction) * (1 - p.percent_mr_reduction),
                      out.flat_mr_reduction + p.flat_mr_reduction,
                      out.received_amp + p.received_amp, out.magic_received_amp + p.magic_received_amp,
                      jnp.maximum(out.attack_speed_cripple, p.attack_speed_cripple))
    return out


class ShieldGrant(NamedTuple):
    """Shields granted to holders this tick, shape (C, S)."""
    amount: Any
    kind: Any
    duration: Any
    decay_hold: Any


def shield_grants(amount: Any, kind: Any = SHIELD_ALL, duration: Any = 0.0,
                  decay_hold: Any = jnp.inf) -> ShieldGrant:
    """Single grant per holder; ``amount`` (C,) with 0 meaning none."""
    a = jnp.asarray(amount, jnp.float32)
    b = lambda v, t=jnp.float32: jnp.broadcast_to(jnp.asarray(v, t), a.shape)[:, None]
    return ShieldGrant(a[:, None], b(kind, jnp.int32), b(duration), b(decay_hold))


class Effects(NamedTuple):
    """Everything a hook emits besides its own state."""
    packets: Packets
    heal: Any               # (C,) self heal that benefits from heal and shield power
    heal_plain: Any         # (C,) self heal without HSP (regen-like, potions)
    mana: Any               # (C,) mana restored
    shields: ShieldGrant    # (C, S)
    slow: Any               # (N,) slow strength applied this tick (strongest wins)
    slow_duration: Any      # (N,) seconds for that slow
    grievous: Any           # (N,) Grievous Wounds duration applied (0 = none)
    gold: Any               # (C,) gold granted
    attack_reset: Any       # (C,) bool
    revive: Any             # (C,) bool: holder's lethal damage this tick is replaced by a revive
    revive_delay: Any       # (C,) seconds of stasis before the revive completes
    revive_hp: Any          # (C,) health on revive completion


def no_effects(n_champions: int, n_units: int) -> Effects:
    zc, zn = jnp.zeros((n_champions,), jnp.float32), jnp.zeros((n_units,), jnp.float32)
    zs = jnp.zeros((n_champions, 0), jnp.float32)
    return Effects(empty_packets(0), zc, zc, zc,
                   ShieldGrant(zs, zs.astype(jnp.int32), zs, zs), zn, zn, zn, zc,
                   jnp.zeros((n_champions,), bool), jnp.zeros((n_champions,), bool), zc, zc)


def merge_effects(parts: list[Effects], n_champions: int, n_units: int) -> Effects:
    out = no_effects(n_champions, n_units)
    for p in parts:
        stronger = p.slow > out.slow
        out = Effects(
            concat_packets(out.packets, p.packets), out.heal + p.heal, out.heal_plain + p.heal_plain,
            out.mana + p.mana,
            ShieldGrant(*(jnp.concatenate([a, b], axis=1) for a, b in zip(out.shields, p.shields))),
            jnp.where(stronger, p.slow, out.slow),
            jnp.where(stronger, p.slow_duration,
                      jnp.where(p.slow == out.slow, jnp.maximum(out.slow_duration, p.slow_duration),
                                out.slow_duration)),
            jnp.maximum(out.grievous, p.grievous), out.gold + p.gold,
            out.attack_reset | p.attack_reset, out.revive | p.revive,
            jnp.where(p.revive, p.revive_delay, out.revive_delay),
            jnp.where(p.revive, p.revive_hp, out.revive_hp))
    return out


def effects(n_champions: int, n_units: int, **fields) -> Effects:
    """Build Effects with defaults for unspecified fields."""
    return no_effects(n_champions, n_units)._replace(**fields)


class StatusFlags(NamedTuple):
    """Holder movement/collision flags (C,) from the ``status`` hook."""
    ghosted: Any            # ignores unit collision (Phantom Dancer)


class ActiveOut(NamedTuple):
    """Per-holder result of an item active request (C,)."""
    used: Any               # bool: the request started a cast
    cast_time: Any          # seconds of cast lockout
    can_move: Any           # bool: movement allowed during the cast
    attack_reset: Any       # bool


# ---- geometry helpers -------------------------------------------------------

def enemy_mask(ctx: Ctx, units: Units) -> Any:
    """(C, N): living, targetable enemy units of each holder."""
    return (units.team[None, :] != ctx.team[:, None]) & units.alive[None, :] & units.targetable[None, :]


def dist_to_point(units: Units, px: Any, py: Any) -> Any:
    """(C, N) center distance from per-holder points to every unit."""
    return jnp.sqrt((units.x[None, :] - px[:, None]) ** 2 + (units.y[None, :] - py[:, None]) ** 2)


def in_circle(units: Units, px: Any, py: Any, radius: Any, *, edge: bool = True) -> Any:
    """(C, N) units whose hitbox (edge rule, U-2) touches the circle."""
    reach = jnp.asarray(radius)[..., None] + (units.radius[None, :] if edge else 0.0)
    return dist_to_point(units, px, py) <= reach


def nearest_k(dist: Any, mask: Any, k: int) -> Any:
    """(C, N) mask of the ``k`` smallest ``dist`` entries within ``mask`` (ties: lower index)."""
    # ``k`` (static, <= 10 in the item modules) rounds of argmin: cheap row reductions on every
    # backend (``lax.top_k`` over 216 columns was the top GPU kernel); argmin keeps the lower index.
    key = jnp.where(mask, dist, jnp.inf)
    cols = jnp.arange(key.shape[-1])
    pick = jnp.zeros(key.shape, bool)
    for _ in range(min(k, key.shape[-1])):
        hit = cols == jnp.argmin(key, axis=-1)[..., None]
        pick, key = pick | hit, jnp.where(hit, jnp.inf, key)
    return mask & pick


def unit_pos(units: Units, idx: Any) -> tuple[Any, Any]:
    i = jnp.clip(idx, 0, units.x.shape[0] - 1)
    return units.x[i], units.y[i]


def target_class(units: Units, idx: Any) -> Any:
    return units.cls[jnp.clip(idx, 0, units.cls.shape[0] - 1)]


def onehot_units(idx: Any, n_units: int) -> Any:
    """(C, N) bool from per-holder unit indices (negative = none)."""
    return (jnp.arange(n_units)[None, :] == idx[:, None]) & (idx[:, None] >= 0)


# ---- report helpers ---------------------------------------------------------

def dealt_by_holder(report: Report, ctx: Ctx, n_units: int, mask: Any = None) -> Any:
    """(C, N) post-mitigation damage holder c dealt to unit n this tick."""
    p, r = report.packets, report.resolved
    sel = p.valid & (r.final > 0.0) if mask is None else p.valid & mask
    src_is = p.src[None, :] == ctx.unit[:, None]                      # (C, P)
    contrib = jnp.where(src_is & sel[None, :], r.final[None, :], 0.0)
    onehot = (p.dst[:, None] == jnp.arange(n_units)[None, :])         # (P, N)
    return contrib @ onehot.astype(jnp.float32)


def hit_by_holder(report: Report, ctx: Ctx, n_units: int, mask: Any) -> Any:
    """(C, N) bool: some selected packet from holder c reached unit n."""
    p = report.packets
    src_is = p.src[None, :] == ctx.unit[:, None]
    sel = (src_is & (p.valid & mask)[None, :]).astype(jnp.float32)
    onehot = (p.dst[:, None] == jnp.arange(n_units)[None, :]).astype(jnp.float32)
    return (sel @ onehot) > 0.0


def taken_by_holder(report: Report, ctx: Ctx, mask: Any = None) -> Any:
    """(C,) post-mitigation damage holder c received (before shields)."""
    p, r = report.packets, report.resolved
    sel = p.valid if mask is None else p.valid & mask
    dst_is = p.dst[None, :] == ctx.unit[:, None]
    return jnp.sum(jnp.where(dst_is & sel[None, :], r.final[None, :], 0.0), axis=1)


def src_class(report: Report, units: Units) -> Any:
    return units.cls[jnp.clip(report.packets.src, 0, units.cls.shape[0] - 1)]


def dst_class(report: Report, units: Units) -> Any:
    return units.cls[jnp.clip(report.packets.dst, 0, units.cls.shape[0] - 1)]


__all__ = [n for n in dir() if not n.startswith("_")] + [
    "CLASS_CHAMPION", "CLASS_MINION", "CLASS_MONSTER", "CLASS_STRUCTURE", "has"]
