"""Shared contract for 26.19 rune effects (RUNES.md §8–9).

Rune kernels follow the item-effect contract (``modern_item_effects.core``):
pure, fixed-shape JAX; (C,) per champion holder, (N,) per world unit; holder
``c`` is world unit ``ctx.unit[c]``. They reuse the item ``Ctx``, ``Units``,
``Effects`` and ``Report`` types. Two differences:

* ownership is the (C, R) page-count matrix ``page`` from
  ``modern_rune_data.page_counts`` (``has_rune(page, id)``);
* every hook also receives ``ev: RuneEvents``, the world events of this tick
  (attacks, casts, CC, kills, summoner casts, blinks, purchases ...) plus
  runtime-managed combat clocks and the post-STAT.50 offensive stats.

Module protocol (all optional except ``init``/``COVERAGE``/``State``):

    COVERAGE: dict[int, str]                         perk id -> what is implemented
    State: NamedTuple; init(n_champions, n_units) -> State
    stats(state, page, ctx, ev) -> ItemStats         STAT.20–50 rune stats (AF unresolved)
    debuffs(state, page, ctx, units, ev) -> Debuffs
    packet_amp(state, page, ctx, units, ev, packets) -> (P,)   DMG.40 additive amp
    packet_block(state, page, ctx, units, ev, packets) -> (P,) DMG.70 flat block (all types)
    heal_mult(state, page, ctx, ev) -> (C,)          × on heals/shields the holder receives
    on_cast / on_attack / on_hit / on_cc / periodic / on_damage / on_takedown
        (state, page, ctx, units, ev) -> (state, Effects)
    post_tick(state, page, ctx, units, ev) -> state  after heals/shields are applied
    outputs(state, page, ctx, ev) -> RuneOutputs     world-applied results

``ev.report`` is the resolved ``Report`` inside ``on_damage`` (main pass,
then the follow-up pass) and ``None`` elsewhere. Damage packets carry rune
provenance as ``item = -perk_id`` (``rune_item``). Rune proc damage is
tagged ``TAG_PROC`` so other runes and items can exclude it. Pet damage
must be emitted with ``src`` = the owner champion's unit and ``TAG_PET``
(packets carry no separate owner field; RUNES U-02).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from ..modern_damage import MAGIC, PHYSICAL, Packets, empty_packets, has
from ..modern_item_effects.core import (CC, Attack, Cast, Ctx, Effects, Kills, Report, Units,  # noqa: F401
                                        effects, enemy_mask, merge_effects, no_effects, onehot_units,
                                        target_class, unit_pos, dealt_by_holder, hit_by_holder,
                                        taken_by_holder, dist_to_point, in_circle, nearest_k,
                                        neutral_debuffs, Debuffs, BIG, src_class, dst_class)
from ..modern_rune_data import (breakpoints, ea, has_rune, level_table, lin, lin_growth,  # noqa: F401
                                rune_catalog, rune_count)


def rune_item(perk_id: int) -> int:
    """Packet provenance value for rune damage."""
    return -int(perk_id)


class CombatClocks(NamedTuple):
    """Runtime-managed combat timers, shape (C,); seconds, -BIG = never.

    ``last_combat``: dealt/took damage (incl. 0) to/from an enemy champion,
    minion, monster or turret (the "damage" combat system, RUNES §1.5).
    ``last_champion_combat``: same, restricted to enemy champions (incl.
    their pets), also counting CC dealt to an enemy champion.
    ``last_hit_by_champion``: health damage > 0 taken from an enemy champion
    ("after taking damage from an enemy champion").
    ``champion_combat_start``: when the current champion-combat episode began
    (an episode starts after ``CHAMPION_COMBAT_GAP`` s without champion combat).
    ``struck_first``: the holder dealt the first damage of the current episode.
    ``last_combat_modern``: the "modern" combat system (RUNES §1.5): also counts
    non-damage CC and hits on invulnerable targets (Relentless Hunter, Flashtraption).
    """
    last_combat: Any
    last_champion_combat: Any
    last_hit_by_champion: Any
    champion_combat_start: Any
    struck_first: Any
    last_combat_modern: Any = None


COMBAT_TIMEOUT = 5.0        # out of combat 5 s after the last event (RUNES §1.5)
CHAMPION_COMBAT_GAP = 10.0  # First Strike OOCTimer: a new episode needs 10 s without champion combat


def init_clocks(n_champions: int) -> CombatClocks:
    z = jnp.full((n_champions,), -BIG, jnp.float32)
    return CombatClocks(z, z, z, z, jnp.zeros((n_champions,), bool), z)


class RuneEvents(NamedTuple):
    """World events and runtime context for one tick. Build with ``rune_events``.

    Shapes: (C,) per holder unless noted; (C, N) holder x unit; (N,) per unit.
    """
    game_time: Any                  # () seconds since game start (Gathering Storm, Conditioning, biscuits)
    attack: Attack                  # basic attack launched/landed this tick
    attack_started: Any             # windup started this tick (Hail of Blades trigger)
    attack_start_target: Any        # int32 unit of that windup
    attack_cancelled: Any           # windup cancelled this tick
    attack_reset: Any               # a Trait_AttackReset effect fired (HoB bonus stack)
    cast: Cast                      # ability cast started this tick
    cast_id: Any                    # int32 instance id of that cast (matches its packets' cast_id)
    cc: CC                          # (C, N) holder applied slow / immobilize this tick
    cc_duration: Any                # (C, N) immobilize duration after tenacity (Glacial zone length)
    impaired: Any                   # (N,) unit has an impairment Cheap Shot accepts (tick start)
    movement_impaired: Any          # (N,) immobilized, grounded or slowed (Approach Velocity)
    impaired_by_holder: Any         # (C, N) holder's own active movement impairment on the unit
    holder_cc_from_champion: Any    # holder is under non-kinematic CC from an enemy champion (Unflinching)
    summoner_cast: Any              # a summoner spell completed its cast/channel this tick (Nimbus)
    summoner_cooldown: Any          # hasted cooldown (s) of that spell
    summoner_is_teleport: Any
    blinked: Any                    # dash, blink, Flash, TP arrival, recall, stealth exit (Sudden Impact)
    flash_cooldown: Any             # remaining Flash cooldown (Hexflash availability)
    hexflash_request: Any           # int32 0 none, 1 start channel, 2 release
    kills: Kills
    deaths: Any                     # (N,) units that died this tick (any killer)
    sight: Any                      # (C, N) holder has direct line of sight to the unit
    visible: Any                    # (C, N) unit visible to the holder's team
    large_monster_kill: Any         # (C,) count
    epic_takedown: Any              # (C,) count
    execute_credit: Any             # (C,) champion kills credited to the holder but dealt by non-champions
    shield_gained: Any              # largest shield initial amount granted to the holder this tick
    shield_gained_duration: Any     # duration (s) of that shield (Shield Bash window = shield life + 2 s)
    summoner_haste: Any             # holder's total summoner spell haste (Hexflash cooldowns)
    cc_cast_id: Any                 # (C, N) int32 cast instance of the CC in ``cc`` (0 = unknown)
    cc_on_hit: Any                  # (C, N) CC in ``cc`` was applied on-hit (Cheap Shot exception)
    purchased: Any                  # int32 item id bought this tick (0 none)
    sold: Any                       # int32 item id sold this tick (0 none)
    potion_drunk: Any               # int32 item id of a potion started this tick (0 none)
    granted: Any                    # int32 item id the world placed in the inventory for this rune (0 none)
    spellbook_request: Any          # int32 summoner spell id to swap to (Unsealed Spellbook, 0 none)
    is_turret: Any                  # (N,) structure unit is a turret (Demolish)
    in_river: Any
    uses_energy: Any                # resource is energy (Presence of Mind)
    adaptive_physical: Any          # champion adaptive type for ties
    own: Any                        # (C, I) item counts (Jack of All Trades, Magical Footwear)
    bonus_ad: Any                   # post-STAT.50 bonus AD (rune formulas read these)
    ap: Any                         # post-STAT.50 AP
    bonus_attack_speed: Any         # post-STAT.50 bonus AS ratio (Lethal Tempo bolt)
    clocks: CombatClocks
    report: Any = None              # Report inside on_damage, else None


def rune_events(ctx: Ctx, n_units: int, **kw) -> RuneEvents:
    """Default (quiet) events for ``ctx``; override any field by keyword."""
    c = ctx.level.shape[0]
    zc, fc = jnp.zeros((c,), jnp.float32), jnp.zeros((c,), bool)
    ic = jnp.zeros((c,), jnp.int32)
    zcn = jnp.zeros((c, n_units), bool)
    base = dict(
        game_time=jnp.asarray(ctx.now, jnp.float32),
        attack=Attack(fc, fc, ic, zc, fc), attack_started=fc, attack_start_target=ic, attack_cancelled=fc,
        attack_reset=fc, cast=Cast(fc, ic, ic - 1), cast_id=ic, cc=CC(zcn, zcn),
        cc_duration=jnp.zeros((c, n_units), jnp.float32), impaired=jnp.zeros((n_units,), bool),
        movement_impaired=jnp.zeros((n_units,), bool), impaired_by_holder=zcn, holder_cc_from_champion=fc,
        summoner_cast=fc, summoner_cooldown=zc, summoner_is_teleport=fc, blinked=fc,
        flash_cooldown=zc, hexflash_request=ic,
        kills=Kills(zc, zc, zc, fc, zcn), deaths=jnp.zeros((n_units,), bool),
        sight=jnp.ones((c, n_units), bool), visible=jnp.ones((c, n_units), bool),
        large_monster_kill=zc, epic_takedown=zc, execute_credit=zc, shield_gained=zc, shield_gained_duration=zc, summoner_haste=zc,
        cc_cast_id=jnp.zeros((c, n_units), jnp.int32), cc_on_hit=zcn, purchased=ic, sold=ic,
        potion_drunk=ic, granted=ic, spellbook_request=ic, is_turret=jnp.zeros((n_units,), bool), in_river=fc, uses_energy=fc, adaptive_physical=~fc, own=None,
        bonus_ad=ctx.bonus_ad, ap=ctx.ap, bonus_attack_speed=ctx.bonus_attack_speed,
        clocks=init_clocks(c), report=None)
    base.update(kw)
    return RuneEvents(**base)


class RuneOutputs(NamedTuple):
    """World-applied rune results, shape (C,) unless noted."""
    grant_item: Any                 # int32 item id to add to the inventory (0 none); repeated each
                                    # tick until the world acknowledges it via ``ev.granted``
    forbid_purchase: Any            # (C, I) bool: purchases the rune blocks (Magical Footwear)
    skill_points: Any               # int32 extra skill points granted this tick
    basic_cd_refund: Any            # fraction of *current* Q/W/E cooldowns removed now (Transcendence)
    ult_cd_refund: Any              # fraction of current R cooldown removed now (Axiom Arcanist)
    move_locked: Any                # MS set to 0 (Hexflash channel)
    blink: Any                      # Hexflash releases a blink this tick
    blink_range: Any
    spellbook_swap_ready: Any       # Unsealed Spellbook may swap a summoner spell now
    first_strike_gold: Any          # gold awarded this tick (also in Effects.gold)
    ghosted: Any                    # ignores unit collision (Nimbus Cloak)


def no_outputs(n_champions: int, n_items: int) -> RuneOutputs:
    zc, fc = jnp.zeros((n_champions,), jnp.float32), jnp.zeros((n_champions,), bool)
    ic = jnp.zeros((n_champions,), jnp.int32)
    return RuneOutputs(ic, jnp.zeros((n_champions, n_items), bool), ic, zc, zc, fc, fc, zc, fc, zc, fc)


def merge_outputs(parts: list[RuneOutputs], n_champions: int, n_items: int) -> RuneOutputs:
    out = no_outputs(n_champions, n_items)
    for p in parts:
        out = RuneOutputs(
            jnp.where(p.grant_item != 0, p.grant_item, out.grant_item), out.forbid_purchase | p.forbid_purchase,
            out.skill_points + p.skill_points, 1 - (1 - out.basic_cd_refund) * (1 - p.basic_cd_refund),
            1 - (1 - out.ult_cd_refund) * (1 - p.ult_cd_refund), out.move_locked | p.move_locked,
            out.blink | p.blink, jnp.maximum(out.blink_range, p.blink_range),
            out.spellbook_swap_ready | p.spellbook_swap_ready, out.first_strike_gold + p.first_strike_gold,
            out.ghosted | p.ghosted)
    return out


# ---- shared rune helpers ----------------------------------------------------

def holder_is_melee(ctx: Ctx) -> Any:
    return ~ctx.is_ranged


def by_range(ctx: Ctx, melee: Any, ranged: Any) -> Any:
    return jnp.where(ctx.is_ranged, ranged, melee)


def adaptive_damage_type(ev: RuneEvents) -> Any:
    """Adaptive damage (RUNES §1.2): physical if bonus AD > AP, magic if
    AP > bonus AD, champion adaptive type on a tie."""
    return jnp.where(ev.bonus_ad > ev.ap, PHYSICAL,
                     jnp.where(ev.ap > ev.bonus_ad, MAGIC, jnp.where(ev.adaptive_physical, PHYSICAL, MAGIC)))


def variable_damage_type(ad_term: Any, ap_term: Any) -> Any:
    """Electrocute/Comet "variable" damage: whichever ratio term contributes
    more; a tie or zero is magic (RUNES §1.2)."""
    return jnp.where(ad_term > ap_term, PHYSICAL, MAGIC)


def enemy_champions(ctx: Ctx, units: Units) -> Any:
    """(C, N) living enemy champion units."""
    from ..modern_damage import CLASS_CHAMPION
    return (units.cls[None, :] == CLASS_CHAMPION) & (units.team[None, :] != ctx.team[:, None]) \
        & units.alive[None, :]


def packets_from_holder(report_or_packets, ctx: Ctx) -> Any:
    """(C, P) packet sourced by holder c."""
    p = report_or_packets.packets if isinstance(report_or_packets, Report) else report_or_packets
    return p.valid[None, :] & (p.src[None, :] == ctx.unit[:, None])


def packets_to_holder(report_or_packets, ctx: Ctx) -> Any:
    p = report_or_packets.packets if isinstance(report_or_packets, Report) else report_or_packets
    return p.valid[None, :] & (p.dst[None, :] == ctx.unit[:, None])


def packet_dst_cls(p: Packets, units: Units) -> Any:
    return units.cls[jnp.clip(p.dst, 0, units.cls.shape[0] - 1)]


def packet_src_cls(p: Packets, units: Units) -> Any:
    return units.cls[jnp.clip(p.src, 0, units.cls.shape[0] - 1)]


def is_rune_packet(p: Packets) -> Any:
    return p.item < 0


def first_instance(p: Packets, sel: Any) -> Any:
    """(C, P) ``sel`` restricted to the first packet of each cast instance per
    (holder row, packet dst); ``cast_id`` 0 makes every packet its own instance."""
    P = p.valid.shape[0]
    idx = jnp.arange(P)
    # Instance groups: sort by (cast_id, src, dst); a group starts where the key changes.
    order = jnp.lexsort((idx, p.dst, p.src, p.cast_id))
    ks = (p.cast_id[order], p.src[order], p.dst[order])
    start = jnp.concatenate([jnp.ones((1,), bool), (ks[0][1:] != ks[0][:-1]) | (ks[1][1:] != ks[1][:-1])
                             | (ks[2][1:] != ks[2][:-1])])
    group = jnp.zeros((P,), jnp.int32).at[order].set(jnp.cumsum(start, dtype=jnp.int32) - 1)
    # First selected packet of each group, per holder row.
    first = jax.vmap(lambda s: jax.ops.segment_min(jnp.where(s, idx, P), group, num_segments=P))(sel)
    return sel & ((p.cast_id == 0)[None, :] | (first[:, group] == idx[None, :]))


def delay_queue(n_champions: int, k: int):
    """Fixed per-holder delayed-damage slots: (due, dst, raw, dtype, flags) each (C, k)."""
    z = jnp.zeros((n_champions, k), jnp.float32)
    zi = jnp.zeros((n_champions, k), jnp.int32)
    return z + BIG, zi, z, zi, zi


__all__ = [n for n in dir() if not n.startswith("_")] + ["has", "empty_packets"]
