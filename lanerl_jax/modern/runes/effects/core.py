"""Rune-effect contract (RUNES.md §8–9): events, outputs, combat clocks and shared kernels.

Rune kernels follow the item-effect contract (``items.effects.core``: pure fixed-shape JAX, (C,) per holder,
(N,) per unit, holder c is unit ``ctx.unit[c]``) and reuse its types, with two differences: ownership is
the (C, R) page-count matrix ``page`` (``has_rune(page, id)``), and every hook also gets ``ev: RuneEvents``.

Module protocol (all optional except ``COVERAGE``/``State``/``init``):

    COVERAGE: dict[int, str]; State: NamedTuple; init(n_champions, n_units) -> State
    stats(state, page, ctx, ev) -> ItemStats                    STAT.20–50 (adaptive force unresolved)
    debuffs(state, page, ctx, units, ev) -> Debuffs
    packet_amp / packet_block(state, page, ctx, units, ev, packets) -> (P,)    DMG.40 amp / DMG.70 block
    heal_mult(state, page, ctx, ev) -> (C,)                     x on heals/shields the holder receives
    on_cast / on_attack / on_hit / on_cc / periodic / on_damage / on_takedown
        (state, page, ctx, units, ev) -> (state, Effects)
    post_tick(state, page, ctx, units, ev) -> state             after heals/shields are applied
    outputs(state, page, ctx, ev) -> RuneOutputs

Rune packets carry ``item = -perk_id`` and ``TAG_PROC``. Pet damage is emitted with ``src`` = the owner's
unit and ``TAG_PET`` (packets have no owner field; U-02).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from ...core.damage import CLASS_CHAMPION, MAGIC, PHYSICAL, Packets

# Rune modules import their whole toolkit from here.
from ...items.effects.core import (BIG, CC, Attack, Cast, Ctx, Effects, Kills, Units, effects, in_circle,  # noqa: F401
                                   onehot_units, target_class, unit_pos)
from ..catalog import breakpoints, ea, has_rune, level_table, lin, lin_growth, rune_catalog  # noqa: F401

COMBAT_TIMEOUT = 5.0        # out of combat 5 s after the last event (RUNES §1.5)
CHAMPION_COMBAT_GAP = 10.0  # First Strike OOCTimer: a new episode needs 10 s without champion combat


def rune_item(perk_id: int) -> int:
    """Packet provenance value of rune damage."""
    return -int(perk_id)


class CombatClocks(NamedTuple):
    """Runtime combat timers (RUNES §1.5), (C,) seconds, -BIG = never."""
    last_combat: Any             # dealt/took damage (incl. 0) to/from an enemy champion, minion, monster or turret
    last_champion_combat: Any    # same, enemy champions only (incl. pets), plus CC dealt to one
    last_hit_by_champion: Any    # health damage > 0 taken from an enemy champion
    champion_combat_start: Any   # start of the episode (a gap of CHAMPION_COMBAT_GAP starts a new one)
    struck_first: Any            # the holder dealt the first damage of the episode
    last_combat_modern: Any      # "modern" system: also non-damage CC and hits on invulnerable targets


def init_clocks(n_champions: int) -> CombatClocks:
    z = jnp.full((n_champions,), -BIG, jnp.float32)
    return CombatClocks(z, z, z, z, jnp.zeros((n_champions,), bool), z)


class RuneEvents(NamedTuple):
    """World events and runtime context of one tick: (C,) per holder unless noted, (C, N) holder x unit."""
    game_time: Any                  # () seconds since game start
    attack: Attack                  # basic attack launched/landed this tick
    attack_started: Any             # windup started this tick (Hail of Blades trigger)
    attack_start_target: Any        # int32 unit of that windup
    attack_cancelled: Any           # windup cancelled this tick
    attack_reset: Any               # a Trait_AttackReset effect fired
    cast: Cast                      # ability cast started this tick
    cast_id: Any                    # int32 instance id of that cast (matches its packets' cast_id)
    cc: CC                          # (C, N) holder applied slow / immobilize this tick
    cc_duration: Any                # (C, N) immobilize duration after tenacity
    impaired: Any                   # (N,) impaired at tick start (Cheap Shot)
    movement_impaired: Any          # (N,) immobilized, grounded or slowed (Approach Velocity)
    impaired_by_holder: Any         # (C, N) holder's own active movement impairment on the unit
    holder_cc_from_champion: Any    # holder is under non-kinematic CC from an enemy champion (Unflinching)
    summoner_cast: Any              # a summoner spell completed its cast/channel this tick
    summoner_cooldown: Any          # hasted cooldown (s) of that spell
    summoner_is_teleport: Any
    blinked: Any                    # dash, blink, Flash, TP arrival, recall, stealth exit (Sudden Impact)
    flash_cooldown: Any             # remaining Flash cooldown
    hexflash_request: Any           # int32 0 none, 1 start channel, 2 release
    kills: Kills
    deaths: Any                     # (N,) units that died this tick (any killer)
    sight: Any                      # (C, N) holder has direct line of sight to the unit
    visible: Any                    # (C, N) unit visible to the holder's team
    large_monster_kill: Any
    epic_takedown: Any
    execute_credit: Any             # champion kills credited to the holder but dealt by non-champions
    shield_gained: Any              # largest shield granted to the holder this tick
    shield_gained_duration: Any     # its duration (s)
    summoner_haste: Any
    cc_cast_id: Any                 # (C, N) int32 cast instance of the CC in ``cc`` (0 = unknown)
    cc_on_hit: Any                  # (C, N) CC in ``cc`` was applied on-hit
    purchased: Any                  # int32 item id bought this tick (0 none)
    sold: Any                       # int32 item id sold this tick (0 none)
    potion_drunk: Any               # int32 item id of a potion started this tick (0 none)
    granted: Any                    # int32 item id the world placed for this rune (0 none)
    spellbook_request: Any          # int32 summoner spell id to swap to (0 none)
    is_turret: Any                  # (N,)
    in_river: Any
    uses_energy: Any
    adaptive_physical: Any          # champion adaptive type for ties
    own: Any                        # (C, I) item counts
    bonus_ad: Any                   # post-STAT.50 bonus AD, AP and bonus AS (rune formulas read these)
    ap: Any
    bonus_attack_speed: Any
    clocks: CombatClocks
    report: Any = None              # resolved Report inside on_damage, else None


def rune_events(ctx: Ctx, n_units: int, **kw) -> RuneEvents:
    """Quiet events for ``ctx``; override any field by keyword."""
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
        large_monster_kill=zc, epic_takedown=zc, execute_credit=zc, shield_gained=zc, shield_gained_duration=zc,
        summoner_haste=zc, cc_cast_id=jnp.zeros((c, n_units), jnp.int32), cc_on_hit=zcn, purchased=ic, sold=ic,
        potion_drunk=ic, granted=ic, spellbook_request=ic, is_turret=jnp.zeros((n_units,), bool), in_river=fc,
        uses_energy=fc, adaptive_physical=~fc, own=None,
        bonus_ad=ctx.bonus_ad, ap=ctx.ap, bonus_attack_speed=ctx.bonus_attack_speed,
        clocks=init_clocks(c), report=None)
    base.update(kw)
    return RuneEvents(**base)


class RuneOutputs(NamedTuple):
    """World-applied rune results, (C,) unless noted."""
    grant_item: Any                 # int32 item id to add; repeated each tick until acknowledged via ``ev.granted``
    forbid_purchase: Any            # (C, I) purchases the rune blocks
    skill_points: Any               # int32 extra skill points this tick
    basic_cd_refund: Any            # fraction of current Q/W/E cooldowns removed now
    ult_cd_refund: Any              # fraction of current R cooldown removed now
    move_locked: Any                # MS set to 0
    blink: Any
    blink_range: Any
    spellbook_swap_ready: Any
    first_strike_gold: Any          # gold awarded this tick (also in Effects.gold)
    ghosted: Any                    # ignores unit collision


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


# ---- shared kernels -------------------------------------------------------------

def by_range(ctx: Ctx, melee: Any, ranged: Any) -> Any:
    return jnp.where(ctx.is_ranged, ranged, melee)


def adaptive_damage_type(ev: RuneEvents) -> Any:
    """Physical if bonus AD > AP, magic if AP > bonus AD, else the champion's adaptive type (RUNES §1.2)."""
    return jnp.where(ev.bonus_ad > ev.ap, PHYSICAL,
                     jnp.where(ev.ap > ev.bonus_ad, MAGIC, jnp.where(ev.adaptive_physical, PHYSICAL, MAGIC)))


def variable_damage_type(ad_term: Any, ap_term: Any) -> Any:
    """Electrocute/Comet: the larger ratio term's type; a tie or zero is magic (RUNES §1.2)."""
    return jnp.where(ad_term > ap_term, PHYSICAL, MAGIC)


def enemy_champions(ctx: Ctx, units: Units) -> Any:
    """(C, N) living enemy champion units."""
    return (units.cls[None, :] == CLASS_CHAMPION) & (units.team[None, :] != ctx.team[:, None]) \
        & units.alive[None, :]


def packet_src_cls(p: Packets, units: Units) -> Any:
    return units.cls[jnp.clip(p.src, 0, units.cls.shape[0] - 1)]


def first_instance(p: Packets, sel: Any) -> Any:
    """(C, P) ``sel`` restricted to the first packet of each (cast_id, src, dst) instance per holder row;
    ``cast_id`` 0 makes every packet its own instance."""
    P = p.valid.shape[0]
    idx = jnp.arange(P)
    order = jnp.lexsort((idx, p.dst, p.src, p.cast_id))
    ks = (p.cast_id[order], p.src[order], p.dst[order])
    start = jnp.concatenate([jnp.ones((1,), bool), (ks[0][1:] != ks[0][:-1]) | (ks[1][1:] != ks[1][:-1])
                             | (ks[2][1:] != ks[2][:-1])])
    group = jnp.zeros((P,), jnp.int32).at[order].set(jnp.cumsum(start, dtype=jnp.int32) - 1)
    first = jax.vmap(lambda s: jax.ops.segment_min(jnp.where(s, idx, P), group, num_segments=P))(sel)
    return sel & ((p.cast_id == 0)[None, :] | (first[:, group] == idx[None, :]))
