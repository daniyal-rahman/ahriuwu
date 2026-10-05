"""Glue between item effects and the damage pipeline for one tick.

The world integrator owns ordering (TICK.*); these helpers do the item-side
folding so every caller applies item contributions the same way:

* ``fold_defense`` / ``fold_offense``: holder defense, target debuffs and item
  penetration stats into ``core.damage.Defense``/``Offense``.
* ``apply_dealt_amp``: per-(holder, target) DMG.40 amps onto packets.
* ``resolve_tick``: resolve packets, compute vamp, return a ``Report``.
* ``apply_effects``: heals (HEAL.10–40), shields (SHIELD.*), slows, Grievous
  Wounds, mana and gold from merged ``Effects``.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core import damage as D
from ..catalog import ItemStats
from .core import Attack, CC, Cast, Ctx, Debuffs, Effects, HolderDefense, Kills, Report, Units, merge_effects


class UnitStatus(NamedTuple):
    """Per-unit timed statuses written by item effects, shape (N,).

    ``slow``/``slow_until`` are bookkeeping for item-only callers; the world tick (``world.tick``)
    routes ``Effects.slow`` into ``mechanics`` CC timers, the single slow state movement reads.
    """
    slow: Any
    slow_until: Any
    grievous_until: Any


def init_status(n_units: int) -> UnitStatus:
    z = jnp.zeros((n_units,), jnp.float32)
    return UnitStatus(z, z, z)


def fold_defense(base: D.Defense, ctx: Ctx, holder: HolderDefense, debuffs: Debuffs, *,
                 shield_power: Any = 0.0, incoming_heal: Any = 0.0) -> D.Defense:
    """Place holder rows at ``ctx.unit`` and add target-side debuffs everywhere.

    Lifeline shields are item base values; heal-and-shield power and the
    holder's incoming heal/shield bonus (Spirit Visage) scale them here,
    exactly once (SHIELD.10/20).
    """
    u = ctx.unit
    champ_mult = 1.0 if holder.champion_received_mult is None else holder.champion_received_mult

    def put(arr, rows):
        return arr.at[u].set(jnp.asarray(rows, arr.dtype))

    d = base._replace(
        received_mult=base.received_mult.at[u].multiply(holder.received_mult),
        basic_attack_mult=base.basic_attack_mult.at[u].multiply(holder.basic_attack_mult),
        crit_taken_mult=base.crit_taken_mult.at[u].multiply(holder.crit_taken_mult),
        champion_attack_block=base.champion_attack_block.at[u].add(holder.champion_attack_block),
        postmit_flat=base.postmit_flat.at[u].add(holder.postmit_flat),
        store_fraction=base.store_fraction.at[u].add(holder.store_fraction),
        lifeline_ready=put(base.lifeline_ready, holder.lifeline_ready),
        lifeline_magic_only=put(base.lifeline_magic_only, holder.lifeline_magic_only),
        lifeline_shield=put(base.lifeline_shield,
                            holder.lifeline_shield * (1.0 + shield_power) * (1.0 + incoming_heal)),
        champion_received_mult=base.champion_received_mult.at[u].multiply(champ_mult),
        lifeline_shield_kind=put(base.lifeline_shield_kind, holder.lifeline_shield_kind),
        lifeline_duration=put(base.lifeline_duration, holder.lifeline_duration),
        lifeline_decay_hold=put(base.lifeline_decay_hold, holder.lifeline_decay_hold),
        lifeline_bonus_health=put(base.lifeline_bonus_health, holder.lifeline_bonus_health),
        spell_shield=base.spell_shield.at[u].set(base.spell_shield[u] | holder.spell_shield))
    return d._replace(
        percent_armor_reduction=1 - (1 - d.percent_armor_reduction) * (1 - debuffs.percent_armor_reduction),
        flat_armor_reduction=d.flat_armor_reduction + debuffs.flat_armor_reduction,
        percent_mr_reduction=1 - (1 - d.percent_mr_reduction) * (1 - debuffs.percent_mr_reduction),
        flat_mr_reduction=d.flat_mr_reduction + debuffs.flat_mr_reduction,
        received_amp=d.received_amp + debuffs.received_amp,
        magic_received_amp=d.magic_received_amp + debuffs.magic_received_amp)


def fold_offense(base: D.Offense, ctx: Ctx, stats: ItemStats) -> D.Offense:
    """Item penetration (static + dynamic ``ItemStats`` of each holder)."""
    u = ctx.unit
    combine = lambda arr, pct: arr.at[u].set(1 - (1 - arr[u]) * (1 - pct))
    return base._replace(
        lethality=base.lethality.at[u].add(stats.lethality),
        percent_armor_pen=combine(base.percent_armor_pen, stats.percent_armor_pen),
        magic_pen=base.magic_pen.at[u].add(stats.magic_pen),
        percent_magic_pen=combine(base.percent_magic_pen, stats.percent_magic_pen))


def apply_dealt_amp(p: D.Packets, ctx: Ctx, amp: Any) -> D.Packets:
    """Add (C, N) holder->target amps to packets sourced by holders (DMG.40)."""
    src_is = p.src[:, None] == ctx.unit[None, :]                   # (P, C)
    per = amp[:, jnp.clip(p.dst, 0, amp.shape[1] - 1)].T           # (P, C)
    extra = jnp.sum(jnp.where(src_is, per, 0.0), axis=1)
    keep = D.has(p.flags, D.PROP_NO_DAMAGE_MOD) | D.has(p.flags, D.TAG_NON_AMPABLE)
    return p._replace(amp=p.amp + jnp.where(keep, 0.0, extra))


def resolve_tick(p: D.Packets, off: D.Offense, dfn: D.Defense, hp: Any, max_hp: Any,
                 shields: D.Shields, now: Any, vamp: D.Vamp, *, lifesteal_scale: Any = None) -> Report:
    res = D.resolve(p, off, dfn, hp, max_hp, shields, now)
    ls, ov = D.vamp_heal_split(p, res, vamp, dfn.unit_class, lifesteal_scale=lifesteal_scale)
    return Report(p, res, ls, ov)


def apply_effects(eff: Effects, ctx: Ctx, hp: Any, max_hp: Any, shields: D.Shields,
                  status: UnitStatus, *, heal_power: Any, incoming_heal: Any, vamp_heal: Any = None,
                  shield_power: Any = None, heal_mult: Any = 1.0) -> tuple[Any, D.Shields, UnitStatus]:
    """Apply heals/shields to holders and slows/GW to units.

    ``heal_power``/``incoming_heal`` are the holders' (C,) heal-and-shield
    power and incoming heal bonus (Spirit Visage). ``vamp_heal`` (N,) from the
    Report is healed here too (no HSP, incoming bonus and GW apply).
    ``heal_mult`` (C,) is a separate multiplier on HSP-type heals and on
    shields the holder receives (Revitalize ×1.10 below 40% HP).
    """
    u = ctx.unit
    now = ctx.now
    gw = status.grievous_until[u] > now
    alive = hp[u] > 0.0
    total = (D.heal_amount(eff.heal, source_power=heal_power, incoming=incoming_heal, grievous=gw) * heal_mult
             + D.heal_amount(eff.heal_plain, incoming=incoming_heal, grievous=gw))
    if vamp_heal is not None:
        total = total + D.heal_amount(vamp_heal[u], incoming=incoming_heal, grievous=gw)
    hp = hp.at[u].set(D.apply_heal(hp[u], max_hp[u], total, alive).astype(hp.dtype))
    sp = heal_power if shield_power is None else shield_power
    for k in range(eff.shields.amount.shape[1]):
        amt = eff.shields.amount[:, k] * (1.0 + sp) * (1.0 + incoming_heal) * heal_mult
        for c in range(u.shape[0]):
            shields = D.grant_shield(shields, u[c], amt[c], eff.shields.kind[c, k], now,
                                     eff.shields.duration[c, k], decay_hold=eff.shields.decay_hold[c, k],
                                     enabled=amt[c] > 0.0)
    active_slow = jnp.where(status.slow_until > now, status.slow, 0.0)
    stronger = eff.slow >= active_slow
    new_slow = jnp.where((eff.slow > 0.0) & stronger, eff.slow, status.slow)
    new_until = jnp.where((eff.slow > 0.0) & stronger, jnp.maximum(now + eff.slow_duration,
                          jnp.where(eff.slow == active_slow, status.slow_until, 0.0)), status.slow_until)
    gw_until = jnp.maximum(status.grievous_until, jnp.where(eff.grievous > 0.0, now + eff.grievous, 0.0))
    return hp, shields, UnitStatus(new_slow, new_until, gw_until)


# ---- reference one-tick orchestration --------------------------------------

EXTRA_ON_HIT_SLOTS = 2   # Runaan's bolts (2 ranged) / Statikk secondary bounces get on-hit re-application
MAIN_PACKET_CAPACITY = 512      # valid packets per tick (world + items) after compaction
FOLLOW_UP_CAPACITY = 256


class ItemTickOut(NamedTuple):
    state: Any               # ItemEffectState
    hp: Any                  # (N,)
    max_hp: Any              # (N,)
    shields: D.Shields
    status: UnitStatus
    packet_overflow: Any     # () valid packets dropped by compaction (must stay 0)
    report: Report           # main resolution pass
    follow_up: Report        # second pass for packets emitted by damage triggers
    effects: Effects         # everything merged (gold, mana, revive, attack_reset ...)
    active: Any              # ActiveOut
    dynamic_stats: ItemStats  # STAT.50 contributions used for this tick
    transforms: tuple        # (from_row, to_row, do) per holder (Tear line)
    consume_row: Any         # (C,) catalog row to consume (-1 none)


def item_tick(state, own, ctx: Ctx, units: Units, *, attack: Attack, cast: Cast, request: Any,
              base_packets: D.Packets, base_offense: D.Offense, base_defense: D.Defense,
              hp: Any, max_hp: Any, shields: D.Shields, status: UnitStatus, kills: Kills,
              holder_stats: ItemStats, cc: CC | None = None) -> ItemTickOut:
    """Items-only tick: ``combat.combat_tick`` with an empty rune page.

    Kept for item-level tests and callers that carry only ``ItemEffectState``.
    It does not sync dynamic max HP and drops second-generation trigger
    packets (no state to carry them); the world integration uses
    ``combat_tick`` with a ``CombatState``.
    """
    from ...runes import effects as RE
    from ...combat import CombatState, combat_tick, empty_page
    from ...runes.effects.core import init_clocks
    c, n = ctx.level.shape[0], units.x.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    cs = CombatState(state, RE.init(c, n), init_clocks(c), z, z, D.empty_packets(0))
    out = combat_tick(cs, own, empty_page(c), ctx, units, attack=attack, cast=cast, request=request,
                      base_packets=base_packets, base_offense=base_offense, base_defense=base_defense,
                      hp=hp, max_hp=max_hp, shields=shields, status=status, kills=kills,
                      holder_stats=holder_stats, cc=cc, sync_max_health=False, carry=False)
    return ItemTickOut(out.state.items, out.hp, out.max_hp, out.shields, out.status, out.packet_overflow,
                       out.report, out.follow_up, out.effects, out.active, out.dynamic_stats, out.transforms,
                       out.consume_row)
