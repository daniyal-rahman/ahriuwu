"""Item-side folding into the damage pipeline for one tick (the world owns the TICK.* order).

``fold_defense``/``fold_offense`` add holder profiles, target debuffs and item penetration to
``core.damage.Defense``/``Offense``; ``resolve_tick`` resolves packets with vamp; ``apply_effects`` applies heals
(HEAL.10-40), shields (SHIELD.*), slows and Grievous Wounds from merged ``Effects``.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core import damage as D
from ..catalog import ItemStats
from .core import Ctx, Debuffs, Effects, HolderDefense, Report

EXTRA_ON_HIT_SLOTS = 2          # Runaan's bolts / Statikk secondary bounces that re-apply on-hit
MAIN_PACKET_CAPACITY = 512      # valid packets per tick (world + items) after compaction
FOLLOW_UP_CAPACITY = 256


class UnitStatus(NamedTuple):
    """Per-unit timed statuses (N,). The world routes ``Effects.slow`` into mechanics CC timers instead of
    ``slow``/``slow_until``, which only item-level callers read."""
    slow: Any
    slow_until: Any
    grievous_until: Any


def init_status(n_units: int) -> UnitStatus:
    z = jnp.zeros((n_units,), jnp.float32)
    return UnitStatus(z, z, z)


def fold_defense(base: D.Defense, ctx: Ctx, holder: HolderDefense, debuffs: Debuffs, *,
                 shield_power: Any = 0.0, incoming_heal: Any = 0.0) -> D.Defense:
    """Write holder rows at ``ctx.unit`` and add target debuffs everywhere.

    Lifeline shields are base values; heal-and-shield power and incoming heal scale them here, once (SHIELD.10/20).
    """
    u = ctx.unit

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
        champion_received_mult=base.champion_received_mult.at[u].multiply(holder.champion_received_mult),
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
    u = ctx.unit
    combine = lambda arr, pct: arr.at[u].set(1 - (1 - arr[u]) * (1 - pct))
    return base._replace(
        lethality=base.lethality.at[u].add(stats.lethality),
        percent_armor_pen=combine(base.percent_armor_pen, stats.percent_armor_pen),
        magic_pen=base.magic_pen.at[u].add(stats.magic_pen),
        percent_magic_pen=combine(base.percent_magic_pen, stats.percent_magic_pen))


def resolve_tick(p: D.Packets, off: D.Offense, dfn: D.Defense, hp: Any, max_hp: Any,
                 shields: D.Shields, now: Any, vamp: D.Vamp, *, lifesteal_scale: Any = None) -> Report:
    res = D.resolve(p, off, dfn, hp, max_hp, shields, now)
    ls, ov = D.vamp_heal_split(p, res, vamp, dfn.unit_class, lifesteal_scale=lifesteal_scale)
    return Report(p, res, ls, ov)


def apply_effects(eff: Effects, ctx: Ctx, hp: Any, max_hp: Any, shields: D.Shields,
                  status: UnitStatus, *, heal_power: Any, incoming_heal: Any, vamp_heal: Any = None,
                  shield_power: Any = None, heal_mult: Any = 1.0) -> tuple[Any, D.Shields, UnitStatus]:
    """Apply holder heals/shields and unit slows/GW.

    ``vamp_heal`` (N,) gets incoming heal and GW but no HSP; ``heal_mult`` (C,) scales HSP-type heals and
    received shields (Revitalize).
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
