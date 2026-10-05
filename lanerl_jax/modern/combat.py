"""One combat tick of every item and rune effect around the world's packets.

This is the reference integration of ``items.effects`` and
``runes.effects`` with the shared pipelines (``core.damage``,
``core.stats``, ``core.stat_pipeline``), in the canonical order of
docs/modern/README.md (hook crosswalk) and RUNES.md §9:

 1. STAT.50: item and rune dynamic stats from the pre-dynamic ``Ctx``;
    adaptive force resolved once from the pre-adaptive bonus AD/AP.
 2. STAT.70: dynamic max-HP changes synced to current HP (gains heal,
    except ``silent_health``; losses clamp).
 3. Action phase: on_cast, on_attack, on_hit (items, item on-hit
    re-applications, then runes), item actives, periodic, on_cc.
 4. Fold defense/debuffs/penetration and the dynamic armor/MR (STAT.40
    multipliers included) into ``Defense``/``Offense``.
 5. Compact (carried + world + emitted packets), DMG.40 amps (items + runes,
    one additive sum), DMG.70 per-packet blocks (Bone Plating), resolve.
 6. Combat clocks, then on_damage (items, then runes).
 7. One follow-up pass for trigger damage; its own trigger packets are
    carried into the next tick (``CARRY_CAPACITY``) instead of being dropped.
 8. on_takedown, heals/shields (HSP, incoming, Revitalize, GW), post_tick,
    shop hooks, rune outputs.

Ctx is the *pre-dynamic* holder context (static base + items + shards); the
returned ``dynamic_stats`` (adaptive force already split into AD/AP) are the
STAT.50 contributions the world adds for its own reads (basic-attack damage,
attack speed, MS, haste). The world must **not** add ``dynamic_stats.health``
to max HP itself: this tick owns that sync and returns ``max_hp``.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from .core import damage as D
from .items.catalog import ItemStats, combine_stats
from .core.stats import resolve_adaptive
from .items.effects.core import Attack, CC, Cast, Ctx, Kills, Units, merge_effects
from .items.effects.runtime import (EXTRA_ON_HIT_SLOTS, FOLLOW_UP_CAPACITY, MAIN_PACKET_CAPACITY, UnitStatus,
                                    apply_effects, fold_defense, fold_offense, resolve_tick)
from .runes.effects.core import (CHAMPION_COMBAT_GAP, CombatClocks, RuneEvents, RuneOutputs, init_clocks,
                                 no_outputs, rune_events)

CARRY_CAPACITY = 64


class CombatState(NamedTuple):
    items: Any                # ItemEffectState
    runes: Any                # RuneEffectState
    clocks: CombatClocks
    dyn_health: Any           # (C,) dynamic max HP currently applied (items + runes, incl. % health)
    dyn_silent: Any           # (C,) part of dyn_health that did not raise current HP
    carry: D.Packets          # second-generation trigger packets for the next tick


def init_combat(n_champions: int, n_units: int) -> CombatState:
    from .items import effects as E
    from .runes import effects as RE
    z = jnp.zeros((n_champions,), jnp.float32)
    return CombatState(E.init(n_champions, n_units), RE.init(n_champions, n_units), init_clocks(n_champions),
                       z, z, D.empty_packets(CARRY_CAPACITY))


class CombatTickOut(NamedTuple):
    state: CombatState
    hp: Any                  # (N,)
    max_hp: Any              # (N,)
    shields: D.Shields
    status: UnitStatus
    packet_overflow: Any     # () valid packets dropped by compaction (must stay 0)
    report: Any              # main resolution pass
    follow_up: Any           # second pass for trigger packets
    effects: Any             # merged Effects (gold, mana, revive, attack_reset ...)
    active: Any              # item ActiveOut
    dynamic_stats: ItemStats  # STAT.50 (AF resolved) items + runes
    transforms: tuple        # (from_row, to_row, do) per holder (Tear line)
    consume_row: Any         # (C,) catalog row to consume (-1 none)
    rune_outputs: RuneOutputs
    events: RuneEvents       # events as seen by rune hooks (clocks, report = follow-up)


def _update_clocks(clocks: CombatClocks, report, ctx: Ctx, units: Units, cc: CC | None) -> CombatClocks:
    """RUNES §1.5 combat systems from this tick's resolved packets and CC."""
    p, r = report.packets, report.resolved
    n = units.x.shape[0]
    cls = units.cls
    scls, dcls = cls[jnp.clip(p.src, 0, n - 1)], cls[jnp.clip(p.dst, 0, n - 1)]
    steam, dteam = units.team[jnp.clip(p.src, 0, n - 1)], units.team[jnp.clip(p.dst, 0, n - 1)]
    combat_cls = lambda k: (k == D.CLASS_CHAMPION) | (k == D.CLASS_MINION) | (k == D.CLASS_MONSTER) \
        | (k == D.CLASS_STRUCTURE)
    enemy = steam != dteam
    out_ = p.valid[None, :] & (p.src[None, :] == ctx.unit[:, None]) & enemy[None, :]
    in_ = p.valid[None, :] & (p.dst[None, :] == ctx.unit[:, None]) & enemy[None, :]
    any_combat = jnp.any((out_ & combat_cls(dcls)[None, :]) | (in_ & combat_cls(scls)[None, :]), axis=1)
    dealt_champ = jnp.any(out_ & (dcls == D.CLASS_CHAMPION)[None, :], axis=1)
    took_champ = jnp.any(in_ & (scls == D.CLASS_CHAMPION)[None, :], axis=1)
    if cc is not None:
        champ_n = (cls == D.CLASS_CHAMPION)[None, :] & (units.team[None, :] != ctx.team[:, None])
        cc_champ = jnp.any((cc.slowed | cc.immobilized) & champ_n, axis=1)
        dealt_champ = dealt_champ | cc_champ
        any_combat = any_combat | jnp.any(cc.slowed | cc.immobilized, axis=1)
    modern = any_combat | jnp.any(out_ | in_, axis=1)      # invulnerable/0-damage hits included
    champ = dealt_champ | took_champ
    hurt = jnp.any(in_ & (scls == D.CLASS_CHAMPION)[None, :] & (r.health_loss > 0.0)[None, :], axis=1)
    now = ctx.now
    new_episode = champ & (now - clocks.last_champion_combat >= CHAMPION_COMBAT_GAP)
    return CombatClocks(
        last_combat=jnp.where(any_combat, now, clocks.last_combat),
        last_champion_combat=jnp.where(champ, now, clocks.last_champion_combat),
        last_hit_by_champion=jnp.where(hurt, now, clocks.last_hit_by_champion),
        champion_combat_start=jnp.where(new_episode, now, clocks.champion_combat_start),
        struck_first=jnp.where(new_episode, dealt_champ & ~took_champ, clocks.struck_first),
        last_combat_modern=jnp.where(modern, now, clocks.last_combat if clocks.last_combat_modern is None
                                     else clocks.last_combat_modern))


def _shield_gained(eff, stats: ItemStats, report, follow_up, ctx: Ctx, dfn: D.Defense, heal_mult, ev):
    """(amount, duration) of the largest shield the holder gained this tick."""
    amt = eff.shields.amount * ((1.0 + stats.heal_shield_power) * (1.0 + stats.incoming_heal) * heal_mult)[:, None]
    pad = lambda a: jnp.concatenate([a, jnp.zeros((a.shape[0], 1), a.dtype)], axis=1)
    k = jnp.argmax(pad(amt), axis=1)
    best = jnp.take_along_axis(pad(amt), k[:, None], axis=1)[:, 0]
    dur = jnp.take_along_axis(pad(eff.shields.duration), k[:, None], axis=1)[:, 0]
    fired = report.resolved.lifeline_fired[ctx.unit] | follow_up.resolved.lifeline_fired[ctx.unit]
    life = jnp.where(fired, dfn.lifeline_shield[ctx.unit], 0.0)
    dur = jnp.where(life > best, dfn.lifeline_duration[ctx.unit], dur)
    best = jnp.maximum(best, life)
    dur = jnp.where(ev.shield_gained > best, ev.shield_gained_duration, dur)
    return jnp.maximum(best, ev.shield_gained), dur


def combat_tick(state: CombatState, own, page, ctx: Ctx, units: Units, *, attack: Attack, cast: Cast,
                request: Any, base_packets: D.Packets, base_offense: D.Offense, base_defense: D.Defense,
                hp: Any, max_hp: Any, shields: D.Shields, status: UnitStatus, kills: Kills,
                holder_stats: ItemStats, cc: CC | None = None, ev: RuneEvents | None = None,
                sync_max_health: bool = True, carry: bool = True) -> CombatTickOut:
    """One tick of all item and rune effects (see module docstring).

    ``page`` is the (C, R) rune page-count matrix (``runes.catalog.page_counts``);
    an all-zero matrix runs no rune behaviour. ``ev`` carries the world
    events runes read; its attack/cast/cc/kills/own fields are overwritten
    with the arguments of this call so items and runes see the same tick.
    """
    from .items import effects as E
    from .runes import effects as RE
    from .items.effects import marksman, spellblade, starters

    c, n = ctx.level.shape[0], units.x.shape[0]
    items, runes = state.items, state.runes
    cc_ev = cc if cc is not None else CC(jnp.zeros((c, n), bool), jnp.zeros((c, n), bool))
    ev = rune_events(ctx, n) if ev is None else ev
    ev = ev._replace(attack=attack, cast=cast, cc=cc_ev, kills=kills, own=own, clocks=state.clocks, report=None)

    # 1. STAT.50 dynamic stats; adaptive force split once (pre-adaptive comparison).
    dyn = combine_stats(E.dynamic_stats(items, own, ctx), RE.stats(runes, page, ctx, ev))
    af = dyn.adaptive_force + holder_stats.adaptive_force
    af_ad, af_ap = resolve_adaptive(af, ctx.bonus_ad + dyn.attack_damage, ctx.ap + dyn.ability_power,
                                    ev.adaptive_physical, jnp)
    dyn = dyn._replace(attack_damage=dyn.attack_damage + af_ad, ability_power=dyn.ability_power + af_ap,
                       adaptive_force=jnp.zeros_like(af))
    stats = combine_stats(holder_stats._replace(adaptive_force=jnp.zeros_like(af)), dyn)
    ev = ev._replace(bonus_ad=ctx.bonus_ad + dyn.attack_damage, ap=ctx.ap + dyn.ability_power,
                     bonus_attack_speed=ctx.bonus_attack_speed + dyn.attack_speed,
                     summoner_haste=stats.summoner_haste)

    # 2. STAT.70 dynamic max-HP sync (Grasp, Overgrowth, Bloodline, Biscuits, elixirs ...).
    u = ctx.unit
    static_max = max_hp[u] - state.dyn_health
    target = dyn.health + dyn.percent_health * (static_max + dyn.health)
    if sync_max_health:
        delta = target - state.dyn_health
        heal_part = jnp.maximum(delta - jnp.maximum(dyn.silent_health - state.dyn_silent, 0.0), 0.0)
        new_max = jnp.maximum(max_hp[u] + delta, 1.0)
        hp = hp.at[u].set(jnp.where(hp[u] > 0.0, jnp.clip(hp[u] + heal_part, 0.0, new_max), hp[u]).astype(hp.dtype))
        max_hp = max_hp.at[u].set(new_max.astype(max_hp.dtype))
        ctx = ctx._replace(max_hp=new_max, hp=hp[u])
        units = units._replace(hp=hp, max_hp=max_hp)
    else:
        target = state.dyn_health
    dyn_silent = dyn.silent_health

    # 3. Action phase: items first, then runes, in each hook.
    parts = []
    items, eff = E.on_cast(items, own, ctx, units, cast); parts.append(eff)
    runes, eff = RE.on_cast(runes, page, ctx, units, ev); parts.append(eff)
    items, eff = E.on_attack(items, own, ctx, units, attack); parts.append(eff)
    runes, eff = RE.on_attack(runes, page, ctx, units, ev); parts.append(eff)
    items, eff = E.on_hit(items, own, ctx, units, attack); parts.append(eff)
    again = spellblade.extra_on_hit_attack(items.spellblade)
    phantom = marksman.phantom_hit_due(items.marksman, ctx) & attack.hit
    again = again._replace(hit=again.hit | phantom, target=jnp.where(phantom, attack.target, again.target))
    items, eff = E.on_hit(items, own, ctx, units, again); parts.append(eff)
    extra = marksman.extra_on_hit_targets(items.marksman, ctx)
    order = jnp.argsort(jnp.where(extra, 0, 1), axis=1)
    for k in range(EXTRA_ON_HIT_SLOTS):
        tgt = order[:, k].astype(jnp.int32)
        ok = jnp.take_along_axis(extra, order[:, k:k + 1], axis=1)[:, 0]
        z = jnp.zeros((c,), jnp.float32)
        items, eff = E.on_hit(items, own, ctx, units, Attack(jnp.zeros((c,), bool), ok, tgt, z, jnp.zeros((c,), bool)))
        parts.append(eff)
    runes, eff = RE.on_hit(runes, page, ctx, units, ev); parts.append(eff)
    items, eff, act = E.active(items, own, ctx, units, request); parts.append(eff)
    ev = ev._replace(potion_drunk=jnp.maximum(ev.potion_drunk, items.consumables.drank))
    items, eff = E.periodic(items, own, ctx, units); parts.append(eff)
    runes, eff = RE.periodic(runes, page, ctx, units, ev); parts.append(eff)
    if cc is not None:
        items, eff = E.on_cc(items, own, ctx, units, cc); parts.append(eff)
    runes, eff = RE.on_cc(runes, page, ctx, units, ev); parts.append(eff)
    pre = merge_effects(parts, c, n)

    # 4. Defense / offense profiles for this pass.
    holder = E.holder_defense(items, own, ctx)
    from .items.effects.core import combine_debuffs
    debuffs = combine_debuffs([E.target_debuffs(items, own, ctx, units), RE.debuffs(runes, page, ctx, units, ev)], n)
    dfn = fold_defense(base_defense, ctx, holder, debuffs, shield_power=stats.heal_shield_power,
                       incoming_heal=stats.incoming_heal)
    armor = (dfn.armor[u] + dyn.armor) * (1.0 + dyn.percent_armor)
    mr = (dfn.magic_resist[u] + dyn.magic_resist) * (1.0 + dyn.percent_magic_resist)
    dfn = dfn._replace(armor=dfn.armor.at[u].set(armor.astype(dfn.armor.dtype)),
                       magic_resist=dfn.magic_resist.at[u].set(mr.astype(dfn.magic_resist.dtype)))
    off = fold_offense(base_offense, ctx, stats)
    vamp = D.Vamp(jnp.zeros((n,), jnp.float32).at[u].set(stats.life_steal),
                  jnp.zeros((n,), jnp.float32).at[u].set(stats.omnivamp))

    def prepare(p, live):
        p = p._replace(amp=p.amp + E.packet_amp(items, own, ctx, live, p) + RE.packet_amp(runes, page, ctx, live, ev, p))
        return p._replace(block=p.block + RE.packet_block(runes, page, ctx, live, ev, p))

    # 5. Main resolution.
    packets, overflow = D.compact_packets(D.concat_packets(state.carry, base_packets, pre.packets),
                                          MAIN_PACKET_CAPACITY)
    packets = prepare(packets, units)
    report = resolve_tick(packets, off, dfn, hp, max_hp, shields, ctx.now, vamp)
    hp, max_hp, shields = report.resolved.hp, report.resolved.max_hp, report.resolved.shields
    live = units._replace(hp=hp, max_hp=max_hp, alive=units.alive & (hp > 0.0))
    clocks = _update_clocks(state.clocks, report, ctx, units, cc)
    ctx_d = ctx._replace(hp=hp[u], max_hp=max_hp[u])

    # 6. On-damage triggers.
    ev_d = ev._replace(report=report, clocks=clocks)
    items, eff_i = E.on_damage(items, own, ctx_d, live, report)
    runes, eff_r = RE.on_damage(runes, page, ctx_d, live, ev_d)

    # 7. Follow-up pass; its trigger packets carry into the next tick.
    follow, overflow2 = D.compact_packets(D.concat_packets(eff_i.packets, eff_r.packets), FOLLOW_UP_CAPACITY)
    follow = prepare(follow, live)
    follow_up = resolve_tick(follow, off, dfn, hp, max_hp, shields, ctx.now, vamp)
    hp, max_hp, shields = follow_up.resolved.hp, follow_up.resolved.max_hp, follow_up.resolved.shields
    live = live._replace(hp=hp, max_hp=max_hp, alive=live.alive & (hp > 0.0))
    clocks = _update_clocks(clocks, follow_up, ctx, units, None)
    ctx_d = ctx._replace(hp=hp[u], max_hp=max_hp[u])
    ev_f = ev._replace(report=follow_up, clocks=clocks)
    items, eff_i2 = E.on_damage(items, own, ctx_d, live, follow_up)
    runes, eff_r2 = RE.on_damage(runes, page, ctx_d, live, ev_f)
    second = D.concat_packets(eff_i2.packets, eff_r2.packets)
    if carry:
        carried, overflow3 = D.compact_packets(second, CARRY_CAPACITY)
    else:
        carried, overflow3 = D.empty_packets(CARRY_CAPACITY), jnp.int32(0)

    # 8. Takedowns, heals/shields, end of tick.
    ev_t = ev_f._replace(report=None)
    items, eff_t = E.on_takedown(items, own, ctx_d, live, kills)
    runes, eff_rt = RE.on_takedown(runes, page, ctx_d, live, ev_t)
    strip = lambda e: e._replace(packets=D.empty_packets(0))
    total = merge_effects([pre] + [strip(e) for e in (eff_i, eff_r, eff_i2, eff_r2, eff_t, eff_rt)], c, n)
    total = total._replace(packets=D.empty_packets(0))
    vamp_heal = report.life_steal_heal + report.omnivamp_heal + follow_up.life_steal_heal + follow_up.omnivamp_heal
    mult = RE.heal_mult(runes, page, ctx_d, ev_t)
    hp, shields, status = apply_effects(total, ctx, hp, max_hp, shields, status,
                                        heal_power=stats.heal_shield_power, incoming_heal=stats.incoming_heal,
                                        vamp_heal=vamp_heal, heal_mult=mult)
    gained, gained_for = _shield_gained(total, stats, report, follow_up, ctx, dfn, mult, ev)
    ev_p = ev_t._replace(shield_gained=gained, shield_gained_duration=gained_for)
    runes = RE.post_tick(runes, page, ctx._replace(hp=hp[u], max_hp=max_hp[u]), live._replace(hp=hp), ev_p)
    items = E.on_shop(items, own, ctx)
    outs = RE.outputs(runes, page, ctx, ev_p)
    new_state = CombatState(items, runes, clocks, target, jnp.where(sync_max_health, dyn_silent, state.dyn_silent),
                            carried)
    return CombatTickOut(new_state, hp, max_hp, shields, status, overflow + overflow2 + overflow3, report, follow_up,
                         total, act, dyn, starters.pending_transforms(items.starters, own),
                         items.consumables.consume_row, outs, ev_p)


def empty_page(n_champions: int) -> Any:
    from .runes.catalog import rune_catalog
    return jnp.zeros((n_champions, len(rune_catalog().ids)), jnp.int32)
