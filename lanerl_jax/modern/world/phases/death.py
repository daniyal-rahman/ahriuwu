"""Phase 9, DEATH: deaths, plates, camp and objective rewards, the economy step, kit takedowns. Writes
``s.obj``. Reads start-of-tick ``s.cc``/``caps`` for the recall interrupt."""
from __future__ import annotations

import jax.numpy as jnp

from ... import champions as K
from ... import economy as E
from ...core import damage as D
from ...core import types as W
from ...items.effects.core import Report
from ...jungle import camps as J
from ...jungle import objectives as OBJ
from ...lane import ai as LA
from ...map import regions as REG
from ..config import N_CHAMPIONS, WorldConfig
from ..scratch import TickScratch
from ..state import ModernOrders, ModernState


def run(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    c, n = N_CHAMPIONS, cfg.n_units
    now, caps, champ, units, st, out, so = sc.now, sc.caps, sc.champ, sc.units, sc.st, sc.out, sc.so
    hp, max_hp, towers, jungle, lane_ai, kits = sc.hp, sc.max_hp, sc.towers, sc.jungle, sc.lane_ai, sc.kits
    level = s.econ.level
    x, y = s.x, s.y                                                   # MOVE's positions
    rp = D.concat_packets(out.report.packets, out.follow_up.packets)
    rr_killed = jnp.concatenate([out.report.resolved.killed, out.follow_up.resolved.killed])
    rr_loss = jnp.concatenate([out.report.resolved.health_loss, out.follow_up.resolved.health_loss])
    killer = jnp.full((n,), -1, jnp.int32).at[jnp.clip(rp.dst, 0, n - 1)].max(
        jnp.where(rp.valid & rr_killed, rp.src, -1).astype(jnp.int32))
    dmg = jnp.zeros((n, n), bool).at[jnp.clip(rp.src, 0, n - 1), jnp.clip(rp.dst, 0, n - 1)].max(rp.valid)
    died = s.alive & (hp <= 0.0)
    death_seen = died[None, :] & s.sight                          # start-of-tick sight, victim alive
    struct = W.is_structure(s.kind)
    towers, plates = LA.structure_damage_events(towers, s.hp, jnp.where(struct, hp, s.hp), now=now)
    minion_died = died & (s.kind == W.KIND_MINION)
    last_hitter = jnp.where(killer < c, killer, -1)
    md = E.MinionDeaths(valid=minion_died, x=s.x, y=s.y, team=s.team, gold=s.bounty_gold, xp=s.bounty_xp,
                        level=s.bounty_level, last_hitter=last_hitter, unit=jnp.arange(n, dtype=jnp.int32))
    sv = plates.plates > 0
    sev = E.StructureEvents(valid=sv | plates.destroyed, unit=jnp.arange(n, dtype=jnp.int32), x=s.x, y=s.y,
                            team=s.team, local_gold=plates.plate_gold + plates.first_turret_gold,
                            global_gold=plates.global_gold, is_turret=plates.destroyed & (s.kind == W.KIND_TURRET),
                            in_top_lane=cfg.unit_lane == 2, is_structure=struct)
    took_health = jnp.zeros((n,), bool).at[jnp.clip(rp.dst, 0, n - 1)].max(rp.valid & (rr_loss > 0))
    in_f = E.in_fountain(x[:c], y[:c], cfg.fountain[s.team[:c], 0], cfg.fountain[s.team[:c], 1])
    dec = E.decimal_level(s.econ.xp, jnp.where(s.econ.quest.complete, E.QUEST_LEVEL_CAP, E.LEVEL_CAP))
    xg = jnp.zeros((c,), jnp.float32)
    gg = jnp.zeros((c,), jnp.float32)
    epic = jnp.zeros((c,), jnp.float32)
    large = jnp.zeros((c,), jnp.float32)
    killed_mon = jnp.zeros((c, n), bool)
    jrw = None
    if cfg.jungle is not None:
        jungle, jrw = J.death_step(jungle, cfg.jungle, units, now=now, died=died, killer=killer,
                                   avg_level=jnp.mean(dec), champion_level=dec, hp=hp[:c], max_hp=max_hp[:c],
                                   mana=champ.mana, max_mana=st.max_mana, champion_died=died[:c],
                                   champion_killer=jnp.where(killer[:c] < c, killer[:c], -1))
        gg, xg = gg + jrw.gold, xg + jrw.xp
        large = large + jrw.large_kills
        killed_mon = killed_mon.at[:, cfg.jungle.slots].set(jrw.killed)
    if so is not None:
        obj, orw = OBJ.objectives_after_damage(s.obj, cfg.objectives, units, rp, rr_loss, died=died, killer=killer,
                                               hp_after=hp, now=now, levels=level, champ=sc.cinfo)
        s = s._replace(obj=obj)
        gg, xg = gg + orw.gold, xg + orw.xp
        epic, large = epic + orw.epic_takedown, large + orw.large_monster_kill
        # Epic takedowns credit the killing team's champions (exact with one champion per team).
        esl = cfg.objectives.slots
        o_td = orw.killed[None, :] & (orw.killer_team[None, :] == s.team[:c][:, None])
        killed_mon = killed_mon.at[:, esl].set(killed_mon[:, esl] | o_td)
    if cfg.regions is not None:
        in_quest = REG.in_quest_lane(x[:c], y[:c], 2, cfg.regions)
        reached, in_jg = REG.homeguard_flags(x[:c], y[:c], s.team[:c], now, units, cfg.unit_lane, lane_ai.lane,
                                             cfg.regions)
    else:
        in_quest, reached, in_jg = ~in_f, jnp.zeros((c,), bool), jnp.zeros((c,), bool)
    recall_ch = None if so is None else jnp.where(so.empowered_recall, 4.0, E.RECALL_CHANNEL)
    pet_gold, pet_xp = (None, None) if cfg.jungle is None else \
        J.minion_reward_mods(jungle, now=now, champion_level=dec, avg_level=jnp.mean(dec))
    einp = E.EconomyInputs(
        now=now, unit=jnp.arange(c, dtype=jnp.int32), x=x[:c], y=y[:c], team=s.team[:c], hp=hp[:c],
        max_hp=max_hp[:c], report=Report(rp, None, None), cc=sc.cc_items,
        final_blow=jnp.where(killer[:c] < c, killer[:c], -1), minion_deaths=md,
        minion_in_lane=LA.minion_in_lane(lane_ai, units, 2), structures=sev,
        last_champion_combat=out.state.clocks.last_champion_combat, in_fountain=in_f,
        in_quest_lane=in_quest, recall_request=orders.recall,
        cancel_action=orders.move | (orders.attack >= 0) | (orders.cast_slot >= 0) | (orders.summoner_slot >= 0),
        health_damage=took_health[:c],
        disabled=caps["stunned"][:c] | caps["silenced"][:c] | (s.cc.root_until[:c] > now),
        reached_endpoint=reached, in_jungle=in_jg, teleported=sc.s_out.teleport_arrive,
        extra_gold=gg, extra_xp=xg, epic=epic, recall_channel=recall_ch, minion_gold_delta=pet_gold,
        minion_xp_mult=pet_xp)
    eco = E.economy_step(s.econ, einp)
    econ = eco.state
    gold_extra = out.effects.gold + sc.s_eff.gold
    econ = econ._replace(gold=jnp.minimum(econ.gold + gold_extra, 100000.0), gold_total=econ.gold_total + gold_extra)
    eco = eco._replace(kills=eco.kills._replace(killed_units=eco.kills.killed_units | killed_mon))
    if jrw is not None:                                               # camp-kill restores
        hp = hp.at[:c].set(jnp.minimum(hp[:c] + jrw.heal, max_hp[:c]))
        champ = champ._replace(mana=jnp.minimum(champ.mana + jrw.mana, st.max_mana))
    kits = K.on_takedown(kits, sc.kctx, units, eco.kills)
    return s, sc._replace(dmg=dmg, died=died, death_seen=death_seen, towers=towers, plates=plates,
                          minion_died=minion_died, took_health=took_health, in_f=in_f, epic=epic, large=large,
                          jrw=jrw, jungle=jungle, eco=eco, econ=econ, hp=hp, champ=champ, kits=kits)
