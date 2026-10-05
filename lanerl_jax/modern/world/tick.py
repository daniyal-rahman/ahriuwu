"""Patch-26.19 modern world tick: every modern system composed in one step.

``step(state, orders, cfg)`` advances the modern Summoner's Rift world (all three lanes,
jungle, objectives, fog) by ``cfg.dt`` (30 Hz default, DAMAGE_AND_STATS §1.3). Subsystems
are pure: they read views (``world/views.py``: ``units_view``, ``KitCtx``, item ``Ctx``) and
return packets, CC, effects and slot writes; only the phases below write ``ModernState``.

Tick order: ``step`` runs one phase module per row (``world/phases/<name>.py``, each with
``run``), passing a ``TickScratch`` (``world/scratch.py``: the tick's data-flow map;
docs/modern/WORLD_IMPLEMENTATION.md has the full table and the one-tick lags):
  1. inputs      fog-filtered orders, walk-in/buffered casts, attack-move, skill points, spawns, terrain
  2. stats       static stats (items, shards, monster buffs, kit); STAT.70 max-HP sync
  2b. objectives epic monsters: spawns, abilities, Rift transformation
  3. casts       shop; kit casts and periodic effects; summoner spells; Smite
  4. ai          turret/minion/monster targets; champion orders, idle acquisition, attack-move
  5. move        route movement, dashes, Flash, Teleport; collision
  6. attack      attack machine, crits, attack packets, missiles, kit on-attack/on-hit
  7. damage      every packet -> combat_tick (items, runes, damage pipeline)
  8. cc_heal     heals, shields and CC (tenacity, slow resist)
  9. death       kills, plates, camp/objective rewards -> economy_step
 10. timers      cooldowns, mana/HP regen, fountain, inventory outputs, wards, terrain ejection
 11. fog         attack reveal, then next tick's visibility
     commit      next state and TickEvents; ``step`` then applies the game-over freeze and dtypes
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import mechanics as M
from ..lane import ai as LA
from .config import N_CHAMPIONS, WorldConfig
from .phases import (
    ai,
    attack,
    casts,
    cc_heal,
    damage,
    death,
    fog,
    inputs,
    move,
    objectives,
    stats,
    timers,
)
from .scratch import TickScratch
from .state import ModernOrders, ModernState, TickEvents

PHASES = (stats, objectives, casts, ai, move, attack, damage, cc_heal, death, timers, fog)   # after inputs


def step(s: ModernState, orders: ModernOrders, cfg: WorldConfig) -> tuple[ModernState, TickEvents]:
    """Advance the modern world one tick (``cfg.dt``); returns ``(state, events)``."""
    s0 = s
    now = s.t + jnp.float32(cfg.dt)
    key, k_crit = jax.random.split(s.key)
    sc = TickScratch(s0=s0, now=now, key=key, k_crit=k_crit)
    s, orders, sc = inputs.run(s, orders, cfg, sc)
    for phase in PHASES:
        s, sc = phase.run(s, orders, cfg, sc)
    new, events = commit(s, cfg, sc)
    # Game over (Nexus destroyed): the world freezes on the final state.
    new = jax.tree.map(lambda a, b: jnp.where(s0.game_over, a, b), s0, new)
    # Keep the carry stable under scan: subsystems may return wider/narrower dtypes.
    return jax.tree.map(lambda a, b: jnp.asarray(b, a.dtype) if hasattr(a, "dtype") else b, s0, new), events


def commit(s: ModernState, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickEvents]:
    """The next state (before the game-over freeze and dtype cast) and the tick's events."""
    c = N_CHAMPIONS
    st, out, cc, died, alive, hp = sc.st, sc.out, sc.cc, sc.died, sc.alive, sc.hp
    cc = cc._replace(**{f: jnp.where(died, 0.0, getattr(cc, f)) for f in M.CCTimers._fields})
    events = TickEvents(out.report, out.follow_up, sc.eco, sc.plates, sc.launched, out.packet_overflow, sc.m_over,
                        sc.shop_code)
    result = LA.game_result(sc.towers)
    # Champion rows of the unit columns mirror this tick's stats (read through WorldUnits).
    champ_cols = dict(ad=s.ad.at[:c].set(st.base_ad + st.bonus_ad),
                      armor=s.armor.at[:c].set(st.base_armor + st.bonus_armor),
                      mr=s.mr.at[:c].set(st.base_mr + st.bonus_mr), arange=s.arange.at[:c].set(sc.reach),
                      aspeed=s.aspeed.at[:c].set(st.attack_speed), mspeed=s.mspeed.at[:c].set(sc.ms[:c]))
    new = s._replace(**champ_cols,
        t=sc.now, tick=s.tick + 1, key=sc.key, kind=sc.kind, alive=alive, x=sc.x, y=sc.y,
        hp=jnp.where(alive, hp, jnp.minimum(hp, 0.0)),
        max_hp=sc.max_hp, att=sc.att, missiles=sc.missiles, cc=cc, champ=sc.champ, kits=sc.kits, summoners=sc.summ,
        combat=out.state, econ=sc.econ, lane_ai=sc.lane_ai, towers=sc.towers, shields=sc.shields, status=sc.status,
        kills=sc.kills, damage_matrix=sc.dmg, death_seen=sc.death_seen, visible=sc.visible, sight=sc.sight,
        reveal=sc.reveal, sub=sc.sub, team=sc.team, spawn_seq=sc.spawn_seq, radius=sc.radius,
        targetable=sc.targetable, jungle=sc.jungle, wards=sc.wards, amove=sc.amove, pending_dash=sc.item_cleanse.dash,
        epic_prev=sc.epic, large_prev=sc.large, game_over=result.over, winner=result.winner)
    return new, events
