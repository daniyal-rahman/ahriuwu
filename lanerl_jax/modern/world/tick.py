"""``step``: one 30 Hz tick of the 26.19 world (DAMAGE_AND_STATS §1.3).

Runs ``inputs`` then ``PHASES`` (``world/phases/<name>.py``), passing a ``TickScratch``, then ``commit``.
Only the phases write ``ModernState``; subsystems read views and return packets, CC, effects and slot writes.
Phase table and one-tick lags: docs/modern/WORLD_IMPLEMENTATION.md "Tick order".
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import mechanics as M
from ..lane import ai as LA
from .config import N_CHAMPIONS, WorldConfig
from .phases import ai, attack, casts, cc_heal, damage, death, fog, inputs, move, objectives, stats, timers
from .scratch import TickScratch
from .state import LastTick, ModernOrders, ModernState, TickEvents

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
    new = jax.tree.map(lambda a, b: jnp.where(s0.game_over, a, b), s0, new)        # a fallen Nexus freezes the world
    # Cast back to the incoming dtypes so scan carries stay stable.
    return jax.tree.map(lambda a, b: jnp.asarray(b, a.dtype) if hasattr(a, "dtype") else b, s0, new), events


def commit(s: ModernState, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickEvents]:
    """The next state (before the game-over freeze and dtype cast) and the tick's events."""
    c = N_CHAMPIONS
    st, out, cc, died, alive, hp = sc.st, sc.out, sc.cc, sc.died, sc.alive, sc.hp
    cc = cc._replace(**{f: jnp.where(died, 0.0, getattr(cc, f)) for f in M.CCTimers._fields})
    events = TickEvents(out.report, out.follow_up, sc.eco, sc.plates, sc.launched, out.packet_overflow, sc.m_over,
                        sc.ray_over, sc.shop_code)
    result = LA.game_result(sc.towers)
    # Champion rows of the unit columns mirror this tick's stats.
    champ_cols = dict(attack_damage=s.attack_damage.at[:c].set(st.base_ad + st.bonus_ad),
                      armor=s.armor.at[:c].set(st.base_armor + st.bonus_armor),
                      magic_resist=s.magic_resist.at[:c].set(st.base_mr + st.bonus_mr),
                      attack_range=s.attack_range.at[:c].set(sc.reach),
                      attack_speed=s.attack_speed.at[:c].set(st.attack_speed),
                      move_speed=s.move_speed.at[:c].set(sc.ms[:c]))
    new = s._replace(**champ_cols,
        t=sc.now, tick=s.tick + 1, key=sc.key, kind=sc.kind, alive=alive, x=sc.x, y=sc.y,
        hp=jnp.where(alive, hp, jnp.minimum(hp, 0.0)),
        max_hp=sc.max_hp, att=sc.att, missiles=sc.missiles, cc=cc, champ=sc.champ, kits=sc.kits, summoners=sc.summ,
        combat=out.state, econ=sc.econ, lane_ai=sc.lane_ai, towers=sc.towers, shields=sc.shields, status=sc.status,
        prev=LastTick(damage_matrix=sc.dmg, death_seen=sc.death_seen, kills=sc.kills, epic=sc.epic, large=sc.large,
                      pending_dash=sc.item_cleanse.dash),
        visible=sc.visible, sight=sc.sight,
        reveal=sc.reveal, sub=sc.sub, team=sc.team, spawn_seq=sc.spawn_seq, radius=sc.radius,
        targetable=sc.targetable, jungle=sc.jungle, wards=sc.wards, amove=sc.amove,
        game_over=result.over, winner=result.winner)
    return new, events
