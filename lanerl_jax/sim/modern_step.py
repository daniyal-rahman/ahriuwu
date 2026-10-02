"""Patch-26.19 modern world tick: every modern system composed in one step.

``step(state, orders, cfg)`` advances the modern Summoner's Rift top-lane world
by ``cfg.dt`` (30 Hz default, DAMAGE_AND_STATS §1.3). It is the integration of:

  modern_world        static config, unit layout (2 champions, 40 minion slots, 30 structures)
  modern_mechanics    attack machine, missiles, CC timers, route movement, blinks
  modern_lane_ai      minion/turret targeting, their attack packets, spawn stats, plates
  modern_champions    Garen/Jax kits (casts, empowered attacks, passives)
  modern_summoners    Flash/Teleport/Ignite/Exhaust/Barrier/Heal/Ghost/Cleanse
  modern_combat       items + runes around the damage pipeline (modern_damage)
  modern_economy      gold, XP, levels, bounty, death/respawn, recall, Homeguard, fountain
  modern_role_quest   top quest (inside economy_step)
  modern_inventory    shop (buy/sell/undo-free), item grants, transforms, consumption

Tick order (README crosswalk TICK.*; ECONOMY §13):
  1. INPUT     orders -> intents; shop actions; skill points
  2. STATS     STAT.00–60 for champions (static items/shards + kit + last tick's dynamic stats)
  3. CASTS     kit casts and periodic effects; summoner spells
  4. AI        minion/turret targets; champion attack targets
  5. MOVE      route movement, dashes, blinks, Teleport; collision
  6. ATTACK    attack machine; launches (crit roll); missiles advance
  7. DAMAGE    all packets -> combat_tick (items, runes, pipeline)
  8. CC/HEAL   CC applied with tenacity; kit/summoner heals and shields
  9. DEATH     minion/structure/champion deaths -> economy_step (gold, XP, levels, respawn)
 10. TIMERS    cooldowns, mana and HP regen (0.5 s), fountain, rune/item outputs

Observations and action decoding for RL are not part of this module (MODERN-005).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import modern_champions as K
from . import modern_damage as D
from . import modern_economy as E
from . import modern_inventory as I
from . import modern_lane_ai as LA
from . import modern_mechanics as M
from . import modern_summoners as S
from . import modern_world_types as W
from .modern_combat import CombatState, combat_tick, init_combat
from .modern_item_data import ItemStats, catalog, combine_stats, zero_stats
from .modern_item_effects.core import CC, Attack, Cast, Ctx, Kills, ShieldGrant, effects, merge_effects
from .modern_item_effects.runtime import UnitStatus, apply_effects, init_status
from .modern_rune_effects.core import rune_events
from .modern_stat_pipeline import ChampionStats, compose, cooldown, sync_max_health
from .modern_world import MAX_MINIONS, N_CHAMPIONS, WorldConfig, unit_ranges

RECALL_MS_LOCK = True
SKILL_ORDERS = {86: (2, 0, 1, 2, 2, 3, 2, 0, 2, 0, 3, 0, 0, 1, 1, 3, 1, 1),     # Garen E>Q>W (modern.skill_ranks)
                24: (2, 0, 1, 1, 1, 3, 1, 2, 1, 2, 3, 2, 2, 0, 0, 3, 0, 0)}     # Jax W>E>Q


class ChampionLayer(NamedTuple):
    """Per-champion world state (C,)."""
    ranks: Any              # (C, 4) int32
    cooldowns: Any          # (C, 4) remaining seconds
    mana: Any
    cast_lock_until: Any
    move_goal: Any          # (C, 2)
    moving: Any             # bool: walk to move_goal
    attack_order: Any       # int32 unit, -1 none
    facing: Any             # (C, 2)
    dash_until: Any         # dash end time (inf = none)
    dash_target: Any        # int32 unit to follow, -1 point
    dash_to: Any            # (C, 2)
    dash_speed: Any
    last_damaged: Any       # last time the champion took damage
    inventory: Any          # modern_inventory.Inventory (C, 7)
    group_cd: Any           # (C, G) item-group purchase cooldowns
    dyn: Any                # ItemStats: last tick's dynamic stats (combat_tick)
    static_max_hp: Any      # STAT max HP excluding dynamic health (for STAT.70 on level/items)
    homeguard_ms: Any
    blinked: Any            # last tick blinked/dashed (Sudden Impact)
    forbid: Any             # (C, I) purchases blocked by runes (Magical Footwear)
    reset_next: Any         # (C,) attack reset requested by items last tick (Titanic, Sheen-like)
    bonus_points: Any       # (C,) extra skill points (Elixir of Skill)
    granted: Any            # (C,) int32 item id placed this tick for a rune grant (ack)


class ModernState(NamedTuple):
    t: Any                  # () seconds
    tick: Any               # () int32
    key: Any                # PRNG key (crit rolls)
    # world units (N,)
    kind: Any
    sub: Any
    team: Any
    alive: Any
    x: Any
    y: Any
    hp: Any
    max_hp: Any
    radius: Any
    armor: Any
    mr: Any
    ad: Any
    arange: Any
    aspeed: Any
    mspeed: Any
    windup: Any
    spawn_seq: Any
    spawn_time: Any
    targetable: Any
    lane_wp: Any            # minion lane waypoint index
    missile_speed: Any      # (N,) ranged attack missile speed, 0 = melee
    m_gold: Any             # minion bounty / xp at spawn
    m_xp: Any
    m_level: Any
    next_seq: Any           # () int32
    wave_index: Any
    unit_index: Any
    att: W.AttackState
    missiles: M.Missiles
    cc: M.CCTimers
    champ: ChampionLayer
    kits: Any
    summoners: Any
    combat: CombatState
    econ: Any
    lane_ai: Any
    towers: Any
    shields: D.Shields
    status: UnitStatus
    kills: Kills            # takedowns credited last tick (fed to item/rune hooks)
    damage_matrix: Any      # (N, N) bool: i damaged j last tick (lane AI, kill credit)
    deaths_prev: Any        # (N,) bool: died last tick (Overgrowth)


class ModernOrders(NamedTuple):
    """One order per champion per tick (C,). Build with ``no_orders``."""
    move: Any               # bool
    move_x: Any
    move_y: Any
    attack: Any             # int32 unit, -1 none
    stop: Any               # bool: clear move and attack
    cast_slot: Any          # int32 -1 / 0..3
    cast_target: Any
    cast_x: Any
    cast_y: Any
    summoner_slot: Any      # int32 -1 / 0 (D) / 1 (F) / 2 (quest TP)
    summoner_target: Any
    summoner_x: Any
    summoner_y: Any
    item_active: Any        # int32 item id, 0 none (potions, Tiamat line, Stridebreaker, elixirs)
    buy: Any                # int32 item id, 0 none
    sell: Any               # int32 item id, 0 none
    recall: Any             # bool
    level_up: Any           # int32 slot to rank (-1 = automatic skill order)


def no_orders(c: int = N_CHAMPIONS) -> ModernOrders:
    z, f, i = jnp.zeros((c,), jnp.float32), jnp.zeros((c,), bool), jnp.zeros((c,), jnp.int32)
    return ModernOrders(f, z, z, i - 1, f, i - 1, i - 1, z, z, i - 1, i - 1, z, z, i, i, i, f, i - 1)


def _shard_stats(cfg: WorldConfig, level) -> ItemStats:
    """Static shard stats per champion (adaptive left unresolved for STAT.50)."""
    from .modern_items import stat_shard_stats
    from .modern_rune_data import RunePage
    parts = []
    for c, lo in enumerate(cfg.loadouts):
        if isinstance(lo.rune_page, RunePage):
            sh = stat_shard_stats(lo.rune_page.shards, level=level[c], adaptive_to_ad=None)
        else:
            sh = zero_stats(())
        parts.append(sh)
    return ItemStats(*(jnp.stack([jnp.asarray(getattr(p, k), jnp.float32) for p in parts])
                       for k in ItemStats._fields))


def init_state(cfg: WorldConfig, *, seed: int = 0) -> ModernState:
    n = cfg.n_units
    c = N_CHAMPIONS
    kind = cfg.unit_kind
    structure = (kind == W.KIND_TURRET) | (kind == W.KIND_INHIBITOR) | (kind == W.KIND_NEXUS)
    towers = LA.init_structures(cfg)
    hp = jnp.where(structure, towers.hp, 0.0).astype(jnp.float32)
    inv = I.inventory_from_ids([list(lo.items) for lo in cfg.loadouts])
    loadout = jnp.asarray([lo.summoners for lo in cfg.loadouts], jnp.int32)
    level = jnp.ones((c,), jnp.int32)
    static = combine_stats(I.inventory_stats(inv), _shard_stats(cfg, level))
    st = compose(cfg.champion_base, level, static, adaptive_physical=cfg.adaptive_physical)
    hp = hp.at[:c].set(st.max_hp)
    zc = jnp.zeros((c,), jnp.float32)
    ranks = jnp.zeros((c, 4), jnp.int32)
    first = jnp.asarray([_skill_order(cfg, i)[0] for i in range(c)])
    ranks = ranks.at[jnp.arange(c), first].set(1)
    champ = ChampionLayer(
        ranks=ranks, cooldowns=jnp.zeros((c, 4), jnp.float32), mana=st.max_mana, cast_lock_until=zc,
        move_goal=cfg.fountain, moving=jnp.zeros((c,), bool), attack_order=jnp.full((c,), -1, jnp.int32),
        facing=jnp.tile(jnp.asarray([[1.0, 1.0]], jnp.float32), (c, 1)) * jnp.asarray([[1.0], [-1.0]]),
        dash_until=zc - 1.0, dash_target=jnp.full((c,), -1, jnp.int32), dash_to=cfg.fountain, dash_speed=zc,
        last_damaged=zc - 1e9, inventory=inv,
        group_cd=jnp.zeros((c, catalog().arrays.groups.shape[1]), jnp.float32),
        dyn=zero_stats((c,)), static_max_hp=st.max_hp, homeguard_ms=zc, blinked=jnp.zeros((c,), bool),
        forbid=jnp.zeros((c, len(catalog().ids)), bool), reset_next=jnp.zeros((c,), bool),
        bonus_points=jnp.zeros((c,), jnp.int32), granted=jnp.zeros((c,), jnp.int32))
    f = lambda v: jnp.asarray(v, jnp.float32)
    radius = jnp.where(kind == W.KIND_CHAMPION, 65.0, jnp.where(structure, towers.radius, 48.0)).astype(jnp.float32)
    alive = (kind == W.KIND_CHAMPION) | structure
    return ModernState(
        t=jnp.float32(0.0), tick=jnp.int32(0), key=jax.random.PRNGKey(seed),
        kind=kind, sub=cfg.unit_sub, team=cfg.unit_team, alive=alive, x=cfg.unit_x, y=cfg.unit_y,
        hp=hp, max_hp=hp, radius=radius,
        armor=jnp.where(structure, towers.armor, 0.0).astype(jnp.float32).at[:c].set(st.base_armor + st.bonus_armor),
        mr=jnp.where(structure, towers.magic_resist, 0.0).astype(jnp.float32).at[:c].set(st.base_mr + st.bonus_mr),
        ad=jnp.where(structure, towers.attack_damage, 0.0).astype(jnp.float32).at[:c].set(st.base_ad + st.bonus_ad),
        arange=jnp.where(structure & (kind == W.KIND_TURRET), towers.attack_range, 0.0).astype(jnp.float32)
        .at[:c].set(st.attack_range),
        aspeed=jnp.where(kind == W.KIND_TURRET, towers.attack_speed, 0.0).astype(jnp.float32).at[:c].set(st.attack_speed),
        mspeed=jnp.zeros((n,), jnp.float32).at[:c].set(st.move_speed),
        windup=jnp.where(kind == W.KIND_TURRET, towers.windup, 0.0).astype(jnp.float32).at[:c].set(st.attack_windup),
        spawn_seq=jnp.arange(n, dtype=jnp.int32), spawn_time=jnp.zeros((n,), jnp.float32),
        targetable=alive & ~structure | (structure & towers.targetable),
        lane_wp=jnp.zeros((n,), jnp.int32),
        missile_speed=jnp.where(kind == W.KIND_TURRET, LA.T.MISSILE_SPEED, 0.0).astype(jnp.float32), m_gold=jnp.zeros((n,), jnp.float32), m_xp=jnp.zeros((n,), jnp.float32),
        m_level=jnp.ones((n,), jnp.int32), next_seq=jnp.int32(n), wave_index=jnp.int32(0), unit_index=jnp.int32(0),
        att=W.init_attack_state(n), missiles=M.init_missiles(64), cc=M.init_cc(n), champ=champ,
        kits=K.init(c, n), summoners=S.init(loadout), combat=init_combat(c, n),
        econ=E.init_economy(c, n, [lo.role for lo in cfg.loadouts]), lane_ai=LA.init_lane_ai(n), towers=towers,
        shields=D.init_shields(n), status=init_status(n),
        kills=Kills(zc, zc, zc, jnp.zeros((c,), bool), jnp.zeros((c, n), bool)),
        damage_matrix=jnp.zeros((n, n), bool), deaths_prev=jnp.zeros((n,), bool))


def _skill_order(cfg: WorldConfig, c: int) -> tuple:
    lo = cfg.loadouts[c]
    return tuple(lo.skill_order) or SKILL_ORDERS[{"Garen": 86, "Jax": 24}[lo.champion]]


def units_view(s: ModernState) -> W.WorldUnits:
    return W.WorldUnits(kind=s.kind, sub=s.sub, team=s.team, alive=s.alive, targetable=s.targetable & s.alive,
                        x=s.x, y=s.y, radius=s.radius, hp=s.hp, max_hp=s.max_hp, armor=s.armor,
                        magic_resist=s.mr, attack_damage=s.ad, attack_range=s.arange, attack_speed=s.aspeed,
                        move_speed=s.mspeed, spawn_seq=s.spawn_seq, spawn_time=s.spawn_time)


class TickEvents(NamedTuple):
    """Diagnostics of one tick (for tests, observations and rewards)."""
    report: Any             # combat main-pass Report
    follow_up: Any
    economy: Any            # modern_economy.EconomyOut
    plates: Any             # modern_lane_ai.PlateEvents
    launched: Any           # (N,) attacks launched
    packet_overflow: Any    # () dropped packets (must stay 0)
    missile_overflow: Any
    shop_code: Any          # (C,) buy/sell result code (0 ok)


# ---- helpers ----------------------------------------------------------------

def _static_stats(s: ModernState, cfg: WorldConfig, kit_stats: ItemStats, level) -> ItemStats:
    inv = I.inventory_stats(s.champ.inventory)
    return combine_stats(inv, _shard_stats(cfg, level), kit_stats)


def _kit_ctx(s: ModernState, cfg: WorldConfig, st: ChampionStats, caps: dict, now, dt) -> K.KitCtx:
    c = N_CHAMPIONS
    lv = s.econ.level
    return K.KitCtx(unit=jnp.arange(c, dtype=jnp.int32), champion_id=cfg.champion_ids, team=s.team[:c],
                    alive=s.alive[:c], level=lv, ranks=s.champ.ranks, x=s.x[:c], y=s.y[:c], mana=s.champ.mana,
                    max_mana=st.max_mana, hp=s.hp[:c], max_hp=s.max_hp[:c], base_ad=st.base_ad, bonus_ad=st.bonus_ad,
                    ap=st.ap, bonus_hp=s.max_hp[:c] - st.base_hp, armor=st.base_armor + st.bonus_armor,
                    magic_resist=st.base_mr + st.bonus_mr, bonus_attack_speed=st.bonus_attack_speed,
                    crit_chance=st.crit_chance, crit_damage=st.crit_damage, ability_haste=st.basic_ability_haste,
                    ultimate_haste=st.ultimate_haste, cooldowns=s.champ.cooldowns, now=now, dt=jnp.float32(dt),
                    silenced=caps["silenced"][:c], stunned=caps["stunned"][:c],
                    in_combat_ms_since_damaged=now - s.champ.last_damaged)


def _item_ctx(s: ModernState, cfg: WorldConfig, st: ChampionStats, now, dt) -> Ctx:
    c = N_CHAMPIONS
    return Ctx(now=now, dt=jnp.float32(dt), unit=jnp.arange(c, dtype=jnp.int32), team=s.team[:c], alive=s.alive[:c],
               level=s.econ.level.astype(jnp.float32), is_ranged=st.attack_range > 300.0, x=s.x[:c], y=s.y[:c],
               facing_x=s.champ.facing[:, 0], facing_y=s.champ.facing[:, 1], moved=jnp.zeros((c,), jnp.float32),
               base_ad=st.base_ad, bonus_ad=st.bonus_ad, ap=st.ap, base_hp=st.base_hp, max_hp=s.max_hp[:c],
               hp=s.hp[:c], base_armor=st.base_armor, bonus_armor=st.bonus_armor, base_mr=st.base_mr,
               bonus_mr=st.bonus_mr, mana=s.champ.mana, max_mana=st.max_mana,
               base_ms=cfg.champion_base.base_ms, move_speed=st.move_speed, crit_chance=st.crit_chance,
               crit_damage=st.crit_damage, life_steal=st.life_steal, bonus_attack_speed=st.bonus_attack_speed,
               ability_haste=st.basic_ability_haste, lethality=st.lethality, heal_shield_power=st.heal_shield_power,
               attack_windup=st.attack_windup, in_combat=(now - s.combat.clocks.last_combat) < 5.0,
               in_shop=I.in_shop_area(s.x[:c], s.y[:c], s.team[:c], ~s.alive[:c]),
               base_mana=cfg.champion_base.base_mana, attack_range=st.attack_range)


def _spawn_minions(s: ModernState, cfg: WorldConfig, now) -> ModernState:
    """MINIONS §2: one unit event per team at a time, 0.8 s apart, into free slots."""
    from . import modern_minions as MM
    ev = MM.spawn_event(s.wave_index, s.unit_index)
    due = (now >= ev.spawn_time_s) & ev.valid
    _, msl, _ = unit_ranges()
    free = (s.kind == W.KIND_NONE) | ((s.kind == W.KIND_MINION) & ~s.alive)
    slot_mask = jnp.zeros_like(free).at[msl].set(free[msl])
    stats = LA.minion_spawn_stats(ev.minion_type, now)
    for team in (0, 1):
        rank = jnp.cumsum(slot_mask) - 1
        pick = slot_mask & (rank == 0) & due
        put = lambda arr, v: jnp.where(pick, jnp.asarray(v, arr.dtype), arr)
        s = s._replace(
            kind=put(s.kind, W.KIND_MINION), sub=put(s.sub, jnp.clip(ev.minion_type, 0, 3)), team=put(s.team, team),
            alive=put(s.alive, True), targetable=put(s.targetable, True),
            x=put(s.x, cfg.minion_spawn[team, 0]), y=put(s.y, cfg.minion_spawn[team, 1]),
            hp=put(s.hp, stats.max_hp), max_hp=put(s.max_hp, stats.max_hp), radius=put(s.radius, stats.radius),
            armor=put(s.armor, stats.armor), mr=put(s.mr, stats.magic_resist), ad=put(s.ad, stats.attack_damage),
            arange=put(s.arange, stats.attack_range), aspeed=put(s.aspeed, stats.attack_speed),
            mspeed=put(s.mspeed, stats.move_speed), windup=put(s.windup, stats.windup),
            missile_speed=put(s.missile_speed, stats.missile_speed),
            spawn_seq=put(s.spawn_seq, s.next_seq + jnp.sum(pick) * 0 + team),
            spawn_time=put(s.spawn_time, now), m_gold=put(s.m_gold, stats.gold), m_xp=put(s.m_xp, stats.xp),
            m_level=put(s.m_level, jnp.max(s.econ.level)), lane_wp=put(s.lane_wp, 0),
            att=s.att._replace(target=put(s.att.target, -1), windup_left=put(s.att.windup_left, 0.0),
                               cooldown_left=put(s.att.cooldown_left, 0.0)),
            cc=M.CCTimers(*(put(v, 0.0) for v in s.cc)))
        slot_mask = slot_mask & ~pick
    nxt = MM.spawn_event(s.wave_index, s.unit_index + 1)
    close = due & ~nxt.valid
    return s._replace(next_seq=s.next_seq + 2 * due.astype(jnp.int32),
                      wave_index=s.wave_index + close.astype(jnp.int32),
                      unit_index=jnp.where(close, 0, s.unit_index + due.astype(jnp.int32)))


def _shop(s: ModernState, cfg: WorldConfig, orders: ModernOrders, st: ChampionStats, forbid) -> tuple[Any, Any, Any]:
    """Buy/sell for each champion in the shop area or dead (ITEMS §1)."""
    cat = catalog()
    ids = jnp.asarray(cat.arrays.item_id)
    inv, gold, gcd = s.champ.inventory, s.econ.gold, s.champ.group_cd
    can = I.in_shop_area(s.x[:N_CHAMPIONS], s.y[:N_CHAMPIONS], s.team[:N_CHAMPIONS], ~s.alive[:N_CHAMPIONS])
    codes = []
    items, stacks, golds, cds, bought, sold = [], [], [], [], [], []
    for c in range(N_CHAMPIONS):
        inv_c = I.Inventory(inv.item[c], inv.stack[c])
        row = jnp.argmax(ids == orders.buy[c])
        want = (orders.buy[c] > 0) & jnp.any(ids == orders.buy[c]) & ~forbid[c, row]
        r = I.buy(inv_c, gold[c], row, can_shop=can[c] & want, level=s.econ.level[c],
                  is_ranged=st.attack_range[c] > 300.0, now=s.t, group_cd_until=gcd[c])
        inv_c, g = r.inv, r.gold
        srow = jnp.argmax(ids == orders.sell[c])
        slot = jnp.argmax(inv_c.item == srow)
        has_it = (orders.sell[c] > 0) & jnp.any(inv_c.item == srow)
        r2 = I.sell(inv_c, g, slot, can_shop=can[c] & has_it)
        inv_c = r2.inv
        items.append(inv_c.item); stacks.append(inv_c.stack); golds.append(r2.gold); cds.append(r.group_cd_until)
        codes.append(jnp.where(want, r.code, jnp.where(has_it, r2.code, 0)))
        bought.append(jnp.where(want & r.ok, orders.buy[c], 0)); sold.append(jnp.where(has_it & r2.ok, orders.sell[c], 0))
    inv = I.Inventory(jnp.stack(items), jnp.stack(stacks))
    return (inv, jnp.stack(golds), jnp.stack(cds), jnp.stack(codes), jnp.stack(bought).astype(jnp.int32),
            jnp.stack(sold).astype(jnp.int32))


def _item_units(s: ModernState) -> Any:
    """``modern_item_effects.core.Units`` view of the world."""
    from .modern_item_effects.core import Units
    return Units(x=s.x, y=s.y, team=s.team, cls=W.damage_class(s.kind), alive=s.alive, hp=s.hp, max_hp=s.max_hp,
                 radius=s.radius, targetable=s.targetable & s.alive,
                 is_siege_or_super=(s.kind == W.KIND_MINION) & (s.sub >= 2),
                 bonus_hp=jnp.zeros_like(s.hp), armor=s.armor, magic_resist=s.mr)


def _skill_up(s: ModernState, cfg: WorldConfig, orders: ModernOrders) -> Any:
    """Spend available skill points: chosen slot, else the champion's default order."""
    ranks = s.champ.ranks
    level = s.econ.level
    points = E.skill_points(level) + s.champ.bonus_points
    spent = jnp.sum(ranks, axis=1)
    order = jnp.asarray([list(_skill_order(cfg, c)) + [0] * (20 - len(_skill_order(cfg, c)))
                         for c in range(N_CHAMPIONS)], jnp.int32)
    auto = jnp.take_along_axis(order, jnp.clip(spent, 0, 19)[:, None], axis=1)[:, 0]
    slot = jnp.where(orders.level_up >= 0, orders.level_up, auto)
    cap = jnp.where(slot == 3, E.max_rank(level, ultimate=True), E.max_rank(level))
    cur = jnp.take_along_axis(ranks, slot[:, None], axis=1)[:, 0]
    ok = (points > spent) & (cur < cap)
    return ranks + (jnp.arange(4)[None, :] == slot[:, None]) * ok[:, None]


def step(s: ModernState, orders: ModernOrders, cfg: WorldConfig) -> tuple[ModernState, TickEvents]:
    """Advance the modern world one tick (``cfg.dt``); returns ``(state, events)``."""
    from . import modern_item_effects as IE
    s0 = s
    dt = cfg.dt
    c, n = N_CHAMPIONS, cfg.n_units
    now = s.t + jnp.float32(dt)
    key, k_crit = jax.random.split(s.key)
    cat = catalog()
    champ = s.champ

    # ---- 1. INPUT -----------------------------------------------------------------------
    attack_order = jnp.where(orders.stop | orders.move, -1, jnp.where(orders.attack >= 0, orders.attack,
                                                                    champ.attack_order))
    moving = jnp.where(orders.stop | (orders.attack >= 0), False, orders.move | champ.moving)
    goal = jnp.where(orders.move[:, None], jnp.stack([orders.move_x, orders.move_y], -1), champ.move_goal)
    ranks = _skill_up(s, cfg, orders)
    champ = champ._replace(attack_order=attack_order.astype(jnp.int32), moving=moving, move_goal=goal, ranks=ranks)
    s = s._replace(champ=champ)
    s = _spawn_minions(s, cfg, now)
    caps = M.capabilities(s.cc, now)

    # ---- 2. STATS -----------------------------------------------------------------------
    level = s.econ.level
    base_static = combine_stats(I.inventory_stats(champ.inventory), _shard_stats(cfg, level))
    st0 = compose(cfg.champion_base, level, base_static, adaptive_physical=cfg.adaptive_physical)
    kit_stats = K.stats(s.kits, _kit_ctx(s, cfg, st0, caps, now, dt))
    static = combine_stats(base_static, kit_stats)
    st_static = compose(cfg.champion_base, level, static, adaptive_physical=cfg.adaptive_physical)
    # STAT.70 for static max-HP changes (level-up, purchases, kit stacks); dynamic health is
    # synced by combat_tick.
    old_total = s.max_hp[:c]
    new_total = st_static.max_hp + s.combat.dyn_health
    hp_c, mx_c = sync_max_health(s.hp[:c], old_total, new_total)
    hp_c = jnp.where(s.alive[:c], hp_c, s.hp[:c])
    s = s._replace(hp=s.hp.at[:c].set(hp_c), max_hp=s.max_hp.at[:c].set(mx_c),
                   champ=champ._replace(static_max_hp=st_static.max_hp))

    # ---- 3. CASTS: shop, kits, summoners --------------------------------------------------
    inv, gold, gcd, shop_code, bought, sold = _shop(s, cfg, orders, st_static, champ.forbid)
    econ = s.econ._replace(gold=gold, gold_total=s.econ.gold_total)
    champ = s.champ._replace(inventory=inv, group_cd=gcd)
    s = s._replace(econ=econ, champ=champ)
    summ_world = combine_stats(static, s.champ.dyn)
    st = compose(cfg.champion_base, level, summ_world, adaptive_physical=cfg.adaptive_physical,
                 slow=caps["slow"][:c])
    units = units_view(s)
    kctx = _kit_ctx(s, cfg, st, caps, now, dt)
    ictx = _item_ctx(s, cfg, st_static, now, dt)
    locked = now < champ.cast_lock_until
    can_cast = caps["can_cast"][:c] & s.alive[:c] & ~locked
    order = W.CastOrder(jnp.where(can_cast, orders.cast_slot, -1).astype(jnp.int32), orders.cast_target,
                        orders.cast_x, orders.cast_y)
    kits, k_cast = K.cast(s.kits, kctx, units, order)
    kits, k_per = K.periodic(kits, kctx, units)
    from .modern_champions.core import merge_out
    kit_out = merge_out([k_cast, k_per], c, n)
    summ_req = W.CastOrder(orders.summoner_slot, orders.summoner_target, orders.summoner_x, orders.summoner_y)
    summ, s_eff, s_out = S.step(s.summoners, ictx, units, request=summ_req, now=now, dt=jnp.float32(dt),
                                summoner_haste=st.summoner_haste,
                                can_cast=caps["can_summoner"][:c] & s.alive[:c],
                                channel_interrupted=caps["stunned"][:c] | ~s.alive[:c],
                                quest_complete=s.econ.quest.complete,
                                took_champion_damage=(now - s.combat.clocks.last_hit_by_champion) < 1e-3,
                                rooted=(s.cc.root_until[:c] > now))

    # ---- 4. AI: minion/turret targets, champion targets ------------------------------------
    towers = LA.turret_tick(s.towers, units, now=now, dt=jnp.float32(dt))
    hp_t, alive_t, targ_t = LA.structure_unit_view(towers, units)
    s = s._replace(hp=hp_t, alive=alive_t, targetable=targ_t)
    units = units_view(s)
    champ_vs_champ = s.damage_matrix & (s.kind == W.KIND_CHAMPION)[:, None] & (s.kind == W.KIND_CHAMPION)[None, :]
    lane_ai, desired, mgoal, stop = LA.select_targets(s.lane_ai, units, s.att, now=now, dt=jnp.float32(dt),
                                                      champion_attacked_champion=champ_vs_champ,
                                                      damage_events=s.damage_matrix)
    t_ok = (attack_order >= 0) & s.alive[jnp.clip(attack_order, 0, n - 1)]
    desired = desired.at[:c].set(jnp.where(t_ok, attack_order, -1))

    # ---- 5. MOVE ------------------------------------------------------------------------------
    tp_lock = s_out.teleport_channel | s_out.teleport_dash
    can_move_c = caps["can_move"][:c] & s.alive[:c] & ~locked & ~tp_lock & (now >= champ.dash_until)
    tgt = jnp.clip(attack_order, 0, n - 1)
    in_range = M.in_attack_range(units._replace(attack_range=units.attack_range.at[:c].set(st.attack_range)),
                                 jnp.full((n,), -1, jnp.int32).at[:c].set(jnp.where(t_ok, attack_order, -1)))[:c]
    chase = t_ok & ~in_range
    cgoal = jnp.where(chase[:, None], jnp.stack([s.x[tgt], s.y[tgt]], -1), goal)
    cact = can_move_c & (chase | (moving & ~t_ok))
    minion = (s.kind == W.KIND_MINION) & s.alive
    gx = jnp.where(minion, mgoal[:, 0], s.x).at[:c].set(cgoal[:, 0])
    gy = jnp.where(minion, mgoal[:, 1], s.y).at[:c].set(cgoal[:, 1])
    ms = LA.minion_move_speed(lane_ai, units, now).at[:c].set(st.move_speed * (1.0 + s_out.bonus_ms_pct)
                                                              + champ.homeguard_ms * cfg.champion_base.base_ms)
    active = (minion & ~stop & caps["can_move"]).at[:c].set(cact)
    x, y, _ = M.move_step(s.x, s.y, gx, gy, ms, active, s.team, s.radius, cfg.routes, cfg.terrain, dt)
    # Kit dashes (Jax Q): follow the target unit at the dash speed (no terrain, it's a leap).
    dash = kit_out.dash
    dstart = dash.active & s.alive[:c]
    dt_ = jnp.clip(dash.target, 0, n - 1)
    dx = jnp.where(dash.target >= 0, s.x[dt_], dash.to_x)
    dy = jnp.where(dash.target >= 0, s.y[dt_], dash.to_y)
    dist = jnp.sqrt((dx - s.x[:c]) ** 2 + (dy - s.y[:c]) ** 2)
    dash_until = jnp.where(dstart, now + dist / jnp.maximum(dash.speed, 1.0), champ.dash_until)
    dash_target = jnp.where(dstart, dash.target, champ.dash_target)
    in_dash = now < dash_until
    ft = jnp.clip(dash_target, 0, n - 1)
    fx, fy = jnp.where(dash_target >= 0, s.x[ft], champ.dash_to[:, 0]), jnp.where(dash_target >= 0, s.y[ft], champ.dash_to[:, 1])
    fd = jnp.sqrt((fx - x[:c]) ** 2 + (fy - y[:c]) ** 2)
    step_d = jnp.minimum(jnp.where(dash.speed > 0, dash.speed, 1400.0) * dt, fd)
    frac = jnp.where(fd > 1e-6, step_d / jnp.maximum(fd, 1e-6), 0.0)
    cx = jnp.where(in_dash, x[:c] + (fx - x[:c]) * frac, x[:c])
    cy = jnp.where(in_dash, y[:c] + (fy - y[:c]) * frac, y[:c])
    # Flash (blink onto walkable terrain) and Teleport arrival.
    blink = s_out.dash.active & s_out.dash.blink
    bx, by = M.blink_point(cx, cy, s_out.dash.to_x, s_out.dash.to_y, jnp.full((c,), S.FLASH_RANGE), s.team[:c],
                           s.radius[:c], cfg.terrain)
    cx, cy = jnp.where(blink, bx, cx), jnp.where(blink, by, cy)
    cx = jnp.where(s_out.teleport_arrive, s_out.teleport_x, cx)
    cy = jnp.where(s_out.teleport_arrive, s_out.teleport_y, cy)
    x, y = x.at[:c].set(cx), y.at[:c].set(cy)
    # Unit collision (static terrain checked by the movement clamp; dynamic terrain deferred).
    from .collision import resolve_collisions
    legacy_kind = jnp.where(s.kind >= W.KIND_TURRET, 3, s.kind)
    ghost = (jnp.zeros((n,), bool).at[:c].set(s_out.ghosted | (now < dash_until)))
    x, y = resolve_collisions(x, y, legacy_kind, s.alive, s.spawn_seq, s.radius, s.radius, ghosted=ghost)
    facing = jnp.stack([x[:c] - s.x[:c], y[:c] - s.y[:c]], -1)
    norm = jnp.linalg.norm(facing, axis=-1, keepdims=True)
    facing = jnp.where(norm > 1e-3, facing / jnp.maximum(norm, 1e-6), champ.facing)
    moved = jnp.sqrt((x[:c] - s.x[:c]) ** 2 + (y[:c] - s.y[:c]) ** 2)
    s = s._replace(x=x, y=y, lane_ai=lane_ai, towers=towers,
                   champ=champ._replace(dash_until=dash_until, dash_target=dash_target.astype(jnp.int32),
                                        dash_to=jnp.where(dstart[:, None], jnp.stack([dx, dy], -1), champ.dash_to),
                                        dash_speed=jnp.where(dstart, dash.speed, champ.dash_speed), facing=facing))
    units = units_view(s)._replace(attack_range=s.arange.at[:c].set(st.attack_range),
                                   attack_speed=s.aspeed.at[:c].set(st.attack_speed))
    ictx = ictx._replace(x=x[:c], y=y[:c], facing_x=facing[:, 0], facing_y=facing[:, 1], moved=moved)

    # ---- 6. ATTACK ------------------------------------------------------------------------------
    kmods = K.attack_mods(s.kits, kctx)
    can_attack = (caps["can_attack"] & s.alive).at[:c].set(
        caps["can_attack"][:c] & s.alive[:c] & ~kmods.cannot_attack & ~locked & ~tp_lock & ~in_dash)
    windup = s.windup.at[:c].set(st.attack_windup)
    reset = jnp.zeros((n,), bool).at[:c].set(kit_out.attack_reset | kmods.attack_reset | champ.reset_next)
    att_prev = s.att
    att, launched = M.attack_step(s.att, units, desired, can_attack=can_attack, windup=windup, dt=dt, reset=reset)
    started = (att.windup_left > 0) & (att_prev.windup_left <= 0)
    cancelled = (att_prev.windup_left > 0) & (att.windup_left <= 0) & ~launched
    atgt = jnp.clip(att.target, 0, n - 1)
    # Champions: crit roll at launch (X-8), Garen Q's spell attack cannot crit.
    imods = IE.attack_mods(s.combat.items, IE_own(champ.inventory), ictx, _item_units(s), att.target[:c])
    no_crit = jnp.zeros((c,), bool) if kmods.cannot_crit is None else kmods.cannot_crit
    roll = jax.random.uniform(k_crit, (c,)) < st.crit_chance
    crit = launched[:c] & ~no_crit & (roll | imods.force_crit)
    crit_mult = 1.0 + (st.crit_damage - 1.0) * jnp.where(imods.force_crit, imods.crit_scale, 1.0)
    vs_struct = (s.kind[atgt[:c]] == W.KIND_TURRET) | (s.kind[atgt[:c]] == W.KIND_INHIBITOR) \
        | (s.kind[atgt[:c]] == W.KIND_NEXUS)
    struct_raw, struct_magic = LA.T.champion_structure_attack(st.base_ad, st.bonus_ad, st.ap)
    craw = jnp.where(vs_struct, struct_raw, (st.base_ad + st.bonus_ad) * jnp.where(crit, crit_mult, 1.0))
    # Melee champions deal x1.2 to turrets (towers.json melee_champion_damage_multiplier; a
    # multiplicative factor, so applying it to raw is equivalent to post-mitigation).
    vs_turret = s.kind[atgt[:c]] == W.KIND_TURRET
    craw = craw * jnp.where(vs_turret & (s.missile_speed[:c] <= 0), 1.2, 1.0)
    cdtype = jnp.where(vs_struct & struct_magic, D.MAGIC, D.PHYSICAL)
    cast_ids = (s.tick * 128 + jnp.arange(n)).astype(jnp.int32) + 1          # < 2^30, never a kit id
    ranged_c = s.missile_speed[:c] > 0
    hit_c = launched[:c] & ~ranged_c
    attack = Attack(launched[:c], hit_c, att.target[:c], jnp.where(launched[:c], craw, 0.0), crit)
    lane_pk = LA.attack_packets(units, W.AttackLaunch(launched & (s.kind != W.KIND_CHAMPION), att.target,
                                                      s.missile_speed > 0, jnp.zeros((n,), bool), cast_ids),
                                now=now, ai=lane_ai)
    lane_pk = lane_pk._replace(cast_id=cast_ids)
    ranged = launched & (s.missile_speed > 0)
    flags_c = jnp.full((c,), D.BASIC_ATTACK, jnp.int32) | jnp.where(crit, D.PROP_CRIT, 0)
    raw_all = lane_pk.raw.at[:c].set(craw)
    dtype_all = lane_pk.dtype.at[:c].set(cdtype)
    flags_all = lane_pk.flags.at[:c].set(flags_c)
    missiles, m_over = M.spawn_missiles(s.missiles, ranged, units, att.target, raw_all, dtype_all, flags_all,
                                        s.missile_speed, cast_ids, jnp.zeros((n,), bool).at[:c].set(crit))
    missiles, arrive = M.advance_missiles(missiles, units, dt)
    direct = D.packets(launched & ~ranged & (att.target >= 0), jnp.arange(n), jnp.maximum(att.target, 0), raw_all,
                       dtype_all, flags_all, cast_id=cast_ids)
    arrived = D.packets(arrive, missiles.src, missiles.dst, missiles.raw, missiles.dtype, missiles.flags,
                        cast_id=missiles.cast_id)
    # Champion ranged hits arriving now are on-hit for items/kits too.
    arrive_c = jnp.zeros((c,), bool).at[jnp.clip(missiles.src, 0, c - 1)].max(arrive & (missiles.src < c))
    attack = attack._replace(hit=attack.hit | arrive_c)
    launch_c = W.AttackLaunch(attack.launched, attack.target, ranged_c, crit, cast_ids[:c])
    kits, k_att = K.on_attack(kits, kctx, units, launch_c)
    kits, k_hit = K.on_hit(kits, kctx, units, launch_c._replace(launched=attack.hit))
    # Crystalline Overgrowth consumed by champion basic attacks on turrets.
    champ_hit = jnp.zeros((n, n), bool).at[:c].set((attack.hit[:, None]
                                                    & (jnp.arange(n)[None, :] == attack.target[:, None])))
    towers, og_pk = LA.overgrowth_packets(s.towers, units, champ_hit, now=now,
                                          team_level=decimal_team_level(s))
    kit_all = merge_out([kit_out, k_att, k_hit], c, n)

    # ---- 7. DAMAGE ---------------------------------------------------------------------------------
    base = D.concat_packets(direct, arrived, og_pk, kit_all.packets, s_eff.packets)
    kdef = K.defense(kits, kctx)
    kdeb = K.debuffs(kits, kctx, units)
    t_armor, t_mr = LA.turret_defense(towers, units, now=now)
    t_mult, t_invuln = LA.structure_defense_mods(towers, units, now=now)
    armor = t_armor.at[:c].set(st_static.base_armor + st_static.bonus_armor)
    mr = t_mr.at[:c].set(st_static.base_mr + st_static.bonus_mr)
    dfn = D.default_defense(n)._replace(
        armor=armor, magic_resist=mr, unit_class=W.damage_class(s.kind),
        received_mult=jnp.ones((n,), jnp.float32).at[:c].set(kdef.received_mult),
        received_mult_all=t_mult,
        invulnerable=t_invuln | jnp.zeros((n,), bool).at[:c].set(s_out.teleport_dash),
        percent_armor_reduction=kdeb.percent_armor_reduction,
        dodge_basic=jnp.zeros((n,), bool).at[:c].set(kdef.dodge_basic),
        aoe_received_mult=jnp.ones((n,), jnp.float32).at[:c].set(kdef.aoe_received_mult))
    off = LA.turret_offense(units, D.default_offense(n)._replace(unit_class=W.damage_class(s.kind),
                                                                dealt_reduction=s_out.exhaust_reduction))
    cc_now = kit_all.cc
    cc_items = CC(cc_now.slow > 0, (cc_now.stun > 0) | (cc_now.root > 0) | (cc_now.knockup > 0))
    deaths_prev = s.deaths_prev
    ev = rune_events(ictx, n, game_time=now, attack_started=started[:c], attack_start_target=att.target[:c],
                     attack_cancelled=cancelled[:c], attack_reset=reset[:c], cast_id=kit_all.cast_id,
                     cc_duration=jnp.maximum(cc_now.stun, cc_now.root), impaired=caps["impaired"],
                     movement_impaired=caps["movement_impaired"],
                     impaired_by_holder=(cc_now.slow > 0) | (cc_now.stun > 0) | (cc_now.root > 0),
                     holder_cc_from_champion=s.cc.champion_cc_until[:c] > now,
                     summoner_cast=s_out.cast_event, summoner_cooldown=s_out.cast_cooldown,
                     summoner_is_teleport=s_out.is_teleport,
                     blinked=s_out.blinked | dstart, flash_cooldown=S.flash_cooldown(summ, now),
                     deaths=deaths_prev, purchased=bought, sold=sold, granted=champ.granted,
                     uses_energy=cfg.uses_energy, adaptive_physical=cfg.adaptive_physical,
                     is_turret=s.kind == W.KIND_TURRET, cc_cast_id=cc_now.cast_id,
                     cc_on_hit=jnp.zeros((c, n), bool))
    out = combat_tick(s.combat, IE_own(champ.inventory), cfg.rune_pages, ictx, _item_units(s), attack=attack,
                      cast=Cast(kit_all.cast_started, kit_all.cast_slot, order.target),
                      request=orders.item_active, base_packets=base, base_offense=off, base_defense=dfn,
                      hp=s.hp, max_hp=s.max_hp, shields=s.shields, status=s.status, kills=s.kills,
                      holder_stats=static, cc=cc_items, ev=ev)
    hp, max_hp, shields, status = out.hp, out.max_hp, out.shields, out.status
    kits, k_dmg = K.on_damage(kits, kctx, units, out.report)

    # ---- 8. CC / heals / shields from kits and summoners ------------------------------------------------
    kit_shields = ShieldGrant(*(jnp.concatenate([u, v], axis=1) for u, v in zip(kit_all.shield, k_dmg.shield)))
    extra = merge_effects([s_eff._replace(packets=D.empty_packets(0)),
                           effects(c, n, heal=kit_all.heal + k_dmg.heal, shields=kit_shields)], c, n)
    hp, shields, status = apply_effects(extra, ictx, hp, max_hp, shields, status,
                                        heal_power=st.heal_shield_power, incoming_heal=summ_world.incoming_heal)
    exhaust_cc = W.no_cc(1, n)._replace(slow=s_out.exhaust_slow[None, :], slow_duration=s_out.exhaust_slow_duration[None, :])
    ten = jnp.zeros((n,), jnp.float32).at[:c].set(
        1.0 - (1.0 - st.tenacity) * (1.0 - kdef.tenacity_bonus) * (1.0 - s_out.tenacity))
    sres = jnp.zeros((n,), jnp.float32).at[:c].set(st.slow_resist)
    cc = M.apply_cc(s.cc, cc_now, ten, sres, now, source_is_champion=jnp.ones((c,), bool),
                    cleansed=jnp.zeros((n,), bool).at[:c].set(s_out.cleanse))
    cc = M.apply_cc(cc, exhaust_cc, ten, sres, now, source_is_champion=jnp.asarray([True]))
    clean_slow = jnp.zeros((n,), bool) if kit_all.cleanse_slow is None else \
        jnp.zeros((n,), bool).at[:c].set(kit_all.cleanse_slow)
    cc = cc._replace(slow_until=jnp.where(clean_slow, now, cc.slow_until))

    # ---- 9. DEATH and economy --------------------------------------------------------------------------
    rp = D.concat_packets(out.report.packets, out.follow_up.packets)
    rr_killed = jnp.concatenate([out.report.resolved.killed, out.follow_up.resolved.killed])
    rr_loss = jnp.concatenate([out.report.resolved.health_loss, out.follow_up.resolved.health_loss])
    killer = jnp.full((n,), -1, jnp.int32).at[jnp.clip(rp.dst, 0, n - 1)].max(
        jnp.where(rp.valid & rr_killed, rp.src, -1).astype(jnp.int32))
    dmg = jnp.zeros((n, n), bool).at[jnp.clip(rp.src, 0, n - 1), jnp.clip(rp.dst, 0, n - 1)].max(rp.valid)
    died = s.alive & (hp <= 0.0)
    struct = (s.kind == W.KIND_TURRET) | (s.kind == W.KIND_INHIBITOR) | (s.kind == W.KIND_NEXUS)
    towers, plates = LA.structure_damage_events(towers, s.hp, jnp.where(struct, hp, s.hp), now=now)
    minion_died = died & (s.kind == W.KIND_MINION)
    last_hitter = jnp.where(killer < c, killer, -1)
    md = E.MinionDeaths(valid=minion_died, x=s.x, y=s.y, team=s.team, gold=s.m_gold, xp=s.m_xp, level=s.m_level,
                        last_hitter=last_hitter, unit=jnp.arange(n, dtype=jnp.int32))
    sv = plates.plates > 0
    sev = E.StructureEvents(valid=sv | plates.destroyed, unit=jnp.arange(n, dtype=jnp.int32), x=s.x, y=s.y,
                            team=s.team, local_gold=plates.plate_gold + plates.first_turret_gold,
                            global_gold=plates.global_gold, is_turret=plates.destroyed & (s.kind == W.KIND_TURRET),
                            in_top_lane=cfg.unit_lane == 2)
    took_health = jnp.zeros((n,), bool).at[jnp.clip(rp.dst, 0, n - 1)].max(rp.valid & (rr_loss > 0))
    in_f = E.in_fountain(x[:c], y[:c], cfg.fountain[s.team[:c], 0], cfg.fountain[s.team[:c], 1])
    from .modern_item_effects.core import Report
    einp = E.EconomyInputs(
        now=now, unit=jnp.arange(c, dtype=jnp.int32), x=x[:c], y=y[:c], team=s.team[:c], hp=hp[:c],
        max_hp=max_hp[:c], report=Report(rp, None, None), cc=cc_items,
        final_blow=jnp.where(killer[:c] < c, killer[:c], -1), minion_deaths=md,
        minion_in_lane=jnp.ones((n,), bool), structures=sev,
        last_champion_combat=out.state.clocks.last_champion_combat, in_fountain=in_f,
        in_quest_lane=~in_f, recall_request=orders.recall,
        cancel_action=orders.move | (orders.attack >= 0) | (orders.cast_slot >= 0) | (orders.summoner_slot >= 0),
        health_damage=took_health[:c], disabled=caps["stunned"][:c] | caps["silenced"][:c] | (s.cc.root_until[:c] > now),
        reached_endpoint=jnp.zeros((c,), bool), in_jungle=jnp.zeros((c,), bool), teleported=s_out.teleport_arrive)
    eco = E.economy_step(s.econ, einp)
    econ = eco.state
    gold_extra = out.effects.gold + s_eff.gold
    econ = econ._replace(gold=jnp.minimum(econ.gold + gold_extra, 100000.0), gold_total=econ.gold_total + gold_extra)
    kits = K.on_takedown(kits, kctx, units, eco.kills)

    # ---- 10. TIMERS, respawn, recall, outputs ----------------------------------------------------------
    alive = s.alive & ~died
    champ_dead = ~alive[:c]
    respawn = eco.respawned
    recall = eco.recalled
    fx_, fy_ = cfg.fountain[s.team[:c], 0], cfg.fountain[s.team[:c], 1]
    x = x.at[:c].set(jnp.where(respawn | recall, fx_, x[:c]))
    y = y.at[:c].set(jnp.where(respawn | recall, fy_, y[:c]))
    alive = alive.at[:c].set(jnp.where(respawn, True, alive[:c]))
    hp = hp.at[:c].set(jnp.where(respawn, max_hp[:c], hp[:c]))
    # Cooldowns: kit starts (hasted), refunds from runes, decrement.
    haste = jnp.stack([st.basic_ability_haste] * 3 + [st.ultimate_haste], -1)
    cds = jnp.maximum(champ.cooldowns - dt, 0.0)
    cds = jnp.where(kit_all.cooldown_start, cooldown(kit_all.base_cooldown, haste), cds)
    ro = out.rune_outputs
    cds = cds.at[:, :3].multiply((1.0 - ro.basic_cd_refund)[:, None]).at[:, 3].multiply(1.0 - ro.ult_cd_refund)
    mana = jnp.clip(champ.mana - kit_all.mana_cost + out.effects.mana + st.mana_regen * dt, 0.0, st.max_mana)
    mana = jnp.where(respawn, st.max_mana, mana)
    # HP regen (0.5 s ticks, §12) and the fountain (+ Homeguard heal).
    pulse = jnp.floor(now / 0.5 + 1e-6) - jnp.floor(s.t / 0.5 + 1e-6)
    regen = jnp.where(alive[:c] & ~respawn, st.hp_regen * 0.5 * pulse, 0.0)
    hp_c = jnp.minimum(hp[:c] + regen, max_hp[:c])
    hp_c, mana = E.fountain_regen(hp_c, max_hp[:c], mana, st.max_mana, in_f & alive[:c], s.t, now,
                                  homeguard=econ.homeguard.active)
    hp = hp.at[:c].set(hp_c)
    # Item outputs: Tear transforms, consumption, rune item grants.
    inv = champ.inventory
    frm, to, do = out.transforms
    inv = I.Inventory(*jax.vmap(lambda it, stk, f_, t_, d_: tuple(I.replace_item(I.Inventory(it, stk), f_, t_, d_)))(
        inv.item, inv.stack, frm, to, do))
    slot = jnp.argmax(inv.item == out.consume_row[:, None], axis=1)
    inv = I.Inventory(*jax.vmap(lambda it, stk, sl, ok: tuple(I.consume_one(I.Inventory(it, stk), sl, ok)))(
        inv.item, inv.stack, slot, out.consume_row >= 0))
    grant_row = jnp.argmax(jnp.asarray(cat.arrays.item_id)[None, :] == ro.grant_item[:, None], axis=1)
    free = (inv.item[:, :6] < 0)
    can_grant = (ro.grant_item > 0) & jnp.any(free, axis=1)
    gslot = jnp.argmax(free, axis=1)
    put = can_grant[:, None] & (jnp.arange(7)[None, :] == gslot[:, None])
    inv = I.Inventory(jnp.where(put, grant_row[:, None], inv.item).astype(jnp.int32),
                      jnp.where(put, 1, inv.stack).astype(jnp.int32))
    lock = jnp.maximum(champ.cast_lock_until, jnp.where(kit_all.cast_started, now + kit_all.cast_lockout, 0.0))
    last_dmg = jnp.where(took_health[:c], now, champ.last_damaged)
    champ = champ._replace(
        cooldowns=cds, mana=mana, cast_lock_until=lock, last_damaged=last_dmg, inventory=inv,
        dyn=out.dynamic_stats, homeguard_ms=eco.homeguard_ms, blinked=s_out.blinked | dstart,
        forbid=ro.forbid_purchase, reset_next=out.effects.attack_reset,
        bonus_points=champ.bonus_points + ro.skill_points, granted=jnp.where(can_grant, ro.grant_item, 0),
        moving=jnp.where(respawn | recall | champ_dead, False, champ.moving),
        attack_order=jnp.where(respawn | recall | champ_dead, -1, champ.attack_order))
    kills_next = eco.kills
    # Dead minions free their slot; dead champions keep theirs.
    kind = jnp.where(minion_died, W.KIND_NONE, s.kind)
    cc = cc._replace(**{f: jnp.where(died, 0.0, getattr(cc, f)) for f in M.CCTimers._fields})
    events = TickEvents(out.report, out.follow_up, eco, plates, launched, out.packet_overflow, m_over, shop_code)
    new = s._replace(
        t=now, tick=s.tick + 1, key=key, kind=kind, alive=alive, x=x, y=y, hp=jnp.where(alive, hp, jnp.minimum(hp, 0.0)),
        max_hp=max_hp, att=att, missiles=missiles, cc=cc, champ=champ, kits=kits, summoners=summ,
        combat=out.state, econ=econ, lane_ai=lane_ai, towers=towers, shields=shields, status=status,
        kills=kills_next, damage_matrix=dmg, deaths_prev=died)
    # Keep the carry stable under scan: subsystems may return wider/narrower dtypes.
    return jax.tree.map(lambda a, b: jnp.asarray(b, a.dtype) if hasattr(a, "dtype") else b, s0, new), events


def IE_own(inv) -> Any:
    return I.owned_counts(inv)


def decimal_team_level(s: ModernState) -> Any:
    """(2,) decimal champion level per team (one champion per team in the lane world)."""
    return E.decimal_level(s.econ.xp, jnp.where(s.econ.quest.complete, 20, 18))[jnp.asarray([0, 1])]
