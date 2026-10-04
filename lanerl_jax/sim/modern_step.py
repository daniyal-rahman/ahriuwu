"""Patch-26.19 modern world tick: every modern system composed in one step.

``step(state, orders, cfg)`` advances the modern Summoner's Rift world (all three lanes,
jungle, objectives, fog) by ``cfg.dt`` (30 Hz default, DAMAGE_AND_STATS §1.3). Subsystems
are pure: they read views (``units_view``, ``KitCtx``, item ``Ctx``) and return packets, CC,
effects and slot writes; only this module writes ``ModernState``.

  modern_world        static config and unit layout (``modern_world.layout()``, 216 slots)
  modern_mechanics    attack machine, missiles, CC timers, route movement, blinks
  modern_lane_ai      minion/turret targeting, their attack packets, spawn stats, plates
  modern_jungle       camps, Scuttle, Smite, pets      modern_objectives  epic monsters, team buffs
  modern_champions    Garen/Jax kits                   modern_summoners   summoner spells
  modern_combat       items + runes around the damage pipeline (modern_damage)
  modern_economy      gold, XP, levels, bounty, respawn, recall, Homeguard, top quest
  modern_inventory    shop, grants, transforms         modern_wards       wards and trinkets
  modern_vision       fog of war                       modern_dynamic_terrain(_rift)  pads, Rift variants

Tick order: ``step`` runs one phase function per row, passing a ``TickScratch`` (the tick's data-flow
map; docs/modern/WORLD_IMPLEMENTATION.md has the full table and the one-tick lags):
  1. INPUT     _input       fog-filtered orders, attack-move, skill points, spawns, this tick's terrain
  2. STATS     _stats       static stats (items, shards, monster buffs, kit); STAT.70 max-HP sync
  2b. OBJ      _objectives  epic monsters: spawns, abilities, Rift transformation
  3. CASTS     _casts       shop; kit casts and periodic effects; summoner spells; Smite
  4. AI        _ai          turret/minion/monster targets; champion orders, idle acquisition, attack-move
  5. MOVE      _move        route movement, dashes, Flash, Teleport; collision
  6. ATTACK    _attack      attack machine, crits, attack packets, missiles, kit on-attack/on-hit
  7. DAMAGE    _damage      every packet -> combat_tick (items, runes, damage pipeline)
  8. CC/HEAL   _cc_heal     heals, shields and CC (tenacity, slow resist)
  9. DEATH     _death       kills, plates, camp/objective rewards -> economy_step
 10. TIMERS    _timers      cooldowns, mana/HP regen, fountain, inventory outputs, wards, terrain ejection
 11. FOG       _fog         attack reveal, then next tick's visibility
     COMMIT    _commit      next state and TickEvents; ``step`` then applies the game-over freeze and dtypes

Observations and actions live in ``obs/modern_builder.py`` and ``train/modern_actions.py``.
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
from . import modern_vision as MV
from .modern_stat_pipeline import ChampionStats, compose, cooldown, sync_max_health
from .modern_world import MAX_MINIONS, N_CHAMPIONS, WorldConfig, unit_ranges
from .modern_world import layout as MW_layout

RECALL_MS_LOCK = True
MOVE_ARRIVE_RADIUS = 5.0     # a move order is complete within this distance of its goal
CAST_BUFFER_S = 0.5          # casts during a lockout / just before a cooldown ends are held this long (U: guess)
CAST_RANGE_SLACK = 5.0       # walk-in casting stops this far inside the spell's range


class QueuedCast(NamedTuple):
    """(C,) a cast held for later (MECHANICS_AUDIT #4/#9): an out-of-range unit-targeted cast the
    champion walks into range for (until a new order), or a cast made during a lockout or within
    ``CAST_BUFFER_S`` of its cooldown ending (fires when allowed, expires after the buffer)."""
    slot: Any               # int32, -1 none
    target: Any             # int32 unit, -1 none
    x: Any
    y: Any
    until: Any              # expiry time (inf: walk-in, until replaced)


def no_queued_cast(c: int) -> QueuedCast:
    z = jnp.zeros((c,), jnp.float32)
    return QueuedCast(jnp.full((c,), -1, jnp.int32), jnp.full((c,), -1, jnp.int32), z, z, z)
CAST_ID_STRIDE = 256        # attack cast ids tick*256+unit+1 stay < 2^30 (K.KIT_ID_BASE) for 2^22 ticks (~38.8 h)
SKILL_ORDERS = {86: (2, 0, 1, 2, 2, 3, 2, 0, 2, 0, 3, 0, 0, 1, 1, 3, 1, 1),     # Garen E>Q>W (modern.skill_ranks)
                24: (2, 0, 1, 1, 1, 3, 1, 2, 1, 2, 3, 2, 2, 0, 0, 3, 0, 0)}     # Jax W>E>Q


class ChampionLayer(NamedTuple):
    """Per-champion world state (C,)."""
    ranks: Any              # (C, 4) int32
    cooldowns: Any          # (C, 4) remaining seconds
    mana: Any
    cast_lock_until: Any    # kit casts and item actives that root the caster: no casts, attacks or movement
    item_cast_until: Any    # item-active cast that allows movement (Stridebreaker): no casts or attacks
    move_goal: Any          # (C, 2)
    moving: Any             # bool: walk to move_goal
    attack_order: Any       # int32 unit, -1 none
    target_seen_at: Any     # (C, 2) where the team last saw the attack target (walked to if it enters fog)
    queued_cast: Any        # QueuedCast: walk-in / buffered cast
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
    last_cast: Any          # (C, 4) game time each Q/W/E/R was last cast (-1e9 never); observed-cast memory
    cs: Any                 # (C,) int32 minions last-hit (creep score)
    seen_cast: Any          # (C, 4) game time each Q/W/E/R was last cast while visible to the enemy team


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
    spawn: Any              # modern_minions.LaneSpawnState: per (team, lane) wave cursors
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
    death_seen: Any         # (C, N) bool: champion i had own sight of unit j when j died last tick (Overgrowth)
    visible: Any            # (2, N) bool: team t sees unit j (modern_vision, end of last tick)
    sight: Any              # (N, N) bool: unit i's own sight of unit j (rune "own sight")
    reveal: Any             # modern_vision.Reveal: attack-reveal circle per champion
    jungle: Any             # modern_jungle.JungleState (camps, Smite, pets, red/blue)
    obj: Any                # modern_objectives.ObjectiveState (grubs, Herald, drakes, Elder, Baron)
    wards: Any              # modern_wards.Wards (ward slots, trinkets)
    amove: Any              # AttackMove: per-champion attack-move order state
    pending_dash: Any       # W.Dash: item-active dash (Rocketbelt) to start next tick
    route_anchor: Any       # (N,) int32 route node each unit steers by (modern_mechanics.move_step); -1 = none
    epic_prev: Any          # (C,) epic takedowns last tick (rune events)
    large_prev: Any         # (C,) large-monster kills last tick (rune events)
    terrain_variant: Any    # () int32 Elemental Rift x Baron-pit terrain variant
    game_over: Any          # () bool: a Nexus fell (the world freezes)
    winner: Any             # () int32 0 blue / 1 red / -1 none


class AttackMove(NamedTuple):
    """Attack-move order per champion (C,)."""
    active: Any
    x: Any
    y: Any
    held: Any               # int32 acquired unit, -1 none
    held_seq: Any


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
    attack_move: Any = None # bool: attack-move to (move_x, move_y)
    ward_kind: Any = None   # int32 -1 none, 0 trinket (place / Oracle sweep), 1 Control Ward
    ward_x: Any = None
    ward_y: Any = None


def no_orders(c: int = N_CHAMPIONS) -> ModernOrders:
    z, f, i = jnp.zeros((c,), jnp.float32), jnp.zeros((c,), bool), jnp.zeros((c,), jnp.int32)
    return ModernOrders(f, z, z, i - 1, f, i - 1, i - 1, z, z, i - 1, i - 1, z, z, i, i, i, f, i - 1,
                        f, i - 1, z, z)


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
    structure = W.is_structure(kind)
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
        item_cast_until=zc,
        move_goal=cfg.fountain, moving=jnp.zeros((c,), bool), attack_order=jnp.full((c,), -1, jnp.int32),
        target_seen_at=cfg.fountain, queued_cast=no_queued_cast(c),
        facing=jnp.tile(jnp.asarray([[1.0, 1.0]], jnp.float32), (c, 1)) * jnp.asarray([[1.0], [-1.0]]),
        dash_until=zc - 1.0, dash_target=jnp.full((c,), -1, jnp.int32), dash_to=cfg.fountain, dash_speed=zc,
        last_damaged=zc - 1e9, inventory=inv,
        group_cd=jnp.zeros((c, catalog().arrays.groups.shape[1]), jnp.float32),
        dyn=zero_stats((c,)), static_max_hp=st.max_hp, homeguard_ms=zc, blinked=jnp.zeros((c,), bool),
        forbid=jnp.zeros((c, len(catalog().ids)), bool), reset_next=jnp.zeros((c,), bool),
        bonus_points=jnp.zeros((c,), jnp.int32), granted=jnp.zeros((c,), jnp.int32),
        last_cast=jnp.full((c, 4), -1e9, jnp.float32), cs=jnp.zeros((c,), jnp.int32),
        seen_cast=jnp.full((c, 4), -1e9, jnp.float32))
    f = lambda v: jnp.asarray(v, jnp.float32)
    radius = jnp.where(kind == W.KIND_CHAMPION, 65.0, jnp.where(structure, towers.radius, 48.0)).astype(jnp.float32)
    alive = (kind == W.KIND_CHAMPION) | structure
    reveal = MV.init_reveal(c)
    from . import modern_jungle as J
    from . import modern_minions as MM
    from . import modern_objectives as OBJ
    from . import modern_wards as WD
    trinkets = [next((i for i in lo.items if i in (3340, 3363, 3364)), 3340) for lo in cfg.loadouts]
    wards = WD.init_wards(c, trinkets)
    jungle = J.init_jungle(cfg.jungle, c, n, seed=seed) if cfg.jungle is not None else None
    obj = OBJ.init_objectives(cfg.objectives, n, c, jax.random.PRNGKey(seed + 1)) if cfg.objectives is not None \
        else None
    zc0 = jnp.zeros((c,), jnp.float32)
    amove = AttackMove(jnp.zeros((c,), bool), zc0, zc0, jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), jnp.int32))
    vis, sight = _visibility(cfg, cfg.unit_x, cfg.unit_y, kind, cfg.unit_sub, cfg.unit_team, alive, reveal,
                             jnp.float32(0.0), wards=wards)
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
        m_level=jnp.ones((n,), jnp.int32), next_seq=jnp.int32(n), spawn=MM.init_lane_spawn(),
        att=W.init_attack_state(n), missiles=M.init_missiles(64), cc=M.init_cc(n), champ=champ,
        kits=K.init(c, n), summoners=S.init(loadout), combat=_init_combat(cfg, c, n),
        econ=E.init_economy(c, n, [lo.role for lo in cfg.loadouts]), lane_ai=LA.init_lane_ai(n), towers=towers,
        shields=D.init_shields(n), status=init_status(n),
        kills=Kills(zc, zc, zc, jnp.zeros((c,), bool), jnp.zeros((c, n), bool)),
        damage_matrix=jnp.zeros((n, n), bool), death_seen=jnp.zeros((c, n), bool),
        visible=vis, sight=sight, reveal=reveal, jungle=jungle, obj=obj, wards=wards, amove=amove,
        pending_dash=W.no_dash(c), route_anchor=M.init_route_anchor(n), epic_prev=zc0, large_prev=zc0, terrain_variant=jnp.int32(0),
        game_over=jnp.asarray(False), winner=jnp.int32(-1))


def _visibility(cfg: WorldConfig, x, y, kind, sub, team, alive, reveal, now, *, wards=None, level=None,
                variant=None, jungle=None):
    """``(visible (2, N), sight (N, N))``; everything live is visible when ``cfg.vision`` is None.

    Wards (``modern_wards``) add sight, stealth and true sight; ``variant`` selects the Elemental
    Rift / Baron-pit brush layout."""
    n = x.shape[0]
    if cfg.vision is None:
        live = alive & (kind != W.KIND_NONE)
        return jnp.broadcast_to(live[None, :], (2, n)), jnp.broadcast_to(live[None, :], (n, n))
    grid = cfg.vision
    if cfg.rift is not None and variant is not None:
        from .modern_dynamic_terrain_rift import vision_for
        grid = vision_for(cfg.rift, variant, cfg.vision)
    kw = {}
    lay = MW_layout()
    if wards is not None:
        from . import modern_wards as WD
        c = N_CHAMPIONS
        lv = jnp.ones((c,), jnp.int32) if level is None else level
        view, oracle = WD.ward_view(wards, now=now, x=x[:c], y=y[:c], team=team[:c], alive=alive[:c], level=lv)
        kw = WD.vision_kwargs(view, oracle, kind, sub, alive, ward_start=lay["ward0"])
    if jungle is not None:                                            # Scuttle Speed Shrines (525 sight)
        from . import modern_jungle as J
        on = jungle.shrine_until > now
        kw["sources"] = (jnp.asarray(J.SHRINE_POS, jnp.float32)[:, 0], jnp.asarray(J.SHRINE_POS, jnp.float32)[:, 1],
                         jnp.where(on, J.SHRINE_SIGHT, 0.0), jungle.shrine_team)
    return MV.visibility(x, y, kind, sub, team, alive, reveal, now, grid, n_fogged=lay["struct0"], **kw)


def _kit_attack(kit_out) -> Any:
    """(C,) unit a kit asks the champion to attack (Jax Q landing on a champion), -1 none."""
    at = kit_out.attack_target
    return jnp.full((N_CHAMPIONS,), -1, jnp.int32) if at is None else at


def refresh_visibility(s: ModernState, cfg: WorldConfig) -> ModernState:
    """Recompute ``visible``/``sight`` from the current positions (after editing a state by hand,
    e.g. scenario resets); ``step`` does this itself at the end of every tick."""
    vis, sight = _visibility(cfg, s.x, s.y, s.kind, s.sub, s.team, s.alive, s.reveal, s.t, wards=s.wards,
                             level=s.econ.level, variant=s.terrain_variant, jungle=s.jungle)
    return s._replace(visible=vis, sight=sight)


def _monster_buff_stats(s: ModernState, cfg: WorldConfig, st0: ChampionStats, now) -> ItemStats:
    """Static bonuses from jungle buffs (Blue/Red/Scuttle shrine) and team objective buffs
    (drake stacks and soul, Hand of Baron)."""
    c = N_CHAMPIONS
    parts = []
    if cfg.jungle is not None:
        from . import modern_jungle as J
        b = J.buff_stats(s.jungle, now=now, level=s.econ.level, max_mana=st0.max_mana, max_hp=s.max_hp[:c],
                         x=s.x[:c], y=s.y[:c], team=s.team[:c],
                         champion_combat_recent=(now - s.combat.clocks.last_champion_combat) < 5.0)
        parts.append(zero_stats((c,))._replace(ability_haste=b.ability_haste, mana_regen=b.mana_per_s,
                                               health_regen=b.hp_regen_per_s, move_speed=b.shrine_ms))
    if cfg.objectives is not None:
        from . import modern_objectives as OBJ
        parts.append(OBJ.team_buff_stats(s.obj, cfg.objectives, s.team[:c], s.alive[:c],
                                         st0.base_ad + st0.bonus_ad, st0.ap, now=now,
                                         out_of_combat=(now - s.combat.clocks.last_combat) >= 5.0))
    out = zero_stats((c,))
    for p in parts:
        out = combine_stats(out, p)
    return out


def _apply_objective_writes(s: ModernState, so, cfg: WorldConfig) -> ModernState:
    """Write the epic slots from ``objectives_step`` (spawns, despawns, relocations, heals)."""
    w = so.writes
    e0 = cfg.objectives.slot0
    sl = slice(e0, e0 + 8)
    n = s.kind.shape[0]

    def put(arr, v, mask=w.write):
        seg = arr[sl]
        return arr.at[sl].set(jnp.where(mask, jnp.asarray(v, arr.dtype), seg))
    new = w.write & w.new_seq
    seq = s.next_seq + jnp.cumsum(new.astype(jnp.int32)) - 1
    s = s._replace(
        kind=put(s.kind, w.kind), sub=put(s.sub, w.sub), team=put(s.team, w.team), alive=put(s.alive, True),
        targetable=put(s.targetable, True), x=put(s.x, w.x), y=put(s.y, w.y), hp=put(s.hp, w.hp),
        max_hp=put(s.max_hp, w.max_hp), armor=put(s.armor, w.armor), mr=put(s.mr, w.mr), ad=put(s.ad, w.ad),
        arange=put(s.arange, w.arange), aspeed=put(s.aspeed, w.aspeed), mspeed=put(s.mspeed, w.mspeed),
        radius=put(s.radius, w.radius), windup=put(s.windup, w.windup),
        missile_speed=put(s.missile_speed, w.missile_speed),
        spawn_seq=put(s.spawn_seq, seq, new), spawn_time=put(s.spawn_time, s.t, new),
        next_seq=s.next_seq + jnp.sum(new.astype(jnp.int32)))
    s = _reset_slots(s, jnp.zeros((n,), bool).at[sl].set(new))
    s = s._replace(kind=put(s.kind, W.KIND_NONE, w.despawn), alive=put(s.alive, False, w.despawn),
                   x=put(s.x, w.rx, w.relocate), y=put(s.y, w.ry, w.relocate))
    hp = s.hp.at[sl].set(jnp.minimum(s.hp[sl] + so.monster_heal, s.max_hp[sl]))
    return s._replace(hp=jnp.where(s.alive, hp, s.hp), terrain_variant=jnp.asarray(so.terrain_variant, jnp.int32))


def _in_brush(cfg: WorldConfig, x, y, variant) -> Any:
    """(C,) bool: the position is a brush cell (vision-grid flag bit 0x1, Rift variant aware); False
    without fog (``cfg.vision`` None: no grid in the config)."""
    if cfg.vision is None:
        return jnp.zeros(x.shape, bool)
    grid = cfg.vision
    if cfg.rift is not None:
        from .modern_dynamic_terrain_rift import vision_for
        grid = vision_for(cfg.rift, variant, cfg.vision)
    h, w = grid.flags.shape
    ix = jnp.floor((x - grid.min_x) / grid.cell_size).astype(jnp.int32)
    iy = jnp.floor((y - grid.min_y) / grid.cell_size).astype(jnp.int32)
    ok = (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h)
    return ok & ((grid.flags[jnp.clip(iy, 0, h - 1), jnp.clip(ix, 0, w - 1)] & 1) != 0)


def _in_river(cfg: WorldConfig, x, y) -> Any:
    if cfg.regions is None:
        return jnp.zeros(x.shape, bool)
    from . import modern_map_regions as REG
    return REG.in_river(x, y, cfg.regions)


def _init_combat(cfg: WorldConfig, c: int, n: int) -> CombatState:
    """``init_combat`` with the jungle items' epic-monster mask (the 8 objective slots)."""
    comb = init_combat(c, n)
    lay = MW_layout()
    epic = jnp.zeros((n,), bool).at[lay["epic0"]:lay["ward0"]].set(cfg.objectives is not None)
    items = comb.items
    return comb._replace(items=items._replace(jungle=items.jungle._replace(epic=epic)))


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

def _static_stats(s: ModernState, cfg: WorldConfig, caps: dict, now, dt) -> tuple[ItemStats, ChampionStats]:
    """``(static, composed)``: items + shards + monster buffs + kit stats, without dynamic stats or slows."""
    level = s.econ.level
    base_static = combine_stats(I.inventory_stats(s.champ.inventory), _shard_stats(cfg, level))
    st0 = compose(cfg.champion_base, level, base_static, adaptive_physical=cfg.adaptive_physical)
    base_static = combine_stats(base_static, _monster_buff_stats(s, cfg, st0, now))
    st0 = compose(cfg.champion_base, level, base_static, adaptive_physical=cfg.adaptive_physical)
    static = combine_stats(base_static, K.stats(s.kits, _kit_ctx(s, cfg, st0, caps, now, dt)))
    return static, compose(cfg.champion_base, level, static, adaptive_physical=cfg.adaptive_physical)


def champion_stats(s: ModernState, cfg: WorldConfig) -> ChampionStats:
    """Displayed champion stats for the current state (items, shards, monster buffs, kit, last
    tick's dynamic stats and slows): the stats the tick uses for casts (observation helper)."""
    caps = M.capabilities(s.cc, s.t)
    static, _ = _static_stats(s, cfg, caps, s.t, cfg.dt)
    return compose(cfg.champion_base, s.econ.level, combine_stats(static, s.champ.dyn),
                   adaptive_physical=cfg.adaptive_physical, slow=caps["slow"][:N_CHAMPIONS])


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
                    in_combat_ms_since_damaged=now - s.champ.last_damaged,
                    attack_target_kind=jnp.where(s.champ.attack_order >= 0,
                                                 s.kind[jnp.clip(s.champ.attack_order, 0, s.kind.shape[0] - 1)],
                                                 W.KIND_NONE).astype(jnp.int32),
                    rooted=s.cc.root_until[:c] > now)


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


def _reset_slots(s: ModernState, mask) -> ModernState:
    """Fresh attack state and CC timers for (re)spawned slots."""
    put = lambda arr, v: jnp.where(mask, jnp.asarray(v, arr.dtype), arr)   # noqa: E731
    return s._replace(att=s.att._replace(target=put(s.att.target, -1), windup_left=put(s.att.windup_left, 0.0),
                                         cooldown_left=put(s.att.cooldown_left, 0.0)),
                      cc=M.CCTimers(*(put(v, 0.0) for v in s.cc)))


def _spawn_minions(s: ModernState, cfg: WorldConfig, now) -> ModernState:
    """MINIONS §2 for every enabled lane (``modern_lane_ai.spawn_lane_minions``)."""
    spawn, wr, _ = LA.spawn_lane_minions(s.spawn, s.towers, s.kind, s.alive, now=now, slot0=MW_layout()["minion0"],
                                         per_lane=W.MAX_MINIONS_PER_LANE, lanes=cfg.lanes)
    pick, st = wr.pick, wr.stats
    put = lambda arr, v: jnp.where(pick, jnp.asarray(v, arr.dtype), arr)   # noqa: E731
    s = s._replace(
        kind=put(s.kind, W.KIND_MINION), sub=put(s.sub, wr.sub), team=put(s.team, wr.team),
        alive=put(s.alive, True), targetable=put(s.targetable, True), x=put(s.x, wr.x), y=put(s.y, wr.y),
        hp=put(s.hp, st.max_hp), max_hp=put(s.max_hp, st.max_hp), radius=put(s.radius, st.radius),
        armor=put(s.armor, st.armor), mr=put(s.mr, st.magic_resist), ad=put(s.ad, st.attack_damage),
        arange=put(s.arange, st.attack_range), aspeed=put(s.aspeed, st.attack_speed),
        mspeed=put(s.mspeed, st.move_speed), windup=put(s.windup, st.windup),
        missile_speed=put(s.missile_speed, st.missile_speed),
        spawn_seq=put(s.spawn_seq, s.next_seq + wr.seq_offset), spawn_time=put(s.spawn_time, now),
        m_gold=put(s.m_gold, st.gold), m_xp=put(s.m_xp, st.xp), m_level=put(s.m_level, jnp.max(s.econ.level)),
        lane_wp=put(s.lane_wp, 0), next_seq=s.next_seq + wr.count, spawn=spawn)
    s = _reset_slots(s, pick)
    if cfg.jungle is not None:
        from . import modern_jungle as J
        jst, w = J.spawn_step(s.jungle, cfg.jungle, now=now, champion_level=s.econ.level)
        names = ("kind", "sub", "team", "alive", "targetable", "x", "y", "hp", "max_hp", "radius", "armor", "mr",
                 "ad", "arange", "aspeed", "mspeed", "windup", "missile_speed", "spawn_time", "spawn_seq")
        arrays, nseq = J.write_spawns(cfg.jungle, w, now=now, next_seq=s.next_seq,
                                      **{k: getattr(s, k) for k in names})
        m0 = cfg.jungle.monster0
        mask = jnp.zeros((s.kind.shape[0],), bool).at[m0:m0 + cfg.jungle.n_slots].set(w.mask)
        s = _reset_slots(s._replace(jungle=J.latch_pets(jst, IE_own(s.champ.inventory)), next_seq=nseq, **arrays),
                         mask)
    return s


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
                 radius=s.radius, targetable=s.targetable & s.alive & (s.kind != W.KIND_WARD),
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
    auto_on = jnp.asarray([lo.auto_skill for lo in cfg.loadouts])
    slot = jnp.where(orders.level_up >= 0, orders.level_up, auto)
    cap = jnp.where(slot == 3, E.max_rank(level, ultimate=True), E.max_rank(level))
    slot = jnp.clip(slot, 0, 3)
    cur = jnp.take_along_axis(ranks, slot[:, None], axis=1)[:, 0]
    ok = (points > spent) & (cur < cap) & ((orders.level_up >= 0) | auto_on)
    return ranks + (jnp.arange(4)[None, :] == slot[:, None]) * ok[:, None]


class TickScratch(NamedTuple):
    """Values passed between the phases of one ``step``: the tick's data-flow map.

    Each comment names the phase that writes the field (later rewrites in brackets). A field exists when
    a later phase needs a value that is not in ``s``, or that must keep its earlier value while ``s`` moves
    on. ``s`` itself is written only where the phase comments say so; ``_commit`` builds the next state."""
    # step (before INPUT)
    s0: Any = None              # step: the incoming state (start-of-tick visibility for wards/reveal; freeze)
    now: Any = None             # step: s.t + dt
    key: Any = None             # step [OBJ]: PRNG key carried to the next tick
    k_crit: Any = None          # step: crit-roll key (ATTACK)
    # 1. INPUT
    champ: Any = None           # INPUT [CASTS from s.champ, AI, DEATH, TIMERS, FOG]: working ChampionLayer; the
                                # next state's champ. STATS and MOVE write s.champ only (see _move).
    vis_c: Any = None           # INPUT: (C, N) the champion's team sees the unit (start of tick)
    in_stasis: Any = None       # INPUT: (C,) item stasis (Zhonya's / Stopwatch)
    caps: Any = None            # INPUT: M.capabilities at now, stasis-gated
    terrain: Any = None         # INPUT: this tick's walkable masks (Rift variant, then structure pads)
    attack_order: Any = None    # INPUT [AI: + idle acquisition]: (C,) ordered attack target
    moving: Any = None          # INPUT: (C,) walking to move_goal
    goal: Any = None            # INPUT [AI: attack-move goal]: (C, 2) champion move goal
    amove: Any = None           # INPUT [AI]: AttackMove
    # 2. STATS
    static: Any = None          # STATS: ItemStats items + shards + monster buffs + kit (no dynamic stats)
    st_static: Any = None       # STATS: ChampionStats composed from ``static``
    # 2b. OBJECTIVES
    so: Any = None              # OBJ: objectives_step output (None without objectives)
    cinfo: Any = None           # OBJ: OBJ.ChampInfo (reused by DEATH)
    # 3. CASTS
    st: Any = None              # CASTS: ChampionStats with last tick's dynamic stats and slows (the tick's stats)
    summ_world: Any = None      # CASTS: ItemStats static + last tick's dynamic stats
    units: Any = None           # CASTS [AI, MOVE: champion range/AS, ATTACK: Hand of Baron]: WorldUnits view
    kctx: Any = None            # CASTS: K.KitCtx
    ictx: Any = None            # CASTS [MOVE: positions, facing, moved]: item Ctx
    locked: Any = None          # CASTS: (C,) cast lockout (no casts, attacks or movement)
    item_casting: Any = None    # CASTS: (C,) item-active cast that allows movement (Stridebreaker)
    cast_order: Any = None      # CASTS: W.CastOrder the kits saw (gated by can_cast)
    kits: Any = None            # CASTS [ATTACK, DAMAGE, DEATH]: kit state
    kit_out: Any = None         # CASTS: merged kit cast + periodic output
    kmods: Any = None           # CASTS: K.attack_mods
    reach: Any = None           # CASTS: (C,) attack range incl. kit extra range
    summ: Any = None            # CASTS: summoner state
    s_eff: Any = None           # CASTS: summoner Effects (packets, heals, gold)
    s_out: Any = None           # CASTS: summoner world outputs (Flash/Teleport, Exhaust, Ghost, Cleanse ...)
    sm: Any = None              # CASTS: Smite output (None without jungle)
    jungle: Any = None          # CASTS [AI, ATTACK, DEATH]: working JungleState (s.jungle stays stale)
    shop_code: Any = None       # CASTS: (C,) buy/sell result code
    bought: Any = None          # CASTS: (C,) item id bought this tick
    sold: Any = None            # CASTS: (C,) item id sold this tick
    # 4. AI
    towers: Any = None          # AI [ATTACK: Overgrowth, DEATH: plates]: working structure state
    lane_ai: Any = None         # AI: lane AI state
    desired: Any = None         # AI: (N,) desired attack target per unit
    lane_goal: Any = None       # AI: (N, 2) lane-minion movement goals
    lane_stop: Any = None       # AI: (N,) lane minions holding position
    mai: Any = None             # AI: jungle monster AI output (None without jungle)
    mon_goal: Any = None        # AI: (N, 2) monster movement goals
    mon_speed: Any = None       # AI: (N,) monster move speed
    mon_move: Any = None        # AI: (N,) monster moving
    mbuf: Any = None            # AI: Hand of Baron minion buffs (None without objectives)
    attack_order_eff: Any = None  # AI: (C,) attack target incl. attack-move acquisition
    t_ok_eff: Any = None        # AI: (C,) champion has a live attack target
    moving_eff: Any = None      # AI: (C,) champion walks to ``goal``
    # 5. MOVE
    tp_lock: Any = None         # MOVE: (C,) Teleport channel/dash (no movement or attacks)
    sres: Any = None            # MOVE: (N,) slow resist
    ms: Any = None              # MOVE: (N,) move speed used this tick
    dstart: Any = None          # MOVE: (C,) dash started this tick (kit or Rocketbelt)
    in_dash: Any = None         # MOVE: (C,) dashing
    # 6. ATTACK
    att: Any = None             # ATTACK: AttackState after the attack machine
    launched: Any = None        # ATTACK: (N,) attacks launched
    started: Any = None         # ATTACK: (N,) windups started
    cancelled: Any = None       # ATTACK: (N,) windups cancelled
    reset: Any = None           # ATTACK: (N,) attack resets
    attack: Any = None          # ATTACK: champion Attack (incl. missile arrivals as hits)
    missiles: Any = None        # ATTACK: Missiles after spawn/advance
    m_over: Any = None          # ATTACK: missile overflow
    direct: Any = None          # ATTACK: melee attack packets
    arrived: Any = None         # ATTACK: missile arrival packets
    og_pk: Any = None           # ATTACK: Crystalline Overgrowth packets
    extra_atk: Any = None       # ATTACK: list of extra monster attack packets (jungle bonus, objectives)
    obj_cc: Any = None          # ATTACK: objective attack CC (None without objectives)
    ward_hits: Any = None       # ATTACK: (2W,) champion hits per ward slot
    ward_hitter: Any = None     # ATTACK: (2W,) hitting unit per ward slot
    kit_all: Any = None         # ATTACK: kit output merged with on_attack/on_hit
    jfx: Any = None             # ATTACK: jungle combat effects (None without jungle)
    # 7. DAMAGE
    out: Any = None             # DAMAGE: combat_tick output
    kdef: Any = None            # DAMAGE: K.defense
    k_dmg: Any = None           # DAMAGE: K.on_damage output
    cc_now: Any = None          # DAMAGE: kit CC this tick
    cc_items: Any = None        # DAMAGE: item CC flags (slowed, hard CC)
    hp: Any = None              # DAMAGE [CC, DEATH, TIMERS]: (N,) working HP
    max_hp: Any = None          # DAMAGE [TIMERS: ward rows]: (N,) working max HP
    shields: Any = None         # DAMAGE [CC]: D.Shields
    status: Any = None          # DAMAGE [CC]: UnitStatus
    # 8. CC / HEAL
    extra: Any = None           # CC: merged summoner/kit/monster heal and mana effects
    item_cleanse: Any = None    # CC: item-active world effects (cleanse, cd rate, mana cost, transform, dash)
    cc: Any = None              # CC: CCTimers after this tick's CC (deaths cleared in _commit)
    # 9. DEATH
    died: Any = None            # DEATH: (N,) died this tick
    minion_died: Any = None     # DEATH: (N,)
    dmg: Any = None             # DEATH: (N, N) damage matrix for the next tick
    death_seen: Any = None      # DEATH: (C, N) own sight of units that died (Overgrowth, next tick)
    plates: Any = None          # DEATH: PlateEvents
    took_health: Any = None     # DEATH: (N,) took health damage
    in_f: Any = None            # DEATH: (C,) in own fountain
    epic: Any = None            # DEATH: (C,) epic takedowns (next tick's rune events)
    large: Any = None           # DEATH: (C,) large-monster kills (next tick's rune events)
    jrw: Any = None             # DEATH: jungle rewards (None without jungle)
    eco: Any = None             # DEATH: EconomyOut
    econ: Any = None            # DEATH [TIMERS: ward gold/XP]: next economy state
    # 10. TIMERS (next-state unit columns and layers)
    alive: Any = None           # TIMERS: (N,)
    x: Any = None               # TIMERS: (N,) final positions
    y: Any = None
    kind: Any = None            # TIMERS: (N,) dead minions freed, ward units mirrored
    sub: Any = None             # TIMERS: (N,) ward rows
    team: Any = None            # TIMERS: (N,) ward rows
    spawn_seq: Any = None       # TIMERS: (N,) ward rows
    radius: Any = None          # TIMERS: (N,) ward rows
    targetable: Any = None      # TIMERS: (N,) ward rows
    wards: Any = None           # TIMERS: Wards
    kills: Any = None           # TIMERS: Kills credited this tick (next tick's hooks)
    cast_now: Any = None        # TIMERS: (C, 4) slot cast this tick
    # 11. FOG
    reveal: Any = None          # FOG: attack-reveal circles
    visible: Any = None         # FOG: (2, N) next tick's visibility
    sight: Any = None           # FOG: (N, N) next tick's own sight


def step(s: ModernState, orders: ModernOrders, cfg: WorldConfig) -> tuple[ModernState, TickEvents]:
    """Advance the modern world one tick (``cfg.dt``); returns ``(state, events)``."""
    s0 = s
    now = s.t + jnp.float32(cfg.dt)
    key, k_crit = jax.random.split(s.key)
    sc = TickScratch(s0=s0, now=now, key=key, k_crit=k_crit)
    s, orders, sc = _input(s, orders, cfg, sc)
    s, sc = _stats(s, orders, cfg, sc)
    s, sc = _objectives(s, orders, cfg, sc)
    s, sc = _casts(s, orders, cfg, sc)
    s, sc = _ai(s, orders, cfg, sc)
    s, sc = _move(s, orders, cfg, sc)
    s, sc = _attack(s, orders, cfg, sc)
    s, sc = _damage(s, orders, cfg, sc)
    s, sc = _cc_heal(s, orders, cfg, sc)
    s, sc = _death(s, orders, cfg, sc)
    s, sc = _timers(s, orders, cfg, sc)
    s, sc = _fog(s, orders, cfg, sc)
    new, events = _commit(s, cfg, sc)
    # Game over (Nexus destroyed): the world freezes on the final state.
    new = jax.tree.map(lambda a, b: jnp.where(s0.game_over, a, b), s0, new)
    # Keep the carry stable under scan: subsystems may return wider/narrower dtypes.
    return jax.tree.map(lambda a, b: jnp.asarray(b, a.dtype) if hasattr(a, "dtype") else b, s0, new), events


def _queue_casts(s: ModernState, cfg: WorldConfig, orders: ModernOrders, champ, seen, new_order, now):
    """Walk-in and buffered casting (wiki Targeting / Cast time; MECHANICS_AUDIT #4/#9).

    Returns ``(orders, queued, chase, walked_in)``: ``orders`` with this tick's cast (an incoming cast
    that can go now, else a queued one that became possible), the queue to keep, champions walking
    into range of a queued target, and champions whose walk-in cast fired this tick (ends the walk). A move, attack,
    stop or attack-move order clears the queue; so does a new cast (it replaces it)."""
    from . import modern_champions as KC
    c, n = N_CHAMPIONS, cfg.n_units
    ar = jnp.arange(c)
    rng = KC.unit_target_ranges(cfg.champion_ids)                                   # (C, 4)

    def needs(slot, target):
        """(far, blocked) for a cast of ``slot`` at ``target`` now."""
        sl = jnp.clip(slot, 0, 3)
        r = rng[ar, sl]
        t = jnp.clip(target, 0, n - 1)
        d = jnp.sqrt((s.x[:c] - s.x[t]) ** 2 + (s.y[:c] - s.y[t]) ** 2)
        far = (slot >= 0) & (target >= 0) & (r > 0) & (d > r + s.radius[t] - CAST_RANGE_SLACK)
        cd = champ.cooldowns[ar, sl]
        blocked = (slot >= 0) & ((now < champ.cast_lock_until) | (now < champ.item_cast_until)
                                 | ((cd > 0) & (cd <= CAST_BUFFER_S)))
        return far, blocked

    incoming = orders.cast_slot >= 0
    far_in, blocked_in = needs(orders.cast_slot, orders.cast_target)
    hold_in = incoming & (far_in | blocked_in)
    q = champ.queued_cast
    keep = (q.slot >= 0) & ~incoming & ~new_order & (now < q.until) \
        & ((q.target < 0) | (seen(q.target) & s.alive[jnp.clip(q.target, 0, n - 1)]))
    q = QueuedCast(*(jnp.where(keep, a, b) for a, b in zip(q, no_queued_cast(c))))
    q = QueuedCast(jnp.where(hold_in, orders.cast_slot, q.slot).astype(jnp.int32),
                   jnp.where(hold_in, orders.cast_target, q.target).astype(jnp.int32),
                   jnp.where(hold_in, orders.cast_x, q.x), jnp.where(hold_in, orders.cast_y, q.y),
                   jnp.where(hold_in, jnp.where(far_in, jnp.inf, now + CAST_BUFFER_S), q.until))
    far_q, blocked_q = needs(q.slot, q.target)
    fire = (q.slot >= 0) & ~hold_in & ~far_q & ~blocked_q
    go_now = incoming & ~hold_in
    orders = orders._replace(
        cast_slot=jnp.where(go_now, orders.cast_slot, jnp.where(fire, q.slot, -1)).astype(jnp.int32),
        cast_target=jnp.where(go_now, orders.cast_target, jnp.where(fire, q.target, orders.cast_target)).astype(jnp.int32),
        cast_x=jnp.where(go_now, orders.cast_x, jnp.where(fire, q.x, orders.cast_x)),
        cast_y=jnp.where(go_now, orders.cast_y, jnp.where(fire, q.y, orders.cast_y)))
    chase = (q.slot >= 0) & far_q & ~fire
    walked_in = fire & jnp.isinf(q.until)                       # a walk-in cast arrived and fired
    q = QueuedCast(*(jnp.where(fire, b, a) for a, b in zip(q, no_queued_cast(c))))
    return orders, q, chase, walked_in


def _input(s: ModernState, orders: ModernOrders, cfg: WorldConfig,
           sc: TickScratch) -> tuple[ModernState, ModernOrders, TickScratch]:
    """1. INPUT: fog-filtered orders, attack-move, stasis, skill points, spawns, this tick's terrain.

    Writes ``s.champ`` (orders, ranks), ``s.amove``, spawns, champion ``targetable``; returns the
    fog-filtered orders."""
    c, n = N_CHAMPIONS, cfg.n_units
    now = sc.now
    champ = s.champ
    # Fog: a champion can only target what its team sees; a target that enters fog is dropped.
    vis_c = s.visible[s.team[:c]]                                                     # (C, N)
    seen = lambda u: (u >= 0) & vis_c[jnp.arange(c), jnp.clip(u, 0, n - 1)]          # noqa: E731
    orders = orders._replace(attack=jnp.where(seen(orders.attack), orders.attack, -1),
                             cast_target=jnp.where(seen(orders.cast_target), orders.cast_target, -1),
                             summoner_target=jnp.where(seen(orders.summoner_target), orders.summoner_target, -1))
    zb = jnp.zeros((c,), bool)
    am_req = zb if orders.attack_move is None else orders.attack_move
    # Item stasis (Zhonya's / Stopwatch): no orders take effect while in stasis (modern_item_actives).
    in_stasis = now < s.combat.items.modern_item_actives.stasis_until
    am_req = am_req & ~in_stasis
    kept = jnp.where(seen(champ.attack_order), champ.attack_order, -1)
    new_order = orders.stop | orders.move | (orders.attack >= 0) | am_req
    attack_order = jnp.where(orders.stop | orders.move | am_req, -1, jnp.where(orders.attack >= 0, orders.attack, kept))
    moving = jnp.where(orders.stop | (orders.attack >= 0) | am_req, False, orders.move | champ.moving)
    goal = jnp.where(orders.move[:, None], jnp.stack([orders.move_x, orders.move_y], -1), champ.move_goal)
    # A live attack target that enters fog: walk to where it was last seen (wiki Basic attack;
    # MECHANICS_AUDIT #10) instead of standing still. The order itself is dropped.
    t_old = jnp.clip(champ.attack_order, 0, n - 1)
    lost = (champ.attack_order >= 0) & ~seen(champ.attack_order) & s.alive[t_old] & ~new_order
    # Remember where this tick's target is seen (including a target just ordered), for later ticks.
    t_now = jnp.clip(attack_order, 0, n - 1)
    seen_at = jnp.where(seen(attack_order)[:, None], jnp.stack([s.x[t_now], s.y[t_now]], -1),
                        champ.target_seen_at)
    moving = moving | lost
    goal = jnp.where(lost[:, None], champ.target_seen_at, goal)
    orders, queued, cast_chase, cast_fired = _queue_casts(s, cfg, orders, champ, seen, new_order, now)
    moving = jnp.where(cast_chase, True, jnp.where(cast_fired, False, moving))
    attack_order = jnp.where(cast_chase, -1, attack_order)          # "move to cast" replaces an attack order
    t_q = jnp.clip(queued.target, 0, n - 1)
    goal = jnp.where(cast_chase[:, None], jnp.stack([s.x[t_q], s.y[t_q]], -1), goal)
    amove = s.amove
    amove = AttackMove(active=jnp.where(new_order, am_req, amove.active),
                       x=jnp.where(am_req, orders.move_x, amove.x), y=jnp.where(am_req, orders.move_y, amove.y),
                       held=jnp.where(new_order, -1, amove.held), held_seq=amove.held_seq)
    ranks = _skill_up(s, cfg, orders)
    champ = champ._replace(attack_order=attack_order.astype(jnp.int32), moving=moving, move_goal=goal, ranks=ranks,
                           target_seen_at=seen_at, queued_cast=queued)
    s = s._replace(champ=champ, amove=amove)
    s = _spawn_minions(s, cfg, now)
    caps = M.capabilities(s.cc, now)
    stasis_n = jnp.zeros((n,), bool).at[:c].set(in_stasis)
    s = s._replace(targetable=s.targetable.at[:c].set(~in_stasis))
    caps = {k: (v & ~stasis_n if k.startswith("can_") else v) for k, v in caps.items()}
    # Terrain this tick: Elemental Rift / Baron-pit variant, then destroyed-structure pads (policy).
    from . import modern_dynamic_terrain as DTR
    terrain = cfg.terrain
    if cfg.rift is not None:
        from .modern_dynamic_terrain_rift import terrain_pair
        terrain = terrain_pair(cfg.rift, s.terrain_variant)
    if cfg.footprints is not None:
        terrain = DTR.walkable_masks(terrain, cfg.footprints, s.alive, DTR.release_mask(cfg.unit_kind))
    return s, orders, sc._replace(champ=champ, vis_c=vis_c, in_stasis=in_stasis, caps=caps, terrain=terrain,
                                  attack_order=attack_order, moving=moving, goal=goal, amove=amove)


def _stats(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """2. STATS: static stats (items, shards, monster buffs, kit); STAT.70 max-HP sync.

    Writes champion ``s.hp``/``s.max_hp`` and ``s.champ.static_max_hp`` (not ``sc.champ``)."""
    c = N_CHAMPIONS
    champ = sc.champ
    static, st_static = _static_stats(s, cfg, sc.caps, sc.now, cfg.dt)
    # STAT.70 for static max-HP changes (level-up, purchases, kit stacks); dynamic health is
    # synced by combat_tick.
    old_total = s.max_hp[:c]
    new_total = st_static.max_hp + s.combat.dyn_health
    hp_c, mx_c = sync_max_health(s.hp[:c], old_total, new_total)
    hp_c = jnp.where(s.alive[:c], hp_c, s.hp[:c])
    s = s._replace(hp=s.hp.at[:c].set(hp_c), max_hp=s.max_hp.at[:c].set(mx_c),
                   champ=champ._replace(static_max_hp=st_static.max_hp))
    return s, sc._replace(static=static, st_static=st_static)


def _objectives(s: ModernState, orders: ModernOrders, cfg: WorldConfig,
                sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """2b. OBJECTIVES: epic monsters (spawns, abilities, Rift transformation). Writes ``s.obj`` and the
    epic slots."""
    if cfg.objectives is None:
        return s, sc._replace(so=None, cinfo=None)
    from . import modern_objectives as OBJ
    c, dt = N_CHAMPIONS, cfg.dt
    now, key, champ, st_static = sc.now, sc.key, sc.champ, sc.st_static
    level = s.econ.level
    cinfo = OBJ.ChampInfo(level=level, bonus_ad=st_static.bonus_ad, ap=st_static.ap,
                          bonus_hp=st_static.max_hp - st_static.base_hp, max_hp=s.max_hp[:c],
                          max_mana=st_static.max_mana, adaptive_physical=cfg.adaptive_physical)
    k_obj, key = jax.random.split(key)
    obj, so = OBJ.objectives_step(s.obj, cfg.objectives, units_view(s), now=now, dt=jnp.float32(dt),
                                  levels=level, damage_matrix=s.damage_matrix, champ=cinfo,
                                  last_damaged=champ.last_damaged,
                                  ult_cast=s.champ.last_cast[:, 3] >= s.t - 1e-6, key=k_obj)
    s = _apply_objective_writes(s._replace(obj=obj), so, cfg)
    return s, sc._replace(so=so, cinfo=cinfo, key=key)


def _casts(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """3. CASTS: shop, kit casts and periodic effects, summoner spells, Smite; kit attack modifiers.

    Writes ``s.econ`` (shop gold) and ``s.champ`` (inventory, group cooldowns)."""
    c, n, dt = N_CHAMPIONS, cfg.n_units, cfg.dt
    now, caps, static, st_static = sc.now, sc.caps, sc.static, sc.st_static
    level = s.econ.level
    inv, gold, gcd, shop_code, bought, sold = _shop(s, cfg, orders, st_static, sc.champ.forbid)
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
    item_casting = now < champ.item_cast_until                       # Stridebreaker: walks, cannot cast/attack
    can_cast = caps["can_cast"][:c] & s.alive[:c] & ~locked & ~item_casting
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
                                rooted=(s.cc.root_until[:c] > now))
    sm = None
    jungle = s.jungle
    if cfg.jungle is not None:
        from . import modern_jungle as J
        jungle, sm = J.smite_step(jungle, cfg.jungle, units, summ_req, s.summoners.spell, now=now,
                                  summoner_haste=st.summoner_haste, alive=s.alive[:c] & caps["can_summoner"][:c])

    kmods = K.attack_mods(kits, kctx)
    reach = st.attack_range + kmods.extra_range                       # Garen Q / Jax W +50 (kit extra range)
    return s, sc._replace(champ=champ, st=st, summ_world=summ_world, units=units, kctx=kctx, ictx=ictx,
                          locked=locked, item_casting=item_casting, cast_order=order, kits=kits, kit_out=kit_out,
                          kmods=kmods, reach=reach, summ=summ, s_eff=s_eff, s_out=s_out, sm=sm, jungle=jungle,
                          shop_code=shop_code, bought=bought, sold=sold)


def _ai(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """4. AI: structure tick, minion/turret/monster targets and goals, champion targets (idle acquisition,
    attack-move).

    Writes structure/monster ``s.hp``/``s.alive``/``s.targetable``/``s.kind`` and ``s.obj``; ``towers`` and
    ``lane_ai`` stay in the scratch until MOVE."""
    c, n, dt = N_CHAMPIONS, cfg.n_units, cfg.dt
    now, units, jungle, so, champ = sc.now, sc.units, sc.jungle, sc.so, sc.champ
    attack_order, moving, amove, goal = sc.attack_order, sc.moving, sc.amove, sc.goal
    reach, vis_c = sc.reach, sc.vis_c
    towers = LA.turret_tick(s.towers, units, now=now, dt=jnp.float32(dt))
    hp_t, alive_t, targ_t = LA.structure_unit_view(towers, units)
    s = s._replace(hp=hp_t, alive=alive_t, targetable=targ_t)
    units = units_view(s)
    champ_vs_champ = s.damage_matrix & (s.kind == W.KIND_CHAMPION)[:, None] & (s.kind == W.KIND_CHAMPION)[None, :]
    lane_ai, desired, mgoal, stop = LA.select_targets(s.lane_ai, units, s.att, now=now, dt=jnp.float32(dt),
                                                      champion_attacked_champion=champ_vs_champ,
                                                      damage_events=s.damage_matrix, visible=s.visible)
    # Monsters: jungle camps and epic objectives choose their own targets and goals.
    mai = None
    m_goal = jnp.stack([s.x, s.y], -1)
    m_speed = jnp.zeros((n,), jnp.float32)
    m_move = jnp.zeros((n,), bool)
    if cfg.jungle is not None:
        from . import modern_jungle as J
        jungle, mai = J.monster_ai(jungle, cfg.jungle, units, s.att, now=now, dt=jnp.float32(dt),
                                   damage_events=s.damage_matrix)
        j0, jn = cfg.jungle.monster0, cfg.jungle.n_slots
        jsl = slice(j0, j0 + jn)
        desired = desired.at[jsl].set(mai.desired)
        m_goal = m_goal.at[jsl].set(jnp.stack([mai.goal_x, mai.goal_y], -1))
        m_speed, m_move = m_speed.at[jsl].set(mai.move_speed), m_move.at[jsl].set(mai.moving)
        live_j = s.alive[jsl] & ~mai.despawn
        s = s._replace(hp=s.hp.at[jsl].set(jnp.where(live_j, jnp.minimum(s.hp[jsl] + mai.heal, s.max_hp[jsl]),
                                                     s.hp[jsl])),
                       alive=s.alive.at[jsl].set(live_j),
                       kind=s.kind.at[jsl].set(jnp.where(mai.despawn, W.KIND_NONE, s.kind[jsl])),
                       targetable=s.targetable.at[jsl].set(mai.targetable & live_j))
    if so is not None:
        from . import modern_objectives as OBJ
        e0 = cfg.objectives.slot0
        esl = slice(e0, e0 + 8)
        desired = desired.at[esl].set(jnp.where(so.can_attack, so.desired, -1))
        m_goal = m_goal.at[esl].set(so.goal)
        m_speed, m_move = m_speed.at[esl].set(so.move_speed), m_move.at[esl].set(so.move_active)
        obj, mbuf = OBJ.baron_minion_buffs(s.obj, cfg.objectives, units_view(s), now=now)
        s = s._replace(obj=obj)
    else:
        mbuf = None
    units = units_view(s)

    # Champions: ordered target, attack-move, or idle auto-acquisition (LA.idle_acquire; chases).
    t_ok = (attack_order >= 0) & s.alive[jnp.clip(attack_order, 0, n - 1)]
    acq = LA.champion_acquisition_range(reach, cfg.champion_base.attack_range)
    unit_c = jnp.arange(c, dtype=jnp.int32)
    auto = LA.idle_acquire(units, unit_c, vis_c, acq)
    idle = ~t_ok & ~moving & ~amove.active & s.alive[:c] & (auto >= 0)
    attack_order = jnp.where(idle, auto, attack_order).astype(jnp.int32)
    t_ok = t_ok | idle
    am = LA.attack_move_step(amove.active & s.alive[:c], amove.x, amove.y, amove.held, amove.held_seq, units,
                             unit_c, vis_c, acq)
    amove = amove._replace(active=am.active, held=am.target, held_seq=am.target_seq)
    am_tgt = am.active & (am.target >= 0) & ~t_ok
    attack_order_eff = jnp.where(am_tgt, am.target, attack_order).astype(jnp.int32)
    t_ok_eff = t_ok | am_tgt
    goal = jnp.where((am.active & ~t_ok_eff)[:, None], jnp.stack([am.goal_x, am.goal_y], -1), goal)
    moving_eff = moving | (am.active & ~t_ok_eff)
    desired = desired.at[:c].set(jnp.where(t_ok_eff, attack_order_eff, -1))
    champ = champ._replace(attack_order=attack_order)
    return s, sc._replace(towers=towers, lane_ai=lane_ai, desired=desired, lane_goal=mgoal, lane_stop=stop,
                          mai=mai, mon_goal=m_goal, mon_speed=m_speed, mon_move=m_move, mbuf=mbuf, units=units,
                          attack_order=attack_order, attack_order_eff=attack_order_eff, t_ok_eff=t_ok_eff,
                          goal=goal, moving_eff=moving_eff, champ=champ, amove=amove, jungle=jungle)


def _move(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """5. MOVE: route movement, dashes, Flash, Teleport; collision.

    Reads: ``now, caps, champ, units, st, reach, locked, s_out`` (Flash/Teleport/Ghost), ``kit_out`` (dash),
    ``kits, kctx`` (kit ghosting), the AI goals ``attack_order_eff, t_ok_eff, goal, moving_eff, lane_goal,
    lane_stop, mon_goal, mon_speed, mon_move``, ``mai, mbuf, so, jungle, lane_ai, towers, terrain, ictx`` and
    ``s.pending_dash``.
    Writes ``s.x, s.y, s.lane_ai, s.towers, s.route_anchor``, dash state and facing in both ``s.champ``
    and ``champ``, and ``tp_lock, sres, ms, dstart, in_dash, units, ictx``."""
    c, n, dt = N_CHAMPIONS, cfg.n_units, cfg.dt
    now, caps, champ, units, st, reach = sc.now, sc.caps, sc.champ, sc.units, sc.st, sc.reach
    locked, s_out, kit_out, kits, kctx = sc.locked, sc.s_out, sc.kit_out, sc.kits, sc.kctx
    attack_order_eff, t_ok_eff, goal, moving_eff = sc.attack_order_eff, sc.t_ok_eff, sc.goal, sc.moving_eff
    mgoal, stop, m_goal, m_speed, m_move = sc.lane_goal, sc.lane_stop, sc.mon_goal, sc.mon_speed, sc.mon_move
    so, mai, mbuf, jungle, lane_ai, towers = sc.so, sc.mai, sc.mbuf, sc.jungle, sc.lane_ai, sc.towers
    terrain, ictx = sc.terrain, sc.ictx
    tp_lock = s_out.teleport_channel | s_out.teleport_dash
    can_move_c = caps["can_move"][:c] & s.alive[:c] & ~locked & ~tp_lock & (now >= champ.dash_until)
    tgt = jnp.clip(attack_order_eff, 0, n - 1)
    in_range = M.in_attack_range(units._replace(attack_range=units.attack_range.at[:c].set(reach)),
                                 jnp.full((n,), -1, jnp.int32).at[:c].set(jnp.where(t_ok_eff, attack_order_eff,
                                                                                    -1)))[:c]
    chase = t_ok_eff & ~in_range
    cgoal = jnp.where(chase[:, None], jnp.stack([s.x[tgt], s.y[tgt]], -1), goal)
    cact = can_move_c & (chase | (moving_eff & ~t_ok_eff))
    minion = (s.kind == W.KIND_MINION) & s.alive
    monster = (s.kind == W.KIND_MONSTER) & s.alive
    gx = jnp.where(minion, mgoal[:, 0], jnp.where(monster, m_goal[:, 0], s.x)).at[:c].set(cgoal[:, 0])
    gy = jnp.where(minion, mgoal[:, 1], jnp.where(monster, m_goal[:, 1], s.y)).at[:c].set(cgoal[:, 1])
    hg_bonus = 0.0 if so is None else so.homeguard_bonus * s.econ.homeguard.active
    sres = jnp.zeros((n,), jnp.float32).at[:c].set(st.slow_resist)
    if mai is not None:                                               # Scuttler: slow immune while not fleeing
        jsl = slice(cfg.jungle.monster0, cfg.jungle.monster0 + cfg.jungle.n_slots)
        sres = sres.at[jsl].set(mai.slow_resist)
    # Gustwalker's Gait from jungle state (brush entry detected at last tick's combat phase: 1 tick lag).
    if cfg.jungle is None:
        gust = 0.0
    else:
        from . import modern_jungle as J
        gust = J.gust_bonus_ms(jungle, now)
    m_ms = LA.minion_move_speed(lane_ai, units, now)
    if mbuf is not None:
        m_ms = jnp.where(minion, jnp.maximum(m_ms, mbuf.ms_floor), m_ms)
    # Non-champion slows apply at use with slow resist (champions: STAT pipeline ``slow=``).
    # Champions: Ghost/Heal, Gustwalker and Homeguard are bonus % MS inside the STAT pipeline, before
    # the soft caps (26.9 replays, REPLAY_FIDELITY; adding them after the caps ran ~60 u fast).
    summ_world = sc.summ_world
    extra_pct = s_out.bonus_ms_pct + gust + champ.homeguard_ms + hg_bonus
    champ_ms = compose(cfg.champion_base, s.econ.level,
                       summ_world._replace(percent_move_speed=summ_world.percent_move_speed + extra_pct),
                       adaptive_physical=cfg.adaptive_physical, slow=caps["slow"][:c]).move_speed
    ms = (jnp.where(monster, m_speed, m_ms) * (1.0 - caps["slow"] * (1.0 - sres))).at[:c].set(champ_ms)
    active = ((minion & ~stop | monster & m_move) & caps["can_move"]).at[:c].set(cact)
    team_mv = jnp.clip(s.team, 0, 1)                                  # neutral monsters walk the blue mask
    x, y, _, anchor = M.move_step(s.x, s.y, gx, gy, ms, active, team_mv, s.radius, cfg.routes, terrain, dt,
                                  s.route_anchor, movers=MW_layout()["ward0"])
    # Kit dashes (Jax Q): follow the target unit at the dash speed (no terrain, it's a leap).
    dash = kit_out.dash
    pd = s.pending_dash                                               # item-active dash (Rocketbelt)
    dash = W.Dash(*(jnp.where(pd.active & ~dash.active, a, b) for a, b in zip(pd, dash)))
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
                           s.radius[:c], terrain)
    cx, cy = jnp.where(blink, bx, cx), jnp.where(blink, by, cy)
    cx = jnp.where(s_out.teleport_arrive, s_out.teleport_x, cx)
    cy = jnp.where(s_out.teleport_arrive, s_out.teleport_y, cy)
    x, y = x.at[:c].set(cx.astype(x.dtype)), y.at[:c].set(cy.astype(y.dtype))
    # Unit collision (modern_collision, COLLISION.md): avoidance steering of movers, then soft separation,
    # pathing radii, never into terrain (movement clearance on the team mask, as the MOVE clamp and eject).
    from . import modern_collision as UC
    ghost = (jnp.zeros((n,), bool).at[:c].set(s_out.ghosted | (now < dash_until) | K.ghosted(kits, kctx))
             | LA.minion_ghosted(lane_ai, units, now))
    # Wards have no collision; structures block through their navgrid pads (terrain), not as unit
    # obstacles: their collision circles reach past the pads that routes are baked around, which left
    # units pinned against them (red stuck at its top inhibitor).
    collide = s.alive & (s.kind != W.KIND_WARD) & ~W.is_structure(s.kind)
    x, y = UC.resolve(s.x, s.y, x, y, radius=UC.pathing_radius(s.kind, s.sub, s.radius), collide=collide,
                      ghosted=ghost, moving=active, goal_x=gx, goal_y=gy, team=team_mv,
                      clearance=jnp.minimum(s.radius, cfg.routes.radius), terrain=terrain, dt=dt,
                      movers=MW_layout()["ward0"])
    facing = jnp.stack([x[:c] - s.x[:c], y[:c] - s.y[:c]], -1)
    norm = jnp.linalg.norm(facing, axis=-1, keepdims=True)
    facing = jnp.where(norm > 1e-3, facing / jnp.maximum(norm, 1e-6), champ.facing)
    moved = jnp.sqrt((x[:c] - s.x[:c]) ** 2 + (y[:c] - s.y[:c]) ** 2)
    # A move order ends on arrival, or where the champion can make no more progress (unreachable or
    # walled-off goal: League walks as far as it can and stops). Then idle auto-acquire resumes
    # (MECHANICS_AUDIT #2; the flag used to stay set for the rest of the game).
    to_goal = jnp.sqrt((x[:c] - champ.move_goal[:, 0]) ** 2 + (y[:c] - champ.move_goal[:, 1]) ** 2)
    stuck = cact & (moved < 0.5) & ~in_dash
    champ = champ._replace(moving=champ.moving & ~(to_goal <= MOVE_ARRIVE_RADIUS) & ~(stuck & ~chase))
    # Dash state and facing go to the working ChampionLayer too: it becomes the next state's ``champ``
    # (writing only ``s.champ`` dropped them every tick, so a Jax Q leap moved only on its start tick).
    champ = champ._replace(dash_until=dash_until, dash_target=dash_target.astype(jnp.int32),
                           dash_to=jnp.where(dstart[:, None], jnp.stack([dx, dy], -1), champ.dash_to),
                           dash_speed=jnp.where(dstart, dash.speed, champ.dash_speed), facing=facing)
    s = s._replace(x=x, y=y, lane_ai=lane_ai, towers=towers, champ=champ, route_anchor=anchor)
    units = units_view(s)._replace(attack_range=s.arange.at[:c].set(reach),
                                   attack_speed=s.aspeed.at[:c].set(st.attack_speed))
    ictx = ictx._replace(x=x[:c], y=y[:c], facing_x=facing[:, 0], facing_y=facing[:, 1], moved=moved)
    return s, sc._replace(champ=champ, tp_lock=tp_lock, sres=sres, ms=ms, dstart=dstart, in_dash=in_dash,
                          units=units, ictx=ictx)


def _minion_pushing(s: ModernState, cfg: WorldConfig, lane_ai, now) -> tuple[Any, Any]:
    """(N,) Minion Pushing (MINIONS §4.4, MECHANICS_AUDIT #6): each lane minion's team bonus damage
    and the divisor on minion damage it takes, from its team's level lead (one champion per team, so
    the champion's level) and its lane's turret lead. Recomputed every tick (the client holds it for
    1 s, so a level-up or turret kill applies up to 1 s earlier here)."""
    from . import modern_minions as MM
    n = s.kind.shape[0]
    level = s.econ.level.astype(jnp.float32)
    team = jnp.clip(s.team, 0, 1)
    turret = (s.kind == W.KIND_TURRET) & s.alive
    alive_t = jnp.stack([jnp.stack([jnp.sum(turret & (s.team == t) & (cfg.unit_lane == l)) for l in range(3)])
                         for t in (0, 1)]).astype(jnp.float32)                                     # (team, lane)
    lane = jnp.clip(lane_ai.lane, 0, 2)
    lvl_adv = level[team] - level[1 - team]
    tow_adv = alive_t[team, lane] - alive_t[1 - team, lane]
    bonus, div = MM.minion_pushing_modifiers(team_level_advantage=lvl_adv, lane_turret_advantage=tow_adv,
                                             time_s=jnp.floor(now))
    minion = (s.kind == W.KIND_MINION) & (lane_ai.lane >= 0)
    return jnp.where(minion, bonus, 0.0), jnp.where(minion, div, 1.0) * jnp.ones((n,), jnp.float32)


def _attack(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """6. ATTACK: attack machine, crits, attack packets, missiles, ward hits, kit on-attack/on-hit,
    Overgrowth, jungle combat effects. Writes ``s.obj``."""
    from . import modern_item_effects as IE
    from .modern_champions.core import merge_out
    c, n, dt = N_CHAMPIONS, cfg.n_units, cfg.dt
    now, caps, champ, units, st, kmods = sc.now, sc.caps, sc.champ, sc.units, sc.st, sc.kmods
    locked, item_casting, tp_lock, in_dash = sc.locked, sc.item_casting, sc.tp_lock, sc.in_dash
    kit_out, kits, kctx, ictx, mbuf, so = sc.kit_out, sc.kits, sc.kctx, sc.ictx, sc.mbuf, sc.so
    lane_ai, jungle, desired, k_crit = sc.lane_ai, sc.jungle, sc.desired, sc.k_crit
    can_attack = (caps["can_attack"] & s.alive).at[:c].set(
        caps["can_attack"][:c] & s.alive[:c] & ~kmods.cannot_attack & ~locked & ~item_casting & ~tp_lock
        & ~in_dash)
    opt = lambda v, d: d if v is None else v                          # noqa: E731
    k_wind, k_per_ = opt(kmods.windup, jnp.zeros((c,))), opt(kmods.period, jnp.zeros((c,)))
    windup = s.windup.at[:c].set(jnp.where(k_wind > 0, k_wind, st.attack_windup))
    period = jnp.zeros((n,), jnp.float32).at[:c].set(k_per_)
    uncancel = jnp.zeros((n,), bool).at[:c].set(opt(kmods.uncancellable, jnp.zeros((c,), bool)))
    reset = jnp.zeros((n,), bool).at[:c].set(kit_out.attack_reset | kmods.attack_reset | champ.reset_next)
    if mbuf is not None:                                              # Hand of Baron empowered minions
        units = units._replace(attack_range=units.attack_range + mbuf.bonus_range,
                               attack_damage=units.attack_damage + mbuf.bonus_ad,
                               attack_speed=units.attack_speed * jnp.where(mbuf.empowered, mbuf.attack_speed_mult,
                                                                           1.0))
    att_prev = s.att
    att, launched = M.attack_step(s.att, units, desired, can_attack=can_attack, windup=windup, dt=dt, reset=reset,
                                  period=period, uncancellable=uncancel)
    started = (att.windup_left > 0) & (att_prev.windup_left <= 0)
    cancelled = (att_prev.windup_left > 0) & (att.windup_left <= 0) & ~launched
    atgt = jnp.clip(att.target, 0, n - 1)
    # Champions: crit roll at launch (X-8), Garen Q's spell attack cannot crit.
    imods = IE.attack_mods(s.combat.items, IE_own(champ.inventory), ictx, _item_units(s), att.target[:c])
    no_crit = jnp.zeros((c,), bool) if kmods.cannot_crit is None else kmods.cannot_crit
    roll = jax.random.uniform(k_crit, (c,)) < st.crit_chance
    crit = launched[:c] & ~no_crit & (roll | imods.force_crit)
    crit_mult = 1.0 + (st.crit_damage - 1.0) * jnp.where(imods.force_crit, imods.crit_scale, 1.0)
    vs_struct = W.is_structure(s.kind[atgt[:c]])
    struct_raw, struct_magic = LA.T.champion_structure_attack(st.base_ad, st.bonus_ad, st.ap)
    craw = jnp.where(vs_struct, struct_raw, (st.base_ad + st.bonus_ad) * jnp.where(crit, crit_mult, 1.0))
    # Melee champions deal x1.2 to turrets (towers.json melee_champion_damage_multiplier; a
    # multiplicative factor, so applying it to raw is equivalent to post-mitigation).
    vs_turret = s.kind[atgt[:c]] == W.KIND_TURRET
    craw = craw * jnp.where(vs_turret & (s.missile_speed[:c] <= 0), 1.2, 1.0)
    cdtype = jnp.where(vs_struct & struct_magic, D.MAGIC, D.PHYSICAL)
    assert n <= CAST_ID_STRIDE
    cast_ids = (s.tick * CAST_ID_STRIDE + jnp.arange(n)).astype(jnp.int32) + 1   # unique per (tick, unit)
    ranged_c = s.missile_speed[:c] > 0
    # Champion basic attacks on wards deal 1 hit each (modern_wards), not damage packets or on-hit.
    on_ward = (s.kind[atgt] == W.KIND_WARD) & (att.target >= 0)
    hit_c = launched[:c] & ~ranged_c & ~on_ward[:c]
    attack = Attack(launched[:c], hit_c, att.target[:c], jnp.where(launched[:c], craw, 0.0), crit)
    push_bonus, push_div = _minion_pushing(s, cfg, lane_ai, now)
    lane_pk = LA.attack_packets(units, W.AttackLaunch(launched & (s.kind != W.KIND_CHAMPION), att.target,
                                                      s.missile_speed > 0, jnp.zeros((n,), bool), cast_ids),
                                now=now, ai=lane_ai, pushing_bonus=push_bonus, pushing_divisor=push_div)
    lane_pk = lane_pk._replace(cast_id=cast_ids)
    msl_speed = s.missile_speed if mbuf is None else jnp.where(mbuf.missile_speed > 0, mbuf.missile_speed,
                                                               s.missile_speed)
    if mbuf is not None:                                              # empowered siege minions vs structures
        t_struct = W.is_structure(s.kind[atgt])
        lane_pk = lane_pk._replace(raw=lane_pk.raw * jnp.where(t_struct & mbuf.empowered,
                                                               mbuf.siege_structure_mult, 1.0))
    extra_atk = []
    launch_all = W.AttackLaunch(launched, att.target, s.missile_speed > 0, jnp.zeros((n,), bool), cast_ids)
    if cfg.jungle is not None:
        from . import modern_jungle as J
        j_main, j_bonus = J.monster_attack_packets(jungle, cfg.jungle, units, launch_all)
        jsl = slice(cfg.jungle.monster0, cfg.jungle.monster0 + cfg.jungle.n_slots)
        lane_pk = lane_pk._replace(raw=lane_pk.raw.at[jsl].set(j_main.raw), dtype=lane_pk.dtype.at[jsl].set(j_main.dtype),
                                   flags=lane_pk.flags.at[jsl].set(j_main.flags))
        extra_atk.append(j_bonus)
    obj_cc = None
    if so is not None:
        from . import modern_objectives as OBJ
        obj, o_raw, o_dtype, o_flags, o_extra, obj_cc = OBJ.objectives_attack(s.obj, cfg.objectives, units,
                                                                            launch_all, now=now)
        s = s._replace(obj=obj)
        esl = slice(cfg.objectives.slot0, cfg.objectives.slot0 + 8)
        on = (jnp.zeros((n,), bool).at[esl].set(True)) & (o_raw > 0)
        lane_pk = lane_pk._replace(raw=jnp.where(on, o_raw, lane_pk.raw), dtype=jnp.where(on, o_dtype, lane_pk.dtype),
                                   flags=jnp.where(on, o_flags, lane_pk.flags))
        extra_atk.append(o_extra)
    ranged = launched & (msl_speed > 0)
    flags_c = jnp.full((c,), D.BASIC_ATTACK, jnp.int32) | jnp.where(crit, jnp.int32(D.PROP_CRIT), jnp.int32(0))
    raw_all = lane_pk.raw.at[:c].set(craw)
    dtype_all = lane_pk.dtype.at[:c].set(cdtype)
    flags_all = lane_pk.flags.at[:c].set(flags_c)
    lay = MW_layout()
    w0 = lay["ward0"]
    missiles, m_over = M.spawn_missiles(s.missiles, ranged & ~on_ward, units, att.target, raw_all, dtype_all,
                                        flags_all, msl_speed, cast_ids, jnp.zeros((n,), bool).at[:c].set(crit))
    missiles, arrive = M.advance_missiles(missiles, units, dt)
    direct = D.packets(launched & ~ranged & (att.target >= 0) & ~on_ward, jnp.arange(n), jnp.maximum(att.target, 0),
                       raw_all, dtype_all, flags_all, cast_id=cast_ids)
    hit_ward = launched & on_ward & (jnp.arange(n) < c)               # champions only hit wards
    w_slot = jnp.clip(att.target - w0, 0, 2 * W.MAX_WARDS_PER_TEAM - 1)
    ward_hits = jnp.zeros((2 * W.MAX_WARDS_PER_TEAM,), jnp.int32).at[w_slot].add(hit_ward.astype(jnp.int32))
    ward_hitter = jnp.full((2 * W.MAX_WARDS_PER_TEAM,), -1, jnp.int32).at[w_slot].max(
        jnp.where(hit_ward, jnp.arange(n), -1).astype(jnp.int32))
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
    jfx = None
    if cfg.jungle is not None:
        # Scorchclaw: enemy champion damaged (last tick's damage_matrix, 1 tick lag); Gustwalker: in brush.
        dm_cc = s.damage_matrix[:c, :c] & (s.team[:c][:, None] != s.team[:c][None, :])
        dmg_champ = jnp.where(jnp.any(dm_cc, axis=1), jnp.argmax(dm_cc, axis=1), -1).astype(jnp.int32)
        jungle, jfx = J.combat_effects(jungle, cfg.jungle, units, ictx, attack_hit=attack.hit,
                                       attack_target=attack.target, damaged_champion=dmg_champ,
                                       in_brush=_in_brush(cfg, s.x[:c], s.y[:c], s.terrain_variant))
    return s, sc._replace(att=att, launched=launched, started=started, cancelled=cancelled, reset=reset,
                          attack=attack, missiles=missiles, m_over=m_over, direct=direct, arrived=arrived,
                          og_pk=og_pk, extra_atk=extra_atk, obj_cc=obj_cc, ward_hits=ward_hits,
                          ward_hitter=ward_hitter, kits=kits, towers=towers, kit_all=kit_all, jfx=jfx,
                          jungle=jungle, units=units)


def _damage(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """7. DAMAGE: every packet -> combat_tick (items, runes, damage pipeline); kit on-damage."""
    from . import modern_item_actives as A
    c, n = N_CHAMPIONS, cfg.n_units
    now, caps, champ, units, kits, kctx, ictx = sc.now, sc.caps, sc.champ, sc.units, sc.kits, sc.kctx, sc.ictx
    so, sm, jfx, kit_all, s_eff, s_out, towers = sc.so, sc.sm, sc.jfx, sc.kit_all, sc.s_eff, sc.s_out, sc.towers
    st_static, att, reset, in_stasis = sc.st_static, sc.att, sc.reset, sc.in_stasis
    more = list(sc.extra_atk)
    if sm is not None:
        more.append(sm.packets)
    if jfx is not None:
        more.append(jfx.packets)
    if so is not None:
        more.append(so.packets)
    base = D.concat_packets(sc.direct, sc.arrived, sc.og_pk, kit_all.packets, s_eff.packets, *more)
    if so is not None:
        from . import modern_objectives as OBJ
        base = OBJ.objectives_packet_mods(s.obj, cfg.objectives, base, units, now=now)
    kdef = K.defense(kits, kctx)
    kdeb = K.debuffs(kits, kctx, units)
    t_armor, t_mr = LA.turret_defense(towers, units, now=now)
    t_mult, t_invuln = LA.structure_defense_mods(towers, units, now=now)
    armor = t_armor.at[:c].set(st_static.base_armor + st_static.bonus_armor)
    mr = t_mr.at[:c].set(st_static.base_mr + st_static.bonus_mr)
    if so is not None:                                                # Baron's Void Corruption
        armor, mr = armor - so.armor_reduction, mr - so.armor_reduction
    dfn = D.default_defense(n)._replace(
        armor=armor, magic_resist=mr, unit_class=W.damage_class(s.kind),
        received_mult=jnp.ones((n,), jnp.float32).at[:c].set(kdef.received_mult),
        received_mult_all=t_mult,
        invulnerable=t_invuln | jnp.zeros((n,), bool).at[:c].set(s_out.teleport_dash | in_stasis),
        percent_armor_reduction=kdeb.percent_armor_reduction,
        dodge_basic=jnp.zeros((n,), bool).at[:c].set(kdef.dodge_basic),
        aoe_received_mult=jnp.ones((n,), jnp.float32).at[:c].set(kdef.aoe_received_mult))
    off = LA.turret_offense(units, D.default_offense(n)._replace(unit_class=W.damage_class(s.kind),
                                                                dealt_reduction=s_out.exhaust_reduction))
    cc_now = kit_all.cc
    cc_items = CC(cc_now.slow > 0, (cc_now.stun > 0) | (cc_now.root > 0) | (cc_now.knockup > 0))
    # Overgrowth (U-15): deaths last tick with own sight latched while the victim was alive
    # (``sight`` is end-of-tick and drops dead units, so it cannot show the death itself).
    deaths_prev = jnp.any(s.death_seen, axis=0)
    ev = rune_events(ictx, n, game_time=now, attack_started=sc.started[:c], attack_start_target=att.target[:c],
                     attack_cancelled=sc.cancelled[:c], attack_reset=reset[:c], cast_id=kit_all.cast_id,
                     cc_duration=jnp.maximum(cc_now.stun, cc_now.root), impaired=caps["impaired"],
                     movement_impaired=caps["movement_impaired"],
                     impaired_by_holder=(cc_now.slow > 0) | (cc_now.stun > 0) | (cc_now.root > 0),
                     holder_cc_from_champion=s.cc.champion_cc_until[:c] > now,
                     summoner_cast=s_out.cast_event, summoner_cooldown=s_out.cast_cooldown,
                     summoner_is_teleport=s_out.is_teleport,
                     blinked=s_out.blinked | sc.dstart, flash_cooldown=S.flash_cooldown(sc.summ, now),
                     deaths=deaths_prev, purchased=sc.bought, sold=sc.sold, granted=champ.granted,
                     uses_energy=cfg.uses_energy, adaptive_physical=cfg.adaptive_physical,
                     is_turret=s.kind == W.KIND_TURRET, cc_cast_id=cc_now.cast_id,
                     cc_on_hit=jnp.zeros((c, n), bool), sight=s.sight[:c] | s.death_seen, visible=sc.vis_c,
                     epic_takedown=s.epic_prev, large_monster_kill=s.large_prev,
                     in_river=_in_river(cfg, s.x[:c], s.y[:c]))
    items0 = s.combat.items
    items0 = items0._replace(modern_item_actives=A.with_aim(items0.modern_item_actives, orders.cast_target,
                                                            orders.cast_x, orders.cast_y))
    item_req = A.request_allowed(orders.item_active, disabled=caps["stunned"][:c], in_stasis=in_stasis)
    out = combat_tick(s.combat._replace(items=items0), IE_own(champ.inventory), cfg.rune_pages, ictx,
                      _item_units(s), attack=sc.attack,
                      cast=Cast(kit_all.cast_started, kit_all.cast_slot, sc.cast_order.target),
                      request=item_req, base_packets=base, base_offense=off, base_defense=dfn,
                      hp=s.hp, max_hp=s.max_hp, shields=s.shields, status=s.status, kills=s.kills,
                      holder_stats=sc.static, cc=cc_items, ev=ev)
    hp, max_hp, shields, status = out.hp, out.max_hp, out.shields, out.status
    kits, k_dmg = K.on_damage(kits, kctx, units, out.report)
    return s, sc._replace(out=out, kdef=kdef, k_dmg=k_dmg, cc_now=cc_now, cc_items=cc_items, hp=hp, max_hp=max_hp,
                          shields=shields, status=status, kits=kits)


def _cc_heal(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """8. CC / HEAL: kit, summoner and monster heals and shields; CC with tenacity and slow resist."""
    from . import modern_item_actives as A
    from .modern_item_effects.core import shield_grants
    c, n = N_CHAMPIONS, cfg.n_units
    now, st, kit_all, k_dmg, s_eff, s_out, out = sc.now, sc.st, sc.kit_all, sc.k_dmg, sc.s_eff, sc.s_out, sc.out
    jfx, so, sm, mai, sres, kdef = sc.jfx, sc.so, sc.sm, sc.mai, sc.sres, sc.kdef
    hp, max_hp, shields, status = sc.hp, sc.max_hp, sc.shields, sc.status
    kit_shields = ShieldGrant(*(jnp.concatenate([u, v], axis=1) for u, v in zip(kit_all.shield, k_dmg.shield)))
    m_heal = jnp.zeros((c,), jnp.float32)
    m_mana = jnp.zeros((c,), jnp.float32)
    m_shield = jnp.zeros((c,), jnp.float32)
    if jfx is not None:
        m_heal, m_shield = m_heal + jfx.heal, jnp.maximum(m_shield, jfx.shield)
    if so is not None:
        m_heal, m_mana, m_shield = m_heal + so.heal, m_mana + so.mana, jnp.maximum(m_shield, so.shield)
    extra = merge_effects([s_eff._replace(packets=D.empty_packets(0)),
                           effects(c, n, heal=kit_all.heal + k_dmg.heal, shields=kit_shields),
                           effects(c, n, heal=m_heal, mana=m_mana,
                                   shields=shield_grants(m_shield, duration=jnp.inf))], c, n)
    hp, shields, status = apply_effects(extra, sc.ictx, hp, max_hp, shields, status,
                                        heal_power=st.heal_shield_power, incoming_heal=sc.summ_world.incoming_heal)
    # Champion-sourced slows without a (C, N) source row: Exhaust, item/rune effects (Rylai's, Stridebreaker,
    # Spellblade fields ...). ``CCTimers`` is the only slow state movement reads; ``status.slow`` is unused.
    slow_cc = W.no_cc(3, n)._replace(slow=jnp.stack([s_out.exhaust_slow, out.effects.slow, extra.slow]),
                                     slow_duration=jnp.stack([s_out.exhaust_slow_duration, out.effects.slow_duration,
                                                              extra.slow_duration]))
    ten = jnp.zeros((n,), jnp.float32).at[:c].set(
        1.0 - (1.0 - st.tenacity) * (1.0 - kdef.tenacity_bonus) * (1.0 - s_out.tenacity))
    if mai is not None:                                               # Scuttler: slow immune, -100% tenacity
        jsl = slice(cfg.jungle.monster0, cfg.jungle.monster0 + cfg.jungle.n_slots)
        ten = ten.at[jsl].set(1.0 - mai.cc_duration_mult)
    champ_cc = sc.cc_now
    for extra_cc in ([] if sm is None else [sm.cc]) + ([] if jfx is None else [jfx.cc]) \
            + ([] if so is None else [so.champion_cc]):
        champ_cc = W.merge_cc(champ_cc, extra_cc)
    item_cleanse = A.world(out.state.items.modern_item_actives, now)
    cc = M.apply_cc(s.cc, champ_cc, ten, sres, now, source_is_champion=jnp.ones((c,), bool),
                    cleansed=jnp.zeros((n,), bool).at[:c].set(s_out.cleanse | item_cleanse.cleanse))
    for mcc in ([] if so is None else [so.cc]) + ([] if sc.obj_cc is None else [sc.obj_cc]):
        cc = M.apply_cc(cc, mcc, ten, sres, now, source_is_champion=jnp.zeros((mcc.stun.shape[0],), bool))
    cc = M.apply_cc(cc, slow_cc, ten, sres, now, source_is_champion=jnp.ones((3,), bool))
    clean_slow = jnp.zeros((n,), bool) if kit_all.cleanse_slow is None else \
        jnp.zeros((n,), bool).at[:c].set(kit_all.cleanse_slow)
    cc = cc._replace(slow_until=jnp.where(clean_slow, now, cc.slow_until))
    return s, sc._replace(extra=extra, hp=hp, shields=shields, status=status, item_cleanse=item_cleanse, cc=cc)


def _death(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """9. DEATH: kills, plates, camp/objective rewards -> economy_step; kit on-takedown. Writes ``s.obj``.

    Reads start-of-tick ``s.cc``/``caps`` for the recall interrupt (``sc.cc`` is this tick's CC)."""
    from .modern_item_effects.core import Report
    c, n = N_CHAMPIONS, cfg.n_units
    now, caps, champ, units, st, out, so = sc.now, sc.caps, sc.champ, sc.units, sc.st, sc.out, sc.so
    hp, max_hp, towers, jungle, lane_ai, kits = sc.hp, sc.max_hp, sc.towers, sc.jungle, sc.lane_ai, sc.kits
    s_out, s_eff = sc.s_out, sc.s_eff
    level = s.econ.level
    x, y = s.x, s.y                                                   # MOVE's positions
    rp = D.concat_packets(out.report.packets, out.follow_up.packets)
    rr_killed = jnp.concatenate([out.report.resolved.killed, out.follow_up.resolved.killed])
    rr_loss = jnp.concatenate([out.report.resolved.health_loss, out.follow_up.resolved.health_loss])
    killer = jnp.full((n,), -1, jnp.int32).at[jnp.clip(rp.dst, 0, n - 1)].max(
        jnp.where(rp.valid & rr_killed, rp.src, -1).astype(jnp.int32))
    dmg = jnp.zeros((n, n), bool).at[jnp.clip(rp.src, 0, n - 1), jnp.clip(rp.dst, 0, n - 1)].max(rp.valid)
    died = s.alive & (hp <= 0.0)
    death_seen = died[None, :] & s.sight[:c]                          # start-of-tick sight, victim alive (Overgrowth)
    struct = W.is_structure(s.kind)
    towers, plates = LA.structure_damage_events(towers, s.hp, jnp.where(struct, hp, s.hp), now=now)
    minion_died = died & (s.kind == W.KIND_MINION)
    last_hitter = jnp.where(killer < c, killer, -1)
    md = E.MinionDeaths(valid=minion_died, x=s.x, y=s.y, team=s.team, gold=s.m_gold, xp=s.m_xp, level=s.m_level,
                        last_hitter=last_hitter, unit=jnp.arange(n, dtype=jnp.int32))
    sv = plates.plates > 0
    sev = E.StructureEvents(valid=sv | plates.destroyed, unit=jnp.arange(n, dtype=jnp.int32), x=s.x, y=s.y,
                            team=s.team, local_gold=plates.plate_gold + plates.first_turret_gold,
                            global_gold=plates.global_gold, is_turret=plates.destroyed & (s.kind == W.KIND_TURRET),
                            in_top_lane=cfg.unit_lane == 2, is_structure=struct)
    took_health = jnp.zeros((n,), bool).at[jnp.clip(rp.dst, 0, n - 1)].max(rp.valid & (rr_loss > 0))
    in_f = E.in_fountain(x[:c], y[:c], cfg.fountain[s.team[:c], 0], cfg.fountain[s.team[:c], 1])
    # Monster and objective deaths: rewards to the killing champion / team (not MinionDeaths).
    dec = E.decimal_level(s.econ.xp, jnp.where(s.econ.quest.complete, E.QUEST_LEVEL_CAP, E.LEVEL_CAP))
    xg = jnp.zeros((c,), jnp.float32)
    gg = jnp.zeros((c,), jnp.float32)
    epic = jnp.zeros((c,), jnp.float32)
    large = jnp.zeros((c,), jnp.float32)
    killed_mon = jnp.zeros((c, n), bool)
    jrw = None
    if cfg.jungle is not None:
        from . import modern_jungle as J
        jungle, jrw = J.death_step(jungle, cfg.jungle, units, now=now, died=died, killer=killer,
                                   avg_level=jnp.mean(dec), champion_level=dec, hp=hp[:c], max_hp=max_hp[:c],
                                   mana=champ.mana, max_mana=st.max_mana, champion_died=died[:c],
                                   champion_killer=jnp.where(killer[:c] < c, killer[:c], -1))
        gg, xg = gg + jrw.gold, xg + jrw.xp
        large = large + jrw.large_kills
        j0 = cfg.jungle.monster0
        killed_mon = killed_mon.at[:, j0:j0 + cfg.jungle.n_slots].set(jrw.killed)
    if so is not None:
        from . import modern_objectives as OBJ
        obj, orw = OBJ.objectives_after_damage(s.obj, cfg.objectives, units, rp, rr_loss, died=died, killer=killer,
                                               hp_after=hp, now=now, levels=level, champ=sc.cinfo)
        s = s._replace(obj=obj)
        gg, xg = gg + orw.gold, xg + orw.xp
        epic, large = epic + orw.epic_takedown, large + orw.large_monster_kill
        # Epic takedowns for K.on_takedown / item+rune hooks: the killing team's champions (the
        # participation window is internal to modern_objectives; exact with one champion per team).
        esl = slice(cfg.objectives.slot0, cfg.objectives.slot0 + 8)
        o_td = orw.killed[None, :] & (orw.killer_team[None, :] == s.team[:c][:, None])
        killed_mon = killed_mon.at[:, esl].set(killed_mon[:, esl] | o_td)
    if cfg.regions is not None:
        from . import modern_map_regions as REG
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
        health_damage=took_health[:c], disabled=caps["stunned"][:c] | caps["silenced"][:c] | (s.cc.root_until[:c] > now),
        reached_endpoint=reached, in_jungle=in_jg, teleported=s_out.teleport_arrive,
        extra_gold=gg, extra_xp=xg, epic=epic, recall_channel=recall_ch, minion_gold_delta=pet_gold,
        minion_xp_mult=pet_xp)
    eco = E.economy_step(s.econ, einp)
    econ = eco.state
    gold_extra = out.effects.gold + s_eff.gold
    econ = econ._replace(gold=jnp.minimum(econ.gold + gold_extra, 100000.0), gold_total=econ.gold_total + gold_extra)
    eco = eco._replace(kills=eco.kills._replace(killed_units=eco.kills.killed_units | killed_mon))
    if jrw is not None:                                               # kill restores, pet egg consumed at evolution
        hp = hp.at[:c].set(jnp.minimum(hp[:c] + jrw.heal, max_hp[:c]))
        champ = champ._replace(mana=jnp.minimum(champ.mana + jrw.mana, st.max_mana))
    kits = K.on_takedown(kits, sc.kctx, units, eco.kills)
    return s, sc._replace(dmg=dmg, died=died, death_seen=death_seen, towers=towers, plates=plates,
                          minion_died=minion_died, took_health=took_health, in_f=in_f, epic=epic, large=large,
                          jrw=jrw, jungle=jungle, eco=eco, econ=econ, hp=hp, champ=champ, kits=kits)


def _timers(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """10. TIMERS: respawn, recall, cooldowns, mana/HP regen, fountain, inventory outputs, wards, ward units,
    terrain ejection. Leaves ``s`` untouched: the next-state columns go to the scratch for FOG/_commit."""
    from . import modern_dynamic_terrain as DTR
    from . import modern_wards as WD
    c, dt = N_CHAMPIONS, cfg.dt
    cat = catalog()
    now, caps, champ, st, out, kit_all, extra = sc.now, sc.caps, sc.champ, sc.st, sc.out, sc.kit_all, sc.extra
    hp, max_hp, eco, econ, jrw, in_f, died = sc.hp, sc.max_hp, sc.eco, sc.econ, sc.jrw, sc.in_f, sc.died
    s0, s_out, in_stasis = sc.s0, sc.s_out, sc.in_stasis
    x, y = s.x, s.y                                                   # MOVE's positions
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
    aw = sc.item_cleanse                                              # item-active world effects
    cd_rate = jnp.concatenate([jnp.broadcast_to(aw.basic_cd_rate[:, None], (c, 3)), jnp.ones((c, 1))], axis=1)
    cds = jnp.maximum(champ.cooldowns - dt * cd_rate, 0.0)
    cds = jnp.where(kit_all.cooldown_start, cooldown(kit_all.base_cooldown, haste), cds)
    ro = out.rune_outputs
    cds = cds.at[:, :3].multiply((1.0 - ro.basic_cd_refund)[:, None]).at[:, 3].multiply(1.0 - ro.ult_cd_refund)
    mana = jnp.clip(champ.mana - kit_all.mana_cost * aw.mana_cost_mult + out.effects.mana + extra.mana
                    + st.mana_regen * dt,
                    0.0, st.max_mana)
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
    frm2, to2, do2 = aw.transform                                     # Seeker's -> Shattered Armguard
    inv = I.Inventory(*jax.vmap(lambda it, stk, f_, t_, d_: tuple(I.replace_item(I.Inventory(it, stk), f_, t_, d_)))(
        inv.item, inv.stack, frm2, to2, do2))
    consume = lambda inv, row, ok: I.Inventory(*jax.vmap(                                  # noqa: E731
        lambda it, stk, r, o: tuple(I.consume_one(I.Inventory(it, stk), jnp.argmax(it == r), o & jnp.any(it == r))))(
        inv.item, inv.stack, row, ok))
    inv = consume(inv, out.consume_row, out.consume_row >= 0)
    if jrw is not None:                                               # final pet evolution consumes the egg
        pet_rows = jnp.asarray([cat.row(i) for i in (1101, 1102, 1103)], jnp.int32)
        held_pet = jnp.max(jnp.where(jnp.isin(inv.item, pet_rows), inv.item, -1), axis=1)
        inv = consume(inv, held_pet, jrw.consume_pet & (held_pet >= 0))
    # Wards and trinkets (modern_wards): placement, hits, expiry, rewards, Control Ward use.
    lay = MW_layout()
    w0, wn = lay["ward0"], 2 * W.MAX_WARDS_PER_TEAM
    ids = jnp.asarray(cat.arrays.item_id)
    trow = inv.item[:, 6]
    trinket_id = jnp.where(trow >= 0, ids[jnp.clip(trow, 0, ids.shape[0] - 1)], 0)
    cw_row = cat.row(2055)
    control_count = jnp.sum(jnp.where(inv.item == cw_row, inv.stack, 0), axis=1)
    wreq = WD.WardRequest(kind=-jnp.ones((c,), jnp.int32) if orders.ward_kind is None else orders.ward_kind,
                          x=jnp.zeros((c,)) if orders.ward_x is None else orders.ward_x,
                          y=jnp.zeros((c,)) if orders.ward_y is None else orders.ward_y)
    wards, wev = WD.ward_step(s.wards, now=now, dt=jnp.float32(dt), request=wreq, x=x[:c], y=y[:c], team=s.team[:c],
                              alive=alive[:c], level=econ.level, trinket_id=trinket_id, control_count=control_count,
                              grid=cfg.ward_grid, can_use=alive[:c] & ~caps["stunned"][:c] & ~in_stasis,
                              trinket_haste=st.item_haste + st.trinket_haste, hits=sc.ward_hits,
                              hitter=sc.ward_hitter, rune_pages=cfg.rune_pages, ward_visible=s0.visible[:, w0:w0 + wn])
    econ = econ._replace(gold=econ.gold + wev.gold, gold_total=econ.gold_total + wev.gold, xp=econ.xp + wev.xp)
    inv = consume(inv, jnp.full((c,), cw_row, jnp.int32), wev.consumed_control)
    grant_row = jnp.argmax(jnp.asarray(cat.arrays.item_id)[None, :] == ro.grant_item[:, None], axis=1)
    free = (inv.item[:, :6] < 0)
    can_grant = (ro.grant_item > 0) & jnp.any(free, axis=1)
    gslot = jnp.argmax(free, axis=1)
    put = can_grant[:, None] & (jnp.arange(7)[None, :] == gslot[:, None])
    inv = I.Inventory(jnp.where(put, grant_row[:, None], inv.item).astype(jnp.int32),
                      jnp.where(put, 1, inv.stack).astype(jnp.int32))
    lock = jnp.maximum(champ.cast_lock_until, jnp.where(kit_all.cast_started, now + kit_all.cast_lockout, 0.0))
    act = out.active                                                  # item actives: Hydra casts (cast time)
    lock = jnp.maximum(lock, jnp.where(act.used & ~act.can_move, now + act.cast_time, 0.0))
    item_lock = jnp.maximum(champ.item_cast_until, jnp.where(act.used & act.can_move, now + act.cast_time, 0.0))
    cast_now = kit_all.cast_started[:, None] & (jnp.arange(4)[None, :] == kit_all.cast_slot[:, None])
    last_dmg = jnp.where(sc.took_health[:c], now, champ.last_damaged)
    champ = champ._replace(
        cooldowns=cds, mana=mana, cast_lock_until=lock, item_cast_until=item_lock, last_damaged=last_dmg, inventory=inv,
        dyn=out.dynamic_stats, homeguard_ms=eco.homeguard_ms, blinked=s_out.blinked | sc.dstart,
        forbid=ro.forbid_purchase, reset_next=out.effects.attack_reset,
        bonus_points=champ.bonus_points + ro.skill_points, granted=jnp.where(can_grant, ro.grant_item, 0),
        cs=champ.cs + eco.kills.minion_kill.astype(jnp.int32),
        last_cast=jnp.where(cast_now, now, champ.last_cast),
        moving=jnp.where(respawn | recall | champ_dead, False, champ.moving),
        attack_order=jnp.where(respawn | recall | champ_dead, -1,
                               jnp.where(_kit_attack(kit_all) >= 0, _kit_attack(kit_all), champ.attack_order)))
    kills_next = eco.kills
    # Dead minions free their slot; dead champions keep theirs.
    kind = jnp.where(sc.minion_died, W.KIND_NONE, s.kind)
    # Ward units mirror the ward slots.
    view, _ = WD.ward_view(wards, now=now, x=x[:c], y=y[:c], team=s.team[:c], alive=alive[:c], level=econ.level)
    wsl = slice(w0, w0 + wn)
    kind = kind.at[wsl].set(jnp.where(view.alive, W.KIND_WARD, W.KIND_NONE))
    alive = alive.at[wsl].set(view.alive)
    x, y = x.at[wsl].set(view.x), y.at[wsl].set(view.y)
    hp, max_hp = hp.at[wsl].set(view.hp), max_hp.at[wsl].set(view.max_hp)
    w_sub = s.sub.at[wsl].set(view.sub)
    w_team = s.team.at[wsl].set(view.team)
    w_seq = s.spawn_seq.at[wsl].set((1 << 24) + wards.slots.seq)
    w_radius = s.radius.at[wsl].set(1.0)
    w_targ = s.targetable.at[wsl].set(view.alive)
    # Units left inside terrain that closed (inhibitor respawn, Rift transformation) step out. The test
    # disk is the movement clearance (route radius), the disk the MOVE clamp keeps walkable: the full
    # unit radius pulled champions walking along a structure pad back every tick.
    mobile = alive & ((kind == W.KIND_CHAMPION) | (kind == W.KIND_MINION))
    x, y = DTR.eject(x, y, jnp.clip(s.team, 0, 1), jnp.minimum(s.radius, cfg.routes.radius), sc.terrain,
                     active=mobile)
    return s, sc._replace(alive=alive, x=x, y=y, hp=hp, max_hp=max_hp, champ=champ, econ=econ, wards=wards,
                          kills=kills_next, kind=kind, sub=w_sub, team=w_team, spawn_seq=w_seq, radius=w_radius,
                          targetable=w_targ, cast_now=cast_now)


def _fog(s: ModernState, orders: ModernOrders, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickScratch]:
    """11. FOG: attack reveal, then next tick's visibility (modern_vision); enemy-seen casts."""
    c = N_CHAMPIONS
    now, s0, x, y, champ = sc.now, sc.s0, sc.x, sc.y, sc.champ
    enemy_team = 1 - s.team[:c]
    hidden = ~s0.visible[enemy_team, jnp.arange(c)] & s0.alive[:c]
    struck = sc.launched[:c] | (sc.kit_all.cast_started & (sc.cast_order.target >= 0))
    reveal = MV.reveal_step(s.reveal, hidden, struck, x[:c], y[:c], now)
    vis_next, sight_next = _visibility(cfg, x, y, sc.kind, sc.sub, sc.team, sc.alive, reveal, now, wards=sc.wards,
                                       level=sc.econ.level, variant=s.terrain_variant, jungle=sc.jungle)
    witnessed = vis_next[enemy_team, jnp.arange(c)] | ~hidden
    champ = champ._replace(seen_cast=jnp.where(sc.cast_now & witnessed[:, None], now, champ.seen_cast))
    return s, sc._replace(reveal=reveal, visible=vis_next, sight=sight_next, champ=champ)


def _commit(s: ModernState, cfg: WorldConfig, sc: TickScratch) -> tuple[ModernState, TickEvents]:
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


def IE_own(inv) -> Any:
    return I.owned_counts(inv)


def decimal_team_level(s: ModernState) -> Any:
    """(2,) decimal champion level per team (one champion per team in the lane world)."""
    return E.decimal_level(s.econ.xp, jnp.where(s.econ.quest.complete, 20, 18))[jnp.asarray([0, 1])]
