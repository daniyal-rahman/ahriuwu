"""World state of the 26.19 modern world: ``ModernState`` (unit columns + every subsystem's state),
the per-champion ``ChampionLayer``, ``ModernOrders`` (one order per champion per tick), ``TickEvents``,
and ``init_state`` (the game at 0:00 for a ``WorldConfig``)."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .. import champions as K
from .. import economy as E
from .. import mechanics as M
from .. import vision as MV
from .. import wards as WD
from ..champions import summoners as S
from ..combat import CombatState, init_combat
from ..core import damage as D
from ..core import types as W
from ..core.stat_pipeline import compose
from ..items import inventory as I
from ..items.catalog import catalog, combine_stats, zero_stats
from ..items.effects.core import Kills
from ..items.effects.runtime import UnitStatus, init_status
from ..jungle import camps as J
from ..jungle import objectives as OBJ
from ..lane import ai as LA
from ..lane import minions as MM
from . import views as V
from .config import N_CHAMPIONS, WorldConfig


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
    inventory: Any          # items.inventory.Inventory (C, 7)
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


class LastTick(NamedTuple):
    """Events of the previous tick that this tick reads: the world's one-tick lags in one place
    (``tick.commit`` writes them; docs/modern/WORLD_IMPLEMENTATION.md). Other carried values: the
    start-of-tick fog ``ModernState.visible``/``sight`` and ``ChampionLayer.dyn``/``reset_next``."""
    damage_matrix: Any      # (N, N) bool: unit i damaged unit j (lane/monster AI aggro, Scorchclaw)
    death_seen: Any         # (C, N) bool: champion c had own sight of unit j when j died (Overgrowth)
    kills: Kills            # takedowns credited (item/rune hooks)
    epic: Any               # (C,) epic-monster takedowns (rune events)
    large: Any              # (C,) large-monster kills (rune events)
    pending_dash: Any       # W.Dash: item-active dash (Rocketbelt) that starts this tick


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
    magic_resist: Any
    attack_damage: Any
    attack_range: Any       # stat range (edge to edge, core.types.WorldUnits)
    attack_speed: Any       # attacks per second
    move_speed: Any
    windup: Any
    spawn_seq: Any
    spawn_time: Any
    targetable: Any
    missile_speed: Any      # (N,) ranged attack missile speed, 0 = melee
    bounty_gold: Any        # lane-minion gold / XP / level fixed at spawn
    bounty_xp: Any
    bounty_level: Any
    next_seq: Any           # () int32
    spawn: Any              # lane.minions.LaneSpawnState: per (team, lane) wave cursors
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
    prev: LastTick          # what the previous tick did, read by this one
    visible: Any            # (2, N) bool: team t sees unit j (vision, end of last tick)
    sight: Any              # (C, N) bool: champion c's own sight of unit j (rune "own sight")
    reveal: Any             # vision.Reveal: attack-reveal circle per champion
    jungle: Any             # jungle.camps.JungleState (camps, Smite, pets, red/blue)
    obj: Any                # jungle.objectives.ObjectiveState (grubs, Herald, drakes, Elder, Baron)
    wards: Any              # wards.Wards (ward slots, trinkets)
    amove: Any              # AttackMove: per-champion attack-move order state
    route_anchor: Any       # (N,) int32 route node each unit steers by (mechanics.move_step); -1 = none
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
    static = combine_stats(I.inventory_stats(inv), V.shard_stats(cfg, level))
    st = compose(cfg.champion_base, level, static, adaptive_physical=cfg.adaptive_physical)
    hp = hp.at[:c].set(st.max_hp)
    zc = jnp.zeros((c,), jnp.float32)
    ranks = jnp.zeros((c, 4), jnp.int32)
    first = jnp.asarray([V.skill_order(cfg, i)[0] for i in range(c)])
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
    radius = jnp.where(kind == W.KIND_CHAMPION, 65.0, jnp.where(structure, towers.radius, 48.0)).astype(jnp.float32)
    alive = (kind == W.KIND_CHAMPION) | structure
    reveal = MV.init_reveal(c)
    trinkets = [next((i for i in lo.items if i in (3340, 3363, 3364)), 3340) for lo in cfg.loadouts]
    wards = WD.init_wards(c, trinkets)
    jungle = J.init_jungle(cfg.jungle, c, n, seed=seed) if cfg.jungle is not None else None
    obj = OBJ.init_objectives(cfg.objectives, n, c, jax.random.PRNGKey(seed + 1)) if cfg.objectives is not None \
        else None
    zc0 = jnp.zeros((c,), jnp.float32)
    amove = AttackMove(jnp.zeros((c,), bool), zc0, zc0, jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), jnp.int32))
    vis, sight = V.visibility(cfg, cfg.unit_x, cfg.unit_y, kind, cfg.unit_sub, cfg.unit_team, alive, reveal,
                             jnp.float32(0.0), wards=wards)
    return ModernState(
        t=jnp.float32(0.0), tick=jnp.int32(0), key=jax.random.PRNGKey(seed),
        kind=kind, sub=cfg.unit_sub, team=cfg.unit_team, alive=alive, x=cfg.unit_x, y=cfg.unit_y,
        hp=hp, max_hp=hp, radius=radius,
        armor=jnp.where(structure, towers.armor, 0.0).astype(jnp.float32).at[:c].set(st.base_armor + st.bonus_armor),
        magic_resist=jnp.where(structure, towers.magic_resist, 0.0).astype(jnp.float32).at[:c].set(st.base_mr + st.bonus_mr),
        attack_damage=jnp.where(structure, towers.attack_damage, 0.0).astype(jnp.float32).at[:c].set(st.base_ad + st.bonus_ad),
        attack_range=jnp.where(structure & (kind == W.KIND_TURRET), towers.attack_range, 0.0).astype(jnp.float32)
        .at[:c].set(st.attack_range),
        attack_speed=jnp.where(kind == W.KIND_TURRET, towers.attack_speed, 0.0).astype(jnp.float32).at[:c].set(st.attack_speed),
        move_speed=jnp.zeros((n,), jnp.float32).at[:c].set(st.move_speed),
        windup=jnp.where(kind == W.KIND_TURRET, towers.windup, 0.0).astype(jnp.float32).at[:c].set(st.attack_windup),
        spawn_seq=jnp.arange(n, dtype=jnp.int32), spawn_time=jnp.zeros((n,), jnp.float32),
        targetable=alive & ~structure | (structure & towers.targetable),
        missile_speed=jnp.where(kind == W.KIND_TURRET, LA.T.MISSILE_SPEED, 0.0).astype(jnp.float32), bounty_gold=jnp.zeros((n,), jnp.float32), bounty_xp=jnp.zeros((n,), jnp.float32),
        bounty_level=jnp.ones((n,), jnp.int32), next_seq=jnp.int32(n), spawn=MM.init_lane_spawn(),
        att=W.init_attack_state(n), missiles=M.init_missiles(64), cc=M.init_cc(n), champ=champ,
        kits=K.init(c, n), summoners=S.init(loadout), combat=_init_combat(cfg, c, n),
        econ=E.init_economy(c, n, [lo.role for lo in cfg.loadouts]), lane_ai=LA.init_lane_ai(n), towers=towers,
        shields=D.init_shields(n), status=init_status(n),
        prev=LastTick(damage_matrix=jnp.zeros((n, n), bool), death_seen=jnp.zeros((c, n), bool),
                      kills=Kills(zc, zc, zc, jnp.zeros((c,), bool), jnp.zeros((c, n), bool)), epic=zc0, large=zc0,
                      pending_dash=W.no_dash(c)),
        visible=vis, sight=sight, reveal=reveal, jungle=jungle, obj=obj, wards=wards, amove=amove,
        route_anchor=M.init_route_anchor(n), terrain_variant=jnp.int32(0),
        game_over=jnp.asarray(False), winner=jnp.int32(-1))


def _init_combat(cfg: WorldConfig, c: int, n: int) -> CombatState:
    """``init_combat`` with the jungle items' epic-monster mask (the 8 objective slots)."""
    comb = init_combat(c, n)
    lay = cfg.layout
    epic = jnp.zeros((n,), bool).at[lay.epic0:lay.ward0].set(cfg.objectives is not None)
    items = comb.items
    return comb._replace(items=items._replace(jungle=items.jungle._replace(epic=epic)))


class TickEvents(NamedTuple):
    """Diagnostics of one tick (for tests, observations and rewards)."""
    report: Any             # combat main-pass Report
    follow_up: Any
    economy: Any            # economy.EconomyOut
    plates: Any             # lane.ai.PlateEvents
    launched: Any           # (N,) attacks launched
    packet_overflow: Any    # () dropped packets (must stay 0)
    missile_overflow: Any
    shop_code: Any          # (C,) buy/sell result code (0 ok)
