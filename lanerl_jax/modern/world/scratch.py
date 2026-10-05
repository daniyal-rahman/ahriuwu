"""``TickScratch``: the values one tick's phases pass to each other (``world/tick.py``)."""
from __future__ import annotations

from typing import Any, NamedTuple


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
