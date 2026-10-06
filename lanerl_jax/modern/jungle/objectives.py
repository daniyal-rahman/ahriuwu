"""Patch-26.19 epic objectives: Voidgrubs, Rift Herald, Drakes, Dragon Soul, Elder, Baron, their AI, team buffs,
rewards and the Elemental Rift.

Rules: docs/modern/OBJECTIVES.md (integration contract §9); values: ``data/26.19/objectives_client.json``
(``data/build_objectives.py``, each tagged CLIENT / WIKI / PATCH / INFERRED-M / INFERRED-L). No Atakhan, Blood
Roses or Feats of Strength on 26.19. Owns the world's 8-slot epic block (``ObjectiveTable.slot0`` = world index of
local slot 0):

    0      pit monster: Voidgrub A (8:00-14:45), Rift Herald (15:00-19:45), Baron Nashor (20:00+)
    1, 2   Voidgrubs B, C
    3      dragon pit: Elemental Drake or Elder Dragon
    4      Rift Herald Mercenary (team-owned, summoned with the Eye of the Herald)
    5..7   Voidmites: camp mites (neutral) or Hunger of the Void summons (team-owned)

Phases: ``objectives_step`` (OBJ), ``team_buff_stats`` (STATS), ``baron_minion_buffs`` (AI),
``objectives_attack`` (ATTACK), ``objectives_packet_mods`` (DAMAGE), ``objectives_after_damage`` (DEATH).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..core import damage as D
from ..core import types as W
from ..data import PATCH_DIR
from ..items.catalog import ItemStats, zero_stats
from ..lane.ai import minion_spawn_stats
from ..map.rift import variant_index

TABLE_PATH = PATCH_DIR / "objectives_client.json"

N_SLOTS = 8
S_PIT, S_GRUB_B, S_GRUB_C, S_DRAGON, S_MERC = 0, 1, 2, 3, 4
S_MITES = (5, 6, 7)

# Local ``mtype``; world ``sub`` = SUB_BASE + mtype, disjoint from jungle.camps.Monster.
T_NONE, T_GRUB, T_HERALD, T_BARON, T_DRAKE, T_ELDER, T_MERC, T_MITE, T_ALLY_MITE = range(9)
N_TYPES = 9
SUB_BASE = 64
# Elements (client MapFlagIndexOverride ids) and the Elder pseudo-element.
E_NONE, E_INFERNAL, E_MOUNTAIN, E_OCEAN, E_CLOUD, E_HEXTECH, E_CHEMTECH = range(7)
ELDER = 7
DRAKE_CHAR = {E_INFERNAL: "sru_dragon_fire", E_MOUNTAIN: "sru_dragon_earth", E_OCEAN: "sru_dragon_water",
              E_CLOUD: "sru_dragon_air", E_HEXTECH: "sru_dragon_hextech", E_CHEMTECH: "sru_dragon_chemtech"}
# Baron forms (pit terrain): 0 Hunting, 1 Territorial, 2 All-Seeing.
F_HUNTING, F_TERRITORIAL, F_ALL_SEEING = range(3)
# Baron ability rotation (WIKI): Acid Pool > Form > Acid Shot > Tentacle Knockup.
B_ACID_POOL, B_FORM, B_ACID_SHOT, B_TENTACLE = range(4)
INF = jnp.float32(1e9)


def _growth(level) -> Any:
    """Champion-style growth (L-1)(0.7025 + 0.0175(L-1)); matches 26.1 (Baron 17,792 at 11, 19,190 at 18)."""
    l = jnp.asarray(level, jnp.float32) - 1.0
    return l * (0.7025 + 0.0175 * l)


@dataclass(frozen=True)
class ObjectiveTable:
    """Host constants. Per-type arrays are indexed by ``mtype`` (9,), per-element ones by element (8,: 1..6
    drakes, 7 Elder)."""
    slot0: int
    hp: np.ndarray
    hp_lvl: np.ndarray
    ad: np.ndarray
    ad_lvl: np.ndarray
    armor: np.ndarray
    armor_lvl: np.ndarray
    mr: np.ndarray
    mr_lvl: np.ndarray
    ms: np.ndarray
    arange: np.ndarray
    aspeed: np.ndarray
    windup: np.ndarray
    radius: np.ndarray
    missile: np.ndarray
    min_level: np.ndarray
    leash: np.ndarray
    e_hp: np.ndarray            # per element (drake/Elder)
    e_hp_lvl: np.ndarray
    e_ad: np.ndarray
    e_aspeed: np.ndarray
    e_pct_current: np.ndarray   # bonus % current-HP physical on attacks
    pit: tuple
    baron_pos: tuple
    dragon_pos: tuple
    r: dict = field(default_factory=dict)       # rules: python floats/lists
    buffs: dict = field(default_factory=dict)   # client buff DataValues

    @property
    def slots(self) -> slice:
        return slice(self.slot0, self.slot0 + N_SLOTS)


@lru_cache(maxsize=4)
def load_table(slot0: int, path: str | None = None) -> ObjectiveTable:
    d = json.loads(Path(path or TABLE_PATH).read_text())
    ch, rules = d["characters"], {k: v["value"] for k, v in d["rules"].items()}
    names = {T_GRUB: "sru_horde", T_HERALD: "sru_riftherald", T_BARON: "sru_baron", T_DRAKE: "sru_dragon_fire",
             T_ELDER: "sru_dragon_elder", T_MERC: "sru_riftherald_mercenary", T_MITE: "sru_horde_mini",
             T_ALLY_MITE: "sru_horde_mini"}
    z = lambda: np.zeros(N_TYPES, np.float32)
    cols = {k: z() for k in ("hp", "hp_lvl", "ad", "ad_lvl", "armor", "armor_lvl", "mr", "mr_lvl", "ms", "arange",
                             "aspeed", "windup", "radius", "missile", "min_level", "leash")}
    radius_wiki = {T_GRUB: 70.0, T_MITE: 30.0, T_ALLY_MITE: 30.0}           # WIKI (no client override)
    missile = {T_GRUB: 1200.0, T_DRAKE: 1500.0, T_ELDER: 1500.0, T_BARON: 1500.0}   # INFERRED-L
    min_level = {T_GRUB: rules["grub_min_level"], T_HERALD: rules["herald_min_level"],
                 T_BARON: rules["baron_min_level"], T_DRAKE: rules["dragon_min_level"],
                 T_ELDER: rules["elder_min_level"], T_MERC: rules["herald_min_level"], T_MITE: rules["grub_min_level"],
                 T_ALLY_MITE: 1}
    leash = {T_GRUB: rules["grub_leash"], T_HERALD: rules["herald_leash"], T_BARON: 1e6,
             T_DRAKE: rules["dragon_leash"], T_ELDER: rules["dragon_leash"], T_MERC: 1e6, T_MITE: rules["grub_leash"],
             T_ALLY_MITE: 1e6}
    for t, nm in names.items():
        c = ch[nm]
        cols["hp"][t], cols["hp_lvl"][t] = c.get("hp", 0.0), c.get("hp_per_level", 0.0)
        cols["ad"][t], cols["ad_lvl"][t] = c.get("ad", 0.0), c.get("ad_per_level", 0.0)
        cols["armor"][t], cols["armor_lvl"][t] = c.get("armor", 0.0), c.get("armor_per_level", 0.0)
        cols["mr"][t], cols["mr_lvl"][t] = c.get("mr", 0.0), c.get("mr_per_level", 0.0)
        cols["ms"][t] = 0.0 if t == T_BARON else c.get("move_speed", 0.0)        # Baron does not move (WIKI)
        cols["arange"][t], cols["aspeed"][t] = c.get("attack_range", 0.0), c.get("attack_speed", 0.0)
        cols["windup"][t] = c.get("attack_cast_time") or 0.3
        cols["radius"][t] = c.get("radius", radius_wiki.get(t, 50.0))
        cols["missile"][t] = missile.get(t, 0.0)
        cols["min_level"][t] = min_level.get(t, 1)
        cols["leash"][t] = leash.get(t, 1000.0)
    cols["ms"][T_ALLY_MITE] = 325.0                                            # INFERRED-L (melee minion)
    e = lambda: np.zeros(8, np.float32)
    e_hp, e_hp_lvl, e_ad, e_as, e_pct = e(), e(), e(), e(), e()
    for el, nm in DRAKE_CHAR.items():
        c = ch[nm]
        e_hp[el], e_hp_lvl[el], e_ad[el], e_as[el] = c["hp"], c["hp_per_level"], c["ad"], c["attack_speed"]
        sp = next(iter(v for v in c["spells"].values() if "PercentCurrentHealthRatio" in v), {})
        e_pct[el] = sp.get("PercentCurrentHealthRatio", 0.0)
    c = ch["sru_dragon_elder"]
    e_hp[ELDER], e_hp_lvl[ELDER], e_ad[ELDER], e_as[ELDER] = c["hp"], c["hp_per_level"], c["ad"], c["attack_speed"]
    camps = d["camps"]
    rules["infernal_splash_radius"] = ch["sru_dragon_fire"]["spells"]["SRU_Dragon_Infernal_Attack"]["AoERadius"]
    return ObjectiveTable(slot0=int(slot0), **cols, e_hp=e_hp, e_hp_lvl=e_hp_lvl, e_ad=e_ad, e_aspeed=e_as,
                          e_pct_current=e_pct,
                          pit=(camps["Horde"]["x"], camps["Horde"]["y"]),
                          baron_pos=(camps["Baron"]["x"], camps["Baron"]["y"]),
                          dragon_pos=(camps["Dragon"]["x"], camps["Dragon"]["y"]),
                          r=rules, buffs=d["shared_spells"])


class ObjectiveState(NamedTuple):
    # schedule
    grubs_spawned: Any          # () bool
    herald_spawned: Any         # () bool
    baron_next: Any             # () next Baron spawn time (INF while alive)
    baron_form: Any             # () int32, -1 until the first Baron spawn
    dragon_next: Any            # () next dragon-pit spawn time (INF while alive)
    elements: Any               # (3,) int32 first drake, second drake, rift element (pre-rolled)
    drakes_killed: Any          # () int32 elemental drakes killed (both teams)
    rift_element: Any           # () int32 0 until decided
    rift_at: Any                # () time the rift transforms (INF)
    stacks: Any                 # (2, 7) int32 Dragon Slayer stacks per team and element
    soul: Any                   # (2,) int32 soul element (0 none)
    elder_ready: Any            # () bool: the next dragon-pit spawn is the Elder
    grub_stacks: Any            # (2,) int32 Touch of the Void stacks
    ability_rot: Any            # () int32 Baron rotation index
    # per champion (C,)
    baron_until: Any
    baron_ad: Any
    baron_ap: Any
    elder_until: Any
    eye_until: Any              # Eye of the Herald held until (-INF none)
    recall_charge: Any          # bool: one Empowered Recall (Glimpse of the Void)
    hunger_cd: Any
    infernal_cd: Any
    hextech_cd: Any
    cloud_until: Any
    cloud_cd: Any
    ocean_until: Any
    ocean_rate: Any             # HP per second of the active Ocean Soul restoration
    ocean_mana_rate: Any
    mountain_up: Any            # bool: Mountain Soul shield granted since the last damage
    soul_pending: Any           # (C, 2) int32 target of a pending Infernal / Hextech proc (-1)
    soul_amount: Any            # (C, 2) damage of the pending procs
    # per slot (8,)
    mtype: Any
    level: Any
    home: Any                   # (8, 2)
    aggro: Any
    target: Any                 # world unit, -1
    patience: Any
    soft_until: Any             # soft reset end (-INF none)
    hard: Any                   # bool: hard reset (ignores attackers)
    last_combat: Any
    cast_until: Any             # ability windup end (-INF none)
    cast_kind: Any              # int32 (Herald 1 charge, 2 swipe; merc 3 leap)
    cast_x: Any
    cast_y: Any
    cast_target: Any
    swipes: Any                 # int32 bitmask of used Herald swipes
    charged: Any                # bool: Herald opened with her charge
    eye_ready: Any              # Herald eye ready time
    facing: Any                 # (8, 2)
    attack_count: Any
    expire: Any                 # mites: despawn time
    next_wave: Any              # grubs: next Voidmite wave
    self_dmg_rate: Any          # grubs: Defensive Measures self damage per second
    self_dmg_until: Any
    leap_count: Any             # mercenary charges done
    leap_done: Any              # mercenary: last structure leapt at (-1)
    damaged_by: Any             # (8, C) last time champion c damaged the slot
    pending: Any                # (8,) bool: summon to realize next tick (Hunger of the Void Voidmites)
    owner: Any                  # (8,) int32 team of team-owned slots (Mercenary, Hunger Voidmites)
    heal_pending: Any           # (8,) HP to add to the slot next tick (grub Defensive Measures)
    # per world unit (N,)
    void_stacks: Any
    void_until: Any
    touch_until: Any
    touch_next: Any
    touch_dmg: Any
    touch_src: Any
    burn_until: Any
    burn_next: Any
    burn_dmg: Any
    burn_src: Any
    exec_at: Any
    exec_src: Any
    exec_lock: Any
    empowered: Any              # bool: Hand of Baron minion empowerment


def init_objectives(table: ObjectiveTable, n_units: int, n_champions: int, key) -> ObjectiveState:
    """``key`` pre-rolls three distinct drake elements (the third is the rift element) and the first Baron
    ability; the Baron form is rolled at his first spawn."""
    c, n, s = n_champions, n_units, N_SLOTS
    k1, k2 = jax.random.split(key)
    perm = jax.random.permutation(k1, jnp.arange(1, 7, dtype=jnp.int32))
    f, i = (lambda v, sh=(): jnp.full(sh, v, jnp.float32)), (lambda v, sh=(): jnp.full(sh, v, jnp.int32))
    b = lambda v, sh=(): jnp.full(sh, v, bool)
    home = jnp.asarray([table.pit, table.pit, table.pit, table.dragon_pos, table.pit, table.pit, table.pit,
                        table.pit], jnp.float32)
    return ObjectiveState(
        grubs_spawned=b(False), herald_spawned=b(False), baron_next=f(table.r["baron_spawn"]), baron_form=i(-1),
        dragon_next=f(table.r["dragon_first_spawn"]), elements=perm[:3], drakes_killed=i(0), rift_element=i(0),
        rift_at=f(1e9), stacks=i(0, (2, 7)), soul=i(0, (2,)), elder_ready=b(False), grub_stacks=i(0, (2,)),
        ability_rot=jax.random.randint(k2, (), 0, 4).astype(jnp.int32),
        baron_until=f(-1e9, (c,)), baron_ad=f(0, (c,)), baron_ap=f(0, (c,)), elder_until=f(-1e9, (c,)),
        eye_until=f(-1e9, (c,)), recall_charge=b(False, (c,)), hunger_cd=f(-1e9, (c,)), infernal_cd=f(-1e9, (c,)),
        hextech_cd=f(-1e9, (c,)), cloud_until=f(-1e9, (c,)), cloud_cd=f(-1e9, (c,)), ocean_until=f(-1e9, (c,)),
        ocean_rate=f(0, (c,)), ocean_mana_rate=f(0, (c,)), mountain_up=b(False, (c,)),
        soul_pending=i(-1, (c, 2)), soul_amount=f(0, (c, 2)),
        mtype=i(T_NONE, (s,)), level=i(1, (s,)), home=home, aggro=b(False, (s,)), target=i(-1, (s,)),
        patience=f(1, (s,)), soft_until=f(-1e9, (s,)), hard=b(False, (s,)), last_combat=f(-1e9, (s,)),
        cast_until=f(-1e9, (s,)), cast_kind=i(0, (s,)), cast_x=f(0, (s,)), cast_y=f(0, (s,)), cast_target=i(-1, (s,)),
        swipes=i(0, (s,)), charged=b(False, (s,)), eye_ready=f(0, (s,)),
        facing=jnp.tile(jnp.asarray([[1.0, 0.0]], jnp.float32), (s, 1)),
        attack_count=i(0, (s,)), expire=f(1e9, (s,)), next_wave=f(0, (s,)), self_dmg_rate=f(0, (s,)),
        self_dmg_until=f(-1e9, (s,)), leap_count=i(0, (s,)), leap_done=i(-1, (s,)), damaged_by=f(-1e9, (s, c)),
        pending=b(False, (s,)), owner=i(W.NEUTRAL, (s,)), heal_pending=f(0, (s,)),
        void_stacks=f(0, (n,)), void_until=f(-1e9, (n,)), touch_until=f(-1e9, (n,)), touch_next=f(0, (n,)),
        touch_dmg=f(0, (n,)), touch_src=i(-1, (n,)), burn_until=f(-1e9, (n,)), burn_next=f(0, (n,)),
        burn_dmg=f(0, (n,)), burn_src=i(-1, (n,)), exec_at=f(1e9, (n,)), exec_src=i(-1, (n,)),
        exec_lock=f(-1e9, (n,)), empowered=b(False, (n,)))


class SlotWrites(NamedTuple):
    """World rows of the 8 slots where ``write``; ``despawn`` frees a slot; ``relocate`` moves a live unit."""
    write: Any
    despawn: Any
    kind: Any
    sub: Any
    team: Any
    x: Any
    y: Any
    hp: Any
    max_hp: Any
    armor: Any
    magic_resist: Any
    attack_damage: Any
    attack_range: Any
    attack_speed: Any
    move_speed: Any
    radius: Any
    windup: Any
    missile_speed: Any
    new_seq: Any            # bool: a fresh spawn (assign a new spawn_seq, reset attack/CC state)
    relocate: Any
    rx: Any
    ry: Any


def unit_write(table: ObjectiveTable, w: SlotWrites, n_units: int) -> W.UnitWrite:
    """The ``write`` rows as a world ``UnitWrite`` (despawns, relocations and heals stay with the caller)."""
    full = lambda v: jnp.zeros((n_units,), jnp.asarray(v).dtype).at[table.slots].set(v)      # noqa: E731
    return W.UnitWrite(mask=full(w.write), new=full(w.write & w.new_seq),
                       **{f: full(getattr(w, f)) for f in W.UNIT_COLUMNS})


class StepOut(NamedTuple):
    writes: SlotWrites
    desired: Any            # (8,) int32 world attack target (-1 none), for the attack machine
    goal: Any               # (8, 2) movement goal
    move_speed: Any         # (8,)
    move_active: Any        # (8,) bool
    can_attack: Any         # (8,) bool (False while winding up an ability or resetting)
    packets: Any            # D.Packets: abilities, DoTs, executes, soul procs, leap, self damage
    cc: Any                 # W.CCOut (1, N): monster-sourced CC (no tenacity source champion)
    champion_cc: Any        # W.CCOut (C, N): champion-sourced CC (Hextech Soul slow)
    monster_heal: Any       # (8,) HP restored to the objective slots (reset regen, grub heal)
    heal: Any               # (C,) HP to restore this tick (Ocean drake / soul)
    mana: Any               # (C,)
    shield: Any             # (C,) Mountain Soul shield amount granted this tick (0 none)
    armor_reduction: Any    # (N,) flat armor and MR reduction (Baron's Void Corruption)
    empowered_recall: Any   # (C,) bool: next recall is Empowered (4 s)
    homeguard_bonus: Any    # (C,) extra homeguard MS fraction (Hand of Baron +50%)
    terrain_variant: Any    # () int32 index into map.rift.RiftTerrain


class Rewards(NamedTuple):
    gold: Any               # (C,)
    xp: Any                 # (C,)
    epic_takedown: Any      # (C,) float count (RuneEvents.epic_takedown, role-quest epic points)
    large_monster_kill: Any  # (C,) float count (RuneEvents.large_monster_kill)
    killed: Any             # (8,) bool slot killed this tick
    killer_team: Any        # (8,) int32
    mtype: Any              # (8,) int32 type of the killed slot
    element: Any            # (8,) int32 drake element (dragon slot), ELDER for the Elder
    soul_granted: Any       # (2,) bool
    rift_decided: Any       # () bool


class ChampInfo(NamedTuple):
    """(C,) numbers the soul effects scale with, from STATS."""
    level: Any
    bonus_ad: Any
    ap: Any
    bonus_hp: Any
    max_hp: Any
    max_mana: Any
    adaptive_physical: Any


class MinionBuffs(NamedTuple):
    empowered: Any          # (N,) bool
    bonus_range: Any
    bonus_ad: Any
    missile_speed: Any      # (N,) 0 = unchanged
    attack_speed_mult: Any
    ms_floor: Any           # (N,) minimum move speed (0 none)
    siege_structure_mult: Any   # (N,) multiplier on siege raw vs structures
    splash_radius: Any


def _world(table: ObjectiveTable):
    return table.slot0 + jnp.arange(N_SLOTS, dtype=jnp.int32)


def monster_level(levels, min_level) -> Any:
    """Average champion level rounded up, at least the monster's minimum (WIKI/PATCH)."""
    avg = jnp.ceil(jnp.mean(jnp.asarray(levels, jnp.float32)) - 1e-6)
    return jnp.maximum(avg, jnp.asarray(min_level, jnp.float32)).astype(jnp.int32)


def slot_stats(table: ObjectiveTable, mtype, element, level, now) -> dict:
    """Stats of an objective monster of ``mtype`` (drake ``element``) at ``level``."""
    t = jnp.clip(jnp.asarray(mtype, jnp.int32), 0, N_TYPES - 1)
    el = jnp.clip(jnp.asarray(element, jnp.int32), 0, 7)
    a = lambda arr: jnp.asarray(arr, jnp.float32)[t]
    g = _growth(level)
    dragon = (t == T_DRAKE) | (t == T_ELDER)
    hp = jnp.where(dragon, jnp.asarray(table.e_hp)[el] + jnp.asarray(table.e_hp_lvl)[el] * g,
                   a(table.hp) + a(table.hp_lvl) * g)
    ad = jnp.where(dragon, jnp.asarray(table.e_ad)[el], a(table.ad) + a(table.ad_lvl) * g)
    aspeed = jnp.where(dragon, jnp.asarray(table.e_aspeed)[el], a(table.aspeed))
    # Hunger of the Void Voidmites: 100% melee-minion HP (26.11), melee-minion AD (INFERRED-L).
    mm = minion_spawn_stats(0, now)
    ally = t == T_ALLY_MITE
    hp = jnp.where(ally, mm.max_hp, hp)
    ad = jnp.where(ally, mm.attack_damage, ad)
    return dict(hp=hp.astype(jnp.float32), ad=ad.astype(jnp.float32),
                armor=(a(table.armor) + a(table.armor_lvl) * g).astype(jnp.float32),
                mr=(a(table.mr) + a(table.mr_lvl) * g).astype(jnp.float32),
                arange=a(table.arange), aspeed=aspeed.astype(jnp.float32), mspeed=a(table.ms),
                radius=a(table.radius), windup=a(table.windup), missile=a(table.missile))


def _interp(points, x) -> Any:
    p = np.asarray(points, np.float32)
    return jnp.interp(jnp.asarray(x, jnp.float32), jnp.asarray(p[:, 0]), jnp.asarray(p[:, 1])).astype(jnp.float32)


def hand_of_baron_bonus(table: ObjectiveTable, now) -> tuple:
    """(AD, AP) latched when Baron dies: 12-48 AD / 20-80 AP over minutes 20-40 (WIKI)."""
    m = jnp.asarray(now, jnp.float32) / 60.0
    return _interp(table.r["hand_of_baron_ad"], m), _interp(table.r["hand_of_baron_ap"], m)


def elder_burn_total(now) -> Any:
    """Aspect of the Dragon burn: 75-225 true over 2.25 s, by minute 25-45 (WIKI/CLIENT)."""
    m = jnp.asarray(now, jnp.float32) / 60.0
    return (75.0 + 150.0 * jnp.clip((m - 25.0) / 20.0, 0.0, 1.0)).astype(jnp.float32)


def comeback_mult(table: ObjectiveTable, levels, team) -> Any:
    """(C,) epic XP multiplier: +25% per level below the enemy team's average, cap 2x (PATCH)."""
    lv = jnp.asarray(levels, jnp.float32)
    team = jnp.asarray(team)
    enemy = jnp.stack([jnp.sum(jnp.where(team != t, lv, 0.0)) / jnp.maximum(jnp.sum(team != t), 1) for t in (0, 1)])
    per, cap = table.r["comeback_xp"]
    deficit = jnp.maximum(enemy[jnp.clip(team, 0, 1)] - lv, 0.0)
    return jnp.minimum(1.0 + per * deficit, cap).astype(jnp.float32)


def terrain_variant(obj: ObjectiveState, now) -> Any:
    el = jnp.where(jnp.asarray(now) >= obj.rift_at, obj.rift_element, 0)
    return variant_index(el, jnp.maximum(obj.baron_form, 0))


def _dist(ax, ay, bx, by):
    return jnp.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _xy(units, i):
    return jnp.stack([units.x[i], units.y[i]])


def _structure(kind):
    return (kind == W.KIND_TURRET) | (kind == W.KIND_INHIBITOR) | (kind == W.KIND_NEXUS)


# ---- INPUT: spawns, schedule, AI, periodic effects --------------------------------------------

def _spawn_rows(table, obj, units, now, levels, key):
    """Spawns and despawns this tick: (obj, spawn, despawn, pos (8, 2), element)."""
    ws = _world(table)
    alive = units.alive[ws] & (units.kind[ws] == W.KIND_MONSTER)
    r = table.r
    pit_free = ~alive[S_PIT]
    in_combat = (now - obj.last_combat) < 5.0
    # Voidgrubs: 3 at 8:00, despawn 14:45 (14:55 in combat).
    grub_spawn = ~obj.grubs_spawned & (now >= r["grubs_spawn"]) & (now < r["grubs_despawn"])
    is_grub = alive & (obj.mtype == T_GRUB)
    grub_gone = is_grub & ((now >= r["grubs_despawn_in_combat"])
                           | ((now >= r["grubs_despawn"]) & ~jnp.any(in_combat & is_grub)))
    pit_free = pit_free | grub_gone[S_PIT]
    herald_spawn = ~obj.herald_spawned & (now >= r["herald_spawn"]) & (now < r["herald_despawn"]) & pit_free \
        & ~grub_spawn
    is_herald = alive & (obj.mtype == T_HERALD)
    herald_gone = is_herald & ((now >= r["herald_despawn_in_combat"]) | ((now >= r["herald_despawn"]) & ~in_combat))
    pit_free = (pit_free & ~herald_spawn) | herald_gone[S_PIT]
    baron_spawn = (now >= obj.baron_next) & pit_free & (now >= r["baron_spawn"])
    dragon_spawn = (now >= obj.dragon_next) & ~alive[S_DRAGON]
    mite = alive & ((obj.mtype == T_MITE) | (obj.mtype == T_ALLY_MITE))
    mite_gone = mite & (now >= obj.expire)
    despawn = grub_gone | herald_gone | mite_gone
    # Pre-rolled elements: first, second, then the rift element for the rest.
    nxt_el = jnp.where(obj.drakes_killed < 2, obj.elements[jnp.clip(obj.drakes_killed, 0, 2)], obj.rift_element)
    d_type = jnp.where(obj.elder_ready, T_ELDER, T_DRAKE)
    d_el = jnp.where(obj.elder_ready, ELDER, nxt_el)
    form = jnp.where(obj.baron_form >= 0, obj.baron_form, jax.random.randint(key, (), 0, 3).astype(jnp.int32))
    spawn = jnp.zeros((N_SLOTS,), bool).at[S_PIT].set(grub_spawn | herald_spawn | baron_spawn) \
        .at[S_GRUB_B].set(grub_spawn).at[S_GRUB_C].set(grub_spawn).at[S_DRAGON].set(dragon_spawn)
    mtype = obj.mtype.at[S_PIT].set(jnp.where(grub_spawn, T_GRUB, jnp.where(
        herald_spawn, T_HERALD, jnp.where(baron_spawn, T_BARON, obj.mtype[S_PIT]))))
    mtype = mtype.at[S_GRUB_B].set(jnp.where(grub_spawn, T_GRUB, mtype[S_GRUB_B]))
    mtype = mtype.at[S_GRUB_C].set(jnp.where(grub_spawn, T_GRUB, mtype[S_GRUB_C]))
    mtype = mtype.at[S_DRAGON].set(jnp.where(dragon_spawn, d_type, mtype[S_DRAGON]))
    mtype = jnp.where(despawn & ~spawn, T_NONE, mtype)
    off = jnp.asarray([[-150.0, 0.0], [150.0, 0.0], [0.0, 150.0]], jnp.float32)     # INFERRED-L grub spread
    pit = jnp.asarray(table.pit, jnp.float32)
    pos = jnp.stack([pit] * N_SLOTS)
    pos = pos.at[S_PIT].set(jnp.where(baron_spawn, jnp.asarray(table.baron_pos, jnp.float32),
                                      jnp.where(grub_spawn, pit + off[0], pit)))
    pos = pos.at[S_GRUB_B].set(pit + off[1]).at[S_GRUB_C].set(pit + off[2])
    pos = pos.at[S_DRAGON].set(jnp.asarray(table.dragon_pos, jnp.float32))
    min_lv = jnp.asarray(table.min_level)[mtype]
    lv = monster_level(levels, min_lv)
    element = jnp.zeros((N_SLOTS,), jnp.int32).at[S_DRAGON].set(d_el)
    obj = obj._replace(
        grubs_spawned=obj.grubs_spawned | grub_spawn, herald_spawned=obj.herald_spawned | herald_spawn,
        baron_next=jnp.where(baron_spawn, INF, obj.baron_next),
        baron_form=jnp.where(baron_spawn, form, obj.baron_form),
        dragon_next=jnp.where(dragon_spawn, INF, obj.dragon_next),
        mtype=mtype, level=jnp.where(spawn, lv, obj.level),
        home=jnp.where(spawn[:, None], pos, obj.home),
        aggro=jnp.where(spawn | despawn, False, obj.aggro), target=jnp.where(spawn | despawn, -1, obj.target),
        patience=jnp.where(spawn, 1.0, obj.patience), hard=jnp.where(spawn, False, obj.hard),
        soft_until=jnp.where(spawn, -INF, obj.soft_until),
        cast_until=jnp.where(spawn | despawn, -INF, obj.cast_until),
        swipes=jnp.where(spawn, 0, obj.swipes), charged=jnp.where(spawn, False, obj.charged),
        attack_count=jnp.where(spawn, 0, obj.attack_count),
        damaged_by=jnp.where(spawn[:, None], -INF, obj.damaged_by),
        next_wave=jnp.where(spawn, 0.0, obj.next_wave), self_dmg_until=jnp.where(spawn, -INF, obj.self_dmg_until))
    return obj, spawn, despawn, pos, element


def _writes_for(table, obj, spawn, despawn, pos, element, team, now, units, level_up):
    """SlotWrites for new spawns and out-of-combat level-ups (HP keeps its fraction)."""
    ws = _world(table)
    st = slot_stats(table, obj.mtype, element, obj.level, now)
    frac = jnp.where(units.max_hp[ws] > 0, units.hp[ws] / jnp.maximum(units.max_hp[ws], 1.0), 1.0)
    write = spawn | level_up
    hp = jnp.where(spawn, st["hp"], st["hp"] * frac)
    z = jnp.zeros((N_SLOTS,), jnp.float32)
    return SlotWrites(write=write, despawn=despawn & ~spawn, kind=jnp.full((N_SLOTS,), W.KIND_MONSTER, jnp.int32),
                      sub=(SUB_BASE + obj.mtype).astype(jnp.int32), team=team.astype(jnp.int32),
                      x=jnp.where(spawn, pos[:, 0], units.x[ws]), y=jnp.where(spawn, pos[:, 1], units.y[ws]),
                      hp=hp, max_hp=st["hp"], armor=st["armor"], magic_resist=st["mr"], attack_damage=st["ad"],
                      attack_range=st["arange"], attack_speed=st["aspeed"], move_speed=st["mspeed"],
                      radius=st["radius"], windup=st["windup"], missile_speed=st["missile"], new_seq=spawn,
                      relocate=jnp.zeros((N_SLOTS,), bool), rx=z, ry=z)


def _element_of_dragon(obj):
    """Element of the live dragon-pit occupant (ELDER for the Elder)."""
    nxt = jnp.where(obj.drakes_killed < 2, obj.elements[jnp.clip(obj.drakes_killed, 0, 2)], obj.rift_element)
    return jnp.where(obj.mtype[S_DRAGON] == T_ELDER, ELDER, nxt)


def objectives_step(obj: ObjectiveState, table: ObjectiveTable, units: W.WorldUnits, *, now, dt, levels,
                    damage_matrix, champ: ChampInfo, last_damaged, ult_cast=None, use_eye=None,
                    key=None) -> tuple[ObjectiveState, StepOut]:
    """Spawns, AI, periodic packets and soul effects. ``damage_matrix`` (N, N): i damaged j last tick;
    ``last_damaged`` (C,) last damage taken; ``ult_cast`` (C,) R cast (Cloud Soul); ``use_eye`` (C,) summon the
    Mercenary; ``key`` rolls the Baron form."""
    now = jnp.asarray(now, jnp.float32)
    c = obj.baron_until.shape[0]
    ws = _world(table)
    key = jax.random.PRNGKey(0) if key is None else key
    ult_cast = jnp.zeros((c,), bool) if ult_cast is None else ult_cast
    use_eye = jnp.zeros((c,), bool) if use_eye is None else use_eye
    levels = jnp.asarray(levels, jnp.int32)
    r = table.r
    obj, spawn, despawn, pos, element = _spawn_rows(table, obj, units, now, levels, key)
    team = jnp.where((obj.mtype == T_MERC) | (obj.mtype == T_ALLY_MITE), obj.owner, W.NEUTRAL).astype(jnp.int32)

    # Eye of the Herald -> Mercenary at the holder (slot 4), level = average champion level.
    holder_ok = use_eye & (now < obj.eye_until) & units.alive[:c]
    summon = jnp.any(holder_ok) & ~(units.alive[ws[S_MERC]] & (obj.mtype[S_MERC] == T_MERC))
    who = jnp.argmax(holder_ok)
    obj = obj._replace(eye_until=jnp.where(holder_ok & summon & (jnp.arange(c) == who), -INF, obj.eye_until))
    spawn = spawn.at[S_MERC].set(summon)
    pos = pos.at[S_MERC].set(_xy(units, who))
    team = team.at[S_MERC].set(jnp.where(summon, units.team[who], team[S_MERC]))
    obj = obj._replace(owner=obj.owner.at[S_MERC].set(team[S_MERC]))
    merc_lv = jnp.maximum(jnp.ceil(jnp.mean(levels.astype(jnp.float32)) - 1e-6),
                          r["herald_min_level"]).astype(jnp.int32)
    obj = obj._replace(mtype=obj.mtype.at[S_MERC].set(jnp.where(summon, T_MERC, obj.mtype[S_MERC])),
                       level=obj.level.at[S_MERC].set(jnp.where(summon, merc_lv, obj.level[S_MERC])),
                       leap_count=obj.leap_count.at[S_MERC].set(jnp.where(summon, 0, obj.leap_count[S_MERC])),
                       leap_done=obj.leap_done.at[S_MERC].set(jnp.where(summon, -1, obj.leap_done[S_MERC])),
                       home=obj.home.at[S_MERC].set(jnp.where(summon, pos[S_MERC], obj.home[S_MERC])),
                       cast_until=obj.cast_until.at[S_MERC].set(jnp.where(summon, -INF, obj.cast_until[S_MERC])))

    # Grub Voidmite waves (4 per grub every 12 s in combat) and Hunger summons share slots 5..7.
    alive = (units.alive[ws] & (units.kind[ws] == W.KIND_MONSTER) & ~despawn) | spawn
    grub_fight = alive & (obj.mtype == T_GRUB) & obj.aggro & (now >= obj.next_wave)
    want_mites = jnp.sum(grub_fight) * int(r["grub_mites_per_wave"])
    mite_free = jnp.asarray([~alive[s] & ~obj.pending[s] for s in S_MITES])
    rank = jnp.cumsum(mite_free) - 1
    mite_new = mite_free & (rank < want_mites)
    src_grub = jnp.argmax(grub_fight)
    for j, s in enumerate(S_MITES):
        spawn = spawn.at[s].set(mite_new[j])
        off = jnp.asarray([60.0 * (j - 1), 60.0], jnp.float32)
        pos = pos.at[s].set(jnp.where(mite_new[j], (_xy(units, ws[src_grub]) + off).astype(jnp.float32), pos[s]))
    obj = obj._replace(next_wave=jnp.where(grub_fight, now + r["grub_mite_wave_period"], obj.next_wave))
    mt = obj.mtype
    for j, s in enumerate(S_MITES):
        mt = mt.at[s].set(jnp.where(mite_new[j], T_MITE, mt[s]))
    obj = obj._replace(mtype=mt, level=jnp.where(spawn & (jnp.arange(N_SLOTS) >= 5),
                                                 monster_level(levels, r["grub_min_level"]), obj.level),
                       expire=jnp.where(spawn & (jnp.arange(N_SLOTS) >= 5), now + r["grub_mite_lifetime"],
                                        obj.expire),
                       aggro=jnp.where(spawn & (jnp.arange(N_SLOTS) >= 5), True, obj.aggro),
                       home=jnp.where((spawn & (jnp.arange(N_SLOTS) >= 5))[:, None], pos, obj.home))
    team = jnp.where(spawn & (obj.mtype == T_MITE), W.NEUTRAL, team)
    # Hunger of the Void Voidmites armed by objectives_after_damage last tick.
    pend = obj.pending & ~alive
    spawn = spawn | pend
    pos = jnp.where(pend[:, None], obj.home, pos)
    team = jnp.where(pend, obj.owner, team).astype(jnp.int32)
    obj = obj._replace(pending=jnp.zeros_like(obj.pending))

    # Out-of-combat level-ups (Delayed Evolution, 30 s; applied to every objective monster).
    target_lv = jnp.maximum(jnp.ceil(jnp.mean(levels.astype(jnp.float32)) - 1e-6).astype(jnp.int32),
                            jnp.asarray(table.min_level)[obj.mtype].astype(jnp.int32))
    lvl_up = alive & ~spawn & (target_lv > obj.level) & ((now - obj.last_combat) >= r["baron_evolution_delay"]) \
        & (obj.mtype != T_MERC) & (obj.mtype != T_MITE) & (obj.mtype != T_ALLY_MITE)
    obj = obj._replace(level=jnp.where(lvl_up, target_lv, obj.level))
    element = element.at[S_DRAGON].set(_element_of_dragon(obj))
    writes = _writes_for(table, obj, spawn, despawn, pos, element, team, now, units, lvl_up)
    obj = obj._replace(aggro=jnp.where(spawn & (obj.mtype == T_MERC), True, obj.aggro))

    # Local view of the objective slots after the writes.
    alive = ((units.alive[ws] & (units.kind[ws] == W.KIND_MONSTER)) | spawn) & ~writes.despawn
    sx = jnp.where(spawn, pos[:, 0], units.x[ws])
    sy = jnp.where(spawn, pos[:, 1], units.y[ws])
    shp = jnp.where(writes.write, writes.hp, units.hp[ws])
    smax = jnp.where(writes.write, writes.max_hp, units.max_hp[ws])
    sad = jnp.where(writes.write, writes.attack_damage, units.attack_damage[ws])
    obj, ai = _monster_ai(obj, table, units, ws, alive, sx, sy, shp, smax, sad, team, now, dt, damage_matrix)

    # Periodic effects: DoTs, Elder executes, grub self damage, soul procs, heals, shields.
    pk, armor_red, obj = _periodic_packets(obj, table, units, ws, alive, now, dt)
    pk = D.concat_packets(ai["packets"], pk)
    obj, soul_pk, ccc, heal, mana, shield = _soul_tick(obj, table, units, champ, now, dt, last_damaged, ult_cast)
    pk = D.concat_packets(pk, soul_pk)
    writes = writes._replace(relocate=ai["relocate"], rx=ai["rx"], ry=ai["ry"])
    recall = (now < obj.baron_until) | obj.recall_charge
    m_heal = jnp.where(alive, ai["reset_heal"] + obj.heal_pending, 0.0).astype(jnp.float32)
    obj = obj._replace(heal_pending=jnp.zeros_like(obj.heal_pending))
    out = StepOut(writes=writes, desired=ai["desired"], goal=ai["goal"], move_speed=ai["speed"],
                  move_active=ai["active"], can_attack=ai["can_attack"], packets=pk, cc=ai["cc"],
                  champion_cc=ccc, monster_heal=m_heal, heal=heal, mana=mana, shield=shield,
                  armor_reduction=armor_red, empowered_recall=recall,
                  homeguard_bonus=jnp.where(now < obj.baron_until, r["baron_homeguard_bonus"], 0.0),
                  terrain_variant=terrain_variant(obj, now))
    return obj, out


def _seg_dist(px, py, ax, ay, bx, by):
    """Distance of points p to segment a-b (broadcast)."""
    vx, vy = bx - ax, by - ay
    l2 = jnp.maximum(vx * vx + vy * vy, 1e-6)
    t = jnp.clip(((px - ax) * vx + (py - ay) * vy) / l2, 0.0, 1.0)
    return jnp.sqrt((px - ax - t * vx) ** 2 + (py - ay - t * vy) ** 2)


def _monster_ai(obj, table, units, ws, alive, sx, sy, shp, smax, sad, team, now, dt, damage_matrix):
    """Aggro, patience/leash resets, target choice, movement and Herald/Mercenary abilities."""
    r = table.r
    n = units.x.shape[0]
    c = obj.baron_until.shape[0]
    mtype = obj.mtype
    idx = jnp.arange(n)
    champ_unit = idx < c
    hostile = alive & ((mtype != T_MERC) & (mtype != T_ALLY_MITE))
    owned = alive & ~hostile
    # Attackers of each slot last tick (any non-neutral unit).
    hit_by = damage_matrix[:, ws].T & (units.team[None, :] != W.NEUTRAL)        # (8, N)
    hit = jnp.any(hit_by, axis=1) & alive
    first = jnp.argmax(hit_by & champ_unit[None, :], axis=1).astype(jnp.int32)
    first = jnp.where(jnp.any(hit_by & champ_unit[None, :], axis=1), first,
                      jnp.argmax(hit_by, axis=1).astype(jnp.int32))
    hx, hy = obj.home[:, 0], obj.home[:, 1]
    leash = jnp.asarray(table.leash)[mtype]
    # Candidates: hostile monsters target champions (Baron and camp mites: any lane unit too).
    cand = units.alive & units.targetable & (units.team != W.NEUTRAL) & (units.kind != W.KIND_NONE)
    cand_champ = cand & (units.kind == W.KIND_CHAMPION)
    any_unit = (mtype == T_BARON) | (mtype == T_MITE)
    pool = jnp.where(any_unit[:, None], cand & ~_structure(units.kind)[None, :], cand_champ[None, :])  # (8, N)
    d_self = _dist(sx[:, None], sy[:, None], units.x[None, :], units.y[None, :])
    d_home = _dist(hx[:, None], hy[:, None], units.x[None, :], units.y[None, :])
    in_leash = d_home <= leash[:, None]
    baron = mtype == T_BARON
    # Baron attacks the nearest unit within his range (stationary); others the nearest champion in leash.
    reach = jnp.where(baron[:, None], d_self <= jnp.asarray(table.arange)[T_BARON] + 200.0, in_leash)
    near = jnp.where(pool & reach, d_self, jnp.inf)
    nearest = jnp.argmin(near, axis=1).astype(jnp.int32)
    has_near = jnp.isfinite(jnp.min(near, axis=1))
    # Soft-reset monsters ignore attackers outside the leash; hard reset ignores all.
    soft = now < obj.soft_until
    hit_inside = jnp.any(hit_by & in_leash, axis=1)
    wake = hostile & hit & ~obj.hard & (~soft | hit_inside)
    new_aggro = wake & ~obj.aggro
    aggro = jnp.where(hostile, obj.aggro | wake, obj.aggro) & alive
    soft = soft & ~(wake & soft)
    tgt = obj.target
    tc = jnp.clip(tgt, 0, n - 1)
    tgt_ok = (tgt >= 0) & pool[jnp.arange(N_SLOTS), tc] & reach[jnp.arange(N_SLOTS), tc]
    tgt = jnp.where(new_aggro, jnp.where(mtype == T_HERALD, first, nearest), tgt)
    tgt = jnp.where(aggro & ~tgt_ok & ~new_aggro, jnp.where(has_near, nearest, -1), tgt)
    tc = jnp.clip(tgt, 0, n - 1)
    valid = aggro & (tgt >= 0) & pool[jnp.arange(N_SLOTS), tc]
    # Patience (WIKI semantics, INFERRED-L rates): drains without a valid in-leash target.
    away = _dist(sx, sy, hx, hy)
    over = jnp.maximum(away - leash, 0.0) / jnp.maximum(leash, 1.0)
    bad = aggro & hostile & (~valid | ~in_leash[jnp.arange(N_SLOTS), tc]) & ~baron
    patience = jnp.where(bad, obj.patience - dt * r["patience_drain"] * (1.0 + over), obj.patience)
    patience = jnp.where(hit_inside & aggro, jnp.minimum(patience + 0.25, 1.0), patience)
    start_soft = aggro & (patience <= 0.0) & ~soft
    soft_until = jnp.where(start_soft, now + r["patience_soft_reset"], jnp.where(wake, -INF, obj.soft_until))
    soft = now < soft_until
    hard = obj.hard | (alive & hostile & ~soft & (obj.soft_until > -1e8) & (obj.soft_until <= now) & ~wake)
    resetting = soft | hard
    aggro = aggro & ~resetting
    tgt = jnp.where(resetting | ~aggro, -1, tgt)
    at_home = away < 50.0
    done = hard & at_home
    hard = hard & ~done
    patience = jnp.where(done | (~aggro & at_home), 1.0, jnp.maximum(patience, 0.0))
    soft_until = jnp.where(done, -INF, soft_until)
    heal_rate = jnp.where(hard, r["patience_hard_heal"], jnp.where(soft, r["patience_soft_heal"], 0.0))
    reset_heal = jnp.where(alive & hostile, jnp.minimum(heal_rate * smax * dt, smax - shp), 0.0)

    # Owned units (Mercenary, Hunger mites): nearest enemy structure, else nearest enemy minion.
    enemy = (units.team[None, :] != team[:, None]) & units.alive[None, :] & units.targetable[None, :] \
        & (units.team[None, :] != W.NEUTRAL)
    struct = enemy & _structure(units.kind)[None, :]
    lane = enemy & (units.kind == W.KIND_MINION)[None, :]
    ds = jnp.where(struct, d_self, jnp.inf)
    dm = jnp.where(lane & (d_self <= 800.0), d_self, jnp.inf)
    near_s = jnp.argmin(ds, axis=1).astype(jnp.int32)
    near_m = jnp.argmin(dm, axis=1).astype(jnp.int32)
    s_in = jnp.min(ds, axis=1) <= jnp.asarray(table.arange)[mtype] + 400.0
    own_t = jnp.where(jnp.isfinite(jnp.min(dm, axis=1)) & ~s_in, near_m,
                      jnp.where(jnp.isfinite(jnp.min(ds, axis=1)), near_s, -1))
    own_t = jnp.where(mtype == T_ALLY_MITE, jnp.where(jnp.isfinite(jnp.min(ds, axis=1)), near_s, -1), own_t)
    tgt = jnp.where(owned, own_t, tgt)
    tc = jnp.clip(tgt, 0, n - 1)

    # Herald: opening charge (2.5 s windup toward the first damager), swipes at 65.75% / 32.75%.
    casting = now < obj.cast_until
    is_her = alive & (mtype == T_HERALD)
    start_charge = is_her & new_aggro & ~obj.charged
    frac = shp / jnp.maximum(smax, 1.0)
    th = jnp.asarray(r["herald_swipe_thresholds"], jnp.float32)
    crossed = (frac[:, None] <= th[None, :]) & ((obj.swipes[:, None] >> jnp.arange(2)[None, :]) & 1 == 0)
    start_swipe = is_her & aggro & ~casting & ~start_charge & jnp.any(crossed, axis=1)
    swipes = jnp.where(start_swipe, obj.swipes | jnp.max(jnp.where(crossed, 1 << jnp.arange(2)[None, :], 0), axis=1),
                       obj.swipes)
    # Mercenary leap at a new structure within range (2.5 s windup).
    is_merc = alive & (mtype == T_MERC)
    leap_t = near_s
    leap_ok = is_merc & ~casting & jnp.isfinite(jnp.min(ds, axis=1)) \
        & (jnp.min(ds, axis=1) <= r["merc_leap_range"]) & (leap_t != obj.leap_done)
    start = start_charge | start_swipe | leap_ok
    kind = jnp.where(start_charge, 1, jnp.where(start_swipe, 2, jnp.where(leap_ok, 3, obj.cast_kind)))
    ft = jnp.where(leap_ok, leap_t, tc)
    cast_x = jnp.where(start, units.x[jnp.clip(ft, 0, n - 1)], obj.cast_x)
    cast_y = jnp.where(start, units.y[jnp.clip(ft, 0, n - 1)], obj.cast_y)
    windup = jnp.where(start_charge, r["herald_charge_windup"],
                       jnp.where(start_swipe, r["herald_swipe_windup"], r["merc_leap_windup"]))
    fire = alive & (obj.cast_until > -1e8) & (obj.cast_until <= now)
    cast_until = jnp.where(start, now + windup, jnp.where(fire | ~alive, -INF, obj.cast_until))
    cast_target = jnp.where(start, ft, obj.cast_target)
    casting = now < cast_until

    raw = jnp.zeros((N_SLOTS, n), jnp.float32)
    dtype = jnp.full((N_SLOTS, n), D.PHYSICAL, jnp.int32)
    knock = jnp.zeros((n,), jnp.float32)
    champs_only = (units.kind == W.KIND_CHAMPION)[None, :]
    foe = (units.team[None, :] != team[:, None]) & units.alive[None, :] & (units.team[None, :] != W.NEUTRAL) \
        & (units.kind[None, :] != W.KIND_NONE)
    # Charge: 200% AD to enemies along the path (width 250, INFERRED-L), knock aside 0.5 s (INFERRED-L).
    seg = _seg_dist(units.x[None, :], units.y[None, :], sx[:, None], sy[:, None], obj.cast_x[:, None],
                    obj.cast_y[:, None])
    f_charge = fire & (obj.cast_kind == 1)
    hit_c = f_charge[:, None] & foe & (seg <= 250.0)
    raw = raw + jnp.where(hit_c, r["herald_charge_ad_ratio"] * sad[:, None], 0.0)
    knock = jnp.maximum(knock, jnp.max(jnp.where(hit_c & champs_only, 0.5, 0.0), axis=0))
    # Swipe: 125% AD in a 350 cone/area in front (INFERRED-L radius).
    f_swipe = fire & (obj.cast_kind == 2)
    hit_s = f_swipe[:, None] & foe & (d_self <= 350.0)
    raw = raw + jnp.where(hit_s, r["herald_swipe_ad_ratio"] * sad[:, None], 0.0)
    # Non-champion bonus (pit Herald): +50%.
    raw = raw * jnp.where(~champs_only & (mtype == T_HERALD)[:, None], r["non_champion_damage_mult"], 1.0)
    # Mercenary leap: 3000 x (2/3)^k true to the structure, she loses 66% current HP.
    f_leap = fire & (obj.cast_kind == 3) & is_merc
    lt = jnp.clip(obj.cast_target, 0, n - 1)
    leap_alive = units.alive[lt] & units.targetable[lt]
    leap_hit = f_leap & leap_alive
    leap_dmg = r["merc_leap_damage"] * jnp.power(r["merc_leap_decay"], obj.leap_count.astype(jnp.float32))
    raw = raw + jnp.where(leap_hit[:, None] & (idx[None, :] == lt[:, None]), leap_dmg[:, None], 0.0)
    dtype = jnp.where(leap_hit[:, None], D.TRUE, dtype)
    self_dmg = jnp.where(leap_hit, r["merc_leap_self_damage"] * shp, 0.0)
    leap_count = jnp.where(leap_hit, obj.leap_count + 1, obj.leap_count)
    leap_done = jnp.where(f_leap, obj.cast_target, obj.leap_done)
    # Charge relocates the Herald to the charge point (dash approximated as an arrival, INFERRED-L).
    relocate = f_charge
    rx, ry = obj.cast_x, obj.cast_y

    valid_pk = (raw > 0) & alive[:, None]
    flags = jnp.where(dtype == D.TRUE, D.TAG_ACTIVE_SPELL, D.TAG_ACTIVE_SPELL | D.TAG_AOE)
    ab = D.packets(valid_pk, jnp.broadcast_to(ws[:, None], (N_SLOTS, n)),
                   jnp.broadcast_to(idx[None, :], (N_SLOTS, n)), raw, dtype, flags)
    selfp = D.packets(self_dmg > 0, ws, ws, self_dmg, D.TRUE, D.PROP_NO_DAMAGE_MOD)
    packets = D.concat_packets(ab, selfp)
    cc = W.no_cc(1, n)._replace(knockup=knock[None, :])

    # Movement: chase the target to attack range; reset -> home (x1.2 speed); casting -> stand.
    tx, ty = units.x[tc], units.y[tc]
    arange = jnp.asarray(table.arange)[mtype]
    dist_t = d_self[jnp.arange(N_SLOTS), tc]
    chase = (tgt >= 0) & (dist_t > arange + units.radius[tc] + jnp.asarray(table.radius)[mtype]) & ~casting
    gx = jnp.where(resetting | (~aggro & hostile), hx, jnp.where(chase, tx, sx))
    gy = jnp.where(resetting | (~aggro & hostile), hy, jnp.where(chase, ty, sy))
    speed = jnp.asarray(table.ms)[mtype] * jnp.where(resetting, r["patience_reset_ms_mult"], 1.0)
    active = alive & ~casting & (speed > 0) & (chase | resetting | (~aggro & hostile & (away > 50.0)))
    desired = jnp.where(alive & ~casting & ~resetting & (tgt >= 0), tgt, -1).astype(jnp.int32)
    face = jnp.stack([tx - sx, ty - sy], -1)
    fn = jnp.linalg.norm(face, axis=-1, keepdims=True)
    facing = jnp.where((tgt >= 0)[:, None] & (fn > 1e-3), face / jnp.maximum(fn, 1e-6), obj.facing)
    last_combat = jnp.where(hit | (aggro & valid), now, obj.last_combat)
    obj = obj._replace(aggro=aggro, target=tgt.astype(jnp.int32), patience=patience.astype(jnp.float32),
                       soft_until=soft_until, hard=hard, last_combat=last_combat, cast_until=cast_until,
                       cast_kind=kind.astype(jnp.int32), cast_x=cast_x, cast_y=cast_y,
                       cast_target=cast_target.astype(jnp.int32), swipes=swipes.astype(jnp.int32),
                       charged=obj.charged | start_charge, facing=facing, leap_count=leap_count.astype(jnp.int32),
                       leap_done=leap_done.astype(jnp.int32))
    return obj, dict(desired=desired, goal=jnp.stack([gx, gy], -1), speed=speed.astype(jnp.float32), active=active,
                     can_attack=alive & ~casting & ~resetting, packets=packets, cc=cc, relocate=relocate,
                     rx=rx, ry=ry, reset_heal=reset_heal)


def _periodic_packets(obj, table, units, ws, alive, now, dt):
    """Touch of the Void and Elder burns (0.5 s / 1 s ticks), Elder executes, grub self damage."""
    n = units.x.shape[0]
    idx = jnp.arange(n)
    # Touch of the Void: true damage every 0.5 s while active.
    t_due = (now < obj.touch_until + 1e-4) & (now >= obj.touch_next) & units.alive
    touch = D.packets(t_due, jnp.maximum(obj.touch_src, 0), idx, obj.touch_dmg, D.TRUE,
                      D.TAG_PERIODIC | D.TAG_PROC | D.TAG_DOES_NOT_AGGRO_JUNGLE)
    touch_next = jnp.where(t_due, obj.touch_next + table.r["touch_dot_tick"], obj.touch_next)
    # Elder burn: three ticks (0.25, 1.25, 2.25 s) of 1/3 of the total each.
    b_due = (obj.burn_until > now - 1e-4) & (now >= obj.burn_next) & units.alive
    burn = D.packets(b_due, jnp.maximum(obj.burn_src, 0), idx, obj.burn_dmg, D.TRUE, D.TAG_PERIODIC | D.TAG_PROC)
    burn_next = jnp.where(b_due, obj.burn_next + 1.0, obj.burn_next)
    # Elder Immolation execute (0.5 s after the trigger): true damage = max HP, PROP_EXECUTE.
    e_due = (obj.exec_at <= now) & units.alive
    ex = D.packets(e_due, jnp.maximum(obj.exec_src, 0), idx, units.max_hp, D.TRUE,
                   D.PROP_EXECUTE | D.PROP_NO_DAMAGE_MOD)
    # Grub Defensive Measures self damage.
    g_due = alive & (now < obj.self_dmg_until)
    self_g = D.packets(g_due, ws, ws, obj.self_dmg_rate * dt, D.TRUE, D.PROP_NO_DAMAGE_MOD | D.TAG_PERIODIC)
    # Void Corruption flat armor/MR reduction.
    red = jnp.where(now < obj.void_until, obj.void_stacks * table.r["baron_void_corruption_per_stack"], 0.0)
    obj = obj._replace(touch_next=touch_next, burn_next=burn_next, exec_at=jnp.where(e_due, INF, obj.exec_at),
                       exec_src=jnp.where(e_due, -1, obj.exec_src),
                       void_stacks=jnp.where(now < obj.void_until, obj.void_stacks, 0.0))
    return D.concat_packets(touch, burn, ex, self_g), red.astype(jnp.float32), obj


def _soul_tick(obj, table, units, champ, now, dt, last_damaged, ult_cast):
    """Pending Infernal/Hextech soul procs (decided last tick), Ocean restoration, Mountain shield,
    Cloud ult speed timer, Ocean drake stack regen."""
    n = units.x.shape[0]
    c = obj.baron_until.shape[0]
    idx = jnp.arange(n)
    b = table.buffs
    team = units.team[:c]
    soul = obj.soul[jnp.clip(team, 0, 1)]
    # Infernal: adaptive AoE (250) around the pending target.
    tgt = obj.soul_pending[:, 0]
    tc = jnp.clip(tgt, 0, n - 1)
    rad = b["SRX_DragonSoulBuffInfernal"]["BlastRadius"]
    near = (_dist(units.x[None, :], units.y[None, :], units.x[tc][:, None], units.y[tc][:, None]) <= rad) \
        & (units.team[None, :] != team[:, None]) & units.alive[None, :] & (units.kind[None, :] != W.KIND_NONE) \
        & ~_structure(units.kind)[None, :] & (tgt >= 0)[:, None]
    dt_inf = jnp.where(champ.adaptive_physical, D.PHYSICAL, D.MAGIC)
    inf_pk = D.packets(near, jnp.arange(c)[:, None], idx[None, :], obj.soul_amount[:, 0][:, None],
                       dt_inf[:, None], D.TAG_PROC | D.TAG_AOE)
    # Hextech: true damage to the primary target and up to 3 enemy champions within 600 of it.
    ht = obj.soul_pending[:, 1]
    hc = jnp.clip(ht, 0, n - 1)
    hb = b["SRX_DragonSoulBuffHextech"]
    dist_h = _dist(units.x[None, :], units.y[None, :], units.x[hc][:, None], units.y[hc][:, None])
    chain = (units.kind[None, :] == W.KIND_CHAMPION) & (units.team[None, :] != team[:, None]) & units.alive[None, :] \
        & (dist_h <= hb["BaseBounceRange"]) & (ht >= 0)[:, None]
    order = jnp.argsort(jnp.where(chain, dist_h, jnp.inf), axis=1)
    rank = jnp.argsort(order, axis=1)
    chain = chain & (rank < int(hb["BaseUnitsToHit"]))
    hex_pk = D.packets(chain, jnp.arange(c)[:, None], idx[None, :], obj.soul_amount[:, 1][:, None], D.TRUE,
                       D.TAG_PROC)
    base = jnp.where(units.attack_range > 300.0, hb["BaseSlowAmountRanged"], hb["BaseSlowAmountMelee"])
    slow = (base[None, :] + (0.5 * champ.bonus_hp / 100.0 + champ.ap / 100.0
                             + 3.0 * champ.bonus_ad / 100.0)[:, None]) / 100.0
    # The slow decays over 2 s; a constant slow at half strength stands in for it (INFERRED-M).
    ccc = W.no_cc(c, n)._replace(slow=jnp.where(chain, 0.5 * slow, 0.0),
                                 slow_duration=jnp.where(chain, hb["SlowDuration"], 0.0))
    obj = obj._replace(soul_pending=jnp.full_like(obj.soul_pending, -1), soul_amount=jnp.zeros_like(obj.soul_amount))
    # Ocean soul restoration (HoT) and Ocean drake stacks (2% missing HP per stack every 5 s).
    heal = jnp.where(now < obj.ocean_until, obj.ocean_rate * dt, 0.0)
    mana = jnp.where(now < obj.ocean_until, obj.ocean_mana_rate * dt, 0.0)
    ocean_stacks = obj.stacks[jnp.clip(team, 0, 1), E_OCEAN].astype(jnp.float32)
    period = b["SRX_DragonBuffOcean"]["HealFrequency"]
    pulse = jnp.floor(now / period + 1e-6) - jnp.floor((now - dt) / period + 1e-6)
    hp = units.hp[:c]
    heal = heal + jnp.where(units.alive[:c], pulse * b["SRX_DragonBuffOcean"]["HealingPercent"] * ocean_stacks
                            * jnp.maximum(units.max_hp[:c] - hp, 0.0), 0.0)
    # Mountain soul: shield after 5 s without damage.
    mb = b["SRX_DragonSoulBuffMountain"]
    calm = (now - last_damaged) >= mb["TimeWithoutTakingDamage"]
    grant = (soul == E_MOUNTAIN) & calm & ~obj.mountain_up & units.alive[:c]
    amount = mb["BaseShieldValue"] + mb["BonusADRatio"] * champ.bonus_ad + mb["APRatio"] * champ.ap \
        + mb["BonusHPRatio"] * champ.bonus_hp
    shield = jnp.where(grant, amount, 0.0)
    # Cloud soul: +45% MS for 6 s after R (30 s cooldown).
    cb = b["SRX_DragonSoulBuffCloud"]
    cloud = (soul == E_CLOUD) & ult_cast & (now >= obj.cloud_cd)
    obj = obj._replace(mountain_up=obj.mountain_up | grant,
                       cloud_until=jnp.where(cloud, now + cb["MSBuffDuration"], obj.cloud_until),
                       cloud_cd=jnp.where(cloud, now + cb["MSBuffCooldown"], obj.cloud_cd))
    return obj, D.concat_packets(inf_pk, hex_pk), ccc, heal.astype(jnp.float32), mana.astype(jnp.float32), \
        shield.astype(jnp.float32)


def team_buff_stats(obj: ObjectiveState, table: ObjectiveTable, team, alive, total_ad, ap, *, now,
                    out_of_combat) -> ItemStats:
    """(C,) bonus ItemStats from Dragon Slayer stacks (Infernal % AD/AP as flat from ``total_ad``/``ap``), Cloud
    Soul MS and Hand of Baron AD/AP (CLIENT values; Ocean heals through ``StepOut.heal``)."""
    b = table.buffs
    tm = jnp.clip(jnp.asarray(team), 0, 1)
    st = obj.stacks[tm].astype(jnp.float32)                    # (C, 7)
    soul = obj.soul[tm]
    pct = b["SRX_DragonBuffInfernal"]["ADandAPPercentIncrease"] * st[:, E_INFERNAL]
    baron = jnp.asarray(now) < obj.baron_until
    cloud_ms = b["SRX_DragonBuffCloud"]["MSAmountPerStack"] * st[:, E_CLOUD] * out_of_combat
    soul_ms = jnp.where(soul == E_CLOUD, b["SRX_DragonSoulBuffCloud"]["PersistentMSValue"]
                        + jnp.where(jnp.asarray(now) < obj.cloud_until, b["SRX_DragonSoulBuffCloud"]["MSAmount"]
                                    - b["SRX_DragonSoulBuffCloud"]["PersistentMSValue"], 0.0), 0.0)
    f = lambda v: jnp.where(alive, jnp.asarray(v, jnp.float32), 0.0)
    z = zero_stats(jnp.shape(team))
    return z._replace(
        attack_damage=f(pct * total_ad + jnp.where(baron, obj.baron_ad, 0.0)),
        ability_power=f(pct * ap + jnp.where(baron, obj.baron_ap, 0.0)),
        percent_armor=f(b["SRX_DragonBuffMountain"]["BonusDefenses"] * st[:, E_MOUNTAIN]),
        percent_magic_resist=f(b["SRX_DragonBuffMountain"]["BonusDefenses"] * st[:, E_MOUNTAIN]),
        slow_resist=f(b["SRX_DragonBuffCloud"]["SRAmountPerStack"] * st[:, E_CLOUD]),
        percent_move_speed=f(cloud_ms + soul_ms),
        ability_haste=f(b["SRX_DragonBuffHextech"]["AbilityHaste"] * st[:, E_HEXTECH]),
        attack_speed=f(b["SRX_DragonBuffHextech"]["AttackSpeed"] * st[:, E_HEXTECH]),
        tenacity=f(b["SRX_DragonBuffChemTech"]["TenacityPerStack"] * st[:, E_CHEMTECH]),
        heal_shield_power=f(b["SRX_DragonBuffChemTech"]["HealShieldPerStack"] * st[:, E_CHEMTECH]))


def baron_minion_buffs(obj: ObjectiveState, table: ObjectiveTable, units: W.WorldUnits, *, now) -> tuple:
    """Hand of Baron minion empowerment (WIKI hysteresis): an unempowered allied minion within 600 of a buffed
    champion empowers every allied minion within 1450 of it; a minion loses it with no buffed champion within
    1500. Hunger Voidmites count as melee minions. The DR is applied by ``objectives_packet_mods``."""
    r = table.r
    c = obj.baron_until.shape[0]
    buffed = (jnp.asarray(now) < obj.baron_until) & units.alive[:c]
    ws = _world(table)
    ally_mite = jnp.zeros_like(units.alive).at[ws].set(obj.mtype == T_ALLY_MITE)
    minion = units.alive & ((units.kind == W.KIND_MINION) | ally_mite)
    same = units.team[None, :] == units.team[:c, None]
    d = _dist(units.x[None, :], units.y[None, :], units.x[:c, None], units.y[:c, None])     # (C, N)
    trig = buffed & jnp.any(same & minion[None, :] & ~obj.empowered[None, :]
                            & (d <= r["baron_minion_acquire_radius"]), axis=1)
    gain = jnp.any(trig[:, None] & same & minion[None, :] & (d <= r["baron_minion_empower_radius"]), axis=0)
    keep = jnp.any(buffed[:, None] & same & (d <= r["baron_minion_lose_radius"]), axis=0)
    emp = minion & ((obj.empowered & keep) | gain)
    sub = jnp.where(ally_mite, 0, jnp.clip(units.sub, 0, 3))
    pick = lambda vals: jnp.asarray(vals, jnp.float32)[sub]
    near_ms = jnp.sum(jnp.where(buffed[:, None] & same & (d <= r["baron_minion_lose_radius"]),
                                units.move_speed[:c, None], 0.0), axis=0) \
        / jnp.maximum(jnp.sum(buffed[:, None] & same & (d <= r["baron_minion_lose_radius"]), axis=0), 1)
    frac, cap = r["baron_minion_ms_floor"]
    sb, sbonus = r["baron_minion_siege_structure"]
    e = lambda v: jnp.where(emp, v, 0.0).astype(jnp.float32)
    out = MinionBuffs(empowered=emp, bonus_range=e(pick(r["baron_minion_range"])),
                      bonus_ad=e(pick(r["baron_minion_ad"])), missile_speed=e(pick(r["baron_minion_missile_speed"])),
                      attack_speed_mult=jnp.where(emp, pick(r["baron_minion_attack_speed_mult"]),
                                                  1.0).astype(jnp.float32),
                      ms_floor=e(jnp.minimum(frac * near_ms, cap)),
                      siege_structure_mult=jnp.where(emp & (sub == 2), sb, 1.0).astype(jnp.float32),
                      splash_radius=e(jnp.where(sub == 2, r["baron_minion_siege_splash"], 0.0)))
    return obj._replace(empowered=emp), out


def objectives_attack(obj: ObjectiveState, table: ObjectiveTable, units: W.WorldUnits, launch: W.AttackLaunch, *,
                      now) -> tuple:
    """Basic attacks launched this tick: ``(obj, raw (N,), dtype, flags, extra Packets, cc CCOut(1, N))``, ``raw``
    non-zero only on objective slots. Riders: Infernal splash, Baron Corrosion (nearest unit with the fewest Void
    Corruption stacks) and his every-6th-attack ability rotation.
    """
    r = table.r
    n = units.x.shape[0]
    ws = _world(table)
    idx = jnp.arange(n)
    mtype = obj.mtype
    la = launch.launched[ws]
    tg = jnp.clip(launch.target[ws], 0, n - 1)
    ad = units.attack_damage[ws]
    t_hp = units.hp[tg]
    el = _element_of_dragon(obj)
    pct = jnp.where(mtype == T_DRAKE, jnp.asarray(table.e_pct_current)[el], 0.0)
    raw_s = ad + pct * t_hp
    raw_s = raw_s + jnp.where(mtype == T_HERALD, r["herald_on_hit_current_hp"] * t_hp, 0.0)
    raw_s = raw_s + jnp.where(mtype == T_MERC, r["merc_on_hit_current_hp"] * units.hp[ws], 0.0)
    epic = (mtype == T_GRUB) | (mtype == T_HERALD) | (mtype == T_DRAKE) | (mtype == T_ELDER) | (mtype == T_BARON)
    non_champ = units.kind[tg] != W.KIND_CHAMPION
    raw_s = raw_s * jnp.where(epic & non_champ, r["non_champion_damage_mult"], 1.0)
    raw_s = jnp.where(la, raw_s, 0.0)
    raw = jnp.zeros((n,), jnp.float32).at[ws].set(raw_s.astype(jnp.float32))
    dtype = jnp.full((n,), D.PHYSICAL, jnp.int32)
    flags = jnp.full((n,), D.BASIC_ATTACK, jnp.int32)

    foe = (units.team[None, :] != W.NEUTRAL) & units.alive[None, :] & (units.kind[None, :] != W.KIND_NONE) \
        & ~_structure(units.kind)[None, :]
    d_t = _dist(units.x[None, :], units.y[None, :], units.x[tg][:, None], units.y[tg][:, None])   # (8, N)
    # Infernal splash: the same damage to other units within 350 of the target.
    splash = (la & (mtype == T_DRAKE) & (el == E_INFERNAL))[:, None] & foe & (d_t <= r["infernal_splash_radius"]) \
        & (idx[None, :] != tg[:, None])
    raw_x = jnp.where(splash, raw_s[:, None], 0.0)
    dtype_x = jnp.full((N_SLOTS, n), D.PHYSICAL, jnp.int32)
    # Baron.
    bl = la[S_PIT] & (mtype[S_PIT] == T_BARON)
    bad = ad[S_PIT]
    bx, by = units.x[ws[S_PIT]], units.y[ws[S_PIT]]
    d_b = _dist(units.x, units.y, bx, by)
    in_b = foe[0] & (d_b <= jnp.asarray(table.arange)[T_BARON] + 200.0)
    # Corrosion: nearest unit with the fewest stacks.
    stacks_now = jnp.where(jnp.asarray(now) < obj.void_until, obj.void_stacks, 0.0)
    key_c = jnp.where(in_b, stacks_now * 1e5 + d_b, jnp.inf)
    corr = jnp.argmin(key_c)
    corr_ok = bl & jnp.isfinite(jnp.min(key_c))
    count = obj.attack_count.at[S_PIT].add(bl.astype(jnp.int32))
    ability = bl & (count[S_PIT] % int(r["baron_ability_every"]) == 0)
    rot = obj.ability_rot
    champs = units.kind == W.KIND_CHAMPION
    tgt_b = tg[S_PIT]
    d_tb = d_t[S_PIT]
    rad = r["baron_aoe_radius"]
    seg = _seg_dist(units.x, units.y, bx, by, units.x[tgt_b], units.y[tgt_b])
    pool_hit = ability & (rot == B_ACID_POOL)
    shot_hit = ability & (rot == B_ACID_SHOT)
    tent_hit = ability & (rot == B_TENTACLE)
    form_hit = ability & (rot == B_FORM)
    a_ratio = r["baron_ability_ad_ratio"]
    m_pool = pool_hit & foe[0] & champs & (d_tb <= rad)
    m_shot = shot_hit & foe[0] & (seg <= 150.0)
    m_tent = tent_hit & foe[0] & (d_tb <= 250.0)
    # Form ability (Hunting: 20% current HP to champions in range; Territorial: 100% AD pull
    # (PATCH; wiki 140%); All-Seeing: 140% AD rift to the two furthest champions within 2200).
    form = obj.baron_form
    far = jnp.where(champs & foe[0] & (d_b <= 2200.0), d_b, -jnp.inf)
    far_rank = jnp.argsort(jnp.argsort(-far))
    m_form_h = form_hit & (form == F_HUNTING) & champs & foe[0] & (d_b <= 2200.0)
    m_form_t = form_hit & (form == F_TERRITORIAL) & champs & foe[0] & (d_b <= 600.0)
    m_form_a = form_hit & (form == F_ALL_SEEING) & jnp.isfinite(far) & (far_rank < 2)
    raw_b = jnp.where(m_pool | m_shot | m_tent, a_ratio * bad, 0.0) + jnp.where(m_form_h, 0.20 * units.hp, 0.0) \
        + jnp.where(m_form_t, 1.0 * bad, 0.0) + jnp.where(m_form_a, 1.4 * bad, 0.0)
    raw_b = raw_b + jnp.where(corr_ok & (idx == corr), r["baron_corrosion_ad_ratio"] * bad, 0.0)
    raw_b = raw_b * jnp.where(~champs, r["non_champion_damage_mult"], 1.0)
    raw_x = raw_x.at[S_PIT].add(jnp.where(bl, raw_b, 0.0))
    dtype_x = dtype_x.at[S_PIT].set(jnp.where(bl, D.MAGIC, D.PHYSICAL))
    extra = D.packets(raw_x > 0, jnp.broadcast_to(ws[:, None], (N_SLOTS, n)),
                      jnp.broadcast_to(idx[None, :], (N_SLOTS, n)), raw_x, dtype_x, D.TAG_ACTIVE_SPELL | D.TAG_AOE)
    slow_s, slow_d = r["baron_acid_slow"]
    cc = W.no_cc(1, n)._replace(knockup=jnp.where(m_tent, r["baron_tentacle_knockup"], 0.0)[None, :],
                                slow=jnp.where(m_pool, slow_s, 0.0)[None, :],
                                slow_duration=jnp.where(m_pool, slow_d, 0.0)[None, :])
    # Void Corruption: +1 per Baron hit (+2 more on the frontal spit = the basic attack), 8 s.
    hit_b = (raw_b > 0) | ((idx == tgt_b) & bl)
    add = hit_b.astype(jnp.float32) + jnp.where((idx == tgt_b) & bl, 2.0, 0.0)
    vs = jnp.minimum(stacks_now + add, r["baron_void_corruption_max"])
    obj = obj._replace(attack_count=count, ability_rot=jnp.where(ability, (rot + 1) % 4, rot).astype(jnp.int32),
                       void_stacks=jnp.where(hit_b, vs, obj.void_stacks),
                       void_until=jnp.where(hit_b, jnp.asarray(now) + r["baron_void_corruption_duration"],
                                            obj.void_until))
    return obj, raw, dtype, flags, extra, cc


def objectives_packet_mods(obj: ObjectiveState, table: ObjectiveTable, packets: D.Packets, units: W.WorldUnits, *,
                           now) -> D.Packets:
    """Scale packet raw (multiplicative, like DMG.60) for Ancient Grudge, Baron's Gaze, Hand of Baron minion DR and
    Chemtech Soul. Executes are untouched."""
    r = table.r
    n = units.x.shape[0]
    c = obj.baron_until.shape[0]
    ws = _world(table)
    src, dst = jnp.clip(packets.src, 0, n - 1), jnp.clip(packets.dst, 0, n - 1)
    champ_src = (units.kind[src] == W.KIND_CHAMPION) & (packets.src < c)
    st = jnp.sum(obj.stacks, axis=1).astype(jnp.float32)                        # (2,)
    is_drake = (dst == ws[S_DRAGON]) & (obj.mtype[S_DRAGON] == T_DRAKE)
    grudge = jnp.where(is_drake & champ_src, 1.0 - r["dragon_vengeance_per_stack"]
                       * jnp.minimum(st[jnp.clip(units.team[src], 0, 1)], 4.0), 1.0)
    gaze = jnp.where((dst == ws[S_PIT]) & (obj.mtype[S_PIT] == T_BARON) & (src == obj.target[S_PIT]),
                     1.0 - r["baron_gaze_reduction"], 1.0)
    emp = obj.empowered[dst]
    sub = jnp.clip(units.sub[dst], 0, 3)
    m = jnp.asarray(now, jnp.float32) / 60.0
    (m0, v0), (m1, v1) = r["baron_minion_champion_dr"]
    dr_ch = v0 + (v1 - v0) * jnp.clip((m - m0) / (m1 - m0), 0.0, 1.0)
    melee_caster = sub <= 1
    aoe = D.has(packets.flags, D.TAG_AOE) | D.has(packets.flags, D.TAG_PERIODIC) | D.has(packets.flags, D.TAG_PROC)
    minion_src = units.kind[src] == W.KIND_MINION
    baron_dr = jnp.where(emp & melee_caster & champ_src, 1.0 - dr_ch, 1.0) \
        * jnp.where(emp & (sub == 0) & minion_src, 1.0 - r["baron_minion_melee_minion_dr"], 1.0) \
        * jnp.where(emp & melee_caster & aoe, 1.0 - r["baron_minion_aoe_dr"], 1.0)
    cb = table.buffs["SRX_DragonSoulBuffChemTech"]
    low = units.hp < cb["TriggerThreshold"] * units.max_hp
    chem = obj.soul[jnp.clip(units.team, 0, 1)] == E_CHEMTECH
    chem_src = champ_src & chem[src] & low[src]
    chem_dst = (units.kind[dst] == W.KIND_CHAMPION) & chem[dst] & low[dst]
    mult = grudge * gaze * baron_dr * jnp.where(chem_src, 1.0 + cb["DamageAmp"], 1.0) \
        * jnp.where(chem_dst, 1.0 - cb["DR"], 1.0)
    mult = jnp.where(D.has(packets.flags, D.PROP_EXECUTE), 1.0, mult)
    return packets._replace(raw=(packets.raw * mult).astype(jnp.float32))


def objectives_after_damage(obj: ObjectiveState, table: ObjectiveTable, units: W.WorldUnits, packets: D.Packets,
                            health_loss, *, died, killer, hp_after, now, levels, champ: ChampInfo) -> tuple:
    """Rewards, buffs and schedules from this tick's resolved ``packets``/``health_loss`` and deaths; arms Touch of
    the Void, Hunger summons (next tick), Elder burns/executes, Herald's eye, soul procs and grub heals."""
    r = table.r
    b = table.buffs
    now = jnp.asarray(now, jnp.float32)
    n = units.x.shape[0]
    c = obj.baron_until.shape[0]
    ws = _world(table)
    levels = jnp.asarray(levels, jnp.int32)
    valid = packets.valid & ((health_loss > 0) | (packets.raw > 0))
    src, dst = jnp.clip(packets.src, 0, n - 1), jnp.clip(packets.dst, 0, n - 1)
    champ_src = valid & (packets.src < c) & (units.kind[src] == W.KIND_CHAMPION)
    tm = jnp.clip(units.team, 0, 1)
    # Damage memory on objective slots (takedowns, XP participation, aggro already via damage_matrix).
    on_slot = (dst[:, None] == ws[None, :]) & champ_src[:, None]                # (P, 8)
    hit_sc = jnp.any(on_slot[:, :, None] & (src[:, None, None] == jnp.arange(c)[None, None, :]), axis=0)  # (8, C)
    damaged_by = jnp.where(hit_sc, now, obj.damaged_by)
    obj = obj._replace(damaged_by=damaged_by,
                       last_combat=jnp.where(jnp.any(on_slot, axis=0), now, obj.last_combat))

    # Herald eye: a champion basic attack from behind when ready -> 12% max HP true (next tick via burn slot).
    her = ws[S_PIT]
    is_her = obj.mtype[S_PIT] == T_HERALD
    fx, fy = obj.facing[S_PIT, 0], obj.facing[S_PIT, 1]
    behind = ((units.x[src] - units.x[her]) * fx + (units.y[src] - units.y[her]) * fy) < 0
    eye_hit = champ_src & (dst == her) & is_her & behind & D.has(packets.flags, D.TAG_BASIC_ATTACK)
    eye = jnp.any(eye_hit) & (now >= obj.eye_ready[S_PIT])
    eye_src = jnp.argmax(eye_hit)
    eye_dmg = r["herald_eye_max_hp_damage"] * units.max_hp[her]

    # Touch of the Void on structures: non-proc champion damage or Hunger Voidmite hits.
    ally_mite_src = jnp.zeros((n,), bool).at[ws].set(obj.mtype == T_ALLY_MITE)[src]
    struct_dst = _structure(units.kind[dst]) & (units.team[dst] != units.team[src])
    gs = obj.grub_stacks[tm[src]]
    touch = valid & struct_dst & ~D.has(packets.flags, D.TAG_PROC) & (gs > 0) & (champ_src | ally_mite_src)
    dmg_tab_m = jnp.asarray(b["SRT_2024_Horde_DoT"]["DamagePerTickMelee"], jnp.float32)
    dmg_tab_r = jnp.asarray(b["SRT_2024_Horde_DoT"]["DamagePerTickRanged"], jnp.float32)
    ranged_src = units.attack_range[src] > 300.0
    per = jnp.where(ranged_src, dmg_tab_r[jnp.clip(gs, 0, 6)], dmg_tab_m[jnp.clip(gs, 0, 6)])
    t_dst = jnp.zeros((n,), bool).at[dst].max(touch)
    t_dmg = jnp.zeros((n,), jnp.float32).at[dst].max(jnp.where(touch, per, 0.0))
    t_src = jnp.full((n,), -1, jnp.int32).at[dst].max(jnp.where(touch, src, -1).astype(jnp.int32))
    fresh = t_dst & (now >= obj.touch_until)
    obj = obj._replace(touch_until=jnp.where(t_dst, now + r["touch_dot_duration"], obj.touch_until),
                       touch_next=jnp.where(fresh, now + r["touch_dot_tick"], obj.touch_next),
                       touch_dmg=jnp.where(t_dst, t_dmg, obj.touch_dmg),
                       touch_src=jnp.where(t_dst, t_src, obj.touch_src))
    # Hunger of the Void: 3 stacks, champion damaged an enemy structure, 15 s cooldown -> Voidmite.
    hunger_c = jnp.zeros((c,), bool).at[jnp.clip(packets.src, 0, c - 1)].max(champ_src & struct_dst) \
        & (obj.grub_stacks[tm[:c]] >= 3) & (now >= obj.hunger_cd)
    mite_alive = units.alive[ws] & ((obj.mtype == T_MITE) | (obj.mtype == T_ALLY_MITE))
    free = jnp.asarray([~mite_alive[s] & ~obj.pending[s] for s in S_MITES])
    want = hunger_c & units.alive[:c]
    crank = jnp.cumsum(want) - 1
    frank = jnp.cumsum(free) - 1
    give = want & (crank < jnp.sum(free))
    # Pending summons are realized next tick by objectives_step via mtype/home/expire.
    mt, home, expire, aggro, lvl = obj.mtype, obj.home, obj.expire, obj.aggro, obj.level
    pend, owner = obj.pending, obj.owner
    for j, s in enumerate(S_MITES):
        who = jnp.argmax(give & (crank == frank[j]))
        ok = free[j] & jnp.any(give & (crank == frank[j]))
        mt = mt.at[s].set(jnp.where(ok, T_ALLY_MITE, mt[s]))
        home = home.at[s].set(jnp.where(ok, _xy(units, who), home[s]))
        expire = expire.at[s].set(jnp.where(ok, now + r["hunger_voidmite_lifetime"], expire[s]))
        aggro = aggro.at[s].set(jnp.where(ok, True, aggro[s]))
        lvl = lvl.at[s].set(jnp.where(ok, 1, lvl[s]))
        pend = pend.at[s].set(pend[s] | ok)
        owner = owner.at[s].set(jnp.where(ok, units.team[who], owner[s]))
    obj = obj._replace(hunger_cd=jnp.where(give, now + b["SRT_2024_Horde_Summoner"]["SummonCD"], obj.hunger_cd),
                       mtype=mt, home=home, expire=expire, aggro=aggro, level=lvl, pending=pend,
                       owner=owner.astype(jnp.int32))

    # Elder: burn + execute (enemy champions, INFERRED-M), 2 s per-target lockout.
    elder = (now < obj.elder_until)
    e_src = champ_src & elder[jnp.clip(packets.src, 0, c - 1)] & (units.kind[dst] == W.KIND_CHAMPION) \
        & (units.team[dst] != units.team[src]) & ~D.has(packets.flags, D.TAG_PROC)
    b_dst = jnp.zeros((n,), bool).at[dst].max(e_src)
    b_srcs = jnp.full((n,), -1, jnp.int32).at[dst].max(jnp.where(e_src, src, -1).astype(jnp.int32))
    burn_tot = elder_burn_total(now)
    obj = obj._replace(burn_until=jnp.where(b_dst, now + 2.25, obj.burn_until),
                       burn_next=jnp.where(b_dst, now + 0.25, obj.burn_next),
                       burn_dmg=jnp.where(b_dst, burn_tot / 3.0, obj.burn_dmg),
                       burn_src=jnp.where(b_dst, b_srcs, obj.burn_src))
    thr = b["ElderDragonBuff"]["ElderExecuteThresholdPercent"]
    arm = b_dst & (hp_after > 0) & (hp_after < thr * units.max_hp) & (now >= obj.exec_lock) \
        & (obj.exec_at > now + 1e3)
    obj = obj._replace(exec_at=jnp.where(arm, now + r["elder_execute_delay"], obj.exec_at),
                       exec_src=jnp.where(arm, b_srcs, obj.exec_src),
                       exec_lock=jnp.where(arm, now + b["ElderDragonBuff"]["PerTargetCooldown"], obj.exec_lock))
    # Herald eye damage rides the burn channel as one tick next tick (not the execute channel).
    obj = obj._replace(burn_until=obj.burn_until.at[her].set(jnp.where(eye, now + 0.5, obj.burn_until[her])),
                       burn_next=obj.burn_next.at[her].set(jnp.where(eye, now, obj.burn_next[her])),
                       burn_dmg=obj.burn_dmg.at[her].set(jnp.where(eye, eye_dmg, obj.burn_dmg[her])),
                       burn_src=obj.burn_src.at[her].set(jnp.where(eye, src[eye_src], obj.burn_src[her])),
                       eye_ready=obj.eye_ready.at[S_PIT].set(jnp.where(eye, now + r["herald_eye_cooldown"],
                                                                       obj.eye_ready[S_PIT])))

    # Souls (Infernal 3 s / Hextech 8 s procs on enemy champions or epic monsters; Ocean on any enemy).
    soul = obj.soul[tm[:c]]
    to_enemy = champ_src & (units.team[dst] != units.team[src]) & (units.kind[dst] != W.KIND_NONE)
    epic_dst = (units.kind[dst] == W.KIND_CHAMPION) \
        | ((units.kind[dst] == W.KIND_MONSTER) & jnp.any(dst[:, None] == ws[None, :4], axis=1))
    atk_or_spell = D.has(packets.flags, D.TAG_BASIC_ATTACK) | D.has(packets.flags, D.TAG_ACTIVE_SPELL)
    cs = jnp.clip(packets.src, 0, c - 1)

    def first_target(mask):
        m = jnp.zeros((c,), bool).at[cs].max(mask)
        t = jnp.full((c,), -1, jnp.int32).at[cs].max(jnp.where(mask, dst, -1).astype(jnp.int32))
        return m, t
    inf_m, inf_t = first_target(to_enemy & epic_dst & atk_or_spell & ~D.has(packets.flags, D.TAG_PROC))
    inf_ok = inf_m & (soul == E_INFERNAL) & (now >= obj.infernal_cd)
    ib = b["SRX_DragonSoulBuffInfernal"]
    inf_amt = ib["BaseDamage"] + ib["BonusADRatio"] * champ.bonus_ad + ib["APRatio"] * champ.ap \
        + ib["BonusHPRatio"] * champ.bonus_hp
    hex_m, hex_t = first_target(to_enemy & (units.kind[dst] == W.KIND_CHAMPION) & atk_or_spell
                                & ~D.has(packets.flags, D.TAG_PROC))
    hex_ok = hex_m & (soul == E_HEXTECH) & (now >= obj.hextech_cd)
    hex_amt = 25.0 + 25.0 * (jnp.clip(levels, 1, 18) - 1) / 17.0
    oc_m, oc_t = first_target(to_enemy & ~D.has(packets.flags, D.TAG_PROC))
    oc_ok = oc_m & (soul == E_OCEAN)
    oc_champ = units.kind[jnp.clip(oc_t, 0, n - 1)] == W.KIND_CHAMPION
    ob = b["SRX_DragonSoulBuffOcean"]
    eff = jnp.where(oc_champ, 1.0, ob["MinionPenalty"])
    oc_heal = (ob["BaseHealValue"] + ob["BonusADRatio"] * champ.bonus_ad + ob["APRatio"] * champ.ap
               + ob["BonusHPRatio"] * champ.bonus_hp) * eff
    oc_mana = (ob["BaseManaValue"] + ob["ManaRatioForTooltip"] * champ.max_mana) * eff
    fresh_oc = oc_ok & (now >= obj.ocean_until)
    obj = obj._replace(
        soul_pending=jnp.stack([jnp.where(inf_ok, inf_t, -1), jnp.where(hex_ok, hex_t, -1)], -1).astype(jnp.int32),
        soul_amount=jnp.stack([jnp.where(inf_ok, inf_amt, 0.0), jnp.where(hex_ok, hex_amt, 0.0)], -1)
        .astype(jnp.float32),
        infernal_cd=jnp.where(inf_ok, now + ib["ProcCooldown"], obj.infernal_cd),
        hextech_cd=jnp.where(hex_ok, now + b["SRX_DragonSoulBuffHextech"]["BaseTriggerCD"], obj.hextech_cd),
        ocean_until=jnp.where(oc_ok, now + ob["HealDuration"], obj.ocean_until),
        ocean_rate=jnp.where(fresh_oc, oc_heal / ob["HealDuration"], obj.ocean_rate),
        ocean_mana_rate=jnp.where(fresh_oc, oc_mana / ob["HealDuration"], obj.ocean_mana_rate),
        mountain_up=obj.mountain_up & ~jnp.zeros((n,), bool).at[dst].max(valid)[:c])

    # ---- objective deaths -> rewards, buffs, schedule ----
    s_alive_before = units.alive[ws] & (units.kind[ws] == W.KIND_MONSTER)
    killed = died[ws] & s_alive_before
    kil = jnp.clip(killer[ws], 0, n - 1)
    k_team = jnp.where(killer[ws] >= 0, units.team[kil], -1)
    k_champ = jnp.where((killer[ws] >= 0) & (killer[ws] < c), killer[ws], -1)
    # A non-champion final blow on a neutral objective credits the most recent damaging champion (INFERRED-M).
    last_c = jnp.argmax(obj.damaged_by, axis=1).astype(jnp.int32)
    has_c = jnp.max(obj.damaged_by, axis=1) > now - 10.0
    k_champ = jnp.where((k_champ < 0) & has_c, last_c, k_champ)
    k_team = jnp.where((k_team < 0) | (k_team == W.NEUTRAL),
                       jnp.where(k_champ >= 0, units.team[jnp.clip(k_champ, 0, n - 1)], -1), k_team)
    mtype = obj.mtype
    ctm = units.team[:c]
    on_team = (ctm[None, :] == k_team[:, None]) & killed[:, None]               # (8, C)
    sx, sy = units.x[ws], units.y[ws]
    near = _dist(units.x[None, :c], units.y[None, :c], sx[:, None], sy[:, None])
    is_k = (jnp.arange(c)[None, :] == k_champ[:, None]) & killed[:, None]
    lv_s = obj.level.astype(jnp.float32)
    cm = comeback_mult(table, levels, ctm)
    tch = lambda t: (mtype == t)[:, None]
    # Gold.
    gold = jnp.where(is_k & tch(T_GRUB), 30.0, 0.0)                              # 30 local (killer)
    gold = gold + jnp.where(is_k & (tch(T_HERALD) | tch(T_BARON) | tch(T_ELDER)), 100.0, 0.0)
    gold = gold + jnp.where(is_k & tch(T_DRAKE), 75.0, 0.0)
    gold = gold + jnp.where(on_team & (tch(T_BARON) | tch(T_ELDER)), 150.0, 0.0)
    gold = gold + jnp.where(is_k & tch(T_MERC), r["merc_kill_gold"], 0.0)
    gold = gold + jnp.where(is_k & tch(T_MITE), 1.0, 0.0)
    # XP.
    in2000 = near <= 2000.0
    dxp0, dxp1 = r["dragon_local_xp"]
    dxp = dxp0 + (dxp1 - dxp0) * jnp.clip((lv_s - 6.0) / 12.0, 0.0, 1.0)
    xp = jnp.where(on_team & in2000 & tch(T_GRUB), 65.0, 0.0)
    xp = xp + jnp.where(on_team & in2000 & tch(T_HERALD), 240.0, 0.0)
    xp = xp + jnp.where(on_team & in2000 & tch(T_DRAKE), dxp[:, None] * cm[None, :], 0.0)
    xp = xp + jnp.where(on_team & (tch(T_BARON) | tch(T_ELDER)), 650.0 * cm[None, :], 0.0)
    xp = xp + jnp.where(on_team & (near <= 600.0) & tch(T_MERC), 200.0, 0.0)
    # Takedowns: killer + team champions that damaged it within 10 s; grubs: only the first grub.
    part = on_team & ((obj.damaged_by > now - 10.0) | is_k)
    epic_k = (mtype == T_GRUB) | (mtype == T_HERALD) | (mtype == T_DRAKE) | (mtype == T_ELDER) | (mtype == T_BARON)
    first_grub = (mtype == T_GRUB) & (jnp.sum(obj.grub_stacks) == 0)
    first_grub = first_grub & (jnp.cumsum(killed & (mtype == T_GRUB)) == 1)
    epic_count = jnp.sum(jnp.where(part & (epic_k & ((mtype != T_GRUB) | first_grub))[:, None], 1.0, 0.0), axis=0)
    large = jnp.sum(jnp.where(is_k & ((mtype == T_GRUB) | (mtype == T_HERALD) | (mtype == T_MERC))[:, None], 1.0, 0.0),
                    axis=0)

    # Team buffs.
    kt = lambda t: jnp.any(killed & (mtype == t))
    team_of = lambda t: k_team[jnp.argmax(killed & (mtype == t))]
    alive_c = units.alive[:c] & ~died[:c]
    # Voidgrubs: Touch of the Void stacks (max 3), Defensive Measures heal on the other grubs.
    gk = killed & (mtype == T_GRUB)
    for tt in (0, 1):
        obj = obj._replace(grub_stacks=obj.grub_stacks.at[tt].set(jnp.minimum(
            obj.grub_stacks[tt] + jnp.sum(gk & (k_team == tt)).astype(jnp.int32), 3)))
    grub_alive = s_alive_before & ~killed & (mtype == T_GRUB)
    gh = jnp.any(gk) & grub_alive
    g_max, g_hp = units.max_hp[ws], hp_after[ws]
    g_heal = jnp.where(gh, r["grub_death_heal_max"] * g_max
                       + r["grub_death_heal_missing"] * jnp.maximum(g_max - g_hp, 0.0), 0.0)
    obj = obj._replace(heal_pending=obj.heal_pending + g_heal,
                       self_dmg_rate=jnp.where(gh, g_heal / r["grub_death_self_damage_duration"], obj.self_dmg_rate),
                       self_dmg_until=jnp.where(gh, now + r["grub_death_self_damage_duration"], obj.self_dmg_until))
    # Herald: Eye to the killer (300 s) and an Empowered Recall charge to participants.
    hk = kt(T_HERALD)
    hkc = k_champ[S_PIT]
    obj = obj._replace(eye_until=jnp.where(hk & (jnp.arange(c) == hkc), now + r["eye_duration"], obj.eye_until),
                       recall_charge=obj.recall_charge | (hk & part[S_PIT]))
    # Baron: Hand of Baron to living members (latched AD/AP), respawn in 6 min.
    bk = kt(T_BARON)
    bt = team_of(T_BARON)
    bad_, bap_ = hand_of_baron_bonus(table, now)
    give_b = bk & (ctm == bt) & alive_c
    obj = obj._replace(baron_until=jnp.where(give_b, now + r["baron_buff_duration"], obj.baron_until),
                       baron_ad=jnp.where(give_b, bad_, obj.baron_ad), baron_ap=jnp.where(give_b, bap_, obj.baron_ap),
                       baron_next=jnp.where(bk, now + r["baron_respawn"], obj.baron_next))
    # Dragons.
    dk = killed[S_DRAGON]
    el = _element_of_dragon(obj)
    dt_ = k_team[S_DRAGON]
    elemental = dk & (mtype[S_DRAGON] == T_DRAKE)
    elder_k = dk & (mtype[S_DRAGON] == T_ELDER)
    stacks = obj.stacks.at[jnp.clip(dt_, 0, 1), jnp.clip(el, 0, 6)].add(jnp.where(elemental & (dt_ >= 0), 1, 0))
    killed_n = obj.drakes_killed + elemental.astype(jnp.int32)
    decide = elemental & (killed_n == int(r["dragons_to_terrain_change"]))
    rift_el = jnp.where(decide, obj.elements[2], obj.rift_element)
    tot = jnp.sum(stacks, axis=1)
    new_soul = elemental & (obj.soul == 0) & (tot >= int(r["dragons_to_soul"])) & (jnp.arange(2) == dt_) \
        & ~jnp.any(obj.soul > 0)
    soul_now = jnp.where(new_soul, rift_el, obj.soul)
    elder_next = obj.elder_ready | jnp.any(new_soul)
    nxt = jnp.where(dk, now + jnp.where(elder_next, r["elder_respawn"], r["dragon_respawn"]), obj.dragon_next)
    give_e = elder_k & (ctm == dt_) & alive_c
    obj = obj._replace(stacks=stacks, drakes_killed=killed_n, rift_element=rift_el,
                       rift_at=jnp.where(decide, now + r["rift_transform_delay"], obj.rift_at), soul=soul_now,
                       elder_ready=elder_next, dragon_next=nxt,
                       elder_until=jnp.where(give_e, now + r["elder_buff_duration"], obj.elder_until))
    # Champion deaths drop Hand of Baron and Aspect of the Dragon.
    cd = died[:c]
    obj = obj._replace(baron_until=jnp.where(cd, -INF, obj.baron_until),
                       elder_until=jnp.where(cd, -INF, obj.elder_until),
                       mtype=jnp.where(killed, T_NONE, obj.mtype), aggro=jnp.where(killed, False, obj.aggro),
                       target=jnp.where(killed, -1, obj.target))
    rewards = Rewards(gold=jnp.sum(gold, axis=0).astype(jnp.float32), xp=jnp.sum(xp, axis=0).astype(jnp.float32),
                      epic_takedown=epic_count.astype(jnp.float32), large_monster_kill=large.astype(jnp.float32),
                      killed=killed, killer_team=k_team.astype(jnp.int32), mtype=mtype,
                      element=jnp.zeros((N_SLOTS,), jnp.int32).at[S_DRAGON].set(el), soul_granted=new_soul,
                      rift_decided=decide)
    return obj, rewards
