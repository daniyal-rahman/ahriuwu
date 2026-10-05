"""Patch-26.19 regular jungle camps, Rift Scuttler, monster AI, jungle rewards, Smite and jungle pets.

Spec: ``docs/modern/JUNGLE.md`` (evidence levels per value). Client data comes from
``modern/data/26.19/jungle_client.json`` (``data/build_modern_jungle.py``); wiki and
patch-note values are the constants below, each tagged CLIENT / WIKI / PATCH /
INFERRED-M / INFERRED-L.

Like ``lane.ai`` this module owns *decisions and rule state* only; the world
tick owns the unit arrays, the attack machine, missiles, movement and damage
resolution. Fixed shapes: S jungle slots (``JungleTable.n_slots`` = 38, inside the
first 40 of the world's ``MAX_MONSTERS`` monster slots; the last 8 are left to the
epic objectives), K = 14 camps, C champions (world units ``0..C-1``), N world units.
Slot ``s`` is world unit ``table.monster0 + s``.

Tick placement (``world.tick`` phases)::

    1 INPUT   state, w = spawn_step(state, tab, now=now, champion_level=lvl)      -> write_spawns(...)
    3 CASTS   state, sm = smite_step(state, tab, units, summoner_request, spells, now=..., ...)
    4 AI      state, ai = monster_ai(state, tab, units, att, now=now, dt=dt, damage_events=s.damage_matrix)
              (desired targets, move goals/speeds, reset heals, despawns, targetable)
    6 ATTACK  pk = monster_attack_packets(state, tab, units, launch)               (monster basic attacks)
              state, fx = combat_effects(state, tab, units, ictx, attack_hit=..., ...)  (red burn, pets, evolutions)
    9 DEATH   state, rw = death_step(state, tab, units, now=now, died=died, killer=killer, ...)
              (gold/XP per champion -> economy, crest grants/transfers, treats, splits, respawn timers)
    2 STATS   buff_stats(state, now=..., level=..., max_mana=..., max_hp=..., ...) next tick (AH, regen)

Nothing here compiles ``world.tick``; every function is pure JAX over the arrays it is given.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..core import damage as D
from ..core.stats import MAGIC, PHYSICAL, TRUE
from ..core.types import (KIND_CHAMPION, KIND_MINION, KIND_MONSTER, KIND_NONE, MAX_MONSTERS, NEUTRAL,
                          AttackLaunch, AttackState, CastOrder, CCOut, WorldUnits, no_cc)
from ..data import PATCH_DIR

DATA = PATCH_DIR / "jungle_client.json"
MAX_JUNGLE_SLOTS = MAX_MONSTERS - 8          # epic objectives own the last 8 monster slots


# ---- monster types (WorldUnits.sub of a KIND_MONSTER slot) --------------------------------------
class Monster:
    BLUE, RED, GROMP, WOLF, WOLF_MINI, RAPTOR, RAPTOR_MINI, KRUG, KRUG_MEDIUM, KRUG_MINI, SCUTTLE = range(11)


N_TYPES = 11
EPIC_SUB_BASE = 16              # proposal for the epic agent: epic monster ``sub`` values start here
CHARACTERS = ("SRU_Blue", "SRU_Red", "SRU_Gromp", "SRU_Murkwolf", "SRU_MurkwolfMini", "SRU_Razorbeak",
              "SRU_RazorbeakMini", "SRU_Krug", "SRU_KrugMini", "SRU_KrugMiniMini", "Sru_Crab")
SMALL, MEDIUM, LARGE = 0, 1, 2
SIZE = (LARGE, LARGE, LARGE, LARGE, SMALL, LARGE, SMALL, LARGE, MEDIUM, SMALL, LARGE)  # CLIENT unit tags
# Bonus damage on attacks, fraction of the target's current HP (WIKI camp pages "Notes").
BONUS_PHYS_CURRENT = (0.05, 0.05, 0.0, 0.03, 0.0, 0.03, 0.0, 0.03, 0.0, 0.0, 0.0)
BONUS_MAGIC_CURRENT = (0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
HP_CURVE = (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2)    # 0 template, 1 buff camps, 2 Rift Scuttler (WIKI)
DEFAULT_ATTACK_RANGE = 100.0                    # INFERRED-M: client record has no attackRange for Blue
MURKWOLF_MINI_AD = 10.0                         # WIKI (client record has no baseDamage)

# Level scaling (WIKI Template:Jungle monster stat; Blue/Red infobox; Rift Scuttler infobox).
AD_MULT = (1.0, 1.0, 1.1, 1.15, 1.2, 1.25, 1.35, 1.45, 1.55, 1.65, 1.8, 1.95, 2.1, 2.25, 2.4, 2.6, 2.8, 3.0)
XP_MULT = (1.0, 1.0, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5)          # capped at monster level 8 (V14.10)
CRAB_GOLD_MULT = (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2)
FIRST_CRAB_HP, FIRST_CRAB_XP = 0.65, 0.20       # WIKI: first Scuttlers 35% less HP, 80% less XP

# ---- camps ------------------------------------------------------------------------------------------
CAMP_BLUE, CAMP_RED, CAMP_GROMP, CAMP_WOLVES, CAMP_RAPTORS, CAMP_KRUGS, CAMP_SCUTTLE = range(7)
CAMP_NAMES = ("Order Blue", "Order Red", "Order OwlBear", "Order Wolves", "Order Wraiths", "Order Small Golems",
              "Chaos Blue", "Chaos Red", "Chaos OwlBear", "Chaos Wolves", "Chaos Wraiths", "Chaos Small Golems",
              "Baron Crab", "Dragon Crab")
CAMP_TYPE = (CAMP_BLUE, CAMP_RED, CAMP_GROMP, CAMP_WOLVES, CAMP_RAPTORS, CAMP_KRUGS) * 2 + (CAMP_SCUTTLE,) * 2
FIRST_SPAWN = {CAMP_BLUE: 55.0, CAMP_RED: 55.0, CAMP_WOLVES: 55.0, CAMP_RAPTORS: 55.0,   # PATCH 26.1
               CAMP_GROMP: 67.0, CAMP_KRUGS: 67.0, CAMP_SCUTTLE: 175.0}
RESPAWN = {CAMP_BLUE: 300.0, CAMP_RED: 300.0, CAMP_GROMP: 135.0, CAMP_WOLVES: 135.0,    # WIKI (26.1 Swiftplay
           CAMP_RAPTORS: 135.0, CAMP_KRUGS: 135.0, CAMP_SCUTTLE: 150.0}                # 270/120 is not SR)
LEASH = {CAMP_BLUE: 650.0, CAMP_RED: 650.0, CAMP_GROMP: 450.0, CAMP_WOLVES: 650.0,     # WIKI infoboxes
         CAMP_RAPTORS: 650.0, CAMP_KRUGS: 650.0, CAMP_SCUTTLE: 1e9}
KRUG_SPLITS = {Monster.KRUG: 4, Monster.KRUG_MEDIUM: 2}           # WIKI: Ancient -> 4 Mini, Krug -> 2 Mini
SPLIT_DELAY = 1.0                                                 # WIKI "Mini Krugs spawn over 1 second"
SPLIT_OFFSETS = ((60.0, 0.0), (-60.0, 0.0), (0.0, 60.0), (0.0, -60.0))   # INFERRED-L
CRAB_UNTARGETABLE_SPAWN = 1.5                                     # CLIENT untargetableSpawnTime

# ---- monster AI (WIKI Monster "Monster behavior & Patience"; rates INFERRED) ----------------------
SWEEP_S = 0.25                  # INFERRED-M target re-evaluation period (lane AI sweep)
SWITCH_COST = 1.0 / 7.0         # WIKI "up to 6 target changes before patience depletes"
PATIENCE_DRAIN = 0.35           # INFERRED-L patience/s outside leash (x(1 + excess/leash))
RECENTLY_HIT_S, RECENTLY_HIT_DRAIN = 1.0, 0.5   # INFERRED-L: being attacked halves the drain
SOFT_RESET_S = 6.0              # WIKI soft reset duration
SOFT_HEAL = 0.06                # WIKI 6% max HP per second during soft reset
HARD_HEAL = 0.25                # INFERRED-L "much faster"
SOFT_RESTORE = 0.5              # INFERRED-L patience restored when hit in leash during a soft reset
GRACE_S = 1.5                   # WIKI no patience loss for 1.5 s after that
PATIENCE_REFILL_DELAY, PATIENCE_REFILL_S = 2.0, 2.0   # WIKI back at camp: after 2 s, over 2 s
MS_IMPATIENT, MS_HARD = 1.2, 2.0    # WIKI +20% once patience runs out; hard-reset speed INFERRED-L
HOME_EPS = 50.0                 # INFERRED-M arrived at camp
SMALL_FOLLOW_RANGE = 700.0      # WIKI smalls share the large monster's patience within 700
MARKED_FOR_DEATH_S = 10.0       # WIKI smalls die 10 s after the large one when out of champion combat
NO_RESET, SOFT, HARD = 0, 1, 2
CRAB_PATROL_HALF, CRAB_FLEE_HALF = 700.0, 1400.0   # INFERRED-L river path half-lengths
CRAB_AXIS = (0.70710678, -0.70710678)              # INFERRED-M: river runs top-left -> bottom-right
CRAB_IDLE_MS_REDUCTION = 100.0  # WIKI base MS -100 out of combat
CRAB_FLEE_S = 3.0               # INFERRED-L flee window after champion damage
CRAB_CC_MULT = 2.0              # WIKI -100% tenacity (CC durations doubled)

# ---- rewards / buffs -----------------------------------------------------------------------------
CREST_DURATION = 120.0          # WIKI Buff data Crest of Insight / Cinders
SHRINE_DURATION, SHRINE_RADIUS, SHRINE_SIGHT = 90.0, 500.0, 525.0   # CLIENT 90 s; WIKI est. radii
SHRINE_POS = ((4400.0, 9600.0), (10500.0, 5170.0))                    # Baron / Dragon river (Scuttler camps, INFERRED-M)
SHRINE_MS, SHRINE_MS_DURATION = 0.30, 1.5                           # CLIENT effect amounts 0.3 / 2.0? (WIKI 1.5)
QUEST_GOLD, QUEST_XP = 10.0, 10.0           # PATCH 26.1 per large monster after the jungle quest
QUEST_MS_IN_COMBAT, QUEST_MS_OUT = 0.04, 0.08   # PATCH 26.1 in jungle/river

# ---- Smite (CLIENT SummonerSmite; WIKI Smite) ------------------------------------------------------
SMITE = 11
SMITE_START_CD = 15.0           # SUMMONER_SPELLS §1.3 start-of-game cooldown (charges unaffected)
SMITE_SECOND_CHARGE_AT = 48.0   # WIKI/PATCH 26.1 recharge starts at 0:48 (26.12 bug fix)
SMITE_FLAGS = D.TAG_PROC | D.PROP_NO_DAMAGE_MOD | D.PROP_NO_OMNIVAMP | D.PROP_SUMMONER

# ---- jungle pets (items 1101/1102/1103; CLIENT item data + WIKI Template:Jungle pet info) ----------
PET_NONE, PET_SCORCHCLAW, PET_GUSTWALKER, PET_MOSSTOMPER = range(4)
PET_ITEMS = {1101: PET_SCORCHCLAW, 1102: PET_GUSTWALKER, 1103: PET_MOSSTOMPER}
FIRST_EVOLUTION, FINAL_EVOLUTION = 15, 35   # WIKI/PATCH 26.1 (client Breakpoint1 still reads 20)
BONUS_TREAT_PERIOD, BONUS_TREAT_PERIOD_ADULT = 60.0, 90.0
BONUS_TREAT_GOLD = 20.0
TREAT_XP, FIRST_LARGE_XP = 80.0, 150.0      # WIKI (PuppyControllerBuff TreatXP / FirstMonsterBonusXP)
COMEBACK_THRESHOLD, COMEBACK_XP = 1.1, 50.0  # WIKI; CLIENT SmiteComebackXP 50
PET_RADIUS, PET_PERIOD = 650.0, 1.0
PET_MONSTER_AMP = 0.10                      # PATCH 26.1 (item module packet_amp)
PET_MONSTER_TAKEN = 0.50                    # PATCH 26.1 junglers take 50% from non-epic monsters
PET_FLAGS = D.TAG_PET | D.TAG_AOE | D.TAG_PROC | D.PROP_NO_OMNIVAMP | D.PROP_NO_DAMAGE_MOD
MONSTER_HUNTER_SHARE, MONSTER_HUNTER_GOLD, MONSTER_HUNTER_END = 0.40, 13.0, 840.0
MINION_XP_REDUCTION, MINION_XP_REDUCTION_END, MINION_XP_BEHIND = 0.70, 1200.0, 1.5
SCORCH_STACKS_PER_S, SCORCH_MAX = 6.0, 100.0   # CLIENT StacksPerHalfSec 3, AvatarMaxStacks 100
GUST_DECAY_S = 1.5                              # CLIENT CampMSDuration
MOSS_OOC_S = 10.0                               # CLIENT AvatarDamageCD

# ---- damage-over-time slots per target (N, 2): 0 Crest of Cinders burn, 1 Scorchclaw burn --------
DOT_RED, DOT_SCORCH = 0, 1
RED_BURN_S, RED_SLOW_S, RED_INSTANCES = 2.0, 3.0, 3     # CLIENT BurnDuration / DebuffDuration; WIKI 3 instances
SCORCH_BURN_S, SCORCH_TICK = 4.0, 1.0
DOT_FLAGS = D.TAG_PERIODIC | D.TAG_BURN | D.TAG_PROC | D.PROP_NO_DAMAGE_MOD


def _f(x):
    return jnp.asarray(x, jnp.float32)


def _i(x):
    return jnp.asarray(x, jnp.int32)


def client() -> dict:
    return json.loads(DATA.read_text())


# =================================================================================================
# static table
# =================================================================================================
class JungleTable(NamedTuple):
    """Static jungle layout and per-type stats (host-built; arrays are jnp)."""
    monster0: int               # world index of slot 0
    n_slots: int
    slot_camp: Any              # (S,) int32
    slot_init_type: Any         # (S,) int32 type spawned with the camp (-1: Krug split reserve)
    slot_parent: Any            # (S,) int32 slot whose Krug death spawns a Mini Krug here (-1 none)
    slot_offset: Any            # (S, 2) split offset from the parent's death position
    slot_home: Any              # (S, 2) client placement (reserves: the camp marker)
    camp_x: Any                 # (K,)
    camp_y: Any
    camp_type: Any              # (K,) int32 CAMP_*
    camp_side: Any              # (K,) int32 0 Order jungle, 1 Chaos jungle, 2 river
    camp_first: Any             # (K,) first spawn time
    camp_respawn: Any           # (K,) respawn delay after the last member dies
    camp_leash: Any             # (K,)
    camp_large: Any             # (K,) int32 slot of the camp's large monster (-1 none)
    camp_onehot: Any            # (S, K) float32
    t_hp: Any                   # (T,) level-1 stats per Monster type
    t_ad: Any
    t_armor: Any
    t_mr: Any
    t_ms: Any
    t_aspeed: Any
    t_range: Any
    t_windup_frac: Any
    t_missile: Any
    t_radius: Any
    t_gold: Any
    t_xp: Any
    t_size: Any
    t_curve: Any
    t_bonus_phys: Any
    t_bonus_magic: Any
    smite_damage: Any           # (3,) by pet stage 600/1000/1400
    smite_pvp: float
    smite_slow: float
    smite_slow_duration: float
    smite_cooldown: float
    smite_recharge: float
    smite_max_ammo: int
    smite_range: float
    smite_aoe_radius: float
    smite_forgiveness: float
    heal_base: float            # Monster_Heal_Mis
    heal_per_level: float
    heal_cap: float
    mana_base: float
    mana_per_level: float
    mana_max_mult: float
    energy_restore: float


def _windup_frac(rec: dict) -> float:
    """Attack windup as a fraction of the attack period (client cast/total or 0.3 + offset)."""
    tot, cast = rec.get("attack_total_time"), rec.get("attack_cast_time")
    if tot and cast:
        return float(cast) / float(tot)
    return 0.3 + float(rec.get("attack_delay_cast_offset_percent") or 0.0)


def _missile(rec: dict) -> float:
    for k, sp in sorted(rec["attack_spells"].items()):
        if k.endswith("BasicAttack") and (sp.get("missile_speed") or 0) > 0:
            return float(sp["missile_speed"])
    return 0.0


def build_table(monster0: int = 0, data: dict | None = None) -> JungleTable:
    """Static table for ``world.config.build_config`` (``monster0`` = world index of the first
    monster slot). 38 slots: per side Blue, Red, Gromp, 3 wolves, 6 raptors, Ancient Krug + Krug
    + 4 Mini-Krug reserve slots (6 minis: 2 reuse the parent slots); then 2 Scuttlers."""
    data = client() if data is None else data
    recs = [data["monsters"][c] for c in CHARACTERS]
    t_hp = [r["hp"] for r in recs]
    t_ad = [r["attack_damage"] if r["attack_damage"] is not None else MURKWOLF_MINI_AD for r in recs]
    camps = {c["name"]: c for c in data["camps"]}
    slot_camp, init_type, parent, offset, home = [], [], [], [], []
    camp_large = []
    for k, name in enumerate(CAMP_NAMES):
        c = camps[name]
        large = -1
        for m in c["members"]:
            t = CHARACTERS.index(m["character"])
            s = len(slot_camp)
            if SIZE[t] == LARGE and large < 0:
                large = s
            slot_camp.append(k); init_type.append(t); home.append((m["x"], m["y"]))
            n_split = KRUG_SPLITS.get(t, 0)
            parent.append(s if n_split else -1); offset.append(SPLIT_OFFSETS[0] if n_split else (0.0, 0.0))
        for m_slot in [s for s in range(len(slot_camp)) if slot_camp[s] == k and parent[s] == s]:
            for j in range(1, KRUG_SPLITS[init_type[m_slot]]):
                slot_camp.append(k); init_type.append(-1); parent.append(m_slot)
                offset.append(SPLIT_OFFSETS[j]); home.append((c["x"], c["y"]))
        camp_large.append(large)
    s = len(slot_camp)
    if s > MAX_JUNGLE_SLOTS:
        raise RuntimeError(f"jungle needs {s} slots > {MAX_JUNGLE_SLOTS}")
    onehot = np.zeros((s, len(CAMP_NAMES)), np.float32)
    onehot[np.arange(s), slot_camp] = 1.0
    ctype = [CAMP_TYPE[k] for k in range(len(CAMP_NAMES))]
    sm = data["smite"]
    dv = sm["data_values"]
    hv = data["monster_kill_heal"]
    f = lambda v: jnp.asarray(v, jnp.float32)
    i = lambda v: jnp.asarray(v, jnp.int32)
    return JungleTable(
        monster0=int(monster0), n_slots=s, slot_camp=i(slot_camp), slot_init_type=i(init_type),
        slot_parent=i(parent), slot_offset=f(offset), slot_home=f(home),
        camp_x=f([camps[n]["x"] for n in CAMP_NAMES]), camp_y=f([camps[n]["y"] for n in CAMP_NAMES]),
        camp_type=i(ctype), camp_side=i([0] * 6 + [1] * 6 + [2] * 2),
        camp_first=f([FIRST_SPAWN[t] for t in ctype]), camp_respawn=f([RESPAWN[t] for t in ctype]),
        camp_leash=f([LEASH[t] for t in ctype]), camp_large=i(camp_large), camp_onehot=jnp.asarray(onehot),
        t_hp=f(t_hp), t_ad=f(t_ad), t_armor=f([r["armor"] for r in recs]), t_mr=f([r["magic_resist"] for r in recs]),
        t_ms=f([r["move_speed"] for r in recs]), t_aspeed=f([r["attack_speed"] for r in recs]),
        t_range=f([r["attack_range"] or DEFAULT_ATTACK_RANGE for r in recs]),
        t_windup_frac=f([_windup_frac(r) for r in recs]), t_missile=f([_missile(r) for r in recs]),
        t_radius=f([r["gameplay_radius"] for r in recs]), t_gold=f([r["gold"] for r in recs]),
        t_xp=f([r["xp"] for r in recs]), t_size=i(SIZE), t_curve=i(HP_CURVE),
        t_bonus_phys=f(BONUS_PHYS_CURRENT), t_bonus_magic=f(BONUS_MAGIC_CURRENT),
        smite_damage=f([dv["SmiteBaseDamage"], dv["SmiteUpgradedDamage"], dv["Smite2ndUpgradedDamage"]]),
        smite_pvp=float(dv["FirstPVPDamage"]), smite_slow=float(dv["SmiteSlowAmount"]),
        smite_slow_duration=float(dv["SmiteSlowDuration"]), smite_cooldown=float(sm["cooldown"]),
        smite_recharge=float(sm["ammo_recharge_time"]), smite_max_ammo=int(sm["max_ammo"]),
        smite_range=float(sm["cast_range"]), smite_aoe_radius=float(sm["cast_radius"]),
        smite_forgiveness=float(sm["forgiveness_range"]),
        heal_base=float(hv["HealBase"]), heal_per_level=float(hv["HealPerLevel"]), heal_cap=float(hv["BaseHealCap"]),
        mana_base=float(hv["ManaBase"]), mana_per_level=float(hv["ManaPerLevel"]),
        mana_max_mult=float(hv["ManaMaxMult"]), energy_restore=float(hv["EnergyRestore"]))


def static_slots(table: JungleTable) -> dict:
    """Host-side per-slot layout for ``build_config``: kind (KIND_NONE until spawned), team NEUTRAL,
    sub (initial Monster type, 0 for reserves), x/y (home). Lists of length ``n_slots``."""
    it = np.asarray(table.slot_init_type)
    home = np.asarray(table.slot_home)
    return {"kind": [KIND_NONE] * table.n_slots, "team": [NEUTRAL] * table.n_slots,
            "sub": [int(max(t, 0)) for t in it], "x": home[:, 0].tolist(), "y": home[:, 1].tolist()}


class MonsterStats(NamedTuple):
    hp: Any
    attack_damage: Any
    armor: Any
    magic_resist: Any
    move_speed: Any
    attack_speed: Any
    attack_range: Any
    windup: Any                 # seconds
    missile_speed: Any
    radius: Any
    gold: Any
    xp: Any


def monster_stats(table: JungleTable, mtype, level, first_crab=False) -> MonsterStats:
    """Stats of a monster of ``mtype`` spawning at camp ``level`` (WIKI level tables)."""
    t = jnp.clip(_i(mtype), 0, N_TYPES - 1)
    lv = jnp.clip(_i(level), 1, 18)
    lf = lv.astype(jnp.float32)
    tmpl = jnp.where(lv < 3, 1.0, jnp.where(lv <= 11, 1.2 + 0.1 * (lf - 3.0), 2.0 + 0.05 * (lf - 11.0)))
    buff = jnp.where(lv < 3, 1.0, 1.0 + 0.1 * (lf - 1.0))
    crab = jnp.where(lv <= 9, 1.0 + 0.1 * (lf - 1.0), 2.0 + 0.2 * (jnp.minimum(lf, 17.0) - 10.0))
    curve = table.t_curve[t]
    mult = jnp.where(curve == 1, buff, jnp.where(curve == 2, crab, tmpl))
    first = jnp.asarray(first_crab, bool) & (t == Monster.SCUTTLE)
    hp = table.t_hp[t] * mult * jnp.where(first, FIRST_CRAB_HP, 1.0)
    ad = table.t_ad[t] * jnp.asarray(AD_MULT, jnp.float32)[lv - 1]
    gold = table.t_gold[t] * jnp.where(t == Monster.SCUTTLE,
                                      jnp.asarray(CRAB_GOLD_MULT, jnp.float32)[jnp.minimum(lv, 17) - 1], 1.0)
    xp = table.t_xp[t] * jnp.asarray(XP_MULT, jnp.float32)[jnp.minimum(lv, 8) - 1] \
        * jnp.where(first, FIRST_CRAB_XP, 1.0)
    aspeed = table.t_aspeed[t]
    return MonsterStats(_f(hp), _f(ad), table.t_armor[t], table.t_mr[t], table.t_ms[t], aspeed, table.t_range[t],
                        _f(table.t_windup_frac[t] / jnp.maximum(aspeed, 1e-3)), table.t_missile[t],
                        table.t_radius[t], _f(gold), _f(xp))


# =================================================================================================
# state
# =================================================================================================
class SmiteState(NamedTuple):
    charges: Any                # (C,) float32
    max_charges: Any            # (C,) 1 until 0:48, then 2
    next_charge_at: Any         # (C,) inf when full
    ready_at: Any               # (C,) 15 s between casts (not hasted)


class PetState(NamedTuple):
    ptype: Any                  # (C,) int32 PET_* latched (effects persist after the egg is consumed)
    treats: Any                 # (C,) int32
    bonus: Any                  # (C,) int32 stored bonus treats
    next_bonus_at: Any          # (C,)
    first_large_done: Any       # (C,) bool
    next_attack_at: Any         # (C,) pet attack clock
    embers: Any                 # (C,) Scorchclaw stacks
    moss_granted_at: Any        # (C,) last Mosstomper shield grant
    gust_peak: Any              # (C,) Gustwalker MS peak of the current decay
    gust_t0: Any                # (C,)
    in_brush: Any               # (C,) bool last tick
    minion_gold: Any            # (C,) Monster Hunter bookkeeping
    monster_gold: Any
    last_combat: Any            # (C,) last tick the holder was in combat (Mosstomper)
    moss_pending: Any           # (C,) bool: grant the Mosstomper shield now (evolution / large kill)


class DotState(NamedTuple):
    until: Any                  # (N, 2)
    next_tick: Any              # (N, 2)
    per_tick: Any               # (N, 2)
    src: Any                    # (N, 2) int32


class JungleState(NamedTuple):
    exists: Any                 # (S,) spawned and not yet dead (jungle bookkeeping)
    mtype: Any                  # (S,) int32 current Monster type
    level: Any                  # (S,) int32 camp level at spawn
    first_crab: Any             # (S,) bool reduced first Scuttler
    spawn_at: Any               # (S,) pending Mini-Krug spawn (inf none)
    spawn_x: Any
    spawn_y: Any
    spawn_level: Any
    untargetable_until: Any     # (S,)
    camp_respawn_at: Any        # (K,) next (re)spawn of the camp (inf while alive / waiting)
    camp_marked: Any            # (K,) bool large monster dead: remaining members marked for death
    camp_combat: Any            # (K,) last champion combat with any member
    camp_engaged: Any           # (K, C) last time champion c damaged the camp
    camp_aggro_start: Any       # (K,)
    target: Any                 # (S,) int32 world unit (-1)
    aggro: Any                  # (S,) bool
    patience: Any               # (S,) [0, 1]
    reset_mode: Any             # (S,) NO_RESET / SOFT / HARD
    reset_start: Any            # (S,)
    grace_until: Any            # (S,)
    home_since: Any             # (S,)
    sweep: Any                  # (S,)
    last_hit: Any               # (S,) last champion damage
    flee_until: Any             # (S,) Scuttler
    flee_sign: Any              # (S,) +-1 flee direction along the river axis
    patrol_sign: Any            # (S,) +-1
    crab_initial_left: Any      # () int32 initial Scuttlers still alive
    crab_respawn_at: Any        # () next Scuttler respawn (inf)
    shrine_team: Any            # (2,) int32 river shrine owner (-1 none); 0 = Baron (top) river
    shrine_until: Any           # (2,)
    key: Any
    blue_until: Any             # (C,) Crest of Insight
    red_until: Any              # (C,) Crest of Cinders
    smite: SmiteState
    pet: PetState
    dots: DotState


def init_jungle(table: JungleTable, n_champions: int, n_units: int, *, seed: int = 0) -> JungleState:
    s, k, c, n = table.n_slots, int(table.camp_x.shape[0]), int(n_champions), int(n_units)
    fs = lambda v: jnp.full((s,), v, jnp.float32)
    fc = lambda v: jnp.full((c,), v, jnp.float32)
    zs, bs = jnp.zeros((s,), jnp.int32), jnp.zeros((s,), bool)
    inf = jnp.float32(jnp.inf)
    smite = SmiteState(fc(1.0), fc(1.0), fc(jnp.inf), fc(SMITE_START_CD))
    pet = PetState(jnp.zeros((c,), jnp.int32), jnp.zeros((c,), jnp.int32), jnp.zeros((c,), jnp.int32), fc(jnp.inf),
                   jnp.zeros((c,), bool), fc(0.0), fc(0.0), fc(-1e9), fc(0.0), fc(-1e9), jnp.zeros((c,), bool),
                   fc(0.0), fc(0.0), fc(-1e9), jnp.zeros((c,), bool))
    dots = DotState(jnp.full((n, 2), -1.0, jnp.float32), jnp.full((n, 2), jnp.inf, jnp.float32),
                    jnp.zeros((n, 2), jnp.float32), jnp.full((n, 2), -1, jnp.int32))
    is_crab = table.camp_type == CAMP_SCUTTLE
    return JungleState(
        exists=bs, mtype=jnp.maximum(table.slot_init_type, 0), level=zs + 1, first_crab=bs, spawn_at=fs(jnp.inf),
        spawn_x=fs(0.0), spawn_y=fs(0.0), spawn_level=zs + 1, untargetable_until=fs(0.0),
        camp_respawn_at=table.camp_first, camp_marked=jnp.zeros((k,), bool), camp_combat=jnp.full((k,), -1e9, jnp.float32),
        camp_engaged=jnp.full((k, c), -1e9, jnp.float32), camp_aggro_start=jnp.zeros((k,), jnp.float32),
        target=zs - 1, aggro=bs, patience=fs(1.0), reset_mode=zs, reset_start=fs(0.0), grace_until=fs(0.0),
        home_since=fs(-1e9), sweep=fs(0.0), last_hit=fs(-1e9), flee_until=fs(-1e9), flee_sign=fs(1.0),
        patrol_sign=fs(1.0), crab_initial_left=jnp.int32(int(np.sum(np.asarray(is_crab)))),
        crab_respawn_at=inf, shrine_team=jnp.full((2,), -1, jnp.int32), shrine_until=jnp.full((2,), -1.0, jnp.float32),
        key=jax.random.PRNGKey(seed), blue_until=fc(-1.0), red_until=fc(-1.0), smite=smite, pet=pet, dots=dots)


# =================================================================================================
# 1. spawns
# =================================================================================================
class SpawnWrites(NamedTuple):
    """(S,) per jungle slot; write where ``mask`` (kind KIND_MONSTER, team NEUTRAL, alive, new spawn_seq,
    spawn_time now, attack/CC timers reset). ``targetable`` comes from ``monster_ai``."""
    mask: Any
    sub: Any
    x: Any
    y: Any
    hp: Any
    radius: Any
    armor: Any
    magic_resist: Any
    attack_damage: Any
    attack_range: Any
    attack_speed: Any
    move_speed: Any
    windup: Any
    missile_speed: Any
    level: Any


def camp_level(champion_level) -> Any:
    """WIKI: average champion level, rounded, at spawn time."""
    return jnp.clip(jnp.round(jnp.mean(_f(champion_level))), 1, 18).astype(jnp.int32)


def spawn_step(state: JungleState, table: JungleTable, *, now, champion_level) -> tuple[JungleState, SpawnWrites]:
    """Camp first spawns/respawns, the Scuttler cycle and pending Mini-Krug splits (phase 1)."""
    now = _f(now)
    lvl = camp_level(champion_level)
    is_crab_camp = table.camp_type == CAMP_SCUTTLE
    due_camp = state.camp_respawn_at <= now                                          # (K,)
    camp_of = table.slot_camp
    regular = due_camp[camp_of] & (table.slot_init_type >= 0)
    # Scuttler cycle after both initial crabs died: one crab at a time, random river (WIKI).
    key, sub = jax.random.split(state.key)
    crab_due = state.crab_respawn_at <= now
    river = jax.random.bernoulli(sub).astype(jnp.int32)                            # 0 Baron, 1 Dragon river
    crab_slot_camp = jnp.where(river == 0, 12, 13)
    crab = crab_due & (camp_of == crab_slot_camp) & is_crab_camp[camp_of]
    initial_crab = regular & is_crab_camp[camp_of]
    split = (state.spawn_at <= now) & ~state.exists
    mask = (regular | crab | split) & ~state.exists
    mtype = jnp.where(split & ~regular, Monster.KRUG_MINI, jnp.where(regular | crab, jnp.maximum(table.slot_init_type, 0),
                                                                     state.mtype)).astype(jnp.int32)
    level = jnp.where(split & ~regular, state.spawn_level, lvl).astype(jnp.int32)
    first = jnp.where(mask, initial_crab, state.first_crab)
    st = monster_stats(table, mtype, level, first)
    x = jnp.where(split & ~regular, state.spawn_x, table.slot_home[:, 0])
    y = jnp.where(split & ~regular, state.spawn_y, table.slot_home[:, 1])
    untarg = jnp.where(mtype == Monster.SCUTTLE, CRAB_UNTARGETABLE_SPAWN, 0.0)
    put = lambda new, old: jnp.where(mask, new, old)
    fresh_camp = _camp_any(table, (regular | crab) & mask)
    state = state._replace(
        exists=state.exists | mask, mtype=put(mtype, state.mtype), level=put(level, state.level), first_crab=first,
        spawn_at=jnp.where(mask, jnp.inf, state.spawn_at).astype(jnp.float32),
        untargetable_until=put(now + untarg, state.untargetable_until).astype(jnp.float32),
        camp_respawn_at=jnp.where(due_camp, jnp.inf, state.camp_respawn_at).astype(jnp.float32),
        camp_marked=jnp.where(fresh_camp, False, state.camp_marked),
        target=put(-1, state.target).astype(jnp.int32), aggro=put(False, state.aggro),
        patience=put(1.0, state.patience).astype(jnp.float32), reset_mode=put(NO_RESET, state.reset_mode).astype(jnp.int32),
        last_hit=put(-1e9, state.last_hit).astype(jnp.float32), flee_until=put(-1e9, state.flee_until).astype(jnp.float32),
        crab_respawn_at=jnp.where(crab_due, jnp.inf, state.crab_respawn_at).astype(jnp.float32), key=key)
    w = SpawnWrites(mask, mtype, _f(x), _f(y), st.hp, st.radius, st.armor, st.magic_resist, st.attack_damage,
                    st.attack_range, st.attack_speed, st.move_speed, st.windup, st.missile_speed, level)
    return state, w


def write_spawns(table: JungleTable, w: SpawnWrites, *, now, next_seq, **arrays) -> tuple[dict, Any]:
    """Convenience scatter of ``SpawnWrites`` into world (N,) arrays passed by name (any subset of
    kind, sub, team, alive, targetable, x, y, hp, max_hp, radius, armor, mr, ad, arange, aspeed, mspeed,
    windup, missile_speed, spawn_time, spawn_seq). Returns ``(arrays, next_seq)``."""
    m0, s = table.monster0, table.n_slots
    seq = next_seq + jnp.cumsum(w.mask.astype(jnp.int32)) - 1
    vals = {"kind": KIND_MONSTER, "sub": w.sub, "team": NEUTRAL, "alive": True,
            "targetable": True, "x": w.x, "y": w.y, "hp": w.hp, "max_hp": w.hp, "radius": w.radius,
            "armor": w.armor, "mr": w.magic_resist, "ad": w.attack_damage, "arange": w.attack_range,
            "aspeed": w.attack_speed, "mspeed": w.move_speed, "windup": w.windup, "missile_speed": w.missile_speed,
            "spawn_time": now, "spawn_seq": seq}
    out = {}
    for name, arr in arrays.items():
        v = jnp.broadcast_to(jnp.asarray(vals[name], arr.dtype), (s,))
        seg = arr[m0:m0 + s]
        out[name] = arr.at[m0:m0 + s].set(jnp.where(w.mask, v, seg))
    return out, next_seq + jnp.sum(w.mask.astype(jnp.int32))


# =================================================================================================
# 2. monster AI
# =================================================================================================
class MonsterAI(NamedTuple):
    """(S,) per jungle slot (world unit ``monster0 + s``)."""
    desired: Any                # int32 world unit to attack (-1 none)
    goal_x: Any
    goal_y: Any
    move_speed: Any             # this tick's speed (impatience / hard reset / Scuttler idle)
    moving: Any                 # bool: move toward the goal (False: stand, e.g. in range)
    targetable: Any             # bool
    heal: Any                   # HP to add this tick (reset regeneration; full heal on arrival)
    despawn: Any                # bool: remove without rewards (marked for death)
    slow_resist: Any            # Scuttler 100% while not fleeing
    cc_duration_mult: Any       # Scuttler x2 (-100% tenacity)


def _slot_idx(table):
    return table.monster0 + jnp.arange(table.n_slots, dtype=jnp.int32)


def _camp_any(table, mask_s):
    """(S, ...) bool -> (K, ...) any over camp members."""
    m = mask_s.astype(jnp.float32)
    return jnp.tensordot(table.camp_onehot.T, m, axes=1) > 0.5


def monster_ai(state: JungleState, table: JungleTable, units: WorldUnits, att: AttackState, *, now, dt,
               damage_events) -> tuple[JungleState, MonsterAI]:
    """Monster aggro, target selection, patience/leash resets, returning home, camp group aggro,
    marked-for-death and the Scuttler's patrol/flee (phase 4).

    ``damage_events[i, j]``: i damaged j last tick (``ModernState.damage_matrix``). Aggro comes
    only from champion damage on any camp member (the whole camp aggroes). Targets: the nearest
    champion that damaged the camp during this aggro episode (straight-line, INFERRED-M for the
    client's path distance), regardless of vision.
    """
    now, dt = _f(now), _f(dt)
    idx = _slot_idx(table)
    c = state.blue_until.shape[0]
    n = units.x.shape[0]
    alive = state.exists & jnp.asarray(units.alive)[idx]
    px, py = _f(units.x)[idx], _f(units.y)[idx]
    camp = table.slot_camp
    hx, hy = table.slot_home[:, 0], table.slot_home[:, 1]
    cx, cy = table.camp_x[camp], table.camp_y[camp]
    leash = table.camp_leash[camp]
    d_home = jnp.sqrt((px - cx) ** 2 + (py - cy) ** 2)
    is_crab = state.mtype == Monster.SCUTTLE
    dmg = jnp.asarray(damage_events, bool)
    champ_ok = jnp.asarray(units.alive)[:c] & jnp.asarray(units.targetable)[:c] & (jnp.asarray(units.kind)[:c] == KIND_CHAMPION)
    hit = dmg[:c][:, idx].T & alive[:, None]                                       # (S, C)
    hit_any = jnp.any(hit, axis=1)
    hits_champ = jnp.any(dmg[idx][:, :c], axis=1) & alive
    camp_hit = _camp_any(table, hit)                                                # (K, C)
    camp_engaged = jnp.where(camp_hit, now, state.camp_engaged)
    camp_combat = jnp.where(jnp.any(camp_hit, axis=1) | _camp_any(table, hits_champ), now, state.camp_combat)
    last_hit = jnp.where(hit_any, now, state.last_hit)

    mode = state.reset_mode
    inside = d_home <= leash
    # Soft reset ends early when attacked inside the leash.
    end_soft = (mode == SOFT) & inside & hit_any
    patience = jnp.where(end_soft, jnp.minimum(1.0, state.patience + SOFT_RESTORE), state.patience)
    grace = jnp.where(end_soft, now + GRACE_S, state.grace_until)
    mode = jnp.where(end_soft, NO_RESET, mode)
    mode = jnp.where((mode == SOFT) & (now - state.reset_start >= SOFT_RESET_S), HARD, mode)
    # Camp group aggro: any member hit by a champion aggroes every member not resetting.
    camp_aggro_now = jnp.any(camp_hit, axis=1)                                      # (K,)
    start = camp_aggro_now & ~_camp_any(table, state.aggro & alive)
    camp_aggro_start = jnp.where(start, now, state.camp_aggro_start)
    aggro = (state.aggro | camp_aggro_now[camp]) & alive & (mode == NO_RESET) & ~is_crab

    engaged = (camp_engaged >= camp_aggro_start[:, None])[camp] & champ_ok[None, :]  # (S, C)
    dx = _f(units.x)[:c][None, :] - px[:, None]
    dy = _f(units.y)[:c][None, :] - py[:, None]
    dist = jnp.sqrt(dx ** 2 + dy ** 2)
    near = jnp.argmin(jnp.where(engaged, dist, jnp.inf), axis=1).astype(jnp.int32)
    has_near = jnp.any(engaged, axis=1)
    cur = state.target
    cur_ok = (cur >= 0) & (cur < c) & engaged[jnp.arange(table.n_slots), jnp.clip(cur, 0, c - 1)]
    timer = state.sweep + dt
    sweep = timer >= SWEEP_S
    pick = aggro & has_near & (~cur_ok | sweep)
    new_t = jnp.where(pick, near, jnp.where(cur_ok & aggro, cur, -1)).astype(jnp.int32)
    switched = aggro & cur_ok & (new_t != cur) & (new_t >= 0)
    patience = patience - jnp.where(switched & (now >= grace), SWITCH_COST, 0.0)
    timer = jnp.where(sweep, 0.0, timer)
    # Leash: drain while the monster or its target is beyond the leash radius, or no target.
    ti = jnp.clip(new_t, 0, n - 1)
    tx, ty = _f(units.x)[ti], _f(units.y)[ti]
    t_home = jnp.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
    excess = jnp.maximum(jnp.maximum(d_home - leash, jnp.where(new_t >= 0, t_home - leash, 0.0)), 0.0)
    recently = (now - last_hit) <= RECENTLY_HIT_S
    rate = PATIENCE_DRAIN * (1.0 + excess / leash) * jnp.where(recently, RECENTLY_HIT_DRAIN, 1.0)
    draining = aggro & ((excess > 0) | (new_t < 0)) & (now >= grace)
    patience = jnp.clip(patience - jnp.where(draining, rate * dt, 0.0), 0.0, 1.0)
    # Smalls share the large monster's patience/reset while it is alive and within 700.
    large = table.camp_large[camp]
    li = jnp.clip(large, 0, table.n_slots - 1)
    follow = (large >= 0) & (large != jnp.arange(table.n_slots)) & alive[li] & \
        (state.mtype[li] != Monster.KRUG_MINI) & \
        (jnp.sqrt((px - px[li]) ** 2 + (py - py[li]) ** 2) <= SMALL_FOLLOW_RANGE)
    patience = jnp.where(follow, patience[li], patience)
    run_out = aggro & (patience <= 0.0) & (mode == NO_RESET)
    mode = jnp.where(run_out, SOFT, mode)
    reset_start = jnp.where(run_out, now, state.reset_start)
    adopt = follow & (mode[li] != NO_RESET) & (mode == NO_RESET)
    mode = jnp.where(adopt, mode[li], mode)
    reset_start = jnp.where(adopt, reset_start[li], reset_start)
    aggro = aggro & (mode == NO_RESET)
    new_t = jnp.where(aggro, new_t, -1).astype(jnp.int32)
    # Return home and recover.
    resetting = (mode != NO_RESET) & alive
    at_home = jnp.sqrt((px - hx) ** 2 + (py - hy) ** 2) <= HOME_EPS
    arrive = resetting & at_home
    mx = jnp.asarray(units.max_hp)[idx]
    hp = jnp.asarray(units.hp)[idx]
    heal = jnp.where(mode == SOFT, SOFT_HEAL, jnp.where(mode == HARD, HARD_HEAL, 0.0)) * mx * dt
    heal = jnp.where(arrive, mx - hp, jnp.where(resetting, heal, 0.0))
    heal = jnp.where(alive, jnp.maximum(heal, 0.0), 0.0)
    mode = jnp.where(arrive, NO_RESET, mode)
    home_since = jnp.where(arrive | (at_home & ~aggro & (state.home_since < -1e8)), now, state.home_since)
    refill = jnp.clip((now - home_since - PATIENCE_REFILL_DELAY) / PATIENCE_REFILL_S, 0.0, 1.0)
    patience = jnp.where(~aggro & (mode == NO_RESET) & at_home, jnp.maximum(patience, refill), patience)
    home_since = jnp.where(~at_home, -1e9, home_since)

    # Movement goals.
    t_r = _f(units.radius)[ti]
    reach = _f(units.attack_range)[idx] + _f(units.radius)[idx] + t_r
    d_t = jnp.sqrt((tx - px) ** 2 + (ty - py) ** 2)
    in_range = (new_t >= 0) & (d_t <= reach)
    base_ms = _f(units.move_speed)[idx]
    ms = base_ms * jnp.where(mode == HARD, MS_HARD, jnp.where((mode == SOFT) | (patience <= 0.0), MS_IMPATIENT, 1.0))
    gx = jnp.where(new_t >= 0, tx, hx)
    gy = jnp.where(new_t >= 0, ty, hy)
    moving = alive & jnp.where(new_t >= 0, ~in_range, jnp.sqrt((px - hx) ** 2 + (py - hy) ** 2) > HOME_EPS)

    # Rift Scuttler: never attacks; patrols the river, flees champion damage.
    ax, ay = CRAB_AXIS
    proj = (px - hx) * ax + (py - hy) * ay
    champ_hit_src = jnp.argmax(hit, axis=1)
    sx, sy = _f(units.x)[champ_hit_src], _f(units.y)[champ_hit_src]
    away = jnp.sign((px - sx) * ax + (py - sy) * ay + 1e-3)
    flee_until = jnp.where(is_crab & hit_any, now + CRAB_FLEE_S, state.flee_until)
    flee_sign = jnp.where(is_crab & hit_any, away, state.flee_sign)
    fleeing = is_crab & (now < flee_until)
    reached = jnp.abs(proj - state.patrol_sign * CRAB_PATROL_HALF) < HOME_EPS
    patrol_sign = jnp.where(is_crab & ~fleeing & reached, -state.patrol_sign, state.patrol_sign)
    goal_proj = jnp.where(fleeing, flee_sign * CRAB_FLEE_HALF, patrol_sign * CRAB_PATROL_HALF)
    gx = jnp.where(is_crab, hx + ax * goal_proj, gx)
    gy = jnp.where(is_crab, hy + ay * goal_proj, gy)
    ms = jnp.where(is_crab, base_ms - jnp.where(fleeing, 0.0, CRAB_IDLE_MS_REDUCTION), ms)
    moving = jnp.where(is_crab, alive, moving)
    heal = jnp.where(is_crab, 0.0, heal)

    # Marked for death: the camp's large monster died and no champion combat for 10 s.
    despawn = alive & state.camp_marked[camp] & ((now - camp_combat[camp]) >= MARKED_FOR_DEATH_S)
    targetable = alive & (now >= state.untargetable_until)
    s_ = state._replace(
        target=new_t, aggro=aggro & ~despawn, patience=_f(patience), reset_mode=mode.astype(jnp.int32),
        reset_start=_f(reset_start), grace_until=_f(grace), home_since=_f(home_since), sweep=_f(timer),
        last_hit=_f(last_hit), flee_until=_f(flee_until), flee_sign=_f(flee_sign), patrol_sign=_f(patrol_sign),
        camp_engaged=_f(camp_engaged), camp_combat=_f(camp_combat), camp_aggro_start=_f(camp_aggro_start))
    s_ = _clear_camps(s_, table, state.exists & ~despawn, now)
    out = MonsterAI(jnp.where(alive & ~is_crab, new_t, -1).astype(jnp.int32), _f(gx), _f(gy), _f(ms), moving & ~despawn,
                    targetable & ~despawn, _f(heal), despawn,
                    jnp.where(is_crab & ~fleeing, 1.0, 0.0).astype(jnp.float32),
                    jnp.where(is_crab, CRAB_CC_MULT, 1.0).astype(jnp.float32))
    return s_, out


def _clear_camps(state: JungleState, table: JungleTable, exists, now) -> JungleState:
    """Update ``exists`` and start the respawn timer of camps whose last member is gone (no
    pending Mini-Krug spawns). The Scuttler cycle is handled separately."""
    was = _camp_any(table, state.exists | jnp.isfinite(state.spawn_at))
    still = _camp_any(table, exists | jnp.isfinite(state.spawn_at))
    cleared = was & ~still & (table.camp_type != CAMP_SCUTTLE)
    return state._replace(
        exists=exists,
        camp_respawn_at=jnp.where(cleared, now + table.camp_respawn, state.camp_respawn_at).astype(jnp.float32),
        camp_marked=jnp.where(cleared, False, state.camp_marked))


# =================================================================================================
# 3. monster basic attacks
# =================================================================================================
def holder_taken_mult(state: JungleState, n_units: int) -> Any:
    """(N,) multiplier on damage *from regular monsters*: 0.5 for jungle-item holders (PATCH 26.1)."""
    c = state.pet.ptype.shape[0]
    return jnp.ones((n_units,), jnp.float32).at[:c].set(jnp.where(state.pet.ptype > 0, PET_MONSTER_TAKEN, 1.0))


def monster_attack_packets(state: JungleState, table: JungleTable, units: WorldUnits,
                           launch: AttackLaunch) -> tuple[D.Packets, D.Packets]:
    """``(main, bonus_magic)`` packets (S,) for jungle-monster basic attacks launched now.

    Main: AD (level-scaled at spawn, read from ``units.attack_damage``) + Blue/Red 5%, Greater
    Wolf/Crimson Raptor/Ancient Krug 3% of the target's current HP, PHYSICAL, ``TAG_BASIC_ATTACK``
    (no life steal). For ranged monsters (Gromp 1800, Crimson Raptor 750 missile speed) the world
    sends ``main`` raw/dtype/flags on its missile. ``bonus_magic``: Gromp's 5% current HP magic,
    emitted at launch (INFERRED-M: one tick-early vs the missile impact). Jungle-item holders take
    50% (scaled raw: a multiplicative received modifier).
    """
    idx = _slot_idx(table)
    n = units.x.shape[0]
    tgt = jnp.asarray(launch.target, jnp.int32)[idx]
    t = jnp.clip(tgt, 0, n - 1)
    mt = state.mtype
    valid = jnp.asarray(launch.launched, bool)[idx] & state.exists & (tgt >= 0) & jnp.asarray(units.alive)[t] \
        & (mt != Monster.SCUTTLE)
    cur = _f(units.hp)[t]
    mult = holder_taken_mult(state, n)[t]
    raw = (_f(units.attack_damage)[idx] + table.t_bonus_phys[mt] * cur) * mult
    mag = table.t_bonus_magic[mt] * cur * mult
    cid = jnp.asarray(launch.cast_id, jnp.int32)[idx]
    main = D.packets(valid, idx, t, jnp.where(valid, raw, 0.0), PHYSICAL, flags=D.TAG_BASIC_ATTACK, cast_id=cid)
    bonus = D.packets(valid & (mag > 0), idx, t, jnp.where(valid, mag, 0.0), MAGIC,
                      flags=D.TAG_PROC, cast_id=cid)
    return main, bonus


# =================================================================================================
# 4. Smite
# =================================================================================================
class SmiteOut(NamedTuple):
    packets: Any                # Packets (C * (1 + S),): target hit, then Primal Smite AoE on jungle slots
    cc: Any                     # CCOut (C, N): champion Smite 20% slow 2 s
    cast: Any                   # (C,) bool
    target: Any                 # (C,) int32
    charges: Any                # (C,) float
    recharge_left: Any          # (C,) seconds to the next charge (inf when full)


def pet_stage(state: JungleState) -> Any:
    """(C,) 0 Smite, 1 Unleashed Smite (15 treats), 2 Primal Smite (35 treats)."""
    t = state.pet.treats
    has = state.pet.ptype > 0
    return jnp.where(has & (t >= FINAL_EVOLUTION), 2, jnp.where(has & (t >= FIRST_EVOLUTION), 1, 0)).astype(jnp.int32)


def _smiteable(state: JungleState, table: JungleTable, units: WorldUnits, team, stage):
    """(C, N) valid Smite targets: large/medium (incl. epic) monsters, enemy lane minions; enemy
    champions after the first evolution (CLIENT mRequiredUnitTags; WIKI)."""
    n = units.x.shape[0]
    kind = jnp.asarray(units.kind)
    sub = jnp.asarray(units.sub, jnp.int32)
    idx = _slot_idx(table)
    in_jungle_slots = jnp.zeros((n,), bool).at[idx].set(True)
    small_type = jnp.asarray(SIZE, jnp.int32)[jnp.clip(sub, 0, N_TYPES - 1)] == SMALL
    small = in_jungle_slots & small_type
    live = jnp.asarray(units.alive) & jnp.asarray(units.targetable)
    monster = (kind == KIND_MONSTER) & ~small
    enemy = jnp.asarray(units.team)[None, :] != team[:, None]
    minion = (kind == KIND_MINION)[None, :] & enemy
    champ = (kind == KIND_CHAMPION)[None, :] & enemy & (stage >= 1)[:, None]
    return live[None, :] & (monster[None, :] | minion | champ)


def smite_step(state: JungleState, table: JungleTable, units: WorldUnits, request: CastOrder, spells, *, now,
               summoner_haste, alive) -> tuple[JungleState, SmiteOut]:
    """Smite charges/recharge and casts (phase 3, next to ``champions.summoners.step``, which
    ignores Smite requests). ``request`` is the summoner CastOrder (slot 0/1 = D/F of ``spells``
    (C, 2), the loadout). Casts while disabled (CLIENT canCastWhileDisabled), not while dead.

    * Charges: 1 at start; max 2 from 0:48 when the recharge starts; 90 s recharge hasted by
      summoner haste (``haste`` read at recharge start, INFERRED-M); 15 s between casts, not
      hasted; first cast possible at 0:15 (start-of-game cooldown, INFERRED-M).
    * Damage (TRUE, proc, no damage modifiers/omnivamp): 600 / 1000 (15 treats) / 1400 (35 treats)
      to monsters and lane minions; Primal Smite also hits other monsters within 210 (CLIENT
      castRadius, AoESmiteRatio 1). Champions (after 15 treats): 40 true + 20% slow for 2 s.
    * Range 500 edge to edge (castRangeUseBoundingBoxes); with no valid unit under the request
      the nearest smiteable monster within 125 of the cursor is taken (CLIENT forgiveness).
    """
    now = _f(now)
    c = state.blue_until.shape[0]
    n = units.x.shape[0]
    sm = state.smite
    haste = _f(summoner_haste) * jnp.ones((c,), jnp.float32)
    rech = table.smite_recharge * 100.0 / (100.0 + haste)
    # Second charge unlocks at 0:48 and starts recharging (26.12 fix: also when dead).
    unlock = (sm.max_charges < table.smite_max_ammo) & (now >= SMITE_SECOND_CHARGE_AT)
    max_c = jnp.where(unlock, float(table.smite_max_ammo), sm.max_charges)
    nxt = jnp.where(unlock & (sm.charges < max_c), SMITE_SECOND_CHARGE_AT + rech, sm.next_charge_at)
    gain = (sm.charges < max_c) & (now >= nxt)
    charges = jnp.where(gain, sm.charges + 1.0, sm.charges)
    nxt = jnp.where(gain, jnp.where(charges < max_c, nxt + rech, jnp.inf), nxt)
    # Request.
    slot = jnp.asarray(request.slot, jnp.int32)
    spells = jnp.asarray(spells, jnp.int32)
    is_smite = ((slot == 0) | (slot == 1)) & (spells[jnp.arange(c), jnp.clip(slot, 0, 1)] == SMITE)
    stage = pet_stage(state)
    team = jnp.asarray(units.team)[:c]
    ok_t = _smiteable(state, table, units, team, stage)                                       # (C, N)
    cx, cy = _f(units.x)[:c], _f(units.y)[:c]
    ux, uy, ur = _f(units.x), _f(units.y), _f(units.radius)
    d = jnp.sqrt((ux[None, :] - cx[:, None]) ** 2 + (uy[None, :] - cy[:, None]) ** 2)
    in_range = d <= table.smite_range + ur[None, :] + ur[:c, None]
    tgt = jnp.asarray(request.target, jnp.int32)
    ti = jnp.clip(tgt, 0, n - 1)
    direct = (tgt >= 0) & ok_t[jnp.arange(c), ti] & in_range[jnp.arange(c), ti]
    rx, ry = _f(request.x), _f(request.y)
    dcur = jnp.sqrt((ux[None, :] - rx[:, None]) ** 2 + (uy[None, :] - ry[:, None]) ** 2)
    kind = jnp.asarray(units.kind)
    fkey = jnp.where(ok_t & in_range & (kind == KIND_MONSTER)[None, :] & (dcur <= table.smite_forgiveness),
                     dcur, jnp.inf)
    fpick = jnp.argmin(fkey, axis=1).astype(jnp.int32)
    target = jnp.where(direct, ti, jnp.where(jnp.isfinite(jnp.min(fkey, axis=1)), fpick, -1)).astype(jnp.int32)
    cast = is_smite & jnp.asarray(alive, bool) & (charges >= 1.0) & (now >= sm.ready_at - 1e-4) & (target >= 0)
    was_full = charges >= max_c
    charges = jnp.where(cast, charges - 1.0, charges)
    nxt = jnp.where(cast & was_full, now + rech, nxt)
    ready = jnp.where(cast, now + table.smite_cooldown, sm.ready_at)
    tsafe = jnp.clip(target, 0, n - 1)
    vs_champ = kind[tsafe] == KIND_CHAMPION
    dmg = jnp.where(vs_champ, table.smite_pvp, table.smite_damage[stage])
    main = D.packets(cast, jnp.arange(c, dtype=jnp.int32), tsafe, jnp.where(cast, dmg, 0.0), TRUE, SMITE_FLAGS)
    # Primal Smite AoE on the other jungle monsters around the target.
    idx = _slot_idx(table)
    tx, ty = ux[tsafe], uy[tsafe]
    da = jnp.sqrt((ux[idx][None, :] - tx[:, None]) ** 2 + (uy[idx][None, :] - ty[:, None]) ** 2)
    aoe = (cast & (stage >= 2) & ~vs_champ)[:, None] & (da <= table.smite_aoe_radius + ur[idx][None, :]) \
        & state.exists[None, :] & jnp.asarray(units.alive)[idx][None, :] & (idx[None, :] != tsafe[:, None])
    aoe_pk = D.packets(aoe, jnp.arange(c, dtype=jnp.int32)[:, None], idx[None, :],
                       jnp.where(aoe, table.smite_damage[2], 0.0), TRUE, SMITE_FLAGS)
    slow_hit = (cast & vs_champ)[:, None] & (jnp.arange(n)[None, :] == tsafe[:, None])
    cc = no_cc(c, n)._replace(slow=jnp.where(slow_hit, table.smite_slow, 0.0).astype(jnp.float32),
                              slow_duration=jnp.where(slow_hit, table.smite_slow_duration, 0.0).astype(jnp.float32))
    state = state._replace(smite=SmiteState(_f(charges), _f(max_c), _f(nxt), _f(ready)))
    return state, SmiteOut(D.concat_packets(main, aoe_pk), cc, cast, jnp.where(cast, target, -1).astype(jnp.int32),
                           _f(charges), _f(jnp.where(charges < max_c, nxt - now, jnp.inf)))


# =================================================================================================
# 5. combat effects: crests on hit, burns, pets, evolution buffs
# =================================================================================================
class JungleEffects(NamedTuple):
    packets: Any                # Packets: burn ticks (N*2) + pet attacks (C*S) + Scorchclaw proc ticks
    cc: Any                     # CCOut (C, N): Crest of Cinders and Scorchclaw slows
    heal: Any                   # (C,) pet heal (benefits from heal power)
    shield: Any                 # (C,) Mosstomper's Courage shield grant (replace, duration inf)
    bonus_ms: Any               # (C,) Gustwalker's Gait MS fraction this tick


def red_slow(level, is_ranged) -> Any:
    """CLIENT BlessingoftheLizardElder SlowPotency: 10/15/25% at 1/6/11 (ranged x0.5)."""
    lv = _f(level)
    s = 0.10 + jnp.where(lv >= 6, 0.05, 0.0) + jnp.where(lv >= 11, 0.10, 0.0)
    return s * jnp.where(jnp.asarray(is_ranged, bool), 0.5, 1.0)


def red_burn_total(level) -> Any:
    """CLIENT DamageOverTime: 15, +3 per level from 6."""
    lv = _f(level)
    return 15.0 + 3.0 * jnp.maximum(lv - 5.0, 0.0)


def pet_damage(ctx) -> Any:
    """CLIENT SummonerSmite PetDPS (26.16): 20-150 by level + 10% bonus AD + 16% AP + 4% bonus HP
    + 25% bonus armor + 25% bonus MR, per pet attack (1/s)."""
    lv = jnp.clip(_f(ctx.level), 1.0, 18.0)
    return 20.0 + 130.0 * (lv - 1.0) / 17.0 + 0.10 * ctx.bonus_ad + 0.16 * ctx.ap \
        + 0.04 * (ctx.max_hp - ctx.base_hp) + 0.25 * ctx.bonus_armor + 0.25 * ctx.bonus_mr


def pet_heal(level) -> Any:
    """CLIENT PetHPS: 6, +2 per level from 4 (6-36)."""
    lv = _f(level)
    return 6.0 + 2.0 * jnp.maximum(lv - 3.0, 0.0)


def moss_shield(level) -> Any:
    """CLIENT item 1103 MaxShield: 200, +20 per level from 11."""
    lv = _f(level)
    return 200.0 + 20.0 * jnp.maximum(lv - 10.0, 0.0)


def pet_type_from_inventory(own) -> Any:
    """(C,) PET_* from ``items.inventory.owned_counts`` (C, I)."""
    from ..items.catalog import catalog
    out = jnp.zeros((own.shape[0],), jnp.int32)
    for iid, p in PET_ITEMS.items():
        out = jnp.where(own[:, catalog().row(iid)] > 0, p, out)
    return out


def _dot_ticks(dots: DotState, kind: int, now, n):
    due = (dots.next_tick[:, kind] <= now + 1e-4) & (dots.next_tick[:, kind] <= dots.until[:, kind] + 1e-4)
    pk = D.packets(due & (dots.src[:, kind] >= 0), jnp.maximum(dots.src[:, kind], 0), jnp.arange(n),
                   jnp.where(due, dots.per_tick[:, kind], 0.0), TRUE, DOT_FLAGS)
    period = 1.0
    nt = jnp.where(due, dots.next_tick[:, kind] + period, dots.next_tick[:, kind])
    return pk, dots._replace(next_tick=dots.next_tick.at[:, kind].set(nt))


def gust_bonus_ms(state: JungleState, now) -> Any:
    """(C,) Gustwalker's Gait MS fraction at ``now`` from the pet state (``combat_effects`` latches the
    brush entry; the world reads this at MOVE, one tick after the entry)."""
    pet = state.pet
    gust = (pet_stage(state) >= 2) & (pet.ptype == PET_GUSTWALKER)
    return jnp.where(gust, pet.gust_peak * jnp.clip(1.0 - (_f(now) - pet.gust_t0) / GUST_DECAY_S, 0.0, 1.0), 0.0)


def combat_effects(state: JungleState, table: JungleTable, units: WorldUnits, ctx, *, attack_hit, attack_target,
                   damaged_champion=None, in_brush=None) -> tuple[JungleState, JungleEffects]:
    """Per-tick jungle combat effects (phase 6, merge ``packets`` into the DAMAGE pass, ``cc`` into
    CC, ``heal``/``shield`` into the heal/shield effects, ``bonus_ms`` into champion move speed).

    * Crest of Cinders: champion basic-attack hits (``attack_hit``/``attack_target`` (C,), on-hit,
      incl. ranged arrivals) slow 10/15/25% (ranged half) for 3 s and burn 15 (+3/level from 6)
      true over 3 instances (on hit, +1 s, +2 s); hits on a burning target only refresh it.
    * Pets (holders of 1101-1103): every 1 s while a jungle monster within 650 targets the holder,
      the pet deals ``pet_damage`` true (AoE/pet, no omnivamp) to each such monster, healing the
      holder ``pet_heal`` (per pet attack). Bonus treats are stored every 60 s (90 s adult).
    * Evolution buffs (35 treats): Scorchclaw embers (6/s, max 100, refilled by large kills;
      ``damaged_champion`` (C,) int32 enemy champion the holder damaged this tick, -1 none:
      consumes them to burn the target and enemies within 250 for 5% max HP true over 4 s and slow
      30% for 3 s, decay not modelled), Gustwalker 30% MS decaying over 1.5 s on entering brush
      (``in_brush`` (C,) optional), Mosstomper shield after 10 s out of combat.
    """
    now = _f(ctx.now)
    c = state.blue_until.shape[0]
    n = units.x.shape[0]
    dots = state.dots
    pet = state.pet
    hit = jnp.asarray(attack_hit, bool) & jnp.asarray(ctx.alive, bool)
    tgt = jnp.asarray(attack_target, jnp.int32)
    ti = jnp.clip(tgt, 0, n - 1)
    tkind = jnp.asarray(units.kind)[ti]
    hostile = (tgt >= 0) & (jnp.asarray(units.team)[ti] != ctx.team) & jnp.asarray(units.alive)[ti] \
        & ((tkind == KIND_CHAMPION) | (tkind == KIND_MINION) | (tkind == KIND_MONSTER))
    # ---- Crest of Cinders on-hit --------------------------------------------------------------
    red_on = hit & hostile & (state.red_until > now)
    on = jnp.zeros((c, n), bool).at[jnp.arange(c), ti].set(red_on)                          # (C, N)
    any_on = jnp.any(on, axis=0)
    src = jnp.argmax(on, axis=0).astype(jnp.int32)
    lvl_src = _f(ctx.level)[src]
    burning = dots.until[:, DOT_RED] >= now
    fresh = any_on & ~burning
    per = red_burn_total(lvl_src) / RED_INSTANCES
    instant = D.packets(fresh, src, jnp.arange(n), jnp.where(fresh, per, 0.0), TRUE, DOT_FLAGS)
    dots = DotState(
        until=dots.until.at[:, DOT_RED].set(jnp.where(any_on, now + RED_BURN_S, dots.until[:, DOT_RED])),
        next_tick=dots.next_tick.at[:, DOT_RED].set(jnp.where(fresh, now + 1.0, dots.next_tick[:, DOT_RED])),
        per_tick=dots.per_tick.at[:, DOT_RED].set(jnp.where(fresh, per, dots.per_tick[:, DOT_RED])),
        src=dots.src.at[:, DOT_RED].set(jnp.where(fresh, src, dots.src[:, DOT_RED])))
    slow = jnp.where(on, red_slow(ctx.level, ctx.is_ranged)[:, None], 0.0)
    slow_d = jnp.where(on, RED_SLOW_S, 0.0)
    red_pk, dots = _dot_ticks(dots, DOT_RED, now, n)
    # ---- pets ---------------------------------------------------------------------------------
    ptype = pet.ptype
    holder = (ptype > 0) & jnp.asarray(ctx.alive, bool)
    idx = _slot_idx(table)
    m_alive = state.exists & jnp.asarray(units.alive)[idx]
    px, py = _f(units.x)[idx], _f(units.y)[idx]
    dpet = jnp.sqrt((px[None, :] - ctx.x[:, None]) ** 2 + (py[None, :] - ctx.y[:, None]) ** 2)
    attacking = m_alive[None, :] & (state.target[None, :] == jnp.arange(c)[:, None]) & (dpet <= PET_RADIUS)
    engaged = holder & jnp.any(attacking, axis=1)
    fire = engaged & (now >= pet.next_attack_at)
    next_attack = jnp.where(fire, now + PET_PERIOD, jnp.where(engaged, pet.next_attack_at, now))
    pdmg = pet_damage(ctx)
    hit_m = fire[:, None] & attacking
    pet_pk = D.packets(hit_m, jnp.arange(c, dtype=jnp.int32)[:, None], idx[None, :],
                       jnp.where(hit_m, pdmg[:, None], 0.0), TRUE, PET_FLAGS)
    heal = jnp.where(fire, pet_heal(ctx.level), 0.0)
    # Bonus treats.
    stage = pet_stage(state)
    start = (ptype > 0) & ~jnp.isfinite(pet.next_bonus_at)
    period = jnp.where(stage >= 2, BONUS_TREAT_PERIOD_ADULT, BONUS_TREAT_PERIOD)
    nb = jnp.where(start, now + period, pet.next_bonus_at)
    store = (ptype > 0) & (now >= nb)
    bonus = jnp.where(store, pet.bonus + 1, pet.bonus)
    nb = jnp.where(store, nb + period, nb)
    # ---- evolution buffs ----------------------------------------------------------------------
    adult = (stage >= 2)
    scorch = adult & (ptype == PET_SCORCHCLAW)
    embers = jnp.where(scorch, jnp.minimum(pet.embers + SCORCH_STACKS_PER_S * ctx.dt, SCORCH_MAX), 0.0)
    dc = jnp.full((c,), -1, jnp.int32) if damaged_champion is None else jnp.asarray(damaged_champion, jnp.int32)
    proc = scorch & (embers >= SCORCH_MAX) & (dc >= 0) & jnp.asarray(ctx.alive, bool)
    embers = jnp.where(proc, 0.0, embers)
    dci = jnp.clip(dc, 0, n - 1)
    dproc = jnp.sqrt((_f(units.x)[None, :] - _f(units.x)[dci][:, None]) ** 2
                     + (_f(units.y)[None, :] - _f(units.y)[dci][:, None]) ** 2)
    burn_area = proc[:, None] & (dproc <= 250.0 + _f(units.radius)[None, :]) & jnp.asarray(units.alive)[None, :] \
        & (jnp.asarray(units.team)[None, :] != ctx.team[:, None]) & (jnp.asarray(units.kind)[None, :] != KIND_NONE)
    sc_on = jnp.any(burn_area, axis=0)
    sc_src = jnp.argmax(burn_area, axis=0).astype(jnp.int32)
    sc_per = 0.05 * _f(units.max_hp) * SCORCH_TICK / SCORCH_BURN_S
    dots = DotState(
        until=dots.until.at[:, DOT_SCORCH].set(jnp.where(sc_on, now + SCORCH_BURN_S, dots.until[:, DOT_SCORCH])),
        next_tick=dots.next_tick.at[:, DOT_SCORCH].set(jnp.where(sc_on, now + SCORCH_TICK, dots.next_tick[:, DOT_SCORCH])),
        per_tick=dots.per_tick.at[:, DOT_SCORCH].set(jnp.where(sc_on, sc_per, dots.per_tick[:, DOT_SCORCH])),
        src=dots.src.at[:, DOT_SCORCH].set(jnp.where(sc_on, sc_src, dots.src[:, DOT_SCORCH])))
    sc_pk, dots = _dot_ticks(dots, DOT_SCORCH, now, n)
    sc_slow = jnp.where(burn_area, 0.30, 0.0)
    stronger = sc_slow > slow
    slow_d = jnp.where(stronger, 3.0, slow_d)
    slow = jnp.maximum(slow, sc_slow)
    gust = adult & (ptype == PET_GUSTWALKER)
    brush = jnp.zeros((c,), bool) if in_brush is None else jnp.asarray(in_brush, bool)
    enter = gust & brush & ~pet.in_brush
    cur_ms = pet.gust_peak * jnp.clip(1.0 - (now - pet.gust_t0) / GUST_DECAY_S, 0.0, 1.0)
    gp = jnp.where(enter & (cur_ms < 0.30), 0.30, pet.gust_peak)
    g0 = jnp.where(enter & (cur_ms < 0.30), now, pet.gust_t0)
    bonus_ms = jnp.where(gust, gp * jnp.clip(1.0 - (now - g0) / GUST_DECAY_S, 0.0, 1.0), 0.0)
    moss = adult & (ptype == PET_MOSSTOMPER) & jnp.asarray(ctx.alive, bool)
    in_combat = jnp.asarray(ctx.in_combat, bool)
    last_combat = jnp.where(in_combat, now, pet.last_combat)
    # ``in_combat`` covers the 5 s after the last combat event, so 10 s out of combat is 5 s after
    # it turns off (INFERRED-M); regrant only if a fight happened since the last grant.
    ooc_long = ~in_combat & ((now - last_combat) >= MOSS_OOC_S - 5.0) & (pet.moss_granted_at < last_combat)
    grant = moss & (pet.moss_pending | ooc_long)
    shield = jnp.where(grant, moss_shield(ctx.level), 0.0)
    moss_at = jnp.where(grant, now, pet.moss_granted_at)
    pet = pet._replace(next_attack_at=_f(next_attack), bonus=bonus.astype(jnp.int32), next_bonus_at=_f(nb),
                       embers=_f(embers), gust_peak=_f(gp), gust_t0=_f(g0), in_brush=brush, moss_granted_at=_f(moss_at),
                       last_combat=_f(last_combat), moss_pending=pet.moss_pending & ~grant)
    cc = no_cc(c, n)._replace(slow=_f(slow), slow_duration=_f(slow_d))
    state = state._replace(dots=dots, pet=pet)
    return state, JungleEffects(D.concat_packets(instant, red_pk, pet_pk, sc_pk), cc, _f(heal), _f(shield),
                                _f(bonus_ms))


# =================================================================================================
# 6. deaths and rewards
# =================================================================================================
class JungleRewards(NamedTuple):
    """Per champion (C,) unless noted; route gold/XP through the economy (GV accrual: monster gold
    counts; XP gets the role-quest XP modifiers like minion XP)."""
    gold: Any
    xp: Any
    heal: Any                   # jungle-item kill heal (route through heal power)
    mana: Any                   # jungle-item kill mana
    energy: Any                 # jungle-item kill energy (energy users)
    large_kills: Any            # int32
    killed: Any                 # (C, S) bool monster takedowns this tick (Kills.killed_units)
    consume_pet: Any            # bool: remove the pet egg from the inventory (final evolution)
    quest_completed: Any        # bool: jungle role quest completed this tick
    shrine_team: Any            # (2,) int32 river Speed Shrine owner (-1 none); 0 Baron river
    shrine_until: Any           # (2,)
    blue_granted: Any           # bool
    red_granted: Any            # bool


def death_step(state: JungleState, table: JungleTable, units: WorldUnits, *, now, died, killer, avg_level,
               champion_level, hp, max_hp, mana, max_mana, champion_died=None, champion_killer=None,
               champion_takedowns=None, minion_gold=None) -> tuple[JungleState, JungleRewards]:
    """Jungle deaths of this tick (phase 9).

    ``died`` (N,) and ``killer`` (N,) int32 (killing-blow source, -1) as computed by the step.
    ``avg_level`` () decimal average champion level; ``champion_level`` (C,) decimal levels;
    ``hp/max_hp/mana/max_mana`` (C,) for the kill heal. Optional: ``champion_died`` (C,) bool and
    ``champion_killer`` (C,) int32 (champion unit or -1) for Crest transfers; ``champion_takedowns``
    (C,) int32 (pet treats); ``minion_gold`` (C,) minion gold earned this tick (Monster Hunter).

    Rewards go to the champion landing the killing blow only (WIKI Experience: "full bounty to the
    killer"); monsters killed by non-champions or despawned give nothing.
    """
    now = _f(now)
    idx = _slot_idx(table)
    c = state.blue_until.shape[0]
    d = jnp.asarray(died, bool)[idx] & state.exists
    k = jnp.asarray(killer, jnp.int32)[idx]
    by = d[:, None] & (k[:, None] == jnp.arange(c)[None, :])                               # (S, C)
    st = monster_stats(table, state.mtype, state.level, state.first_crab)
    size = table.t_size[state.mtype]
    large = by & (size == LARGE)[:, None]
    gold = jnp.sum(jnp.where(by, st.gold[:, None], 0.0), axis=0)
    xp = jnp.sum(jnp.where(by, st.xp[:, None], 0.0), axis=0)
    n_large = jnp.sum(large, axis=0).astype(jnp.int32)
    # ---- jungle item --------------------------------------------------------------------------
    pet = state.pet
    holder = pet.ptype > 0
    stage0 = pet_stage(state)
    lv = _f(champion_level)
    behind = _f(avg_level) - lv
    comeback = jnp.where(behind > COMEBACK_THRESHOLD, COMEBACK_XP * jnp.round(behind), 0.0)
    has_large = n_large > 0
    xp = xp + jnp.where(holder, (TREAT_XP + comeback) * n_large, 0.0)
    xp = xp + jnp.where(holder & has_large & ~pet.first_large_done, FIRST_LARGE_XP, 0.0)
    quest = holder & (stage0 >= 2)
    gold = gold + jnp.where(quest, QUEST_GOLD * n_large, 0.0)
    xp = xp + jnp.where(quest, QUEST_XP * n_large, 0.0)
    take = jnp.zeros((c,), jnp.int32) if champion_takedowns is None else jnp.asarray(champion_takedowns, jnp.int32)
    use_bonus = jnp.where(holder & has_large, jnp.minimum(pet.bonus, jnp.where(stage0 >= 2, 2, 1)), 0)
    treats = pet.treats + jnp.where(holder, n_large + take + jnp.where(stage0 < 2, use_bonus, 0), 0)
    gold = gold + BONUS_TREAT_GOLD * use_bonus
    stage1 = jnp.where(holder & (treats >= FINAL_EVOLUTION), 2, jnp.where(holder & (treats >= FIRST_EVOLUTION), 1, 0))
    completed = (stage1 >= 2) & (stage0 < 2)
    avg = jnp.clip(_f(avg_level), 1.0, 18.0)
    miss_hp = 1.0 - _f(hp) / jnp.maximum(_f(max_hp), 1.0)
    miss_mp = 1.0 - _f(mana) / jnp.maximum(_f(max_mana), 1.0)
    max_heal = jnp.minimum(table.heal_base + table.heal_per_level * avg, table.heal_cap)
    heal = jnp.where(holder & has_large, max_heal * jnp.clip(1.25 * miss_hp, 0.0, 1.0), 0.0)
    mana_r = jnp.where(holder & has_large & (_f(max_mana) > 0),
                       (table.mana_base + table.mana_per_level * avg)
                       * jnp.minimum(1.0 + 1.25 * miss_mp, table.mana_max_mult), 0.0)
    energy = jnp.where(holder & has_large, table.energy_restore, 0.0)
    embers = jnp.where(has_large & (pet.ptype == PET_SCORCHCLAW), SCORCH_MAX, pet.embers)
    gust = has_large & (pet.ptype == PET_GUSTWALKER) & (stage1 >= 2)
    moss_now = has_large & (pet.ptype == PET_MOSSTOMPER) & (stage1 >= 2)
    mg = jnp.zeros((c,), jnp.float32) if minion_gold is None else _f(minion_gold)
    monster_gold = pet.monster_gold + gold
    pet = pet._replace(
        treats=treats.astype(jnp.int32), bonus=(pet.bonus - use_bonus).astype(jnp.int32),
        first_large_done=pet.first_large_done | (holder & has_large), embers=_f(embers),
        gust_peak=jnp.where(gust, 0.45, pet.gust_peak).astype(jnp.float32),
        gust_t0=jnp.where(gust, now, pet.gust_t0).astype(jnp.float32),
        moss_pending=pet.moss_pending | moss_now | (completed & (pet.ptype == PET_MOSSTOMPER)),
        minion_gold=_f(pet.minion_gold + mg), monster_gold=_f(monster_gold))
    # ---- crests -------------------------------------------------------------------------------
    blue_k = by & (state.mtype == Monster.BLUE)[:, None]
    red_k = by & (state.mtype == Monster.RED)[:, None]
    blue_g, red_g = jnp.any(blue_k, axis=0), jnp.any(red_k, axis=0)
    blue_until = jnp.where(blue_g, now + CREST_DURATION, state.blue_until)
    red_until = jnp.where(red_g, now + CREST_DURATION, state.red_until)
    if champion_died is not None:
        cd = jnp.asarray(champion_died, bool)
        ck = jnp.full((c,), -1, jnp.int32) if champion_killer is None else jnp.asarray(champion_killer, jnp.int32)
        had_b, had_r = cd & (blue_until > now), cd & (red_until > now)
        to = (ck >= 0) & (ck < c)
        gain_b = jnp.zeros((c,), bool).at[jnp.clip(ck, 0, c - 1)].max(had_b & to)
        gain_r = jnp.zeros((c,), bool).at[jnp.clip(ck, 0, c - 1)].max(had_r & to)
        blue_until = jnp.where(cd, -1.0, jnp.where(gain_b, now + CREST_DURATION, blue_until))
        red_until = jnp.where(cd, -1.0, jnp.where(gain_r, now + CREST_DURATION, red_until))
    # ---- Scuttler shrine and cycle --------------------------------------------------------------
    crab_d = d & (state.mtype == Monster.SCUTTLE)
    river = jnp.where(table.slot_camp == 12, 0, 1)
    crab_by = by & (state.mtype == Monster.SCUTTLE)[:, None]
    killer_team = jnp.asarray(units.team)[jnp.clip(k, 0, units.x.shape[0] - 1)]
    crab_kill = jnp.any(crab_by, axis=1)
    sh_new = jnp.stack([jnp.any(crab_kill & (river == r)) for r in (0, 1)])
    sh_team = jnp.stack([jnp.max(jnp.where(crab_kill & (river == r), killer_team, -1)) for r in (0, 1)])
    shrine_team = jnp.where(sh_new, sh_team, jnp.where(state.shrine_until > now, state.shrine_team, -1))
    shrine_until = jnp.where(sh_new, now + SHRINE_DURATION, state.shrine_until)
    n_crab = jnp.sum(crab_d).astype(jnp.int32)
    initial_dead = jnp.sum(crab_d & state.first_crab).astype(jnp.int32)
    left = state.crab_initial_left - initial_dead
    later_dead = n_crab - initial_dead
    crab_respawn = jnp.where((initial_dead > 0) & (left <= 0) & (state.crab_initial_left > 0), now + RESPAWN[CAMP_SCUTTLE],
                             state.crab_respawn_at)
    crab_respawn = jnp.where(later_dead > 0, now + RESPAWN[CAMP_SCUTTLE], crab_respawn)
    # ---- Krug splits, marked for death, camp timers -------------------------------------------
    splitting = d & ((state.mtype == Monster.KRUG) | (state.mtype == Monster.KRUG_MEDIUM))
    par = table.slot_parent
    pi = jnp.clip(par, 0, table.n_slots - 1)
    child = (par >= 0) & splitting[pi]
    dx, dy = _f(units.x)[idx], _f(units.y)[idx]
    spawn_at = jnp.where(child, now + SPLIT_DELAY, state.spawn_at)
    spawn_x = jnp.where(child, dx[pi] + table.slot_offset[:, 0], state.spawn_x)
    spawn_y = jnp.where(child, dy[pi] + table.slot_offset[:, 1], state.spawn_y)
    spawn_level = jnp.where(child, jnp.maximum(state.level[pi] - 1, 1), state.spawn_level)
    large_dead = _camp_any(table, d & (size == LARGE))
    camp_marked = state.camp_marked | large_dead
    s_ = state._replace(
        spawn_at=_f(spawn_at), spawn_x=_f(spawn_x), spawn_y=_f(spawn_y), spawn_level=spawn_level.astype(jnp.int32),
        camp_marked=camp_marked, blue_until=_f(blue_until), red_until=_f(red_until), pet=pet,
        crab_initial_left=jnp.maximum(left, 0).astype(jnp.int32), crab_respawn_at=_f(crab_respawn),
        shrine_team=shrine_team.astype(jnp.int32), shrine_until=_f(shrine_until),
        target=jnp.where(d, -1, state.target).astype(jnp.int32), aggro=state.aggro & ~d)
    s_ = _clear_camps(s_, table, state.exists & ~d, now)
    return s_, JungleRewards(_f(gold), _f(xp), _f(heal), _f(mana_r), _f(energy), n_large, by.T, completed, completed,
                             s_.shrine_team, s_.shrine_until, blue_g, red_g)


# =================================================================================================
# 7. stats and economy hooks
# =================================================================================================
class BuffStats(NamedTuple):
    ability_haste: Any          # (C,) Crest of Insight 10/15/20 at 1/6/11
    mana_per_s: Any             # (C,) Crest of Insight 5 + 1% max mana per second (mana or energy)
    hp_regen_per_s: Any         # (C,) Crest of Cinders 0.5/1/1.5/3% max HP per 5 s out of champion combat
    shrine_ms: Any              # (C,) Speed Shrine 30% for 1.5 s (apply when > 0)


def buff_stats(state: JungleState, *, now, level, max_mana, max_hp, x, y, team, champion_combat_recent,
               dealt_damage_recent=None, shrine_pos=None) -> BuffStats:
    """Crest stats for the STATS phase (CLIENT CrestoftheAncientGolem / BlessingoftheLizardElder
    calculations) and the Speed Shrine bonus (``shrine_pos`` (2, 2): river shrine centres, default
    the Scuttler camp markers - INFERRED-M; the shrine also grants SHRINE_SIGHT vision to its team:
    feed ``state.shrine_team/shrine_until`` to ``vision`` as a ward-like source)."""
    now = _f(now)
    lv = _f(level)
    blue = state.blue_until > now
    red = state.red_until > now
    ah = jnp.where(blue, 10.0 + jnp.where(lv >= 6, 5.0, 0.0) + jnp.where(lv >= 11, 5.0, 0.0), 0.0)
    mana = jnp.where(blue, 5.0 + 0.01 * _f(max_mana), 0.0)
    pct = 0.005 + jnp.where(lv >= 4, 0.005, 0.0) + jnp.where(lv >= 6, 0.005, 0.0) + jnp.where(lv >= 11, 0.015, 0.0)
    regen = jnp.where(red & ~jnp.asarray(champion_combat_recent, bool), pct * _f(max_hp) / 5.0, 0.0)
    pos = jnp.asarray(SHRINE_POS, jnp.float32) if shrine_pos is None else _f(shrine_pos)
    d = jnp.sqrt((_f(x)[:, None] - pos[None, :, 0]) ** 2 + (_f(y)[:, None] - pos[None, :, 1]) ** 2)
    own = (state.shrine_team[None, :] == jnp.asarray(team, jnp.int32)[:, None]) & (state.shrine_until[None, :] > now)
    calm = jnp.ones_like(blue) if dealt_damage_recent is None else ~jnp.asarray(dealt_damage_recent, bool)
    shrine = jnp.any(own & (d <= SHRINE_RADIUS), axis=1) & calm
    return BuffStats(_f(ah), _f(mana), _f(regen), jnp.where(shrine, SHRINE_MS, 0.0).astype(jnp.float32))


def minion_reward_mods(state: JungleState, *, now, champion_level, avg_level) -> tuple[Any, Any]:
    """``(gold_delta, xp_mult)`` (C,) on lane-minion rewards for jungle-item holders (WIKI):
    XP -70% at 0:00 shrinking linearly to 0 at 20:00 unless 1.5+ levels behind; Monster Hunter
    (minion gold > 40% of monster gold, before 14:00): -13 gold and -50% XP per minion."""
    now = _f(now)
    holder = state.pet.ptype > 0
    behind = (_f(avg_level) - _f(champion_level)) >= MINION_XP_BEHIND
    red = MINION_XP_REDUCTION * jnp.clip(1.0 - now / MINION_XP_REDUCTION_END, 0.0, 1.0)
    hunter = holder & (now < MONSTER_HUNTER_END) & \
        (state.pet.minion_gold > MONSTER_HUNTER_SHARE * state.pet.monster_gold)
    mult = jnp.where(holder & ~behind, 1.0 - red, 1.0) * jnp.where(hunter, 0.5, 1.0)
    return jnp.where(hunter, -MONSTER_HUNTER_GOLD, 0.0).astype(jnp.float32), _f(mult)


def quest_jungle_ms(state: JungleState, *, in_jungle, in_combat) -> Any:
    """(C,) jungle-quest MS fraction in the jungle/river (PATCH 26.1: 4%, 8% out of combat)."""
    done = pet_stage(state) >= 2
    return jnp.where(done & jnp.asarray(in_jungle, bool),
                     jnp.where(jnp.asarray(in_combat, bool), QUEST_MS_IN_COMBAT, QUEST_MS_OUT), 0.0).astype(jnp.float32)


def pet_mana_regen(state: JungleState, *, level, mana, max_mana, in_jungle) -> Any:
    """(C,) mana/s in jungle or river for holders (WIKI): (8% + L*8*0.1/1.3%) of missing mana."""
    pct = 0.08 + _f(level) * 8.0 * 0.1 / 1.3 / 100.0
    return jnp.where((state.pet.ptype > 0) & jnp.asarray(in_jungle, bool),
                     pct * jnp.maximum(_f(max_mana) - _f(mana), 0.0), 0.0).astype(jnp.float32)


def latch_pets(state: JungleState, own) -> JungleState:
    """Adopt pet types from the inventory (keep them after the egg is consumed)."""
    p = pet_type_from_inventory(own)
    return state._replace(pet=state.pet._replace(ptype=jnp.where(p > 0, p, state.pet.ptype).astype(jnp.int32)))
