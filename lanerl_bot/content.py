"""Champion and minion numbers, read from the server's own content files.

Nothing in here is typed by hand. If the server's content changes, these change with
it, and the tests that depend on them fail loudly rather than passing against a stale
constant someone copied out of a wiki.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

# The NFS export is mounted at /srv/nfs on danilogin and /mnt/nfs on desktop.
# Never hardcode either: resolve everything from this file's own location.
_PROJECTS = Path(__file__).resolve().parents[2]
_VENDOR = str(_PROJECTS / "lanerl-vendor")
_REPO = _PROJECTS / "ahriuwu-lanerl"

CONTENT = Path(_VENDOR + "/LoLServer/Content/LeagueSandbox-Default")

# Global constants the engine divides attack timings by
# (GameServerLib/Content/GlobalData/GlobalData.cs).
GCD_ATTACK_DELAY = 1.600
GCD_ATTACK_DELAY_CAST_PERCENT = 0.300

# Map1 wave composition, from Content/.../Maps/Map1/LevelScript.cs
SPAWN_INTERVAL_MS = 30_000
FIRST_WAVE_MS = 90_000
INTRA_WAVE_GAP_MS = 800

STARTING_GOLD = 475.0


def _f(v, default=0.0) -> float:
    if v is None:
        return default
    return float(v)


@functools.lru_cache(maxsize=None)
def _stats_json(name: str) -> dict:
    p = CONTENT / "Stats" / name / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"no content for {name}: {p}")
    return json.loads(p.read_text())["Values"]["Data"]


@dataclass(frozen=True)
class UnitStats:
    """The subset of a unit's content record the last-hit model needs."""

    name: str
    base_hp: float
    base_damage: float
    armor: float
    move_speed: float
    attack_range: float
    gold_on_death: float
    exp_on_death: float
    collision_radius: float
    attack_delay_offset_pct: float
    attack_delay_cast_offset_pct: float
    hp_regen: float

    @property
    def attack_period(self) -> float:
        """Seconds between autos at base attack speed."""
        return GCD_ATTACK_DELAY * (1.0 + self.attack_delay_offset_pct)

    @property
    def windup(self) -> float:
        """Seconds from starting an auto to the damage landing."""
        return self.attack_period * (
            GCD_ATTACK_DELAY_CAST_PERCENT + self.attack_delay_cast_offset_pct
        )


@functools.lru_cache(maxsize=None)
def unit(name: str) -> UnitStats:
    d = _stats_json(name)
    return UnitStats(
        name=name,
        base_hp=_f(d.get("BaseHP")),
        base_damage=_f(d.get("BaseDamage")),
        armor=_f(d.get("Armor")),
        move_speed=_f(d.get("MoveSpeed")),
        attack_range=_f(d.get("AttackRange")),
        gold_on_death=_f(d.get("GoldGivenOnDeath")),
        exp_on_death=_f(d.get("ExpGivenOnDeath")),
        collision_radius=_f(d.get("PathfindingCollisionRadius")),
        attack_delay_offset_pct=_f(d.get("AttackDelayOffsetPercent")),
        attack_delay_cast_offset_pct=_f(d.get("AttackDelayCastOffsetPercent")),
        hp_regen=_f(d.get("BaseStaticHPRegen")),
    )


@dataclass(frozen=True)
class ChampionStats:
    name: str
    base_hp: float
    hp_per_level: float
    base_damage: float
    damage_per_level: float
    armor: float
    armor_per_level: float
    attack_range: float
    move_speed: float
    attack_delay_offset_pct: float
    attack_delay_cast_offset_pct: float
    is_melee: bool

    def attack_damage(self, level: int) -> float:
        """Stats.LevelUp applies growth * (0.65 + 0.035 * level) per level up."""
        ad = self.base_damage
        for lv in range(1, level):
            ad += self.damage_per_level * (0.65 + 0.035 * lv)
        return ad

    def max_hp(self, level: int) -> float:
        hp = self.base_hp
        for lv in range(1, level):
            hp += self.hp_per_level * (0.65 + 0.035 * lv)
        return hp

    @property
    def attack_period(self) -> float:
        return GCD_ATTACK_DELAY * (1.0 + self.attack_delay_offset_pct)

    @property
    def windup(self) -> float:
        return self.attack_period * (
            GCD_ATTACK_DELAY_CAST_PERCENT + self.attack_delay_cast_offset_pct
        )


@functools.lru_cache(maxsize=None)
def champion(name: str = "Garen") -> ChampionStats:
    d = _stats_json(name)
    return ChampionStats(
        name=name,
        base_hp=_f(d.get("BaseHP")),
        hp_per_level=_f(d.get("HPPerLevel")),
        base_damage=_f(d.get("BaseDamage")),
        damage_per_level=_f(d.get("DamagePerLevel")),
        armor=_f(d.get("Armor")),
        armor_per_level=_f(d.get("ArmorPerLevel")),
        attack_range=_f(d.get("AttackRange")),
        move_speed=_f(d.get("MoveSpeed")),
        attack_delay_offset_pct=_f(d.get("AttackDelayOffsetPercent")),
        attack_delay_cast_offset_pct=_f(d.get("AttackDelayCastOffsetPercent")),
        is_melee=str(d.get("IsMelee", "")).lower() == "true",
    )


@functools.lru_cache(maxsize=None)
def spell(name: str) -> dict:
    p = CONTENT / "Spells" / name / f"{name}.json"
    return json.loads(p.read_text())["Values"]["SpellData"]


def spell_cooldowns(name: str) -> list[float]:
    d = spell(name)
    return [_f(d.get(f"Cooldown{i}")) for i in range(1, 6)]


def spell_effect(name: str, effect: int) -> list[float]:
    d = spell(name)
    return [_f(d.get(f"Effect{effect}Level{i}Amount")) for i in range(1, 6)]


# The four lane-minion models Map1 actually spawns for each side.
BLUE_MINIONS = {
    "melee": "Blue_Minion_Basic",
    "caster": "Blue_Minion_Wizard",
    "cannon": "Blue_Minion_MechCannon",
    "super": "Blue_Minion_MechMelee",
}
RED_MINIONS = {
    "melee": "Red_Minion_Basic",
    "caster": "Red_Minion_Wizard",
    "cannon": "Red_Minion_MechCannon",
    "super": "Red_Minion_MechMelee",
}
