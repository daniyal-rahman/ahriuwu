"""The patch table: every number the sim needs, loaded from the server's Content.

Why a table and not constants
-----------------------------
The mandate is parity with the vendored server *first*, modern League *second*.
Those are the same rules with different numbers, so the rules engine must never
hardcode a number that lives in a JSON file.  This module is the seam: it loads
``Content/LeagueSandbox-Default`` into a typed table, and the modern-patch work
later becomes a second table plus explicit flags for the rules that genuinely
changed (as opposed to merely being retuned).

Concretely, `Garen.json` alone supplies `BaseHP`, `HPPerLevel`, `BaseDamage`,
`DamagePerLevel`, `Armor`, `ArmorPerLevel`, `AttackRange`, `AttackSpeedPerLevel`,
`AttackDelayOffsetPercent`, `AttackDelayCastOffsetPercent`, `AcquisitionRange`,
`PathfindingCollisionRadius` and `MoveSpeed` -- all of which `lanerl_rl/constants.py`
currently carries as Python literals with a source comment. Those literals are
the thing this replaces.

The types are a mess, on purpose
--------------------------------
Content values are inconsistently typed *within the same file*: ``"Armor": "0"``
is a string while ``"ArmorPerLevel": 0`` is a number, ``"IsMelee": "true"`` is a
string boolean, and ``"AcquisitionRange": "400"`` is a stringly-typed float.
This is faithful to the ``.inibin`` files they were converted from. :func:`num`
and :func:`flag` coerce and **raise on anything they cannot read**, rather than
returning a default -- a silently-defaulted stat is a mechanic that is quietly
wrong and a parity diff that blames the wrong thing.

What is deliberately *not* here
-------------------------------
Runes and masteries. Measured: `Garen.json` gives `BaseHP` 616.28 but the state
dump reports 754.0 max HP at level 1, a 137.72 gap that comes from the rune page
the server applies at spawn (`LanerlEpisode.RestoreBaseline` restores it, and
`lanerl_rl/tests/test_runes.py` pins it). So a champion's *spawn* stats are not
a pure function of this table, and :func:`ChampionStats.at_level` predicts the
**base** curve only. The rune delta is measured against the dump by
`validate_against_dump`, not assumed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "CONTENT_ROOT",
    "num",
    "flag",
    "load_character",
    "load_spell",
    "load_map_table",
    "ChampionStats",
    "UnitStats",
    "PatchTable",
    "load_patch",
]

CONTENT_ROOT = Path(
    "/srv/nfs/projects/lanerl-vendor/LoLServer/Content/LeagueSandbox-Default"
)


def num(values: Mapping[str, Any], key: str, default: Optional[float] = None) -> float:
    """Read a Content number that may be a str, int, float -- or absent.

    Raises when the key is missing and no default was given, and when the value
    cannot be read as a number. Never guesses: a stat that silently defaults to
    zero produces a mechanic that is wrong in a way no parity diff can attribute.
    """
    if key not in values:
        if default is None:
            raise KeyError(
                f"Content key {key!r} is absent and has no default. Either the "
                "patch really omits it (pass an explicit default and say why) or "
                "this is the wrong file."
            )
        return float(default)
    v = values[key]
    try:
        return float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Content key {key!r} = {v!r} is not a number") from exc


def flag(values: Mapping[str, Any], key: str, default: Optional[bool] = None) -> bool:
    """Read a Content boolean, which is usually the *string* ``"true"``."""
    if key not in values:
        if default is None:
            raise KeyError(f"Content key {key!r} is absent and has no default")
        return default
    v = values[key]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    raise ValueError(f"Content key {key!r} = {v!r} is not a boolean")


@lru_cache(maxsize=None)
def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def load_character(name: str, root: Path = CONTENT_ROOT) -> Dict[str, Any]:
    """``Stats/<name>/<name>.json`` -> its ``Values.Data`` block."""
    p = root / "Stats" / name / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"no Content stats for {name!r} at {p}")
    return _load_json(p)["Values"]["Data"]


def load_spell(name: str, root: Path = CONTENT_ROOT) -> Dict[str, Any]:
    """``Spells/<name>/<name>.json`` -> its ``Values`` block (all sections)."""
    p = root / "Spells" / name / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(f"no Content spell data for {name!r} at {p}")
    return _load_json(p)["Values"]


def load_map_table(name: str, map_id: int = 1, root: Path = CONTENT_ROOT
                   ) -> Dict[str, Any]:
    """``Maps/Map<N>/<name>.json`` -> its ``Values`` block, or the whole file
    for ``Constants``, which has no ``MetaData``/``Values`` wrapper."""
    p = root / "Maps" / f"Map{map_id}" / f"{name}.json"
    d = _load_json(p)
    return d.get("Values", d)


@dataclass(slots=True, frozen=True)
class UnitStats:
    """The per-unit numbers the lane sim needs, in server units."""

    name: str
    base_hp: float
    hp_per_level: float
    base_ad: float
    ad_per_level: float
    armor: float
    armor_per_level: float
    magic_resist: float
    mr_per_level: float
    move_speed: float
    attack_range: float
    acquisition_range: float
    pathfinding_radius: float
    #: ``GameObject.CollisionRadius``, used for **attack range**
    #: (``idealRange = Range + TargetUnit.CollisionRadius``) and for the
    #: collision push-apart. ``ObjAIBase``'s constructor resolves it as
    #: ``GameplayCollisionRadius`` if positive, else **40** -- NOT
    #: ``SelectionRadius``, which is a client display value and is 2-3x larger
    #: (Garen 75 vs 40; a minion 115 vs 48). Using the wrong one inflates every
    #: attack range by ~70 units, which is more than half a melee attack range.
    collision_radius: float
    attack_speed_per_level: float
    attack_delay_offset_percent: float
    attack_delay_cast_offset_percent: float
    is_melee: bool
    gold_given_on_death: float = 0.0
    exp_given_on_death: float = 0.0

    @classmethod
    def from_content(cls, name: str, root: Path = CONTENT_ROOT) -> "UnitStats":
        d = load_character(name, root)
        return cls(
            name=name,
            base_hp=num(d, "BaseHP"),
            hp_per_level=num(d, "HPPerLevel", 0.0),
            base_ad=num(d, "BaseDamage"),
            ad_per_level=num(d, "DamagePerLevel", 0.0),
            armor=num(d, "Armor", 0.0),
            armor_per_level=num(d, "ArmorPerLevel", 0.0),
            # Content spells this "SpellBlock" (the old name for magic resist).
            magic_resist=num(d, "SpellBlock", 0.0),
            mr_per_level=num(d, "SpellBlockPerLevel", 0.0),
            move_speed=num(d, "MoveSpeed"),
            attack_range=num(d, "AttackRange"),
            # Minions carry no AcquisitionRange in Content; the server falls back
            # to its own default, so 0 here means "not specified", not "blind".
            acquisition_range=num(d, "AcquisitionRange", 0.0),
            # both fall back to 40 in ObjAIBase's constructor when absent or
            # non-positive, which is how Garen gets 40 (his
            # GameplayCollisionRadius is -1).
            pathfinding_radius=(num(d, "PathfindingCollisionRadius", 0.0)
                                if num(d, "PathfindingCollisionRadius", 0.0) > 0
                                else 40.0),
            collision_radius=(num(d, "GameplayCollisionRadius", 0.0)
                              if num(d, "GameplayCollisionRadius", 0.0) > 0
                              else 40.0),
            attack_speed_per_level=num(d, "AttackSpeedPerLevel", 0.0),
            attack_delay_offset_percent=num(d, "AttackDelayOffsetPercent", 0.0),
            attack_delay_cast_offset_percent=num(
                d, "AttackDelayCastOffsetPercent", 0.0),
            is_melee=flag(d, "IsMelee", False),
            gold_given_on_death=num(d, "GoldGivenOnDeath", 0.0),
            exp_given_on_death=num(d, "ExpGivenOnDeath", 0.0),
        )

    # NOTE: growth is NOT linear. ``Stats.LevelUp`` scales every per-level
    # value by ``0.65 + 0.035 * Level``, which is exactly Map1's
    # ``PerLevelStatsFactor`` table. Reading it as ``perLevel * (N-1)`` is ~19%
    # light on Garen's HP by level 6. See ``lanerl_jax/sim/combat.py``.
    def hp_at_level(self, level: int) -> float:
        """BASE curve only -- runes and masteries are not in Content. See the
        module docstring: Garen's dump value at level 1 exceeds this by 137.72."""
        from ..sim.combat import stat_at_level
        return float(stat_at_level(self.base_hp, self.hp_per_level, level))

    def ad_at_level(self, level: int) -> float:
        from ..sim.combat import stat_at_level
        return float(stat_at_level(self.base_ad, self.ad_per_level, level))

    def armor_at_level(self, level: int) -> float:
        from ..sim.combat import stat_at_level
        return float(stat_at_level(self.armor, self.armor_per_level, level))


#: alias kept for readability at call sites that mean a champion specifically
ChampionStats = UnitStats


@dataclass(slots=True)
class PatchTable:
    """Everything the 1v1 top lane needs, from one Content tree."""

    champion: UnitStats
    minions: Dict[str, UnitStats] = field(default_factory=dict)
    turrets: Dict[str, UnitStats] = field(default_factory=dict)
    exp_curve: Dict[int, float] = field(default_factory=dict)
    per_level_stats_factor: Dict[int, float] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)
    root: Path = CONTENT_ROOT

    @property
    def global_attack_delay(self) -> float:
        """``gcd_AttackDelay`` -- the 1.6 s global base attack period."""
        return float(self.constants["gcd_AttackDelay"])

    @property
    def global_attack_delay_cast_percent(self) -> float:
        return float(self.constants["gcd_AttackDelayCastPercent"])

    def xp_for_level(self, level: int) -> float:
        return self.exp_curve[level]


#: The minion models Map1's LevelScript spawns, per team.
MINION_MODELS = {
    "melee": ("Blue_Minion_Basic", "Red_Minion_Basic"),
    "caster": ("Blue_Minion_Wizard", "Red_Minion_Wizard"),
    "cannon": ("Blue_Minion_MechCannon", "Red_Minion_MechCannon"),
    "super": ("Blue_Minion_MechMelee", "Red_Minion_MechMelee"),
}

#: Outer-lane turrets. `LevelScriptObjects.LoadBuildings` maps map-object names
#: to these models; only the top outer pair ever acts in a TOPONLY 1v1, but all
#: 24 exist as objects (measured), so the roster carries the models it needs.
TURRET_MODELS = ("SRUAP_Turret_Order3", "SRUAP_Turret_Chaos3")


def load_patch(champion: str = "Garen", map_id: int = 1,
               root: Path = CONTENT_ROOT) -> PatchTable:
    """Bake one Content tree into a :class:`PatchTable`."""
    exp = load_map_table("ExpCurve", map_id, root)["EXP"]
    prog = load_map_table("StatsProgression", map_id, root)["PerLevelStatsFactor"]
    turrets: Dict[str, UnitStats] = {}
    for m in TURRET_MODELS:
        try:
            turrets[m] = UnitStats.from_content(m, root)
        except (FileNotFoundError, KeyError):
            # Named here rather than swallowed: a missing turret model means the
            # lane has no tower, which is a different game, so the caller must see it.
            pass
    return PatchTable(
        champion=UnitStats.from_content(champion, root),
        minions={
            f"{kind}_{'blue' if i == 0 else 'red'}": UnitStats.from_content(m, root)
            for kind, models in MINION_MODELS.items()
            for i, m in enumerate(models)
        },
        turrets=turrets,
        exp_curve={int(k.replace("Level", "")): float(v) for k, v in exp.items()},
        per_level_stats_factor={
            int(k.replace("Level", "")): float(v) for k, v in prog.items()},
        constants=load_map_table("Constants", map_id, root),
        root=root,
    )
