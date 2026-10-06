"""Champion stat composition in STAT.* order (DAMAGE_AND_STATS §3, §8-11), pure elementwise JAX.

STAT.00 base -> .10 growth -> .20 flat -> .30 additive % -> .40 multiplicative -> .50 adaptive force -> .60 caps
-> .70 max-HP sync. Every bonus source arrives combined in one ``ItemStats``.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..data.champions import champion
from ..items.catalog import ItemStats
from .stats import level_growth_sum, resolve_adaptive

AS_MIN, AS_MAX = 0.2, 1.0 / 0.333          # client gcd_AttackMaxDelay 5.0 / gcd_AttackMinDelay 0.333
AS_UNCAPPED = 1e3                           # Hail of Blades lifts the cap (RUNES §4.3)
HASTE_CAP = 500.0                           # wiki
TENACITY_FLOOR_SECONDS = 0.3                # §10.2


class ChampionBase(NamedTuple):
    """Client champion record values, (C,)."""
    base_hp: Any
    hp_per_level: Any
    base_ad: Any
    ad_per_level: Any
    base_armor: Any
    armor_per_level: Any
    base_mr: Any
    mr_per_level: Any
    base_ms: Any
    attack_range: Any
    attack_speed: Any               # AS at level 1 with no bonus
    attack_speed_ratio: Any
    attack_speed_per_level: Any     # percent (3.65 = 3.65%)
    windup_percent: Any             # 0.3 + mAttackDelayCastOffsetPercent
    windup_modifier: Any            # mAttackDelayCastOffsetPercentAttackSpeedRatio (default 1.0)
    hp_regen: Any                   # per second
    hp_regen_per_level: Any
    base_mana: Any = 0.0
    mana_per_level: Any = 0.0
    mana_regen: Any = 0.0
    mana_regen_per_level: Any = 0.0


# AbilityResourceSlotInfo hashed fields: base mana, per level, regen, regen per level (identified on Jax's record).
AR_BASE, AR_PER_LEVEL, AR_REGEN, AR_REGEN_PER_LEVEL = "{726ee5cd}", "{6216bf7b}", "{c4ab3550}", "{3a509002}"


def champion_base(names) -> ChampionBase:
    """Host-side ``ChampionBase`` from the pinned 26.19 champion records."""
    def field(name, key, default=0.0):
        rec = champion(name)["character"]
        return float(rec[key]["baseValue"]) if key in rec else default

    def resource(name, key):
        par = champion(name)["character"].get("primaryAbilityResource", {})
        return float(par[key]["baseValue"]) if key in par else 0.0

    rows = []
    for name in names:
        attack = champion(name)["character"]["basicAttack"]
        rows.append(ChampionBase(
            base_hp=field(name, "baseHPModifiable"), hp_per_level=field(name, "hpPerLevelModifiable"),
            base_ad=field(name, "baseDamageModifiable"), ad_per_level=field(name, "damagePerLevelModifiable"),
            base_armor=field(name, "baseArmorModifiable"), armor_per_level=field(name, "armorPerLevelModifiable"),
            base_mr=field(name, "baseMR"), mr_per_level=field(name, "mrPerLevel"),
            base_ms=field(name, "baseMoveSpeedModifiable"), attack_range=field(name, "attackRangeModifiable"),
            attack_speed=field(name, "attackSpeedModifiable"),
            attack_speed_ratio=field(name, "attackSpeedRatioModifiable", field(name, "attackSpeedModifiable")),
            attack_speed_per_level=field(name, "attackSpeedPerLevelModifiable"),
            windup_percent=0.3 + float(attack.get("mAttackDelayCastOffsetPercent", 0.0)),
            windup_modifier=float(attack.get("mAttackDelayCastOffsetPercentAttackSpeedRatio", 1.0)),
            hp_regen=field(name, "baseStaticHPRegenModifiable"),
            hp_regen_per_level=field(name, "hpRegenPerLevelModifiable"),
            base_mana=resource(name, AR_BASE), mana_per_level=resource(name, AR_PER_LEVEL),
            mana_regen=resource(name, AR_REGEN), mana_regen_per_level=resource(name, AR_REGEN_PER_LEVEL)))
    return ChampionBase(*(jnp.asarray([getattr(r, f) for r in rows], jnp.float32) for f in ChampionBase._fields))


class ChampionStats(NamedTuple):
    """Composed stats, (C,). ``bonus_*`` = total - base."""
    base_ad: Any
    bonus_ad: Any
    ap: Any
    base_hp: Any
    max_hp: Any
    base_armor: Any
    bonus_armor: Any
    base_mr: Any
    bonus_mr: Any
    attack_speed: Any            # attacks per second after caps
    bonus_attack_speed: Any      # growth + sources, uncapped
    attack_period: Any
    attack_windup: Any
    move_speed: Any              # after slows and soft caps
    basic_ability_haste: Any     # Q/W/E, capped
    ultimate_haste: Any          # R, capped
    item_haste: Any
    summoner_haste: Any
    trinket_haste: Any
    tenacity: Any
    slow_resist: Any
    crit_chance: Any
    crit_damage: Any             # total multiplier
    life_steal: Any
    omnivamp: Any
    heal_shield_power: Any
    lethality: Any
    percent_armor_pen: Any
    magic_pen: Any
    percent_magic_pen: Any
    hp_regen: Any
    max_mana: Any
    mana_regen: Any
    attack_range: Any


def soft_cap_move_speed(raw: Any) -> Any:
    """§9.2 soft caps."""
    return jnp.where(raw > 490.0, 0.5 * raw + 230.0,
                     jnp.where(raw > 415.0, 0.8 * raw + 83.0,
                               jnp.where(raw >= 220.0, raw,
                                         jnp.where(raw >= 0.0, 0.5 * raw + 110.0, 0.01 * raw + 110.0))))


def move_speed(base_ms: Any, flat: Any = 0.0, additive_pct: Any = 0.0, multiplicative_pct: Any = 0.0,
               slow: Any = 0.0, slow_resist: Any = 0.0, bonus_ms_amp: Any = 0.0,
               celerity_flat_pct: Any = 0.0) -> Any:
    """§9.1 raw MS, then soft caps. ``slow``: strongest active slow. Celerity scales every other bonus term by
    ``1 + bonus_ms_amp`` and adds its own ``celerity_flat_pct`` to the additive bucket (RUNES §5.6)."""
    amp = 1.0 + bonus_ms_amp
    raw = (base_ms + flat * amp) * (1.0 + additive_pct * amp + celerity_flat_pct) \
        * (1.0 + multiplicative_pct * amp) * (1.0 - slow * (1.0 - slow_resist))
    return soft_cap_move_speed(raw)


def attack_speed(base_as: Any, ratio: Any, bonus_as: Any, multiplicative: Any = 0.0,
                 cripple: Any = 0.0, cap_lift: Any = 0.0) -> Any:
    """§8.1 ``(AS_base + ratio*bonus) * (1+mult) * (1-cripple)``, clamped."""
    a = (base_as + ratio * bonus_as) * (1.0 + multiplicative) * (1.0 - cripple)
    return jnp.clip(a, AS_MIN, jnp.where(jnp.asarray(cap_lift) > 0.0, AS_UNCAPPED, AS_MAX))


def windup(base_as: Any, attack_speed_now: Any, windup_percent: Any, windup_modifier: Any) -> Any:
    """§8.2 windup interpolated toward ``T*pct`` by the champion modifier."""
    base = windup_percent / base_as
    return base + windup_modifier * (windup_percent / attack_speed_now - base)


def cooldown(base_cd: Any, haste: Any) -> Any:
    """§10.1 ``cd * 100 / (100 + haste)``."""
    return base_cd * 100.0 / (100.0 + jnp.minimum(haste, HASTE_CAP))


def rescale_cooldown(remaining: Any, old_haste: Any, new_haste: Any) -> Any:
    """§10.1: a haste change rescales the remaining cooldown."""
    return remaining * (100.0 + jnp.minimum(old_haste, HASTE_CAP)) / (100.0 + jnp.minimum(new_haste, HASTE_CAP))


def tenacity_total(group_a: Any, group_b: Any = 0.0, group_c: Any = 0.0) -> Any:
    """§10.2: groups (each ``1 - prod(1 - t)``) add, capped at 1."""
    return jnp.minimum(group_a + group_b + group_c, 1.0)


def cc_duration(duration: Any, tenacity: Any, affected: Any = True) -> Any:
    """§10.2: never reduced below 0.3 s; negative tenacity lengthens."""
    t = jnp.where(affected, tenacity, 0.0)
    reduced = jnp.maximum(jnp.minimum(duration, TENACITY_FLOOR_SECONDS), duration * (1.0 - t))
    return jnp.where(t <= 0.0, duration * (1.0 - t), reduced)


def compose(base: ChampionBase, level: Any, bonus: ItemStats, *, adaptive_physical: Any = True,
            slow: Any = 0.0, cripple: Any = 0.0, extra_tenacity_b: Any = 0.0,
            extra_tenacity_c: Any = 0.0) -> ChampionStats:
    """STAT.00-60 for (C,) champions. Growth counts as base except attack speed growth, which is bonus AS."""
    g = level_growth_sum(level, jnp)
    base_hp = base.base_hp + base.hp_per_level * g
    base_ad = base.base_ad + base.ad_per_level * g
    base_armor = base.base_armor + base.armor_per_level * g
    base_mr = base.base_mr + base.mr_per_level * g
    af_ad, af_ap = resolve_adaptive(bonus.adaptive_force, bonus.attack_damage, bonus.ability_power,
                                    adaptive_physical, jnp)
    bonus_ad = bonus.attack_damage + af_ad
    ap = bonus.ability_power + af_ap
    max_hp = jnp.maximum((base_hp + bonus.health) * (1.0 + bonus.percent_health), 1.0)
    armor = (base_armor + bonus.armor) * (1.0 + bonus.percent_armor)
    mr = (base_mr + bonus.magic_resist) * (1.0 + bonus.percent_magic_resist)
    bonus_as = base.attack_speed_per_level / 100.0 * g + bonus.attack_speed
    aspd = attack_speed(base.attack_speed, base.attack_speed_ratio, bonus_as,
                        bonus.multiplicative_attack_speed, cripple, bonus.attack_speed_cap_lift)
    slow_resist = jnp.minimum(bonus.slow_resist, 1.0)
    ms = move_speed(base.base_ms, bonus.move_speed, bonus.percent_move_speed, 0.0, slow, slow_resist,
                    bonus.bonus_ms_amp)
    ah = bonus.ability_haste
    basic_ah = jnp.minimum(ah + bonus.basic_ability_haste, HASTE_CAP)
    ult_ah = jnp.minimum(ah + bonus.ultimate_haste, HASTE_CAP)
    hp_regen = (base.hp_regen + base.hp_regen_per_level * g) * (1.0 + bonus.percent_base_health_regen) \
        + bonus.health_regen
    mana_regen = (base.mana_regen + base.mana_regen_per_level * g) * (1.0 + bonus.percent_base_mana_regen) \
        + bonus.mana_regen
    shape = jnp.broadcast_shapes(jnp.shape(base.base_hp), jnp.shape(level))
    return ChampionStats(*(jnp.broadcast_to(jnp.asarray(v, jnp.float32), shape) for v in ChampionStats(
        base_ad=base_ad, bonus_ad=bonus_ad, ap=ap, base_hp=base_hp, max_hp=max_hp,
        base_armor=base_armor, bonus_armor=armor - base_armor, base_mr=base_mr, bonus_mr=mr - base_mr,
        attack_speed=aspd, bonus_attack_speed=bonus_as, attack_period=1.0 / aspd,
        attack_windup=windup(base.attack_speed, aspd, base.windup_percent, base.windup_modifier),
        move_speed=ms, basic_ability_haste=basic_ah, ultimate_haste=ult_ah,
        item_haste=bonus.item_haste, summoner_haste=bonus.summoner_haste, trinket_haste=bonus.trinket_haste,
        tenacity=tenacity_total(bonus.tenacity, extra_tenacity_b, extra_tenacity_c), slow_resist=slow_resist,
        crit_chance=jnp.clip(bonus.crit_chance, 0.0, 1.0), crit_damage=2.0 + bonus.crit_damage,
        life_steal=bonus.life_steal, omnivamp=bonus.omnivamp, heal_shield_power=bonus.heal_shield_power,
        lethality=bonus.lethality, percent_armor_pen=bonus.percent_armor_pen, magic_pen=bonus.magic_pen,
        percent_magic_pen=bonus.percent_magic_pen, hp_regen=hp_regen,
        max_mana=base.base_mana + base.mana_per_level * g + bonus.mana, mana_regen=mana_regen,
        attack_range=base.attack_range)))


def sync_max_health(hp: Any, old_max: Any, new_max: Any, heal_on_gain: Any = None) -> tuple[Any, Any]:
    """STAT.70 (§11): gains raise HP by ``heal_on_gain`` (default the delta; not a heal), losses only clamp."""
    delta = new_max - old_max
    gain = jnp.maximum(delta, 0.0) if heal_on_gain is None else jnp.clip(heal_on_gain, 0.0, jnp.maximum(delta, 0.0))
    return jnp.clip(hp + gain, 0.0, new_max), new_max
