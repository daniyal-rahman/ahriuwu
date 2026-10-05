"""Patch 26.19 lane and Nexus turret rules, independent of legacy C# parity.

Times are absolute game seconds. Scalar state is vmappable. Call advance before
hits; resolve projectiles on impact, and discard them if their turret has died.
Geometry/visibility and identifying an aggressive champion are caller inputs.
Sources/remaining uncertainties: docs/modern/TOWERS.md (authoritative),
modern/data/26.19/towers.json. The per-tick driver is ``lane.ai``.

Tiers 0..3 are turrets. Tiers 4/5 reuse the same state for the non-attacking
inhibitor and Nexus buildings (HP, regen, respawn; no plates, Overgrowth,
Bulwark or Reinforced Armor; TOWERS §1.4).
"""
from typing import NamedTuple

import jax.numpy as jnp

from ..core.stats import armor_after_modifiers, magic_resist_after_modifiers, mitigation_multiplier

OUTER, INNER, INHIBITOR, NEXUS = range(4)
INHIBITOR_BUILDING, NEXUS_BUILDING = 4, 5
TIER_MAX_HP = (9000., 5000., 4750., 3500., 4000., 5500.)
MAX_HP = 9000.0
ATTACK_SPEED = 0.833                 # client SR_* attackSpeed (TOWERS §1.2, D8)
ATTACK_PERIOD = 1.0 / ATTACK_SPEED   # 1.20048 s (was 1.2)
WINDUP_S = 0.139 / ATTACK_SPEED      # 0.1669 s [INFERRED M]
ATTACK_RANGE = 750.0  # edge to edge
GAMEPLAY_RADIUS = 88.4
MISSILE_SPEED = 1200.0
ARMOR_PENETRATION = 0.3              # item 1500, turret's own attacks
BACKDOOR_RADIUS = 1000.0             # unpublished; default (TOWERS §4.4, U2)
BULWARK_RADIUS = 850.0
PROTECTION_RADIUS = 1400.0           # champion-protection victim radius (§3.3)
BUILDING_ARMOR, BUILDING_MR = 20.0, 0.0   # inhibitor/Nexus (wiki, TOWERS C5/U9)
NEXUS_TURRET_RESPAWN_S, INHIBITOR_RESPAWN_S = 180.0, 300.0
# Crystalline Overgrowth, item 1524 mDataValues (SR; TOWERS §6.2).
OG_PROC_COOLDOWN = 90.0
OG_START_TIME = 60.0
OG_END_TIME = 300.0
OG_END_MULT_MIN = 1.65
OG_END_MULT_MAX = 2.15
OG_DAMAGE_BASE_PCT = 1.6
OG_DAMAGE_PER_LEVEL_PCT = 0.4
# Ordered priority categories; champion kit pets are outside this world module.
PET, CANNON_SUPER, MIST_WALKER, MELEE, CASTER, LOW_PRIORITY_PET, CHAMPION = range(7)


class TurretState(NamedTuple):
    hp: jnp.ndarray
    max_hp: jnp.ndarray
    tier: jnp.ndarray
    respawn_at: jnp.ndarray
    plates: jnp.ndarray  # number already claimed, including final destruction
    bulwark_until: jnp.ndarray  # four independently expiring stacks
    backdoor_until: jnp.ndarray  # backdoor disabled strictly before this time
    growth_since: jnp.ndarray  # time targetable, or last consumption
    growth_active: jnp.ndarray
    warm_stacks: jnp.ndarray
    warm_until: jnp.ndarray


class HitResult(NamedTuple):
    state: TurretState
    damage: jnp.ndarray
    plates: jnp.ndarray
    local_gold: jnp.ndarray
    destroyed: jnp.ndarray
    overgrowth_damage: jnp.ndarray


def init_turret(tier=OUTER, targetable_since=10.0):
    """Use infinite targetable_since for a locked turret, then unlock explicitly."""
    hp = jnp.take(jnp.asarray(TIER_MAX_HP, jnp.float32), tier)
    return TurretState(hp, hp, jnp.int32(tier), jnp.float32(jnp.inf), jnp.int32(0),
                       jnp.zeros(4, jnp.float32), jnp.float32(0),
                       jnp.float32(targetable_since), jnp.bool_(False),
                       jnp.int32(0), jnp.float32(0))


def init_outer_turret(targetable_since=10.0):
    return init_turret(OUTER, targetable_since)


def unlock(state, now):
    """Begin lane overgrowth startup when the preceding structure falls."""
    return state._replace(growth_since=jnp.where(jnp.isinf(state.growth_since), now, state.growth_since))


def advance(state, now, enemy_minion_near, enemy_unit_near):
    """Caller supplies independently checked protection/growth proximity.

    A qualifying minion (or Herald) removes backdoor immediately and refreshes
    its 3s grace. Enemies suppress only initial crystal activation, not an
    already active crystal, and suppression never pauses its damage clock.
    """
    alive = state.hp > 0
    return state._replace(
        backdoor_until=jnp.where(enemy_minion_near & alive, now + 3., state.backdoor_until),
        growth_active=alive & (state.tier < NEXUS) & (state.growth_active | (
            (now >= state.growth_since + OG_PROC_COOLDOWN) & ~jnp.asarray(enemy_unit_near))),
        warm_stacks=jnp.where(now >= state.warm_until, 0, state.warm_stacks))


def decay_steps(now):
    # First step at 11:00; fourth and last at 14:00 (Riot 26.1 wording).
    return jnp.clip(jnp.floor(jnp.asarray(now) / 60.) - 10., 0., 4.)


def resistance(state, now, nearby_enemy_champions):
    count = jnp.clip(jnp.asarray(nearby_enemy_champions), 1, 5)
    per_stack = 30. + 5. * (count - 1)
    return 60. - jnp.where(state.tier == OUTER, 15. * decay_steps(now), 0.) + per_stack * jnp.sum(state.bulwark_until > now)


def plate_value(now, tier=OUTER):
    return jnp.where(tier >= NEXUS, 0., 120. - jnp.where(tier == OUTER, 10. * decay_steps(now), 0.))


def outer_attack_damage(now):
    return 182. + 12. * jnp.clip(jnp.floor((jnp.asarray(now) - 30.) / 60.) + 1., 0., 14.)


def attack_damage(tier, now):
    inner_growth = 16. * jnp.clip(jnp.floor((jnp.asarray(now) - 180.) / 60.) + 1., 0., 15.)
    return jnp.where(tier == OUTER, outer_attack_damage(now),
                     jnp.where(tier == NEXUS, 165., 187.) + inner_growth)


def regenerate_and_respawn(state, now, dt):
    """Base turrets heal within segments; Nexus returns after180s at40% HP.

    Inhibitor (tier 4) regenerates 15 HP/s and Nexus (tier 5) 20 HP/s,
    uncapped; a dead inhibitor returns at full HP 300 s after death.
    World still enforces inhibitor-based vulnerability separately. Destroyed
    lane turrets never respawn. Regeneration cannot restore lost plate rewards.
    """
    frac = state.hp / state.max_hp
    low = jnp.where(state.tier == NEXUS, .4, .3)
    high = jnp.where(state.tier == NEXUS, .7, .75)
    cap = jnp.where(frac <= low, low, jnp.where(frac <= high, high, 1.)) * state.max_hp
    cap = jnp.where(state.tier >= INHIBITOR_BUILDING, state.max_hp, cap)
    rate = jnp.take(jnp.asarray([0., 0., 3., 6., 15., 20.], jnp.float32), jnp.clip(state.tier, 0, 5))
    hp = jnp.where(state.hp > 0, jnp.minimum(cap, state.hp + rate * jnp.maximum(dt, 0.)), 0.)
    respawn = ((state.tier == NEXUS) | (state.tier == INHIBITOR_BUILDING)) & (state.hp <= 0) & (now >= state.respawn_at)
    back = jnp.where(state.tier == NEXUS, .4, 1.)
    return state._replace(hp=jnp.where(respawn, state.max_hp * back, hp),
                          respawn_at=jnp.where(respawn, jnp.inf, state.respawn_at),
                          warm_stacks=jnp.where(respawn, 0, state.warm_stacks),
                          warm_until=jnp.where(respawn, 0., state.warm_until))


def resistance_multiplier(resist):
    """Shared ``core.stats.mitigation_multiplier`` (negative branch kept)."""
    return mitigation_multiplier(jnp.asarray(resist), jnp)


def in_attack_range(center_distance, target_radius):
    """Edge-to-edge turret range: ``d <= 750 + 88.4 + r_target`` (§1.3, U5)."""
    return center_distance <= ATTACK_RANGE + GAMEPLAY_RADIUS + target_radius


def champion_structure_attack(base_ad, bonus_ad, ap):
    """Return (raw amount, magic flag). Both bonus AD and AP contribute."""
    return base_ad + bonus_ad + .6 * ap, .6 * ap > bonus_ad


def overgrowth_level_fractions(average_team_level):
    """Client item-1524 curve: ``(min, max)`` fractions of turret max HP.

    ``min = (1.6 + 0.4 L)%`` (exact, a per-level data value) and
    ``max = min * (1.65 + 0.5 * clip((L-1)/17, 0, 1))`` (Riot lerp, INFERRED M;
    TOWERS §6.2, D1). Replaces the former linear interpolation of the max
    endpoint (L9 outer: 882.3 instead of 957.7). ``L`` is not clamped, so the
    level-19/20 top-quest cap extends ``min`` linearly while the multiplier
    lerp clamps at 18 (TOWERS D2/U6, README X-1).
    """
    level = jnp.asarray(average_team_level)
    base = (OG_DAMAGE_BASE_PCT + OG_DAMAGE_PER_LEVEL_PCT * level) / 100.
    level_fraction = jnp.clip((level - 1.) / 17., 0., 1.)
    end_mult = OG_END_MULT_MIN + (OG_END_MULT_MAX - OG_END_MULT_MIN) * level_fraction
    return base, base * end_mult


def overgrowth_damage(state, now, minimum_fraction, maximum_fraction):
    """Crystal true damage from explicit level fractions (``overgrowth_level_fractions``).

    Growth is measured from appearance (``growth_since + 90``, fast-forwarded
    through suppression): 60 s hold at the minimum, then a 240 s linear ramp.
    """
    hold = OG_PROC_COOLDOWN + OG_START_TIME
    ramp = jnp.clip((now - state.growth_since - hold) / (OG_END_TIME - OG_START_TIME), 0., 1.)
    return state.max_hp * (minimum_fraction + ramp * (maximum_fraction - minimum_fraction))


def apply_turret_damage(state, now, physical, magic, true, *,
                        nearby_enemy_champions=1, melee_champion=False,
                        champion_attack=False, armor_pen_flat=0., armor_pen_percent=0.,
                        magic_pen_flat=0., magic_pen_percent=0.,
                        growth_min_fraction=None, growth_max_fraction=None, average_team_level=1.):
    """Apply a damage packet and report plate/gold events for reward sharing.

    Default Overgrowth uses the client item-1524 level curve; pass both
    fractions to override it.
    Overgrowth is a separate turret-owned true packet, not melee-amplified.
    For minion hits caller scales AD by .60 (.84 cannon) before this function.
    Reward eligibility/sharing and first-turret global state belong to world.
    Bulwark acquired by this packet mitigates subsequent packets only.
    """
    estimated_min, estimated_max = overgrowth_level_fractions(average_team_level)
    if growth_min_fraction is None:
        growth_min_fraction = estimated_min
    if growth_max_fraction is None:
        growth_max_fraction = estimated_max
    resist = resistance(state, now, nearby_enemy_champions)
    # Shared resist order (DAMAGE_AND_STATS §4.2): negative resist survives
    # (decayed outer turret + reduction), flat pen cannot push positive below 0.
    armor = armor_after_modifiers(resist, percent_penetration=jnp.clip(armor_pen_percent, 0., 1.),
                                  flat_penetration=armor_pen_flat, xp=jnp)
    mr = magic_resist_after_modifiers(resist, percent_penetration=jnp.clip(magic_pen_percent, 0., 1.),
                                      flat_penetration=magic_pen_flat, xp=jnp)
    normal = (jnp.maximum(physical, 0.) * resistance_multiplier(armor)
              + jnp.maximum(magic, 0.) * resistance_multiplier(mr) + jnp.maximum(true, 0.))
    # Reinforced Armor exists on all four turret tiers, never on buildings.
    backdoor = (now >= state.backdoor_until) & (state.tier < INHIBITOR_BUILDING)
    proc = state.growth_active & champion_attack & ~backdoor & (state.hp > 0)
    crystal = jnp.where(proc, overgrowth_damage(state, now, growth_min_fraction, growth_max_fraction), 0.)
    damage = (normal * jnp.where(melee_champion, 1.2, 1.) + crystal) * jnp.where(backdoor, .2, 1.)
    hp = jnp.maximum(state.hp - damage, 0.)
    thresholds = state.max_hp * jnp.asarray([.9, .75, .55, .3, 0.])
    plates = jnp.where(state.tier >= NEXUS, 0, jnp.maximum(state.plates, jnp.sum(hp <= thresholds, dtype=jnp.int32)))
    gained = plates - state.plates
    slots = jnp.arange(4)
    expiry = jnp.where((slots >= state.plates) & (slots < plates), now + 20., state.bulwark_until)
    new = state._replace(hp=hp, plates=plates, bulwark_until=expiry,
                         respawn_at=jnp.where((state.hp > 0) & (hp <= 0), now + respawn_delay(state.tier), state.respawn_at),
                         growth_since=jnp.where(proc, now, state.growth_since),
                         growth_active=state.growth_active & ~proc & (hp > 0))
    return HitResult(new, state.hp - hp, gained, gained * plate_value(now, state.tier),
                     (state.hp > 0) & (hp <= 0), crystal)


def respawn_delay(tier):
    """Seconds until a destroyed structure returns (inf: never)."""
    return jnp.where(tier == NEXUS, NEXUS_TURRET_RESPAWN_S,
                     jnp.where(tier == INHIBITOR_BUILDING, INHIBITOR_RESPAWN_S, jnp.inf))


def warming_multiplier(stacks):
    """Warming Up / heat ramp vs champions: 1.0, 1.5, 2.0, 2.5 (item 1500)."""
    return 1. + .5 * jnp.clip(stacks, 0, 3)


def champion_shot_impact(state, now, target_armor):
    """Post-armor champion damage and ramp state; ramp survives target changes."""
    stacks = jnp.where(now < state.warm_until, state.warm_stacks, 0)
    armor = jnp.where(target_armor > 0, target_armor * .7, target_armor)
    damage = attack_damage(state.tier, now) * (1. + .5 * stacks) * resistance_multiplier(armor)
    alive = state.hp > 0
    return state._replace(warm_stacks=jnp.where(alive, jnp.minimum(stacks + 1, 3), stacks),
                          warm_until=jnp.where(alive, now + 5., state.warm_until)), jnp.where(alive, damage, 0.)


MINION_SHOT_FRACTION = (.45, .70, .14, .07)   # melee, caster, siege (outer), super


def minion_shot_fraction(kind, tier=OUTER):
    """Turret shot as a fraction of the minion's max HP (items 1508-1511).

    Siege 14/11/8 % by outer/inner/inhibitor-or-Nexus tier; super 7 %
    (client 1511 tooltip; the former 5 % had no source, TOWERS D3).
    """
    kind = jnp.asarray(kind)
    tier = jnp.asarray(tier)
    siege = jnp.where(tier == OUTER, .14, jnp.where(tier == INNER, .11, .08))
    return jnp.where(kind == 2, siege, jnp.take(jnp.asarray(MINION_SHOT_FRACTION), jnp.clip(kind, 0, 3)))


def minion_shot_damage(max_hp, kind, armor=0., tier=OUTER, mitigated=False):
    """Turret shot damage on a lane minion.

    Default (README X-3, MINIONS U-14): the percent-max-HP amount is the HP
    the minion loses, unaffected by its armor. ``mitigated=True`` gives the
    TOWERS §4.2 reading (physical, armor after the turret's 30 % pen).
    """
    raw = max_hp * minion_shot_fraction(kind, tier)
    eff = jnp.where(armor > 0, armor * (1. - ARMOR_PENETRATION), armor)
    return jnp.where(mitigated, raw * resistance_multiplier(eff), raw)


def select_target(current, eligible, distance, priority, aggressive_champion):
    """Stable lock; champion protection preempts it. -1 means no valid target.

    Arrays have nonzero fixed capacity. Caller masks enemies/visibility/range/
    targetability. aggressive_champion means champion-origin damage attempt
    against an ally within 1400 of turret, including blocked/zero damage.
    Ties use slot order as a deterministic simulation convention.
    """
    indices = jnp.arange(eligible.shape[0])
    nearest_aggressor = jnp.argmin(jnp.where(eligible & aggressive_champion, distance, jnp.inf))
    best_priority = jnp.min(jnp.where(eligible, priority, 100))
    nearest = jnp.argmin(jnp.where(eligible & (priority == best_priority), distance, jnp.inf))
    valid_current = (current >= 0) & jnp.any(eligible & (indices == current))
    chosen = jnp.where(valid_current, current, nearest)
    chosen = jnp.where(jnp.any(eligible & aggressive_champion), nearest_aggressor, chosen)
    return jnp.where(jnp.any(eligible), chosen, -1)


def local_reward_eligible(alive, distance, last_assist, now, direct_killer=False):
    return (alive & (distance <= 1200.)) | (now - last_assist <= 10.) | direct_killer
