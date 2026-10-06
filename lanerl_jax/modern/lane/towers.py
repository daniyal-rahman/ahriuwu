"""26.19 turret and building rules on one vmappable per-structure state (docs/modern/TOWERS.md, data/26.19/towers.json).

Tiers 0..3 are turrets; 4/5 reuse the state for the inhibitor and Nexus (HP, regen, respawn; no plates, Overgrowth,
Bulwark or backdoor, §1.4). Times are absolute game seconds. The per-tick driver is ``lane.ai``.
"""
from typing import NamedTuple

import jax.numpy as jnp

OUTER, INNER, INHIBITOR, NEXUS = range(4)
INHIBITOR_BUILDING, NEXUS_BUILDING = 4, 5
TIER_MAX_HP = (9000., 5000., 4750., 3500., 4000., 5500.)
ATTACK_SPEED = 0.833                 # client SR_* attackSpeed (§1.2)
WINDUP_S = 0.139 / ATTACK_SPEED      # INFERRED M
ATTACK_RANGE = 750.0                 # edge to edge
GAMEPLAY_RADIUS = 88.4
MISSILE_SPEED = 1200.0
ARMOR_PENETRATION = 0.3              # item 1500
BACKDOOR_RADIUS = 1000.0             # unpublished default (§4.4)
BULWARK_RADIUS = 850.0
PROTECTION_RADIUS = 1400.0           # champion-protection victim radius (§3.3)
BUILDING_ARMOR, BUILDING_MR = 20.0, 0.0   # wiki
NEXUS_TURRET_RESPAWN_S, INHIBITOR_RESPAWN_S = 180.0, 300.0
# Crystalline Overgrowth, item 1524 mDataValues (§6.2).
OG_PROC_COOLDOWN = 90.0
OG_START_TIME = 60.0
OG_END_TIME = 300.0
OG_END_MULT_MIN = 1.65
OG_END_MULT_MAX = 2.15
OG_DAMAGE_BASE_PCT = 1.6
OG_DAMAGE_PER_LEVEL_PCT = 0.4
# Target priority classes, lowest first (§3).
PET, CANNON_SUPER, MIST_WALKER, MELEE, CASTER, LOW_PRIORITY_PET, CHAMPION = range(7)
MINION_SHOT_FRACTION = (.45, .70, .14, .07)   # of minion max HP: melee, caster, siege (outer), super (items 1508-11)


class TurretState(NamedTuple):
    hp: jnp.ndarray
    max_hp: jnp.ndarray
    tier: jnp.ndarray
    respawn_at: jnp.ndarray
    plates: jnp.ndarray          # claimed, including final destruction
    bulwark_until: jnp.ndarray   # four independently expiring stacks
    backdoor_until: jnp.ndarray  # backdoor protection off before this time
    growth_since: jnp.ndarray    # time targetable, or last consumption
    growth_active: jnp.ndarray
    warm_stacks: jnp.ndarray
    warm_until: jnp.ndarray


def init_turret(tier=OUTER, targetable_since=10.0):
    """``targetable_since=inf``: locked until ``unlock``."""
    hp = jnp.take(jnp.asarray(TIER_MAX_HP, jnp.float32), tier)
    return TurretState(hp, hp, jnp.int32(tier), jnp.float32(jnp.inf), jnp.int32(0),
                       jnp.zeros(4, jnp.float32), jnp.float32(0),
                       jnp.float32(targetable_since), jnp.bool_(False),
                       jnp.int32(0), jnp.float32(0))


def unlock(state, now):
    """Start the Overgrowth clock when the preceding structure falls."""
    return state._replace(growth_since=jnp.where(jnp.isinf(state.growth_since), now, state.growth_since))


def advance(state, now, enemy_minion_near, enemy_unit_near):
    """An enemy lane minion near refreshes the 3 s backdoor grace. Enemies near only suppress a crystal's initial
    appearance, never an active crystal or its damage clock."""
    alive = state.hp > 0
    return state._replace(
        backdoor_until=jnp.where(enemy_minion_near & alive, now + 3., state.backdoor_until),
        growth_active=alive & (state.tier < NEXUS) & (state.growth_active | (
            (now >= state.growth_since + OG_PROC_COOLDOWN) & ~jnp.asarray(enemy_unit_near))),
        warm_stacks=jnp.where(now >= state.warm_until, 0, state.warm_stacks))


def decay_steps(now):
    """Outer-turret decay: first step at 11:00, last at 14:00."""
    return jnp.clip(jnp.floor(jnp.asarray(now) / 60.) - 10., 0., 4.)


def resistance(state, now, nearby_enemy_champions):
    """Armor = MR: 60 (outer -15 per decay step) + Bulwark ``30 + 5 (n - 1)`` per live stack (§5.2-5.3)."""
    count = jnp.clip(jnp.asarray(nearby_enemy_champions), 1, 5)
    per_stack = 30. + 5. * (count - 1)
    return 60. - jnp.where(state.tier == OUTER, 15. * decay_steps(now), 0.) + per_stack * jnp.sum(state.bulwark_until > now)


def plate_value(now, tier=OUTER):
    return jnp.where(tier >= NEXUS, 0., 120. - jnp.where(tier == OUTER, 10. * decay_steps(now), 0.))


def outer_attack_damage(now):
    return 182. + 12. * jnp.clip(jnp.floor((jnp.asarray(now) - 30.) / 60.) + 1., 0., 14.)


def attack_damage(tier, now):
    """§1.1 AD growth by tier and time."""
    inner_growth = 16. * jnp.clip(jnp.floor((jnp.asarray(now) - 180.) / 60.) + 1., 0., 15.)
    return jnp.where(tier == OUTER, outer_attack_damage(now),
                     jnp.where(tier == NEXUS, 165., 187.) + inner_growth)


def regenerate_and_respawn(state, now, dt):
    """Inhibitor/Nexus turrets regen 3/6 HP/s within HP segments, inhibitor 15/s and Nexus 20/s uncapped. Nexus
    turrets return 180 s after death at 40 %, inhibitors 300 s after at full HP; lane turrets never return."""
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


def in_attack_range(center_distance, target_radius):
    """Edge-to-edge: ``d <= 750 + 88.4 + r_target`` (§1.3)."""
    return center_distance <= ATTACK_RANGE + GAMEPLAY_RADIUS + target_radius


def champion_structure_attack(base_ad, bonus_ad, ap):
    """``(raw, is_magic)`` of a champion attack on a structure: bonus AD and 60 % AP both count."""
    return base_ad + bonus_ad + .6 * ap, .6 * ap > bonus_ad


def overgrowth_level_fractions(average_team_level):
    """Item-1524 ``(min, max)`` fractions of max HP (§6.2): ``min = (1.6 + 0.4 L)%`` (level unclamped),
    ``max = min * (1.65 + 0.5 clip((L-1)/17, 0, 1))`` (INFERRED M lerp)."""
    level = jnp.asarray(average_team_level)
    base = (OG_DAMAGE_BASE_PCT + OG_DAMAGE_PER_LEVEL_PCT * level) / 100.
    level_fraction = jnp.clip((level - 1.) / 17., 0., 1.)
    end_mult = OG_END_MULT_MIN + (OG_END_MULT_MAX - OG_END_MULT_MIN) * level_fraction
    return base, base * end_mult


def overgrowth_damage(state, now, minimum_fraction, maximum_fraction):
    """Crystal true damage: 60 s at the minimum after appearing (``growth_since + 90``), then a 240 s ramp."""
    hold = OG_PROC_COOLDOWN + OG_START_TIME
    ramp = jnp.clip((now - state.growth_since - hold) / (OG_END_TIME - OG_START_TIME), 0., 1.)
    return state.max_hp * (minimum_fraction + ramp * (maximum_fraction - minimum_fraction))


def respawn_delay(tier):
    """Seconds until a destroyed structure returns (inf: never)."""
    return jnp.where(tier == NEXUS, NEXUS_TURRET_RESPAWN_S,
                     jnp.where(tier == INHIBITOR_BUILDING, INHIBITOR_RESPAWN_S, jnp.inf))


def warming_multiplier(stacks):
    """Warming Up vs champions: 1.0, 1.5, 2.0, 2.5 (item 1500)."""
    return 1. + .5 * jnp.clip(stacks, 0, 3)


def minion_shot_fraction(kind, tier=OUTER):
    """Turret shot as a fraction of the minion's max HP; siege 14/11/8 % by outer/inner/deeper tier."""
    kind = jnp.asarray(kind)
    tier = jnp.asarray(tier)
    siege = jnp.where(tier == OUTER, .14, jnp.where(tier == INNER, .11, .08))
    return jnp.where(kind == 2, siege, jnp.take(jnp.asarray(MINION_SHOT_FRACTION), jnp.clip(kind, 0, 3)))


def select_target(current, eligible, distance, priority, aggressive_champion):
    """Stable lock, preempted by champion protection (nearest aggressor); else nearest of the best class; -1 none.

    ``aggressive_champion``: champion-origin damage attempt on an ally within 1400 of the turret. Ties: slot order."""
    indices = jnp.arange(eligible.shape[0])
    nearest_aggressor = jnp.argmin(jnp.where(eligible & aggressive_champion, distance, jnp.inf))
    best_priority = jnp.min(jnp.where(eligible, priority, 100))
    nearest = jnp.argmin(jnp.where(eligible & (priority == best_priority), distance, jnp.inf))
    valid_current = (current >= 0) & jnp.any(eligible & (indices == current))
    chosen = jnp.where(valid_current, current, nearest)
    chosen = jnp.where(jnp.any(eligible & aggressive_champion), nearest_aggressor, chosen)
    return jnp.where(jnp.any(eligible), chosen, -1)
