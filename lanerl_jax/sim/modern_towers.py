"""Patch 26.19 outer-lane turret rules, independent of legacy C# parity.

Times are absolute game seconds. Scalar state is vmappable. Call advance before
hits; resolve projectiles on impact, and discard them if their turret has died.
Geometry/visibility and identifying an aggressive champion are caller inputs.
Sources/remaining uncertainties: data/modern/26.19/towers.json.
"""
from typing import NamedTuple
import jax.numpy as jnp

MAX_HP = 9000.0
ATTACK_PERIOD = 1.2
ATTACK_RANGE = 750.0  # edge to edge
GAMEPLAY_RADIUS = 88.4
MISSILE_SPEED = 1200.0
# Ordered priority categories; champion kit pets are outside this world module.
PET, CANNON_SUPER, MIST_WALKER, MELEE, CASTER, LOW_PRIORITY_PET, CHAMPION = range(7)


class TurretState(NamedTuple):
    hp: jnp.ndarray
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


def init_outer_turret(targetable_since=10.0):
    """10s targetability gives wiki's 1:40 initial crystal appearance."""
    return TurretState(jnp.float32(MAX_HP), jnp.int32(0),
                       jnp.zeros(4, jnp.float32), jnp.float32(0),
                       jnp.float32(targetable_since), jnp.bool_(False),
                       jnp.int32(0), jnp.float32(0))


def advance(state, now, enemy_minion_near, enemy_unit_near):
    """Caller supplies independently checked protection/growth proximity.

    A qualifying minion (or Herald) removes backdoor immediately and refreshes
    its 3s grace. Enemies suppress only initial crystal activation, not an
    already active crystal, and suppression never pauses its damage clock.
    """
    alive = state.hp > 0
    return state._replace(
        backdoor_until=jnp.where(enemy_minion_near & alive, now + 3., state.backdoor_until),
        growth_active=alive & (state.growth_active | (
            (now >= state.growth_since + 90.) & ~jnp.asarray(enemy_unit_near))),
        warm_stacks=jnp.where(now >= state.warm_until, 0, state.warm_stacks))


def decay_steps(now):
    # First step at 11:00; fourth and last at 14:00 (Riot 26.1 wording).
    return jnp.clip(jnp.floor(jnp.asarray(now) / 60.) - 10., 0., 4.)


def resistance(state, now, nearby_enemy_champions):
    count = jnp.clip(jnp.asarray(nearby_enemy_champions), 1, 5)
    per_stack = 30. + 5. * (count - 1)
    return 60. - 15. * decay_steps(now) + per_stack * jnp.sum(state.bulwark_until > now)


def plate_value(now):
    return 120. - 10. * decay_steps(now)


def outer_attack_damage(now):
    return 182. + 12. * jnp.clip(jnp.floor((jnp.asarray(now) - 30.) / 60.) + 1., 0., 14.)


def resistance_multiplier(resist):
    resist = jnp.asarray(resist)
    return jnp.where(resist >= 0, 100. / (100. + jnp.maximum(resist, 0)),
                     2. - 100. / (100. - jnp.minimum(resist, 0)))


def champion_structure_attack(base_ad, bonus_ad, ap):
    """Return (raw amount, magic flag). Both bonus AD and AP contribute."""
    return base_ad + bonus_ad + .6 * ap, .6 * ap > bonus_ad


def overgrowth_damage(state, now, minimum_fraction, maximum_fraction):
    """Fractions are explicit: published level endpoints do not define curve.

    Caller must provide patch-reviewed level scaling; silently interpolating
    the level-1 and level-18 endpoints would invent an undocumented formula.
    Time interpolation, unlike level interpolation, is documented as linear.
    """
    ramp = jnp.clip((now - state.growth_since - 150.) / 240., 0., 1.)
    return MAX_HP * (minimum_fraction + ramp * (maximum_fraction - minimum_fraction))


def apply_turret_damage(state, now, physical, magic, true, *,
                        nearby_enemy_champions=1, melee_champion=False,
                        champion_attack=False, armor_pen_flat=0., armor_pen_percent=0.,
                        magic_pen_flat=0., magic_pen_percent=0.,
                        growth_min_fraction=jnp.nan, growth_max_fraction=jnp.nan):
    """Apply a damage packet and report plate/gold events for reward sharing.

    Overgrowth coefficients must be supplied for champion procs; omitted values
    deliberately yield NaN on a proc rather than silently disabling crystals.
    Overgrowth is a separate turret-owned true packet, not melee-amplified.
    For minion hits caller scales AD by .60 (.84 cannon) before this function.
    Reward eligibility/sharing and first-turret global state belong to world.
    Bulwark acquired by this packet mitigates subsequent packets only.
    """
    resist = resistance(state, now, nearby_enemy_champions)
    armor = jnp.maximum(0., resist * (1. - jnp.clip(armor_pen_percent, 0., 1.)) - armor_pen_flat)
    mr = jnp.maximum(0., resist * (1. - jnp.clip(magic_pen_percent, 0., 1.)) - magic_pen_flat)
    normal = (jnp.maximum(physical, 0.) * resistance_multiplier(armor)
              + jnp.maximum(magic, 0.) * resistance_multiplier(mr) + jnp.maximum(true, 0.))
    backdoor = now >= state.backdoor_until
    proc = state.growth_active & champion_attack & ~backdoor & (state.hp > 0)
    crystal = jnp.where(proc, overgrowth_damage(state, now, growth_min_fraction, growth_max_fraction), 0.)
    damage = (normal * jnp.where(melee_champion, 1.2, 1.) + crystal) * jnp.where(backdoor, .2, 1.)
    hp = jnp.maximum(state.hp - damage, 0.)
    thresholds = MAX_HP * jnp.asarray([.9, .75, .55, .3, 0.])
    plates = jnp.maximum(state.plates, jnp.sum(hp <= thresholds))
    gained = plates - state.plates
    slots = jnp.arange(4)
    expiry = jnp.where((slots >= state.plates) & (slots < plates), now + 20., state.bulwark_until)
    new = state._replace(hp=hp, plates=plates, bulwark_until=expiry,
                         growth_since=jnp.where(proc, now, state.growth_since),
                         growth_active=state.growth_active & ~proc & (hp > 0))
    return HitResult(new, state.hp - hp, gained, gained * plate_value(now),
                     (state.hp > 0) & (hp <= 0), crystal)


def champion_shot_impact(state, now, target_armor):
    """Post-armor champion damage and ramp state; ramp survives target changes."""
    stacks = jnp.where(now < state.warm_until, state.warm_stacks, 0)
    armor = jnp.where(target_armor > 0, target_armor * .7, target_armor)
    damage = outer_attack_damage(now) * (1. + .5 * stacks) * resistance_multiplier(armor)
    alive = state.hp > 0
    return state._replace(warm_stacks=jnp.where(alive, jnp.minimum(stacks + 1, 3), stacks),
                          warm_until=jnp.where(alive, now + 5., state.warm_until)), jnp.where(alive, damage, 0.)


def minion_shot_damage(max_hp, kind, armor=0.):
    """Outer turret percent-max-HP physical damage, with 30% armor penetration."""
    # Input kind 0 melee, 1 caster, 2 cannon, 3 super (not target priority).
    return (max_hp * jnp.take(jnp.asarray([.45, .7, .14, .05]), kind)
            * resistance_multiplier(jnp.where(armor > 0, armor * .7, armor)))


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
