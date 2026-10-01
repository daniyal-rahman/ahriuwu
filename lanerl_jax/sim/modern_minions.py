"""Pure JAX rules for 26.19 Summoner's Rift lane minions.

Scope is the top lane wave schedule, composition, profile facts, and target
selection/Call-for-Help rules.  Movement/pathing, combat resolution, inhibitor
respawn suppression, and the server's fuzzy target reevaluation timer belong
to the world step.  All clock values are seconds and map coordinates are XZ.

The ``*_PROFILE`` values are current Wiki base/profile values, not a claim
that a minion keeps those values after its 90-second upgrades. The per-upgrade
stat curve is not exposed here: the Wiki gives current endpoints, but does not
document a complete per-upgrade curve for every stat. In particular, do not
linearly interpolate those endpoints.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

__all__ = [
    "MinionType", "TargetKind", "TargetPriority", "SpawnEvent",
    "MinionProfile", "MELEE_PROFILE",
    "CASTER_PROFILE", "CANNON_PROFILE", "SUPER_PROFILE",
    "WAVE_FIRST_S", "WAVE_UNIT_GAP_S", "wave_spawn_time",
    "wave_index_at", "wave_interval_s", "cannon_wave",
    "wave_composition", "spawn_event", "upgrade_index_at",
    "select_target", "target_priority", "shared_xp_fraction",
    "minion_pushing_modifiers", "gold_bounty",
    "lane_minion_current_hp_bonus", "call_for_help_applies",
    "call_for_help_trigger",
]


class MinionType:
    # Matches sim.targeting.MinionType on the existing LaneState path.
    MELEE = 0
    CASTER = 1
    CANNON = 2
    SUPER = 3
    NONE = -1


class TargetKind:
    CHAMPION = 1
    MINION = 2
    TURRET = 3


class TargetPriority:
    CHAMPION_ATTACKING_ALLIED_CHAMPION = 1
    MINION_ATTACKING_ALLIED_CHAMPION = 2
    MINION_ATTACKING_ALLIED_MINION = 3
    TURRET_ATTACKING_ALLIED_MINION = 4
    CLOSEST_MINION = 5
    CLOSEST_CHAMPION = 6
    UNPRIORITIZED = 7


class MinionProfile(NamedTuple):
    # Profile stats at base/upgrades endpoints. Growth between endpoints is
    # deliberately not synthesized until a complete current curve is sourced.
    health_base: float
    health_cap: float
    attack_damage_base: float
    attack_damage_cap: float
    attack_speed: float
    attack_range: float
    armor_base: float
    armor_cap: float
    magic_resist: float
    gameplay_radius: float
    pathing_radius: float
    gold_base: float
    gold_per_upgrade: float
    xp_base: float
    move_speed_base: float


# League Wiki current pages, standard SR. These fields are useful as factual
# profile data; callers needing time-varying stats must supply the separately
# verified game state rather than assume a linear curve.
MELEE_PROFILE = MinionProfile(465., 1550., 11., 80., 1.25, 110., 0., 20., 0.,
                              48., 35.7437, 20., 0., 62., 350.)
CASTER_PROFILE = MinionProfile(284., 600., 21., 125., .667, 550., 0., 0., 0.,
                               48., 35.7437, 14., 0., 31., 350.)
CANNON_PROFILE = MinionProfile(835., 5850., 37.5, 126., 1., 300., 0., 0., 0.,
                               65., 55.7437, 50., 1., 75., 350.)
SUPER_PROFILE = MinionProfile(1600., 7500., 180., 480., .85, 170., 100., 100.,
                              -30., 65., 55.5208, 50., 1., 75., 350.)

WAVE_FIRST_S = 30.
WAVE_UNIT_GAP_S = .792


class SpawnEvent(NamedTuple):
    minion_type: jax.Array
    spawn_time_s: jax.Array
    unit_index: jax.Array
    valid: jax.Array


def wave_spawn_time(wave_index: jax.Array) -> jax.Array:
    """Spawn time for zero-based waves, exactly honoring the 30:00 gap.

    Wave 0 is 00:30. Wave 27 is 14:00, wave 65 is 29:50, and wave 66 is
    30:10. At/after 30:10 waves continue every 20 seconds.
    """
    i = jnp.asarray(wave_index, dtype=jnp.int32)
    early = 30. + 30. * i.astype(jnp.float32)
    mid = 840. + 25. * (i - 27).astype(jnp.float32)
    late = 1810. + 20. * (i - 66).astype(jnp.float32)
    return jnp.where(i < 27, early, jnp.where(i < 66, mid, late))


def wave_interval_s(time_s: jax.Array) -> jax.Array:
    """Interval from a wave at ``time_s`` to its successor (30/25/20 sec)."""
    t = jnp.asarray(time_s, jnp.float32)
    return jnp.where(t < 840., 30., jnp.where(t < 1790., 25., 20.))


def wave_index_at(time_s: jax.Array) -> jax.Array:
    """Index of the most recent wave at ``time_s``; -1 before 00:30."""
    t = jnp.asarray(time_s, jnp.float32)
    early = jnp.floor((t - 30.) / 30.).astype(jnp.int32)
    mid = 27 + jnp.floor((t - 840.) / 25.).astype(jnp.int32)
    late = 66 + jnp.floor((t - 1810.) / 20.).astype(jnp.int32)
    idx = jnp.where(t < 840., early, jnp.where(t < 1810., mid, late))
    # The interval 30:00--30:10 contains no wave: retain wave 65.
    idx = jnp.where((t >= 1800.) & (t < 1810.), 65, idx)
    return jnp.where(t < WAVE_FIRST_S, -1, idx).astype(jnp.int32)


def cannon_wave(wave_index: jax.Array) -> jax.Array:
    """Whether the zero-based wave has its normal siege/cannon minion.

    The first cannon is wave index 2 (01:30). After the rate change, the
    phase is anchored at index 28 (14:25), then index 54 (25:15) begins every
    wave. A super replaces the cannon at composition time, not here.
    """
    i = jnp.asarray(wave_index, jnp.int32)
    return jnp.where(i < 27, (i >= 2) & (i % 3 == 2),
                     jnp.where(i < 54, (i >= 28) & ((i - 28) % 2 == 0),
                               i >= 54))


def wave_composition(wave_index: jax.Array, time_s: jax.Array,
                     enemy_inhibitor_down: jax.Array = False,
                     all_enemy_inhibitors_down: jax.Array = False):
    """Return per-wave counts ``(supers, melee, cannon, casters)``.

    ``enemy_inhibitor_down`` means the inhibitor in this lane is down.
    ``all_enemy_inhibitors_down`` overrides it with two supers in this lane.
    A super wave replaces the cannon; post-14:00 melee pruning only applies
    when a cannon actually spawns. At 30:00 each wave loses one caster.
    """
    i = jnp.asarray(wave_index, jnp.int32)
    t = jnp.asarray(time_s, jnp.float32)
    supers = jnp.where(jnp.asarray(all_enemy_inhibitors_down, bool), 2,
                       jnp.where(jnp.asarray(enemy_inhibitor_down, bool), 1, 0))
    has_cannon = cannon_wave(i) & (supers == 0)
    melee = 3 - ((t >= 840.) & has_cannon).astype(jnp.int32)
    casters = 3 - (t >= 1800.).astype(jnp.int32)
    return supers.astype(jnp.int32), melee, has_cannon.astype(jnp.int32), casters


def spawn_event(wave_index: jax.Array, unit_index: jax.Array,
                enemy_inhibitor_down: jax.Array = False,
                all_enemy_inhibitors_down: jax.Array = False) -> SpawnEvent:
    """Unit type and event time for one unit in a wave.

    Unit order is supers, melee, cannon, casters; each successive unit event
    is 0.792 seconds after the previous event. Indices beyond wave length are
    marked invalid and return ``MinionType.NONE``.
    """
    i = jnp.asarray(wave_index, jnp.int32)
    u = jnp.asarray(unit_index, jnp.int32)
    t = wave_spawn_time(i)
    ns, nm, nc, nr = wave_composition(i, t, enemy_inhibitor_down,
                                      all_enemy_inhibitors_down)
    after_super = u - ns
    after_melee = after_super - nm
    after_cannon = after_melee - nc
    kind = jnp.where(u < ns, MinionType.SUPER,
           jnp.where(after_super < nm, MinionType.MELEE,
           jnp.where(after_melee < nc, MinionType.CANNON,
           jnp.where(after_cannon < nr, MinionType.CASTER, MinionType.NONE))))
    valid = (i >= 0) & (u >= 0) & (kind != MinionType.NONE)
    kind = jnp.where(valid, kind, MinionType.NONE).astype(jnp.int32)
    return SpawnEvent(kind, t + WAVE_UNIT_GAP_S * u.astype(jnp.float32), u, valid)


def upgrade_index_at(time_s: jax.Array) -> jax.Array:
    """Number of 90-second minion upgrade events elapsed since 00:30."""
    t = jnp.asarray(time_s, jnp.float32)
    return jnp.where(t < 30., 0,
                     1 + jnp.floor((t - 30.) / 90.).astype(jnp.int32))


def target_priority(candidate_kind: jax.Array, attacking_victim_kind: jax.Array) -> jax.Array:
    """Build Wiki's 26.10+ minion target rank from candidate behavior.

    ``attacking_victim_kind`` is the kind of unit the candidate currently
    attacks, or 0 if it is not attacking an allied unit. A champion attacking
    an allied minion intentionally has no special rank after 26.10; it falls
    through to ordinary closest-champion rank 6. Other classes receive 7 and
    should be handled by a separate target path (such as structure pushing).
    """
    ck = jnp.asarray(candidate_kind, jnp.int32)
    vk = jnp.asarray(attacking_victim_kind, jnp.int32)
    return jnp.select(
        [(ck == TargetKind.CHAMPION) & (vk == TargetKind.CHAMPION),
         (ck == TargetKind.MINION) & (vk == TargetKind.CHAMPION),
         (ck == TargetKind.MINION) & (vk == TargetKind.MINION),
         (ck == TargetKind.TURRET) & (vk == TargetKind.MINION),
         ck == TargetKind.MINION],
        [TargetPriority.CHAMPION_ATTACKING_ALLIED_CHAMPION,
         TargetPriority.MINION_ATTACKING_ALLIED_CHAMPION,
         TargetPriority.MINION_ATTACKING_ALLIED_MINION,
         TargetPriority.TURRET_ATTACKING_ALLIED_MINION,
         TargetPriority.CLOSEST_MINION],
        default=jnp.where(ck == TargetKind.CHAMPION,
                          TargetPriority.CLOSEST_CHAMPION,
                          TargetPriority.UNPRIORITIZED)).astype(jnp.int32)


def shared_xp_fraction(nearby_enemy_champions: jax.Array) -> jax.Array:
    """26.1 SR XP share per champion (1..6); zero if none are nearby."""
    n = jnp.asarray(nearby_enemy_champions, jnp.int32)
    fractions = jnp.asarray([0., 1., .65, 13. / 30., .325, .26, 13. / 60.], jnp.float32)
    return fractions[jnp.clip(n, 0, 6)]


def gold_bounty(minion_type: jax.Array, upgrade_index: jax.Array) -> jax.Array:
    """Current Summoner's Rift minion kill-gold bounty at an upgrade count."""
    kind = jnp.asarray(minion_type, jnp.int32)
    upgrades = jnp.maximum(jnp.asarray(upgrade_index, jnp.int32), 0)
    return jnp.select(
        [kind == MinionType.MELEE, kind == MinionType.CASTER,
         (kind == MinionType.CANNON) | (kind == MinionType.SUPER)],
        [20., 14., 50. + upgrades.astype(jnp.float32)], default=0.)


def lane_minion_current_hp_bonus(attacker_minion_type: jax.Array,
                                 target_current_hp: jax.Array) -> jax.Array:
    """26.09+ extra on-hit physical damage to lane minions by current HP."""
    kind = jnp.asarray(attacker_minion_type, jnp.int32)
    fraction = jnp.select(
        [kind == MinionType.MELEE, kind == MinionType.CASTER,
         kind == MinionType.CANNON],
        [.02, .035, .05], default=0.)
    return jnp.asarray(target_current_hp, jnp.float32) * fraction


def minion_pushing_modifiers(*, team_level_advantage: jax.Array,
                             lane_turret_advantage: jax.Array,
                             time_s: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Wiki SR minion-pushing buff: bonus damage and incoming-damage divisor.

    Inputs are nonnegative advantages measured in champion levels and lane
    turrets (the advantaged minion team's levels/turrets minus the opponent's,
    each floored at zero). Level advantage caps at 3. The buff starts at
    03:30, applies to existing minions, and modifiers update with advantages.
    Returns ``(bonus_damage_fraction, damage_divisor)``; divide minion-on-
    minion incoming damage by the latter.
    """
    level = jnp.minimum(jnp.maximum(jnp.asarray(team_level_advantage, jnp.float32), 0.), 3.)
    towers = jnp.maximum(jnp.asarray(lane_turret_advantage, jnp.float32), 0.)
    active = jnp.asarray(time_s, jnp.float32) >= 210.
    bonus = (.05 + .05 * towers) * level
    divisor = 1. + towers * level
    return jnp.where(active, bonus, 0.), jnp.where(active, divisor, 1.)


def select_target(candidate_priority: jax.Array, distance_sq: jax.Array,
                  valid: jax.Array, current_target: jax.Array = -1) -> jax.Array:
    """Choose the lowest priority number, then nearest; retain incumbent ties.

    Priority numbers should be the six documented post-26.10 ranks:
    1 enemy champion attacking allied champion; 2 enemy minion attacking
    allied champion; 3 enemy minion attacking allied minion; 4 enemy turret
    attacking allied minion; 5 closest enemy minion; 6 closest enemy champion.
    Since a held target is replaced only by strictly higher priority, equal
    priority candidates cannot steal it. `valid=False` for allies/dead/out of
    acquisition range candidates. The caller maintains collection-order ties.
    """
    p = jnp.asarray(candidate_priority, jnp.int32)
    d2 = jnp.asarray(distance_sq, jnp.float32)
    ok = jnp.asarray(valid, bool)
    n = p.shape[-1]
    safe_cur = jnp.clip(jnp.asarray(current_target, jnp.int32), 0, n - 1)
    has_cur = (current_target >= 0) & ok[safe_cur]
    cur_p = p[safe_cur]
    eligible = ok & (~has_cur | (p < cur_p))
    best_p = jnp.min(jnp.where(eligible, p, jnp.iinfo(jnp.int32).max), axis=-1)
    best_d = jnp.min(jnp.where(eligible & (p == best_p), d2, jnp.inf), axis=-1)
    first = jnp.argmax((eligible & (p == best_p) & (d2 == best_d)).astype(jnp.int32), axis=-1)
    any_target = jnp.any(eligible, axis=-1)
    return jnp.where(any_target, first, jnp.where(has_cur, safe_cur, -1)).astype(jnp.int32)


def call_for_help_applies(distance: jax.Array, victim_is_allied_champion_under_attack: jax.Array,
                          acquisition_range: jax.Array) -> jax.Array:
    """Distance gate for an eligible attacker/victim Call-for-Help pair.

    Normally 500 units; the allied champion being attacked by an enemy
    champion may be up to 1000 units away. ``acquisition_range`` is explicit
    because minion attack range is minion-specific in the current Wiki.
    """
    r = jnp.where(victim_is_allied_champion_under_attack, 1000., acquisition_range)
    return jnp.asarray(distance, jnp.float32) <= r


def call_for_help_trigger(*, champion_hit_enemy_champion: jax.Array,
                          champion_in_path: jax.Array,
                          no_other_target_in_attack_range: jax.Array,
                          outside_turret_range: jax.Array,
                          minion_attacking_turret: jax.Array,
                          is_first_wave: jax.Array) -> jax.Array:
    """Documented Call-for-Help trigger; caller separately applies distance.

    The 26.10 change removes an enemy champion attacking an allied minion as
    a trigger/priority. Basic attacks, most targeted abilities, and some
    explicitly similar abilities damaging an enemy champion still trigger.
    Minions attacking a turret ignore the signal after wave one.
    """
    damage_call = jnp.asarray(champion_hit_enemy_champion, bool)
    path_call = (jnp.asarray(champion_in_path, bool)
                 & jnp.asarray(no_other_target_in_attack_range, bool)
                 & jnp.asarray(outside_turret_range, bool))
    blocked = jnp.asarray(minion_attacking_turret, bool) & ~jnp.asarray(is_first_wave, bool)
    return (damage_call | path_call) & ~blocked
