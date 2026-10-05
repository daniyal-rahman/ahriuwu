"""Pure JAX rules for 26.19 Summoner's Rift lane minions.

Scope is the wave schedule, composition, per-upgrade stats, profile facts,
target selection/Call-for-Help rules, movement-speed timing and bounties
(docs/modern/MINIONS.md). The per-tick AI that drives these rules is
``lane.ai``; movement/pathing and combat resolution belong to the world
step. All clock values are seconds and map coordinates are XZ.

Per-upgrade stats follow the client CLASSIC ``MinionUpgradeConfig``
(MINIONS §1.3): ``stat(U) = base + min(Up*U + UpLate*max(U-5, 0), MaxBonus)``
with the upgrade index ``U`` latched at spawn (U-1 default). The ``*_PROFILE``
tuples are U=1 (first-wave) snapshots of that formula.
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
    "call_for_help_trigger", "MinionUpgradeStats", "minion_upgrade_stats",
    "melee_armor", "ACQUISITION_RANGE", "FIRST_ACQUISITION_RANGE",
    "WAKE_UP_RANGE", "WINDUP_S", "MISSILE_SPEED", "ATTACK_SPEED",
    "ATTACK_RANGE", "GAMEPLAY_RADIUS", "XP_BASE", "base_move_speed",
    "sidelane_bonus_move_speed", "move_speed_soft_cap",
    "MINION_SLAYER_FRACTION", "CFH_GENERIC_RADIUS", "CFH_CHAMPION_RADIUS",
    "N_LANES", "LaneSpawnState", "init_lane_spawn", "super_count", "wave_unit_type",
    "LaneSpawnDue", "lane_spawn_step", "first_wave_ghost_s",
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
    # U=1 snapshot (``health_base``/``attack_damage_base``/``gold_base``) and
    # stat caps of the client upgrade formula; ``minion_upgrade_stats`` gives
    # every other upgrade index.
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


# Client CharacterRecords + CLASSIC barracks config (MINIONS §1.1-1.4).
MELEE_PROFILE = MinionProfile(465., 1550., 11., 80., 1.25, 110., 0., 20., 0.,
                              48., 35.7437, 20., 0., 62., 350.)
CASTER_PROFILE = MinionProfile(284., 600., 21., 125., .667, 550., 0., 0., 0.,
                               48., 35.7437, 14., 0., 31., 350.)
CANNON_PROFILE = MinionProfile(835., 5850., 37.5, 126., 1., 300., 0., 0., 0.,
                               65., 55.7437, 50., 1., 75., 350.)
# Super AD at U=1 is 180 + 5 = 185 (the old 180 mixed in the U=0 value).
SUPER_PROFILE = MinionProfile(1600., 7500., 185., 480., .85, 170., 100., 100.,
                              -30., 65., 55.5208, 50., 1., 75., 350.)

WAVE_FIRST_S = 30.
# Client ``MinionSpawnIntervalSecs`` 0.800000011920929 (MINIONS §2.4, U-13);
# the former 0.792 had no client source.
WAVE_UNIT_GAP_S = .8

# Per-type tables, indexed by MinionType (melee, caster, siege, super).
# Client values (MINIONS §1.1, §3.2); first-wave ranges are melee/caster only.
ATTACK_SPEED = (1.25, .667, 1., .85)
ATTACK_RANGE = (110., 550., 300., 170.)
GAMEPLAY_RADIUS = (48., 48., 65., 65.)
WINDUP_S = (.393, .47, .3, .5 / 1.44 / .85)
MISSILE_SPEED = (0., 650., 1200., 0.)        # 0 = melee (hit at launch)
ACQUISITION_RANGE = (750., 700., 750., 600.)
FIRST_ACQUISITION_RANGE = (1000., 900., 750., 600.)
WAKE_UP_RANGE = (450., 635., 750., 600.)
XP_BASE = (62., 31., 75., 75.)
# Minion Slayer / lane-minion bonus vs lane minions, x target current HP
# (items 1509/1510/1508; MINIONS §4.2).
MINION_SLAYER_FRACTION = (.02, .035, .05, 0.)
# Call-for-Help listener radii (wiki; MINIONS §3.2/§3.3, U-9).
CFH_GENERIC_RADIUS = 500.
CFH_CHAMPION_RADIUS = 1000.

# MinionUpgradeConfig (CLASSIC Order barracks {147211fb}; MINIONS §1.2).
_BASE_HP = (430., 275., 750., 1500.)
_HP_UP = (35., 9., 85., 100.)
_HP_MAX_BONUS = (1120., 325., 5100., 6000.)
_BASE_AD = (11., 19.5, 36., 180.)
_AD_UP = (0., 1.5, 1.5, 5.)
_AD_UP_LATE = (3., 2.5, 2.5, 0.)
_AD_MAX_BONUS = (69., 105.5, 90., 300.)
_BASE_ARMOR = (0., 0., 0., 100.)
_BASE_MR = (0., 0., 0., -30.)
UPGRADES_BEFORE_LATE = 5


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
                     all_enemy_inhibitors_down: jax.Array = False,
                     inhibitor_respawn_at: jax.Array = jnp.inf):
    """Return per-wave counts ``(supers, melee, cannon, casters)``.

    ``enemy_inhibitor_down`` means the inhibitor in this lane is down.
    ``all_enemy_inhibitors_down`` overrides it with two supers in this lane
    (``SpawnCountPerInhibitorDown [1,1,2]``). A super wave replaces the cannon.
    Melee counts follow the client rotation independently of the cannon/super
    (``3`` before 14:00, ``[2,3]`` by wave parity until 25:00, then ``2``;
    MINIONS §2.3, U-12). At 30:00 each wave loses one caster. Supers stop two
    waves before the lane inhibitor respawns (``inhibitor_respawn_at``;
    MINIONS §2.3, INFERRED L on the comparison).
    """
    i = jnp.asarray(wave_index, jnp.int32)
    t = jnp.asarray(time_s, jnp.float32)
    supers = jnp.where(jnp.asarray(all_enemy_inhibitors_down, bool), 2,
                       jnp.where(jnp.asarray(enemy_inhibitor_down, bool), 1, 0))
    respawn_soon = (jnp.asarray(inhibitor_respawn_at, jnp.float32) - t) < 2. * wave_interval_s(t)
    supers = jnp.where(respawn_soon, 0, supers)
    has_cannon = cannon_wave(i) & (supers == 0)
    melee = jnp.where(t < 840., 3, jnp.where(t < 1500., jnp.where(i % 2 == 0, 2, 3), 2))
    casters = 3 - (t >= 1800.).astype(jnp.int32)
    return (supers.astype(jnp.int32), melee.astype(jnp.int32),
            has_cannon.astype(jnp.int32), casters)


def spawn_event(wave_index: jax.Array, unit_index: jax.Array,
                enemy_inhibitor_down: jax.Array = False,
                all_enemy_inhibitors_down: jax.Array = False,
                inhibitor_respawn_at: jax.Array = jnp.inf) -> SpawnEvent:
    """Unit type and event time for one unit in a wave.

    Unit order is supers, melee, cannon, casters; each successive unit event
    is 0.8 seconds after the previous event. Indices beyond wave length are
    marked invalid and return ``MinionType.NONE``.
    """
    i = jnp.asarray(wave_index, jnp.int32)
    u = jnp.asarray(unit_index, jnp.int32)
    t = wave_spawn_time(i)
    ns, nm, nc, nr = wave_composition(i, t, enemy_inhibitor_down,
                                      all_enemy_inhibitors_down, inhibitor_respawn_at)
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


class MinionUpgradeStats(NamedTuple):
    max_hp: jax.Array
    attack_damage: jax.Array
    armor: jax.Array
    magic_resist: jax.Array
    gold: jax.Array
    xp: jax.Array


def melee_armor(upgrade_index: jax.Array) -> jax.Array:
    """Melee ``ArmorUpgradeGrowth 0.085`` (wiki shape, MINIONS §1.3, U-2 L).

    ``0.085*(U-6)*(U-5)/2`` from U=6, capped at 20: 0 @6, 0.085 @7, 0.85 @10.
    """
    u = jnp.asarray(upgrade_index, jnp.float32)
    return jnp.where(u >= 6., jnp.minimum(.085 * (u - 6.) * (u - 5.) / 2., 20.), 0.)


def minion_upgrade_stats(minion_type: jax.Array, upgrade_index: jax.Array,
                         team: jax.Array = 0) -> MinionUpgradeStats:
    """Client per-upgrade stats of a lane minion latched at upgrade ``U``.

    ``bonus(U) = Up*U + UpLate*max(U-5, 0)``, capped at ``Max*`` (a cap on
    the bonus). Gold: siege/super ``min(49+U, 90)``; Chaos (team 1) supers
    have no GoldUpgrade in the client and stay at 49 (MINIONS §1.3).
    """
    k = jnp.clip(jnp.asarray(minion_type, jnp.int32), 0, 3)
    u = jnp.maximum(jnp.asarray(upgrade_index, jnp.int32), 0).astype(jnp.float32)
    late = jnp.maximum(u - UPGRADES_BEFORE_LATE, 0.)
    tab = lambda v: jnp.asarray(v, jnp.float32)[k]
    hp = tab(_BASE_HP) + jnp.minimum(tab(_HP_UP) * u, tab(_HP_MAX_BONUS))
    ad = tab(_BASE_AD) + jnp.minimum(tab(_AD_UP) * u + tab(_AD_UP_LATE) * late, tab(_AD_MAX_BONUS))
    armor = tab(_BASE_ARMOR) + jnp.where(k == MinionType.MELEE, melee_armor(u), 0.)
    return MinionUpgradeStats(hp, ad, armor, tab(_BASE_MR),
                              gold_bounty(k, u.astype(jnp.int32), team), tab(XP_BASE))


def base_move_speed(time_s: jax.Array) -> jax.Array:
    """350 + 25 per increase at 10:30, 15:30, 20:30, 25:30 (MINIONS §2.6 A, U-3).

    Global: applies to living minions immediately (default, INFERRED L).
    """
    t = jnp.asarray(time_s, jnp.float32)
    steps = jnp.clip(jnp.floor((t - 630.) / 300.) + 1., 0., 4.)
    return 350. + 25. * steps


def sidelane_bonus_move_speed(wave_number: jax.Array, lane: jax.Array,
                              wave_time_s: jax.Array, since_spawn_s: jax.Array) -> jax.Array:
    """Flat side-lane bonus MS (MINIONS §2.7). ``wave_number`` is 1-based.

    Lanes 0 (bot) and 2 (top) only, waves 2+ spawning before 14:00:
    ``B = max(0, 120 - 4.5 n)`` stepping down by 15 every 7 s, removed at 25 s.
    """
    n = jnp.asarray(wave_number, jnp.float32)
    tau = jnp.asarray(since_spawn_s, jnp.float32)
    b = jnp.maximum(0., 120. - 4.5 * n)
    step = jnp.floor(jnp.maximum(tau, 0.) / 7.)
    bonus = jnp.maximum(0., b - 15. * step)
    ok = ((jnp.asarray(lane) != 1) & (n >= 2.) & (jnp.asarray(wave_time_s) < 840.)
          & (tau >= 0.) & (tau < 25.))
    return jnp.where(ok, bonus, 0.)


def move_speed_soft_cap(raw: jax.Array) -> jax.Array:
    """Movement-speed soft caps (DAMAGE_AND_STATS §9.2; MINIONS §2.7, U-11)."""
    raw = jnp.asarray(raw, jnp.float32)
    return jnp.where(raw > 490., .5 * raw + 230., jnp.where(raw > 415., .8 * raw + 83., raw))


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
    """SR XP share per champion (1..6); zero if none are nearby.

    Client ``ExperienceModData.mPlayerMinionSplitXp`` floats (MINIONS §5.2).
    """
    n = jnp.asarray(nearby_enemy_champions, jnp.int32)
    fractions = jnp.asarray([0., 1., .65, .433, .325, .26, .217], jnp.float32)
    return fractions[jnp.clip(n, 0, 6)]


def gold_bounty(minion_type: jax.Array, upgrade_index: jax.Array,
                team: jax.Array = 0) -> jax.Array:
    """Minion kill-gold bounty at a (spawn-latched) upgrade count.

    Melee 20, caster 14, siege ``min(49 + U, 90)`` (client goldGivenOnDeath
    49 + GoldUpgrade 1, GoldMax 90 as a total cap). Supers: Order the same,
    Chaos (team 1) a flat 49 because its barracks lacks GoldUpgrade
    (MINIONS §1.3, client data, probably a Riot data bug).
    """
    kind = jnp.asarray(minion_type, jnp.int32)
    upgrades = jnp.maximum(jnp.asarray(upgrade_index, jnp.int32), 0)
    scaled = jnp.minimum(49. + upgrades.astype(jnp.float32), 90.)
    chaos_super = (kind == MinionType.SUPER) & (jnp.asarray(team) == 1)
    return jnp.select(
        [kind == MinionType.MELEE, kind == MinionType.CASTER, chaos_super,
         (kind == MinionType.CANNON) | (kind == MinionType.SUPER)],
        [20., 14., 49., scaled], default=0.)


def lane_minion_current_hp_bonus(attacker_minion_type: jax.Array,
                                 target_current_hp: jax.Array) -> jax.Array:
    """26.09+ extra on-hit physical damage to lane minions by current HP."""
    kind = jnp.asarray(attacker_minion_type, jnp.int32)
    fraction = jnp.where((kind >= 0) & (kind <= 3),
                         jnp.asarray(MINION_SLAYER_FRACTION, jnp.float32)[jnp.clip(kind, 0, 3)], 0.)
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
    minion incoming damage by the latter. The client refreshes it every 1.0 s
    (``mvm_UpdateInterval``): callers evaluate it on 1-s boundaries and hold
    the result; a one-champion team's average level is that champion's level.
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
                          generic_radius: jax.Array = CFH_GENERIC_RADIUS) -> jax.Array:
    """Listener-to-victim distance gate for a Call-for-Help pair.

    500 units generally; an allied champion attacked by an enemy champion may
    be up to 1000 units away (wiki, MINIONS §3.3, U-9). The generic radius is
    no longer the listener's acquisition range (MINIONS §9 diff); the
    attacker must separately be inside the listener's scan range.
    """
    r = jnp.where(victim_is_allied_champion_under_attack, CFH_CHAMPION_RADIUS, generic_radius)
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


# --- all-lane spawn schedule (MINIONS §2.2-2.4; LANES_TERRAIN.md §1) ---------
N_LANES = 3                       # geometry lane ids: 0 bot, 1 mid, 2 top
N_TEAMS = 2


class LaneSpawnState(NamedTuple):
    """Per (team, lane) wave cursor, shape (2, 3).

    ``supers`` is the super count latched when the wave's first unit spawns
    (-1: not latched yet), so an inhibitor dying or respawning during the
    4-5 s spawn window cannot reorder or duplicate units of that wave
    (INFERRED M: the client builds a wave's unit list once per wave).
    """
    wave: jax.Array               # int32 zero-based wave index of the next unit
    unit: jax.Array               # int32 unit index inside that wave
    supers: jax.Array             # int32 latched super count, -1 = none yet


def init_lane_spawn() -> LaneSpawnState:
    z = jnp.zeros((N_TEAMS, N_LANES), jnp.int32)
    return LaneSpawnState(z, z, z - 1)


def super_count(enemy_inhibitor_down, all_enemy_inhibitors_down, inhibitor_respawn_at, time_s):
    """Supers per wave in one lane: 1 if this lane's enemy inhibitor is down,
    2 in every lane if all three are (``SpawnCountPerInhibitorDown [1,1,2]``),
    0 within two wave intervals of that inhibitor's respawn (MINIONS §2.3).

    With all three down, a lane whose inhibitor is about to respawn still
    counts as "all down" until it actually respawns (INFERRED L)."""
    t = jnp.asarray(time_s, jnp.float32)
    down = jnp.asarray(enemy_inhibitor_down, bool)
    s = jnp.where(jnp.asarray(all_enemy_inhibitors_down, bool), 2, jnp.where(down, 1, 0))
    soon = (jnp.asarray(inhibitor_respawn_at, jnp.float32) - t) < 2. * wave_interval_s(t)
    return jnp.where(down & soon, 0, s).astype(jnp.int32)


def wave_unit_type(wave_index, unit_index, supers):
    """Type of unit ``unit_index`` of wave ``wave_index`` with ``supers`` supers.

    Same order and counts as ``spawn_event`` (supers, melee, cannon, casters;
    a super replaces the cannon), but from an explicit super count.
    Returns ``(type, valid)``; ``MinionType.NONE`` past the wave's end."""
    i = jnp.asarray(wave_index, jnp.int32)
    u = jnp.asarray(unit_index, jnp.int32)
    ns = jnp.maximum(jnp.asarray(supers, jnp.int32), 0)
    t = wave_spawn_time(i)
    nm = jnp.where(t < 840., 3, jnp.where(t < 1500., jnp.where(i % 2 == 0, 2, 3), 2))
    nc = (cannon_wave(i) & (ns == 0)).astype(jnp.int32)
    nr = 3 - (t >= 1800.).astype(jnp.int32)
    a = u - ns
    b = a - nm
    c = b - nc
    kind = jnp.where(u < ns, MinionType.SUPER, jnp.where(a < nm, MinionType.MELEE,
                     jnp.where(b < nc, MinionType.CANNON, jnp.where(c < nr, MinionType.CASTER, MinionType.NONE))))
    valid = (i >= 0) & (u >= 0) & (kind != MinionType.NONE)
    return jnp.where(valid, kind, MinionType.NONE).astype(jnp.int32), valid


class LaneSpawnDue(NamedTuple):
    """Units due this tick, shape (2, 3) by (team, lane)."""
    due: jax.Array                # bool
    minion_type: jax.Array        # int32 (NONE where not due)
    wave: jax.Array               # int32 wave index of the unit
    unit: jax.Array               # int32 unit index inside its wave


def lane_spawn_step(state: LaneSpawnState, now, *, enemy_inhibitor_down=False,
                    all_enemy_inhibitors_down=False, inhibitor_respawn_at=jnp.inf):
    """Advance the per-(team, lane) wave cursors to ``now``.

    Inputs broadcast to (2, 3) ``[team, lane]`` and describe *that team's
    enemy*: ``enemy_inhibitor_down[t, l]`` = the inhibitor of team ``1-t`` in
    lane ``l`` is dead (``lane.ai.wave_inhibitor_inputs`` builds
    them from ``TowersState``). At most one unit per (team, lane) per tick:
    the 0.8 s stagger exceeds any tick length. Returns ``(state, due)``.
    """
    shape = (N_TEAMS, N_LANES)
    now = jnp.asarray(now, jnp.float32)
    down = jnp.broadcast_to(jnp.asarray(enemy_inhibitor_down, bool), shape)
    all_down = jnp.broadcast_to(jnp.asarray(all_enemy_inhibitors_down, bool), shape)
    respawn = jnp.broadcast_to(jnp.asarray(inhibitor_respawn_at, jnp.float32), shape)
    wave = jnp.broadcast_to(jnp.asarray(state.wave, jnp.int32), shape)
    unit = jnp.broadcast_to(jnp.asarray(state.unit, jnp.int32), shape)
    latched = jnp.broadcast_to(jnp.asarray(state.supers, jnp.int32), shape)
    t_wave = wave_spawn_time(wave)
    supers = jnp.where(latched >= 0, latched, super_count(down, all_down, respawn, t_wave))
    kind, valid = wave_unit_type(wave, unit, supers)
    due = valid & (now >= t_wave + WAVE_UNIT_GAP_S * unit.astype(jnp.float32))
    _, more = wave_unit_type(wave, unit + 1, supers)
    close = due & ~more
    new = LaneSpawnState(
        wave=jnp.where(close, wave + 1, wave).astype(jnp.int32),
        unit=jnp.where(close, 0, jnp.where(due, unit + 1, unit)).astype(jnp.int32),
        supers=jnp.where(close, -1, jnp.where(due, supers, latched)).astype(jnp.int32))
    return new, LaneSpawnDue(due, jnp.where(due, kind, MinionType.NONE).astype(jnp.int32), wave, unit)


def first_wave_ghost_s(lane):
    """First-wave ghosting after spawn: 28 s side lanes, 18 s mid (MINIONS §2.8, WIKI M)."""
    return jnp.where(jnp.asarray(lane) == 1, 18., 28.)
