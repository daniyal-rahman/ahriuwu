"""26.19 lane-minion rules: wave schedule and composition, per-upgrade stats, move speed, bounties (MINIONS.md).

Pure JAX; times in seconds. The per-tick driver is ``lane.ai``. Stats follow the client ``MinionUpgradeConfig``
(§1.3): ``stat(U) = base + min(Up*U + UpLate*max(U-5, 0), MaxBonus)`` with ``U`` latched at spawn.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class MinionType:
    MELEE = 0
    CASTER = 1
    CANNON = 2
    SUPER = 3
    NONE = -1


class TargetPriority:
    """Post-26.10 minion target classes (MINIONS §3.1); lower wins."""
    CHAMPION_ATTACKING_ALLIED_CHAMPION = 1
    MINION_ATTACKING_ALLIED_CHAMPION = 2
    MINION_ATTACKING_ALLIED_MINION = 3
    TURRET_ATTACKING_ALLIED_MINION = 4
    CLOSEST_MINION = 5
    CLOSEST_CHAMPION = 6
    UNPRIORITIZED = 7


WAVE_FIRST_S = 30.
WAVE_UNIT_GAP_S = .8            # client MinionSpawnIntervalSecs (§2.4)

# Per-type tables indexed by MinionType (client, §1.1, §3.2).
ATTACK_SPEED = (1.25, .667, 1., .85)
ATTACK_RANGE = (110., 550., 300., 170.)
GAMEPLAY_RADIUS = (48., 48., 65., 65.)
WINDUP_S = (.393, .47, .3, .5 / 1.44 / .85)
MISSILE_SPEED = (0., 650., 1200., 0.)        # 0 = melee (hit at launch)
ACQUISITION_RANGE = (750., 700., 750., 600.)
FIRST_ACQUISITION_RANGE = (1000., 900., 750., 600.)
WAKE_UP_RANGE = (450., 635., 750., 600.)
XP_BASE = (62., 31., 75., 75.)
MINION_SLAYER_FRACTION = (.02, .035, .05, 0.)   # x target current HP vs lane minions (§4.2)
CFH_GENERIC_RADIUS = 500.                       # Call-for-Help listener radii (wiki, §3.3)
CFH_CHAMPION_RADIUS = 1000.

# MinionUpgradeConfig (CLASSIC Order barracks, §1.2).
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


def wave_spawn_time(wave_index: jax.Array) -> jax.Array:
    """Zero-based wave -> spawn time: every 30 s from 0:30, 25 s from 14:00, 20 s from 30:10 (no 30:00 wave)."""
    i = jnp.asarray(wave_index, dtype=jnp.int32)
    early = 30. + 30. * i.astype(jnp.float32)
    mid = 840. + 25. * (i - 27).astype(jnp.float32)
    late = 1810. + 20. * (i - 66).astype(jnp.float32)
    return jnp.where(i < 27, early, jnp.where(i < 66, mid, late))


def wave_interval_s(time_s: jax.Array) -> jax.Array:
    """Interval from a wave at ``time_s`` to its successor."""
    t = jnp.asarray(time_s, jnp.float32)
    return jnp.where(t < 840., 30., jnp.where(t < 1790., 25., 20.))


def wave_index_at(time_s: jax.Array) -> jax.Array:
    """Index of the most recent wave at ``time_s``; -1 before 0:30."""
    t = jnp.asarray(time_s, jnp.float32)
    early = jnp.floor((t - 30.) / 30.).astype(jnp.int32)
    mid = 27 + jnp.floor((t - 840.) / 25.).astype(jnp.int32)
    late = 66 + jnp.floor((t - 1810.) / 20.).astype(jnp.int32)
    idx = jnp.where(t < 840., early, jnp.where(t < 1810., mid, late))
    idx = jnp.where((t >= 1800.) & (t < 1810.), 65, idx)
    return jnp.where(t < WAVE_FIRST_S, -1, idx).astype(jnp.int32)


def cannon_wave(wave_index: jax.Array) -> jax.Array:
    """Wave has a cannon: every 3rd from index 2, every 2nd from index 28, every wave from 54 (before supers)."""
    i = jnp.asarray(wave_index, jnp.int32)
    return jnp.where(i < 27, (i >= 2) & (i % 3 == 2),
                     jnp.where(i < 54, (i >= 28) & ((i - 28) % 2 == 0),
                               i >= 54))


def upgrade_index_at(time_s: jax.Array) -> jax.Array:
    """Number of 90 s upgrade events elapsed since 0:30."""
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
    """Melee ``ArmorUpgradeGrowth``: ``0.085*(U-6)*(U-5)/2`` from U=6, capped at 20 (wiki shape, §1.3)."""
    u = jnp.asarray(upgrade_index, jnp.float32)
    return jnp.where(u >= 6., jnp.minimum(.085 * (u - 6.) * (u - 5.) / 2., 20.), 0.)


def minion_upgrade_stats(minion_type: jax.Array, upgrade_index: jax.Array,
                         team: jax.Array = 0) -> MinionUpgradeStats:
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
    """350 + 25 at 10:30, 15:30, 20:30, 25:30, applied to living minions at once (§2.6)."""
    t = jnp.asarray(time_s, jnp.float32)
    steps = jnp.clip(jnp.floor((t - 630.) / 300.) + 1., 0., 4.)
    return 350. + 25. * steps


def sidelane_bonus_move_speed(wave_number: jax.Array, lane: jax.Array,
                              wave_time_s: jax.Array, since_spawn_s: jax.Array) -> jax.Array:
    """Side-lane bonus MS (§2.7), 1-based ``wave_number`` >= 2 before 14:00: ``max(0, 120 - 4.5 n)`` minus 15
    every 7 s, gone at 25 s."""
    n = jnp.asarray(wave_number, jnp.float32)
    tau = jnp.asarray(since_spawn_s, jnp.float32)
    b = jnp.maximum(0., 120. - 4.5 * n)
    step = jnp.floor(jnp.maximum(tau, 0.) / 7.)
    bonus = jnp.maximum(0., b - 15. * step)
    ok = ((jnp.asarray(lane) != 1) & (n >= 2.) & (jnp.asarray(wave_time_s) < 840.)
          & (tau >= 0.) & (tau < 25.))
    return jnp.where(ok, bonus, 0.)


def move_speed_soft_cap(raw: jax.Array) -> jax.Array:
    """DAMAGE_AND_STATS §9.2 upper soft caps."""
    raw = jnp.asarray(raw, jnp.float32)
    return jnp.where(raw > 490., .5 * raw + 230., jnp.where(raw > 415., .8 * raw + 83., raw))


def gold_bounty(minion_type: jax.Array, upgrade_index: jax.Array,
                team: jax.Array = 0) -> jax.Array:
    """Melee 20, caster 14, siege/super ``min(49 + U, 90)``; Chaos supers a flat 49 (no client GoldUpgrade, §1.3)."""
    kind = jnp.asarray(minion_type, jnp.int32)
    upgrades = jnp.maximum(jnp.asarray(upgrade_index, jnp.int32), 0)
    scaled = jnp.minimum(49. + upgrades.astype(jnp.float32), 90.)
    chaos_super = (kind == MinionType.SUPER) & (jnp.asarray(team) == 1)
    return jnp.select(
        [kind == MinionType.MELEE, kind == MinionType.CASTER, chaos_super,
         (kind == MinionType.CANNON) | (kind == MinionType.SUPER)],
        [20., 14., 49., scaled], default=0.)


def minion_pushing_modifiers(*, team_level_advantage: jax.Array,
                             lane_turret_advantage: jax.Array,
                             time_s: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Minion-pushing buff from 3:30 (wiki): ``(bonus damage fraction, incoming minion damage divisor)``.

    Advantages are the leading team's levels (capped at 3) and lane turrets over the opponent's, floored at 0."""
    level = jnp.minimum(jnp.maximum(jnp.asarray(team_level_advantage, jnp.float32), 0.), 3.)
    towers = jnp.maximum(jnp.asarray(lane_turret_advantage, jnp.float32), 0.)
    active = jnp.asarray(time_s, jnp.float32) >= 210.
    bonus = (.05 + .05 * towers) * level
    divisor = 1. + towers * level
    return jnp.where(active, bonus, 0.), jnp.where(active, divisor, 1.)


# --- all-lane spawn schedule (§2.2-2.4) ---------------------------------------------------------------------------
N_LANES = 3                       # 0 bot, 1 mid, 2 top
N_TEAMS = 2


class LaneSpawnState(NamedTuple):
    """Per (team, lane) wave cursor, (2, 3). ``supers`` is latched at the wave's first unit (-1: not yet), so an
    inhibitor dying or respawning mid-wave cannot reorder that wave (INFERRED M)."""
    wave: jax.Array
    unit: jax.Array
    supers: jax.Array


def init_lane_spawn() -> LaneSpawnState:
    z = jnp.zeros((N_TEAMS, N_LANES), jnp.int32)
    return LaneSpawnState(z, z, z - 1)


def super_count(enemy_inhibitor_down, all_enemy_inhibitors_down, inhibitor_respawn_at, time_s):
    """Supers per wave in one lane (§2.3): 1 if this lane's enemy inhibitor is down, 2 if all three are, 0 within
    two wave intervals of this lane's inhibitor respawning (an all-down lane keeps 2 until it respawns)."""
    t = jnp.asarray(time_s, jnp.float32)
    down = jnp.asarray(enemy_inhibitor_down, bool)
    s = jnp.where(jnp.asarray(all_enemy_inhibitors_down, bool), 2, jnp.where(down, 1, 0))
    soon = (jnp.asarray(inhibitor_respawn_at, jnp.float32) - t) < 2. * wave_interval_s(t)
    return jnp.where(down & soon, 0, s).astype(jnp.int32)


def wave_unit_type(wave_index, unit_index, supers):
    """``(type, valid)`` of unit ``unit_index``: supers, melee, cannon (replaced by supers), casters (§2.3).

    Melee: 3 before 14:00, [2, 3] by wave parity until 25:00, then 2; one caster fewer from 30:00."""
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
    """Units due this tick, (2, 3) by (team, lane)."""
    due: jax.Array
    minion_type: jax.Array        # NONE where not due
    wave: jax.Array
    unit: jax.Array


def lane_spawn_step(state: LaneSpawnState, now, *, enemy_inhibitor_down=False,
                    all_enemy_inhibitors_down=False, inhibitor_respawn_at=jnp.inf):
    """Advance the (team, lane) cursors to ``now``: ``(state, due)``, at most one unit per cursor per tick.

    Inputs broadcast to (2, 3) and describe that team's *enemy* inhibitors (``lane.ai.wave_inhibitor_inputs``)."""
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
    """First-wave ghosting after spawn: 28 s side lanes, 18 s mid (§2.8, wiki)."""
    return jnp.where(jnp.asarray(lane) == 1, 18., 28.)
