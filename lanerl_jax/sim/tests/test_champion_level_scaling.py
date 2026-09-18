"""A champion's attack damage grows with level, the way ``Stats.LevelUp`` does.

Before this was wired in, ``sim/step.py``'s ``tick`` computed a champion's
outgoing damage from ``P("attack_damage")`` alone -- a **static** per-profile
value baked once in ``profiles.build_profile_tables`` from the champion's
level-1(+rune) stats, gathered fresh every tick but never a function of
``state.level``. ``state.level`` itself was already correct (``rewards.
level_for_xp``, driven by proximity-shared XP off every minion death within
1600 units -- see ``rewards.death_rewards``), so a champion visibly leveled up
over an episode while still swinging for exactly its level-1 damage the whole
time.

That is not a cosmetic gap. ``Stats.LevelUp``
(``LoLServer/GameServerLib/GameObjects/Stats/Stats.cs:270-271``)::

    statsLevelUp.AttackDamage.BaseValue = GetLevelUpStatValue(AttackDamagePerLevel.BaseValue);
    statsLevelUp.AttackDamage.FlatBonus = GetLevelUpStatValue(AttackDamagePerLevel.FlatBonus);

adds to ``AttackDamage`` on every level gained, through the same non-linear
per-level curve as HP, armour and magic resist (``combat.stat_at_level`` /
``growth_sum``). The wire confirms the server's real number moves: Garen's
``LanerlControl.cs``-emitted ``"ad"`` field is ``Stats.AttackDamage.Total``
directly, and that module's own comment records the old Python stack paying
for re-deriving it once already ("read 57.88 against a real 73.14, a 21%
under-report"). J1 gate 3 (the last-hit oracle, ``lanerl_jax/parity/
last_hit_drive.py``) reads its "would this attack kill" threshold from exactly
this number, so a flat champion AD directly narrows how often a minion falls
into the one-shot band as the episode goes on -- see
``lanerl_jax/parity/hp_band.py``.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch
from lanerl_jax.sim.combat import growth_sum
from lanerl_jax.sim.init import (MASTERY_AD_PER_LEVEL_BONUS, RUNE_AD_BONUS,
                                 init_lane, lane_params)
from lanerl_jax.sim.movement_jax import TICK_MS
from lanerl_jax.sim.profiles import profile_id
from lanerl_jax.sim.state import Kind, Team
from lanerl_jax.sim.step import tick
from lanerl_jax.sim.targeting import MinionType

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)

BLUE_CHAMP, RED_CHAMP = 0, 1
VICTIM = 2   # a free minion slot


@functools.lru_cache(maxsize=1)
def _ticker():
    patch = load_patch()
    params = lane_params(patch)
    return jax.jit(lambda s: tick(s, params, TICK_MS, None, None)), params


def _base_state():
    return init_lane(load_patch(), include_all_turrets=False)


def _swing_landing_this_tick(s, level: int):
    """Blue champion mid-windup, one tick from landing a hit on a full-health,
    zero-armour melee minion standing in range -- so the WHOLE of the damage
    this tick is the champion's own attack damage, unmitigated, and
    attributable to nothing else (no cooldown gate, no missile travel time,
    no armour term).
    """
    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    x[BLUE_CHAMP], y[BLUE_CHAMP] = 5000.0, 5000.0
    x[RED_CHAMP], y[RED_CHAMP] = 0.0, 30000.0   # out of the way

    kind = np.asarray(s.kind).copy()
    team = np.asarray(s.team).copy()
    alive = np.asarray(s.alive).copy()
    hp = np.asarray(s.hp).copy()
    max_hp = np.asarray(s.max_hp).copy()
    model = np.asarray(s.model).copy()
    kind[VICTIM] = Kind.LANE_MINION
    team[VICTIM] = Team.RED
    alive[VICTIM] = True
    model[VICTIM] = profile_id(Kind.LANE_MINION, MinionType.MELEE, Team.RED)
    x[VICTIM], y[VICTIM] = 5060.0, 5000.0   # well inside Garen's attack range
    hp[VICTIM] = max_hp[VICTIM] = 10_000.0  # will not die mid-test

    target = np.asarray(s.target).copy()
    is_attacking = np.asarray(s.is_attacking).copy()
    aa_windup = np.asarray(s.aa_windup).copy()
    aa_cooldown = np.asarray(s.aa_cooldown).copy()
    has_aa = np.asarray(s.has_auto_attacked).copy()
    lvl = np.asarray(s.level).copy()

    target[BLUE_CHAMP] = VICTIM
    is_attacking[BLUE_CHAMP] = True
    aa_windup[BLUE_CHAMP] = 1e-6     # lands on THIS tick's decrement
    aa_cooldown[BLUE_CHAMP] = 0.0
    has_aa[BLUE_CHAMP] = False
    lvl[BLUE_CHAMP] = level

    return s.replace(
        x=jnp.asarray(x), y=jnp.asarray(y), kind=jnp.asarray(kind),
        team=jnp.asarray(team), alive=jnp.asarray(alive), hp=jnp.asarray(hp),
        max_hp=jnp.asarray(max_hp), model=jnp.asarray(model),
        target=jnp.asarray(target), is_attacking=jnp.asarray(is_attacking),
        aa_windup=jnp.asarray(aa_windup), aa_cooldown=jnp.asarray(aa_cooldown),
        has_auto_attacked=jnp.asarray(has_aa), level=jnp.asarray(lvl),
    )


def _damage_dealt_at_level(level: int) -> float:
    s = _swing_landing_this_tick(_base_state(), level)
    hp_before = float(np.asarray(s.hp)[VICTIM])
    step, _ = _ticker()
    s2 = step(s)
    hp_after = float(np.asarray(s2.hp)[VICTIM])
    return hp_before - hp_after


def test_champion_ad_at_level_1_matches_the_flat_baseline():
    """Control: level 1 must reproduce the old, still-correct number -- the
    fix must not have shifted the baseline everything else in this project was
    already checked against (``sim/init.py``'s ``RUNE_AD_BONUS`` measurement).
    """
    patch = load_patch()
    expected = patch.champion.base_ad + RUNE_AD_BONUS
    got = _damage_dealt_at_level(1)
    assert got == pytest.approx(expected, abs=0.05)


def test_champion_ad_grows_with_level_the_way_stats_levelup_does():
    """The fix. Before it, this failed: level 9 dealt exactly the level-1
    number, because ``tick()`` never read ``state.level`` for AD at all.
    """
    patch = load_patch()
    ad = patch.champion.base_ad
    # `STAT-002`: the server's slope is NOT Content's `DamagePerLevel` alone.
    # `Brute Force` writes `AttackDamagePerLevel.FlatBonus = 0.55`, and
    # `Stats.LevelUp` grows `AttackDamage` through that term as well as
    # `.BaseValue` (`Stats.cs:270-271`). This test used Content's 3.5 and
    # passed anyway while the sim ran 2.67 AD light at level 7, because it
    # rebuilt the expectation from the same wrong number the sim used --
    # the dumped ladder in `test_lane.py` is what actually pins this.
    per_level = patch.champion.ad_per_level + MASTERY_AD_PER_LEVEL_BONUS
    assert per_level > 0, "the test is meaningless if Garen's AD curve is flat"

    dmg_1 = _damage_dealt_at_level(1)
    dmg_9 = _damage_dealt_at_level(9)

    expected_9 = ad + per_level * growth_sum(9) + RUNE_AD_BONUS
    assert dmg_9 == pytest.approx(expected_9, abs=0.05)
    assert dmg_9 > dmg_1 + 10.0, (
        f"level 9 AD ({dmg_9:.2f}) should be well above level 1's "
        f"({dmg_1:.2f}) -- Stats.LevelUp grows AttackDamage every level"
    )


def test_minion_and_turret_ad_do_not_move_with_the_level_field():
    """``ad_per_level`` is 0 for every non-champion profile row
    (``profiles.py``), so even if a minion's ``level`` field were ever
    misread as something other than the constant 1 every unit starts at
    (``state.py``'s ``empty_state``), its damage must not change with it --
    minions have no level in the server at all.
    """
    s = _base_state()
    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    kind = np.asarray(s.kind).copy()
    team = np.asarray(s.team).copy()
    alive = np.asarray(s.alive).copy()
    hp = np.asarray(s.hp).copy()
    max_hp = np.asarray(s.max_hp).copy()
    model = np.asarray(s.model).copy()
    target = np.asarray(s.target).copy()
    is_attacking = np.asarray(s.is_attacking).copy()
    aa_windup = np.asarray(s.aa_windup).copy()
    aa_cooldown = np.asarray(s.aa_cooldown).copy()
    has_aa = np.asarray(s.has_auto_attacked).copy()
    lvl = np.asarray(s.level).copy()

    ATTACKER = 2
    kind[ATTACKER] = Kind.LANE_MINION
    team[ATTACKER] = Team.BLUE
    alive[ATTACKER] = True
    model[ATTACKER] = profile_id(Kind.LANE_MINION, MinionType.MELEE, Team.BLUE)
    x[ATTACKER], y[ATTACKER] = 5000.0, 5000.0

    kind[VICTIM] = Kind.LANE_MINION
    team[VICTIM] = Team.RED
    alive[VICTIM] = True
    model[VICTIM] = profile_id(Kind.LANE_MINION, MinionType.MELEE, Team.RED)
    x[VICTIM], y[VICTIM] = 5060.0, 5000.0
    hp[VICTIM] = max_hp[VICTIM] = 10_000.0

    target[ATTACKER] = VICTIM
    is_attacking[ATTACKER] = True
    aa_windup[ATTACKER] = 1e-6
    aa_cooldown[ATTACKER] = 0.0
    has_aa[ATTACKER] = False

    base = dict(
        x=jnp.asarray(x), y=jnp.asarray(y), kind=jnp.asarray(kind),
        team=jnp.asarray(team), alive=jnp.asarray(alive), hp=jnp.asarray(hp),
        max_hp=jnp.asarray(max_hp), model=jnp.asarray(model),
        target=jnp.asarray(target), is_attacking=jnp.asarray(is_attacking),
        aa_windup=jnp.asarray(aa_windup), aa_cooldown=jnp.asarray(aa_cooldown),
        has_auto_attacked=jnp.asarray(has_aa),
    )
    step, _ = _ticker()

    lvl1 = lvl.copy()
    lvl1[ATTACKER] = 1
    s1 = s.replace(level=jnp.asarray(lvl1), **base)
    hp_before = float(np.asarray(s1.hp)[VICTIM])
    dmg_lvl1 = hp_before - float(np.asarray(step(s1).hp)[VICTIM])

    lvl9 = lvl.copy()
    lvl9[ATTACKER] = 9
    s9 = s.replace(level=jnp.asarray(lvl9), **base)
    dmg_lvl9 = hp_before - float(np.asarray(step(s9).hp)[VICTIM])

    assert dmg_lvl1 == pytest.approx(dmg_lvl9, abs=1e-4)
