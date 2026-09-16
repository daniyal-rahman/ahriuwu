"""Garen's E (Judgment) -- the farming and trading spell.

`constants.GAREN_SKILL_ORDER` takes it at level 1 and maxes it first, with the
reason stated in the source: *"E first: it is the farming and trading spell."*
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch
from lanerl_jax.sim.init import init_lane, lane_params
from lanerl_jax.sim.orders import OrderKind, Orders, apply_orders
from lanerl_jax.sim.profiles import profile_id
from lanerl_jax.sim.spells import (
    E_COOLDOWNS,
    E_DURATION_S,
    E_MINION_MULTIPLIER,
    E_RADIUS,
    E_TICK_MS,
    BuffId,
    Slot,
    e_damage_at_rank,
)
from lanerl_jax.sim.state import TU_SLICE, Kind, Team
from lanerl_jax.sim.step import step_decision
from lanerl_jax.sim.targeting import MinionType

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)

GAREN_AD_L1 = 78.134765625        # with the rune page, measured
DECISIONS_PER_S = 30


@functools.lru_cache(maxsize=1)
def _stepper():
    """One jitted step, reused across every test in this file.

    Calling ``step_decision`` straight from a Python loop re-traces on every
    call; the first version of this file did that and took minutes instead of
    seconds. The params are identical for all of these tests, so one cached
    closure serves them all.
    """
    params = lane_params(load_patch())
    return jax.jit(lambda st: step_decision(st, params)), params


def _run(s, n):
    step, _ = _stepper()
    for _ in range(n):
        s = step(s)
    return s


def _lane_with_minions(n_minions=4, dist=200.0, hp=455.0, e_rank=1):
    """Blue Garen well away from any turret, with red minions at ``dist``."""
    patch = load_patch()
    s = init_lane(patch)
    kind = np.asarray(s.kind).copy()
    team = np.asarray(s.team).copy()
    alive = np.asarray(s.alive).copy()
    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    hpv = np.asarray(s.hp).copy()
    mx = np.asarray(s.max_hp).copy()
    model = np.asarray(s.model).copy()
    x[0], y[0] = 6000.0, 6000.0           # nowhere near a turret
    x[1], y[1] = 6000.0, 12000.0          # the other champion, parked
    for j in range(n_minions):
        i = 2 + j
        kind[i] = Kind.LANE_MINION
        team[i] = Team.RED
        alive[i] = True
        model[i] = profile_id(Kind.LANE_MINION, MinionType.MELEE, Team.RED)
        x[i], y[i] = 6000.0 + dist, 6000.0 + j * 10.0
        hpv[i] = mx[i] = hp
    lvl = np.asarray(s.spell_level).copy()
    lvl[0, Slot.E] = e_rank
    return s.replace(kind=jnp.asarray(kind), team=jnp.asarray(team),
                     alive=jnp.asarray(alive), x=jnp.asarray(x),
                     y=jnp.asarray(y), hp=jnp.asarray(hpv),
                     max_hp=jnp.asarray(mx), model=jnp.asarray(model),
                     spell_level=jnp.asarray(lvl))


def _cast_e(s):
    return apply_orders(s, Orders(
        kind=jnp.asarray([OrderKind.CAST_E, OrderKind.NOOP], jnp.int8),
        x=jnp.zeros(2), y=jnp.zeros(2),
        target=jnp.asarray([-1, -1], jnp.int8)))


def test_damage_formula_matches_the_buff_script():
    """``10 + 12.5*(rank-1) + AD*(0.35 + 0.05*(rank-1))``, snapshotted at cast."""
    for rank, want in ((1, 10 + GAREN_AD_L1 * 0.35),
                       (3, 10 + 25.0 + GAREN_AD_L1 * 0.45),
                       (5, 10 + 50.0 + GAREN_AD_L1 * 0.55)):
        got = float(e_damage_at_rank(jnp.int32(rank), jnp.float32(GAREN_AD_L1)))
        assert got == pytest.approx(want, abs=1e-3), f"rank {rank}"


def test_casting_starts_the_spin():
    s = _cast_e(_lane_with_minions())
    assert int(s.buff_id[0, 0]) == BuffId.GAREN_E
    assert float(s.buff_duration[0, 0]) == pytest.approx(E_DURATION_S)


def test_spinning_damages_nearby_enemies():
    s = _lane_with_minions(dist=200.0)
    before = float(s.hp[2])
    s = _run(_cast_e(s), int(E_DURATION_S * DECISIONS_PER_S) + 5)
    after = float(s.hp[2])
    assert after < before, "the spin dealt no damage"
    # six 500 ms ticks over 3 s, minions at 0.75x, melee minions have 0 armour
    per_tick = float(e_damage_at_rank(jnp.int32(1), jnp.float32(GAREN_AD_L1)))
    assert before - after == pytest.approx(
        per_tick * E_MINION_MULTIPLIER * 6, rel=0.25)


def test_enemies_outside_the_radius_are_untouched():
    """Checked at a fixed instant, because the minions close the gap.

    The first version parked a minion at 450 units, spun for 3 s and asserted
    no damage. It took 168 -- because red minions acquire the blue champion at
    600 and **walk into him**. That is the sim being right and the test being
    wrong. So: start far enough that the minion is still outside 330 when the
    first 500 ms spin tick lands (minions cover ~172 units in that time), and
    assert at exactly that point.
    """
    # Beyond the minion's 600 acquisition range, so it has no reason to close
    # (these are hand-placed and carry no lane waypoints, so nothing else moves
    # them either).
    s = _lane_with_minions(dist=700.0)
    before = float(s.hp[2])
    s = _run(_cast_e(s), int(E_DURATION_S * DECISIONS_PER_S) + 5)
    d = float(((s.x[2] - s.x[0]) ** 2 + (s.y[2] - s.y[0]) ** 2) ** 0.5)
    assert d > E_RADIUS, f"the minion closed to {d:.0f}"
    assert float(s.hp[2]) == pytest.approx(before)


def test_an_enemy_that_walks_INTO_the_radius_starts_taking_damage():
    """The complement of the test above, and the reason it had to be written
    that way: the spin is centred on a moving caster and re-evaluated every
    tick, so range is not decided once at cast."""
    # Inside the minion's 600 acquisition range: it targets the champion and
    # walks to its own attack range (110 + 40 = 150), well inside the spin.
    s = _lane_with_minions(dist=450.0)
    before = float(s.hp[2])
    s = _run(_cast_e(s), int(E_DURATION_S * DECISIONS_PER_S))
    d = float(((s.x[2] - s.x[0]) ** 2 + (s.y[2] - s.y[0]) ** 2) ** 0.5)
    assert d < E_RADIUS, f"the minion never closed (d={d:.0f})"
    assert float(s.hp[2]) < before


def test_minions_take_three_quarters():
    """Without the 0.75x, E clears waves far too fast -- and wave clear is
    exactly the mechanic the agent is being trained to use."""
    assert E_MINION_MULTIPLIER == 0.75


def test_autoattacks_are_suppressed_while_spinning():
    """``SetStatus(CanAttack, false)`` for the duration."""
    step, _ = _stepper()
    s = _cast_e(_lane_with_minions(dist=150.0))
    s = s.replace(target=s.target.at[0].set(2))
    saw_attack = False
    for _ in range(int(E_DURATION_S * DECISIONS_PER_S) - 10):
        s = step(s)
        saw_attack |= bool(s.is_attacking[0])
    assert not saw_attack, "a spinning Garen must not auto-attack"


def test_the_cooldown_starts_when_the_spin_ENDS():
    """``OnDeactivate`` sets it, so it runs from the end and not from the cast."""
    step, _ = _stepper()
    s = _cast_e(_lane_with_minions())
    mid = None
    for k in range(int(E_DURATION_S * DECISIONS_PER_S) + 30):
        s = step(s)
        if k == 20:
            mid = float(s.spell_cooldown[0, Slot.E])
    assert mid == pytest.approx(0.0, abs=1e-3), "cooldown ran during the spin"
    assert float(s.spell_cooldown[0, Slot.E]) == pytest.approx(
        E_COOLDOWNS[0], rel=0.1), "rank-1 cooldown is 13 s"


def test_e_cannot_be_recast_while_on_cooldown():
    s = _run(_cast_e(_lane_with_minions()),
             int(E_DURATION_S * DECISIONS_PER_S) + 10)
    assert int(s.buff_id[0, 0]) == BuffId.NONE
    assert int(_cast_e(s).buff_id[0, 0]) == BuffId.NONE


def test_an_unlearned_e_does_nothing_here():
    """The SERVER would grant the effect -- ``Spell.Cast`` never checks the
    level, which `constants.py` flags as the reason the action mask matters.
    The sim gates on rank instead, so a masking bug upstream fails loudly rather
    than silently handing out a free spell."""
    s = _lane_with_minions(e_rank=0)
    assert int(_cast_e(s).buff_id[0, 0]) == BuffId.NONE


def test_turrets_are_immune_to_the_spin():
    s = _lane_with_minions()
    ti = TU_SLICE.start
    s = s.replace(x=s.x.at[ti].set(6100.0), y=s.y.at[ti].set(6000.0),
                  team=s.team.at[ti].set(Team.RED))
    before = float(s.hp[ti])
    s = _run(_cast_e(s), int(E_DURATION_S * DECISIONS_PER_S) + 5)
    assert float(s.hp[ti]) == pytest.approx(before)


def test_tick_cadence_and_duration():
    assert E_TICK_MS == 500.0 and E_DURATION_S == 3.0 and E_RADIUS == 330.0


# ------------------------------------------------------------ skill order --
def test_skill_order_matches_lanerl_rl_constants():
    """The canonical copy lives in `lanerl_rl/constants.py`; this mirrors it.

    That file records why it matters: the order previously existed in **three**
    places that disagreed (`LanerlConfig.SkillOrder`, `lanerl_bot.build`, and
    `obs.AbilityBook`, which encoded a fourth again). A skill order that differs
    between the server and the observation is not cosmetic -- the action mask
    then forbids a spell the champion HAS and offers one it does not, and
    casting an unlearned spell is **not** a no-op on the server, because nothing
    in `Spell.Cast` checks the level.

    Mirrored rather than imported so the sim has no import-time dependency on
    the torch-bearing package. This test is the thing that stops them drifting.
    """
    from lanerl_jax.sim.spells import SKILL_ORDER
    from lanerl_rl import constants as C

    assert SKILL_ORDER == C.GAREN_SKILL_ORDER


@pytest.mark.parametrize("level,want", [
    (1, (0, 0, 1, 0)),      # E first: the farming and trading spell
    (2, (1, 0, 1, 0)),
    (6, (1, 1, 3, 1)),      # R at 6
    (9, (2, 1, 5, 1)),      # E maxed at 9
    (13, (5, 1, 5, 2)),
    (18, (5, 5, 5, 3)),
])
def test_ranks_at_level(level, want):
    from lanerl_jax.sim.spells import ranks_for_level

    assert ranks_for_level(level) == want


def test_r_is_capped_at_three_ranks():
    """``Champion.LevelUpSpell`` caps R. The skill order never exceeds it, but
    the cap is applied rather than assumed -- `constants.py` notes that
    `GAREN_COOLDOWNS["R"]` carries padding entries for ranks 4-5 that do not
    match the JSON's own filler, precisely because they are unreachable."""
    from lanerl_jax.sim.spells import ranks_for_level

    assert ranks_for_level(18)[3] == 3


def test_e_ranks_up_as_the_champion_levels():
    """End-to-end: rank follows level, which follows XP, with no state of its own.

    Note the fixture sets **XP**, not level. Writing `level` directly does not
    stick, because the tick recomputes it from XP every step -- the first
    version of this test set level=9 with 2500 XP and got level 6 back. That is
    the right behaviour (one source of truth) and a fixture that fights it is
    testing nothing.
    """
    from lanerl_jax.sim.spells import RANKS_BY_LEVEL, Slot

    step, _ = _stepper()
    patch = load_patch()
    s = _lane_with_minions()
    s = s.replace(xp=s.xp.at[0].set(float(patch.xp_for_level(9)) + 1.0))
    s = step(s)
    assert int(s.level[0]) == 9
    assert int(s.spell_level[0, Slot.E]) == RANKS_BY_LEVEL[9][Slot.E]
    assert int(s.spell_level[0, Slot.E]) == 5, "E is maxed by level 9"
