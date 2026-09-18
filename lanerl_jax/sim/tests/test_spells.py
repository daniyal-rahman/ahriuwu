"""Garen's kit: E (Judgment), Q (Decisive Strike), W (Courage), R (Demacian Justice).

`constants.GAREN_SKILL_ORDER` takes E at level 1 and maxes it first, with the
reason stated in the source: *"E first: it is the farming and trading spell."*
E's tests came first for that reason and are the model for everything below.

The kit is exercised both as pure spell/buff formulas and through ``tick``.
In particular, Q's damage and silence are integration behavior: they land
only when the post-skip empowered autoattack connects.
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
                                 RUNE_ARMOR_BONUS, init_lane, lane_params)
from lanerl_jax.sim.orders import OrderKind, Orders, apply_orders
from lanerl_jax.sim.profiles import profile_id
from lanerl_jax.sim.spells import (
    E_COOLDOWNS,
    E_DURATION_S,
    E_MINION_MULTIPLIER,
    E_RADIUS,
    E_TICK_MS,
    Q_BUFF_DURATION,
    Q_BUFF_SLOT,
    Q_COOLDOWN,
    Q_HASTE_BUFF_SLOT,
    R_BASE_PER_RANK,
    R_CAST_TIME_S,
    R_CAST_RANGE,
    R_COOLDOWNS,
    R_MISSING_HP_FRAC,
    R_PENDING_BUFF_SLOT,
    W_BUFF_SLOT,
    W_COOLDOWNS,
    W_DAMAGE_MULT,
    W_PASSIVE_ARMOR_PCT,
    W_PASSIVE_BUFF_SLOT,
    W_PASSIVE_MR_PCT,
    BuffId,
    Slot,
    e_damage_at_rank,
    q_damage_at_rank,
    q_haste_duration_at_rank,
    q_silence_duration_at_rank,
    r_damage_at_rank,
    step_buffs,
    w_duration_at_rank,
)
from lanerl_jax.sim.state import TU_SLICE, Kind, MoveOrder, Team
from lanerl_jax.sim.step import step_decision, tick
from lanerl_jax.sim.targeting import MinionType

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)

GAREN_AD_L1 = 78.134765625        # with the rune page, measured
#: Base Armor/SpellBlock at level 1 (no per-level growth yet) --
#: `Content/LeagueSandbox-Default/Stats/Garen/Garen.json` `Data.Armor` and
#: `Data.SpellBlock`. Deliberately different from each other so a test that
#: mixes them up (e.g. mitigating magic damage against Armor) is caught.
GAREN_ARMOR_L1 = 27.536
GAREN_MR_L1 = 32.1
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
    """Blue Garen well away from any turret, with red minions at ``dist``.

    ``include_all_turrets=False`` for a genuinely empty arena. With the full
    map placed, "well away from any turret" stops being true anywhere useful:
    blue's mid-lane turret sits at (5448, 6169), 673 units from where this
    puts the relocated turret, and it opened fire on it -- which read as the
    spin damaging a turret.
    """
    patch = load_patch()
    s = init_lane(patch, include_all_turrets=False)
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


def _cast_e(s, params=None):
    return apply_orders(s, Orders(
        kind=jnp.asarray([OrderKind.CAST_E, OrderKind.NOOP], jnp.int8),
        x=jnp.zeros(2), y=jnp.zeros(2),
        target=jnp.asarray([-1, -1], jnp.int8)), params)


def _cast_q(s, caster=0):
    kind = [OrderKind.NOOP, OrderKind.NOOP]
    kind[caster] = OrderKind.CAST_Q
    return apply_orders(s, Orders(
        kind=jnp.asarray(kind, jnp.int8), x=jnp.zeros(2), y=jnp.zeros(2),
        target=jnp.asarray([-1, -1], jnp.int8)))


def _cast_w(s, caster=0):
    kind = [OrderKind.NOOP, OrderKind.NOOP]
    kind[caster] = OrderKind.CAST_W
    return apply_orders(s, Orders(
        kind=jnp.asarray(kind, jnp.int8), x=jnp.zeros(2), y=jnp.zeros(2),
        target=jnp.asarray([-1, -1], jnp.int8)))


def _cast_r(s, caster=0, target=1):
    kind = [OrderKind.NOOP, OrderKind.NOOP]
    tgt = [-1, -1]
    kind[caster] = OrderKind.CAST_R
    tgt[caster] = target
    return apply_orders(s, Orders(
        kind=jnp.asarray(kind, jnp.int8), x=jnp.zeros(2), y=jnp.zeros(2),
        target=jnp.asarray(tgt, jnp.int8)))


def _at_level(s, patch, level):
    """Set champion 0's XP so ``tick()`` derives ``level`` (and, from it, spell
    ranks) the next time it runs. Writing ``level``/``spell_level`` directly
    does not stick -- both are recomputed from XP every tick -- so a fixture
    has to go through XP the same way `test_e_ranks_up_as_the_champion_levels`
    does for E.
    """
    return s.replace(xp=s.xp.at[0].set(float(patch.xp_for_level(level)) + 1.0))


def _step_buffs_from(s, params, **overrides):
    """Call :func:`step_buffs` directly against fixture ``s``, bypassing
    ``tick()``/``step_decision`` entirely.

    Used for the parts of Q/W/R that ``tick()`` cannot exercise yet (R's
    magic-resist mitigation -- ``step.py`` does not pass ``magic_resist`` to
    ``step_buffs`` today, see ``spells.py``'s module docstring) and for
    controlled multi-call sequences (e.g. "run 500 ticks with no cast in
    between") that would otherwise need an equally long, slower `_run` loop
    through the whole tick pipeline for no extra fidelity.
    """
    kwargs = dict(
        buff_id=s.buff_id, buff_elapsed=s.buff_elapsed,
        buff_duration=s.buff_duration, buff_power=s.buff_power,
        spell_cooldown=s.spell_cooldown, spell_level=s.spell_level,
        x=s.x, y=s.y, kind=s.kind, team=s.team, alive=s.alive,
        armor=params["armor"][s.model],
        magic_resist=params["magic_resist"][s.model],
        hp=s.hp, max_hp=s.max_hp,
    )
    kwargs.update(overrides)
    return step_buffs(**kwargs)


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


# ------------------------------------------------------------------ Q ------

def test_q_damage_formula_matches_the_buff_script():
    """``30 + 25*(rank-1) + 1.4*AD`` physical -- Characters/Garen/Q.cs:150-152.

    Tested in isolation, like E's damage formula, because nothing in this sim
    can yet trigger the empowered swing that would deal this damage inside a
    real episode (see this file's and ``spells.py``'s module docstrings) --
    the formula is still exactly what the server computes, and a regression
    here would otherwise be invisible until the auto-attack hook lands.
    """
    for rank, want in ((1, 30 + GAREN_AD_L1 * 1.4),
                       (3, 30 + 50.0 + GAREN_AD_L1 * 1.4),
                       (5, 30 + 100.0 + GAREN_AD_L1 * 1.4)):
        got = float(q_damage_at_rank(jnp.int32(rank), jnp.float32(GAREN_AD_L1)))
        assert got == pytest.approx(want, abs=1e-3), f"rank {rank}"


def test_q_silence_and_haste_durations_match_the_script():
    """``1.5 + 0.25*(rank-1)`` (silence, Q.cs:142) and ``1.5 + 0.75*(rank-1)``
    (haste, Q.cs:77) share the same shape and differ only in the rank
    coefficient -- exactly the kind of pair that gets transposed by accident.
    Checked at three ranks each so a swapped 0.25/0.75 would show up as a
    wrong number rather than a coincidentally-matching one at rank 1.
    """
    for rank, want_silence, want_haste in (
            (1, 1.5, 1.5), (3, 2.0, 3.0), (5, 2.5, 4.5)):
        got_silence = float(q_silence_duration_at_rank(jnp.int32(rank)))
        got_haste = float(q_haste_duration_at_rank(jnp.int32(rank)))
        assert got_silence == pytest.approx(want_silence), f"silence rank {rank}"
        assert got_haste == pytest.approx(want_haste), f"haste rank {rank}"


def test_qs_cooldown_is_flat_not_rank_scaled():
    """``GarenQ.json``'s ``Cooldown1``-``Cooldown5`` are all ``"8.0000"``, and
    the buff's own ``OnDeactivate`` hardcodes ``SetCooldown(8)``
    (``Buffs/Garen/GarenQ.cs:98``) rather than reading a rank-indexed table --
    unlike E, W and R, which all scale with rank. Catches whoever "fixes"
    this into a per-rank table because that is what every other spell in the
    kit does.
    """
    assert Q_COOLDOWN == 8.0


def test_casting_q_opens_the_empowerment_and_haste_windows():
    s = _lane_with_minions(e_rank=0)
    s = s.replace(spell_level=s.spell_level.at[0, Slot.Q].set(1))
    s = _cast_q(s)
    assert int(s.buff_id[0, Q_BUFF_SLOT]) == BuffId.GAREN_Q
    assert int(s.buff_id[0, Q_HASTE_BUFF_SLOT]) == BuffId.GAREN_Q_HASTE
    assert float(s.buff_duration[0, Q_BUFF_SLOT]) == pytest.approx(Q_BUFF_DURATION)
    assert float(s.buff_duration[0, Q_HASTE_BUFF_SLOT]) == pytest.approx(1.5)
    # Q.cs:85 -- `spell.SetCooldown(0)` overwrites the engine's default
    # cast-time cooldown in the same event; the real 8s starts only when the
    # window closes (see the timing test below).
    assert float(s.spell_cooldown[0, Slot.Q]) == pytest.approx(0.0)


def test_q_haste_multiplies_the_real_movement_budget():
    """`GarenQHaste.OnActivate` writes +35% MoveSpeed.PercentBonus."""
    params = lane_params(load_patch())
    s = _lane_with_minions(n_minions=0, e_rank=0)
    s = s.replace(
        spell_level=s.spell_level.at[0, Slot.Q].set(1),
        move_order=s.move_order.at[0].set(MoveOrder.MOVE_TO),
        n_waypoints=s.n_waypoints.at[0].set(2),
        waypoint_key=s.waypoint_key.at[0].set(1),
        waypoints=s.waypoints.at[0, 0].set(
            jnp.asarray([6000.0, 6000.0], dtype=s.waypoints.dtype))
                           .at[0, 1].set(
            jnp.asarray([9000.0, 6000.0], dtype=s.waypoints.dtype)),
    )
    s = _cast_q(s)
    before = float(s.x[0])
    s = tick(s, params)
    want = float(params["move_speed"][s.model[0]]) * 1.35 * (1000.0 / 60.0) / 1000.0
    # The movement integrator/state are float32, while this host-side expected
    # value is built in Python float64.
    assert float(s.x[0] - before) == pytest.approx(want, abs=3e-4)


def test_e_snapshots_level_scaled_ad_when_params_are_supplied():
    """Orders are before tick, so E must calculate the same live AD itself."""
    params = lane_params(load_patch())
    s = _lane_with_minions(n_minions=0, e_rank=1)
    s = s.replace(level=s.level.at[0].set(6))
    s = _cast_e(s, params)
    expected_ad = (
        params["attack_damage"][s.model[0]]
        + params["ad_per_level"][s.model[0]] * growth_sum(jnp.int8(6), jnp))
    expected = 10.0 + expected_ad * 0.35
    assert float(s.buff_power[0, 0]) == pytest.approx(float(expected), abs=1e-4)


def test_e_first_buff_update_hits_immediately():
    """`GarenE.TimeSinceLastTick` starts at 500 ms, so the first positive
    `OnUpdate(diff)` deals a tick rather than waiting another half second.
    """
    params = lane_params(load_patch())
    s = _lane_with_minions(n_minions=1, dist=100.0, e_rank=1)
    s = _cast_e(s, params)
    before = float(s.hp[2])
    bs = _step_buffs_from(s, params)
    want = float(e_damage_at_rank(jnp.int32(1), jnp.float32(GAREN_AD_L1)))
    assert float(bs.damage_dealt[2]) == pytest.approx(
        want * E_MINION_MULTIPLIER, abs=1e-3)
    assert float(bs.damage_dealt[2]) > 0
    assert before == pytest.approx(float(s.hp[2]))  # direct buff step is pure


def test_q_cannot_be_recast_while_the_window_is_open():
    """``SealSpellSlot`` (Q.cs:84, unsealed only at ``GarenQ.cs:97``) locks the
    real spell slot for as long as the empowerment window is open, regardless
    of what the cooldown timer reads -- and the timer reads 0 the instant Q is
    cast (previous test), so gating on cooldown alone would wrongly allow a
    recast here. Advances the window partway through first, so a wrongly
    "successful" recast is visible as the elapsed timer snapping back to 0.
    """
    s = _lane_with_minions(e_rank=0)
    s = s.replace(spell_level=s.spell_level.at[0, Slot.Q].set(1))
    s = _cast_q(s)
    s = s.replace(buff_elapsed=s.buff_elapsed.at[0, Q_BUFF_SLOT].set(2.0))
    s2 = _cast_q(s)
    assert float(s2.buff_elapsed[0, Q_BUFF_SLOT]) == pytest.approx(2.0), \
        "a recast while the window is open must not refresh it"


def test_an_unlearned_q_does_nothing():
    s = _lane_with_minions(e_rank=0)
    assert int(_cast_q(s).buff_id[0, Q_BUFF_SLOT]) == BuffId.NONE


def test_qs_cooldown_starts_when_the_window_closes_not_at_cast():
    """Mirrors E's ``test_the_cooldown_starts_when_the_spin_ENDS``: the
    engine's default cast-time cooldown is overwritten to 0 by Q's own script
    (see the two tests above), and the real 8s is set only once the 4.5s
    empowerment window naturally expires (``GarenQ.cs:98``). This fixture
    deliberately has no held target, so it exercises that natural-expiry path;
    the empowered-hit test below covers the earlier close.
    """
    step, _ = _stepper()
    patch = load_patch()
    s = _lane_with_minions()
    s = _at_level(s, patch, 2)      # ranks_for_level(2) == (1, 0, 1, 0): Q=1
    s = step(s)
    assert int(s.spell_level[0, Slot.Q]) == 1
    s = _cast_q(s)
    mid = None
    # Only a few decisions past the window's natural close (unlike E's
    # equivalent test, which waits a full extra second) -- Q's cooldown is
    # 8s against E's 13s, so the same one-second buffer used there would be a
    # 12.5% decay by itself and blow past a 10% tolerance for no reason.
    n_ticks = int(Q_BUFF_DURATION * DECISIONS_PER_S) + 3
    for k in range(n_ticks):
        s = step(s)
        if k == 5:
            mid = float(s.spell_cooldown[0, Slot.Q])
    assert mid == pytest.approx(0.0, abs=1e-3), "cooldown ran during the window"
    assert float(s.spell_cooldown[0, Slot.Q]) == pytest.approx(
        Q_COOLDOWN, rel=0.1)


def test_q_skips_once_then_lands_the_replacement_auto_damage():
    """`GarenQ.OnActivate` cancels then skips one swing; `GarenQAttack`, not
    native `AutoAttackHit`, deals the following swing's complete damage and
    ends the window early. A normal attack at this level would be 81.05
    damage; Q rank 1 is `30 + 1.4 * AD`.
    """
    step, _ = _stepper()
    patch = load_patch()
    s = _lane_with_minions(n_minions=1, dist=60.0, hp=10_000.0, e_rank=0)
    s = _at_level(s, patch, 2)  # Q rank 1 under the fixed skill order
    s = step(s)
    assert int(s.spell_level[0, Slot.Q]) == 1
    s = s.replace(target=s.target.at[0].set(2))
    s = _cast_q(s)
    before = float(s.hp[2])
    hit = None
    for _ in range(90):
        s = step(s)
        if float(s.hp[2]) < before:
            hit = before - float(s.hp[2])
            break
    assert hit is not None, "Q's post-skip empowered swing never landed"
    # `STAT-002`: the server's slope is `DamagePerLevel + Brute Force`, not
    # Content's `DamagePerLevel` alone. Q scales `1.4 * AD`, so reconstructing
    # the expectation from the Content value alone understates it by
    # `1.4 * 0.55 * growth_sum(2)` -- eleven times this assertion's tolerance.
    ad_l2 = (patch.champion.base_ad + RUNE_AD_BONUS
             + (patch.champion.ad_per_level + MASTERY_AD_PER_LEVEL_BONUS)
             * float(growth_sum(2)))
    expected = float(q_damage_at_rank(jnp.int32(1), jnp.float32(ad_l2)))
    assert hit == pytest.approx(expected, abs=0.05)
    assert float(s.silenced_ms[2]) == pytest.approx(1500.0, abs=40.0)
    assert int(s.buff_id[0, Q_BUFF_SLOT]) == BuffId.NONE
    assert float(s.spell_cooldown[0, Slot.Q]) > Q_COOLDOWN - 1.0


# ------------------------------------------------------------------ W ------

def test_w_duration_formula_matches_the_script():
    """``2 + rank - 1`` -- Characters/Garen/W.cs:51. Recomputed the long way
    here (rather than just calling ``w_duration_at_rank`` and trusting it) so
    a typo collapsing it to e.g. ``rank`` or ``rank + 2`` is caught."""
    for rank in range(1, 6):
        want = 2 + rank - 1
        got = float(w_duration_at_rank(jnp.int32(rank)))
        assert got == pytest.approx(want), f"rank {rank}"


def test_w_constants():
    """Pins the three hardcoded numbers from ``Buffs/Garen/GarenW.cs:54`` and
    ``GarenWPassive.cs:34,36`` in one place, so a future source-tracking pass
    has a single spot to update if the patch's numbers ever move."""
    assert W_DAMAGE_MULT == 0.7
    assert W_PASSIVE_ARMOR_PCT == 0.20
    assert W_PASSIVE_MR_PCT == 0.20


def test_casting_w_opens_the_window_and_sets_the_cooldown_immediately():
    """Contrast with Q and E: nothing in ``W.cs``/``GarenW.cs`` overrides the
    engine's default cast-time cooldown (``Spell.cs:1017-1021``), so the full
    rank cooldown is set the instant W is cast, not when the active window
    closes. Getting this backwards (treating W like Q/E) would let W be
    recast the moment its short active window ends, roughly 20s early.
    """
    s = _lane_with_minions(e_rank=0)
    s = s.replace(spell_level=s.spell_level.at[0, Slot.W].set(3))
    s = _cast_w(s)
    assert int(s.buff_id[0, W_BUFF_SLOT]) == BuffId.GAREN_W
    assert float(s.buff_duration[0, W_BUFF_SLOT]) == pytest.approx(4.0)  # 2+3-1
    assert float(s.spell_cooldown[0, Slot.W]) == pytest.approx(W_COOLDOWNS[2])


def test_an_unlearned_w_does_nothing():
    s = _lane_with_minions(e_rank=0)
    out = _cast_w(s)
    assert int(out.buff_id[0, W_BUFF_SLOT]) == BuffId.NONE
    assert float(out.spell_cooldown[0, Slot.W]) == pytest.approx(0.0)


def test_ws_active_damage_reduction_never_reaches_hp_bug_compat():
    """``AttackableUnit.cs:551,558,585,606,612-616``: ``PostMitigationDamage``
    is copied into a stale local BEFORE ``GarenW.cs:54``'s ``PreTakeDamage``
    listener mutates the ``DamageData`` field it reads from -- the real HP
    subtraction (and lifesteal) never see the 0.7x, only a cosmetic
    damage-number packet does. So ``damage_multiplier`` must be
    unconditionally 1.0, **including while the window is genuinely open**
    (checked via ``buff_id`` below, independent of the multiplier) --
    reproducing the server's bug rather than the intended mechanic this sim
    used to implement. Also checks the window still genuinely opens and
    expires on schedule (``buff_id``/``buff_elapsed``, unaffected by this
    fix), the same boundary E's and Q's cooldown tests check.
    """
    patch = load_patch()
    params = lane_params(patch)
    s = _lane_with_minions(e_rank=0)
    s = s.replace(spell_level=s.spell_level.at[0, Slot.W].set(1))  # 2s window
    s = _cast_w(s)
    bs = _step_buffs_from(s, params)
    assert int(s.buff_id[0, W_BUFF_SLOT]) == BuffId.GAREN_W, \
        "the window is genuinely open"
    assert float(bs.damage_multiplier[0]) == pytest.approx(1.0), \
        "no real damage reduction even while W is active -- see the docstring"
    assert float(bs.damage_multiplier[1]) == pytest.approx(1.0)

    s = s.replace(buff_id=bs.buff_id, buff_elapsed=bs.buff_elapsed,
                 spell_cooldown=bs.spell_cooldown)
    for _ in range(int(2.5 * 60)):      # 2.5s of 60Hz ticks, past the 2s window
        bs = _step_buffs_from(s, params)
        s = s.replace(buff_id=bs.buff_id, buff_elapsed=bs.buff_elapsed,
                      spell_cooldown=bs.spell_cooldown)
    assert int(s.buff_id[0, W_BUFF_SLOT]) == BuffId.NONE, \
        "the window still genuinely expires"
    assert float(bs.damage_multiplier[0]) == pytest.approx(1.0)


def test_garenwpassive_is_granted_once_on_rank_up_and_is_permanent():
    """The trap this test exists to catch: ``GarenWPassive`` is granted by
    ``OnLevelUpSpell`` firing on the SPELL OBJECT the moment W's rank first
    becomes 1 (``Characters/Garen/W.cs:26-46``) -- a listener registered from
    champion spawn, independent of ever casting W. The natural-looking
    implementation is to grant it from ``cast_w`` instead, which would leave a
    Garen who puts a point in W at level 3 and never presses it again without
    the permanent +20%/+20% the real server already gives him from that
    level onward. This drives ``step_buffs`` directly, with no cast in
    between, specifically to prove the grant is NOT cast-triggered.

    Also checks it never expires (``infiniteduration``, W.cs:46) across many
    ticks once granted, and that it only ever transitions from absent to
    present, never the reverse.
    """
    patch = load_patch()
    params = lane_params(patch)
    s = _lane_with_minions(e_rank=0)
    assert int(s.spell_level[0, Slot.W]) == 0

    for _ in range(120):    # unranked and never cast: must stay ungranted
        bs = _step_buffs_from(s, params)
        s = s.replace(buff_id=bs.buff_id, buff_elapsed=bs.buff_elapsed)
    assert int(s.buff_id[0, W_PASSIVE_BUFF_SLOT]) == BuffId.NONE

    s = s.replace(spell_level=s.spell_level.at[0, Slot.W].set(1))  # rank-up, no cast
    bs = _step_buffs_from(s, params)
    assert int(bs.buff_id[0, W_PASSIVE_BUFF_SLOT]) == BuffId.GAREN_W_PASSIVE
    # The raw `PercentBaseBonus`/`PercentBonus` pair the server writes
    # (`GarenWPassive.cs:34-37`) -- NOT a single +20% multiplier, see
    # `spells.py`'s W-passive citation for why these compose around
    # `FlatBonus` differently and must stay separate.
    assert float(bs.armor_percent_base_bonus[0]) == pytest.approx(-W_PASSIVE_ARMOR_PCT)
    assert float(bs.armor_percent_bonus[0]) == pytest.approx(W_PASSIVE_ARMOR_PCT)
    assert float(bs.mr_percent_base_bonus[0]) == pytest.approx(-W_PASSIVE_MR_PCT)
    assert float(bs.mr_percent_bonus[0]) == pytest.approx(W_PASSIVE_MR_PCT)
    s = s.replace(buff_id=bs.buff_id, buff_elapsed=bs.buff_elapsed)

    for _ in range(600):    # permanent: survives an arbitrarily long stretch
        bs = _step_buffs_from(s, params)
        s = s.replace(buff_id=bs.buff_id, buff_elapsed=bs.buff_elapsed)
    assert int(s.buff_id[0, W_PASSIVE_BUFF_SLOT]) == BuffId.GAREN_W_PASSIVE
    assert float(bs.armor_percent_bonus[0]) == pytest.approx(W_PASSIVE_ARMOR_PCT)


def test_w_passive_is_committed_on_the_same_tick_as_xp_rank_up():
    """A post-tick observation must not expose rank-one W without its passive.

    XP is the authoritative progression input, so this drives the real
    level/rank derivation rather than manufacturing a W rank directly.  The
    passive is installed by W's level-up listener, not deferred until the
    next ``UpdateBuffs`` call.
    """
    step, _ = _stepper()
    patch = load_patch()
    s = _at_level(_lane_with_minions(e_rank=0), patch, 3)
    assert int(s.spell_level[0, Slot.W]) == 0

    s = step(s)

    assert int(s.spell_level[0, Slot.W]) == 1
    assert int(s.buff_id[0, W_PASSIVE_BUFF_SLOT]) == BuffId.GAREN_W_PASSIVE


def test_w_passive_composes_via_stat_total_through_a_real_autoattack():
    """``GarenWPassive.cs:34-37``: ``Armor.PercentBonus += 0.2;
    Armor.PercentBaseBonus -= 0.2`` composes to
    ``(0.8*(BaseValue+BaseBonus) + FlatBonus) * 1.2`` (``combat.stat_total``),
    NOT ``Total_before * 1.2``. Champion 0's Armor has a nonzero ``FlatBonus``
    here (the rune page, ``RUNE_ARMOR_BONUS = 9.0`` -- see
    ``spells.py``'s W-passive citation for why a rune lands there and not in
    ``BaseValue``/``BaseBonus``), so at level 1 the correct composition is a
    small NET INCREASE (~+1.9%), clearly distinct from both "no passive" and
    from the flat ``*1.2`` bug this replaces (a bigger increase). Verified
    through a REAL auto-attack landing (``step.tick``'s own damage pipeline,
    not an independently re-derived formula), so this exercises exactly the
    code path ``step.py`` changed and fails against the pre-fix flat-percent
    formula.
    """
    from lanerl_jax.sim.combat import post_mitigation_damage, stat_total

    patch = load_patch()
    params = lane_params(patch)
    s = _lane_with_minions(n_minions=0, e_rank=0)
    # champion 1 in melee range of champion 0
    s = s.replace(x=s.x.at[1].set(s.x[0] + 100.0), y=s.y.at[1].set(s.y[0]))
    # grant champion 0's W passive by ranking W -- no cast, matching the real
    # OnLevelUpSpell trigger (see the "granted once on rank-up" test above).
    s = s.replace(spell_level=s.spell_level.at[0, Slot.W].set(1))
    # champion 1 attacks champion 0.
    s = apply_orders(s, Orders(
        kind=jnp.asarray([OrderKind.NOOP, OrderKind.ATTACK], jnp.int8),
        x=jnp.zeros(2), y=jnp.zeros(2), target=jnp.asarray([-1, 0], jnp.int8)))

    hp0 = float(s.hp[0])
    dealt = None
    for i in range(29):        # well under the 30-tick (500ms) regen boundary
        s = tick(s, params)
        hp1 = float(s.hp[0])
        if hp1 < hp0:
            dealt = hp0 - hp1
            break
        hp0 = hp1
    assert dealt is not None, "champion 1's auto-attack should have landed by now"

    raw_ad = GAREN_AD_L1
    armor_before_flat = GAREN_ARMOR_L1                       # BaseValue+BaseBonus
    armor_eff = stat_total(armor_before_flat, 0.0, -W_PASSIVE_ARMOR_PCT,
                           RUNE_ARMOR_BONUS, W_PASSIVE_ARMOR_PCT)
    expect = float(post_mitigation_damage(raw_ad, armor_eff, np))
    old_buggy = float(post_mitigation_damage(
        raw_ad, (armor_before_flat + RUNE_ARMOR_BONUS) * (1.0 + W_PASSIVE_ARMOR_PCT), np))
    assert armor_eff > armor_before_flat + RUNE_ARMOR_BONUS, \
        "at level 1 the real formula is still a net INCREASE, ~+1.9%"
    assert dealt == pytest.approx(expect, abs=0.05)
    assert dealt != pytest.approx(old_buggy, abs=0.05), \
        "must not match the flat *1.2 formula this replaces"


# ------------------------------------------------------------------ R ------

def _lane_for_r(dist=300.0, r_rank=1):
    """Blue and red Garen close enough to be within R's 400-unit cast range
    by default; ``dist`` overrides that for the range-gating test."""
    s = _lane_with_minions(e_rank=0)
    s = s.replace(spell_level=s.spell_level.at[0, Slot.R].set(r_rank))
    s = s.replace(x=s.x.at[1].set(s.x[0] + dist), y=s.y.at[1].set(s.y[0]))
    return s


def test_r_damage_formula_and_missing_hp_scaling():
    """``175*rank + missingHpFrac[rank]*missingHP`` -- Characters/Garen/R.cs:28-29.
    Checked at each of R's three ranks (``R_MISSING_HP_FRAC``), and that more
    missing HP means more damage at a fixed rank -- the execute-scaling half
    of the kit, and the half most likely to get the wrong sign if the missing-
    HP term were accidentally subtracted instead of added.
    """
    for rank, frac in zip((1, 2, 3), R_MISSING_HP_FRAC):
        for missing in (0.0, 300.0, 900.0):
            want = R_BASE_PER_RANK * rank + frac * missing
            got = float(r_damage_at_rank(jnp.int32(rank), jnp.float32(missing)))
            assert got == pytest.approx(want, abs=1e-2), \
                f"rank {rank} missing {missing}"
    lo = float(r_damage_at_rank(jnp.int32(1), jnp.float32(100.0)))
    hi = float(r_damage_at_rank(jnp.int32(1), jnp.float32(900.0)))
    assert hi > lo


def test_r_cooldowns():
    assert R_COOLDOWNS == (160.0, 120.0, 80.0)


def test_r_can_only_target_the_enemy_champion():
    """``GarenR.json``'s ``TextFlags`` is ``AffectEnemies | AffectHeroes``
    with none of ``AffectMinions``/``AffectTurrets``/``AffectBuildings``/
    ``AffectNeutral`` -- unlike Q, whose ``TextFlags`` has all four. There is
    also no ``AffectFriends``, so a self-target (the only "ally" that exists
    here) must fail exactly like a minion target does.
    """
    s = _lane_for_r()
    minion_idx = 2       # first red minion from _lane_with_minions
    assert int(_cast_r(s, 0, minion_idx).buff_id[minion_idx, R_PENDING_BUFF_SLOT]) \
        == BuffId.NONE
    assert int(_cast_r(s, 0, 0).buff_id[0, R_PENDING_BUFF_SLOT]) == BuffId.NONE
    assert int(_cast_r(s, 0, 1).buff_id[1, R_PENDING_BUFF_SLOT]) \
        == BuffId.GAREN_R_PENDING


def test_r_rejects_an_already_dead_enemy_before_starting_its_cast():
    """Spell target validation happens at cast ingress.  This differs from a
    target dying *during* GarenR's uncancellable 0.435-second windup, which
    still lets the caster finish and starts its cooldown.
    """
    s = _lane_for_r()
    s = s.replace(alive=s.alive.at[1].set(False))
    out = _cast_r(s, caster=0, target=1)
    assert int(out.buff_id[1, R_PENDING_BUFF_SLOT]) == BuffId.NONE
    assert float(out.r_cast_ms[0]) == pytest.approx(0.0)


def test_r_respects_cast_range():
    """``Spells/GarenR/GarenR.json`` ``"CastRange": "400.0000"`` -- an
    engine-level ``SpellData`` targeting rule rather than a content-script
    one, but sourced from the same JSON as the cooldown table and applied the
    same way a client would refuse to send the order out of range.
    """
    s_far = _lane_for_r(dist=R_CAST_RANGE + 50.0)
    assert int(_cast_r(s_far, 0, 1).buff_id[1, R_PENDING_BUFF_SLOT]) == BuffId.NONE
    s_near = _lane_for_r(dist=R_CAST_RANGE - 50.0)
    assert int(_cast_r(s_near, 0, 1).buff_id[1, R_PENDING_BUFF_SLOT]) \
        == BuffId.GAREN_R_PENDING


def test_an_unlearned_r_does_nothing():
    s = _lane_for_r(r_rank=0)
    assert int(_cast_r(s, 0, 1).buff_id[1, R_PENDING_BUFF_SLOT]) == BuffId.NONE


def test_rs_damage_is_magical_not_physical():
    """``Characters/Garen/R.cs:33`` applies ``DAMAGE_TYPE_MAGICAL`` -- worth
    stating plainly because modern-patch League's Demacian Justice is
    physical and this server targets patch 4.20, not modern League. Garen's
    Armor and Magic Resist are deliberately different
    (``GAREN_ARMOR_L1``/``GAREN_MR_L1``, both cited from ``Garen.json``), so
    mitigating against the wrong stat fails this test instead of passing it
    by coincidence.
    """
    patch = load_patch()
    params = lane_params(patch)
    s = _lane_for_r()
    s = _cast_r(s, caster=0, target=1)
    assert int(s.buff_id[1, R_PENDING_BUFF_SLOT]) == BuffId.GAREN_R_PENDING

    armor = params["armor"][s.model]
    magic_resist = params["magic_resist"][s.model]
    assert float(magic_resist[1]) != float(armor[1]), \
        "fixture must actually distinguish the two stats to test this"
    # R is non-instant: no damage on its first buff update, then snapshot the
    # target's health when the engine's 0.435-second cast timer completes.
    bs = _step_buffs_from(s, params, armor=armor, magic_resist=magic_resist)
    assert float(bs.damage_dealt[1]) == pytest.approx(0.0)
    s = s.replace(buff_elapsed=bs.buff_elapsed.at[1, R_PENDING_BUFF_SLOT].set(
        R_CAST_TIME_S))
    bs = _step_buffs_from(s, params, armor=armor, magic_resist=magic_resist)

    missing = float(s.max_hp[1] - s.hp[1])
    raw = float(r_damage_at_rank(jnp.int32(1), jnp.float32(missing)))
    want = raw * (100.0 / (100.0 + float(magic_resist[1])))
    wrong = raw * (100.0 / (100.0 + float(armor[1])))
    assert float(bs.damage_dealt[1]) == pytest.approx(want, rel=1e-4)
    assert float(bs.damage_dealt[1]) != pytest.approx(wrong, rel=1e-4)
    assert int(bs.dealt_by[1]) == 0, "attribution must credit the caster (0)"


def test_r_falls_back_to_armor_only_when_magic_resist_is_not_supplied():
    """The direct helper's optional fallback remains useful to make an
    omitted magic-resist input explicit; production ``step.py`` passes MR.
    """
    patch = load_patch()
    params = lane_params(patch)
    s = _lane_for_r()
    s = _cast_r(s, caster=0, target=1)
    armor = params["armor"][s.model]
    s = s.replace(buff_elapsed=s.buff_elapsed.at[1, R_PENDING_BUFF_SLOT].set(
        R_CAST_TIME_S))
    bs = _step_buffs_from(s, params, magic_resist=None)
    raw = float(r_damage_at_rank(
        jnp.int32(1), jnp.float32(float(s.max_hp[1] - s.hp[1]))))
    want = raw * (100.0 / (100.0 + float(armor[1])))
    assert float(bs.damage_dealt[1]) == pytest.approx(want, rel=1e-4)


def test_rs_cooldown_starts_when_its_noninstant_cast_finishes():
    """R lacks ``InstantCast``, so `Spell.FinishCasting` -- which transitions
    to cooldown -- runs after `(1 - .13) * .5 = .435` seconds, not at order
    ingress. This also pins that a second R order during the windup is denied
    by the simulated casting state.
    """
    params = lane_params(load_patch())
    patch = load_patch()
    s = _lane_for_r()
    s = _at_level(s, patch, 6)      # ranks_for_level(6) == (1, 1, 3, 1): R=1
    s = tick(s, params)
    assert int(s.spell_level[0, Slot.R]) == 1
    s = _cast_r(s, caster=0, target=1)
    assert float(s.spell_cooldown[0, Slot.R]) == pytest.approx(0.0)
    assert int(_cast_r(s, caster=0, target=1).buff_id[1, R_PENDING_BUFF_SLOT]) \
        == BuffId.GAREN_R_PENDING

    # Place the pending R just before its final cast-timer decrement, then
    # execute exactly one buff update. At finish the normal cooldown begins.
    s = s.replace(buff_elapsed=s.buff_elapsed.at[1, R_PENDING_BUFF_SLOT].set(
        R_CAST_TIME_S))
    bs = _step_buffs_from(s, params)
    assert float(bs.spell_cooldown[0, Slot.R]) == pytest.approx(R_COOLDOWNS[0])


def test_r_windup_locks_orders_then_finishes_hold_and_delayed_hit():
    """R is the only non-instant Garen combat spell here. Its engine casting
    state refuses Move/Attack/Q while live, then `FinishCasting` both lands R
    and leaves the owner in Hold with a reset one-point path.
    """
    params = lane_params(load_patch())
    s = _lane_for_r()
    s = s.replace(
        spell_level=s.spell_level.at[0, Slot.Q].set(1),
        aa_cooldown=s.aa_cooldown.at[0].set(0.8),
        aa_windup=s.aa_windup.at[0].set(0.1),
        is_attacking=s.is_attacking.at[0].set(True),
        move_order=s.move_order.at[0].set(MoveOrder.MOVE_TO),
        n_waypoints=s.n_waypoints.at[0].set(2),
        waypoint_key=s.waypoint_key.at[0].set(1),
        waypoints=s.waypoints.at[0, 0].set(
            jnp.asarray([6000.0, 6000.0], dtype=s.waypoints.dtype))
                           .at[0, 1].set(
            jnp.asarray([6500.0, 6000.0], dtype=s.waypoints.dtype)),
    )
    s = _cast_r(s)
    assert float(s.r_cast_ms[0]) == pytest.approx(R_CAST_TIME_S * 1000.0)
    assert float(s.aa_cooldown[0]) == pytest.approx(0.0)
    assert float(s.aa_windup[0]) == pytest.approx(0.0)
    assert not bool(s.is_attacking[0])

    blocked_move = apply_orders(s, Orders(
        kind=jnp.asarray([OrderKind.MOVE, OrderKind.NOOP], jnp.int8),
        x=jnp.asarray([9000.0, 0.0]), y=jnp.zeros(2),
        target=jnp.asarray([-1, -1], jnp.int8)))
    assert int(blocked_move.n_waypoints[0]) == 2
    blocked_attack = apply_orders(s, Orders(
        kind=jnp.asarray([OrderKind.ATTACK, OrderKind.NOOP], jnp.int8),
        x=jnp.zeros(2), y=jnp.zeros(2), target=jnp.asarray([1, -1], jnp.int8)))
    assert int(blocked_attack.target[0]) == -1
    blocked_q = _cast_q(s)
    assert int(blocked_q.buff_id[0, Q_BUFF_SLOT]) == BuffId.NONE

    # Complete both fixed-shape timer representations on one simulation tick.
    s = s.replace(
        r_cast_ms=s.r_cast_ms.at[0].set(1.0),
        buff_elapsed=s.buff_elapsed.at[1, R_PENDING_BUFF_SLOT].set(R_CAST_TIME_S),
    )
    hp_before = float(s.hp[1])
    s = tick(s, params)
    assert float(s.r_cast_ms[0]) == pytest.approx(0.0)
    assert int(s.move_order[0]) == MoveOrder.HOLD
    assert int(s.n_waypoints[0]) == 1 and int(s.waypoint_key[0]) == 1
    assert float(s.hp[1]) < hp_before
    assert float(s.spell_cooldown[0, Slot.R]) == pytest.approx(R_COOLDOWNS[0])


def test_r_cast_cancels_on_caster_death_without_starting_cooldown():
    """Generic `CastCancelCheck` resets an R whose owner dies mid-windup.
    The target-side pending mailbox is purged on the following buff update.
    """
    params = lane_params(load_patch())
    s = _cast_r(_lane_for_r())
    s = s.replace(hp=s.hp.at[0].set(0.0))
    s = tick(s, params)       # establishes caster death and clears cast lock
    assert not bool(s.alive[0])
    assert float(s.r_cast_ms[0]) == pytest.approx(0.0)
    s = tick(s, params)       # target mailbox observes dead caster and cancels
    assert int(s.buff_id[1, R_PENDING_BUFF_SLOT]) == BuffId.NONE
    assert float(s.spell_cooldown[0, Slot.R]) == pytest.approx(0.0)


def test_death_clears_owner_q_empowerment_and_haste():
    """A dead Garen must not resume Q/haste after respawn."""
    params = lane_params(load_patch())
    s = _lane_with_minions(n_minions=0, e_rank=0)
    s = s.replace(spell_level=s.spell_level.at[0, Slot.Q].set(1))
    s = _cast_q(s)
    assert int(s.buff_id[0, Q_BUFF_SLOT]) == BuffId.GAREN_Q
    s = s.replace(hp=s.hp.at[0].set(0.0))
    out = tick(s, params)
    assert not bool(out.alive[0])
    assert int(out.buff_id[0, Q_BUFF_SLOT]) == BuffId.NONE
    assert int(out.buff_id[0, Q_HASTE_BUFF_SLOT]) == BuffId.NONE
