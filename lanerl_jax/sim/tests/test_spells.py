"""Garen's kit: E (Judgment), Q (Decisive Strike), W (Courage), R (Demacian Justice).

`constants.GAREN_SKILL_ORDER` takes E at level 1 and maxes it first, with the
reason stated in the source: *"E first: it is the farming and trading spell."*
E's tests came first for that reason and are the model for everything below.

Q, W and R each hit a real wall described in ``spells.py``'s module docstring:
Q's damage/silence only land when the caster's *next auto-attack* connects (an
``autoattack.py``/``step.py`` event this module cannot see), and W's 0.7x
damage multiplier and permanent Armor/MR passive only take effect once
``step.py`` folds the values ``step_buffs`` now computes into its own damage
and mitigation math. Tests below that depend on either are written against
the level this project actually implements them at -- the pure formulas, the
buff/cooldown bookkeeping, and (for W) the *values* step.py must consume --
not against a full in-game hit that cannot happen yet. Each says so.
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
    Q_BUFF_DURATION,
    Q_BUFF_SLOT,
    Q_COOLDOWN,
    Q_HASTE_BUFF_SLOT,
    R_BASE_PER_RANK,
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
from lanerl_jax.sim.state import TU_SLICE, Kind, Team
from lanerl_jax.sim.step import step_decision
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


def _cast_e(s):
    return apply_orders(s, Orders(
        kind=jnp.asarray([OrderKind.CAST_E, OrderKind.NOOP], jnp.int8),
        x=jnp.zeros(2), y=jnp.zeros(2),
        target=jnp.asarray([-1, -1], jnp.int8)))


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
    empowerment window naturally expires (``GarenQ.cs:98``). Since this sim
    cannot yet see an early-landing empowered swing (the module docstring's Q
    section explains why), a full window is also the ONLY way it closes here
    -- so the lockout measured below (``Q_BUFF_DURATION + Q_COOLDOWN``) is an
    upper bound on the real server's, not the number the server would also
    produce if the swing connects early.
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


def test_ws_multiplier_applies_regardless_of_damage_source_and_expires_on_time():
    """Exercises ``step_buffs``'s ``damage_multiplier`` output directly, since
    nothing in ``step.py`` multiplies it into a unit's damage total yet (see
    this file's and ``spells.py``'s module docstrings) -- this pins the
    CONTRACT value ``step.py`` must consume. ``GarenW.cs:54``'s
    ``PreTakeDamage`` has no attacker-type or damage-type filter at all, which
    is exactly why a single per-unit scalar (rather than one multiplier per
    damage source) is the right shape: wiring it once in ``step.py`` covers
    auto-attacks, missiles and turret shots alike, with no separate hook per
    source needed. Also checks it turns back off the tick the window expires,
    the same boundary E's and Q's cooldown tests check.
    """
    patch = load_patch()
    params = lane_params(patch)
    s = _lane_with_minions(e_rank=0)
    s = s.replace(spell_level=s.spell_level.at[0, Slot.W].set(1))  # 2s window
    s = _cast_w(s)
    bs = _step_buffs_from(s, params)
    assert float(bs.damage_multiplier[0]) == pytest.approx(W_DAMAGE_MULT)
    assert float(bs.damage_multiplier[1]) == pytest.approx(1.0), \
        "a unit without the buff must not be discounted"

    s = s.replace(buff_id=bs.buff_id, buff_elapsed=bs.buff_elapsed,
                 spell_cooldown=bs.spell_cooldown)
    for _ in range(int(2.5 * 60)):      # 2.5s of 60Hz ticks, past the 2s window
        bs = _step_buffs_from(s, params)
        s = s.replace(buff_id=bs.buff_id, buff_elapsed=bs.buff_elapsed,
                      spell_cooldown=bs.spell_cooldown)
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
    assert float(bs.armor_pct_bonus[0]) == pytest.approx(W_PASSIVE_ARMOR_PCT)
    assert float(bs.mr_pct_bonus[0]) == pytest.approx(W_PASSIVE_MR_PCT)
    s = s.replace(buff_id=bs.buff_id, buff_elapsed=bs.buff_elapsed)

    for _ in range(600):    # permanent: survives an arbitrarily long stretch
        bs = _step_buffs_from(s, params)
        s = s.replace(buff_id=bs.buff_id, buff_elapsed=bs.buff_elapsed)
    assert int(s.buff_id[0, W_PASSIVE_BUFF_SLOT]) == BuffId.GAREN_W_PASSIVE
    assert float(bs.armor_pct_bonus[0]) == pytest.approx(W_PASSIVE_ARMOR_PCT)


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
    bs = _step_buffs_from(s, params, armor=armor, magic_resist=magic_resist)

    raw = float(s.buff_power[1, R_PENDING_BUFF_SLOT])
    want = raw * (100.0 / (100.0 + float(magic_resist[1])))
    wrong = raw * (100.0 / (100.0 + float(armor[1])))
    assert float(bs.damage_dealt[1]) == pytest.approx(want, rel=1e-4)
    assert float(bs.damage_dealt[1]) != pytest.approx(wrong, rel=1e-4)
    assert int(bs.dealt_by[1]) == 0, "attribution must credit the caster (0)"


def test_r_falls_back_to_armor_only_when_magic_resist_is_not_supplied():
    """Pins the documented stopgap in ``step_buffs``: today's ``step.py``
    call site does not pass ``magic_resist`` (see ``spells.py``'s module
    docstring), so until it does, R's mitigation silently reuses ``armor``.
    This is the one test allowed to rely on that fallback -- every other test
    above passes ``magic_resist`` explicitly so a bug in the real path is not
    masked by it.
    """
    patch = load_patch()
    params = lane_params(patch)
    s = _lane_for_r()
    s = _cast_r(s, caster=0, target=1)
    armor = params["armor"][s.model]
    bs = _step_buffs_from(s, params, magic_resist=None)
    raw = float(s.buff_power[1, R_PENDING_BUFF_SLOT])
    want = raw * (100.0 / (100.0 + float(armor[1])))
    assert float(bs.damage_dealt[1]) == pytest.approx(want, rel=1e-4)


def test_rs_cooldown_starts_at_cast_like_w_not_like_q_or_e():
    """No ``SetCooldown`` call anywhere in ``R.cs``: like W, this is the
    engine's unmodified default (``Spell.cs:1017-1021``), starting the
    instant R is cast -- there is no window to wait for, since R has none.
    """
    step, _ = _stepper()
    patch = load_patch()
    s = _lane_for_r()
    s = _at_level(s, patch, 6)      # ranks_for_level(6) == (1, 1, 3, 1): R=1
    s = step(s)
    assert int(s.spell_level[0, Slot.R]) == 1
    s = _cast_r(s, caster=0, target=1)
    assert float(s.spell_cooldown[0, Slot.R]) == pytest.approx(R_COOLDOWNS[0])
