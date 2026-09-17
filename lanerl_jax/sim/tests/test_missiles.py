"""Basic-attack missiles -- the mechanic that makes a ranged minion's damage
conditional on its target surviving the flight.

See ``sim/missiles.py``'s docstring for the exact server rule this ports:
``!IsMelee`` and an empty ``BasicAttack`` script both have to hold before a
swing becomes a missile instead of instant damage, and in this slice the
second half is never false -- including for a lane turret, which looked like
the exception until the turret model itself turned out to be wrong (see
``test_a_lane_turrets_damage_also_lands_after_flight``). A missile whose
target dies or leaves range in flight is removed with **no damage** -- not
damage redirected, not damage banked, none; a naive "instant damage, just
delayed" model would still pay out on a target that died mid-flight, which is
the load-bearing test below.
"""
from __future__ import annotations

import functools

import jax
import numpy as np
import pytest

import jax.numpy as jnp

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch
from lanerl_jax.sim.init import ALL_TURRETS, TOP_LANE_PATH, init_lane, lane_params
from lanerl_jax.sim.profiles import build_profile_tables, profile_id
from lanerl_jax.sim.state import (Kind, MoveOrder, N_MISSILES, TU_SLICE, Team,
                                  TurretTier)
from lanerl_jax.sim.step import step_decision, tick
from lanerl_jax.sim.movement_jax import TICK_MS
from lanerl_jax.sim.targeting import MinionType

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)


@functools.lru_cache(maxsize=1)
def _ticker():
    """One jitted ``tick``, reused across every test in this file.

    Calling ``tick`` straight from a Python loop re-traces on every call; see
    ``test_spells.py``'s ``_stepper`` for the same fix applied there. No wave
    spawning (``lane_path=None``) -- every test here places its own two units.
    """
    patch = load_patch()
    params = lane_params(patch)
    return jax.jit(lambda s: tick(s, params, TICK_MS, None, None)), params


def _n_missiles(s) -> int:
    return int(np.asarray(s.missile_alive).sum())


def _two_minions(a_type, a_team, a_xy, b_type, b_team, b_xy, hp_a=1000.0, hp_b=1000.0):
    """Exactly two lane minions, alone in ``init_lane``'s isolated arena.

    The two champions stay wherever ``init_lane`` spawns them, tens of
    thousands of units from anything placed near ``(5000, 5000)``, which is
    well outside any acquisition range -- so they never need to be moved out
    of the way by hand.
    """
    patch = load_patch()
    s = init_lane(patch, include_all_turrets=False)
    kind = np.asarray(s.kind).copy()
    team = np.asarray(s.team).copy()
    alive = np.asarray(s.alive).copy()
    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    hp = np.asarray(s.hp).copy()
    mhp = np.asarray(s.max_hp).copy()
    model = np.asarray(s.model).copy()

    A, B = 2, 3
    for i, (mtype, tm, (px, py), hpv) in enumerate(
        ((a_type, a_team, a_xy, hp_a), (b_type, b_team, b_xy, hp_b))
    ):
        idx = A if i == 0 else B
        kind[idx] = Kind.LANE_MINION
        team[idx] = tm
        alive[idx] = True
        model[idx] = profile_id(Kind.LANE_MINION, mtype, tm)
        x[idx], y[idx] = px, py
        hp[idx] = mhp[idx] = hpv

    return s.replace(kind=jnp.asarray(kind), team=jnp.asarray(team),
                     alive=jnp.asarray(alive), x=jnp.asarray(x), y=jnp.asarray(y),
                     hp=jnp.asarray(hp), max_hp=jnp.asarray(mhp),
                     model=jnp.asarray(model))


def _run_until_missile(s, step, limit=80):
    """Advance until a missile is in flight, and return the tick it took."""
    for i in range(1, limit + 1):
        s = step(s)
        if _n_missiles(s) > 0:
            return s, i
    raise AssertionError(f"no missile appeared within {limit} ticks")


def test_a_casters_damage_lands_after_flight_not_on_the_swing():
    """The whole point of a missile: the hit is not simultaneous with the swing.

    A caster minion is ranged with an empty basic-attack script, so its swing
    (``FinishCasting``) creates a ``SpellMissile`` instead of applying damage,
    and that missile only pays out once it physically covers the distance at
    ITS OWN ``MissileSpeed`` -- 650 u/s for this minion, read from its own
    ``Blue_Minion_WizardBasicAttack.json`` (``sim/missiles.py``), not the
    engine's 500 default. Modelling this as instant damage merely shifted in
    time would still be wrong here: it would also pay out a target that is
    *going* to survive the flight window, which this test would not catch, so
    ``test_a_missile_whose_target_dies_in_flight...`` below covers that half
    separately.

    130 units is chosen so the flight time is a whole number of ticks --
    ``130 / 650`` s is exactly 12 ticks at 60 Hz -- so the tick the damage
    lands on is asserted exactly, not just "eventually".
    """
    dist = 130.0
    hp0 = 1000.0
    s = _two_minions(MinionType.CASTER, Team.BLUE, (5000.0, 5000.0),
                     MinionType.MELEE, Team.RED, (5000.0 + dist, 5000.0), hp_b=hp0)
    step, params = _ticker()
    victim = 3

    s, _ = _run_until_missile(s, step)
    assert float(s.hp[victim]) == pytest.approx(hp0), \
        "damage must not land on the same tick the caster swings"

    tables = build_profile_tables(load_patch())
    caster = profile_id(Kind.LANE_MINION, MinionType.CASTER, Team.BLUE)
    ad = float(tables["attack_damage"][caster])
    speed = float(tables["missile_speed"][caster])
    assert speed == pytest.approx(650.0)
    flight_ticks = round((dist / speed) / (TICK_MS / 1000.0))
    assert flight_ticks == 12

    for _ in range(flight_ticks - 1):
        s = step(s)
        assert float(s.hp[victim]) == pytest.approx(hp0), "landed before covering the distance"
    s = step(s)
    assert float(s.hp[victim]) == pytest.approx(hp0 - ad)  # armor 0, no mitigation
    assert _n_missiles(s) == 0, "the missile must be gone once it lands"


def test_a_missile_whose_target_dies_in_flight_deals_no_damage():
    """The load-bearing case: ``SpellMissile.Update``'s ``else`` branch.

    ``if (HasTarget() && !TargetUnit.IsDead && Targetable) Move(); else
    SetToRemove();`` -- the ``else`` is unconditional and carries no damage
    application at all. Get this wrong (e.g. by applying the missile's stored
    damage regardless of what happened to the target while it was in the air)
    and a winning side is credited for kills its attacks never actually
    landed, which is exactly the free lunch that made the pre-missile sim's
    lane run away to one side instead of oscillating like the server's.

    The victim is killed by hand -- ``alive = False`` -- while its hp is left
    at 50, a value the missile's stored damage (23) would visibly change if it
    were wrongly applied. That decouples the assertion from hp's floor-at-zero
    clamp, which would make "no damage" and "some damage, then floored" look
    identical if hp had been set to 0 instead.
    """
    dist = 300.0  # far enough that the missile is still mid-flight when killed
    s = _two_minions(MinionType.CASTER, Team.BLUE, (5000.0, 5000.0),
                     MinionType.MELEE, Team.RED, (5000.0 + dist, 5000.0), hp_b=455.0)
    step, _ = _ticker()
    victim = 3

    s, _ = _run_until_missile(s, step)
    assert _n_missiles(s) == 1

    alive = np.asarray(s.alive).copy()
    hp = np.asarray(s.hp).copy()
    alive[victim] = False
    hp[victim] = 50.0
    s = s.replace(alive=jnp.asarray(alive), hp=jnp.asarray(hp))

    s = step(s)
    assert _n_missiles(s) == 0, "a missile whose target just died must be removed, not keep flying"
    assert float(s.hp[victim]) == pytest.approx(50.0)

    # Keep going well past when it would otherwise have arrived, to rule out
    # a merely-deferred application rather than a suppressed one.
    for _ in range(30):
        s = step(s)
    assert float(s.hp[victim]) == pytest.approx(50.0)


def test_a_melee_minions_damage_is_instant_not_a_missile():
    """Melee is ``IsMelee``, so ``FinishCasting`` never reaches the missile
    branch regardless of anything else -- this is the branch the caster tests
    above must NOT affect."""
    s = _two_minions(MinionType.MELEE, Team.BLUE, (5000.0, 5000.0),
                     MinionType.MELEE, Team.RED, (5000.0 + 100.0, 5000.0))
    step, _ = _ticker()
    victim = 3
    hp0 = float(s.hp[victim])
    ever_missile = False
    hit_tick = None
    for i in range(1, 60):
        s = step(s)
        ever_missile = ever_missile or _n_missiles(s) > 0
        if hit_tick is None and float(s.hp[victim]) < hp0:
            hit_tick = i
    assert hit_tick is not None, "the melee minion never landed a hit"
    assert not ever_missile, "a melee attacker must never create a missile"


def test_garens_damage_is_instant_not_a_missile():
    """Garen is a champion, not a minion, but the gate is ``fires_missile``
    (i.e. ``!IsMelee``), not ``Kind`` -- so this needs its own test rather than
    trusting the minion test to cover it. A champion only auto-acquires a
    target while attack-moving, so the target is handed to it directly here,
    exactly as an accepted attack order would.
    """
    patch = load_patch()
    s = init_lane(patch, include_all_turrets=False)
    kind = np.asarray(s.kind).copy()
    team = np.asarray(s.team).copy()
    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    hp = np.asarray(s.hp).copy()
    mhp = np.asarray(s.max_hp).copy()
    target = np.asarray(s.target).copy()

    garen, victim = 0, 2
    x[1], y[1] = 0.0, 20000.0        # the other champion, out of the way
    alive = np.asarray(s.alive).copy()
    kind[victim] = Kind.LANE_MINION
    team[victim] = Team.RED
    # `alive` was missing here, and the omission was invisible for as long as
    # the champion's target code never checked whether its target was alive.
    # Once fog-of-war wiring made a held target require visibility -- and
    # visibility implies alive -- Garen correctly refused to attack a corpse
    # and this test failed with "Garen never landed a hit". The fixture was
    # always wrong; a correct change is what surfaced it.
    alive[victim] = True
    x[victim], y[victim] = 5100.0, 5000.0
    hp[victim] = mhp[victim] = 1000.0
    x[garen], y[garen] = 5000.0, 5000.0
    target[garen] = victim

    s = s.replace(kind=jnp.asarray(kind), team=jnp.asarray(team),
                 alive=jnp.asarray(alive),
                 x=jnp.asarray(x), y=jnp.asarray(y), hp=jnp.asarray(hp),
                 max_hp=jnp.asarray(mhp), target=jnp.asarray(target))
    step, _ = _ticker()
    hp0 = float(s.hp[victim])
    ever_missile = False
    hit_tick = None
    for i in range(1, 60):
        s = step(s)
        ever_missile = ever_missile or _n_missiles(s) > 0
        if hit_tick is None and float(s.hp[victim]) < hp0:
            hit_tick = i
    assert hit_tick is not None, "Garen never landed a hit"
    assert not ever_missile, "Garen must never create a missile"


def test_a_lane_turrets_damage_also_lands_after_flight():
    """The case a naive rule gets wrong -- in the OPPOSITE direction from what
    it looks like.

    A lane turret is ranged (``attack_range`` 750), which is exactly the
    property that would make "ranged units fire missiles" look like the right
    shortcut. The one-line reason it looked wrong instead -- "it has a real
    ``BasicAttack.cs`` script, so ``HasEmptyScript`` is false" -- was checked
    against ``SRUAP_Turret_Order3``/``Chaos3``, which are Map11 turrets.
    ``lanerl/cfg/garen1v1.json`` pins map 1, whose outer turrets are
    ``OrderTurretNormal``/``ChaosTurretWorm``, and **neither has a script
    anywhere in Content**. So the shortcut's conclusion was right for the
    wrong turret and wrong for the real one: a map-1 turret's damage is a
    missile, at its own ``MissileSpeed`` (1200 u/s, not a caster's 650), same
    as any other ranged attacker.

    100 units at 1200 u/s is exactly 5 ticks, so -- as in the caster test --
    the landing tick is asserted exactly.
    """
    patch = load_patch()
    s = init_lane(patch)  # the real 24-turret map, not the isolated arena
    kind = np.asarray(s.kind).copy()
    team = np.asarray(s.team).copy()
    alive = np.asarray(s.alive).copy()
    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    hp = np.asarray(s.hp).copy()
    mhp = np.asarray(s.max_hp).copy()
    model = np.asarray(s.model).copy()

    turret_team, tx, ty, _, _ = ALL_TURRETS[1]     # blue's top outer turret
    assert turret_team == Team.BLUE
    turret_idx = TU_SLICE.start + 1
    # 200, not 100: since the collision-parity pass turrets are OBSTACLES
    # (`IsCollisionObject` excludes only LevelProp/Particle/SpellMissile/Region;
    # only `IsCollisionAffected` excludes BaseTurret), so anything inside
    # minion PathfindingRadius + 1 + turret PathfindingRadius = 35.74 + 1 +
    # 88.40 = 125.14 units of a turret's centre is pushed out on the first
    # tick. At 100 the victim was standing inside the turret and got displaced
    # mid-flight, so the precomputed `flight_ticks` no longer matched. 200 is
    # clear of the footprint and still well inside the turret's 750 range, so
    # this stays a test of missile flight rather than of collision.
    dist = 200.0
    hp0 = 1000.0

    victim = 2
    kind[victim] = Kind.LANE_MINION
    team[victim] = Team.RED
    alive[victim] = True
    model[victim] = profile_id(Kind.LANE_MINION, MinionType.MELEE, Team.RED)
    x[victim], y[victim] = tx + dist, ty
    hp[victim] = mhp[victim] = hp0
    # both champions parked far from this turret so neither of them acts
    x[0], y[0] = tx, ty - 20_000.0
    x[1], y[1] = tx, ty - 21_000.0

    s = s.replace(kind=jnp.asarray(kind), team=jnp.asarray(team),
                 alive=jnp.asarray(alive), x=jnp.asarray(x), y=jnp.asarray(y),
                 hp=jnp.asarray(hp), max_hp=jnp.asarray(mhp), model=jnp.asarray(model))
    step, _ = _ticker()

    s, _ = _run_until_missile(s, step)
    assert float(s.hp[victim]) == pytest.approx(hp0), \
        "damage must not land on the same tick the turret fires"

    tables = build_profile_tables(patch)
    r = profile_id(Kind.TURRET, TurretTier.OUTER, Team.BLUE)
    speed = float(tables["missile_speed"][r])
    assert speed == pytest.approx(1200.0)
    flight_ticks = round((dist / speed) / (TICK_MS / 1000.0))
    assert flight_ticks == 10

    for _ in range(flight_ticks - 1):
        s = step(s)
        assert float(s.hp[victim]) == pytest.approx(hp0), "landed before covering the distance"
    s = step(s)
    assert float(s.hp[victim]) < hp0, "the turret never actually landed the hit"
    assert _n_missiles(s) == 0
    assert bool(s.alive[turret_idx]), "the turret itself must be unaffected by any of this"


def test_a_missile_homes_on_a_moving_target():
    """``GetTargetPosition`` reads ``TargetUnit.Position`` every ``Update``, so
    a missile re-aims at where its target IS, not where it stood at launch --
    which is what makes a ranged minion's damage conditional on the target
    still being reachable, rather than merely delayed.

    The victim is a champion with no target of its own (a champion only
    re-paths onto a target it holds; with none, its hand-set waypoints are
    never overwritten by that logic) sent walking 3000 units down a straight
    line right after the missile launches. The missile still lands, and it
    lands more than 500 units from where the victim stood at launch -- proof
    it tracked the move rather than flying at a fixed point and getting lucky.
    """
    patch = load_patch()
    s = init_lane(patch, include_all_turrets=False)
    kind = np.asarray(s.kind).copy()
    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    target = np.asarray(s.target).copy()

    caster, victim = 2, 1
    kind[caster] = Kind.LANE_MINION
    team = np.asarray(s.team).copy()
    team[caster] = Team.BLUE
    model = np.asarray(s.model).copy()
    model[caster] = profile_id(Kind.LANE_MINION, MinionType.CASTER, Team.BLUE)
    alive = np.asarray(s.alive).copy()
    alive[caster] = True
    hp = np.asarray(s.hp).copy()
    mhp = np.asarray(s.max_hp).copy()
    hp[caster] = mhp[caster] = 1000.0     # a freshly-used minion slot starts at hp 0
    x[caster], y[caster] = 5000.0, 5000.0
    x[victim], y[victim] = 5500.0, 5000.0
    target[victim] = -1
    x[0], y[0] = 0.0, 0.0
    target[0] = -1

    s = s.replace(kind=jnp.asarray(kind), team=jnp.asarray(team), model=jnp.asarray(model),
                 alive=jnp.asarray(alive), x=jnp.asarray(x), y=jnp.asarray(y),
                 hp=jnp.asarray(hp), max_hp=jnp.asarray(mhp), target=jnp.asarray(target))
    step, _ = _ticker()
    s, _ = _run_until_missile(s, step)
    launch_x, launch_y = float(s.x[victim]), float(s.y[victim])
    hp0 = float(s.hp[victim])

    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    wp = np.asarray(s.waypoints).copy()
    n_wp = np.asarray(s.n_waypoints).copy()
    wp_key = np.asarray(s.waypoint_key).copy()
    move_order = np.asarray(s.move_order).copy()
    target = np.asarray(s.target).copy()
    wp[victim, 0] = [x[victim], y[victim]]
    wp[victim, 1] = [x[victim] + 3000.0, y[victim]]
    n_wp[victim] = 2
    wp_key[victim] = 1
    move_order[victim] = MoveOrder.MOVE_TO
    target[victim] = -1
    s = s.replace(x=jnp.asarray(x), y=jnp.asarray(y), waypoints=jnp.asarray(wp),
                 n_waypoints=jnp.asarray(n_wp), waypoint_key=jnp.asarray(wp_key),
                 move_order=jnp.asarray(move_order), target=jnp.asarray(target))

    hit_tick = None
    for i in range(1, 300):
        s = step(s)
        assert int(s.target[victim]) == -1, \
            "the victim re-acquired a target and would stop moving, invalidating the test"
        if float(s.hp[victim]) < hp0:
            hit_tick = i
            break
    assert hit_tick is not None, "the missile lost a moving target entirely"
    moved = ((float(s.x[victim]) - launch_x) ** 2 + (float(s.y[victim]) - launch_y) ** 2) ** 0.5
    assert moved > 500.0, "the target barely moved; this would not distinguish homing from luck"


@pytest.mark.slow
def test_missile_overflow_stays_zero_over_a_long_run():
    """``step_missiles`` reports ``overflow`` -- launches that found no free
    slot, i.e. damage silently dropped because ``N_MISSILES`` was too small.
    It is not threaded out through ``step_decision``, but its own slot
    allocation makes the peak *live* count a proof rather than a proxy:
    launches are handed the free slots in rank order and the rest are
    dropped, so on any tick that still had a free slot left over, nothing was
    dropped. An overflowing tick's post-launch live count is therefore always
    exactly ``N_MISSILES`` (every slot filled), by construction of
    ``step_missiles``'s ``slot`` assignment. So: peak live count staying
    strictly below the cap for the whole run is equivalent to overflow being
    zero for the whole run.

    A few thousand ticks (~100 s) rather than the full 600 s recording that
    ``N_MISSILES`` was actually sized against -- that measurement is its own
    script, not a test, because it takes minutes to run. This is the fast
    regression check that a future change has not silently made the cap too
    small again.
    """
    patch = load_patch()
    s = init_lane(patch)
    params = lane_params(patch)
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))
    f = jax.jit(lambda st: step_decision(st, params, lane_path=path))
    max_live = 0
    for _ in range(3000):
        s = f(s)
        max_live = max(max_live, _n_missiles(s))
    assert max_live < N_MISSILES, (
        f"missile count reached the cap ({max_live} >= {N_MISSILES}); "
        "some launches this run may have overflowed")
