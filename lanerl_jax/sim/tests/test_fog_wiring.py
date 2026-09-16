"""Fog of war, wired into the tick: a held target is only kept while a
teammate can still see it.

Before this module existed, ``tick`` called ``step_minion_ai`` with
``visible=state.alive`` -- i.e. every living unit was "visible" to everyone,
so nothing acquisition did ever actually respected vision even though
``fog.visible_to`` already implemented the rule correctly in isolation
(``lanerl_jax/obs/tests/test_obs.py`` tests the rule itself). The champion and
turret target-retention paths in ``step.tick`` never checked vision at all,
which is the exact mechanism behind the scripted last-hit oracle "seeing"
enemy minions still at their own spawn and walking the champion 8 deaths deep
into enemy territory (``lanerl_jax/parity/last_hit_drive.py``'s "KNOWN
ASYMMETRY" note).

These tests exercise the wiring at the level actually reachable in this
patch's numbers. ``sim/step.py`` documents why the minion and turret
acquisition/candidate paths are mathematically unaffected here (every
acquisition/attack range in ``profiles.py`` is smaller than the matching
viewer's own vision radius, so a unit's own sight always covers anything it
could otherwise acquire) -- the one place vision genuinely changes behaviour
in this sim's own AI is a CHAMPION's held target, which carries no distance
cap at all, and a TURRET's held target surviving its own owner's death.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch
from lanerl_jax.sim.init import init_lane, lane_params
from lanerl_jax.sim.movement_jax import TICK_MS
from lanerl_jax.sim.profiles import profile_id
from lanerl_jax.sim.state import Kind, Team
from lanerl_jax.sim.step import tick
from lanerl_jax.sim.targeting import MinionType

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)

BLUE_CHAMP, RED_CHAMP = 0, 1
VICTIM = 2                    # a free minion slot
BLUE_TURRET, RED_TURRET = 42, 43   # TU_SLICE.start with include_all_turrets=False


@functools.lru_cache(maxsize=1)
def _ticker():
    """One jitted ``tick``, reused across every test here (see
    ``test_missiles.py``'s ``_ticker`` for why: re-tracing per call is slow and
    unnecessary since ``params`` never changes)."""
    patch = load_patch()
    params = lane_params(patch)
    return jax.jit(lambda s: tick(s, params, TICK_MS, None, None)), params


def _base_state():
    """``init_lane``'s isolated arena: two champions, only the two outer
    turrets, no minions until one is placed by hand."""
    return init_lane(load_patch(), include_all_turrets=False)


def _place_minion(s, idx, team, x, y, hp=1000.0):
    kind = np.asarray(s.kind).copy()
    t = np.asarray(s.team).copy()
    alive = np.asarray(s.alive).copy()
    px = np.asarray(s.x).copy()
    py = np.asarray(s.y).copy()
    h = np.asarray(s.hp).copy()
    mh = np.asarray(s.max_hp).copy()
    model = np.asarray(s.model).copy()

    kind[idx] = Kind.LANE_MINION
    t[idx] = team
    alive[idx] = True
    model[idx] = profile_id(Kind.LANE_MINION, MinionType.MELEE, team)
    px[idx], py[idx] = x, y
    h[idx] = mh[idx] = hp

    return s.replace(kind=jnp.asarray(kind), team=jnp.asarray(t),
                     alive=jnp.asarray(alive), x=jnp.asarray(px),
                     y=jnp.asarray(py), hp=jnp.asarray(h),
                     max_hp=jnp.asarray(mh), model=jnp.asarray(model))


def test_a_champions_held_target_is_kept_while_a_teammate_still_sees_it():
    """Control for the test below: vision must not drop a target that is
    genuinely still visible, or the fix would just be "champions never
    attack".
    """
    s = _base_state()
    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    x[BLUE_CHAMP], y[BLUE_CHAMP] = 5000.0, 5000.0
    x[RED_CHAMP], y[RED_CHAMP] = 0.0, 30000.0     # out of the way
    s = s.replace(x=jnp.asarray(x), y=jnp.asarray(y))
    s = _place_minion(s, VICTIM, Team.RED, 5100.0, 5000.0)   # 100 from champ
    target = np.asarray(s.target).copy()
    target[BLUE_CHAMP] = VICTIM
    s = s.replace(target=jnp.asarray(target))

    step, _ = _ticker()
    s = step(s)
    assert int(s.target[BLUE_CHAMP]) == VICTIM, (
        "a target 100 units away, well inside the champion's own 1200-unit "
        "vision, must not be dropped"
    )


def test_a_champions_held_target_is_dropped_once_it_leaves_every_sight_bubble():
    """``ObjAIBase.UpdateTarget`` (``AI/ObjAIBase.cs:1183``)::

        else if (TargetUnit.IsDead || (...) || !TargetUnit.IsVisibleByTeam(Team))
        {
            if (IsAttacking) CancelAutoAttack(!HasAutoAttacked, true);
            SetTargetUnit(null, true);
            return;
        }

    A champion's held target carries NO distance cap anywhere else in this
    branch (unlike the minion and turret rules, which re-check
    ``AcquisitionRange``/attack range every reevaluation) -- this invisibility
    check is the ONLY thing that ever releases a champion from a target that
    has wandered off. Before fog was wired in, ``keep_champ`` was just
    ``is_champ & (state.target >= 0)``: a champion held its target forever,
    which is exactly the failure mode described in
    ``last_hit_drive.py``'s fog-of-war note -- a target the champion's team
    can no longer see kept dragging movement and combat toward it.
    """
    s = _base_state()
    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    x[BLUE_CHAMP], y[BLUE_CHAMP] = 5000.0, 5000.0
    x[RED_CHAMP], y[RED_CHAMP] = 0.0, 30000.0     # out of the way, and not
                                                   # a blue teammate anyway
    s = s.replace(x=jnp.asarray(x), y=jnp.asarray(y))
    # ~9500 units from the champion and from both outer turrets (574,10220)
    # and (3911,13654) -- nothing blue is within any bubble of it.
    s = _place_minion(s, VICTIM, Team.RED, 5000.0, 20000.0)
    target = np.asarray(s.target).copy()
    target[BLUE_CHAMP] = VICTIM
    s = s.replace(target=jnp.asarray(target))

    step, _ = _ticker()
    s = step(s)
    assert int(s.target[BLUE_CHAMP]) == -1, (
        "a target no blue unit can see must be dropped, not held indefinitely"
    )


def test_a_turret_drops_a_target_that_died_instead_of_holding_the_corpse():
    """``BaseTurret : ObjAIBase`` (``AI/BaseTurret.cs:17``) runs the same
    generic ``UpdateTarget`` as a champion, so the SAME ``IsDead ... ||
    !IsVisibleByTeam`` check above applies to it -- ``TurretAI`` itself
    (``Content/.../TurretAI.cs``) only ever drops a target for leaving ATTACK
    range, nothing else.

    Before this was wired in, ``turret_acquire``'s own "holding" branch kept
    ``current_target`` whenever it held one and no enemy champion was diving,
    with no check on whether that index was even still alive -- so a turret
    that had just killed its target stayed "locked on" to the corpse's slot
    (motionless, still in attack range) and went idle instead of picking a new
    target, even with other live enemies sitting in range.
    """
    s = _base_state()
    s = _place_minion(s, VICTIM, Team.RED,
                      float(s.x[BLUE_TURRET]) + 100.0, float(s.y[BLUE_TURRET]))
    target = np.asarray(s.target).copy()
    target[BLUE_TURRET] = VICTIM
    alive = np.asarray(s.alive).copy()
    alive[VICTIM] = False              # already dead when the tick starts
    s = s.replace(target=jnp.asarray(target), alive=jnp.asarray(alive))

    step, _ = _ticker()
    s = step(s)
    assert int(s.target[BLUE_TURRET]) == -1, (
        "a turret must release a dead target instead of holding its corpse"
    )


def test_a_turret_keeps_a_live_visible_target_in_range():
    """Control for the test above: a turret's own sight (800) already covers
    its own attack range (750, ``profiles.py``), so a live target sitting in
    range must be kept -- the fix above must not turn into "turrets never
    hold a target".
    """
    s = _base_state()
    s = _place_minion(s, VICTIM, Team.RED,
                      float(s.x[BLUE_TURRET]) + 100.0, float(s.y[BLUE_TURRET]))
    target = np.asarray(s.target).copy()
    target[BLUE_TURRET] = VICTIM
    s = s.replace(target=jnp.asarray(target))

    step, _ = _ticker()
    s = step(s)
    assert int(s.target[BLUE_TURRET]) == VICTIM
