"""Fog of war: what the agent may and may not know about what it cannot see."""

from __future__ import annotations

import numpy as np
import pytest

from lanerl_rl import constants as C
from lanerl_rl.frame import ApproxFogModel, visible_ids_for
from lanerl_rl.obs import ObservationBuilder
from lanerl_rl.scenarios import make_frame, top_lane_scenario, unit

BLUE_TURRET = C.TOP_OUTER_TURRET[C.TEAM_BLUE]


def _isolated_frame(t_ms: int, enemy_xy, blue_xy=BLUE_TURRET):
    """Blue alone at its turret; red parked far away with nothing granting vision."""
    return make_frame(
        t_ms,
        [
            unit(1001, "champion", C.TEAM_BLUE, blue_xy[0], blue_xy[1], hp=671, mhp=671,
                 gold=500.0, xp=0.0, lvl=1),
            unit(1002, "champion", C.TEAM_RED, enemy_xy[0], enemy_xy[1], hp=671, mhp=671,
                 gold=500.0, xp=0.0, lvl=1),
            unit(4001, "turret", C.TEAM_BLUE, BLUE_TURRET[0], BLUE_TURRET[1], hp=1550, mhp=1550),
        ],
    )


def test_buildings_are_never_fogged(quiet_fog):
    """ObjBuilding / BaseTurret override IsAffectedByFoW to false."""
    f = _isolated_frame(100_000, (12000, 12000))
    visible, _src = visible_ids_for(f, C.TEAM_BLUE, quiet_fog)
    assert 4001 in visible  # own turret
    assert 1002 not in visible  # distant enemy champion


def test_fogged_enemy_has_valid_zero(quiet_fog):
    """The fog gate, at the two places it is read: the valid bit and the mask.

    ``E_VISIBLE_NOW`` used to be checked alongside ``valid`` because the two
    were genuinely different -- a fogged unit kept its slot with ``valid = 0``
    and ``visible_now = 0``.  Only visible units are slotted now, so
    ``visible_now`` IS ``valid`` and asserting both would be asserting one
    thing twice.  What is left is the pair that actually gates the network:
    the valid bit and the ``entity_pad_mask`` derived from it.
    """
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = b.build(_isolated_frame(100_000, (12000, 12000)))
    lo, hi = C.SLOT_ENEMY_CHAMP
    assert o.entities[lo, C.E_VALID] == 0.0
    assert o.entity_pad_mask[lo]
    assert o.global_vec[C.G_ENEMY_VISIBLE] == 0.0


def test_never_seen_enemy_leaves_the_slot_completely_empty(quiet_fog):
    """The worst failure mode: dx=dy=0 with valid=1 reads as 'on top of me'."""
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = b.build(_isolated_frame(100_000, (12000, 12000)))
    lo, _ = C.SLOT_ENEMY_CHAMP
    row = o.entities[lo]
    assert row[C.E_VALID] == 0.0
    # Never seen -> no memory at all -> the row is entirely zero.
    assert np.all(row == 0.0)


def test_a_remembered_enemy_does_not_keep_its_slot(quiet_fog):
    """See the enemy, then lose vision: the row must VANISH, not go stale.

    The old contract was the opposite one -- keep the slot, drop ``valid``,
    start a staleness counter, and carry the last known offset so the network
    could reason about where he might be now.  That was safe only while the row
    also carried ``age_s`` / ``reach_radius`` / ``last_heading`` to say how much
    to distrust it, and those were built on a horizon constant that was already
    unreachable (``ENEMY_MEMORY_HORIZON_S = 60`` against ``FORGET_S = 8``).
    A remembered position emitted with no expiry information reads to the
    network as a sighting, which is a confident wrong answer rather than a
    missing one -- so the slot is now released and the recurrent core carries
    the memory.

    What has NOT changed is the thing this test has always really been about:
    the stale offset must not be presented as a live one.  Asserted here as the
    absence of the row, and across a whole trajectory by
    :func:`test_stale_position_never_appears_in_a_valid_slot`.
    """
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    near = (BLUE_TURRET[0] + 300, BLUE_TURRET[1] + 300)

    seen = b.build(_isolated_frame(100_000, near))
    lo, _ = C.SLOT_ENEMY_CHAMP
    assert seen.entities[lo, C.E_VALID] == 1.0
    assert seen.global_vec[C.G_ENEMY_VISIBLE] == 1.0
    seen_dx = float(seen.entities[lo, C.E_DS])
    seen_dy = float(seen.entities[lo, C.E_DN])
    assert (seen_dx, seen_dy) != (0.0, 0.0), "probe needs a non-zero offset to look for"

    lost = b.build(_isolated_frame(100_500, (12000, 12000)))
    assert lost.entities[lo, C.E_VALID] == 0.0, "fogged enemy must not be valid"
    assert lost.entity_pad_mask[lo], "fogged enemy must be masked out of attention"
    assert np.all(lost.entities[lo] == 0.0), "the fogged enemy's row must be released"
    assert lost.global_vec[C.G_ENEMY_VISIBLE] == 0.0
    # The remembered offset is nowhere in the table, not merely not in slot 0.
    for s in range(C.N_SLOTS):
        assert not (
            float(lost.entities[s, C.E_DS]) == pytest.approx(seen_dx, abs=1e-6)
            and float(lost.entities[s, C.E_DN]) == pytest.approx(seen_dy, abs=1e-6)
        ), f"slot {s} carries the fogged enemy's last known offset"


def test_stale_position_never_appears_in_a_valid_slot(quiet_fog):
    """No slot with valid=1 may carry a fogged unit's position, live or stale."""
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    near = (BLUE_TURRET[0] + 300, BLUE_TURRET[1] + 300)
    b.build(_isolated_frame(100_000, near))  # establish the memory

    for step in range(1, 12):
        t = 100_000 + 100 * step
        f = _isolated_frame(t, (12000 + 50 * step, 12000))
        o = b.build(f)
        visible, _ = visible_ids_for(f, C.TEAM_BLUE, quiet_fog)
        assert 1002 not in visible

        # Everything below is in LANE-LOCAL coordinates -- ``transform.point``
        # maps world -> (s, n).
        ax, ay = b.transform.point(BLUE_TURRET[0], BLUE_TURRET[1])
        sx, sy = b.transform.point(*near)
        stale_ds = (sx - ax) / C.NORM_XY
        stale_dn = (sy - ay) / C.NORM_XY
        live = f.units[1002]
        lx, ly = b.transform.point(live.x, live.y)
        live_ds = (lx - ax) / C.NORM_XY
        live_dn = (ly - ay) / C.NORM_XY

        for s in range(C.N_SLOTS):
            if o.entities[s, C.E_VALID] < 0.5:
                continue
            dx, dy = float(o.entities[s, C.E_DS]), float(o.entities[s, C.E_DN])
            assert not (
                abs(dx - stale_ds) < 1e-6 and abs(dy - stale_dn) < 1e-6
            ), f"valid slot {s} leaks the STALE position of the fogged enemy"
            assert not (
                abs(dx - live_ds) < 1e-6 and abs(dy - live_dn) < 1e-6
            ), f"valid slot {s} leaks the LIVE position of the fogged enemy"


def test_memory_is_forgotten_after_the_horizon(quiet_fog):
    """``FORGET_S`` must actually drop the entry, not just hide it.

    Checked against ``builder.memory`` rather than the entity table.  The table
    stopped being able to show this the moment fogged units stopped being
    slotted: a remembered unit and a forgotten one produce the same all-zero
    row from the first fogged tick onwards, so the old assertion ("a
    long-forgotten enemy must leave no trace" in ``entities``) is now true
    ``FORGET_S`` seconds before the thing it is named after happens.  The
    surviving property is the one the horizon exists for -- an unbounded
    memory dict in a long game -- and that is what is asserted.
    """
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    near = (BLUE_TURRET[0] + 300, BLUE_TURRET[1] + 300)
    b.build(_isolated_frame(100_000, near))
    lo, _ = C.SLOT_ENEMY_CHAMP
    assert 1002 in b.memory.units, "probe needs the enemy to have been seen at all"

    t = 100_000
    while t < 100_000 + C.FORGET_S * 1000 - 200:
        t += 100
        b.build(_isolated_frame(t, (12000, 12000)))
    assert 1002 in b.memory.units, "dropped BEFORE the horizon; the horizon means nothing"

    for _ in range(10):
        t += 100
        o = b.build(_isolated_frame(t, (12000, 12000)))
    assert 1002 not in b.memory.units, "a long-forgotten enemy must leave no trace"
    assert np.all(o.entities[lo] == 0.0)


def test_fogged_enemy_hp_is_not_reported(quiet_fog):
    """A health bar you cannot see is not a number you may have.

    ``E_HP_KNOWN`` is gone -- with only visible units slotted, the flag was
    degenerate for everything except the off-screen case below -- so the check
    is now that the fogged enemy's health does not appear anywhere: not in its
    old slot, not in any other.
    """
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    near = (BLUE_TURRET[0] + 300, BLUE_TURRET[1] + 300)
    seen = b.build(_isolated_frame(100_000, near))
    lo, _ = C.SLOT_ENEMY_CHAMP
    seen_hp = float(seen.entities[lo, C.E_HP_FRAC])
    assert seen_hp > 0.0, "probe needs a health bar that WAS readable a tick ago"

    o = b.build(_isolated_frame(100_100, (12000, 12000)))
    assert o.entities[lo, C.E_HP_FRAC] == 0.0
    champ_i = C.ENTITY_TYPE_INDEX["champion"]
    enemy_i = C.ENTITY_TEAMS.index("enemy")
    for s in range(C.N_SLOTS):
        row = o.entities[s]
        assert not (row[C.E_TYPE_ONEHOT][champ_i] and row[C.E_TEAM_ONEHOT][enemy_i]), (
            f"slot {s} still describes the fogged enemy champion"
        )


def test_visible_but_off_screen_gives_position_without_health(quiet_fog):
    """Minimap gives position; a health bar needs the unit on screen.

    ``E_VISIBLE_NOW`` / ``E_ON_SCREEN`` / ``E_HP_KNOWN`` were deleted: with
    only visible units slotted, ``visible_now`` is exactly ``valid``, and the
    other two are readable off the row itself -- an unreadable bar leaves
    ``hp_frac`` at 0 while the position is written regardless.  The gating is
    the property, so it is checked differentially: the SAME frame, built once
    with the unit off screen and once with it on, must differ in hp and agree
    in position.  Asserting hp == 0 alone would also pass on a builder that
    never writes hp at all.
    """
    near = (BLUE_TURRET[0] + 400, BLUE_TURRET[1])
    f = _isolated_frame(100_000, near)
    lo, _ = C.SLOT_ENEMY_CHAMP

    off = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog, screen_radius=200.0).build(f)
    on = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog, screen_radius=5000.0).build(f)

    assert off.entities[lo, C.E_VALID] == 1.0, "visible on the minimap means valid"
    assert off.entities[lo, C.E_HP_FRAC] == 0.0, "an off-screen health bar is not readable"
    assert on.entities[lo, C.E_HP_FRAC] > 0.0, "an on-screen health bar must be read"
    # Position is known either way, and it is the same position.
    assert abs(float(off.entities[lo, C.E_DS])) + abs(float(off.entities[lo, C.E_DN])) > 0.0
    assert float(off.entities[lo, C.E_DS]) == pytest.approx(float(on.entities[lo, C.E_DS]))
    assert float(off.entities[lo, C.E_DN]) == pytest.approx(float(on.entities[lo, C.E_DN]))


def test_critic_sees_through_the_fog(quiet_fog):
    """The whole point of the asymmetric critic."""
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    f = _isolated_frame(100_000, (12000, 12000))
    o = b.build(f)
    assert o.entities[C.SLOT_ENEMY_CHAMP[0], C.E_VALID] == 0.0
    assert o.priv_vec[C.P_E_ALIVE] == 1.0
    assert o.priv_vec[C.P_E_VISIBLE_TO_ME] == 0.0
    assert o.priv_vec[C.P_E_DIST] > 0.0
    assert o.priv_entities[C.SLOT_ENEMY_CHAMP[0], C.E_VALID] == 1.0


def test_approx_fog_matches_the_server_rule_on_a_dense_frame():
    """Radius rule agreement between our model and a server-annotated frame."""
    f = top_lane_scenario(with_visibility=True)
    model = ApproxFogModel(warn=False)
    for team in (C.TEAM_BLUE, C.TEAM_RED):
        from_server, src = visible_ids_for(f, team, model)
        assert src == "server"
        approx = model.visible_ids(f, team)
        assert from_server == approx
