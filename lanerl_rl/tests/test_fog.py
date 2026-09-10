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
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = b.build(_isolated_frame(100_000, (12000, 12000)))
    lo, hi = C.SLOT_ENEMY_CHAMP
    assert o.entities[lo, C.E_VALID] == 0.0
    assert o.entities[lo, C.E_VISIBLE_NOW] == 0.0
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
    assert o.global_vec[C.G_ENEMY_MEM_VALID] == 0.0


def test_remembered_enemy_is_masked_but_its_last_position_is_kept(quiet_fog):
    """See the enemy, then lose vision: valid must drop, memory must persist."""
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    near = (BLUE_TURRET[0] + 300, BLUE_TURRET[1] + 300)

    seen = b.build(_isolated_frame(100_000, near))
    lo, _ = C.SLOT_ENEMY_CHAMP
    assert seen.entities[lo, C.E_VALID] == 1.0
    assert seen.global_vec[C.G_ENEMY_VISIBLE] == 1.0
    seen_dx = float(seen.entities[lo, C.E_DS])
    seen_dy = float(seen.entities[lo, C.E_DN])

    lost = b.build(_isolated_frame(100_500, (12000, 12000)))
    assert lost.entities[lo, C.E_VALID] == 0.0, "fogged enemy must not be valid"
    assert lost.entity_pad_mask[lo], "fogged enemy must be masked out of attention"
    assert lost.entities[lo, C.E_STALENESS] > 0.0, "staleness must start counting"
    # The row still carries the LAST KNOWN offset, not the live one.
    assert float(lost.entities[lo, C.E_DS]) == pytest.approx(seen_dx, abs=1e-6)
    assert float(lost.entities[lo, C.E_DN]) == pytest.approx(seen_dy, abs=1e-6)
    assert lost.global_vec[C.G_ENEMY_MEM_VALID] == 1.0
    assert lost.global_vec[C.G_ENEMY_UNSEEN_TIME] > 0.0


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
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    near = (BLUE_TURRET[0] + 300, BLUE_TURRET[1] + 300)
    b.build(_isolated_frame(100_000, near))
    lo, _ = C.SLOT_ENEMY_CHAMP

    t = 100_000
    for _ in range(int(C.FORGET_S * 1000 / 100) + 5):
        t += 100
        o = b.build(_isolated_frame(t, (12000, 12000)))
    assert np.all(o.entities[lo] == 0.0), "a long-forgotten enemy must leave no trace"


def test_fogged_enemy_hp_is_not_reported(quiet_fog):
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    near = (BLUE_TURRET[0] + 300, BLUE_TURRET[1] + 300)
    b.build(_isolated_frame(100_000, near))
    o = b.build(_isolated_frame(100_100, (12000, 12000)))
    lo, _ = C.SLOT_ENEMY_CHAMP
    assert o.entities[lo, C.E_HP_KNOWN] == 0.0
    assert o.entities[lo, C.E_HP_FRAC] == 0.0


def test_visible_but_off_screen_gives_position_without_health(quiet_fog):
    """Minimap gives position; a health bar needs the unit on screen."""
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog, screen_radius=200.0)
    near = (BLUE_TURRET[0] + 400, BLUE_TURRET[1])
    o = b.build(_isolated_frame(100_000, near))
    lo, _ = C.SLOT_ENEMY_CHAMP
    assert o.entities[lo, C.E_VISIBLE_NOW] == 1.0
    assert o.entities[lo, C.E_ON_SCREEN] == 0.0
    assert o.entities[lo, C.E_HP_KNOWN] == 0.0
    assert o.entities[lo, C.E_HP_FRAC] == 0.0
    assert o.entities[lo, C.E_DIST] > 0.0


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
