import numpy as np

from lanerl_jax.modern.bc import fog
from lanerl_jax.modern.bc.replay import Unit


def test_team_sight_uses_all_live_replay_allies_and_each_frame_only():
    units = [Unit(1, "champion", 0, "Garen"), Unit(2, "champion", 1, "Jax"),
             Unit(3, "champion", 0, "Lux"), Unit(4, "minion", 0, "melee")]
    # Lux is not in the observation slots but supplies sight to the enemy Jax.
    x = np.array([[0., 3000., 2000., 1000.], [0., 3000., 2000., 1000.],
                  [0., 3000., np.nan, 1000.]])
    y = np.zeros_like(x)
    alive = np.ones_like(x, bool)
    alive[1, 2] = False
    slots = np.tile([0, 1, -1], (3, 1))
    vis = fog.team_visibility(units, x, y, alive, slots)
    np.testing.assert_array_equal(vis[:, 0, 1], [True, False, False])
    assert vis[:, 0, 0].all() and vis[:, 1, 1].all()
    assert not vis[:, :, 2].any()  # template-only slots never gain replay provenance
    assert not vis[:, 1, 0].any()  # enemy sight does not reveal things to the other team


def test_radius_boundary_and_dead_or_unknown_targets():
    units = [Unit(1, "turret", 0, "outer"), Unit(2, "champion", 1, "Jax")]
    x = np.array([[0., 1350.], [0., 1350.01], [0., 500.], [0., 500.]])
    y = np.zeros_like(x)
    y[3, 1] = np.nan
    alive = np.ones_like(x, bool)
    alive[2, 1] = False
    vis = fog.team_visibility(units, x, y, alive, np.tile([0, 1], (4, 1)))
    np.testing.assert_array_equal(vis[:, 0, 1], [True, False, False, False])


def test_fog_provenance_is_explicit_about_the_exception():
    meta = fog.provenance()
    assert meta["decoded"] is False and meta["profile"] == fog.PROFILE
    assert "wards" in meta["omitted"] and "brush" in meta["omitted"]


def test_fog_exception_does_not_clear_other_source_holds():
    from lanerl_jax.modern.bc.sources import REPLAY_ONLY_BLOCKERS
    assert not any("visibility" in reason for reason in REPLAY_ONLY_BLOCKERS)
    assert any("sim-kit" in reason for reason in REPLAY_ONLY_BLOCKERS)
