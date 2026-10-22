from types import SimpleNamespace

import numpy as np

from lanerl_jax.modern.bc import convert as C
from lanerl_jax.modern.bc.replay import Action, Unit
from lanerl_jax.modern.core import types as W


def fixture():
    units = [Unit(1, "champion", 0, "Garen"), Unit(2, "champion", 1, "Jax"),
             Unit(3, "turret", 0, "outer")]
    g = SimpleNamespace(t=np.array([10., 11.]), units=units, me=1, opponent=2,
                        index=lambda uid: next(i for i, u in enumerate(units) if u.id == uid),
                        x=np.array([[100., 200., 300.], [100., 200., 300.]]),
                        y=np.zeros((2, 3)), hp=np.array([[100., 100., 50.], [100., 100., np.nan]]),
                        max_hp=np.full((2, 3), 100.), alive=np.ones((2, 3), bool),
                        turret_id={(0, 2, "outer"): 3})
    cfg = SimpleNamespace(layout=SimpleNamespace(n_units=6, minion0=2, monster0=3, ward0=3, struct0=3),
                          unit_kind=np.array([W.KIND_CHAMPION, W.KIND_CHAMPION, W.KIND_MINION,
                                              W.KIND_TURRET, W.KIND_TURRET, W.KIND_NEXUS]),
                          unit_sub=np.zeros(6, int), unit_team=np.array([0, 1, 0, 0, 1, 0]),
                          unit_lane=np.full(6, 2))
    return g, cfg


def test_structure_state_is_received_and_missing_state_stays_unknown():
    g, cfg = fixture()
    # cfg intentionally has no positions: they must come from the replay.
    fields, slots, _ = C._unit_fields(g, cfg, np.stack([g.x[:, 0], g.y[:, 0]], -1))
    np.testing.assert_array_equal(fields["x"][:, 3], [300., 300.])
    assert fields["hp"][0, 3] == 50. and np.isnan(fields["hp"][1, 3])
    assert fields["max_hp"][1, 3] == 100.
    assert not fields["alive"][:, 4:].any()  # absent turret and template Nexus
    assert not fields["kind"][:, 4:].any()
    assert (slots[:, 4:] == -1).all()


def test_summoner_timers_use_received_slots_and_no_cooldown_constants():
    g, _ = fixture()
    g.spell_rank = np.ones((2, 2, 4), int)
    g.mana = g.cs = g.level = g.xp = g.gold = np.ones(2)
    g.champion_stats = np.ones((2, 2, 11))
    g.last_champion_combat = np.full(2, -1e9)
    g.summoner_slot = {4: 1, 5: 0}
    g.actions = [Action(10., "cast", slot=4), Action(10., "cast", slot=5),
                 Action(10., "cast", slot=13)]
    g.cooldown_events = [(10., 4, 100., 100., 0)]
    fields, _ = C._own_fields(g, 0, 2)
    assert np.isnan(fields["summoner_ready_at"][:, 0, 0]).all()
    np.testing.assert_array_equal(fields["summoner_ready_at"][:, 0, 1], [110., 110.])
    assert np.isnan(fields["cooldowns"][:, 0]).all()
    assert np.isnan(fields["recall_channeling"][:, 0]).all()
