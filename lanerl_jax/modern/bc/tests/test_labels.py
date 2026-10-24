"""Labels: the screen inverse, clamping, and the labelling rules on hand-made action lists."""
import numpy as np

from lanerl_jax.modern.actions import MODERN_BUTTON_INDEX as B
from lanerl_jax.modern.actions import screen_usage
from lanerl_jax.modern.bc.labels import label_actions, lane_to_screen, screen_cell
from lanerl_jax.modern.bc.replay import Action, canonical_summoners
from lanerl_jax.modern.screen import N_SCREEN_X, N_SCREEN_Y, screen_to_lane, target_on_screen

AXIS, NORMAL = np.array([1.0, 0.0]), np.array([0.0, 1.0])


def test_lane_to_screen_inverts_the_camera_model():
    rng = np.random.default_rng(1)
    sx, sy = rng.uniform(0.02, 0.98, 500), rng.uniform(0.02, 0.98, 500)
    ds, dn = (np.asarray(v, float) for v in screen_to_lane(sx.astype(np.float32), sy.astype(np.float32)))
    bx, by = lane_to_screen(ds, dn)
    assert np.abs(bx - sx).max() < 1e-4 and np.abs(by - sy).max() < 1e-4


def test_screen_cell_centre_and_clamped_far_click():
    col, row, clamped = screen_cell(0.0, 0.0)
    assert (col, row, clamped) == (N_SCREEN_X // 2, N_SCREEN_Y // 2, False)
    col, row, clamped = screen_cell(6000.0, -2500.0)
    assert clamped and 0 <= col < N_SCREEN_X and 0 <= row < N_SCREEN_Y
    # The clamped cell's centre is on screen and keeps the click's direction.
    ds, dn = (float(v) for v in screen_to_lane(np.float32((col + 0.5) / N_SCREEN_X),
                                                np.float32((row + 0.5) / N_SCREEN_Y)))
    assert bool(target_on_screen(ds, dn)) and ds > 0 and dn < 0


def _game(actions, n=20):
    t = 100.0 + 0.1 * np.arange(n)
    me = np.tile(np.array([[1000.0, 1000.0]]), (n, 1))
    minion, champ = 7, 9
    units = {minion: np.tile(np.array([[1300.0, 1050.0]]), (n, 1)),
             champ: np.tile(np.array([[1200.0, 900.0]]), (n, 1))}
    return t, me, units, {minion, champ}, minion, champ


def test_move_attack_repeat_and_cast_rules():
    t, me, units, enemies, minion, champ = _game([])
    actions = [Action(100.05, "path", point=(1400.0, 1100.0), n_waypoints=3),      # decision 0: move
               Action(100.31, "attack", target=minion),                             # decision 3: attack
               Action(100.72, "attack", target=minion),                             # decision 7: repeat -> held click
               Action(101.02, "cast", slot=0, mode="self"),                                      # decision 10: Q, no click
               Action(101.21, "cast", slot=3, target=champ),                        # decision 12: R on the champion
               Action(101.41, "cast", slot=7),                                      # decision 14: item: no button
               Action(101.63, "cast", slot=5, point=(1100.0, 1300.0))]              # decision 16: a summoner at a point
    lab = label_actions(t, actions, me, units, AXIS, NORMAL, enemies, summoner_slot={4: 1, 5: 0})
    assert lab.button[0] == B["move"] and lab.click_valid[0] and not lab.click_on_unit[0]
    assert lab.button[3] == B["move"] and lab.click_on_unit[3] and lab.click_valid[3]
    assert lab.button[7] == B["move"] and lab.held[7] and lab.click_on_unit[7] and lab.n_repeat_attacks == 1
    assert lab.button[10] == B["q"] and not lab.click_valid[10] and lab.uses_screen[10] == 1.0
    assert lab.button[12] == B["r"] and lab.click_on_unit[12]
    assert not lab.representable[14] and lab.button[14] == B["noop"]
    assert lab.button[16] == B["summoner_d"] and lab.click_valid[16] and not lab.click_on_unit[16]
    assert np.array_equal(lab.uses_screen, np.asarray(screen_usage(lab.button), np.float32))
    assert lab.target_id[3] == lab.target_id[7] == minion
    assert lab.target_id[12] == champ
    assert lab.target_id[0] == lab.target_id[10] == lab.target_id[16] == -1
    # The click on the minion lands right of centre and slightly up the lane (ds > 0, dn > 0).
    assert lab.screen_x[3] > N_SCREEN_X // 2 and lab.screen_y[3] < N_SCREEN_Y // 2


def test_stop_is_a_one_point_path_at_the_champion_while_walking():
    t, me, units, enemies, minion, _ = _game([])
    actions = [Action(100.05, "path", point=(1800.0, 1000.0), n_waypoints=2),
               Action(100.52, "path", point=(1010.0, 1000.0), n_waypoints=1),       # at me, walking: stop
               Action(101.02, "path", point=(1010.0, 1000.0), n_waypoints=1),       # not walking any more: noop
               Action(101.52, "path", point=(1800.0, 1000.0), n_waypoints=2),
               Action(101.72, "path", point=(1000.0, 1000.0), n_waypoints=1),
               Action(101.95, "attack", target=minion)]                            # an attack follows: noop
    lab = label_actions(t, actions, me, units, AXIS, NORMAL, enemies)
    assert lab.button[5] == B["stop"] and lab.button[10] == B["noop"] and lab.button[17] == B["noop"]


def test_priority_and_superseded_count():
    t, me, units, enemies, minion, _ = _game([])
    actions = [Action(100.01, "path", point=(1400.0, 1100.0), n_waypoints=2),
               Action(100.05, "cast", slot=2, mode="self")]                                       # same decision: the cast wins
    lab = label_actions(t, actions, me, units, AXIS, NORMAL, enemies)
    assert lab.button[0] == B["e"] and lab.n_superseded == 1


def test_canonical_summoners_put_flash_in_d():
    assert canonical_summoners([14, 4]) == ((4, 14), {4: 1, 5: 0})
    assert canonical_summoners([4, 12]) == ((4, 12), {4: 0, 5: 1})
    assert canonical_summoners([14, 12]) == ((14, 12), {4: 0, 5: 1})


def test_held_labels_follow_the_active_order():
    """Between events the active order is the label: the move destination while walking, the attack target
    while auto-attacking; a cast of R ends it, standing still ends a move, `hold=False` fills nothing."""
    t, me, units, enemies, minion, champ = _game([])
    walking = me + np.arange(len(t))[:, None] * np.array([30.0, 0.0])            # 30 units per decision, +x
    actions = [Action(100.05, "path", point=(1800.0, 1000.0), n_waypoints=2),      # decision 0: move
               Action(100.52, "attack", target=minion),                             # decision 5: attack
               Action(101.02, "cast", slot=3, target=champ)]                        # decision 10: R
    lab = label_actions(t, actions, walking, units, AXIS, NORMAL, enemies)
    assert not lab.held[0] and lab.button[0] == B["move"]
    assert lab.held[1:5].all() and (lab.button[1:5] == B["move"]).all() and not lab.click_on_unit[1:5].any()
    assert lab.click_valid[1:5].all() and lab.screen_x[2] < lab.screen_x[1]       # the destination comes closer
    assert not lab.held[5] and lab.held[6:10].all() and lab.click_on_unit[6:10].all()
    assert (lab.target_id[5:10] == minion).all()
    assert lab.target_id[10] == champ
    assert lab.button[10] == B["r"] and not lab.held[10] and not lab.held[11:].any()
    assert (lab.button[11:] == B["noop"]).all()
    # Standing still ends a held move; without `hold` nothing is filled.
    lab = label_actions(t, actions[:1], me, units, AXIS, NORMAL, enemies)
    assert lab.held[1] and not lab.held[3:].any()
    lab = label_actions(t, actions, walking, units, AXIS, NORMAL, enemies, hold=False)
    assert not lab.held.any() and lab.button[1] == B["noop"]


def test_cast_targeting_uses_received_metadata_without_champion_rules():
    from lanerl_jax.modern.bc.labels import cast_rules
    assert cast_rules("Garen") == cast_rules("Jax") == cast_rules("Nobody")
    assert cast_rules("Garen").self_cast == frozenset({13})
    t, me, units, enemies, minion, champ = _game([])
    actions = [Action(100.05, "cast", slot=0, target=champ, mode="unit"),
               Action(100.52, "cast", slot=0, mode="unit"),
               Action(101.02, "cast", slot=1, mode="self"),
               Action(101.52, "cast", slot=2)]
    lab = label_actions(t, actions, me, units, AXIS, NORMAL, enemies, hold=False)
    assert lab.button[0] == B["q"] and lab.click_on_unit[0]
    assert not lab.representable[5]  # unit mode without a target is unknown, not a self cast
    assert lab.button[10] == B["w"] and not lab.click_valid[10]
    assert not lab.representable[15]  # silent targeting metadata cannot borrow Garen's self-cast rule


def test_received_self_and_allied_summoner_targets():
    t, me, units, enemies, minion, champ = _game([])
    actor, ally = 10, 11
    units[actor] = me
    units[ally] = np.tile([[1100., 1000.]], (len(t), 1))
    actions = [Action(100.05, "cast", slot=2, target=actor, mode="unit"),
               Action(100.52, "cast", slot=5, target=ally, mode="unit"),
               Action(101.02, "cast", slot=0, target=ally, mode="unit")]
    lab = label_actions(t, actions, me, units, AXIS, NORMAL, enemies, me=actor, hold=False)
    assert lab.representable[[0, 5, 10]].all()
    assert not lab.click_valid[0] and lab.click_on_unit[[5, 10]].all()
    assert lab.button[5] == B["summoner_f"]


def test_a_cast_record_on_an_unknown_slot_does_not_displace_the_real_cast():
    """Gnar's Q arrives with a second cast record on its mega-form slot (47) in the same decision."""
    import numpy as np
    from lanerl_jax.modern.actions import MODERN_BUTTON_INDEX as b
    from lanerl_jax.modern.bc.labels import CastRules, label_actions
    from lanerl_jax.modern.bc.replay import Action
    t = np.arange(0.0, 2.0, 0.1)
    me = np.zeros((len(t), 2))
    acts = [Action(0.5, "cast", point=(300.0, 0.0), slot=0), Action(0.5, "cast", point=(300.0, 0.0), slot=47)]
    rules = CastRules(self_cast=frozenset({13}), unit_target=frozenset())
    for order in (acts, acts[::-1]):
        L = label_actions(t, order, me, {}, np.array([1.0, 0.0]), np.array([0.0, 1.0]), set(), {4: 0, 5: 1}, hold=False, rules=rules)
        assert L.button[5] == b["q"] and L.representable[5] and L.click_valid[5]
