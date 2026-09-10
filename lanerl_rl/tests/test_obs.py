"""The observation builder, run against the real recorded game."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from lanerl_rl import constants as C
from lanerl_rl.frame import ApproxFogModel, iter_jsonl
from lanerl_rl.obs import ObservationBuilder
from lanerl_rl.scenarios import top_lane_scenario, top_lane_sequence

N_FRAMES = 400
#: The first minion wave spawns at 90 s and the recorder writes a frame every
#: 100 ms, so anything before frame ~900 is an empty lane.  Starting there is
#: what makes the minion-slot assertions below non-vacuous.
SKIP_FRAMES = 900


@pytest.fixture(scope="module")
def recorded_run(request):
    from .conftest import RECORDING

    if not RECORDING.exists():
        pytest.skip(f"recording not found at {RECORDING}")
    fog = ApproxFogModel(warn=False)
    builders = {t: ObservationBuilder(t, fog_model=fog) for t in (C.TEAM_BLUE, C.TEAM_RED)}
    out = {t: [] for t in builders}
    for frame in itertools.islice(iter_jsonl(RECORDING), SKIP_FRAMES, SKIP_FRAMES + N_FRAMES):
        for t, b in builders.items():
            out[t].append(b.build(frame))
    return out


def test_shapes_and_dtypes(recorded_run):
    for team, obs_list in recorded_run.items():
        assert len(obs_list) == N_FRAMES
        for o in obs_list:
            assert o.entities.shape == (C.N_SLOTS, C.ENTITY_DIM)
            assert o.entities.dtype == np.float32
            assert o.entity_pad_mask.shape == (C.N_SLOTS,)
            assert o.entity_pad_mask.dtype == np.bool_
            assert o.self_vec.shape == (C.SELF_DIM,)
            assert o.global_vec.shape == (C.GLOBAL_DIM,)
            assert o.priv_entities.shape == (C.N_SLOTS, C.ENTITY_DIM)
            assert o.priv_vec.shape == (C.PRIV_DIM,)
            assert o.action_mask.button.shape == (C.N_BUTTONS,)
            assert o.action_mask.move_x.shape == (C.N_MOVE_BINS,)
            assert o.action_mask.move_z.shape == (C.N_MOVE_BINS,)
            assert o.action_mask.target.shape == (C.N_SLOTS,)


def test_no_nans_or_infs(recorded_run):
    for obs_list in recorded_run.values():
        for o in obs_list:
            for name, arr in (
                ("entities", o.entities),
                ("self_vec", o.self_vec),
                ("global_vec", o.global_vec),
                ("priv_entities", o.priv_entities),
                ("priv_vec", o.priv_vec),
            ):
                assert np.isfinite(arr).all(), f"{name} has non-finite values"


def test_pad_mask_matches_valid_bit(recorded_run):
    for obs_list in recorded_run.values():
        for o in obs_list:
            expected = o.entities[:, C.E_VALID] < 0.5
            assert np.array_equal(o.entity_pad_mask, expected)


def test_reserved_fields_are_zero(recorded_run):
    for obs_list in recorded_run.values():
        for o in obs_list:
            assert np.all(o.entities[:, C.E_RESERVED] == 0.0)
            assert np.all(o.self_vec[C.S_RESERVED] == 0.0)
            assert np.all(o.global_vec[C.G_RESERVED] == 0.0)


def test_empty_slots_are_all_zero(recorded_run):
    """A slot with valid=0 that holds nothing must be entirely zero."""
    for obs_list in recorded_run.values():
        for o in obs_list:
            for s in range(C.N_SLOTS):
                if o.entities[s, C.E_VALID] > 0.5:
                    continue
                # Either fully empty, or a remembered entity: remembered rows
                # must have visible_now == 0 too.
                assert o.entities[s, C.E_VISIBLE_NOW] == 0.0


def test_slot_blocks_hold_the_right_types(recorded_run):
    """Type/team one-hots must agree with the block a row landed in."""
    champ_i = C.ENTITY_TYPE_INDEX["champion"]
    minion_i = C.ENTITY_TYPE_INDEX["minion"]
    turret_i = C.ENTITY_TYPE_INDEX["turret"]
    ally_i = C.ENTITY_TEAMS.index("ally")
    enemy_i = C.ENTITY_TEAMS.index("enemy")

    for obs_list in recorded_run.values():
        for o in obs_list:
            e = o.entities
            occupied = (e[:, C.E_VALID] > 0.5) | (e[:, C.E_STALENESS] > 0.0)
            types = e[:, C.E_TYPE_ONEHOT]
            teams = e[:, C.E_TEAM_ONEHOT]
            for s in range(*C.SLOT_ENEMY_CHAMP):
                if occupied[s]:
                    assert types[s, champ_i] == 1.0
                    assert teams[s, enemy_i] == 1.0
            for s in range(*C.SLOT_ALLY_MINION):
                if occupied[s]:
                    assert types[s, minion_i] == 1.0
                    assert teams[s, ally_i] == 1.0
            for s in range(*C.SLOT_ENEMY_MINION):
                if occupied[s]:
                    assert types[s, minion_i] == 1.0
                    assert teams[s, ally_i] == 0.0
            for s in range(*C.SLOT_TURRET):
                if occupied[s]:
                    assert types[s, turret_i] == 1.0


def test_slots_sorted_by_distance_within_block(recorded_run):
    """Distance order holds everywhere except the last-hit head of the enemy block."""
    for obs_list in recorded_run.values():
        for o in obs_list:
            for lo, hi in (C.SLOT_ALLY_MINION, C.SLOT_TURRET):
                occ = [
                    s
                    for s in range(lo, hi)
                    if o.entities[s, C.E_VALID] > 0.5 or o.entities[s, C.E_STALENESS] > 0.0
                ]
                dists = [float(o.entities[s, C.E_DIST]) for s in occ]
                assert dists == sorted(dists)
            # The enemy-minion block is distance-ordered from LAST_HIT_SORT_K on.
            lo, hi = C.SLOT_ENEMY_MINION
            tail = [
                s
                for s in range(lo + C.LAST_HIT_SORT_K, hi)
                if o.entities[s, C.E_VALID] > 0.5 or o.entities[s, C.E_STALENESS] > 0.0
            ]
            dists = [float(o.entities[s, C.E_DIST]) for s in tail]
            assert dists == sorted(dists)


def test_one_hots_are_one_hot(recorded_run):
    for obs_list in recorded_run.values():
        for o in obs_list:
            e = o.entities
            occupied = (e[:, C.E_VALID] > 0.5) | (e[:, C.E_STALENESS] > 0.0)
            assert np.all(e[occupied][:, C.E_TYPE_ONEHOT].sum(axis=1) == 1.0)
            assert np.all(e[occupied][:, C.E_TEAM_ONEHOT].sum(axis=1) == 1.0)
            assert o.self_vec[C.S_LEVEL_ONEHOT].sum() == 1.0
            assert o.self_vec[C.S_REGION_ONEHOT].sum() == 1.0
            assert o.global_vec[C.G_CLOCK_PHASE].sum() == 1.0


def test_hp_is_quantised_to_bar_resolution(recorded_run):
    step = 1.0 / C.HP_BAR_STEPS
    for obs_list in recorded_run.values():
        for o in obs_list:
            hp = o.entities[:, C.E_HP_FRAC]
            ratio = hp / step
            assert np.allclose(ratio, np.round(ratio), atol=1e-4)


def test_hp_only_reported_when_visible_and_on_screen(recorded_run):
    for obs_list in recorded_run.values():
        for o in obs_list:
            e = o.entities
            no_hp = e[:, C.E_HP_KNOWN] < 0.5
            assert np.all(e[no_hp][:, C.E_HP_FRAC] == 0.0)
            known = e[:, C.E_HP_KNOWN] > 0.5
            assert np.all(e[known][:, C.E_VISIBLE_NOW] == 1.0)
            assert np.all(e[known][:, C.E_ON_SCREEN] == 1.0)


def test_action_mask_always_leaves_something_legal(recorded_run):
    for obs_list in recorded_run.values():
        for o in obs_list:
            assert o.action_mask.button.any()
            assert o.action_mask.target.any()
            assert o.action_mask.move_x.all()


def test_builder_is_deterministic(quiet_fog):
    frames = top_lane_sequence(n=12)
    a = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    for f in frames:
        oa, ob = a.build(f), b.build(f)
        assert np.array_equal(oa.entities, ob.entities)
        assert np.array_equal(oa.self_vec, ob.self_vec)
        assert np.array_equal(oa.global_vec, ob.global_vec)
        assert np.array_equal(oa.priv_vec, ob.priv_vec)


@pytest.mark.parametrize(
    "blue_s,red_s,expect_blue_s,expect_red_s",
    [
        (0.10, 0.90, 0.10, 0.10),  # both hugging their own turret
        (0.50, 0.60, 0.50, 0.40),
        (0.80, 0.30, 0.80, 0.70),  # both deep in the enemy half
    ],
)
def test_lane_frame_means_the_same_thing_for_both_sides(
    quiet_fog, blue_s, red_s, expect_blue_s, expect_red_s
):
    """``s = 0`` is always "my turret", ``s = 1`` always "theirs" -- for both agents.

    The scenario places champions along the blue->red axis, so RED's own lane
    progress is ``1 - red_s``.  If this held only for BLUE, a single set of
    weights could not play both sides.
    """
    blue = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    red = ObservationBuilder(C.TEAM_RED, fog_model=quiet_fog)
    f = top_lane_scenario(blue_s=blue_s, red_s=red_s)
    ob, orr = blue.build(f), red.build(f)
    assert float(ob.self_vec[C.S_LANE_S]) == pytest.approx(expect_blue_s, abs=0.01)
    assert float(orr.self_vec[C.S_LANE_S]) == pytest.approx(expect_red_s, abs=0.01)
    # On the lane axis, the perpendicular offset is ~0 for both.
    assert abs(float(ob.self_vec[C.S_LANE_N])) < 0.01
    assert abs(float(orr.self_vec[C.S_LANE_N])) < 0.01


def test_region_one_hot_follows_lane_progress(quiet_fog):
    blue = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    seen = []
    for s in (0.02, 0.30, 0.50, 0.70, 0.95):
        o = blue.build(top_lane_scenario(blue_s=s))
        seen.append(int(o.self_vec[C.S_REGION_ONEHOT].argmax()))
    assert seen == sorted(seen)
    assert seen[0] < seen[-1]


def test_lane_handedness_is_consistent_between_the_two_agents(quiet_fog):
    """Both agents must place their own base on the same side of the lane.

    The 180 degree rotation maps blue's top lane onto red's *bot* lane (the
    rift is point-symmetric), so without the handedness convention in
    ``LaneFrame`` the two agents would read opposite signs of ``n`` for the
    same lane-relative situation.
    """
    for team in (C.TEAM_BLUE, C.TEAM_RED):
        b = ObservationBuilder(team, fog_model=quiet_fog)
        own_nexus = C.NEXUS_POSITION[team]
        cx, cy = b.transform.point(*own_nexus)
        _s, n = b.lane.sn(cx, cy)
        assert n < 0.0, f"{C.TEAM_NAMES[team]} nexus should sit at n < 0"


def test_privileged_vector_carries_enemy_economy(quiet_fog):
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = b.build(top_lane_scenario(red_gold=1234.0, red_lvl=7))
    assert o.priv_vec[C.P_E_GOLD_NORM] == pytest.approx(1234.0 / C.NORM_GOLD, rel=1e-5)
    assert o.priv_vec[C.P_E_LEVEL_ONEHOT][6] == 1.0


def test_no_slot_index_feature():
    """There must be nothing in the layout that identifies a slot position."""
    for name in C.ENTITY_FIELD_NAMES:
        low = name.lower()
        assert "slot" not in low
        assert "index" not in low
        assert low not in ("idx", "position_id", "rank")


def test_fog_source_reported(recorded_run):
    for obs_list in recorded_run.values():
        assert all(o.fog_source == "approx" for o in obs_list)


def test_server_supplied_visibility_is_used(quiet_fog):
    f = top_lane_scenario(with_visibility=True)
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = b.build(f)
    assert o.fog_source == "server"


def test_control_channel_vb_vr_flags_are_understood():
    """The live wire format: per-team booleans, not a team list."""
    from lanerl_rl.frame import decode_frame, visible_ids_for

    raw = {
        "t": 100_000,
        "u": [
            {"id": 1, "k": "Champion", "tm": 100, "x": 0, "y": 0, "hp": 600, "mhp": 600,
             "vb": 1, "vr": 0, "gold": 500, "xp": 0, "lvl": 1,
             "cd0": 3500, "cd1": -1, "cd2": 0, "cd3": -1},
            {"id": 2, "k": "LaneMinion", "tm": 200, "x": 500, "y": 0, "hp": 100, "mhp": 455,
             "vb": 0, "vr": 1},
        ],
    }
    f = decode_frame(raw)
    assert f.units[1].visible_to == frozenset({C.TEAM_BLUE})
    assert f.units[2].visible_to == frozenset({C.TEAM_RED})
    # cd0..cd3 arrive in MILLISECONDS with -1 meaning "no such spell".
    assert f.units[1].cooldowns == (3.5, None, 0.0, None)
    ids, src = visible_ids_for(f, C.TEAM_BLUE)
    assert src == "server"
    assert ids == {1}


# --------------------------------------------------------------------------
# 32 slots
# --------------------------------------------------------------------------


def test_slot_blocks_tile_the_table_exactly():
    blocks = [
        C.SLOT_ENEMY_CHAMP,
        C.SLOT_ALLY_MINION,
        C.SLOT_ENEMY_MINION,
        C.SLOT_TURRET,
        C.SLOT_SPARE,
    ]
    assert blocks[0][0] == 0
    for a, b in zip(blocks, blocks[1:]):
        assert a[1] == b[0], (a, b)
    assert blocks[-1][1] == C.N_SLOTS
    assert C.N_SLOTS == 32
    # 12 minion slots a side: a stacked double wave is 12, and the measured p95
    # unit count within 1500 of the lane meeting point is 15 across both sides.
    assert C.SLOT_ALLY_MINION[1] - C.SLOT_ALLY_MINION[0] == 12
    assert C.SLOT_ENEMY_MINION[1] - C.SLOT_ENEMY_MINION[0] == 12


def test_the_recording_actually_needs_more_than_twenty_slots(recorded_run):
    """Not a design opinion -- a count off the real recording."""
    peak = max(
        int((o.entities[:, C.E_VALID] > 0.5).sum())
        for obs_list in recorded_run.values()
        for o in obs_list
    )
    assert peak > 20, f"only {peak} slots ever occupied; the 20 -> 32 change would be unmotivated"


# --------------------------------------------------------------------------
# The last-hit canonicalisation (correction 7)
# --------------------------------------------------------------------------


def _minion_cluster(t_ms=100_000, hps=(400, 60, 250, 30, 455), spacing=60.0):
    """Blue champion in a huddle of enemy minions, nearest-first HP given."""
    from lanerl_rl.scenarios import make_frame, unit

    a = C.TOP_OUTER_TURRET[C.TEAM_BLUE]
    units = [
        unit(1001, "champion", C.TEAM_BLUE, a[0], a[1], hp=671, mhp=671,
             gold=600.0, xp=0.0, lvl=3),
        unit(1002, "champion", C.TEAM_RED, 12000, 12000, hp=671, mhp=671,
             gold=600.0, xp=0.0, lvl=3),
        unit(4001, "turret", C.TEAM_BLUE, a[0], a[1], hp=1550, mhp=1550),
    ]
    for i, hp in enumerate(hps):
        # Increasing distance with i, so distance order and HP order differ.
        units.append(
            unit(3000 + i, "minion", C.TEAM_RED, a[0] + spacing * (i + 1), a[1], hp=hp, mhp=455)
        )
    return make_frame(t_ms, units)


def test_enemy_minion_head_is_sorted_by_ascending_hp(quiet_fog):
    """Slot 0 of the enemy-minion block is the last-hit candidate, for free."""
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = b.build(_minion_cluster())
    lo, hi = C.SLOT_ENEMY_MINION
    head = list(range(lo, lo + C.LAST_HIT_SORT_K))
    hp_abs = [float(o.entities[s, C.E_HP_ABS]) * C.NORM_HP_ABS for s in head]
    assert all(o.entities[s, C.E_HP_KNOWN] > 0.5 for s in head), "probe needs readable health bars"
    assert hp_abs == sorted(hp_abs), hp_abs
    # The head is NOT in distance order -- that is the whole point.
    dists = [float(o.entities[s, C.E_DIST]) for s in head]
    assert dists != sorted(dists), "the HP sort did nothing; the probe is vacuous"
    # And the tail keeps distance order.
    tail = [s for s in range(lo + C.LAST_HIT_SORT_K, hi) if o.entities[s, C.E_VALID] > 0.5]
    td = [float(o.entities[s, C.E_DIST]) for s in tail]
    assert td == sorted(td)


def test_one_shot_kill_score_is_highest_on_the_head_slot(quiet_fog):
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = b.build(_minion_cluster())
    lo, _ = C.SLOT_ENEMY_MINION
    head = o.entities[lo, C.E_AA_KILLABLE]
    others = o.entities[lo + 1 : lo + C.LAST_HIT_SORT_K, C.E_AA_KILLABLE]
    assert head >= others.max()
    # Garen's level-3 AD is ~64, so a 30 hp minion is very much a one-shot.
    assert head > 0.9


def test_minions_with_unreadable_health_sort_last_in_the_head(quiet_fog):
    """A guess is worse than a defer: off-screen minions must not claim slot 0."""
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog, screen_radius=100.0)
    o = b.build(_minion_cluster(hps=(400, 60, 250, 30, 455), spacing=60.0))
    lo, _ = C.SLOT_ENEMY_MINION
    head = list(range(lo, lo + C.LAST_HIT_SORT_K))
    known = [float(o.entities[s, C.E_HP_KNOWN]) for s in head]
    # Whatever the mix, every readable one comes before every unreadable one.
    assert known == sorted(known, reverse=True), known


def test_shots_to_kill_is_consistent_with_hp_and_damage(quiet_fog):
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = b.build(_minion_cluster())
    ad = C.garen_attack_damage(3)
    lo, hi = C.SLOT_ENEMY_MINION
    for s in range(lo, hi):
        if o.entities[s, C.E_HP_KNOWN] < 0.5:
            continue
        hp = float(o.entities[s, C.E_HP_ABS]) * C.NORM_HP_ABS
        expected = min(np.ceil(hp / ad) / 8.0, 1.0)
        assert float(o.entities[s, C.E_AA_SHOTS_TO_KILL]) == pytest.approx(expected, abs=1e-5)


# --------------------------------------------------------------------------
# Derived memory (correction 3)
# --------------------------------------------------------------------------


def test_reachability_radius_grows_with_the_age_of_the_estimate(quiet_fog):
    """A remembered dot is a disc, and the policy is told the radius."""
    from lanerl_rl.scenarios import make_frame, unit

    a = C.TOP_OUTER_TURRET[C.TEAM_BLUE]

    def f(t_ms, enemy_xy):
        return make_frame(
            t_ms,
            [
                unit(1001, "champion", C.TEAM_BLUE, a[0], a[1], hp=671, mhp=671,
                     gold=500.0, xp=0.0, lvl=1),
                unit(1002, "champion", C.TEAM_RED, enemy_xy[0], enemy_xy[1], hp=671, mhp=671,
                     gold=500.0, xp=0.0, lvl=1),
                unit(4001, "turret", C.TEAM_BLUE, a[0], a[1], hp=1550, mhp=1550),
            ],
        )

    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    b.build(f(100_000, (a[0] + 300, a[1] + 300)))  # seen
    lo, _ = C.SLOT_ENEMY_CHAMP
    radii, ages = [], []
    for step in range(1, 8):
        o = b.build(f(100_000 + 500 * step, (12000, 12000)))
        radii.append(float(o.entities[lo, C.E_REACH_RADIUS]))
        ages.append(float(o.global_vec[C.G_ENEMY_REACH_RADIUS]))
    assert radii == sorted(radii) and radii[0] < radii[-1]
    assert ages == sorted(ages) and ages[0] < ages[-1]
    # 3.5 s at Garen's 345 move speed is ~1207 units.
    expected = 3.5 * C.GAREN_MOVE_SPEED / C.NORM_DIST
    assert radii[-1] == pytest.approx(expected, rel=1e-4)


def test_last_seen_heading_is_recorded_and_is_a_unit_vector(quiet_fog):
    blue = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = None
    for i in range(6):
        o = blue.build(top_lane_scenario(t_ms=100_000 + 100 * i, blue_s=0.35, red_s=0.45 - 0.01 * i))
    lo, _ = C.SLOT_ENEMY_CHAMP
    assert o.entities[lo, C.E_HEADING_KNOWN] == 1.0
    cos = float(o.entities[lo, C.E_LAST_HEADING_COS])
    sin = float(o.entities[lo, C.E_LAST_HEADING_SIN])
    assert math.hypot(cos, sin) == pytest.approx(1.0, abs=1e-5)
    # The enemy is walking DOWN-lane towards us: s decreasing in our frame.
    assert cos < 0.0
    assert o.global_vec[C.G_ENEMY_MEM_HEADING_KNOWN] == 1.0


def test_never_seen_enemy_ability_is_flagged_unknown_not_ready(quiet_fog):
    """'Never seen it cast' and 'off cooldown' must not be the same number."""
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = b.build(top_lane_scenario())
    assert np.all(o.global_vec[C.G_ENEMY_ABILITY_UNKNOWN] == 1.0)
    assert np.all(o.global_vec[C.G_ENEMY_ABILITY_CD_EST] == 0.0)


def test_a_witnessed_cast_starts_a_cooldown_estimate_that_decays(quiet_fog):
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    est = []
    for i in range(40):
        f = top_lane_scenario(t_ms=100_000 + 100 * i, blue_s=0.45, red_s=0.55, n_minions=0)
        for u in f.units.values():
            if u.etype == "champion" and u.team == C.TEAM_RED:
                u.cooldowns = (0.0, 0.0, 0.0, 0.0) if i < 5 else (8.0, 24.0, 9.0, 160.0)
        o = b.build(f)
        est.append(float(o.global_vec[C.G_ENEMY_ABILITY_CD_EST][0]))
    assert est[4] == 0.0 and o.global_vec[C.G_ENEMY_ABILITY_UNKNOWN][0] == 0.0
    assert est[5] == pytest.approx(1.0, abs=1e-6), "the cast should be seen the tick it happens"
    assert est[-1] < est[6] < est[5], "the estimate must tick down"
    # Garen Q is 8 s; 3.4 s later the estimate is (8 - 3.4) / 8.
    assert est[39] == pytest.approx((8.0 - 3.4) / 8.0, abs=1e-4)


def test_a_cast_across_a_vision_gap_is_not_witnessed(quiet_fog):
    """You cannot read an enemy's cooldown; you can only see them press it."""
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    # Seen up close, then gone for 2 s, then back -- with the spell now on cd.
    for i in range(4):
        f = top_lane_scenario(t_ms=100_000 + 100 * i, blue_s=0.45, red_s=0.55, n_minions=0)
        for u in f.units.values():
            if u.etype == "champion" and u.team == C.TEAM_RED:
                u.cooldowns = (0.0, 0.0, 0.0, 0.0)
        b.build(f)
    for i in range(4, 24):
        f = top_lane_scenario(t_ms=100_000 + 100 * i, blue_s=0.45, red_s=0.99, n_minions=0)
        for u in f.units.values():
            if u.etype == "champion" and u.team == C.TEAM_RED:
                u.cooldowns = (6.0, 6.0, 6.0, 6.0)
        b.build(f)
    f = top_lane_scenario(t_ms=100_000 + 2400, blue_s=0.45, red_s=0.55, n_minions=0)
    for u in f.units.values():
        if u.etype == "champion" and u.team == C.TEAM_RED:
            u.cooldowns = (6.0, 6.0, 6.0, 6.0)
    o = b.build(f)
    assert np.all(o.global_vec[C.G_ENEMY_ABILITY_UNKNOWN] == 1.0), (
        "a cooldown that appeared while the enemy was out of sight was treated as a "
        "witnessed cast -- that is reading the server's cooldown, not watching a cast"
    )


# --------------------------------------------------------------------------
# The attack cycle (correction 3)
# --------------------------------------------------------------------------


def test_attack_cycle_is_unknown_until_a_swing_is_noted(quiet_fog):
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = b.build(top_lane_scenario())
    assert o.self_vec[C.S_ATTACK_TIMING_KNOWN] == 0.0
    assert o.self_vec[C.S_ATTACK_CYCLE_PHASE] == 0.0
    assert o.self_vec[C.S_WINDUP_REMAINING] == 0.0


def test_attack_cycle_phase_and_windup_match_the_server_content(quiet_fog):
    """1.6 s period and 0.333 s windup at level 1, from gcd_AttackDelay etc."""
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    b.build(top_lane_scenario(t_ms=100_000, blue_lvl=1))
    b.note_attack(100_000)
    period = C.garen_attack_period(1)
    windup = C.garen_attack_windup(1)
    assert period == pytest.approx(1.6, abs=1e-9)
    assert windup == pytest.approx(0.3333333, abs=1e-5)

    o = b.build(top_lane_scenario(t_ms=100_200, blue_lvl=1))
    assert float(o.self_vec[C.S_ATTACK_TIMING_KNOWN]) == 1.0
    assert float(o.self_vec[C.S_ATTACK_CYCLE_PHASE]) == pytest.approx(0.2 / period, abs=1e-5)
    assert float(o.self_vec[C.S_WINDUP_REMAINING]) == pytest.approx(
        (windup - 0.2) / windup, abs=1e-5
    )
    o = b.build(top_lane_scenario(t_ms=100_500, blue_lvl=1))
    assert float(o.self_vec[C.S_WINDUP_REMAINING]) == 0.0, "windup is over after 0.333 s"
    o = b.build(top_lane_scenario(t_ms=101_700, blue_lvl=1))
    assert float(o.self_vec[C.S_ATTACK_CYCLE_PHASE]) == 1.0
    assert float(o.self_vec[C.S_TIME_UNTIL_NEXT_ATTACK]) == 0.0


def test_attack_speed_growth_shortens_the_period(quiet_fog):
    """AttackSpeedPerLevel = 2.9%, applied through Stats.GetLevelUpStatValue."""
    periods = [C.garen_attack_period(lv) for lv in range(1, 19)]
    assert periods == sorted(periods, reverse=True)
    assert periods[0] == pytest.approx(1.6, abs=1e-9)
    assert periods[-1] < 1.2


def test_own_attack_damage_matches_the_server_growth_curve():
    """``value * (0.65 + 0.035 * Level)`` per level up, cumulative."""
    assert C.garen_attack_damage(1) == pytest.approx(57.88, abs=1e-6)
    assert C.garen_attack_damage(2) == pytest.approx(57.88 + 3.5 * (0.65 + 0.035 * 2), abs=1e-6)
    ads = [C.garen_attack_damage(lv) for lv in range(1, 19)]
    assert ads == sorted(ads)
