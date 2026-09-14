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
            assert o.action_mask.screen_x.shape == (C.N_SCREEN_X,)
            assert o.action_mask.screen_y.shape == (C.N_SCREEN_Y,)
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


def test_empty_slots_are_all_zero(recorded_run):
    """A slot with valid=0 must be entirely zero.

    This used to be weaker than its own name: a slot could legitimately hold a
    remembered-but-fogged entity with ``valid = 0`` and a stale position, so all
    the check could ask for was ``visible_now == 0``.  Since only VISIBLE units
    are slotted at all, ``valid = 0`` now means the slot holds NOTHING, and the
    check can assert the thing it was always about -- no residue of any kind
    under a masked row, because ``entity_pad_mask`` is the only thing standing
    between that residue and the attention encoder.
    """
    for obs_list in recorded_run.values():
        for o in obs_list:
            for s in range(C.N_SLOTS):
                if o.entities[s, C.E_VALID] > 0.5:
                    continue
                assert np.all(o.entities[s] == 0.0), f"slot {s} is masked but not empty"


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
            # "Occupied" used to also mean "remembered but fogged" (staleness
            # > 0 with valid = 0).  Fogged units get no slot now, so occupancy
            # is exactly the valid bit.
            occupied = e[:, C.E_VALID] > 0.5
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


def _slot_dist(o, s: int) -> float:
    """Distance to the entity in slot ``s``, in normalised units.

    ``E_DIST`` was deleted from the layout: it is ``hypot(E_DS, E_DN)``, i.e. a
    function of two fields already present, and the deleted precomputed
    quantities are where the bugs lived.  The DISTANCE ORDER it was used to
    check is still a real property of the slot assignment (``_assign_slots``
    sorts on it), so the tests recompute it here rather than dropping the
    checks.
    """
    return math.hypot(float(o.entities[s, C.E_DS]), float(o.entities[s, C.E_DN]))


def _is_sorted(xs, tol: float = 1e-6) -> bool:
    """Non-decreasing, with a float32 tolerance.

    ``_assign_slots`` sorts on a float64 distance; the rows are stored as
    float32 ``ds``/``dn`` and the distance is recomputed from them here, so two
    genuinely-ordered neighbours can come back out of order by an ulp.  1e-6 is
    ~10 float32 ulps at these magnitudes and ~3 game units at NORM_XY = 3000,
    i.e. far below anything a real mis-sort would produce.
    """
    return all(a <= b + tol for a, b in zip(xs, xs[1:]))


def test_slots_sorted_by_distance_within_block(recorded_run):
    """Distance order holds everywhere except the last-hit head of the enemy block."""
    for obs_list in recorded_run.values():
        for o in obs_list:
            for lo, hi in (C.SLOT_ALLY_MINION, C.SLOT_TURRET):
                occ = [s for s in range(lo, hi) if o.entities[s, C.E_VALID] > 0.5]
                dists = [_slot_dist(o, s) for s in occ]
                assert _is_sorted(dists), dists
            # The enemy-minion block is distance-ordered from LAST_HIT_SORT_K on.
            lo, hi = C.SLOT_ENEMY_MINION
            tail = [
                s
                for s in range(lo + C.LAST_HIT_SORT_K, hi)
                if o.entities[s, C.E_VALID] > 0.5
            ]
            dists = [_slot_dist(o, s) for s in tail]
            assert _is_sorted(dists), dists


def test_one_hots_are_one_hot(recorded_run):
    """Exactly one bin set per one-hot -- the off-by-one tripwire.

    Three of the five one-hots this used to check no longer exist: the 18-wide
    self level one-hot (19 inputs for one integer, redundant with
    ``S_LEVEL_NORM``), the 6 region one-hots (all thresholds on ``lane_s`` /
    ``lane_n``) and the 6 clock-phase one-hots (a function of the clock).  The
    same class of bug is still reachable through the scalars that replaced
    them -- an index or a scale written raw instead of normalised -- so those
    are range-checked here instead of being dropped.
    """
    for obs_list in recorded_run.values():
        for o in obs_list:
            e = o.entities
            occupied = e[:, C.E_VALID] > 0.5
            assert np.all(e[occupied][:, C.E_TYPE_ONEHOT].sum(axis=1) == 1.0)
            assert np.all(e[occupied][:, C.E_TEAM_ONEHOT].sum(axis=1) == 1.0)
            lvl = float(o.self_vec[C.S_LEVEL_NORM]) * C.MAX_LEVEL
            assert 1 <= round(lvl) <= C.MAX_LEVEL
            assert lvl == pytest.approx(round(lvl), abs=1e-5), "level_norm is not level/18"
            assert 0.0 <= float(o.global_vec[C.G_CLOCK_NORM]) <= 2.0


def test_hp_is_quantised_to_bar_resolution(recorded_run):
    step = 1.0 / C.HP_BAR_STEPS
    for obs_list in recorded_run.values():
        for o in obs_list:
            hp = o.entities[:, C.E_HP_FRAC]
            ratio = hp / step
            assert np.allclose(ratio, np.round(ratio), atol=1e-4)


def test_action_mask_always_leaves_something_legal(recorded_run):
    for obs_list in recorded_run.values():
        for o in obs_list:
            assert o.action_mask.button.any()
            assert o.action_mask.target.any()
            assert o.action_mask.screen_x.all()


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


def test_enemy_minions_are_in_plain_distance_order_by_default(quiet_fog):
    """The last-hit pre-sort is OFF, and this test used to assert the opposite.

    ``LAST_HIT_SORT_K`` used to be 4: the nearest four enemy-minion slots were
    re-sorted by ascending HP so the last-hit candidate always sat at a fixed
    index. That is a crutch. ``hp_frac`` is already a per-slot feature for
    every candidate, so sorting adds no information -- it only removes the
    COMPARISON from the network's job, and the target head learns "click slot
    13" instead of "find the minion that is about to die".

    The skill it skipped is the one that matters as soon as the decision stops
    being trivial: two near-dead minions, a contested deny, a minion weighed
    against the champion. This repo has been burned by exactly this shape
    before -- the movement head was blind for weeks because a free crutch
    meant the signal was never learned.

    Set ``LAST_HIT_SORT_K`` above 0 to put it back; the behaviour is still
    covered by the test below.
    """
    assert C.LAST_HIT_SORT_K == 0, (
        "the pre-sort is back on; if that is deliberate, re-collect demos and "
        "retrain BC -- a BC target head trained under one value reads slot "
        "positions that mean something else under the other"
    )
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = b.build(_minion_cluster())
    lo, hi = C.SLOT_ENEMY_MINION
    live = [s for s in range(lo, hi) if o.entities[s, C.E_VALID] > 0.5]
    dists = [_slot_dist(o, s) for s in live]
    assert _is_sorted(dists), (
        f"enemy minions are not in distance order: {dists}. Every other block "
        f"in the table is distance-ordered; this one must be too."
    )


def test_the_last_hit_pre_sort_still_works_when_switched_back_on(quiet_fog, monkeypatch):
    """Kept so the ablation is a real switch rather than dead code.

    If someone turns it back on to A/B it, it must actually do the thing.
    """
    monkeypatch.setattr(C, "LAST_HIT_SORT_K", 4)
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog)
    o = b.build(_minion_cluster())
    lo, hi = C.SLOT_ENEMY_MINION
    head = list(range(lo, lo + 4))
    hp = [float(o.entities[s, C.E_HP_FRAC]) for s in head]
    assert all(h > 0.0 for h in hp), "probe needs readable health bars"
    assert hp == sorted(hp), hp
    dists = [_slot_dist(o, s) for s in head]
    assert dists != sorted(dists), "the HP sort did nothing; the probe is vacuous"


def test_minions_with_unreadable_health_sort_last_in_the_head(quiet_fog, monkeypatch):
    """A guess is worse than a defer: off-screen minions must not claim slot 0.

    ``E_HP_KNOWN`` no longer exists, so readability is read off the only thing
    that distinguishes the two cases in the new layout: an unreadable bar
    leaves ``hp_frac`` at 0.0 while every minion in this cluster is alive with
    hp > 1/120 of its maximum, so a zero here can only mean "not read".  The
    non-vacuity assertion below is new and load-bearing for exactly that
    reason -- a mix of both kinds is what makes the ordering claim mean
    anything, and with a sentinel value rather than a flag it has to be
    checked.
    """
    # A property OF THE SORT, so it only means anything with the sort on --
    # it is off by default now (see
    # test_enemy_minions_are_in_plain_distance_order_by_default).
    monkeypatch.setattr(C, "LAST_HIT_SORT_K", 4)
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog, screen_radius=100.0)
    o = b.build(_minion_cluster(hps=(400, 60, 250, 30, 455), spacing=60.0))
    lo, _ = C.SLOT_ENEMY_MINION
    head = list(range(lo, lo + 4))
    known = [1.0 if float(o.entities[s, C.E_HP_FRAC]) > 0.0 else 0.0 for s in head]
    assert 1.0 in known and 0.0 in known, f"probe has no mix of readable/unreadable: {known}"
    # Whatever the mix, every readable one comes before every unreadable one.
    assert known == sorted(known, reverse=True), known


# --------------------------------------------------------------------------
# Derived memory (correction 3)
# --------------------------------------------------------------------------


def _cd_sequence(builder, gap: bool):
    """Enemy seen at 0 cd, then goes on cd; ``gap`` hides them while it happens."""
    hidden_s = 0.99 if gap else 0.55
    for i in range(4):
        f = top_lane_scenario(t_ms=100_000 + 100 * i, blue_s=0.45, red_s=0.55, n_minions=0)
        for u in f.units.values():
            if u.etype == "champion" and u.team == C.TEAM_RED:
                u.cooldowns = (0.0, 0.0, 0.0, 0.0)
        builder.build(f)
    for i in range(4, 24):
        f = top_lane_scenario(t_ms=100_000 + 100 * i, blue_s=0.45, red_s=hidden_s, n_minions=0)
        for u in f.units.values():
            if u.etype == "champion" and u.team == C.TEAM_RED:
                u.cooldowns = (6.0, 6.0, 6.0, 6.0)
        builder.build(f)
    f = top_lane_scenario(t_ms=100_000 + 2400, blue_s=0.45, red_s=0.55, n_minions=0)
    for u in f.units.values():
        if u.etype == "champion" and u.team == C.TEAM_RED:
            u.cooldowns = (6.0, 6.0, 6.0, 6.0)
    return builder.build(f)


def test_a_cast_across_a_vision_gap_is_not_witnessed(quiet_fog):
    """You cannot read an enemy's cooldown; you can only see them press it.

    The separate ``never_observed`` flag is gone from the layout -- the 8
    enemy-cooldown ESTIMATE fields it belonged to were built on
    ``ENEMY_COOLDOWN_ASSUMED``, whose documented safety direction was the
    opposite of what the code did.  What survives is
    ``time_since_observed_cast``, which is 1.0 both for "never seen him cast
    it" and for "saw it long enough ago that it is certainly back up".  That
    conflation is why this needs the positive control below: on its own,
    ``since == 1.0`` after the gap would also be satisfied by a dead feature.
    """
    hidden = _cd_sequence(ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog), gap=True)
    assert np.all(hidden.global_vec[C.G_ENEMY_ABILITY_SINCE_CAST] == 1.0), (
        "a cooldown that appeared while the enemy was out of sight was treated as a "
        "witnessed cast -- that is reading the server's cooldown, not watching a cast"
    )

    # Positive control: the identical cooldown history, watched the whole way
    # through, IS a witnessed cast, so the same field drops well below 1.0.
    # Without this the assertion above passes on a feature that never moves.
    watched = _cd_sequence(ObservationBuilder(C.TEAM_BLUE, fog_model=quiet_fog), gap=False)
    assert np.all(watched.global_vec[C.G_ENEMY_ABILITY_SINCE_CAST] < 1.0), (
        "the witnessed-cast feature never fires at all, so the vision-gap check "
        "above proves nothing"
    )


# --------------------------------------------------------------------------
# The attack cycle (correction 3)
# --------------------------------------------------------------------------


def test_attack_speed_growth_shortens_the_period(quiet_fog):
    """AttackSpeedPerLevel = 2.9%, applied through Stats.GetLevelUpStatValue."""
    periods = [C.garen_attack_period(lv) for lv in range(1, 19)]
    assert periods == sorted(periods, reverse=True)
    assert periods[0] == pytest.approx(1.6, abs=1e-9)
    assert periods[-1] < 1.2