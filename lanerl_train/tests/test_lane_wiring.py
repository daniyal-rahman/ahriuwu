

def test_move_label_round_trips_and_both_sides_agree():
    """BC's move labels must decode back to the direction the bot walked.

    move_bins_for inverts decode_action. It first used LaneTransform.vector,
    which is lane-local -> WORLD, not its inverse -- an involution only for the
    old MirrorTransform. Blue's labels therefore pointed the wrong way while
    red's were right, BC averaged the two sides to the centre bin, and the
    prior learned to stand still: 288 units from spawn over 300 s, against a
    lane 11,866 units away.

    Both sides must map "toward the enemy" to the SAME bins -- that is what a
    canonical frame is for.
    """
    import math
    from lanerl_rl import constants as C
    from lanerl_train.lane_wiring import make_lane_adapters
    from lanerl_train.collect_demos import move_bins_for
    from lanerl_rl.scenarios import top_lane_scenario, encode_frame

    ad = make_lane_adapters(train_step_source=lambda: 0)
    raw = encode_frame(top_lane_scenario(t_ms=120_000, n_minions=4))
    seen = {}
    for side, team in (("blue", C.TEAM_BLUE), ("red", C.TEAM_RED)):
        a = ad.adapter_factory(0, side)
        a.build(raw, side)
        own = C.TOP_OUTER_TURRET[team]
        foe = C.TOP_OUTER_TURRET[C.TEAM_RED if team == C.TEAM_BLUE else C.TEAM_BLUE]
        ux, uy = foe[0] - own[0], foe[1] - own[1]
        ch = next(u for u in raw["u"] if u.get("k") == "Champion" and u.get("tm") == team)
        bx, bz = move_bins_for(a, raw, team, ch["x"] + ux, ch["y"] + uy)
        seen[side] = (bx, bz)
        # decode exactly as env.decode_action does
        tx, tz = float(C.MOVE_BIN_VALUES[bx]), float(C.MOVE_BIN_VALUES[bz])
        nn = math.hypot(tx, tz) or 1.0
        wx, wy = a.builder.transform.vector(tx / nn, tz / nn)
        d = math.hypot(ux, uy)
        assert (wx * ux + wy * uy) / d > 0.9, f"{side}: label does not decode toward the enemy"
    assert seen["blue"] == seen["red"], (
        f"blue and red disagree on 'toward the enemy': {seen} -- BC will average "
        f"them to the centre bin and stand still"
    )
