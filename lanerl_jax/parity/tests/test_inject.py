"""Turret model inference for the Tier-1 injector.

`inject_snapshot` writes a recorded server state into the JAX sim so a
one-step differential can run against it. Every field it cannot recover
honestly is refused rather than guessed (`infer_minion_model`'s contract);
`infer_turret_model` exists because a turret's TIER used to not need
recovering at all -- every turret shared one profile, so `next(profiles for
this team)` always found the only candidate. Once turrets carry per-tier
profiles that shortcut silently starts returning the OUTER row for every
turret regardless of which one it actually is, which is exactly the kind of
regression a one-step diff would otherwise blame on the wrong mechanic.
"""
from __future__ import annotations

import pytest

from lanerl_jax.parity.inject import (
    inject_snapshot,
    infer_turret_model,
    reconstruct_waypoints,
    reconstruct_waypoints_relaxed,
    replay_wave_states,
    xp_bounds_for_level,
)
from lanerl_jax.parity.trace import Snapshot, parse_row, parse_stream
from lanerl_jax.parity.one_step import OneStepResult
from lanerl_jax.sim.init import lane_params
from lanerl_jax.sim.init import ALL_TURRETS, TOP_LANE_PATH
from lanerl_jax.sim.profiles import PROFILES, profile_id
from lanerl_jax.sim.spells import BuffId, E_BUFF_SLOT, W_PASSIVE_BUFF_SLOT
from lanerl_jax.sim.state import Kind, Team
from lanerl_jax.sim.waves import FIRST_WAVE_MS, WaveState, step_waves


def _minion(team, x, y, order):
    """A dump-shaped, full-health melee minion for temporal injection tests."""
    return parse_row(
        f"LaneMinion|{team}|{int(x * 16)},{int(y * 16)}|296960/296960|A|"
        f"{order}|2|-|-|1|")


def _state_for(snapshot, previous=None):
    return inject_snapshot(snapshot, WaveState(FIRST_WAVE_MS, 0, 0),
                           lane_params(), PROFILES,
                           previous_snapshot=previous)


def test_infer_turret_model_recovers_every_tier_not_just_outer():
    """Before this existed, injecting ANY turret entity resolved to
    ``next(r for r, (k, _, t) in enumerate(profiles) if k == TURRET and t ==
    team)`` -- the first turret row for that team, full stop. With one row per
    team that was harmless by construction; with five it silently mislabels
    every inner, inhibitor, nexus and fountain turret as an outer one. This
    walks every real placed turret and checks the recovered row matches its
    OWN tier, not always the first (outer) one.
    """
    for team, x, y, _hp, tier in ALL_TURRETS:
        row, reason = infer_turret_model(x, y, team)
        assert row == profile_id(Kind.TURRET, tier, team), (
            f"({x},{y}) team={team}: got row {row} ({reason}), "
            f"expected tier {tier}'s row")


def test_infer_turret_model_matches_within_the_dumps_own_quantisation():
    """The dump quantises to 1/16 of a unit; a position off by a fraction of
    that must still resolve to the same turret, not the nearest DIFFERENT one.
    """
    team, x, y, _hp, tier = ALL_TURRETS[1]     # blue's top outer turret
    row, _ = infer_turret_model(x + 1 / 32, y - 1 / 16, team)
    assert row == profile_id(Kind.TURRET, tier, team)


def test_infer_turret_model_refuses_rather_than_guessing_when_far_off():
    """No known turret sits at the champion spawn or in the middle of a
    lane -- and this must say so, not hand back its nearest (irrelevant)
    match. `infer_minion_model` refuses the same way for an unrecognised
    minion; this is the turret side of that same contract.
    """
    row, reason = infer_turret_model(6000.0, 6000.0, Team.BLUE)
    assert row is None
    assert "too far" in reason


def test_infer_turret_model_does_not_cross_teams():
    """Blue's outer turret position must not resolve to a RED profile row even
    though `ALL_TURRETS` also has red entries -- the team argument has to
    actually gate the search, not just tiebreak it.
    """
    team, x, y, _hp, tier = ALL_TURRETS[1]     # blue's top outer turret
    assert team == Team.BLUE
    row, _ = infer_turret_model(x, y, Team.BLUE)
    assert row == profile_id(Kind.TURRET, tier, Team.BLUE)
    # Asking for RED at BLUE's outer turret's position must not silently
    # return blue's row -- it must fail to find a nearby RED turret instead
    # (the nearest red turret is thousands of units away).
    row_wrong_team, reason = infer_turret_model(x, y, Team.RED)
    assert row_wrong_team is None, reason


def test_replay_wave_states_does_not_double_count_the_pairing_ticks_own_spawn():
    """Regression for the off-by-one this module's docstring now documents:
    an earlier version paired `trace[i]` with the counters as they stood
    BEFORE `trace[i]`'s own tick, so if `trace[i]` was itself a spawn tick
    (its own population already includes the new minion, since the dump is
    written after that tick's LevelScript.Update ran), `tick()` -- stepping
    FROM the injected `trace[i]` -- would re-evaluate the exact same
    threshold at the exact same game time and spawn AGAIN on top of the unit
    already in the dump.

    `replay_wave_states` must instead return, for index i, counters that
    already reflect `trace[i].t_ms`'s own decision: re-running `step_waves`
    at that SAME game time must spawn nothing more.
    """
    snaps = [Snapshot(t_ms=int(FIRST_WAVE_MS)), Snapshot(t_ms=int(FIRST_WAVE_MS) + 17)]
    states = replay_wave_states(snaps)
    assert len(states) == 2

    # The state paired with the FIRST (spawn) tick must already have consumed
    # that tick's own arrival -- one minion per barrack has already spawned.
    assert states[0].minion_number == 1

    # Re-evaluating step_waves at the SAME game time the state was already
    # advanced through must be a no-op: the bug this guards against would
    # instead spawn a second minion here.
    spawned_again = step_waves(states[0], float(snaps[0].t_ms))
    assert spawned_again == [], (
        "replay_wave_states paired trace[i] with pre-trace[i] counters -- "
        "the sim will spawn on top of a minion the dump already shows")


def test_reconstruct_waypoints_relaxed_never_refuses():
    """Unlike the strict (exact) reconstruction, the relaxed fallback must
    always return a usable waypoint array -- that is its entire purpose: a
    guess for the case the strict version refuses, not a second refusal."""
    # Far off any known corridor vertex -- the strict version refuses this.
    off_x, off_y = 6000.0, 6000.0
    strict = reconstruct_waypoints(off_x, off_y, Team.BLUE)
    assert strict[0] is None, "test fixture should be a case the strict path refuses"

    relaxed = reconstruct_waypoints_relaxed(off_x, off_y, Team.BLUE)
    wp, key, n, reason = relaxed
    assert wp is not None
    assert n == len(TOP_LANE_PATH)
    assert 0 <= key < n
    assert "guess" in reason.lower()


def test_reconstruct_waypoints_relaxed_agrees_with_strict_on_corridor():
    """On the corridor, where the strict path succeeds, the relaxed fallback
    should recover the identical waypoint key -- it is the same projection,
    just without the tolerance gate."""
    x, y = TOP_LANE_PATH[3]
    strict_wp, strict_key, strict_n, _ = reconstruct_waypoints(x, y, Team.BLUE)
    relaxed_wp, relaxed_key, relaxed_n, _ = reconstruct_waypoints_relaxed(
        x, y, Team.BLUE)
    assert strict_wp is not None
    assert relaxed_key == strict_key
    assert relaxed_n == strict_n


def test_injection_recovers_lane_ai_cursor_and_labels_off_corridor_guess():
    """The new persistent LaneMinionAI cursor must never default to zero.

    Dumps omit `currentWaypointIndex`, but the immutable team-relative path is
    known. On-corridor rows recover its upcoming index; a chase-displaced row
    uses the same nearest/upcoming projection only as an explicitly labelled
    guess.
    """
    x, y = TOP_LANE_PATH[3]
    state, report = _state_for(Snapshot(t_ms=1017, entities=[
        _minion(100, x, y, 1),
        _minion(200, 6000, 6000, 1),
    ]))
    blue, red = report.notes
    _, expected, _, _ = reconstruct_waypoints(x, y, Team.BLUE)
    assert int(state.lane_waypoint_key[blue.slot]) == expected
    assert "on-corridor upcoming" in blue.lane_waypoint_recovery
    assert int(state.lane_waypoint_key[red.slot]) != 0
    assert "OFF-CORRIDOR" in red.lane_waypoint_recovery


def test_temporal_injection_recovers_only_a_unique_attack_to_target():
    """ATTACK_TO says a minion has a held target, but never names it.

    It is safe to inject one only when the current dump leaves exactly one
    modelled enemy in its acquisition range.  This is deliberately stricter
    than re-running nearest-target acquisition: incumbent hysteresis makes a
    closest candidate a guess, not evidence.
    """
    current = Snapshot(t_ms=1017, entities=[
        _minion(100, 0, 0, 3),       # ATTACK_TO
        _minion(200, 120, 0, 1),     # the unique plausible held target
        _minion(200, 1000, 0, 1),    # outside minion acquisition range
    ])
    previous = Snapshot(t_ms=1000, entities=[
        _minion(100, -3, 0, 3),
        _minion(200, 118, 0, 1),
        _minion(200, 998, 0, 1),
    ])
    state, report = _state_for(current, previous)
    source = report.notes[0]

    assert int(state.target[source.slot]) == report.notes[1].slot
    assert not bool(state.is_attacking[source.slot])
    assert float(state.aa_windup[source.slot]) == 0.0
    assert "unique ATTACK_TO" in source.target_recovery
    assert "windup ruled out" in source.attack_recovery
    # The quadtree is rebuilt before movement, while each dump is after it:
    # prior position is a useful temporal proxy, never claimed as exact.
    assert float(state.collision_x[source.slot]) == -3.0
    assert "temporal proxy" in source.collision_cache_recovery
    provenance = report.provenance_counts()
    assert provenance[
        "target_recovery=unique ATTACK_TO candidate among injected lane entities "
        "(unmodelled targetable kinds remain a blind spot)"] == 1


def test_target_recovery_refuses_a_hysteresis_ambiguous_attack_to():
    """Two legal enemies are not disambiguated by the row's position/order."""
    current = Snapshot(t_ms=1017, entities=[
        _minion(100, 0, 0, 3),
        _minion(200, 120, 0, 1),
        _minion(200, 160, 0, 1),
    ])
    state, report = _state_for(current)
    source = report.notes[0]
    assert int(state.target[source.slot]) == -1
    assert "unresolved ATTACK_TO target (2" in source.target_recovery
    assert "current-position fallback" in source.collision_cache_recovery


def test_collision_proxy_refuses_a_nonadjacent_prior_snapshot():
    """A dropped log interval must not become a stale collision-cache claim."""
    current = Snapshot(t_ms=1100, entities=[_minion(100, 10, 0, 2)])
    old = Snapshot(t_ms=1000, entities=[_minion(100, 0, 0, 2)])
    state, report = _state_for(current, old)
    note = report.notes[0]
    assert float(state.collision_x[note.slot]) == 10.0
    assert note.collision_cache_recovery == "current-position fallback"


def test_xp_bounds_and_dumped_cooldowns_are_not_conflated_with_a_point_xp_value():
    """Level determines an interval; spell cooldown is a separate exact field."""
    champ = parse_row(
        "Champion|100|0,0|772096/772096|A|1|0|-|-|1|GarenE+GarenWPassive+Mystery"
        "|59269,28197,45219,353280,1024|2|486400|0|0|0"
        "|1:0|1:2048|1:5120|-1:-1")
    state, report = _state_for(Snapshot(t_ms=1017, entities=[champ]))
    note = report.notes[0]
    lower, upper = xp_bounds_for_level(2, lane_params()["xp_curve"])

    assert note.xp_bounds == (lower, upper)
    assert float(state.xp[note.slot]) == lower
    assert upper is not None and lower < upper
    assert float(state.spell_cooldown[note.slot, 1]) == 2.0
    assert float(state.spell_cooldown[note.slot, 2]) == 5.0
    # Exact identity survives, but no made-up elapsed duration/power creates a
    # fictitious full-length Garen E on the following tick.
    assert int(state.buff_id[note.slot, E_BUFF_SLOT]) == BuffId.GAREN_E
    assert float(state.buff_duration[note.slot, E_BUFF_SLOT]) == 0.0
    assert int(state.buff_id[note.slot, W_PASSIVE_BUFF_SLOT]) == BuffId.GAREN_W_PASSIVE
    assert "phase/power intentionally unresolved" in note.buff_recovery
    assert "Mystery" in note.buff_recovery
    assert note.cooldown_recovery == "exact dumped cooldowns"


def test_level_cap_has_no_fabricated_xp_ceiling():
    lower, upper = xp_bounds_for_level(18, lane_params()["xp_curve"])
    assert lower > 0
    assert upper is None


def test_hidden_state_provenance_is_visible_in_the_one_step_report():
    """A report cannot silently make a defaulted timer look like agreement."""
    result = OneStepResult(
        recovery_counts={
            "attack_recovery=auto-attack clock unobservable from dump": 3,
            "xp_bounds=within-level interval (lower bound injected)": 2,
        })
    text = result.report()
    assert "hidden-state injection provenance (not agreement)" in text
    assert "auto-attack clock unobservable" in text
    assert "within-level interval" in text


def test_internal_stream_restores_identity_attack_ai_maps_and_missile():
    """The diagnostic oracle must replace defaults, not merely parse them."""
    blue = _minion(100, 0, 0, 3)
    red = _minion(200, 120, 0, 1)
    rows = [
        "LANERL_STATEHASH t=1000 n=2 h=0000000000000001",
        "LANERL_STATEROW t=1000 " + "LaneMinion|100|0,0|296960/296960|A|3|2|-|-|1|",
        "LANERL_STATEROW t=1000 " + "LaneMinion|200|1920,0|296960/296960|A|1|2|-|-|1|",
        "LANERL_INTERNAL t=1000 ai id=77 kind=LaneMinion team=100 x=0 y=0 "
        "xbits=0 ybits=0 target=88,LaneMinion,200,1920,0 wpkey=1 "
        "wps=0,0;1920,0 coll=-48,0 collbits=-1069547520,0 "
        "aacd=512 aastate=1 aacast=0 "
        "aadelay=100 aawindup=245 attacking=1 hasaa=0 aitimer=128000 "
        "ailocal=92160000 aitsa=3072 aiprio=4 aiwp=3 aihad=1 "
        "aiignore=88:92672000 aihelp=88:2",
        "LANERL_INTERNAL t=1000 ai id=88 kind=LaneMinion team=200 x=1920 y=0 "
        "xbits=1123024896 ybits=0 target=0,-,0,0,0 wpkey=1 "
        "wps=1920,0;2000,0 coll=none collbits=none "
        "aacd=0 aastate=0 aacast=0 aadelay=0 "
        "aawindup=0 attacking=0 hasaa=0 aitimer=0 ailocal=92160000 "
        "aitsa=0 aiprio=14 aiwp=2 aihad=0 aiignore=none aihelp=none",
        "LANERL_INTERNAL t=1000 missile id=99 kind=SpellMissile x=960 y=0 "
        "owner=77 target=88 speed=665600 damage=24576",
    ]
    snapshot = parse_stream(rows)[0]
    state, report = _state_for(snapshot)
    source, dest = report.notes

    assert int(state.target[source.slot]) == dest.slot
    assert float(state.aa_cooldown[source.slot]) == pytest.approx(0.5)
    assert float(state.aa_windup[source.slot]) == pytest.approx(245 / 1024)
    assert bool(state.is_attacking[source.slot])
    assert float(state.ai_timer[source.slot]) == pytest.approx(125.0)
    assert float(state.ai_local_time[source.slot]) == pytest.approx(90000.0)
    assert float(state.time_since_attack[source.slot]) == pytest.approx(3.0)
    assert int(state.target_priority[source.slot]) == 4
    assert int(state.lane_waypoint_key[source.slot]) == 3
    assert float(state.ignore_until[source.slot, dest.slot]) == pytest.approx(90500.0)
    assert int(state.help_priority[source.slot, dest.slot]) == 2
    assert bool(state.missile_alive[0])
    assert int(state.missile_source[0]) == source.slot
    assert int(state.missile_tx[0]) == dest.slot
    assert float(state.missile_speed[0]) == pytest.approx(650.0)
    assert float(state.missile_damage[0]) == pytest.approx(24.0)
    assert float(state.collision_x[source.slot]) == pytest.approx(-3.0)
    assert bool(state.collision_present[source.slot])
    assert not bool(state.collision_present[dest.slot])
    assert "exact cached position" in source.collision_cache_recovery
    assert "float32 bits" in source.collision_cache_recovery
    assert source.position_recovery == "exact float32 bits from diagnostic stream"
    assert "exact NetId" in source.target_recovery
    assert "exact cooldown" in source.attack_recovery


def test_an_injected_champion_in_lane_is_not_standing_in_its_own_fountain():
    """`spawn_x/spawn_y` is the fountain, and `step.tick` reads it as one.

    This field used to be set to the champion's CURRENT position, on the
    reasoning that its only consumer was respawn. `step.tick`'s fountain-heal
    block tests `(x - spawn_x)^2 + (y - spawn_y)^2 <= 1000^2` against the same
    field, so every injected champion sat in its own fountain and was healed
    15% of max HP per second of simulated time. Tier 1 could not see it (one
    16.667 ms tick never reaches the 1,000 ms pulse period) and every
    free-running differential was dominated by it.
    """
    from lanerl_jax.sim.init import CHAMPION_SPAWN
    from lanerl_jax.sim.step import _FOUNTAIN_RADIUS

    # a champion standing in the top lane, nowhere near a base
    champ = parse_row(
        "Champion|100|52923,204346|400000/772096|A|2|2|-|-|1|GarenPassive"
        "|59269,28197,45219,353280,1024|1|486400|0|0|1"
        "|1:0|0:-1|1:5120|-1:-1")
    state, report = _state_for(Snapshot(t_ms=120_000, entities=[champ]))
    slot = report.notes[0].slot
    sx, sy = float(state.spawn_x[slot]), float(state.spawn_y[slot])
    assert (sx, sy) == pytest.approx(CHAMPION_SPAWN[Team.BLUE])
    d = ((float(state.x[slot]) - sx) ** 2 + (float(state.y[slot]) - sy) ** 2) ** 0.5
    assert d > _FOUNTAIN_RADIUS, (
        f"an injected lane champion is {d:.0f} u from its recorded spawn, "
        f"inside the {_FOUNTAIN_RADIUS:.0f} u fountain-heal radius")
