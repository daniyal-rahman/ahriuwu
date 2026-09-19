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
    # This fixture's `aacd=512` (0.5 s) does not land on this minion profile's
    # own cooldown grid (it is a synthetic value, not sampled from a real
    # attack clock), so `snap_cooldown_to_tick_grid` correctly REJECTS the
    # recovery and reports why rather than silently keeping a stale "exact"
    # label from before that function existed -- the numeric assertion above
    # (`aa_cooldown == 0.5`, i.e. the dump wins on rejection) is the load-
    # bearing check; this only confirms the attempt happened and is honest
    # about its outcome.
    assert "cooldown" in source.attack_recovery
    assert "not snapped" in source.attack_recovery


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


def test_the_dumped_windup_is_snapped_back_onto_the_server_cast_grid():
    """`AA-002`. The rounded `aawindup` alone gets the hit tick wrong.

    `Spell.Update` (`Spell.cs:1643-1647`) accumulates `CurrentDelayTime` by a
    whole tick and compares `>=` against a fixed threshold, so the remaining
    wind-up only ever takes the values `W - k*dt`. The dump rounds that to
    1/1024 s. For the blue melee lane minion `W/dt` is **20.0016**, so the last
    wind-up tick has 2.67e-5 s left -- 18x below the dump's 1/2048 rounding
    threshold, published as a flat `aawindup=0`.

    Read literally that says "no wind-up left", and the sim's `winding` gate is
    `is_attacking & (aa_windup > 0)`: the hit is dropped on the tick the server
    lands it, and fired one tick early on the tick before, where the rounded
    17/1024 s sits just under `dt`. Two mismatched unit-ticks per swing, in
    opposite directions -- which is why `AA-002` measured a symmetric sign
    split and an 18x melee/ranged asymmetry at the same time.

    The guard walks every profile's whole cast clock and asserts the injected
    value reproduces the server's own `(k+1)*dt >= W` decision **after** the
    float32 round-trip the sim actually performs.
    """
    import numpy as np

    from lanerl_jax.parity.inject import snap_windup_to_tick_grid
    from lanerl_jax.sim.movement_jax import TICK_MS

    params = lane_params()
    dt = TICK_MS / 1000.0
    dt32 = np.float32(dt)
    checked = 0
    for row in range(len(PROFILES)):
        full = float(params["attack_windup"][row])
        if full <= 0.0:
            continue
        # every tick the server can still be STATE_CASTING on
        for k in range(int(np.ceil(full / dt))):
            remaining = full - k * dt
            dumped = float(np.round(remaining * 1024.0) / 1024.0)
            snapped, _why = snap_windup_to_tick_grid(dumped, row, 1.0, params)
            s32 = np.float32(snapped)
            sim_hits = bool(s32 > 0) and bool(s32 - dt32 <= 0)
            server_hits = remaining <= dt
            assert sim_hits == server_hits, (
                f"profile {PROFILES[row]} (W={full:.6f}s, W/dt={full / dt:.4f}) "
                f"tick {k}: server {'hits' if server_hits else 'waits'} but the "
                f"injected wind-up {snapped:.8f}s makes the sim "
                f"{'hit' if sim_hits else 'wait'}")
            checked += 1
    assert checked > 100, "the profile table stopped carrying wind-ups"


def test_a_windup_the_profile_table_cannot_explain_is_left_alone():
    """The snap is a recovery, not an override.

    If the port's threshold is not the server's, `W - k*dt` lands further from
    the dumped value than the dump could have rounded. Inventing phase there
    would silently paper over a wrong `attack_windup` constant -- the exact
    failure mode `STAT-001` was, one field over -- so the dump wins and the
    reason says why.
    """
    from lanerl_jax.parity.inject import snap_windup_to_tick_grid

    params = lane_params()
    row = profile_id(Kind.LANE_MINION, 0, Team.BLUE)
    value, why = snap_windup_to_tick_grid(0.0083, row, 1.0, params)
    assert value == 0.0083
    assert "not snapped" in why


def test_the_dumped_cooldown_is_snapped_back_onto_the_servers_cooldown_grid():
    """The `aa_cooldown` analogue of `AA-002`: same clock shape, same dump,
    same fix.

    `ObjAIBase.Update` (`ObjAIBase.cs:1103-1105`) decrements
    `_autoAttackCurrentCooldown` by exactly one tick every Update it is
    positive, from `1.0f / Stats.GetTotalAttackSpeed()` (`:1263`), so the
    remaining cooldown only ever takes the values `period - k*dt` for a
    profile's fixed `period` -- `AA-002`'s wind-up argument with a different
    constant. `LanerlStateDump` rounds it to 1/1024 s (`aacd=`,
    `LanerlStateDump.cs:205`), which is where the corpus's
    82.3%-of-misses "<=2 quanta" `aa_cooldown` mode comes from: two
    independent ~0.5-quantum roundings of the SAME grid point (this tick's
    dump and the next tick's), not a real difference between the two ticks.

    This walks every profile's whole cooldown clock (skipping the single tick
    nearest the gate, which the dump's `Math.Max(0f, .)` clamp -- applied
    BEFORE quantisation -- makes genuinely ambiguous with every already-fired
    tick, and which `snap_cooldown_to_tick_grid` deliberately leaves alone)
    and asserts the snap recovers the exact grid point.
    """
    import numpy as np

    from lanerl_jax.parity.inject import snap_cooldown_to_tick_grid
    from lanerl_jax.sim.movement_jax import TICK_MS

    params = lane_params()
    dt = TICK_MS / 1000.0
    checked = 0
    for row in range(len(PROFILES)):
        period = float(params["attack_period"][row])
        if not period > 0.0 or period > 100.0:
            continue
        for k in range(int(np.ceil(period / dt)) + 1):
            remaining = period - k * dt
            if remaining <= 1.0 / 2048.0:
                continue  # the ambiguous gate tick; not this function's job
            dumped = round(remaining * 1024.0) / 1024.0
            if dumped <= 0.0:
                continue
            snapped, why = snap_cooldown_to_tick_grid(dumped, row, 1.0, params)
            assert abs(snapped - remaining) < 1e-6, (
                f"profile {PROFILES[row]} (period={period:.6f}s) tick {k}: "
                f"dumped {dumped:.6f}s snapped to {snapped:.6f}s ({why}), "
                f"expected the true remainder {remaining:.6f}s")
            checked += 1
    assert checked > 100, "the profile table stopped carrying attack periods"


def test_a_cooldown_the_profile_table_cannot_explain_is_left_alone():
    """The snap is a recovery, not an override -- same guard as `AA-002`'s."""
    from lanerl_jax.parity.inject import snap_cooldown_to_tick_grid

    params = lane_params()
    row = profile_id(Kind.LANE_MINION, 0, Team.BLUE)
    value, why = snap_cooldown_to_tick_grid(0.0083, row, 1.0, params)
    assert value == 0.0083
    assert "not snapped" in why


def test_a_dumped_zero_cooldown_is_left_at_the_gate_clamp_not_invented():
    """`Math.Max(0f, remaining)` runs BEFORE the dump's quantisation
    (`LanerlAim.AutoAttackCooldownRemaining`), so a dumped `0` collapses every
    already-fired tick together with the single not-yet-fired tick within
    rounding of the gate. Disambiguating that needs the previous tick's own
    dump, which is outside this function's contract -- so a dumped `0` must
    come back unchanged, never a manufactured small positive residue.
    """
    from lanerl_jax.parity.inject import snap_cooldown_to_tick_grid

    params = lane_params()
    row = profile_id(Kind.LANE_MINION, 0, Team.BLUE)
    value, why = snap_cooldown_to_tick_grid(0.0, row, 1.0, params)
    assert value == 0.0
    assert "gate clamp" in why


def test_the_dumped_action_timer_is_snapped_back_onto_the_server_tick_grid():
    """`INJ-003`. The 250 ms sweep cannot fire from a rounded timer.

    ``LaneMinionAI.minionActionTimer`` is zeroed at every sweep and otherwise
    only ``+= delta``, and the server's free-run ``deltaTime`` is
    ``(float)REFRESH_RATE`` (`Game.cs:333`), so it is an exact float32
    accumulation of ``1000f/60f``. The fifteenth accumulation is
    **250.0000305**, so the trigger ``minionActionTimer >= 250.0f``
    (`LaneMinionAI.cs:90`) is crossed by 0.0000305 ms. `LanerlStateDump`
    rounds the field to 1/1024 ms, and the fourteenth accumulation
    (233.3333588) rounds **down** to 233.3330078 -- 0.00035 ms low, 11.5x the
    crossing margin -- so an injected minion is one rounding short of its own
    sweep and never takes it.

    That is not a rare miss: over the whole gate-1 corpus the sim's rule fired
    on **0 of 395,366** minion tick-pairs against the server's 26,537 sweeps.
    Every Tier-1 minion-controller figure taken before this was measured with
    the regular sweep switched off.

    This walks the server's own accumulation and asserts that, at every point
    on it, the *snapped* value reproduces the server's trigger decision and the
    *dumped* value does not always do so.
    """
    import numpy as np

    from lanerl_jax.parity.inject import snap_action_timer_to_tick_grid
    from lanerl_jax.sim.minion_ai import ACTION_TIMER_MS

    f32 = np.float32
    dt = f32(f32(1000.0) / f32(60.0))
    t = f32(0.0)
    raw_wrong = 0
    for _k in range(1, 17):
        t = f32(t + dt)
        dumped = round(float(t) * 1024) / 1024.0        # the dump's own value
        snapped, why = snap_action_timer_to_tick_grid(dumped)
        server_fires = bool(f32(t + dt) >= f32(ACTION_TIMER_MS))
        snapped_fires = bool(f32(f32(snapped) + dt) >= f32(ACTION_TIMER_MS))
        raw_fires = bool(f32(f32(dumped) + dt) >= f32(ACTION_TIMER_MS))
        assert snapped_fires == server_fires, (
            f"tick {_k}: snapped timer disagrees with the server's own "
            f"trigger ({why})")
        raw_wrong += raw_fires != server_fires
    assert raw_wrong > 0, (
        "the dumped value now reproduces the trigger on its own -- either the "
        "dump gained precision or this test stopped exercising the boundary")


def test_an_injected_minion_one_tick_from_its_sweep_actually_takes_it():
    """The WIRING, which the arithmetic test above cannot see.

    A guard that calls ``snap_action_timer_to_tick_grid`` directly still passes
    with the snap present in the module and absent from ``inject_snapshot`` --
    which is the shape of half the silent failures this project has logged. So
    this one goes end to end: dump a minion whose ``aitimer`` is the server's
    own fourteenth accumulation, inject it, step one tick, and check that the
    controller actually re-evaluated.

    ``ai_timer == 0`` after the tick is ``minionActionTimer = 0`` at
    `LaneMinionAI.cs:97`, which only runs on the branch that also calls
    ``UpdateMoveOrder(ReevaluateBehavior(delta))`` -- so it is the observable
    for "the sweep ran", and the sweep running is what 2,381 of the 4,791
    whole-corpus move-order misses were waiting for.
    """
    import numpy as np

    from lanerl_jax.sim.step import tick

    f32 = np.float32
    dt = f32(f32(1000.0) / f32(60.0))
    t = f32(0.0)
    for _ in range(14):
        t = f32(t + dt)
    q_dumped = round(float(t) * 1024)          # exactly what the dump writes
    assert q_dumped == 238933, q_dumped

    rows = [
        "LANERL_STATEHASH t=1000 n=2 h=0000000000000001",
        "LANERL_STATEROW t=1000 LaneMinion|100|0,0|296960/296960|A|1|2|-|-|1|",
        "LANERL_STATEROW t=1000 LaneMinion|200|1920,0|296960/296960|A|1|2|-|-|1|",
        "LANERL_INTERNAL t=1000 ai id=77 kind=LaneMinion team=100 x=0 y=0 "
        "xbits=0 ybits=0 target=0,-,0,0,0 wpkey=1 wps=0,0 "
        "coll=none collbits=none aacd=0 aastate=0 aacast=0 aadelay=0 "
        f"aawindup=0 attacking=0 hasaa=0 aitimer={q_dumped} ailocal=92160000 "
        "aitsa=0 aiprio=14 aiwp=2 aihad=0 aiignore=none aihelp=none",
        "LANERL_INTERNAL t=1000 ai id=88 kind=LaneMinion team=200 x=1920 y=0 "
        "xbits=1123024896 ybits=0 target=0,-,0,0,0 wpkey=1 wps=1920,0 "
        "coll=none collbits=none aacd=0 aastate=0 aacast=0 aadelay=0 "
        "aawindup=0 attacking=0 hasaa=0 aitimer=0 ailocal=92160000 "
        "aitsa=0 aiprio=14 aiwp=2 aihad=0 aiignore=none aihelp=none",
    ]
    snapshot = parse_stream(rows)[0]
    state, report = _state_for(snapshot)
    slot = report.notes[0].slot
    assert "exact tick grid" in report.notes[0].ai_timer_recovery

    out = tick(state, lane_params())
    assert float(out.ai_timer[slot]) == 0.0, (
        "the injected minion did not take its 250 ms sweep: the dump's "
        f"{q_dumped}/1024 ms is one rounding short of the trigger and nothing "
        "put it back on the server's tick grid (INJ-003)")


def test_an_action_timer_that_is_not_on_the_tick_grid_is_left_alone():
    """The snap is a recovery, not an override -- same contract as the
    wind-up's. A value the grid cannot explain means the assumed tick stride
    is wrong, and snapping there would hide that rather than fix it.
    """
    from lanerl_jax.parity.inject import snap_action_timer_to_tick_grid

    value, why = snap_action_timer_to_tick_grid(123.456)
    assert value == 123.456
    assert "not snapped" in why
