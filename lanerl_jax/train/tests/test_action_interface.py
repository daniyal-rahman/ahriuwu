"""The policy's action/observation interface, end to end, in both frames.

Three properties the scripted last-hitter (`train/scripted_policy.py`) relies
on, and so does every trained policy:

(a) screen clicks resolve to living units under the cursor in both frames,
    independently of observation slot ordering;
(b) a MOVE on a screen cell whose raw decode is standable (the rest are
    snapped to the nearest standable cell, `PATH-010`) decodes to a world
    point that projects back into the same cell, seen from the agent's own (red: mirrored) frame;
(c) the features the scripted player reads -- enemy minion HP bar, subtype,
    position, own AD -- equal the underlying ``LaneState`` after the builder's
    documented normalisation, and own AD is the damage a swing really deals.

Round-trip and feature checks also exercise deliberately broken mappings:
shifted cells, red decoded without its mirror, and red features built in
blue's frame.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.obs.builder import (HP_BAR_STEPS, NORM_AD, NORM_DIST,
                                    SLOT_ENEMY_MINION, build_observation)
from lanerl_jax.obs.frame import delta_to_lane, make_lane_frame
from lanerl_jax.sim.combat import growth_sum, post_mitigation_damage
from lanerl_jax.sim.config import SimConfig
from lanerl_jax.sim.init import TOP_LANE_PATH, TOP_OUTER_TURRET, init_lane, spawn_minion
from lanerl_jax.sim.orders import OrderKind
from lanerl_jax.sim.profiles import PROFILES, profile_id
from lanerl_jax.sim.state import Kind, Team
from lanerl_jax.sim.step import env_step
from lanerl_jax.sim.targeting import MinionType
from lanerl_jax.train import scripted_policy as SP
from lanerl_jax.train.actions import move_snap_table, orders_from
from lanerl_jax.train.trainer import BLUE_NEXUS, RED_NEXUS
from lanerl_rl import constants as C
from lanerl_rl.projection import centred_on, world_to_screen

AM = C.BUTTON_INDEX["attack_move"]
MOVE = C.BUTTON_INDEX["move"]

FRAMES = (make_lane_frame(TOP_OUTER_TURRET[Team.BLUE], TOP_OUTER_TURRET[Team.RED],
                          BLUE_NEXUS),
          make_lane_frame(TOP_OUTER_TURRET[Team.RED], TOP_OUTER_TURRET[Team.BLUE],
                          RED_NEXUS))
BLUE_FRAME = FRAMES[0]           # what the trainer passes to orders_from

B_POS = (1500.0, 11900.0)
R_POS = (2300.0, 12700.0)


def _state(level=1, turrets=True):
    """Both champions mid-lane, each facing a mixed, damaged enemy wave."""
    params = SimConfig.unit_test().params
    s = init_lane(include_all_turrets=turrets)
    x = s.x.at[0].set(B_POS[0]).at[1].set(R_POS[0])
    y = s.y.at[0].set(B_POS[1]).at[1].set(R_POS[1])
    s = s.replace(x=x, y=y, collision_x=x, collision_y=y,
                  level=s.level.at[:2].set(level),
                  # xp consistent with the level, or the next tick drops it
                  xp=s.xp.at[:2].set(params["xp_to_reach_level"][level] + 1.0))
    path = jnp.asarray(np.asarray(TOP_LANE_PATH, np.float32))
    rng = np.random.default_rng(0)
    types = (MinionType.MELEE, MinionType.CASTER, MinionType.CANNON)
    hp_fracs = []
    for team, (cx, cy) in ((Team.RED, B_POS), (Team.BLUE, R_POS)):
        for k in range(7):
            mt = types[k % 3]
            prof = profile_id(Kind.LANE_MINION, mt, team)
            ang = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(90, 650)
            s = spawn_minion(s, team, prof, params["max_hp"][prof], path,
                             spawn_xy=(cx + r * np.cos(ang), cy + r * np.sin(ang)))
            hp_fracs.append(rng.uniform(0.03, 1.0))
    # damage them: slot order is spawn order
    alive_m = np.where(np.asarray(s.alive) & (np.asarray(s.kind) == Kind.LANE_MINION))[0]
    hp = np.asarray(s.hp).copy()
    hp[alive_m] = hp[alive_m] * np.asarray(hp_fracs)[: len(alive_m)]
    return s.replace(hp=jnp.asarray(hp)), params


def _obs(state, params, frames=FRAMES):
    return [build_observation(state, i, frames[i], params=params) for i in (0, 1)]


# --------------------------------------------------------------------------
# (a) screen click -> unit, independent of observation slot ordering
# --------------------------------------------------------------------------
@pytest.mark.parametrize("me", [0, 1])
def test_a_click_hits_visible_enemy_in_each_screen_frame(me):
    state, params = _state(turrets=False)
    # Keep only the two champions and put the enemy exactly under a valid
    # screen cell. No slot table is needed to resolve a world-space hit.
    state = state.replace(alive=state.alive.at[2:].set(False))
    action = (jnp.asarray([AM, AM]), jnp.asarray([58, 58]), jnp.asarray([25, 25]))
    empty = jnp.full((2, C.N_SLOTS), -1, jnp.int32)
    click = orders_from(action, state, empty, BLUE_FRAME, snap_moves=False)
    enemy = 1 - me
    state = state.replace(x=state.x.at[enemy].set(click.x[me]),
                          y=state.y.at[enemy].set(click.y[me]))
    orders = orders_from(action, state, empty, BLUE_FRAME)
    assert int(orders.kind[me]) == OrderKind.ATTACK
    assert int(orders.target[me]) == enemy
    # A bogus pointer table cannot change which physical unit is clicked.
    wrong = jnp.full((2, C.N_SLOTS), 57, jnp.int32)
    again = orders_from(action, state, wrong, BLUE_FRAME)
    for a, b in zip(orders, again):
        np.testing.assert_array_equal(a, b)
    # Moving the click away must miss the unit, even with a table naming it.
    missed = orders_from((action[0], jnp.asarray([38, 38]), action[2]),
                         state, jnp.full_like(empty, enemy), BLUE_FRAME)
    assert int(missed.kind[me]) == OrderKind.ATTACK_MOVE
    assert int(missed.target[me]) == -1


@pytest.mark.parametrize("me", [0, 1])
def test_a_dead_unit_under_click_does_not_resolve_to_attack(me):
    state, _ = _state(turrets=False)
    state = state.replace(alive=state.alive.at[2:].set(False))
    action = (jnp.asarray([AM, AM]), jnp.asarray([58, 58]), jnp.asarray([25, 25]))
    slots = jnp.full((2, C.N_SLOTS), 1 - me, jnp.int32)
    click = orders_from(action, state, slots, BLUE_FRAME, snap_moves=False)
    enemy = 1 - me
    state = state.replace(x=state.x.at[enemy].set(click.x[me]),
                          y=state.y.at[enemy].set(click.y[me]),
                          alive=state.alive.at[enemy].set(False))
    orders = orders_from(action, state, slots, BLUE_FRAME)
    assert int(orders.kind[me]) == OrderKind.ATTACK_MOVE
    assert int(orders.target[me]) == -1


# --------------------------------------------------------------------------
# (b) screen cell -> world -> screen cell
# --------------------------------------------------------------------------
_CAM = centred_on(0.0, 0.0)


def _raw_cell_standable(x, y):
    """Is the RAW (pre-snap) decoded point on a cell a 35-u champion can
    stand on?  Only those points are passed through by the `PATH-010` snap;
    the rest are moved to the nearest standable cell centre and so, by
    design, need not project back into the clicked screen cell."""
    t = move_snap_table()
    nx = (float(x) - t.min_x) / t.cell_size
    ny = (float(y) - t.min_y) / t.cell_size
    if not (0 <= nx < t.width and 0 <= ny < t.height):
        return False
    return bool(t.standable[int(ny) * t.width + int(nx)])


def _roundtrip_failures(state, decode_state=None, cell_shift=0):
    """Decode every executable cell for both champions, re-express the world
    point in the champion's OWN frame and project it back to the screen.

    Cells whose raw decode is off-grid or unstandable at r=35 are skipped
    (the snap moves them on purpose, `PATH-010`); returns
    ``(fails, checked, skipped)``."""
    decode_state = state if decode_state is None else decode_state
    _, _, ok = SP.screen_grid()
    ys, xs = np.where(ok)
    sel = np.arange(0, len(xs), 7)             # ~700 of 5k cells, spread out
    xs, ys = xs[sel], ys[sel]
    fails = checked = skipped = 0
    slots = jnp.full((2, C.N_SLOTS), -1, jnp.int32)
    for sx, sy in zip(xs, ys):
        action = (jnp.asarray([MOVE, MOVE]), jnp.asarray([sx + cell_shift] * 2),
                  jnp.asarray([sy, sy]))
        o = orders_from(action, decode_state, slots, BLUE_FRAME)
        raw = orders_from(action, decode_state, slots, BLUE_FRAME,
                          snap_moves=False)
        for me in (0, 1):
            if not _raw_cell_standable(raw.x[me], raw.y[me]):
                skipped += 1
                continue
            checked += 1
            ds, dn = delta_to_lane(FRAMES[me], o.x[me] - state.x[me],
                                   o.y[me] - state.y[me])
            px, py = world_to_screen(_CAM, float(ds), float(dn))
            cx, cy = int(np.floor(px * C.N_SCREEN_X)), int(np.floor(py * C.N_SCREEN_Y))
            if (cx, cy) != (sx, sy):
                fails += 1
    return fails, checked, skipped


def test_b_move_cell_round_trips_in_both_frames():
    state, _ = _state()
    fails, n, skipped = _roundtrip_failures(state)
    # 453 of the 1,380 sampled (cell, side) pairs decode off-grid or onto an
    # unstandable cell from these mid-lane positions and are snapped (452 of
    # them no longer round-trip; one snapped centre still lands in its cell).
    assert n > 900 and fails == 0 and skipped == 453, (fails, n, skipped)


def test_b_detects_red_decoded_without_its_mirror():
    state, _ = _state()
    # the decoder picks the red flip from state.team; lie that red is blue
    no_mirror = state.replace(team=state.team.at[1].set(Team.BLUE))
    fails, n, _ = _roundtrip_failures(state, decode_state=no_mirror)
    assert fails > n // 4, (fails, n)


def test_b_detects_cell_off_by_one():
    state, _ = _state()
    fails, n, _ = _roundtrip_failures(state, cell_shift=1)
    assert n > 900 and fails == n, (fails, n)


def test_b_scripted_click_on_a_minion_lands_on_it():
    """The scripted player's `cell_for_offset` click on each visible enemy
    minion decodes to a world point within one cell of that minion."""
    state, params = _state()
    obs = _obs(state, params)
    worst = 0.0
    for me in (0, 1):
        o = obs[me]
        for k in range(*SLOT_ENEMY_MINION):
            if o.entities[k, 0] < 0.5:
                continue
            ds = o.entities[k, 1] * NORM_DIST
            dn = o.entities[k, 2] * NORM_DIST
            if float(jnp.hypot(ds, dn)) > SP.CLICK_RADIUS:
                continue
            sx, sy = SP.cell_for_offset(ds, dn)
            action = (jnp.asarray([MOVE, MOVE]), jnp.asarray([sx, sx]),
                      jnp.asarray([sy, sy]))
            ords = orders_from(action, state, jnp.stack([q.slot_unit for q in obs]),
                               BLUE_FRAME)
            u = int(o.slot_unit[k])
            d = float(jnp.hypot(ords.x[me] - state.x[u], ords.y[me] - state.y[u]))
            worst = max(worst, d)
    assert worst < 40.0, worst


# --------------------------------------------------------------------------
# (c) features vs LaneState
# --------------------------------------------------------------------------
def _feature_failures(state, obs, params, slot_tables=None, frames=FRAMES):
    tab = SP.static_tables()
    fails = []
    for me in (0, 1):
        o = obs[me]
        table = o.slot_unit if slot_tables is None else slot_tables[me]
        for k in range(*SLOT_ENEMY_MINION):
            if o.entities[k, 0] < 0.5:
                continue
            u = int(table[k])
            e = np.asarray(o.entities[k])
            # position, own frame
            ds, dn = delta_to_lane(frames[me], state.x[u] - state.x[me],
                                   state.y[u] - state.y[me])
            if not np.allclose(e[1:3] * NORM_DIST, [float(ds), float(dn)], atol=0.05):
                fails.append((me, k, "pos"))
            # distance the scripted player derives == true centre distance
            true_d = float(jnp.hypot(state.x[u] - state.x[me], state.y[u] - state.y[me]))
            if abs(float(np.hypot(*(e[1:3] * NORM_DIST))) - true_d) > 0.05:
                fails.append((me, k, "dist"))
            # subtype -> max hp / armour the scripted player assumes
            sub = e[13:16]
            if abs(float(sub @ tab["max_hp"]) - float(state.max_hp[u])) > 1e-3:
                fails.append((me, k, "max_hp"))
            if abs(float(sub @ tab["armor"])
                   - float(params["armor"][int(state.model[u])])) > 1e-4:
                fails.append((me, k, "armor"))
            # hp bar and the scripted upper bound brackets true hp
            true = float(state.hp[u])
            if abs(e[3] - np.round(true / float(state.max_hp[u]) * HP_BAR_STEPS)
                   / HP_BAR_STEPS) > 1e-6:
                fails.append((me, k, "hp_frac"))
            hi = (e[3] + 0.5 / HP_BAR_STEPS) * float(sub @ tab["max_hp"])
            if not (hi + 1e-3 >= true >= hi - float(state.max_hp[u]) / HP_BAR_STEPS - 1e-3):
                fails.append((me, k, "hp_bound"))
        # own AD
        m = int(state.model[me])
        ad = float(params["attack_damage"][m] + params["ad_per_level"][m]
                   * growth_sum(state.level[me], jnp))
        if abs(float(o.self_vec[C.S_AD]) * NORM_AD - ad) > 1e-3:
            fails.append((me, "ad"))
    return fails


@pytest.mark.parametrize("level", [1, 6])
def test_c_scripted_features_equal_lane_state_both_frames(level):
    state, params = _state(level=level)
    obs = _obs(state, params)
    assert _feature_failures(state, obs, params) == []


def test_c_detects_red_features_in_blue_frame():
    state, params = _state()
    bad = _obs(state, params, frames=(FRAMES[0], FRAMES[0]))
    fails = _feature_failures(state, bad, params)
    assert any(f[0] == 1 and f[2] == "pos" for f in fails)


def test_c_detects_slot_table_off_by_one():
    state, params = _state()
    obs = _obs(state, params)
    shifted = [jnp.roll(o.slot_unit, 1) for o in obs]
    assert len(_feature_failures(state, obs, params, slot_tables=shifted)) > 5


@pytest.mark.parametrize("mtype,level", [(MinionType.MELEE, 1),
                                         (MinionType.CANNON, 1),
                                         (MinionType.CASTER, 6)])
def test_c_obs_ad_is_the_damage_a_swing_deals(mtype, level):
    """Isolated arena: blue attacks one red minion in range through
    `orders_from`; the first hp drop equals obs AD after the armour the
    scripted player assumes for that subtype."""
    params = SimConfig.unit_test().params
    sim = SimConfig.training(route_artifact=None)
    s = init_lane(include_all_turrets=False)
    x = s.x.at[0].set(B_POS[0]).at[1].set(B_POS[0] + 3000)
    y = s.y.at[0].set(B_POS[1]).at[1].set(B_POS[1] + 3000)
    s = s.replace(x=x, y=y, collision_x=x, collision_y=y,
                  level=s.level.at[:2].set(level),
                  # xp consistent with the level, or the next tick drops it
                  xp=s.xp.at[:2].set(params["xp_to_reach_level"][level] + 1.0))
    prof = profile_id(Kind.LANE_MINION, mtype, Team.RED)
    path = jnp.asarray(np.asarray(TOP_LANE_PATH, np.float32))
    s = spawn_minion(s, Team.RED, prof, params["max_hp"][prof], path,
                     spawn_xy=(B_POS[0] + 120.0, B_POS[1]))
    obs = _obs(s, params)
    k = int(np.argmax(np.asarray(obs[0].slot_unit) == 2))
    assert int(obs[0].slot_unit[k]) == 2 and k >= SLOT_ENEMY_MINION[0]
    tab = SP.static_tables()
    sub = np.asarray(obs[0].entities[k, 13:16])
    want = float(obs[0].self_vec[C.S_AD]) * NORM_AD * 100.0 / (100.0 + float(sub @ tab["armor"]))
    sx, sy = SP.cell_for_offset(obs[0].entities[k, 1] * NORM_DIST,
                                obs[0].entities[k, 2] * NORM_DIST)
    action = (jnp.asarray([AM, 0]), jnp.asarray([sx, 0]), jnp.asarray([sy, 0]))
    orders = orders_from(action, s, jnp.stack([o.slot_unit for o in obs]), BLUE_FRAME)
    hp0 = float(s.hp[2])
    drop = 0.0
    for _ in range(40):
        s = env_step(s, orders, sim)
        orders = orders._replace(kind=jnp.zeros_like(orders.kind))   # noop after
        drop = hp0 - float(s.hp[2])
        if drop > 0:
            break
    assert drop == pytest.approx(want, abs=1e-3), (drop, want)
    ref = float(post_mitigation_damage(
        float(params["attack_damage"][0] + params["ad_per_level"][0]
              * growth_sum(level, np)), float(params["armor"][prof])))
    assert want == pytest.approx(ref, abs=1e-3)


def test_c_no_attack_readiness_feature_exists():
    """Documented limitation, pinned: the self vector carries no auto-attack
    cooldown/windup, so no player can time a swing from the observation."""
    names = [n for n in dir(C) if n.startswith("S_")]
    assert not any(("AA" in n or "ATTACK" in n or "SWING" in n) for n in names), names
