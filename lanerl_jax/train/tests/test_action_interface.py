"""The policy's action/observation interface, end to end, in both frames.

Three properties the scripted last-hitter (`train/scripted_policy.py`) relies
on, and so does every trained policy:

(a) every valid target slot decodes to an ATTACK on exactly the unit whose
    features the observation wrote into that slot;
(b) a MOVE on a screen cell whose raw decode is standable (the rest are
    snapped to the nearest standable cell, `PATH-010`) decodes to a world
    point that projects back into the same cell, seen from the agent's own (red: mirrored) frame;
(c) the features the scripted player reads -- enemy minion HP bar, subtype,
    position, own AD -- equal the underlying ``LaneState`` after the builder's
    documented normalisation, and own AD is the damage a swing really deals.

Each check is a function returning its failures, so every test also shows the
check FAILS on a deliberately broken interface: slot table off by one, blue's
slot table used for red, red decoded without its mirror, red's features built
in blue's frame.
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


def _expected_row(state, me, u, frame):
    """The 16 features of unit ``u`` for observer ``me``, rebuilt from LaneState."""
    dx = state.x[u] - state.x[me]
    dy = state.y[u] - state.y[me]
    ds, dn = delta_to_lane(frame, dx, dy)
    hp = np.round(float(state.hp[u] / state.max_hp[u]) * HP_BAR_STEPS) / HP_BAR_STEPS
    k = int(state.kind[u])
    t = int(state.team[u])
    mine = int(state.team[me])
    row = np.zeros(16, np.float32)
    row[0] = 1
    row[1], row[2], row[3] = float(ds) / NORM_DIST, float(dn) / NORM_DIST, hp
    row[4:10] = [k == Kind.CHAMPION, k == Kind.LANE_MINION, k == Kind.TURRET, 0, 0, 0]
    row[10:13] = [t == mine, t != mine, t == Team.NEUTRAL]
    if k == Kind.LANE_MINION:
        mt = PROFILES[int(state.model[u])][1]
        row[13:16] = [mt == MinionType.MELEE, mt == MinionType.CASTER,
                      mt == MinionType.CANNON]
    return row


# --------------------------------------------------------------------------
# (a) slot -> unit
# --------------------------------------------------------------------------
def _slot_failures(state, obs, slot_tables):
    """For every valid slot of each champion, press attack_move on it and check
    the order is ATTACK on the unit whose LaneState reproduces that row."""
    fails = []
    for k in range(C.N_SLOTS):
        valid = [bool(o.entities[k, 0] > 0.5) for o in obs]
        action = (jnp.asarray([AM, AM]), jnp.asarray([48, 48]),
                  jnp.asarray([27, 27]), jnp.asarray([k, k]))
        orders = orders_from(action, state, jnp.stack(slot_tables), BLUE_FRAME)
        for me in (0, 1):
            if not valid[me]:
                continue
            if int(orders.kind[me]) != OrderKind.ATTACK:
                fails.append((me, k, "not ATTACK"))
                continue
            u = int(orders.target[me])
            row = np.asarray(obs[me].entities[k])
            want = _expected_row(state, me, u, FRAMES[me])
            if not np.allclose(row, want, atol=1e-5):
                fails.append((me, k, u))
    return fails


def test_a_every_valid_slot_attacks_the_unit_in_that_slot_both_frames():
    state, params = _state()
    obs = _obs(state, params)
    n_valid = [int((o.entities[:, 0] > 0.5).sum()) for o in obs]
    n_enemy_min = [int((o.entities[slice(*SLOT_ENEMY_MINION), 0] > 0.5).sum())
                   for o in obs]
    assert min(n_enemy_min) >= 5 and min(n_valid) >= 10, (n_valid, n_enemy_min)
    assert _slot_failures(state, obs, [o.slot_unit for o in obs]) == []


def test_a_detects_slot_table_off_by_one():
    state, params = _state()
    obs = _obs(state, params)
    shifted = [jnp.roll(o.slot_unit, 1) for o in obs]
    assert len(_slot_failures(state, obs, shifted)) > 10


def test_a_detects_red_decoded_with_blue_slot_table():
    state, params = _state()
    obs = _obs(state, params)
    swapped = [obs[0].slot_unit, obs[0].slot_unit]      # red reads blue's table
    fails = _slot_failures(state, obs, swapped)
    assert any(me == 1 for me, *_ in fails)


def test_a_invalid_slot_is_not_an_attack():
    state, params = _state()
    obs = _obs(state, params)
    for me in (0, 1):
        empty = np.where(np.asarray(obs[me].entity_pad_mask))[0]
        assert len(empty)
        k = int(empty[0])
        action = (jnp.asarray([AM, AM]), jnp.asarray([48, 48]),
                  jnp.asarray([27, 27]), jnp.asarray([k, k]))
        orders = orders_from(action, state,
                             jnp.stack([o.slot_unit for o in obs]), BLUE_FRAME)
        assert int(orders.kind[me]) == OrderKind.MOVE


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
                  jnp.asarray([sy, sy]), jnp.asarray([0, 0]))
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
                      jnp.asarray([sy, sy]), jnp.asarray([0, 0]))
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
    action = (jnp.asarray([AM, 0]), jnp.asarray([48, 0]), jnp.asarray([27, 0]),
              jnp.asarray([k, 0]))
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
