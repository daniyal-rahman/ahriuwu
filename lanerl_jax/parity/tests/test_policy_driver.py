"""PARITY-001: the creation-rank NetId map, the policy action log, the driver.

No server and no network: the policy's jitted act is replaced by a stub, so
what is tested is the plumbing the gate's correctness rests on -- that a
server NetId names the same unit as a sim slot, and that the log carries
enough to re-derive it.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.parity import policy_driver as pd
from lanerl_jax.parity.policy_driver import (CreationRankMap, PolicyActionLog,
                                             PolicyPairDriver)
from lanerl_jax.sim.init import TOP_LANE_PATH, init_lane, lane_params, spawn_minion
from lanerl_jax.sim.orders import OrderKind
from lanerl_jax.sim.profiles import PROFILES
from lanerl_jax.sim.state import CH_SLICE, MI_SLICE, TU_SLICE, Kind, Team

REPO = Path(__file__).resolve().parents[3]


def _spawn_wave(state, n_pairs: int):
    """`step.py`'s wave write: red then blue, pair by pair."""
    params = lane_params()
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))
    red_row = PROFILES.index((Kind.LANE_MINION, 0, Team.RED))
    blue_row = PROFILES.index((Kind.LANE_MINION, 0, Team.BLUE))
    for _ in range(n_pairs):
        state = spawn_minion(state, Team.RED, red_row,
                             params["max_hp"][red_row], path[::-1])
        state = spawn_minion(state, Team.BLUE, blue_row,
                             params["max_hp"][blue_row], path)
    return state


def _wire(state, netid_of_seq: dict, t: int = 0) -> dict:
    """A control-channel frame listing every live unit of ``state``."""
    s = {k: np.asarray(getattr(state, k)) for k in
         ("kind", "alive", "team", "x", "y", "spawn_seq")}
    units = []
    for i in np.flatnonzero(s["alive"] & (s["kind"] != Kind.NONE)):
        k = int(s["kind"][i])
        tm = 100 if int(s["team"][i]) == Team.BLUE else 200
        name = {Kind.CHAMPION: "Champion", Kind.LANE_MINION: "LaneMinion",
                Kind.TURRET: "LaneTurret"}[k]
        units.append({"id": netid_of_seq[int(s["spawn_seq"][i])], "k": name,
                      "tm": tm, "x": float(s["x"][i]), "y": float(s["y"][i]),
                      "hp": 100, "mhp": 100, "mt": 0, "dead": False,
                      "ad": 78., "ap": 0., "ar": 30., "mr": 30., "se": [1, 1, 1, 1]})
    return {"t": t, "u": units}


def _two_wave_state():
    """Wave 1 (3 pairs), two of its minions die, wave 2 (3 pairs) reuses slots.

    Server NetIds are a creation-order counter with GAPS (missiles, buffs and
    particles take NetIds too), and the map's turrets carry fixed high ids.
    """
    base = init_lane()
    seq = np.asarray(base.spawn_seq)
    netid_of_seq = {}
    for i in range(TU_SLICE.start, TU_SLICE.stop):
        netid_of_seq[int(seq[i])] = 0xFF000000 + i            # map objects
    netid_of_seq[int(seq[CH_SLICE.start + Team.BLUE])] = 0x40000010
    netid_of_seq[int(seq[CH_SLICE.start + Team.RED])] = 0x40000011
    first = int(base.next_spawn_seq)
    for r in range(12):                                      # 12 minions total
        netid_of_seq[first + r] = 0x40000100 + 67 * r + (r % 3)
    s1 = _spawn_wave(base, 3)
    # kill the first red and the second blue minion of wave 1
    seq1 = np.asarray(s1.spawn_seq)
    alive = np.asarray(s1.alive).copy()
    dead_seqs = (first + 0, first + 3)
    for d in dead_seqs:
        alive[int(np.flatnonzero((seq1 == d) & alive)[0])] = False
    s1k = s1.replace(alive=jnp.asarray(alive))
    s2 = _spawn_wave(s1k, 3)
    return base, s1, s2, netid_of_seq, first, dead_seqs


def test_creation_rank_map_on_a_synthetic_two_wave_spawn():
    base, s1, s2, netid_of_seq, first, dead_seqs = _two_wave_state()
    m = CreationRankMap(base)
    assert m.first_minion_seq == first
    # Frames as the server would send them: wave 1 alive, then after the
    # deaths and wave 2. The dead minions' NetIds were seen in the first frame.
    m.observe(_wire(s1, netid_of_seq))
    m.observe(_wire(s2, netid_of_seq))

    f = {k: np.asarray(getattr(s2, k)) for k in ("spawn_seq", "alive", "kind", "team")}
    # every live sim minion is named by exactly the NetId of its creation rank
    live = np.flatnonzero(f["alive"] & (f["kind"] == Kind.LANE_MINION))
    assert len(live) == 10
    for i in live:
        nid = netid_of_seq[int(f["spawn_seq"][i])]
        assert m.sim_slot(nid, f["spawn_seq"], f["alive"], f["kind"]) == int(i)
        assert m.spawn_seq_of(nid) == int(f["spawn_seq"][i])
        assert m.netid_of_spawn_seq(int(f["spawn_seq"][i])) == nid
    # wave 2 reused the dead minions' slots: the dead NetIds must NOT resolve
    # to the new occupants (that would retarget an attack onto another unit)
    for d in dead_seqs:
        assert m.sim_slot(netid_of_seq[d], f["spawn_seq"], f["alive"], f["kind"]) is None
    reused = [int(i) for i in live if int(f["spawn_seq"][i]) >= first + 6]
    assert set(reused) & set(int(np.flatnonzero(np.asarray(s1.spawn_seq) == d)[0])
                             for d in dead_seqs)
    # red before blue: even ranks are red, odd ranks blue, in both engines
    for r, nid in enumerate(m.minions):
        i = m.sim_slot(nid, f["spawn_seq"], f["alive"], f["kind"])
        if i is not None:
            assert int(f["team"][i]) == (Team.RED if r % 2 == 0 else Team.BLUE)
    # champions by team, turrets by position
    assert m.sim_slot(0x40000010, f["spawn_seq"], f["alive"], f["kind"]) == 0
    assert m.sim_slot(0x40000011, f["spawn_seq"], f["alive"], f["kind"]) == 1
    ti = TU_SLICE.start + 5
    assert m.sim_slot(0xFF000000 + ti, f["spawn_seq"], f["alive"], f["kind"]) == ti
    # the table decision_to_orders consumes holds every resolvable NetId
    table = m.slot_table(f["spawn_seq"], f["alive"], f["kind"])
    assert all(table[netid_of_seq[int(f["spawn_seq"][i])]] == int(i) for i in live)
    assert not any(netid_of_seq[d] in table for d in dead_seqs)


def test_rank_map_persists_and_refuses_a_different_placement():
    base, s1, s2, netid_of_seq, first, _ = _two_wave_state()
    m = CreationRankMap(base)
    m.observe(_wire(s2, netid_of_seq))
    back = CreationRankMap(base).load_json(json.loads(json.dumps(m.to_json())))
    assert back.minions == m.minions and back.champions == m.champions
    assert back.turrets == m.turrets
    bad = dict(m.to_json(), first_minion_seq=first + 1)
    with pytest.raises(ValueError, match="first_minion_seq"):
        CreationRankMap(base).load_json(bad)


def test_policy_action_log_round_trip(tmp_path):
    log = PolicyActionLog(meta={"checkpoint": "x"})
    log.append(16, {"t": "cast", "slot": 2}, {"t": "noop"},
               {"kind": OrderKind.CAST_E, "x": 1.0, "y": 2.0, "target": -1,
                "target_netid": 0, "target_spawn_seq": None}, None)
    log.append(50, {"t": "attack", "id": 1073743694}, {"t": "level", "slot": 2},
               {"kind": OrderKind.ATTACK, "x": 0.0, "y": 0.0, "target": 7,
                "target_netid": 1073743694, "target_spawn_seq": 26},
               {"kind": OrderKind.NOOP, "rank_up": True, "slot": 2})
    log.ranks = {"first_minion_seq": 26, "champions": {"0": 5, "1": 6},
                 "turrets": {}, "minions": [1073743694]}
    p = tmp_path / "log.json"
    log.save(p)
    back = PolicyActionLog.load(p)
    assert back.to_json() == json.loads(json.dumps(log.to_json()))
    wire = back.to_action_log()
    assert wire.t_ms == [16, 50] and wire.blue[1] == {"t": "attack", "id": 1073743694}
    assert back.rank_map().spawn_seq_of(1073743694) == 26
    d = log.to_json()
    d["red_sim"] = d["red_sim"][:1]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="lengths disagree"):
        PolicyActionLog.load(p)


def _stub_act(target_slot: int):
    def make(policy, params, *, deterministic, team):
        def act(state, key, visibility=None, hud=None):
            return (OrderKind.ATTACK, 0.0, 0.0, target_slot, pd.BUTTONS.index("attack_move"))
        return act
    return make


def test_pair_driver_logs_wire_netid_and_spawn_seq(monkeypatch):
    base, s1, s2, netid_of_seq, first, _ = _two_wave_state()
    monkeypatch.setattr(pd, "_make_act", _stub_act(MI_SLICE.start))
    drv = PolicyPairDriver(None, None, red="policy")
    frame = _wire(s1, netid_of_seq, t=90014)
    for u in frame["u"]:
        if u["k"] == "Champion":            # ranks already at the table's level
            u.update(lvl=1, sl=[0, 0, 1, 0], cd0=0, cd1=0, cd2=0, cd3=0)
    out = drv(frame, 0)
    # the rebuilder puts the first minion ON THE WIRE in the first free slot
    first_wire_minion = next(u for u in frame["u"] if u["k"] == "LaneMinion")
    nid = first_wire_minion["id"]
    assert out["blue"] == {"t": "click", "button": "attack_move", "x": 0.0, "y": 0.0} == out["red"]
    rec = drv.log.blue_sim[0]
    assert rec["kind"] == OrderKind.ATTACK and rec["target"] == MI_SLICE.start
    assert rec["target_netid"] == nid
    inv = {v: k for k, v in netid_of_seq.items()}
    assert rec["target_spawn_seq"] == inv[nid]
    assert drv.log.t_ms == [90014] and len(drv.log) == 1
    assert drv.log.rank_map().spawn_seq_of(nid) == inv[nid]
    # a champion missing a rank gets a `level` order, logged as a sim NOOP
    for u in frame["u"]:
        if u["k"] == "Champion" and u["tm"] == 200:
            u["sl"] = [0, 0, 0, 0]
    frame["t"] = 90048
    out = drv(frame, 1)
    assert out["red"] == {"t": "level", "slot": 2}
    assert drv.log.red_sim[1]["rank_up"] and drv.log.red_sim[1]["kind"] == OrderKind.NOOP
    assert drv.counts["red"]["level"] == 1


def test_pair_driver_idle_red_sends_noops(monkeypatch):
    monkeypatch.setattr(pd, "_make_act", _stub_act(-1))
    drv = PolicyPairDriver(None, None, red="idle")
    frame = {"t": 16, "u": [
        {"id": 5, "k": "Champion", "tm": 100, "x": 0, "y": 0, "hp": 1, "mhp": 1, "dead": False,
         "lvl": 1, "sl": [0, 0, 1, 0], "ad": 78, "ap": 0, "ar": 30, "mr": 30, "se": [1, 1, 1, 1]},
        {"id": 6, "k": "Champion", "tm": 200, "x": 9, "y": 9, "hp": 1, "mhp": 1, "dead": False,
         "lvl": 1, "sl": [0, 0, 0, 0]}]}
    out = drv(frame, 0)
    assert out["red"] == {"t": "noop"} and drv.log.red_sim[0] is None
    assert drv(None, 1) is None and len(drv.log) == 1


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "rl_eval_vs_server_under_test", REPO / "tools" / "rl_eval_vs_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_tool_imports_the_one_driver_copy():
    tool = _load_tool()
    assert tool.make_driver is pd.make_driver
    assert tool.StateRebuilder is pd.StateRebuilder
    assert tool.order_to_wire is pd.order_to_wire


def test_eval_replay_maps_attacks_through_creation_rank(tmp_path):
    """`--replay`: slot -> spawn_seq -> NetId, instead of dropping the attack."""
    tool = _load_tool()
    base, s1, s2, netid_of_seq, first, _ = _two_wave_state()
    lines = [
        {"kind": OrderKind.ATTACK, "x": 0, "y": 0, "target": 9,
         "target_spawn_seq": first + 1},                       # a blue minion
        {"kind": OrderKind.ATTACK, "x": 0, "y": 0, "target": 1,
         "target_spawn_seq": int(np.asarray(base.spawn_seq)[1])},  # red champion
        {"kind": OrderKind.ATTACK, "x": 0, "y": 0, "target": 3},   # old format
        {"kind": OrderKind.ATTACK, "x": 0, "y": 0, "target": 3,
         "target_spawn_seq": first + 40},                      # never created
        {"kind": OrderKind.MOVE, "x": 10.0, "y": 20.0, "target": -1},
    ]
    p = tmp_path / "orders.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines))
    counts = {}
    drive = tool.replay_driver(p, counts)
    # The driver sees every frame of a live run, so it has seen wave 1 BEFORE
    # two of its minions died. Rank is only correct over the COMPLETE NetId
    # history -- a map that had only seen the post-death frame would shift
    # every later rank by two (that is the failure this line prevents).
    drive.ranks.observe(_wire(s1, netid_of_seq))
    frame = _wire(s2, netid_of_seq)
    got = [drive(frame) for _ in lines]
    assert got[0] == {"t": "attack", "id": netid_of_seq[first + 1]}
    assert got[1] == {"t": "attack", "id": 0x40000011}
    assert got[2] == {"t": "noop"} and got[3] == {"t": "noop"}
    assert got[4] == {"t": "move", "x": 10.0, "y": 20.0}
    assert counts["attack_mapped"] == 2 and counts["attack_dropped"] == 2


# ---------------------------------------------------------------------------
# OBS-04 / OBS-05 / OBS-06: the rebuilt state means what the sim's state means
# ---------------------------------------------------------------------------

def _champs(t, *, blue=None, red=None):
    """A frame with the two champions only; ``blue``/``red`` override fields.
    Red stands 400 u from blue: visible (champion vision 1200) and on screen."""
    def one(nid, tm, x, y, over):
        u = {"id": nid, "k": "Champion", "tm": tm, "x": x, "y": y, "hp": 600,
             "mhp": 672, "gold": 475, "lvl": 1, "cs": 0, "rc": 0,
             "vb": 1, "vr": 1, "dead": False,
             "sl": [0, 0, 1, 0], "cd0": 0, "cd1": 0, "cd2": 0, "cd3": 0}
        u.update(over or {})
        return u
    return {"t": t, "u": [one(11, 100, 5000, 9000, blue),
                          one(12, 200, 5400, 9000, red)]}


@pytest.mark.parametrize('team', [0, 1])
def test_hidden_wire_fields_cannot_change_actual_policy_inputs(team):
    """Exercise raw wire -> rebuilder -> fog/HUD -> actual policy call."""
    import copy
    from lanerl_jax.train.policy import ActionLogits
    from lanerl_rl.constants import N_SCREEN_X, N_SCREEN_Y
    captured = []
    class CapturePolicy:
        def apply(self, params, *actor_inputs):
            assert len(actor_inputs) == 4  # No IDs/diagnostic slot table.
            jax.debug.callback(lambda *a: captured.append(tuple(np.array(x) for x in a)),
                               *actor_inputs, ordered=True)
            return ActionLogits(jnp.zeros(len(pd.BUTTONS)).at[pd.BUTTONS.index('noop')].set(9),
                                jnp.zeros(N_SCREEN_X), jnp.zeros(N_SCREEN_Y), jnp.float32(0))
    base = _champs(1000)
    for unit in base['u']:
        unit.update(ad=78., ap=0., ar=30., mr=30., se=[1,1,1,1])
    key = 'vb' if team == 0 else 'vr'
    hidden = copy.deepcopy(base)
    hidden['t'] = 1033
    hidden['u'][1-team][key] = 0
    changed = copy.deepcopy(hidden)
    changed['u'][1-team].update(id=987654, hp=17., mhp=999., gold=9000., cs=53,
        lvl=18, sl=[5,5,5,3], cd0=5000, cd1=6000, cd2=7000, cd3=8000,
        ad=400., ar=200., mr=150., x=5300., y=9200.)
    changed['u'].reverse()
    results = []
    for last in (hidden, changed):
        driver = pd.PolicyDriver(CapturePolicy(), {}, team=team, deterministic=True)
        # Equal witnessed history, then one hidden frame before perturbation.
        driver.step(base)
        driver.step(hidden)
        final = copy.deepcopy(last); final['t'] = 1066
        driver.step(final)
        jax.effects_barrier()
        results.append(captured[-1])
    for a, b in zip(*results):
        np.testing.assert_array_equal(a, b)
    # Non-vacuity: the actual call sees own HUD and excludes the enemy row.
    assert results[0][0].shape == (32, 16)
    assert bool(results[0][1][0])
    assert results[0][2][10] == np.float32(78./200.)


def test_gold_is_earnings_not_wallet():
    """OBS-04: the sim's gold is earnings from 0 and never drops."""
    from lanerl_jax.parity.policy_divergence import STARTING_GOLD as DIV_START
    assert pd.STARTING_GOLD == DIV_START == 475.0
    rb = pd.StateRebuilder()
    seq = [(16, 475, 0.0), (89_984, 475, 0.0),       # no ambient before 90 s
           (90_548, 476, 1.0), (91_081, 477, 2.0),   # ambient, floor-rounded
           (95_000, 498, 23.0),                      # a 21 g last-hit
           (95_033, 48, 23.0),                       # a PURCHASE: not income
           (96_000, 50, 25.0)]
    for t, wallet, want in seq:
        st, _ = rb.rebuild(_champs(t, blue={"gold": wallet}, red={"gold": 475}))
        assert float(st.gold[CH_SLICE.start + Team.BLUE]) == pytest.approx(want), t
        assert float(st.gold[CH_SLICE.start + Team.RED]) == 0.0
    # first sight mid-game: whatever is above the starting wallet was earned
    st, _ = pd.StateRebuilder().rebuild(_champs(200_000, blue={"gold": 861}))
    assert float(st.gold[0]) == pytest.approx(386.0)


def _obs_ms(rb, frame):
    st, _ = rb.rebuild(frame)
    return np.asarray(st.observed_enemy_cast_ms)


def test_witnessed_enemy_cast_from_cooldown_rises():
    """OBS-05: mirrors `orders._record_observed_enemy_casts`.

    An E spin rises TWICE on the wire (GarenECancel's ~1000 ms, then the rank
    cooldown at the end) and is ONE cast; the clock starts at the frame the
    order was issued (the previous frame), as the sim's does at ingress.
    """
    rb = pd.StateRebuilder()
    m = _obs_ms(rb, _champs(1000))
    assert (m == -1.0).all()                             # never seen
    m = _obs_ms(rb, _champs(1033, red={"cd2": 983}))     # red E start
    assert m[Team.BLUE, 2] == pytest.approx(33.0)        # pressed at 1000
    assert m[Team.RED, 2] == -1.0                        # not its own cast
    assert (m[Team.BLUE, [0, 1, 3]] == -1.0).all()
    m = _obs_ms(rb, _champs(2500, red={"cd2": 0}))
    assert m[Team.BLUE, 2] == pytest.approx(1500.0)
    # the spin ENDS: cd2 rises to the rank cooldown -- not a second cast
    m = _obs_ms(rb, _champs(4033, red={"cd2": 9000}))
    assert m[Team.BLUE, 2] == pytest.approx(3033.0)
    # W: its cooldown starts at the press
    m = _obs_ms(rb, _champs(4066, red={"cd1": 24000, "cd2": 8967}))
    assert m[Team.BLUE, 1] == pytest.approx(33.0)
    assert m[Team.BLUE, 2] == pytest.approx(3066.0)
    # R: the cooldown lands at FinishCasting, 435 ms after the press
    m = _obs_ms(rb, _champs(5000, red={"cd1": 23000, "cd2": 8000,
                                       "cd3": 160000}))
    assert m[Team.BLUE, 3] == pytest.approx(5000 - 4066 + 435.0)
    # and symmetric: red witnesses blue
    m = _obs_ms(rb, _champs(5033, blue={"cd2": 990},
                            red={"cd1": 23000, "cd2": 8000, "cd3": 160000}))
    assert m[Team.RED, 2] == pytest.approx(33.0)


def test_unwitnessed_casts_are_not_recorded():
    """Off screen (> 1800 u), fogged, or a dead observer: nothing recorded."""
    for red_pos, blue_over in (({"x": 7000}, None),   # 2000 u: off screen
                               ({"x": 6500, "vb": 0}, None),  # authoritative fog
                               ({"vb": 0}, None),    # nearby but hidden in brush
                               ({}, {"hp": 0, "dead": True})):      # observer dead
        rb = pd.StateRebuilder()
        rb.rebuild(_champs(1000, red=red_pos, blue=blue_over))
        m = _obs_ms(rb, _champs(1033, red={**red_pos, "cd2": 983},
                                blue=blue_over))
        assert m[Team.BLUE, 2] == -1.0, (red_pos, blue_over)
    # judged on the frame the order was ISSUED: red walks into view on the
    # rise frame, but was fogged when it pressed
    rb = pd.StateRebuilder()
    rb.rebuild(_champs(1000, red={"x": 6500, "vb": 0}))
    m = _obs_ms(rb, _champs(1033, red={"cd2": 983}))
    assert m[Team.BLUE, 2] == -1.0
    # the first frame has no previous cooldown, so it is never an edge
    m = _obs_ms(pd.StateRebuilder(), _champs(16, red={"cd2": 983}))
    assert (m == -1.0).all()


def test_observed_cast_reaches_the_observation():
    """The rebuilt memory lands in `global_vec` as the builder renders it."""
    from lanerl_jax.obs.builder import build_observation
    from lanerl_jax.sim.spells import E_COOLDOWNS
    rb = pd.StateRebuilder()
    rb.rebuild(_champs(1000))
    st, _ = rb.rebuild(_champs(1033, red={"cd2": 983}))
    blue_frame, _ = pd._lane_frames()
    g = np.asarray(build_observation(st, 0, blue_frame,
                                     params=lane_params()).global_vec)
    assert g[2 + 2] == pytest.approx(33.0 / (E_COOLDOWNS[0] * 1000.0), abs=1e-6)
    assert g[2 + 1] == 1.0                               # W never seen


def test_w_passive_granted_from_the_first_w_rank_and_sticky():
    """OBS-06: `grant_w_passive` -- the builder's armour/MR read it."""
    from lanerl_jax.obs.builder import build_observation
    rb = pd.StateRebuilder()
    blue_frame, _ = pd._lane_frames()
    p = lane_params()
    st, _ = rb.rebuild(_champs(16, blue={"lvl": 3, "sl": [1, 0, 1, 0]}))
    assert not bool(st.buffs.w_passive[0])
    ar0 = float(build_observation(st, 0, blue_frame, params=p).self_vec[12])
    st, _ = rb.rebuild(_champs(50, blue={"lvl": 3, "sl": [1, 1, 1, 0]}))
    assert bool(st.buffs.w_passive[0]) and not bool(st.buffs.w_passive[1])
    ar1 = float(build_observation(st, 0, blue_frame, params=p).self_vec[12])
    # (base*0.8 + 9)*1.2 against base + 9: the rune's flat armour gains 20%
    assert ar1 > ar0
    st, _ = rb.rebuild(_champs(80, blue={"lvl": 3, "sl": [1, 1, 1, 0], "hp": 0, "dead": True}))
    assert bool(st.buffs.w_passive[0])                   # never removed


# ---------------------------------------------------------------------------
# SERVER-001: the server cast freeze, detected from the wire's `mo`
# ---------------------------------------------------------------------------

def _mo_frames(stretches, *, team=100, t_end=20_000, dt=33, hp_zero=()):
    """Frames every `dt` ms; `mo` is CastSpell inside any [start, end) stretch.
    A minion with mo=15 and the other champion ride along: neither may count."""
    out = []
    for t in range(0, t_end, dt):
        casting = any(s <= t < e for s, e in stretches)
        dead = any(s <= t < e for s, e in hp_zero)
        out.append({"t": t, "u": [
            {"id": 1, "k": "Champion", "tm": team, "x": 10, "y": 20,
             "hp": 0 if dead else 500, "dead": dead,
             "mo": pd.WIRE_ORDER_CAST_SPELL if casting else 2},
            {"id": 2, "k": "Champion", "tm": 300 - team, "x": 0, "y": 0,
             "hp": 500, "mo": 3, "dead": False},
            {"id": 3, "k": "LaneMinion", "tm": team, "x": 0, "y": 0, "mo": 15}]})
    return out


def test_cast_freeze_ignores_legitimate_windups():
    # Q attack 250 ms, R 435 ms, recall pill 500 ms: the measured healthy max
    det = pd.scan_cast_freeze(_mo_frames([(1000, 1250), (4000, 4435), (8000, 8501)]))
    assert not det.invalid and det.frozen == [] and det.reason() is None


def test_cast_freeze_flags_a_stuck_champion_by_team_through_a_death():
    det = pd.scan_cast_freeze(_mo_frames([(5000, 20_000)], team=100,
                                         hp_zero=[(9000, 12_000)]))
    assert det.invalid and det.frozen_teams() == [100]
    (s,) = det.report()["frozen"]
    assert s["start_t_ms"] == 5016 and s["last_t_ms"] == 19_998
    assert s["died_while_stuck"] is True
    assert "SERVER-001" in det.reason() and "blue" in det.reason()


def test_cast_freeze_threshold_and_stretches_reset_on_any_other_order():
    thr = int(pd.CAST_FREEZE_MS)
    # two stretches just under the threshold, split by one non-cast frame
    det = pd.scan_cast_freeze(_mo_frames([(1000, 1000 + thr - 50),
                                          (1000 + thr, 1000 + 2 * thr - 50)]))
    assert not det.invalid
    det = pd.scan_cast_freeze(_mo_frames([(1000, 1000 + thr + 50)]))
    assert det.invalid


def test_cast_freeze_without_mo_on_the_wire_is_unchecked_not_passing():
    frames = _mo_frames([])
    for f in frames:
        for u in f["u"]:
            u.pop("mo", None)
    det = pd.scan_cast_freeze(frames)
    assert det.unchecked and det.invalid and not det.frozen
    assert "impossible" in det.reason()


def test_cast_freeze_scans_a_recorded_jsonl_stream():
    lines = [json.dumps(f) for f in _mo_frames([(2000, 9000)], team=200)]
    det = pd.scan_cast_freeze(iter(lines + ["", "  "]))
    assert det.frozen_teams() == [200] and "red" in det.reason()


def test_policy_loading_rejects_pointer_checkpoint_before_deserializing(tmp_path):
    checkpoint = tmp_path / "ckpt_latest.msgpack"
    checkpoint.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="manifest"):
        pd.load_params(str(checkpoint))
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {"train": {"policy": {}}}}))
    with pytest.raises(ValueError, match="entity-pointer"):
        pd.load_params(str(checkpoint))


def test_jitted_policy_driver_uses_three_heads_and_emits_click():
    from lanerl_jax.train.policy import ActionLogits
    from lanerl_rl.constants import N_SCREEN_X, N_SCREEN_Y

    class FixedPolicy:
        def apply(self, params, *obs):
            return ActionLogits(
                jnp.zeros(len(pd.BUTTONS)).at[pd.BUTTONS.index("attack_move")].set(10),
                jnp.zeros(N_SCREEN_X).at[N_SCREEN_X // 2].set(10),
                jnp.zeros(N_SCREEN_Y).at[N_SCREEN_Y // 2].set(10), jnp.float32(0))

    driver = pd.PolicyDriver(FixedPolicy(), {}, deterministic=True)
    state = init_lane()
    step = driver.decide(state, np.arange(state.x.size, dtype=np.int64) + 1073741824)
    assert step.wire["t"] == "click"
    assert step.wire["button"] == "attack_move"
    assert set(step.wire) == {"t", "button", "x", "y"}
    assert np.isfinite([step.wire["x"], step.wire["y"]]).all()


def test_v2_checkpoint_cannot_silently_load_under_authoritative_life_contract(tmp_path):
    checkpoint = tmp_path/'initial.msgpack'
    checkpoint.write_bytes(b'not deserialized')
    (tmp_path/'manifest.json').write_text(json.dumps({'config':{'train':{'policy':{
        'action_interface':'screen-click-v2', 'observation_interface':'viewport-structured-v2'}}}}))
    with pytest.raises(ValueError, match='v3 authoritative life-state'):
        pd.load_params(str(checkpoint))


def test_pair_actor_positive_hp_corpse_cannot_emit_gameplay_commands(monkeypatch):
    monkeypatch.setattr(pd, '_make_act', _stub_act(-1))
    frame = _champs(1000, blue=dict(dead=True, hp=35, ad=78., ap=0., ar=30., mr=30.,
                                   se=[1,1,1,1]))
    driver = PolicyPairDriver(None, None, red='idle')
    assert driver(frame, 0)['blue'] == {'t':'noop'}
    assert driver.log.blue_sim[0]['kind'] == OrderKind.NOOP
    # Same positive HP after authoritative respawn permits normal control.
    frame['u'][0]['dead'] = False
    assert driver(frame, 1)['blue']['t'] == 'click'


def test_pending_rank_up_never_asks_for_a_point_the_champion_does_not_have():
    """OPS-004: the server ranked E to 5 at level 9 where the table expected Q;
    asking for Q anyway spun the rank loop forever and starved the champion of
    decisions. The wire's spent points are authoritative."""
    from lanerl_jax.parity.policy_driver import pending_rank_up
    assert pending_rank_up({"lvl": 9, "sl": [2, 1, 5, 1]}) is None       # 9 points spent
    assert pending_rank_up({"lvl": 10, "sl": [2, 1, 5, 1]}) == 0         # a point left: Q
    assert pending_rank_up({"lvl": 6, "sl": [1, 1, 3, 0]}) == 3          # R at 6
    assert pending_rank_up({"lvl": 1, "sl": [0, 0, 0, 0]}) == 2          # E first
    assert pending_rank_up({"lvl": 12, "sl": [5, 1, 5, 2]}) is None      # 13 > 12: nothing owed
