"""PARITY-001: the divergence report on synthetic traces with a known answer.

The server side is a hand-written `LANERL_STATEHASH`/`STATEROW` log in the
dump's own grammar; the sim side is a fake engine whose every tick is known.
So "first divergence at tick 5 in `Champion.hp`" is asserted, not observed.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.parity import policy_divergence as g
from lanerl_jax.parity.policy_driver import PolicyActionLog
from lanerl_jax.parity.trace import StatQ
from lanerl_jax.sim.init import TOP_LANE_PATH, init_lane, lane_params, spawn_minion
from lanerl_jax.sim.orders import OrderKind
from lanerl_jax.sim.profiles import PROFILES
from lanerl_jax.sim.spells import Slot
from lanerl_jax.sim.state import Kind, Team

REPO = Path(__file__).resolve().parents[3]
BLUE_NID, RED_NID = 0x40000010, 0x40000011
MINION_NIDS = [0x40000100, 0x40000143]          # red rank 0, blue rank 1


def _row(e) -> str:
    """One Entity -> a `LanerlStateDump.Describe` row body (11 or 21 parts)."""
    ai = e.ai
    parts = [e.kind, str(e.team), f"{e.q_x},{e.q_y}", f"{e.q_hp}/{e.q_max_hp}",
             "D" if e.dead else "A", str(ai.move_order), str(ai.waypoints),
             ai.cast_spell, ai.channel_spell, "1" if ai.can_move else "0",
             "+".join(ai.buffs)]
    if e.champ is not None:
        c = e.champ
        parts += ["80010,37413,45220,353280,1024", str(c.level),
                  str(c.q_gold + int(g.STARTING_GOLD * StatQ)),   # the WALLET
                  str(c.minions_killed), str(c.deaths), "0"]
        parts += [f"{lv}:{cd}" for lv, cd in c.spells]
    return "|".join(parts)


def _write_log(path: Path, snaps, internals=None) -> Path:
    lines = []
    for j, s in enumerate(snaps):
        lines.append(f"LANERL_STATEHASH t={s.t_ms} n={len(s.entities)} "
                     f"h={j:016x}")
        lines += [f"LANERL_STATEROW t={s.t_ms} {_row(e)}" for e in s.entities]
        for team, hasaa in (internals or {}).get(j, {}).items():
            lines.append(f"LANERL_INTERNAL t={s.t_ms} ai id={BLUE_NID} "
                         f"kind=Champion team={team} x=0 y=0 hasaa={int(hasaa)}")
        lines.append("some interleaved server chatter")
    path.write_text("\n".join(lines) + "\n")
    return path


def _t(j: int) -> int:
    return int(round(j * 1000.0 / 60.0))


def _initial_numpy_state():
    s = init_lane()
    params = lane_params()
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))
    for team in (Team.RED, Team.BLUE):
        row = PROFILES.index((Kind.LANE_MINION, 0, team))
        s = spawn_minion(s, team, row, params["max_hp"][row],
                         path[::-1] if team == Team.RED else path)
    return SimpleNamespace(**{k: np.array(getattr(s, k)) for k in g._FETCH},
                           buffs=jax.tree.map(np.array, s.buffs))


class FakeEngine:
    """Deterministic stand-in for `TrainingStepEngine`: blue walks +1 u/tick."""

    def __init__(self):
        self.applied = []
        self.j = 0

    def init(self):
        self.j = 0
        return _initial_numpy_state()

    def tick(self, st):
        st = copy.deepcopy(st)
        st.t_ms = np.float32(st.t_ms + g.TICK_MS)
        st.x[0] += 1.0
        self.j += 1
        return st

    def apply(self, st, orders):
        self.applied.append((self.j, int(orders.kind[0]), int(orders.target[0]),
                             int(orders.kind[1])))
        return st


def _sim_snaps(n: int):
    eng = FakeEngine()
    st = eng.init()
    out = [g.render_sim_snapshot(g.fetch(st), 0)]
    for j in range(1, n):
        st = eng.tick(st)
        out.append(g.render_sim_snapshot(g.fetch(st), _t(j)))
    return out, st


def _log(t_ms, blue, red):
    log = PolicyActionLog()
    for t, b, r in zip(t_ms, blue, red):
        log.append(t, b, r, None, None)
    log.ranks = {"first_minion_seq": int(init_lane().next_spawn_seq),
                 "champions": {"0": BLUE_NID, "1": RED_NID}, "turrets": {},
                 "minions": MINION_NIDS}
    return log


def _hp_bump(snap, team, delta_q):
    import dataclasses
    ents = [dataclasses.replace(e, q_hp=e.q_hp + delta_q)
            if e.kind == "Champion" and e.team == team else e for e in snap.entities]
    return dataclasses.replace(snap, entities=ents)


def test_first_divergence_is_reported_at_the_known_tick_and_field(tmp_path):
    snaps, _ = _sim_snaps(12)
    server = list(snaps)
    server[5] = _hp_bump(server[5], 100, -5 * 1024)       # blue 5 hp lower, tick 5
    log_path = _write_log(tmp_path / "srv.log", server, internals={7: {100: True}})
    # decisions are logged at TRUNCATED times (16, 50) -> ticks 1 and 3
    log = _log([16, 50],
               [{"t": "attack", "id": MINION_NIDS[1]}, {"t": "level", "slot": 2}],
               [{"t": "noop"}, {"t": "attack", "id": 0x7777}])   # unknown NetId
    eng = FakeEngine()
    rep = g.replay_and_diff(log, log_path, engine=eng)

    blue_minion_slot = int(np.flatnonzero(
        (_initial_numpy_state().kind == Kind.LANE_MINION)
        & (_initial_numpy_state().team == Team.BLUE))[0])
    assert eng.applied == [(1, OrderKind.ATTACK, blue_minion_slot, OrderKind.NOOP),
                           (3, OrderKind.NOOP, -1, OrderKind.NOOP)]
    fd = rep["first_divergence"]
    assert fd["t_ms"] == _t(5) and fd["tick"] == 5 and fd["decision"] == 1
    assert fd["fields"] == ["Champion.hp"]
    assert list(rep["first_divergence_by_field"]) == ["Champion.hp"]
    assert rep["diverged_ticks_by_field"] == {"Champion.hp": 1}
    assert rep["ticks_compared"] == 12 and rep["decisions_applied"] == 2
    assert rep["attack_orders_unmapped_in_sim"] == {"blue": 0, "red": 1}
    # AA hits come from the INTERNAL stream on the server, state on the sim
    assert rep["counters"]["server"]["blue"]["aa_hits"] == 1
    assert rep["counters"]["sim"]["blue"]["aa_hits"] == 0
    # gold is compared as earnings: the wallet's 475 is not a divergence
    assert rep["counters"]["server"]["blue"]["gold"] == 0.0


def test_a_dump_gap_or_a_late_start_refuses_to_align(tmp_path):
    snaps, _ = _sim_snaps(6)
    log = _log([16], [{"t": "noop"}], [{"t": "noop"}])
    p = _write_log(tmp_path / "gap.log", snaps[:3] + snaps[4:])
    with pytest.raises(ValueError, match="not one tick"):
        g.replay_and_diff(log, p, engine=FakeEngine())
    p = _write_log(tmp_path / "late.log", snaps[1:])
    with pytest.raises(ValueError, match="not at the reset"):
        g.replay_and_diff(log, p, engine=FakeEngine())


def _with_champ(snap, team, *, buffs=None, e_cd_s=None):
    import dataclasses
    out = []
    for e in snap.entities:
        if e.kind == "Champion" and e.team == team:
            if buffs is not None:
                e = dataclasses.replace(e, ai=dataclasses.replace(e.ai, buffs=buffs))
            if e_cd_s is not None:
                sp = list(e.champ.spells)
                sp[2] = (sp[2][0], int(round(e_cd_s * StatQ)))
                e = dataclasses.replace(e, champ=dataclasses.replace(
                    e.champ, spells=tuple(sp)))
        out.append(e)
    return dataclasses.replace(snap, entities=out)


def test_server_vs_server_compare_finds_a_cooldown_divergence(tmp_path):
    snaps, _ = _sim_snaps(8)
    a = list(snaps)
    b = list(snaps)
    b[4] = _with_champ(b[4], 200, e_cd_s=0.5)
    ra = _write_log(tmp_path / "a.log", a)
    rb = _write_log(tmp_path / "b.log", b)
    rep = g.compare_server_logs(ra, rb)
    assert rep["first_divergence"]["t_ms"] == _t(4)
    assert rep["first_divergence"]["fields"] == ["Champion.spell.E.cooldown"]
    assert rep["ticks_missing"] == 0
    assert g.compare_server_logs(ra, ra)["first_divergence"] is None


def test_counters_separate_spin_starts_ends_and_the_evals_double_count():
    """One server spin = TWO rising edges of cd2 (`GarenECancel`, then rank cd)."""
    snaps, _ = _sim_snaps(1)
    base = snaps[0]
    seq = [((), 0.0), (("GarenE",), 1.0), (("GarenE",), 0.4), (("GarenE",), 0.0),
           ((), 9.0), ((), 8.9), ((), 0.0), (("GarenE",), 1.0), ((), 8.0)]
    c = g.EventCounters()
    for i, (buffs, cd) in enumerate(seq):
        s = _with_champ(base, 100, buffs=buffs, e_cd_s=cd)
        s = g.project_server_snapshot(s, gold_offset=0.0)
        c.update(type(s)(t_ms=i * 1000, entities=s.entities))
    b = c.as_dict()["blue"]
    assert b["e_spin_starts"] == 2 and b["e_spin_ends"] == 2
    assert b["e_cd_rising_edges"] == 4
    assert b["e_cancels"] == 1             # the second spin lasted 1 s
    assert b["casts_accepted"] == 2


def test_render_projects_e_cooldown_during_the_spin_and_keeps_dead_champions():
    st = _initial_numpy_state()
    st.buffs.e.active[0] = True
    st.buffs.e.elapsed_s[0] = 0.25
    st.spell_cooldown[0, Slot.E] = 0.0
    st.spell_level[0] = [0, 0, 1, 0]
    st.alive[1] = False
    snap = g.render_sim_snapshot(g.fetch(st), 1000)
    blue, red = snap.champion(100), snap.champion(200)
    assert blue.ai.buffs == ("GarenE",)
    assert blue.champ.spells[2] == (1, int(round(0.75 * StatQ)))
    assert red is not None and red.dead


def test_server_projection_clamps_ready_cooldowns_and_filters_buffs():
    snaps, _ = _sim_snaps(1)
    s = _with_champ(snaps[0], 100, buffs=("GarenE", "GarenPassive"),
                    e_cd_s=-1 / 60)
    p = g.project_server_snapshot(s)
    e = p.champion(100)
    assert e.ai.buffs == ("GarenE",) and e.champ.spells[2][1] == 0
    assert e.champ.q_gold == s.champion(100).champ.q_gold - 475 * 1024


def test_floor_scoring():
    rep = {"first_divergence": {"t_ms": 500},
           "counters": {"sim": {"blue": {"cs": 3}, "red": {"cs": 1}},
                        "server": {"blue": {"cs": 5}, "red": {"cs": 1}}},
           "position_resync": {"intervals": 10, "unexplained": 0,
                               "path001": 2, "path001_rate": 0.2}}
    assert g.score_against_floor(rep, None)["verdict"] == "UNSCORED"
    floor = {"first_divergence": {"t_ms": 400},
             "counters": {"a": {"blue": {"cs": 5}, "red": {"cs": 1}},
                          "b": {"blue": {"cs": 3}, "red": {"cs": 1}}}}
    assert g.score_against_floor(rep, floor)["verdict"] == "PASS"
    floor["first_divergence"]["t_ms"] = 600
    v = g.score_against_floor(rep, floor)
    assert v["verdict"] == "FAIL" and "first divergence" in v["failures"][0]
    floor["first_divergence"]["t_ms"] = 400
    floor["counters"]["b"]["blue"]["cs"] = 4
    v = g.score_against_floor(rep, floor)
    assert v["verdict"] == "FAIL" and v["failures"] == [
        "blue.cs: sim-vs-server gap 2 > floor 1"]
    # position: scored per decision; the PATH-001 rate is reported, not gated
    floor["counters"]["b"]["blue"]["cs"] = 3
    v = g.score_against_floor(rep, floor)
    assert v["verdict"] == "PASS" and v["path001_rate"] == 0.2
    rep["position_resync"]["unexplained"] = 1
    v = g.score_against_floor(rep, floor)
    assert v["verdict"] == "FAIL" and "champion position: 1 of 10" in v["failures"][0]
    del rep["position_resync"]
    v = g.score_against_floor(rep, floor)
    assert v["verdict"] == "FAIL" and "not scored" in v["failures"][0]


def test_free_run_champion_position_is_reported_not_scored_while_routing_is_approx():
    import dataclasses
    assert g.ROUTING_APPROX
    assert not g._is_scored("Champion.pos.x") and not g._is_scored("Champion.pos.y")
    assert g._is_scored("LaneMinion.pos.x") and g._is_scored("Champion.hp")
    tr = g.DivergenceTracker(g.gate_tolerance())
    snaps, _ = _sim_snaps(2)
    moved = dataclasses.replace(snaps[1], entities=[
        dataclasses.replace(e, q_x=e.q_x + 32) if e.kind == "Champion" else e
        for e in snaps[1].entities])
    tr.update(snaps[1], moved, tick=1, decision=0)
    out = tr.as_dict()
    assert out["first_divergence"] is None
    assert "Champion.pos.x" in out["reported_only_first_by_field"]


def _f32_bits(v: float) -> int:
    return int(np.array([v], np.float32).view(np.int32)[0])


def _internal(t, team, x, y, key, wps):
    q = ";".join(f"{int(round(a * 16))},{int(round(b * 16))}" for a, b in wps)
    xb, yb = _f32_bits(x), _f32_bits(y)
    return (f"LANERL_INTERNAL t={t} ai id={BLUE_NID if team == 100 else RED_NID} "
            f"kind=Champion team={team} x=0 y=0 xbits={xb} ybits={yb} cr=1 "
            f"pr=1 target=0,-,0,0,0 wpkey={key} wps={q} coll=0,0 "
            f"collbits={xb},{yb} aacd=0 hasaa=0")


def test_internal_champion_rows_parse_exact_bits_and_waypoints():
    line = _internal(17, 200, 13850.6640625, 14375.5, 1,
                     [(13850.6640625, 14375.5), (12635.390625, 14141.2265625)])
    team, c = g._parse_champ_internal(line)
    assert team == 200 and (c.x, c.y) == (13850.6640625, 14375.5)
    assert (c.cx, c.cy) == (c.x, c.y) and c.key == 1
    assert c.remaining == ((12635.375, 14141.25),)    # the dump's 1/16 quantum
    _, neg = g._parse_champ_internal(
        _internal(0, 100, -26.75, 257.5, 1, [(0.0, 0.0)]))
    assert neg.x == -26.75


class FakeResyncEngine(FakeEngine):
    """`FakeEngine` plus the resync surface: blue walks +1 u/tick, a blue
    MOVE installs the route [(9, 9)], anything else keeps the resynced one."""

    def __init__(self):
        super().__init__()
        self.routes = {}

    def resync(self, st, champ):
        st = copy.deepcopy(st)
        for team, c in champ.items():
            i = g.CHAMP_SLOT[team]
            st.x[i], st.y[i] = np.float32(c.x), np.float32(c.y)
        self.routes = {t: list(c.remaining) for t, c in champ.items()}
        return st

    def apply(self, st, orders):
        st = super().apply(st, orders)
        if int(orders.kind[0]) == OrderKind.MOVE:
            self.routes[100] = [(9.0, 9.0)]
        return st

    def champion_routes(self, st):
        return {t: {"x": float(st.x[i]), "y": float(st.y[i]),
                    "remaining": self.routes[t], "route_status": 0}
                for t, i in g.CHAMP_SLOT.items()}


class FakeGrid:
    """Host `GetPath`: a fixed route to (7, 7), or null for goal (8, 8)."""

    def __init__(self):
        self.calls = []

    def get_path(self, src, goal, radius):
        self.calls.append((src, goal))
        return None if goal == (8.0, 8.0) else [src, (7.0, 7.0)]


def _resync_log(tmp_path, blue_orders, blue_srv_goal):
    """Seven ticks, decisions at 1, 3, 5. The server's blue runs +1 u/tick
    like the fake sim and holds ``[pos, blue_srv_goal]`` from tick 3; both
    champions are 1 u off the sim from tick 5 on."""
    snaps, _ = _sim_snaps(7)
    st0 = _initial_numpy_state()
    bx, by = float(st0.x[0]), float(st0.y[0])
    rx, ry = float(st0.x[1]), float(st0.y[1])
    lines = []
    for j, s in enumerate(snaps):
        lines.append(f"LANERL_STATEHASH t={s.t_ms} n={len(s.entities)} h={j:016x}")
        lines += [f"LANERL_STATEROW t={s.t_ms} {_row(e)}" for e in s.entities]
        off = 1.0 if j >= 5 else 0.0
        blue_now = (bx + j + off, by)
        blue_wps = [blue_now, blue_srv_goal] if j == 3 else [blue_now]
        lines.append(_internal(s.t_ms, 100, *blue_now, 1, blue_wps))
        red_now = (rx, ry + off)
        lines.append(_internal(s.t_ms, 200, *red_now, 1, [red_now, (rx, ry + 50)]))
    log_path = tmp_path / "srv.log"
    log_path.write_text("\n".join(lines) + "\n")
    log = _log([16, 50, 83], blue_orders, [{"t": "noop"}] * 3)
    return log, log_path, (bx, by)


def test_resync_scores_position_per_decision_and_classifies_mismatches(tmp_path):
    """Intervals 1->3 and 3->5 per team. At tick 5 blue is 1 u off after a
    MOVE whose sim route differs from the server's and the host port
    reproduces the server's (PATH-001); red is 1 u off with identical
    waypoints (unexplained)."""
    log, log_path, (bx, by) = _resync_log(
        tmp_path, [{"t": "noop"}, {"t": "move", "x": 7.0, "y": 7.0},
                   {"t": "noop"}], (7.0, 7.0))
    grid = FakeGrid()
    rs = g.replay_resync(log, log_path, engine=FakeResyncEngine(), grid=grid,
                         radius=35.0)
    assert (rs["intervals"], rs["within_tol"], rs["path001"], rs["unexplained"]) \
        == (4, 2, 1, 1)
    assert rs["path001_rate"] == 0.25 and rs["max_err_u"] == 1.0
    assert grid.calls == [((bx + 3, by), (7.0, 7.0))]      # the float32 click
    assert rs["unexplained_by_reason"] == {"same waypoints, still off": 1}
    ex = rs["unexplained_examples"]
    assert len(ex) == 1 and ex[0]["team"] == 200 and ex[0]["waypoints_agree"]
    assert rs["by_order"]["move"] == {"intervals": 1, "within_tol": 0,
                                      "path001": 1, "unexplained": 0}


def test_resync_fails_a_server_null_the_sim_routed_around(tmp_path):
    """`PATH-008`: the host port says GetPath is null and the server walked
    ``[Position, click]``; a sim that routed instead is NOT a tie-break."""
    log, log_path, _ = _resync_log(
        tmp_path, [{"t": "noop"}, {"t": "move", "x": 8.0, "y": 8.0},
                   {"t": "noop"}], (8.0, 8.0))
    rs = g.replay_resync(log, log_path, engine=FakeResyncEngine(),
                         grid=FakeGrid(), radius=35.0)
    assert rs["path001"] == 0 and rs["unexplained"] == 2
    assert rs["unexplained_by_reason"] == {
        "server GetPath null the sim did not reproduce (PATH-008)": 1,
        "same waypoints, still off": 1}


def test_replay_wire_driver_remaps_netids_by_rank():
    log = _log([16], [{"t": "attack", "id": MINION_NIDS[1]}],
               [{"t": "attack", "id": BLUE_NID}])
    drv = g.ReplayWireDriver(log)
    frame = {"t": 16, "u": [
        {"id": 900, "k": "Champion", "tm": 100, "x": 0, "y": 0},
        {"id": 901, "k": "Champion", "tm": 200, "x": 1, "y": 1},
        {"id": 950, "k": "LaneMinion", "tm": 200, "x": 2, "y": 2},
        {"id": 951, "k": "LaneMinion", "tm": 100, "x": 3, "y": 3}]}
    out = drv(frame, 0)
    assert out == {"blue": {"t": "attack", "id": 951},
                   "red": {"t": "attack", "id": 900}}
    assert drv(frame, 5) == {"blue": {"t": "noop"}, "red": {"t": "noop"}}


def test_gate_step_config_is_the_trainers():
    """`STRUCT-003`: the gate's engine steps a `SimConfig` that equals the one
    `make_train` actually closes over, except `step_ticks` (the gate diffs
    every tick, `PARITY-001`). Compared as objects, not as a regex over the
    trainer's source; the allow-list and its ledger citations live in
    `sim/tests/test_sim_config.py`."""
    from lanerl_jax.sim.config import DEFAULT_ROUTE_ARTIFACT
    from lanerl_jax.train.run_train import DEFAULT_ROUTE_ARTIFACT as TRAIN_DEFAULT
    from lanerl_jax.train.trainer import TrainConfig, make_train

    engine = g.TrainingStepEngine(use_route_table=False)
    built = make_train(TrainConfig(n_envs=2, rollout_steps=2, n_updates=1))
    assert engine.sim_config.differing_fields(built.sim_config) == {"step_ticks"}
    assert (engine.sim_config.step_ticks, built.sim_config.step_ticks) == (1, 2)
    # routed by default, from the artifact `run_train` trains on
    assert g._default_route_artifact() == DEFAULT_ROUTE_ARTIFACT == TRAIN_DEFAULT
    # and the trainer can only step through the config object
    src = (REPO / "lanerl_jax" / "train" / "trainer.py").read_text()
    assert "step_decision(" not in src and "apply_orders(" not in src
    assert "env_apply(state, orders, sim)" in src and "env_advance(ordered, sim)" in src
