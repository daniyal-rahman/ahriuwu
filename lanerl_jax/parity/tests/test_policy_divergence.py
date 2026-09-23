"""PARITY-001: the divergence report on synthetic traces with a known answer.

The server side is a hand-written `LANERL_STATEHASH`/`STATEROW` log in the
dump's own grammar; the sim side is a fake engine whose every tick is known.
So "first divergence at tick 5 in `Champion.hp`" is asserted, not observed.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.parity import policy_divergence as g
from lanerl_jax.parity.policy_driver import PolicyActionLog
from lanerl_jax.parity.trace import StatQ
from lanerl_jax.sim.init import TOP_LANE_PATH, init_lane, lane_params, spawn_minion
from lanerl_jax.sim.orders import OrderKind
from lanerl_jax.sim.profiles import PROFILES
from lanerl_jax.sim.spells import BuffId, E_BUFF_SLOT, Slot
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
    return SimpleNamespace(**{k: np.array(getattr(s, k)) for k in g._FETCH})


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
    st.buff_id[0, E_BUFF_SLOT] = BuffId.GAREN_E
    st.buff_elapsed[0, E_BUFF_SLOT] = 0.25
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
                        "server": {"blue": {"cs": 5}, "red": {"cs": 1}}}}
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


def test_gate_step_flags_are_the_trainers():
    """`STRUCT-003` until a shared SimConfig exists: pin the copy to the source."""
    src = (REPO / "lanerl_jax" / "train" / "trainer.py").read_text()
    calls = re.findall(r"step_decision\((.*?)\)\n", src, re.S)
    assert calls, "trainer no longer calls step_decision -- re-read the gate"
    flags = dict(re.findall(r"(collision_terrain|defer_collision_terrain)=(\w+)",
                            " ".join(calls)))
    assert {k: v == "True" for k, v in flags.items()} == g.TRAINING_STEP_FLAGS
    assert "route_table=route_table, terrain=terrain" in src
