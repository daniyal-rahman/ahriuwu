"""The cross-process league pool, and the exploiter targeting rules.

These exist because the league's failure mode is silence. The sampler itself
was implemented, unit-tested and uncalled for weeks while every log looked
normal (see test_run.py's league wiring tests). The same shape of bug is
available here: a shared pool that publishes nothing, or an exploiter that
targets nobody, both degrade to ordinary self-play without erroring.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from lanerl_train.league import Snapshot
from lanerl_train.shared_pool import (
    LEAGUE_EXPLOITER,
    MAIN,
    MAIN_EXPLOITER,
    SharedSnapshotDir,
)


@pytest.fixture()
def league(tmp_path):
    ck = tmp_path / "ck.pt"
    ck.write_text("weights")
    return tmp_path / "league", ck


def _dir(root, agent_id, agent_type):
    return SharedSnapshotDir(root, agent_id=agent_id, agent_type=agent_type)


def test_agents_see_each_others_snapshots(league):
    """The whole point: without this a 'league' is N self-play runs."""
    root, ck = league
    main = _dir(root, "run-main", MAIN)
    expl = _dir(root, "run-expl", MAIN_EXPLOITER)
    main.publish(Snapshot(id="snap@200", step=200, path=str(ck), created_s=1.0))
    expl.publish(Snapshot(id="snap@100", step=100, path=str(ck), created_s=1.5))

    assert [s.id for s in expl.scan(exclude_self=True)] == ["run-main:snap@200"]
    assert {s.id for s in main.scan()} == {"run-main:snap@200", "run-expl:snap@100"}


def test_ids_are_namespaced_by_agent(league):
    """Two lineages both snapshot at step 200; they must not collide.

    Every agent names its snapshots snap@<update>, so without the agent id in
    the key the pool would hold one entry where it should hold two, and the
    win-rate table would blend two different opponents into one record.
    """
    root, ck = league
    a = _dir(root, "run-a", MAIN)
    b = _dir(root, "run-b", LEAGUE_EXPLOITER)
    a.publish(Snapshot(id="snap@200", step=200, path=str(ck), created_s=1.0))
    b.publish(Snapshot(id="snap@200", step=200, path=str(ck), created_s=1.0))
    assert len({s.id for s in a.scan()}) == 2


def test_a_snapshot_whose_checkpoint_is_gone_is_not_offered(league):
    """A dangling entry would be drawn, fail to load, and fall back to the
    live mirror -- a league quietly playing itself while the logs disagree."""
    root, _ = league
    ck = root.parent / "temp.pt"
    ck.write_text("w")
    d = _dir(root, "run-a", MAIN)
    d.publish(Snapshot(id="snap@1", step=1, path=str(ck), created_s=1.0))
    assert len(d.scan()) == 1
    ck.unlink()
    assert d.scan() == []


def test_latest_of_type_is_what_a_main_exploiter_targets(league):
    root, ck = league
    main = _dir(root, "run-main", MAIN)
    other = _dir(root, "run-le", LEAGUE_EXPLOITER)
    for step in (100, 300, 200):
        main.publish(Snapshot(id=f"snap@{step}", step=step, path=str(ck),
                              created_s=float(step)))
    other.publish(Snapshot(id="snap@999", step=999, path=str(ck), created_s=999.0))

    latest = other.latest_of_type(MAIN)
    assert latest is not None and latest.id == "run-main:snap@300", (
        "a main exploiter must target the newest MAIN snapshot, not the newest "
        "snapshot of any type -- otherwise it exploits another exploiter"
    )


def test_disabled_without_a_directory(league):
    """Single-agent runs must be completely unaffected."""
    d = SharedSnapshotDir(None, agent_id="x", agent_type=MAIN)
    assert not d.enabled
    d.publish(Snapshot(id="s", step=1, path="/nonexistent", created_s=1.0))
    assert d.scan() == [] and d.counts() == {} and d.latest_of_type(MAIN) is None


def test_a_half_written_record_does_not_break_a_scan(league):
    """Readers scan while writers write. publish() renames atomically, but a
    stray or corrupt file must not take the league down either."""
    root, ck = league
    d = _dir(root, "run-a", MAIN)
    d.publish(Snapshot(id="snap@1", step=1, path=str(ck), created_s=1.0))
    (root / "garbage.json").write_text("{not json")
    assert [s.id for s in d.scan()] == ["run-a:snap@1"]
