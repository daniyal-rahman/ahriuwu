"""A checkpoint pool shared between training processes, via a directory.

AlphaStar's league is not one agent playing its own past selves. It is several
concurrently-training agents of different TYPES, all drawing opponents from one
pool that they all contribute to (Vinyals et al., Nature 2019):

    main agent        35% self-play, 50% PFSP over all past players, 15% PFSP
                      over forgotten main players and past main exploiters.
                      Never reset.
    main exploiter    plays the CURRENT main agents, to find their weaknesses.
                      Reset to the supervised (here: behaviour-cloned)
                      parameters once it beats them.
    league exploiter  PFSP over all past players, to find weaknesses in the
                      league as a whole. Reset the same way.

The exploiters are the part that makes the league more than self-play: a main
agent playing only its own history can cycle, and has no pressure to be robust
to a strategy nobody in its history happened to try.

``CheckpointPool`` is in-memory and per-process, persisted inside one run's
``state.json``, so two training processes cannot see each other's snapshots at
all. This is the smallest thing that fixes that: every agent writes a small
JSON per snapshot into one directory, and every agent reads the directory back
before it samples. No server, no lock protocol -- a snapshot file is written
once, atomically, and never modified, so a reader either sees a complete file
or does not see it yet.

Why a directory rather than one process with several learners: the actors are
already separate OS processes for GIL reasons (see ``procactor``), each agent
needs its own optimiser and its own CUDA context anyway, and a crash in one
lineage should not take the league with it. Separate Slurm jobs also let the
league be resized without touching the trainer.

Scale warning, recorded because it is the whole cost of this feature: on a
16-core box, three lineages means ~16 server instances each instead of 48, so
each learns at roughly a third of the single-agent rate. AlphaStar ran twelve
agents on a TPU cluster for 44 days.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from .league import Snapshot

log = logging.getLogger("lanerl_train.shared_pool")

#: Agent types, matching the paper's three roles.
MAIN = "main"
MAIN_EXPLOITER = "main_exploiter"
LEAGUE_EXPLOITER = "league_exploiter"
AGENT_TYPES = (MAIN, MAIN_EXPLOITER, LEAGUE_EXPLOITER)


class SharedSnapshotDir:
    """Snapshots published by every agent in the league, in one directory.

    One JSON per snapshot, named ``<agent_id>@<step>.json``. Written to a
    temporary file and renamed, because ``rename`` within a filesystem is
    atomic: a reader scanning concurrently sees either nothing or a complete
    record, never a half-written one. That matters because every agent scans
    this directory on every sample.
    """

    def __init__(self, path: Optional[Path], agent_id: str, agent_type: str) -> None:
        self.path = Path(path) if path else None
        self.agent_id = agent_id
        self.agent_type = agent_type
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def publish(self, snap: Snapshot) -> None:
        """Announce one snapshot to the rest of the league."""
        if self.path is None:
            return
        rec = {
            "id": f"{self.agent_id}:{snap.id}",
            "step": int(snap.step),
            "path": str(snap.path or ""),
            "created_s": float(snap.created_s or time.time()),
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
        }
        name = f"{self.agent_id}@{snap.step:09d}.json"
        try:
            fd, tmp = tempfile.mkstemp(dir=str(self.path), suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(rec, fh)
            os.replace(tmp, self.path / name)
        except OSError:
            # A league that cannot publish is a league that silently shrinks to
            # self-play, which is the failure this whole module exists to end.
            log.error("could not publish snapshot %s to the shared pool at %s",
                      snap.id, self.path, exc_info=True)

    def scan(self, exclude_self: bool = False) -> List[Snapshot]:
        """Every snapshot any agent has published, newest last.

        A record whose checkpoint file has gone is skipped rather than
        returned: a stale entry would be drawn by PFSP, fail to load in the
        actor, and fall back to the live mirror -- a league quietly playing
        itself while the logs say otherwise.
        """
        if self.path is None:
            return []
        out: List[Snapshot] = []
        try:
            files = sorted(self.path.glob("*.json"))
        except OSError:
            return []
        for f in files:
            try:
                rec = json.loads(f.read_text())
            except Exception:
                continue  # mid-write or corrupt; it will be there next scan
            if exclude_self and rec.get("agent_id") == self.agent_id:
                continue
            p = rec.get("path")
            if not p or not Path(p).exists():
                continue
            out.append(Snapshot(id=rec["id"], step=int(rec.get("step", 0)),
                                path=p, created_s=float(rec.get("created_s", 0.0))))
        out.sort(key=lambda s: (s.created_s, s.step))
        return out

    def by_type(self, agent_type: str) -> List[Snapshot]:
        """Snapshots published by agents of one type, newest last.

        The main exploiter needs exactly this: the MAIN agents' snapshots and
        nobody else's.
        """
        if self.path is None:
            return []
        out: List[Snapshot] = []
        for f in sorted(self.path.glob("*.json")):
            try:
                rec = json.loads(f.read_text())
            except Exception:
                continue
            if rec.get("agent_type") != agent_type:
                continue
            p = rec.get("path")
            if not p or not Path(p).exists():
                continue
            out.append(Snapshot(id=rec["id"], step=int(rec.get("step", 0)),
                                path=p, created_s=float(rec.get("created_s", 0.0))))
        out.sort(key=lambda s: (s.created_s, s.step))
        return out

    def latest_of_type(self, agent_type: str) -> Optional[Snapshot]:
        snaps = self.by_type(agent_type)
        return snaps[-1] if snaps else None

    def counts(self) -> Dict[str, int]:
        """How many snapshots each agent type has published, for the log."""
        out: Dict[str, int] = {}
        if self.path is None:
            return out
        for f in self.path.glob("*.json"):
            try:
                rec = json.loads(f.read_text())
            except Exception:
                continue
            t = str(rec.get("agent_type", "?"))
            out[t] = out.get(t, 0) + 1
        return out
