"""``lanerl_train`` -- self-play training orchestration for the 1v1 Garen mirror.

This package is the *orchestration* half of the stack.  It owns how many
simulators run, who plays whom, what gets measured, and how a run survives being
killed.  It deliberately owns none of the RL semantics: observations, the
policy, the action decode and PPO live in ``lanerl_rl``; the scripted anchor and
the fast episode reset live in ``lanerl_bot`` and the C# server.  Those are read
here through :mod:`lanerl_train.protocols`, never imported concretely, so the
two halves can move independently.

Modules
-------
``paths``       resolve everything from ``__file__``; translate paths across nodes
``ports``       one verified-free port block per instance (a shared port is a silent kill)
``protocols``   the interfaces the RL side is expected to satisfy
``serverlog``   incremental parsing of a server's stdout (CS@10, fatals)
``vec``         N instances, one batched forward, loud death handling and restart
``league``      opponent sampling: 40% latest / 40% PFSP / 15% uniform / 5% anchors
``eval``        CS@10, win rate vs frozen bots, Bradley-Terry Elo, the rot signature
``run``         async actors -> central learner, bounded staleness, resumable
``selfplay``    the seam: instance planning, opponent slots, episode accounting
``slurm``       submission with ``-o`` paths valid on the node that opens them
"""

from __future__ import annotations

__all__ = [
    "paths",
    "ports",
    "protocols",
    "serverlog",
    "vec",
    "league",
    "eval",
    "run",
    "selfplay",
    "slurm",
]
__version__ = "0.1.0"
