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
``anchor_eval`` actually PLAYS the frozen bot on the eval cadence and scores it
``run``         async actors -> central learner, bounded staleness, resumable
``selfplay``    the seam: instance planning, opponent slots, episode accounting
``slurm``       submission with ``-o`` paths valid on the node that opens them
``runstats``    one run directory -> the handful of numbers worth comparing
``compare``     ``python -m lanerl_train.compare runs/a runs/b`` -- what differed?
``seeds``       one config across N seeds, and a 95% interval over the results

After the run
-------------
Everything below reads a finished (or running) run directory and needs no
server, no GPU and no checkpoint::

    # what was different between two runs, config and headline metrics
    python -m lanerl_train.compare runs/a runs/b

    # plan N seeds of one config (prints the commands; launches nothing)
    python -m lanerl_train.seeds plan --run-name abl-lr --seeds 0 1 2 -- --lr 1e-4

    # ...and once they have finished, mean +- 95% t interval across them
    python -m lanerl_train.seeds aggregate runs/abl-lr-s{0,1,2}

Evaluating against something that does not move
-----------------------------------------------
In a symmetric mirror ``score`` is 0.5 by construction, so a run with no frozen
opponent cannot tell you whether the policy improved.  ``--eval-every`` now
plays real games against the scripted Garen; ``--anchors`` selects which
difficulties, and a named anchor whose resource is missing **fails at startup**
rather than logging a warning and skipping.  ``--eval-every 0`` is how a run
says deliberately that it is not evaluated.

Environment switches
--------------------
``LANERL_OBS_STRICT=0``  skip the observation guards in
    :mod:`lanerl_rl.obs` (finite / shape / range / mask consistency).  Measured
    cost with them on: 66 us on a 535 us ``build()``.
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
    "anchor_eval",
    "run",
    "selfplay",
    "slurm",
    "runstats",
    "compare",
    "seeds",
]
__version__ = "0.1.0"
