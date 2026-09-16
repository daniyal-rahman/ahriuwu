"""J1 gate 3: an oracle last-hitter must score the same CS@10 in both places.

WHY THIS GATE IS DIFFERENT FROM EVERYTHING ELSE IN ``lanerl_jax/parity``
-------------------------------------------------------------------------
Every other instrument here diffs *state* -- positions and health tick by
tick, or the distribution of live minions. Both can hold inside tolerance
while the thing the simulator is *for* is broken: the minion-population
comparison (``lanerl_jax/sim/tests/test_lane.py``) sat at +17% while the lane
underneath it was collapsing to one side. This gate instead runs one fixed,
deterministic policy -- :mod:`lanerl_jax.parity.last_hit_oracle` -- against
both implementations and compares the score it gets. See that module's
docstring for why the policy is a greedy last-hitter and not a "perfect" one,
and ``lanerl_jax/parity/last_hit_drive.py`` for the two drivers and the two
known asymmetries between them (no fog of war in ``LaneState``; no per-level
champion stat growth in the sim's tick).

THE FIRST VERSION OF THIS GATE MEASURED PATHING, NOT LAST-HITTING
--------------------------------------------------------------------
Recorded in full in ``lanerl_jax/parity/last_hit_drive.py``'s module
docstring; the short version: a first cut handed ``decide()`` control from the
champion's spawn point, decision one. When nothing is killable, the oracle's
fallback is "walk to the centroid of the enemy minions", which from blue's
fountain is a single ~13,500-unit order -- and the sim has no pathfinding
(``sim/orders.py``: "on device there is no A*, so a move order becomes the
two-point line directly"), so that order went straight through terrain and the
champion never reached the wave. That version scored sim CS 0-1 against the
server's CS 16 (382 attacks against the sim's 0-11) -- a real number, but one
that measured the pathing approximation, not last-hitting, which was never
actually exercised. The fix (below) is a scripted, terrain-safe approach that
both drivers walk identically before either ever calls ``decide()``; see
:data:`~lanerl_jax.parity.last_hit_drive.APPROACH_WAYPOINTS`. This is
recorded here because a gate that silently measures the wrong thing is
precisely the failure mode gate 3 exists to catch, and it caught itself.

HOW THE TOLERANCE BELOW WAS CHOSEN
-----------------------------------
Measured, not assumed, on 2026-09-16, before the approach fix (the fix only
changes the driving script, not whether the server draws from any RNG this
config reaches, so the conclusion still applies): the same 600 s oracle-driven
episode was run against two freshly booted servers with different
``bot_seed``s (4242 and 1337; everything else identical -- ``toponly=True``,
``bot_teams="none"``, ``step_ticks=2``). ``bot_teams="none"`` means no
in-server scripted bot ever touches either champion, so if the seed does not
reach any RNG this configuration draws from, the two runs should be
bit-identical -- and they were::

    seed 4242:  cs=16  attacks=382  moves=13065  holds=4553
    seed 1337:  cs=16  attacks=382  moves=13065  holds=4553

Identical decision-by-decision, not just in the final CS. So the server's own
run-to-run spread for this exact scripted policy, at this exact config, is
**zero** -- consistent with the determinism already established in commit
``7b6d640`` (``docs/JAX_REWRITE_PLAN.md`` section 1.2). There is no measured
noise floor to build slack out of, so the tolerance is ``0``: an exact CS
match. If a future re-measurement ever finds nonzero server-side spread (a new
RNG draw reachable from this config, say), that number -- not a guess -- is
what should replace this constant.

WHAT THE GATE CURRENTLY FINDS, AND WHY THIS TEST IS EXPECTED TO FAIL
----------------------------------------------------------------------
Measured 2026-09-16 with the approach fix in place, same policy, same 600 s,
``LANERL_TOPONLY=1``, one server run (the budget for this pass was one
rebaseline run; see the module docstring for why the server number moved from
the pre-fix 16)::

    sim (JAX):  cs=2   approach_decisions=1285  attacks=39   moves=14095  holds=2581
    server:     cs=10  approach_decisions=1286  attacks=535  moves=12777  holds=3402

``approach_decisions`` landing within one decision of each other (1285 vs
1286, ~43 s) is the point of the restructuring: both champions now cover the
walk-in at the same rate, so the walk is no longer a variable. What is left is
an 8-CS gap (5x) against a tolerance of 0, and it shows up as attack
*opportunities*, not as a difference in what happens once an attack is
ordered: the server's champion gets offered a kill 535 times to the sim's 39,
a 14x gap that happens upstream of last-hitting. The leading suspect, per
``last_hit_drive``'s module docstring, is that ``LaneState`` still has no fog
of war: the sim driver's candidate list is every live red minion on the map,
so once engaged its centroid-walk fallback can still be pulled toward a wave
the server-side champion cannot see yet, even though the walk-IN is now
identical. Per the instructions this gate was built under: report the
disagreement, do not loosen the tolerance to make it pass.

Neither the fog gap nor the champion-level-scaling gap (also documented in
``last_hit_drive``) is this test's to fix. Implementing fog in ``LaneState``
or level-scaled champion stats in the tick are sim changes with their own
parity surface; this file's job is to keep making the disagreement visible
until one of them lands.
"""
from __future__ import annotations

import os

import pytest

from lanerl_jax.parity.last_hit_drive import (
    DECISIONS_600S,
    run_oracle_in_sim,
    run_oracle_on_server,
)

#: Exact match. See the module docstring: two 600 s server runs under
#: different bot_seeds produced bit-identical decision streams, so there is no
#: measured run-to-run noise to build slack out of.
CS_TOLERANCE = 0

#: Override for a quick local smoke run, e.g. ``LANERL_LAST_HIT_DECISIONS=1800
#: pytest -k last_hit_gate``. The gate's own number -- CS@10, 18,000 decisions
#: -- is the default because a shorter run answers a different question (CS at
#: some other clock, which nothing else reports against).
DECISIONS = int(os.environ.get("LANERL_LAST_HIT_DECISIONS", str(DECISIONS_600S)))


@pytest.mark.slow
@pytest.mark.skipif(
    not __import__("lanerl_train.paths", fromlist=["paths"]).server_available(),
    reason="vendored server build not available",
)
def test_oracle_scores_the_same_cs_in_sim_and_server():
    """J1 gate 3. Boots one real server; takes several minutes.

    Both sides walk :data:`~lanerl_jax.parity.last_hit_drive.APPROACH_WAYPOINTS`
    before either ever calls the oracle -- see the module docstring for why a
    first version of this test measured pathing instead of last-hitting, and
    why that walk-in is now scripted rather than oracle-driven.

    See the module docstring for the measured tolerance and for the gap this
    is currently finding. This assertion is written to the gate's real
    criterion, not to what happens to pass today -- so as of 2026-09-16 it
    FAILS, and that failure is the deliverable: sim CS=2 against the server's
    CS=10 over the same 600 s, same policy, same scripted approach.
    """
    sim = run_oracle_in_sim(decisions=DECISIONS)
    server = run_oracle_on_server(
        decisions=DECISIONS, port_base=46000, bot_seed=4242, tag="last_hit_gate")

    gap = sim.cs - server.cs
    assert abs(gap) <= CS_TOLERANCE, (
        f"oracle CS@10 disagrees between sim and server by {gap:+d} "
        f"(tolerance {CS_TOLERANCE}): "
        f"sim cs={sim.cs} approach_decisions={sim.approach_decisions} "
        f"attacks={sim.attacks} moves={sim.moves} holds={sim.holds}; "
        f"server cs={server.cs} approach_decisions={server.approach_decisions} "
        f"attacks={server.attacks} moves={server.moves} holds={server.holds} "
        f"log={server.log_path}. "
        "This is the behavioural gate (J1 gate 3, docs/JAX_REWRITE_PLAN.md); "
        "the approach walk-in is no longer the cause (approach_decisions should "
        "match within ~1 across both sides) -- the leading suspect is LaneState "
        "having no fog of war, so the sim's post-handover centroid-walk can "
        "still be pulled toward minions the server-side champion cannot see "
        "yet (see lanerl_jax/parity/last_hit_drive.py's module docstring) -- "
        "do not raise CS_TOLERANCE to silence this, fix the cause instead."
    )
