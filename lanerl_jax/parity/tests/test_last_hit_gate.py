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
and ``lanerl_jax/parity/last_hit_drive.py`` for the two drivers, both now
resolved (a driver-side fog substitute; champion AD level-scaling in
``sim/step.py``) but still worth reading for what each one changed.

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

THE "10 CS / 535 ATTACKS" FIGURE IN docs/JAX_REWRITE_PLAN.md IS STALE
--------------------------------------------------------------------
Traced by commit timestamp, not assumption. That figure was written in
``89c5d58`` at 13:13:53Z; ``d463533`` ("the gate stops standing under the
enemy turret") landed 37 minutes later at 13:50:25Z and moved
:data:`~lanerl_jax.parity.last_hit_drive.APPROACH_WAYPOINTS` from
``TOP_LANE_PATH[:7]`` (ending 412 units from red's outer turret, inside its
750 range) to ``[:6]``. So "535 attacks" was measured with the champion
parked inside the enemy turret's attack range, and nobody re-measured the
server baseline after the position fix landed -- every commit after
``d463533`` (the fog fix, the call-for-help toggle, the turret/W-ramp
resolution) built on the corrected position without anyone going back to
refresh the one server number that was taken before it. It does not
reproduce today (confirmed: two independent runs, one on a contended login
node and one on a clean ``desktop`` slurm allocation, agree with each other
and disagree with the doc). **The plan's J1 status needs correcting, not this
test.**

WHAT THE GATE CURRENTLY FINDS, AND WHY THIS TEST IS EXPECTED TO FAIL
----------------------------------------------------------------------
Re-measured 2026-09-16 against the corrected ``[:6]`` position, same policy,
same 600 s, ``LANERL_TOPONLY=1``, on a clean ``desktop`` slurm allocation
(``slurm/parity_g3.sbatch``), server-side ``LANERL_AUTOBUY=0`` (see
``run_oracle_on_server``'s docstring for why -- checked, and confirmed inert
for this specific run's cs/attacks/deaths either way, but philosophically
still the right default for a gate that is supposed to isolate last-hitting
from an item system the sim does not have)::

    sim (JAX):  cs=9  approach_decisions=6998  attacks=473  moves=0     holds=10529  deaths=5
    server:     cs=4  approach_decisions=3197  attacks=86   moves=0     holds=14717  deaths=1

This is the OPPOSITE direction from the stale figure above: the sim now
scores MORE CS and gets MORE attack opportunities than the server, not fewer.
``moves=0`` on both sides is expected and correct -- the oracle's
``hold_position=True`` fallback never returns a move order post-handover (see
``last_hit_oracle.decide``), so every post-approach decision is an attack or a
hold, on both sides, by construction.

What was checked and ruled OUT as the cause of the remaining gap:

* **Champion attack damage not scaling with level** -- real, C#-verified
  (``Stats.LevelUp``, ``Stats.cs:270-271``) and FIXED here (``sim/step.py``'s
  ``tick``, ``profiles.py``'s ``ad_per_level`` column,
  ``lanerl_jax/sim/tests/test_champion_level_scaling.py``). Moved the sim's
  in-band count by less than 1% on this exact scenario (468 -> 473,
  ``lanerl_jax/parity/hp_band.py``'s own before/after report), because the
  sim's champion barely levels (1..5) while it keeps dying and losing its
  proximity-XP window -- a real bug, correctly fixed, just not the dominant
  term here.
* **``LANERL_AUTOBUY`` / Doran's Shield** -- real, C#-verified asymmetry
  (``LanerlHooks.AutoBuyUndriven`` buys the server's champion +80 max HP and
  +1.2 HP/s regen for free, at boot, that the sim has nowhere -- see
  ``run_oracle_in_sim``'s docstring). Controlled for in the instrument
  (``run_server_band`` defaults it off). A clean, same-script, same-node A/B
  on the server (autobuy on vs off, everything else identical) produced
  IDENTICAL cs/attacks/deaths -- the one death that occurs on the server is
  not prevented or even delayed meaningfully by the extra HP pool. Real
  mechanism, not the explanation for this pattern.
* **``enable_call_for_help=True``** -- tried on the strength of
  ``docs/CALL_FOR_HELP_SWITCH_RATE.md`` part 6 (measured to help a
  champion-in-lane scenario). Made this gate dramatically WORSE (sim cs
  9 -> 0, deaths 5 -> 8), because that scenario (``StandInWave``) never
  attacks and so only ever exercises call-for-help's release side, while this
  oracle attacks routinely and every landed swing is itself a
  ``CHAMPION_ATTACKING_MINION`` (priority 5, beats any minion's own 6-9) call
  for help that recruits fresh aggressors onto the champion. Reverted; see
  ``last_hit_drive.run_oracle_in_sim``'s inline comment for the citation.

What was NOT ruled out, and is the leading open suspect: the sim's champion
dies 5 times to the server's 1, and each death costs a full fountain-to-lane
walk (``approach_decisions`` 6998 vs 3197) that the champion cannot farm
during. A direct state-level test (not checked in, see the session that
produced this comment) found six red minions simultaneously locked onto the
idle champion moments before one sim death, consistent with the ALREADY
DOCUMENTED, ALREADY MEASURED pile-up-without-release behaviour recorded in
``lanerl_jax.sim.targeting.call_for_help_map``'s own docstring ("the server
RELEASED... while the sim released never... sim mean 4.14 and max 12
simultaneous attackers"). Enabling call-for-help is the documented fix for
exactly that pile-up and was just shown, above, to make an ACTIVE oracle
worse rather than better -- so this remains open. Per the instructions this
gate was built under: report the disagreement, do not loosen the tolerance to
make it pass.
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

    ``autobuy=False``: see ``run_oracle_on_server``'s docstring -- this gate
    isolates last-hitting, and the server's champion should not be scoring a
    free defensive item the sim has no way to model.

    See the module docstring for the measured tolerance and for the gap this
    is currently finding, including why the number this test used to compare
    against (server CS=10) was stale and what replaced it. This assertion is
    written to the gate's real criterion, not to what happens to pass today --
    so as of 2026-09-16 it FAILS, and that failure is the deliverable: sim
    CS=9 against the server's CS=4 over the same 600 s, same policy, same
    scripted approach -- the gap did not close, but its DIRECTION reversed
    from every previously recorded measurement, which is itself evidence the
    old baseline was never comparable to begin with (see the module
    docstring's staleness note).
    """
    sim = run_oracle_in_sim(decisions=DECISIONS)
    server = run_oracle_on_server(
        decisions=DECISIONS, port_base=46000, bot_seed=4242, tag="last_hit_gate",
        autobuy=False)

    # Deaths are reported first because they dominate CS: each one costs a
    # respawn plus a walk back that the champion cannot farm during
    # (approach_decisions balloons from ~1300 for a deathless run to 6998 for
    # the sim's 5 deaths here). NOT asserted to be zero any more -- unlike the
    # old [:7]-waypoint bug this guarded against (a champion parked inside the
    # enemy turret's range, dying to TOWER fire), the server's own current
    # baseline under the CORRECTED [:6] position sits RIGHT AT a life/death
    # boundary late in the episode: four independent runs (one contended, two
    # on a clean slurm allocation, one via this exact test) agree EXACTLY on
    # cs=4/attacks=86/approach_decisions=3197 every time, but split 3:1 on
    # whether that boundary tick reads as death=0 or death=1 -- a genuine
    # near-miss, not the settled "0" the old comment claimed. Asserting a
    # specific value here would be pinning a coin flip, not guarding an
    # invariant; deaths are logged, not asserted.
    if sim.deaths or server.deaths:
        print(f"deaths: sim={sim.deaths} server={server.deaths} "
              f"(each one costs a fountain walk; see approach_decisions below)")

    gap = sim.cs - server.cs
    assert abs(gap) <= CS_TOLERANCE, (
        f"oracle CS@10 disagrees between sim and server by {gap:+d} "
        f"(tolerance {CS_TOLERANCE}): "
        f"sim cs={sim.cs} approach_decisions={sim.approach_decisions} "
        f"attacks={sim.attacks} moves={sim.moves} holds={sim.holds} "
        f"deaths={sim.deaths}; "
        f"server cs={server.cs} approach_decisions={server.approach_decisions} "
        f"attacks={server.attacks} moves={server.moves} holds={server.holds} "
        f"deaths={server.deaths} log={server.log_path}. "
        "This is the behavioural gate (J1 gate 3, docs/JAX_REWRITE_PLAN.md); "
        "champion-AD level-scaling and LANERL_AUTOBUY have both been checked "
        "and ruled out as the explanation (see the module docstring) -- the "
        "leading open suspect is the sim's minion-pile-up-without-release "
        "behaviour (lanerl_jax.sim.targeting.call_for_help_map's own "
        "docstring), and enabling call-for-help for this scenario has been "
        "tried and makes it WORSE, not better (see the module docstring) -- "
        "do not raise CS_TOLERANCE to silence this, fix the cause instead."
    )
