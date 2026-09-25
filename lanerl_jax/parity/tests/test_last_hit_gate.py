"""J1 gate 3: an oracle last-hitter must score the same CS@10 in both places.

WHY THIS GATE IS DIFFERENT FROM EVERYTHING ELSE IN ``lanerl_jax/parity``
-------------------------------------------------------------------------
Every other instrument here diffs *state* -- positions and health tick by
tick, or the distribution of live minions. Both can hold inside tolerance
while the thing the simulator is *for* is broken: the minion-population
comparison (``lanerl_jax/sim/tests/test_lane.py``) sat at +17% while the lane
underneath it was collapsing to one side. This gate instead runs one fixed,
deterministic policy -- :mod:`lanerl_jax.parity.archive.last_hit_oracle` -- against
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
:data:`~lanerl_jax.parity.archive.last_hit_drive.APPROACH_WAYPOINTS`. This is
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
:data:`~lanerl_jax.parity.archive.last_hit_drive.APPROACH_WAYPOINTS` from
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

    sim (JAX):  cs=7  approach_decisions=1768  attacks=72   moves=0     holds=16160  deaths=1
    server:     cs=4  approach_decisions=3197  attacks=86   moves=0     holds=14717  deaths=1

The current source-faithful fixture scores more CS in the sim despite the sim
now getting *fewer* attack-decision frames.  ``hp_band.py`` makes the reason
observable: the 72 sim frames form 7 lethal windows (mean 10.29 frames), while
the 86 server frames form only 4 windows (mean 21.50 frames).  One sim window
currently becomes one CS; this is a minion HP/crossover cadence discrepancy,
not a duplicated-attack-order counter or an excess death/pathing exposure.
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
* **Call for help disabled** -- not a valid candidate or baseline.  The server
  broadcasts aggro on every landed hit; canonical ``step_decision`` now enables
  that source-derived mechanism by default.  The old OFF ablation was a
  regression-shaped tuning experiment and is retained only as history, not
  gate evidence.

The remainder of this historical death/isolation investigation predates the
current canonical call-for-help fixture. It remains useful provenance for the
old cs=9/473-attack run, but it is **not** the leading explanation now: the
fresh fixture has one death on each side and isolates the remaining gap to the
number of distinct HP-band crossovers.

**The release rule itself is not the cause -- checked against the C# source
directly, not inferred.** ``lanerl_jax.sim.minion_ai``'s
``test_a_minion_holding_the_idle_champion_is_NOT_displaced_by_a_fresh_minion``
is a direct, source-derived test (``LaneMinionAI.cs``'s ``ReevaluateBehavior``
returns ``AttackTo`` the instant ``targetIsStillValid``, never reaching the
priority-comparing ``FoundNewTarget()`` while a valid incumbent holds) proving
the sim already matches the server: a minion that has validly acquired the
champion is not released by a fresh candidate merely entering range, on
either side. A minion sitting on the champion is therefore not, by itself,
a targeting bug -- it is what the rule produces once that minion has nothing
else nearby worth switching to.

**That reframes the question as WHERE the champion ends up relative to its
own wave**, which ``lanerl_jax/parity/archive/isolation.py`` (archived 2026-09-23) measures directly, in
both engines, over this same scripted scenario (both drivers walk to the
identical ``APPROACH_WAYPOINTS`` coordinate, so the destination itself is
not in question)::

                                sim      server    ratio
    0 allies within 1500 u    32.3%     19.4%      1.7x
    nearest-ally > 3000 u     13.1%      7.0%      1.9x
    mean nearest-ally dist     1263       874      +45%
    median nearest-ally dist    280       306      ~equal

The medians agreeing while the tails diverge is the real finding: the
champion's TYPICAL position relative to its own wave is right in both
engines, but the sim shows a measurably heavier tail of decisions where he
has no ally nearby at all. That is consistent with -- not proof of -- the
excess deaths: more isolated decisions is more opportunity for an
unrecoverable, individually-correct lock-on to accumulate, but this
instrument does not trace any specific death back to a specific isolated
stretch, and no other contributor has been ruled out. Read the causal step
as an inference resting on a real, measured tail effect, not as closed.

**Handoff.** The mechanism behind a wider equilibrium is very likely
lane-equilibrium/collision separation -- ``sim/collision.py`` applies one
push-apart per unit per tick from a pre-tick snapshot where the server
resolves collisions sequentially and can push a unit several times in one
tick, each visible to the next (that module's own booked-approximation
note). That is J1 gate 1's territory (the sequential-collision port is in
flight there as of 2026-09-16); this gate's job stops at making the tail
effect measurable and reproducible for whoever picks it up next, not at
diagnosing its root cause -- noted here so gate 1 and gate 2 (whose own
lane-equilibrium work this bears on just as much) can find it. Per the
instructions this gate was built under: report the disagreement, do not
loosen the tolerance to make it pass.
"""
from __future__ import annotations

import inspect
import os

import pytest

from lanerl_jax.parity.archive.last_hit_drive import (
    DECISIONS_600S,
    run_oracle_in_sim,
    run_oracle_on_server,
)
from lanerl_jax.sim.step import step_decision

#: Exact match. See the module docstring: two 600 s server runs under
#: different bot_seeds produced bit-identical decision streams, so there is no
#: measured run-to-run noise to build slack out of.
CS_TOLERANCE = 0

#: Override for a quick local smoke run, e.g. ``LANERL_LAST_HIT_DECISIONS=1800
#: pytest -k last_hit_gate``. The gate's own number -- CS@10, 18,000 decisions
#: -- is the default because a shorter run answers a different question (CS at
#: some other clock, which nothing else reports against).
DECISIONS = int(os.environ.get("LANERL_LAST_HIT_DECISIONS", str(DECISIONS_600S)))


def test_gate3_canonical_call_for_help_default_is_enabled():
    """The server always broadcasts on damage; OFF is only an explicit ablation."""
    assert inspect.signature(step_decision).parameters[
        "enable_call_for_help"].default is True


def test_gate3_canonical_sim_path_is_routed():
    """The server and production simulator both route Move orders.

    ``table_disabled`` is deliberately opt-in so a raw two-point segment can
    remain available for PATH-006 isolation without quietly becoming gate
    evidence again.
    """
    assert inspect.signature(run_oracle_in_sim).parameters[
        "table_disabled"].default is False


@pytest.mark.slow
@pytest.mark.skipif(
    not __import__("lanerl_train.paths", fromlist=["paths"]).server_available(),
    reason="vendored server build not available",
)
def test_oracle_scores_the_same_cs_in_sim_and_server():
    """J1 gate 3. Boots one real server; takes several minutes.

    Both sides walk :data:`~lanerl_jax.parity.archive.last_hit_drive.APPROACH_WAYPOINTS`
    before either ever calls the oracle -- see the module docstring for why a
    first version of this test measured pathing instead of last-hitting, and
    why that walk-in is now scripted rather than oracle-driven.

    ``autobuy=False``: see ``run_oracle_on_server``'s docstring -- this gate
    isolates last-hitting, and the server's champion should not be scoring a
    free defensive item the sim has no way to model.

    See the module docstring for the measured tolerance and current source-
    faithful gap. The assertion is written to the gate's real criterion, not
    to what happens to pass today: the current full fixture fails at sim CS=7
    versus server CS=4, even though it has fewer eligible decision frames
    (72 versus 86). The deliverable is the measured HP-window disagreement,
    not a loosened tolerance.
    """
    sim = run_oracle_in_sim(decisions=DECISIONS)
    server = run_oracle_on_server(
        decisions=DECISIONS, port_base=46000, bot_seed=4242, tag="last_hit_gate",
        autobuy=False)

    # Deaths are logged rather than asserted.  The current canonical run has
    # one on each side, so death/respawn exposure cannot explain its CS gap;
    # future mechanics work may nevertheless move this near-boundary value.
    if sim.deaths or server.deaths:
        print(f"deaths: sim={sim.deaths} server={server.deaths} "
              f"(each one costs a fountain walk; see approach_decisions below)")

    gap = sim.cs - server.cs
    assert abs(gap) <= CS_TOLERANCE, (
        f"oracle CS@10 disagrees between sim and server by {gap:+d} "
        f"(tolerance {CS_TOLERANCE}): "
        f"sim cs={sim.cs} approach_decisions={sim.approach_decisions} "
            f"walks={sim.walks} "
        f"attacks={sim.attacks} moves={sim.moves} holds={sim.holds} "
        f"deaths={sim.deaths}; "
        f"server cs={server.cs} approach_decisions={server.approach_decisions} "
            f"walks={server.walks} "
        f"attacks={server.attacks} moves={server.moves} holds={server.holds} "
        f"deaths={server.deaths} log={server.log_path}. "
        "This is the behavioural gate (J1 gate 3, docs/JAX_REWRITE_PLAN.md); "
        "canonical call-for-help is enabled (the server has no OFF mode). "
        "The focused HP-band diagnostic shows distinct lethal windows, not "
        "duplicated ATTACK orders; do not raise CS_TOLERANCE to silence this, "
        "fix the HP/crossover cause instead."
    )
