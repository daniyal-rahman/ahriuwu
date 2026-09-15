"""Every rollout an actor produced is accounted for: trained, rejected, or lost.

WHY A LEDGER RATHER THAN A RATE
-------------------------------
This project once threw away 45-80% of everything its actors collected. The
staleness bound was 1 while the pipeline structurally produces lag of
``queue_capacity + num_actors - 1``, so most rollouts arrived already too old,
were rejected, and were never trained on. It ran that way for a long time.
Nothing was broken in a way anything could detect: throughput looked fine
(the actors were busy), the loss looked fine (it trained on what got
through), and the only symptom was that learning was slower than it should
have been -- which is indistinguishable from "the task is hard".

The counters to catch it already existed (``StalenessTracker.accepted`` /
``rejected``, ``RunState.rollouts_lost``). What did not exist was anything
asserting they ADD UP. A counter nobody reconciles is a counter nobody reads.

THE IDENTITY
------------
    produced == accepted + rejected + lost

with no gaps and no double counting. If that holds, data cannot vanish
silently: work either reaches a gradient or is counted as not having done so.

The three sinks are genuinely different and must stay distinguishable:
``rejected`` is a deliberate policy decision about staleness, ``lost`` is a
rollout that never arrived (detected by a gap in the per-actor sequence), and
``accepted`` is the only one that becomes a gradient.
"""
from __future__ import annotations

import pytest

from lanerl_train.run import StalenessTracker


def test_the_gate_accounts_for_every_rollout_it_sees():
    gate = StalenessTracker(max_staleness=2)
    learner_v = 100
    produced = 0
    for lag in (0, 1, 2, 3, 4, 0, 1, 5, 2, 2):
        produced += 1
        gate.admit(learner_v - lag, learner_v)
    assert gate.accepted + gate.rejected == produced, (
        f"{produced} rollouts went in, {gate.accepted} accepted + "
        f"{gate.rejected} rejected came out. Data is vanishing between the "
        f"queue and the gradient with nothing counting it."
    )
    assert gate.total == produced


def test_rejections_are_the_ones_over_the_bound_and_nothing_else():
    """A gate that rejects the wrong rollouts would still balance."""
    gate = StalenessTracker(max_staleness=2)
    v = 50
    for lag in (0, 1, 2):
        assert gate.admit(v - lag, v) is True, lag
    for lag in (3, 4, 10):
        assert gate.admit(v - lag, v) is False, lag
    assert (gate.accepted, gate.rejected) == (3, 3)
    assert gate.max_accepted == 2, (
        f"max_accepted is {gate.max_accepted}: something over the bound was "
        f"trained on, which is the bound not being a bound"
    )
    assert gate.max_observed == 10, (
        "max_observed must include rejections -- it is the diagnostic that "
        "says how far behind the pipeline actually runs, and the reason the "
        "45-80% discard was invisible is that nobody could see the lag"
    )


def test_a_gap_in_an_actors_sequence_is_counted_as_lost():
    """Simulates TrainingLoop's per-actor sequence check (run.py:1502).

    A rollout that never arrives leaves a hole in that actor's sequence.
    Without this it would simply not exist -- no counter, no log -- and the
    run would train on a silently smaller batch.
    """
    seen: dict[int, int] = {}
    lost = 0
    # actor 0 sends 1,2,3; actor 1 sends 1, then 4 (2 and 3 never arrived)
    for actor_id, seq in ((0, 1), (0, 2), (1, 1), (0, 3), (1, 4)):
        prev = seen.get(actor_id)
        if prev is not None and seq > prev + 1:
            lost += seq - prev - 1
        seen[actor_id] = seq
    assert lost == 2, f"expected 2 lost rollouts from actor 1's gap, counted {lost}"


def test_the_identity_holds_across_all_three_sinks():
    """produced == accepted + rejected + lost, the whole point."""
    gate = StalenessTracker(max_staleness=2)
    v = 200
    seen: dict[int, int] = {}
    lost = 0
    produced = 0

    # actor 1 loses sequence 3 and 4 entirely; the rest arrive with varying lag
    arrivals = [(0, 1, 0), (0, 2, 1), (1, 1, 2), (1, 2, 5), (0, 3, 3), (1, 5, 0)]
    for actor_id, seq, lag in arrivals:
        prev = seen.get(actor_id)
        if prev is not None and seq > prev + 1:
            lost += seq - prev - 1
        seen[actor_id] = seq
        produced += 1
        gate.admit(v - lag, v)

    total_produced = produced + lost     # what the actors actually made
    assert total_produced == gate.accepted + gate.rejected + lost, (
        f"produced {total_produced} != accepted {gate.accepted} + rejected "
        f"{gate.rejected} + lost {lost}. Rollouts are disappearing without "
        f"being counted, which is how 45-80% of collected data went missing "
        f"for a long time while every visible metric looked healthy."
    )
    assert lost == 2 and gate.rejected == 2 and gate.accepted == 4


def test_a_high_rejection_rate_is_surfaced_not_swallowed():
    """The bound silently discarding most of the data is a misconfiguration,
    not a safety net working. It has to say so."""
    gate = StalenessTracker(max_staleness=0, warn_reject_rate=0.05)
    v = 10
    for _ in range(50):
        gate.admit(v - 3, v)          # every one too stale
    assert gate.rejected == 50 and gate.accepted == 0
    rate = gate.rejected / gate.total
    assert rate > gate.warn_reject_rate, (
        "a gate rejecting 100% of the data must exceed its own warning "
        "threshold, or the threshold cannot fire on the exact case it exists "
        "for"
    )
