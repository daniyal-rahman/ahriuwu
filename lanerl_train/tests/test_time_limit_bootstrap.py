"""A time-limit ending is a TRUNCATION, not a termination.

THE DISTINCTION
---------------
An episode that ends because the agent died has genuinely no future: the
return really does stop, and ``done = 1`` is correct. An episode that ends
because the clock ran out has a perfectly good future that we simply chose
not to simulate, and its last transition should be bootstrapped with
``gamma * V(s_T)``. Treating the second like the first teaches the critic
that the world ends on a timer.

WHAT THE CODE DOES TODAY
------------------------
``lane_wiring.collect_rollout`` has the end reason in hand -- ``prev_dones``
is ``Dict[int, str]``, carrying ``"time"`` or ``"death_team_<n>"`` -- and
ignores it::

    if i in prev_dones:
        done_t[row] = 1.0
        if ctx.terminal_valid:
            rewards[row] = float(ctx.terminal_values.get(team, 0.0))

so every ending, including the clock, is a termination.

This matters here because with ``--no-end-on-death`` EVERY training episode
ends on the clock. There is no configuration in which the truncation path is
rare.

HONEST MAGNITUDE FOR THIS PROJECT
---------------------------------
The literature figure for this bug is 20-40% on MuJoCo, and that does NOT
transfer directly: there episodes are short relative to the discount horizon,
so most steps are affected. Here an episode is 18,000 decisions and the
horizon is 3,600 (120 s at 30 Hz), and ``gamma**3600 ~ 0.37``, so the missing
bootstrap has meaningful weight over roughly the last fifth of an episode and
decays away before that. Real, bounded, and biased in one direction: the
critic is taught that value decays to zero towards the end of every game,
when in a real lane it does not.

WHY THIS IS XFAIL RATHER THAN FIXED
-----------------------------------
The fix needs ``V(s_T)``, and ``s_T`` is ``result.terminal_obs[i]`` -- which
is available -- but the recurrent state that goes with it is the state the
actor held BEFORE the driver zeroed it for the new episode. Threading that
out of the hot path is a real change to rollout collection, and doing it
carelessly would corrupt every rollout rather than bias the tail of one. It
is worth doing properly rather than quickly.
"""
from __future__ import annotations

import inspect

import pytest

from lanerl_train import lane_wiring


def test_the_end_reason_is_available_where_done_is_set():
    """The information needed for the fix is already in scope.

    This is the part that is not in doubt, and it is worth pinning: if
    someone later changes ``prev_dones`` to a set of instance ids, the fix
    below becomes much harder and nothing would have complained.
    """
    src = inspect.getsource(lane_wiring.collect_rollout)
    assert "prev_dones: Dict[int, str]" in src, (
        "prev_dones no longer carries the end REASON, so truncation can no "
        "longer be told from termination at the point where done is set"
    )
    assert "if i in prev_dones:" in src


@pytest.mark.xfail(
    reason=(
        "KNOWN OPEN BUG: a time-limit ending is recorded as a termination. "
        "collect_rollout sets done_t[row] = 1.0 for every entry in "
        "prev_dones without consulting the reason, so the clock running out "
        "is indistinguishable from a death and the last transition is never "
        "bootstrapped with gamma*V(s_T).\n"
        "\n"
        "With --no-end-on-death every training episode ends on the clock, so "
        "this is the common path, not a corner. Magnitude here is smaller "
        "than the 20-40% quoted for MuJoCo -- an episode is 18,000 decisions "
        "against a 3,600-decision horizon, so it biases roughly the last "
        "fifth -- but it is one-directional: the critic learns value decays "
        "to zero at the end of every game.\n"
        "\n"
        "The fix needs V(s_T) together with the recurrent state the actor "
        "held before the driver zeroed it for the new episode. terminal_obs "
        "is available; the state is not, without threading it out of the hot "
        "path."
    ),
    strict=False,
)
def test_a_time_limit_ending_is_not_recorded_as_a_termination():
    src = inspect.getsource(lane_wiring.collect_rollout)
    block = src[src.index("if i in prev_dones:"):]
    block = block[:block.index("buffer.add(")]
    assert ('prev_dones[i]' in block or 'reason' in block), (
        "done is set for every episode end without looking at the reason: "
        "the clock running out is being trained on as if the world ended"
    )
