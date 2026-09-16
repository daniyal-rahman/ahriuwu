"""A scripted last-hitter, for J1 gate 3.

WHY THIS EXISTS, AND WHY IT IS DIFFERENT FROM EVERY OTHER PARITY CHECK
---------------------------------------------------------------------
Every other instrument in this project compares *state*: a tick-by-tick diff of
positions and health (Tier 1), or a distribution of live minions (Tier 3). Both
can agree while the thing the simulator is *for* is broken.

The minion-population comparison proved exactly that today. It sat inside its
own tolerance -- median 24 against the server's 21 -- while underneath it the
lane was collapsing to one side and ending the episode at 2 blue minions
against 28 red. A distribution over a total hides which half of it moved.

This measures **behaviour**: run one fixed policy in both implementations and
compare the score it gets. If the sim's CS@10 differs from the server's under
the same policy, something the agent can feel is different, whatever the state
diff says.

WHY A GREEDY LAST-HITTER AND NOT A "PERFECT" ONE
------------------------------------------------
The plan says "an oracle last-hitter -- perfect timing from exact HP". Perfect
timing is not implementable against either side without predicting every other
attacker's windup, and an oracle that has to model the simulator to act is a
second simulator to get wrong.

What the gate actually needs is a policy that is **identical in both**, cheap to
state, and sensitive to the mechanics under test -- attack range, windup, the
attack clock, damage mitigation, and kill attribution. Greedy last-hitting is
all four, and it is deterministic given the observation, so any CS difference is
a difference in the simulation and not in the policy.

The decision rule, in full:

1. Of the enemy minions within reach, take those this attack would kill.
2. Among those, the lowest absolute HP -- it is closest to being taken by
   someone else. Ties by unit id, so the two implementations agree.
3. If none is killable, walk to the centroid of the enemy minions (or hold if
   there are none). Deliberately no harassing: an attack that does not last-hit
   changes the wave's balance, and this policy exists to measure farming, not
   to play well.

THE CAVEAT THAT MATTERS
-----------------------
This scores what the policy *achieves*, not what is achievable. A sim whose
minions die slightly faster gives the same greedy policy a different CS without
anything being wrong with last-hitting itself. So a difference here is a
question, not a verdict -- but it is the only question in the gate list that is
asked in the units the agent is optimised in.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np

__all__ = ["MinionView", "ChampView", "Decision", "decide", "post_mitigation"]


class MinionView(NamedTuple):
    """One enemy minion, as both implementations can describe it."""

    uid: int
    x: float
    y: float
    hp: float
    armor: float
    collision_radius: float


class ChampView(NamedTuple):
    x: float
    y: float
    attack_damage: float
    attack_range: float


class Decision(NamedTuple):
    #: uid to attack, or None
    attack: Optional[int]
    #: where to walk, or None to hold
    move: Optional[Tuple[float, float]]


def post_mitigation(damage: float, resist: float) -> float:
    """``Stats.GetPostMitigationDamage``. Mirrors :mod:`lanerl_jax.sim.combat`.

    Duplicated rather than imported so the oracle stays free of JAX and can be
    driven against the server without importing the simulator it is judging.
    """
    if damage <= 0.0:
        return 0.0
    pct = 100.0 / (100.0 + resist)
    if resist < 0:
        pct = 2.0 - pct
    return damage * pct


def decide(champ: ChampView, minions: Sequence[MinionView],
           lethal_epsilon: float = 0.0, hold_position: bool = True) -> Decision:
    """The whole policy. Pure, deterministic, and identical on both sides.

    ``lethal_epsilon`` widens the kill test to absorb the fact that damage is
    quantised to 1/1024 on the wire; leave it at 0 when reading exact state.
    """
    if not minions:
        return Decision(attack=None, move=None)

    killable = []
    for m in minions:
        reach = champ.attack_range + m.collision_radius
        if (m.x - champ.x) ** 2 + (m.y - champ.y) ** 2 > reach * reach:
            continue
        dmg = post_mitigation(champ.attack_damage, m.armor)
        if m.hp <= dmg + lethal_epsilon:
            killable.append(m)

    if killable:
        # lowest HP first, then uid -- the tie-break must not depend on the
        # order the two implementations happen to enumerate units in
        best = min(killable, key=lambda m: (m.hp, m.uid))
        return Decision(attack=best.uid, move=None)

    if hold_position:
        # HOLD, do not chase. The fallback used to be "walk to the centroid of
        # the visible enemy minions", and it was the single worst thing in this
        # gate.
        #
        # A centroid is inside the wave. So the policy walked the champion into
        # the middle of the enemy minions and stood there: 9 deaths in 600 s of
        # a 10-minute episode, and every death costs a respawn plus a walk back
        # the sim cannot even path. Worse, it made the gate depend on the FOG
        # model, because which minions are visible decides where the centroid
        # is -- so a difference in vision became a difference in position, and
        # a difference in CS, none of which is about last-hitting.
        #
        # Holding makes position a constant of the experiment on both sides.
        # The approach script has already put the champion in lane; from there
        # it last-hits whatever walks into range and does nothing else. CS is
        # lower for both implementations, which does not matter: the gate
        # compares two numbers, it does not need either to be large.
        return Decision(attack=None, move=None)

    cx = float(np.mean([m.x for m in minions]))
    cy = float(np.mean([m.y for m in minions]))
    return Decision(attack=None, move=(cx, cy))
