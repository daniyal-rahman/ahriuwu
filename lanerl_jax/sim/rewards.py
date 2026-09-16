"""Gold, CS, experience and levelling on death.

The asymmetry between gold and XP is the whole of laning
--------------------------------------------------------
``AttackableUnit.Die``::

    // experience: SHARED among enemy champions within ExpRadius2 of the corpse
    champs = GetChampionsInRangeFromTeam(Position, ExpRadius2,
                                         GetEnemyTeam(Team), alive: true);
    expPerChamp = Stats.ExpGivenOnDeath.Total / champs.Count;

``Champion.OnKill``::

    // gold: the KILLER only, and only for a Minion
    if (deathData.Unit is Minion) {
        ChampStats.MinionsKilled += 1;
        gold = deathData.Unit.Stats.GoldGivenOnDeath.Total;
        if (gold <= 0) return;
        AddGold(deathData.Unit, gold);
    }

So **experience is proximity-shared and gold is last-hit-only**.  That single
asymmetry is why last-hitting is a skill at all: standing near a dying wave is
enough for levels, and landing the killing blow is required for income.  An
agent that models minion HP badly still levels normally and earns nothing, which
is exactly the failure mode the reward head was built to detect.

``ExpRadius2`` is 1600 (``ObjAIBaseVariables``, overridable as
``ai_ExpRadius2``).  Note it is measured from the **corpse**, not from the
killer.

Levelling
---------
``Champion.AddExperience`` crosses the ``ExpCurve`` thresholds and calls
``Stats.LevelUp`` once per level gained, and level-ups are not linear -- see
:mod:`lanerl_jax.sim.combat`.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .state import Kind

__all__ = ["EXP_RADIUS", "AMBIENT_GOLD_DELAY_MS", "AMBIENT_GOLD_AMOUNT",
           "AMBIENT_GOLD_INTERVAL_MS", "DeathRewards", "death_rewards",
           "level_for_xp", "ambient_gold"]

#: Ambient ("passive") gold, measured from a 600 s idle server run rather than
#: derived. The Content constants are `AmbientGoldAmount 9.5` /
#: `AmbientGoldInterval 5.0`, but the observed behaviour is **0.9502 every
#: ~517 ms** -- a tenth of each, with the 17 ms being one tick of overshoot on a
#: 500 ms timer decremented by 16.667 per tick. The rate works out the same
#: (1.9/s); the granularity does not, and granularity is what a reward signal
#: sees.
#:
#: Starts at `ObjAIBaseVariables.AmbientGoldDelay` = 90 s, and the first tick
#: fires immediately at 90 s rather than 500 ms later.
AMBIENT_GOLD_DELAY_MS = 90_000.0
AMBIENT_GOLD_AMOUNT = 0.95
AMBIENT_GOLD_INTERVAL_MS = 500.0

#: ``GlobalData.ObjAIBaseVariables.ExpRadius2``.
EXP_RADIUS = 1600.0


class DeathRewards(NamedTuple):
    gold: Any        # (N,) gold gained by each unit this tick
    xp: Any          # (N,) experience gained
    cs: Any          # (N,) minions killed this tick


def death_rewards(*, died: jax.Array, killer: jax.Array, x: jax.Array,
                  y: jax.Array, team: jax.Array, kind: jax.Array,
                  alive: jax.Array, gold_on_death: jax.Array,
                  xp_on_death: jax.Array,
                  exp_radius: float = EXP_RADIUS) -> DeathRewards:
    """Distribute the rewards for every unit that died this tick.

    Args:
      died:   ``(N,)`` bool, units that died on this tick
      killer: ``(N,)`` index of whoever landed the killing blow, -1 if none

    Gold and CS go to ``killer`` when the killer is a champion and the victim is
    a minion. Experience is split among **living enemy champions within
    ``exp_radius`` of the victim**, which is computed from the victim's position
    and so is not the same set as "champions near the killer".
    """
    n = x.shape[0]
    is_champ = kind == Kind.CHAMPION
    victim_is_minion = kind == Kind.LANE_MINION

    # ---- gold and CS: the killer alone -----------------------------------
    k = jnp.clip(killer, 0, n - 1)
    pays = died & victim_is_minion & (killer >= 0) & is_champ[k] \
        & (gold_on_death > 0)
    gold = jnp.zeros((n,), x.dtype).at[k].add(
        jnp.where(pays, gold_on_death, jnp.zeros_like(gold_on_death)))
    cs = jnp.zeros((n,), jnp.int32).at[k].add(jnp.where(pays, 1, 0))

    # ---- experience: shared by proximity to the CORPSE --------------------
    d2 = (x[None, :] - x[:, None]) ** 2 + (y[None, :] - y[:, None]) ** 2
    # row = victim, col = candidate champion
    eligible = (
        died[:, None] & is_champ[None, :] & alive[None, :]
        & (team[None, :] != team[:, None])
        & (d2 <= exp_radius * exp_radius)
    )
    share_count = jnp.sum(eligible, axis=1)
    per = jnp.where(share_count > 0, xp_on_death / jnp.maximum(share_count, 1),
                    jnp.zeros_like(xp_on_death))
    xp = jnp.sum(jnp.where(eligible, per[:, None], jnp.zeros_like(d2)), axis=0)

    return DeathRewards(gold=gold, xp=xp.astype(x.dtype), cs=cs)


def level_for_xp(xp: jax.Array, curve: jax.Array) -> jax.Array:
    """Level implied by total experience.

    ``curve`` is ``(18,)`` cumulative thresholds, ``curve[i]`` being the XP to
    reach level ``i+1`` (so ``curve[0] == 0``). Returns a level in 1..18.
    """
    return (1 + jnp.sum(xp[:, None] >= curve[None, :], axis=1) - 1).astype(jnp.int8)


def ambient_gold(t_ms: Any, gold_timer: Any, is_champion: Any, xp: Any = None):
    """One tick of ``Champion.Update``'s ambient gold block.

    Returns ``(gold_gained, new_timer)``. Ambient **experience** is not modelled
    because ``ChampionVariables.AmbientXPAmount`` is **0.0** -- the block exists
    in the server and pays nothing.
    """
    import jax.numpy as jnp

    generating = is_champion & (t_ms >= AMBIENT_GOLD_DELAY_MS)
    fires = generating & (gold_timer <= 0)
    gained = jnp.where(fires, AMBIENT_GOLD_AMOUNT, 0.0)
    new_timer = jnp.where(
        fires, AMBIENT_GOLD_INTERVAL_MS,
        jnp.where(generating, gold_timer - (1000.0 / 60.0), gold_timer))
    return gained.astype(gold_timer.dtype), new_timer
