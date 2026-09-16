"""Health regeneration -- base, and Garen's passive.

Not modelling this is why the sim's champion died seven times in a 600 s
oracle-driven episode while the server's died **zero** times. It is the whole
of that gap: the server's Garen heals through minion chip damage, and ours
never healed at all.

TWO CLOCKS, AND THEY ARE INDEPENDENT
------------------------------------
**Base regen** is a stat, applied from ``AttackableUnit.Update``::

    _statUpdateTimer += diff;
    while (_statUpdateTimer >= 500) {
        Stats.Update(_statUpdateTimer);            // 500 ms accumulator
        _statUpdateTimer -= 500;
    }

and ``Stats.Update`` does ``CurrentHealth += HealthRegeneration.Total * diff *
0.001f`` with ``diff`` in milliseconds -- so each call adds ``Total * 0.5`` and
the stored rate is **HP per second**, whatever League's per-5-seconds display
convention suggests. There is no combat gate on it at all. On this map minions
and turrets are both 0.0; Garen is 1.568 + 0.1 per level.

**Garen's passive ("Perseverance")** is a separate mechanism on its own ~1 s
accumulator inside the ``GarenPassiveHeal`` buff script -- a direct ``TakeHeal``
call, not a ``HealthRegeneration`` modifier, so it would NOT come along for free
with generic regen::

    healingTimer += diff;
    if (healingTimer > 1000f) {
        float healAmount = HEALTH_PERCENTAGES[GetCorrectLevelIndex(level)]
                         * unit.Stats.HealthPoints.Total;
        unit.TakeHeal(unit, healAmount);
        healingTimer = 0;
    }

with ``HEALTH_PERCENTAGES = {0.004, 0.008, 0.02}`` and brackets at level 11 and
16, and an out-of-combat requirement of ``OUT_OF_COMBAT_COOLDOWNS = {9, 6, 4}``
seconds on the same brackets.

THE PART THAT DECIDES A LANE
----------------------------
``CharScriptGaren.ShouldPassiveTurnOff`` returns **false** -- meaning the
passive keeps running -- when the attacker's ``UnitTags`` is any of::

    Minion, Minion_Lane, Minion_Lane_Siege, Minion_Lane_Super, Minion_Summon

So **ordinary minion damage does not interrupt Garen's passive regeneration**.
In a lane that is the difference between healing continuously while farming and
never healing at all, and it is why the server's Garen survives 600 s of minion
chip damage without dying once.

Note this exception list is also where the server has a real bug (see
``docs/PORT_AUDIT_COMBAT.md``'s UnitTag row, and ``sim/step.py``'s
``breaks_combat_pair``, which is what actually computes this -- this module
only consumes the resulting ``ms_since_damaged``): ``UnitTag`` declares no
explicit flag values, so BOTH cannon's and SUPER minions' OR-ed tag collide
with ``Monster`` and are NOT in the exceptions list, despite
``Minion_Lane_Siege``/``Minion_Lane_Super`` being named in it -- the list
checks an exact raw-integer value, and a real minion's tag is the OR of
several. Cannon and super autoattacks therefore DO break the passive below
Garen's own level 11, and stop doing so from level 11 onward (a second check,
keyed on the SAME Monster-value collision, gated on the DEFENDER's level) --
while melee and caster autoattacks never do, at any level. That is reproduced
here, because parity means reproducing the server rather than what the
mechanic obviously intended.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .state import Kind

__all__ = ["STAT_TICK_MS", "GAREN_HEAL_PCT", "GAREN_HEAL_PERIOD_MS",
           "OUT_OF_COMBAT_MS", "garen_heal_bracket", "step_regen", "RegenOut"]

#: ``AttackableUnit.Update``'s stat accumulator.
STAT_TICK_MS = 500.0
#: ``GarenPassiveHeal.HEALTH_PERCENTAGES``, by level bracket.
GAREN_HEAL_PCT = (0.004, 0.008, 0.02)
#: ``CharScriptGaren.OUT_OF_COMBAT_COOLDOWNS``, seconds, same brackets.
OUT_OF_COMBAT_MS = (9000.0, 6000.0, 4000.0)
#: the passive's own internal accumulator
GAREN_HEAL_PERIOD_MS = 1000.0


def garen_heal_bracket(level: Any, xp: Any = jnp) -> Any:
    """``GetCorrectLevelIndex``: 0 below 11, 1 from 11, 2 from 16."""
    return (level >= 11).astype(jnp.int32) + (level >= 16).astype(jnp.int32)


class RegenOut(NamedTuple):
    hp: jax.Array
    stat_timer: jax.Array
    heal_timer: jax.Array


def step_regen(*, hp, max_hp, alive, kind, level, hp_regen, stat_timer,
               heal_timer, ms_since_damaged, delta_ms: float) -> RegenOut:
    """One tick of base regen plus Garen's passive.

    ``ms_since_damaged`` is time since the unit last took damage that counts as
    combat -- minion autoattacks deliberately do NOT reset it, see the module
    docstring.
    """
    dtype = hp.dtype
    can_regen = alive & (hp > 0) & (hp < max_hp)

    # ---- base regen: 500 ms accumulator, no combat gate -------------------
    st = stat_timer + delta_ms
    fires = st >= STAT_TICK_MS
    # Stats.Update(_statUpdateTimer) is called with the accumulated value
    # BEFORE the subtraction, so the amount is rate * accumulated_ms / 1000.
    base = jnp.where(fires & can_regen, hp_regen * st * 0.001, 0.0)
    st = jnp.where(fires, st - STAT_TICK_MS, st)

    # ---- Garen's passive: its own 1 s accumulator, gated on out-of-combat --
    bracket = garen_heal_bracket(level)
    pct = jnp.asarray(GAREN_HEAL_PCT, dtype)[bracket]
    ooc = jnp.asarray(OUT_OF_COMBAT_MS, dtype)[bracket]
    eligible = (kind == Kind.CHAMPION) & alive & (ms_since_damaged >= ooc)
    ht = jnp.where(eligible, heal_timer + delta_ms, jnp.zeros_like(heal_timer))
    heals = eligible & (ht > GAREN_HEAL_PERIOD_MS)
    passive = jnp.where(heals & (hp > 0), pct * max_hp, 0.0)
    ht = jnp.where(heals, jnp.zeros_like(ht), ht)

    # The clamp lives INSIDE the server's guard:
    #
    #     if (HealthRegeneration.Total > 0 && CurrentHealth < HealthPoints.Total
    #         && CurrentHealth > 0) {
    #         newHealth = CurrentHealth + ...;
    #         CurrentHealth = Math.Min(HealthPoints.Total, newHealth);
    #     }
    #
    # so a unit that regenerates nothing is not touched at all. Clamping every
    # living unit instead killed a test fixture outright: it carried hp = 1.0
    # with max_hp = 0, and `min(1.0, 0.0)` zeroed it with no attacker, so the
    # minion died and no champion was credited the kill.
    gain = (base + passive).astype(dtype)
    new_hp = jnp.where(gain > 0, jnp.minimum(hp + gain, max_hp), hp)
    return RegenOut(hp=new_hp, stat_timer=st.astype(dtype),
                    heal_timer=ht.astype(dtype))
