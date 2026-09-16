"""Garen's kit: Q (Decisive Strike), W (Courage), E (Judgment), R (Demacian Justice).

Why E first
-----------
``constants.GAREN_SKILL_ORDER`` takes it at level 1 and maxes it first, with the
reason stated: *"E first: it is the farming and trading spell."* A Garen with no
abilities cannot clear a wave the way the policy will be trained to, so this is
the first one that changes what the agent can do rather than how accurately it
does it. E was built and tested first for that reason; Q, W and R follow the
same pattern (rank tables, cooldowns, cast gating, buff slots) but each hits a
different wall in how far that pattern can go without touching ``state.py`` or
``step.py``, which are owned by another agent. Each section below says exactly
where its wall is.

The spell, from ``Characters/Garen/E.cs`` and ``Buffs/Garen/GarenE.cs``
---------------------------------------------------------------------
``OnSpellPostCast`` adds a **3-second** ``GarenE`` buff to the caster and swaps
the E slot for ``GarenECancel`` on a 1 s cooldown. While the buff is up::

    damage = 10 + 12.5*(rank-1) + AD * (0.35 + 0.05*(rank-1))     # at activate
    every 500 ms:
        units = GetUnitsInRange(Owner.Position, 330f, true)
        for each enemy ObjAIBase that is not a building or turret:
            tick = damage * (0.75 if Minion else 1.0)
            TakeDamage(tick, PHYSICAL, SPELL)

and the caster has ``CanAttack`` cleared and ``Ghosted`` set for the duration,
so **E suppresses auto-attacks and passes through unit collision**. On
deactivate the E slot comes back and its cooldown starts: 13/12/11/10/9 s by
rank.

Four details that are easy to get wrong, three of which the server got wrong
first and has fixed in comments worth preserving:

* **The AD ratio is snapshotted at cast**, not recomputed per tick.
* **Minions take 0.75x.** Without it E clears waves far too fast, which is
  precisely the mechanic the agent is being trained to use.
* **The radius is 330 and it is centred on the caster**, who moves while
  spinning -- so it is re-evaluated every tick, not once.
* **Turrets and buildings are immune to it.**

The crit roll uses a *seeded* RNG in the server (``new Random(0x6A3E17)``), a
fix for an unseeded `new Random()` per tick sitting directly in an RL reward
signal. Garen's crit chance is 0 and the comparison is ``<`` against
``Next(0, 100)``, so the roll never fires; it is modelled as always-false and no
key is consumed. Flagged rather than hidden: it stops being correct the moment
a crit source enters the kit.

Q -- Decisive Strike, from ``Characters/Garen/Q.cs`` and ``Buffs/Garen/GarenQ.cs``
-----------------------------------------------------------------------------
Q is **not** an instant nuke. ``OnSpellPreCast`` (Q.cs:76-79) adds two buffs to
the caster and nothing else: ``GarenQ`` (4.5 s -- the empowerment window) and
``GarenQHaste`` (movement speed, ``1.5 + 0.75*(rank-1)`` s). The buff's own
``OnActivate`` (``Buffs/Garen/GarenQ.cs:47-66``) is what actually does anything:
it cancels Garen's in-flight swing, skips the very next one, and registers a
listener that force-swaps whichever auto-attack fires after that to
``GarenQAttack``. The damage (``30 + 25*(rank-1) + 1.4*AD`` physical) and the
silence (``1.5 + 0.25*(rank-1)`` s) are dealt from *that* empowered swing's own
``OnSpellPostCast`` (Q.cs:150-152, :142-144) -- i.e. whenever the champion's
next auto-attack actually lands, which can be anywhere from 0 to 4.5 s after
the Q order was issued, against whatever ``TargetUnit`` happens to be at that
moment. A comment in the buff script (``GarenQ.cs:47-56``) explicitly forbids
"fixing" this into an instant hit: doing so was measured to make Q *worse* as
a last-hit tool, because the real mechanic is an empowered auto, not a nuke.

That is the wall. This module implements everything that is genuinely
independent of the auto-attack system -- the pure damage/silence/haste
formulas (tested directly against the C# below), the rank/cooldown gating,
and the buff bookkeeping for the empowerment window and the haste window,
including the ``SealSpellSlot`` recast-lock (Q.cs:84, ``GarenQ.cs:97``) and the
cooldown, which is genuinely simple: the engine's default
``CurrentCooldown = GetCooldown()`` at cast (``Spell.cs:1017-1021``) is
overwritten to 0 in the same event by Q's own ``OnSpellPostCast``
(Q.cs:85, ``spell.SetCooldown(0)``), and the real 8 s cooldown is set only when
the empowerment window ends (``GarenQ.cs:98``, hardcoded, not rank-scaled --
``GarenQ.json``'s ``Cooldown1``-``Cooldown5`` are all ``"8.0000"`` too, so the
override and the JSON agree). What this module does **not** do is deal Q's
damage or apply its silence, because that requires knowing when the caster's
next auto-attack lands -- a fact ``autoattack.py``/``step.py`` compute, not
this module. See the docstring on :func:`cast_q` for the exact hook this
needs. Until that lands, casting Q in the sim opens the window, blocks a
recast, and starts the cooldown on schedule, but the empowered swing itself
deals ordinary auto-attack damage.

One more consequence of "the buff always starts a fresh window": our sim
cannot see the early-landing case (window closes as soon as the empowered
swing connects, which can be well under 4.5 s after cast) because that also
needs the auto-attack hook above. Until it lands, Q is on cooldown for a full
4.5 s window plus 8 s here, which is a strict upper bound on the real
lockout, not the real number.

W -- Courage, from ``Characters/Garen/W.cs`` and ``Buffs/Garen/GarenW*.cs``
----------------------------------------------------------------------
Two independent effects share the slot:

* ``GarenW`` (W.cs:51-52): an active window, duration ``2 + rank - 1`` s
  (i.e. rank+1), during which a ``PreTakeDamage`` listener
  (``Buffs/Garen/GarenW.cs:54``) multiplies **every** instance of
  post-mitigation damage Garen takes by 0.7 -- minion autoattack, turret shot,
  champion hit, with no source filter. Unlike Q/E, nothing here overrides the
  engine's default cooldown-at-cast, so ``GarenW.json``'s ``Cooldown1``-
  ``Cooldown5`` (24/23/22/21/20 s) are the real, unmodified cooldown, and it
  starts **at cast**, not at some later deactivation -- the opposite of E and Q.
* ``GarenWPassive`` (``Buffs/Garen/GarenWPassive.cs:34-37``): +20% Armor, +20%
  Magic Resist, ``infiniteduration`` (i.e. permanent). Read ``W.cs:26-46``
  carefully: the listener that grants it is registered once when the *spell
  object itself* is constructed (``OnActivate(ObjAIBase, Spell)``, the
  ``ISpellScript`` lifecycle hook -- not the buff's ``OnActivate``), and it
  fires on ``OnLevelUpSpell`` the moment W's rank first becomes 1. **This is
  not tied to ever casting W.** A Garen who puts a point in W at level 3 and
  never presses W again still has the permanent mitigation from that level
  onward. Modelled here the same way: :func:`step_buffs` grants it the tick
  ``spell_level[..., Slot.W]`` first becomes >= 1, independent of
  :func:`cast_w`.

What this module does: the buff bookkeeping for both (duration, expiry,
cooldown-at-cast for the active window, permanent-and-granted-once for the
passive), plus the pure constants (``W_DAMAGE_MULT``, ``W_PASSIVE_ARMOR_PCT``,
``W_PASSIVE_MR_PCT``) and, from :func:`step_buffs`, the *per-unit multiplier
and percent-bonus values* a caller needs to actually apply these. What it
cannot do alone: multiply W's 0.7 into damage that lands via auto-attacks and
missiles (computed in ``step.py``/``autoattack.py``/``missiles.py``, not here),
or fold the passive's Armor/MR percent bonus into the ``armor``/``magic_resist``
arrays those same modules gather from ``params`` (a static per-profile-row
table with no notion of a per-unit buff). See :class:`BuffStep`'s
``damage_multiplier``, ``armor_pct_bonus`` and ``mr_pct_bonus`` fields for the
exact values to wire in, and the module-level "INTEGRATION NEEDED" note below
for where.

R -- Demacian Justice, from ``Characters/Garen/R.cs``
------------------------------------------------------
The one spell in the kit that is a genuine instant, unconditional, single-target
hit -- no buff, no windup modelled in the script, the whole thing happens
synchronously in ``OnSpellPostCast`` (R.cs:23-39)::

    percentMissingHP = [0.2857, 0.3333, 0.4][rank - 1]
    damage = 175 * rank + percentMissingHP * (MaxHP - CurrentHP)
    Target.TakeDamage(owner, damage, DAMAGE_TYPE_MAGICAL, DAMAGE_SOURCE_SPELL, false)

**The damage type is magical** (R.cs:33), which is worth stating plainly
because modern-patch League's Demacian Justice is physical: this is patch
4.20 and matches this server's actual script, not the wiki. There is no
``is Minion``/``is BaseTurret`` branch, but ``GarenR.json``'s ``TextFlags``
(``"AffectEnemies | AffectHeroes"``, no ``AffectMinions``/``AffectTurrets``/
``AffectBuildings``/``AffectNeutral``/``AffectFriends``) restrict a legal cast
to an *enemy champion* -- an engine-level ``SpellData`` targeting rule rather
than a content-script one, and outside the audit's stated scope, but real and
cited from the same JSON, and enforced here (see ``orders.apply_orders``).
Cooldown (``GarenR.json`` ``Cooldown1``-``Cooldown3``: 160/120/80 s) is,
like W's, the unmodified engine default starting at cast -- R.cs never calls
``SetCooldown``.

This is fully implementable inside this module: the raw (pre-mitigation)
damage is computed at cast from data ``apply_orders`` already has
(``state.hp``, ``state.max_hp``), mirroring E's "snapshot everything at cast"
discipline, and the mitigation step (against ``MagicResist``, not ``Armor`` --
this is a magic-damage spell) happens in :func:`step_buffs` the same way E's
own damage is mitigated against ``Armor`` there. The one real gap: this
project's ``UnitParams``/``LaneState`` split keeps ``magic_resist`` in
``params`` (gathered per-profile-row in ``step.py``), and the existing
``step.py`` call to :func:`step_buffs` does not pass it -- only ``armor`` was
ever needed before R existed. See :func:`step_buffs`'s ``magic_resist``
parameter for the exact one-line addition ``step.py`` needs; until it is
added, :func:`step_buffs` falls back to reusing ``armor`` (documented in
place, wrong whenever Armor != Magic Resist, and dead code for every existing
test because none of them cast R).

INTEGRATION NEEDED in ``step.py`` (not made here -- reported instead per the
brief): three precise, additive hooks, none of which change existing
behaviour when Garen's W/R are never cast:

1. Add ``magic_resist=P("magic_resist")`` to the existing
   ``step_buffs(...)`` call.
2. Right after that call, compute
   ``armor_eff = P("armor") * (1.0 + bs.armor_pct_bonus)`` and
   ``magic_resist_eff = P("magic_resist") * (1.0 + bs.mr_pct_bonus)``, and use
   them in place of ``P("armor")``/``P("magic_resist")`` wherever mitigation is
   computed downstream in that tick (``step_autoattack``'s ``target_resist``,
   ``step_missiles``'s ``armor``) -- this is Garen's W passive.
3. Multiply the final per-unit damage total by ``bs.damage_multiplier`` before
   ``hp = jnp.maximum(state.hp - dealt, ...)`` -- this is Garen's W active
   window.

Q's auto-attack-empowerment hook (damage + silence delivered on the next
landed swing while ``GarenQ`` is active, then the buff deactivates early) is
a fourth, structurally different ask -- it needs the auto-attack resolution
in ``autoattack.py``/``step.py`` to notice the buff and a way to signal back
"that swing was the empowered one" -- and is *not* attempted here; seeGarenQ's
section above.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .combat import post_mitigation_damage
from .state import Kind

__all__ = [
    "Slot", "BuffId", "E_RADIUS", "E_DURATION_S", "E_TICK_MS",
    "E_MINION_MULTIPLIER", "E_COOLDOWNS", "SKILL_ORDER", "RANKS_BY_LEVEL",
    "e_damage_at_rank", "cast_e",
    "E_BUFF_SLOT", "W_BUFF_SLOT", "W_PASSIVE_BUFF_SLOT", "Q_BUFF_SLOT",
    "Q_HASTE_BUFF_SLOT", "R_PENDING_BUFF_SLOT",
    "Q_BUFF_DURATION", "Q_COOLDOWN",
    "q_haste_duration_at_rank", "q_silence_duration_at_rank",
    "q_damage_at_rank", "cast_q",
    "W_DURATIONS", "W_COOLDOWNS", "W_DAMAGE_MULT", "W_PASSIVE_ARMOR_PCT",
    "W_PASSIVE_MR_PCT", "w_duration_at_rank", "cast_w",
    "R_COOLDOWNS", "R_BASE_PER_RANK", "R_MISSING_HP_FRAC", "R_CAST_RANGE",
    "r_damage_at_rank", "cast_r", "enemy_champion_index",
    "BuffStep", "step_buffs", "ranks_for_level",
]

#: ``constants.GAREN_SKILL_ORDER`` -- one entry per champion level, 0=Q 1=W 2=E 3=R.
#:
#: **THE canonical copy is `lanerl_rl/constants.py`**, which records that this
#: order previously existed in three places that disagreed
#: (``LanerlConfig.SkillOrder``, ``lanerl_bot.build``, and ``obs.AbilityBook``,
#: which encoded a fourth order again). That is not cosmetic: the action mask
#: then forbids a spell the champion HAS and offers one it does not -- and
#: casting an unlearned spell is **not** a no-op on the server, because nothing
#: in ``Spell.Cast`` checks the level.
#:
#: Mirrored here rather than imported so the sim has no import-time dependency
#: on the torch-bearing package; `test_skill_order_matches_constants` fails if
#: they drift.
SKILL_ORDER = (2, 0, 1, 2, 2, 3, 2, 2, 0, 0, 3, 0, 0, 1, 1, 3, 1, 1)


class Slot:
    """``CharData``'s ``Spell1``-``Spell4`` -- ``LeagueSandbox-Default/Stats/Garen/Garen.json``
    ``Data``: ``Spell1=GarenQ, Spell2=GarenW, Spell3=GarenE, Spell4=GarenR``."""
    Q, W, E, R = 0, 1, 2, 3


class BuffId:
    """0 is "empty slot", so a zeroed buff table holds no buffs."""
    NONE = 0
    GAREN_E = 1
    GAREN_W = 2
    GAREN_W_PASSIVE = 3
    GAREN_Q = 4
    GAREN_Q_HASTE = 5
    #: Marks a unit as having just been hit by an R cast against it; consumed
    #: and cleared by :func:`step_buffs` the very next time it runs. Not a
    #: real buff on the server -- R has none -- this is purely a same-tick
    #: mailbox so R's mitigation can happen where the other spell damage does.
    GAREN_R_PENDING = 6


#: Fixed lanes in the ``(N, MAX_BUFFS)`` buff table, chosen once here rather
#: than searched for at runtime -- see `cast_e`'s docstring for why a free-slot
#: search is not worth it for a kit this small. ``MAX_BUFFS`` is 8; six slots
#: are spoken for, two remain.
E_BUFF_SLOT = 0
W_BUFF_SLOT = 1
W_PASSIVE_BUFF_SLOT = 2
Q_BUFF_SLOT = 3
Q_HASTE_BUFF_SLOT = 4
R_PENDING_BUFF_SLOT = 5

#: ``GetUnitsInRange(Owner.Position, 330f, true)``
E_RADIUS = 330.0
#: ``AddBuff("GarenE", 3f, ...)``
E_DURATION_S = 3.0
#: ``TimeSinceLastTick >= 500.0f``
E_TICK_MS = 500.0
#: minions take three quarters
E_MINION_MULTIPLIER = 0.75
#: ``constants.GAREN_COOLDOWNS["E"]`` -- Spells/GarenE/GarenE.json
E_COOLDOWNS = (13.0, 12.0, 11.0, 10.0, 9.0)


def ranks_for_level(level: int) -> tuple:
    """Rank in each slot at a champion level, from :data:`SKILL_ORDER`.

    R is capped at 3 ranks (``Champion.LevelUpSpell``); the skill order puts a
    point in it at levels 6, 11 and 16 and never again, so the cap is not
    reached by this order anyway -- but it is applied rather than assumed.
    """
    ranks = [0, 0, 0, 0]
    for slot in SKILL_ORDER[:max(0, min(level, len(SKILL_ORDER)))]:
        cap = 3 if slot == Slot.R else 5
        ranks[slot] = min(ranks[slot] + 1, cap)
    return tuple(ranks)


#: ``(19, 4)`` lookup: ranks at each champion level, index 0 unused.
RANKS_BY_LEVEL = tuple(ranks_for_level(l) for l in range(0, 19))


def e_damage_at_rank(rank: jax.Array, attack_damage: jax.Array) -> jax.Array:
    """``10 + 12.5*(rank-1) + AD*(0.35 + 0.05*(rank-1))``, snapshotted at cast."""
    r = jnp.maximum(rank.astype(attack_damage.dtype), 1.0)
    return 10.0 + 12.5 * (r - 1.0) + attack_damage * (0.35 + 0.05 * (r - 1.0))


def cast_e(buff_id, buff_elapsed, buff_duration, buff_power,
           spell_cooldown, want_cast, rank, attack_damage, slot=E_BUFF_SLOT):
    """Start the spin for every unit whose ``want_cast`` is set and E is ready.

    Writes into buff slot ``slot``. A general free-slot search is not worth the
    gather here: Garen has exactly one buff that does anything in lane, and the
    slot is a fixed lane in the table.
    """
    ready = want_cast & (spell_cooldown[:, Slot.E] <= 0) & (rank > 0)
    dmg = e_damage_at_rank(rank, attack_damage)
    return (
        buff_id.at[:, slot].set(
            jnp.where(ready, jnp.int8(BuffId.GAREN_E), buff_id[:, slot])),
        buff_elapsed.at[:, slot].set(
            jnp.where(ready, 0.0, buff_elapsed[:, slot])),
        buff_duration.at[:, slot].set(
            jnp.where(ready, E_DURATION_S, buff_duration[:, slot])),
        buff_power.at[:, slot].set(jnp.where(ready, dmg, buff_power[:, slot])),
        ready,
    )


# --------------------------------------------------------------------- Q ---
#: ``AddBuff("GarenQ", 4.5f, ...)`` -- Characters/Garen/Q.cs:78.
Q_BUFF_DURATION = 4.5
#: ``ownerSpell.SetCooldown(8)`` -- Buffs/Garen/GarenQ.cs:98. Flat across all
#: five ranks: ``Spells/GarenQ/GarenQ.json``'s ``Cooldown1``-``Cooldown5`` are
#: all ``"8.0000"`` too, so the hardcoded override and the JSON agree -- this
#: is not an oversight where the script forgot to scale it by rank.
Q_COOLDOWN = 8.0


def q_haste_duration_at_rank(rank: jax.Array) -> jax.Array:
    """``1.5 + 0.75*(rank-1)`` -- Characters/Garen/Q.cs:77."""
    r = jnp.maximum(rank.astype(jnp.float32), 1.0)
    return 1.5 + 0.75 * (r - 1.0)


def q_silence_duration_at_rank(rank: jax.Array) -> jax.Array:
    """``1.5 + 0.25*(rank-1)`` -- Characters/Garen/Q.cs:142.

    Formula only: nothing in this module *applies* the silence, because it is
    dealt from the empowered swing landing, not from the cast -- see the
    module docstring's Q section.
    """
    r = jnp.maximum(rank.astype(jnp.float32), 1.0)
    return 1.5 + 0.25 * (r - 1.0)


def q_damage_at_rank(rank: jax.Array, attack_damage: jax.Array) -> jax.Array:
    """``30 + 25*(rank-1) + 1.4*AD`` physical -- Characters/Garen/Q.cs:150-152.

    ``DealSpellDamage`` reads ``owner.Stats.AttackDamage.Total`` when the
    empowered swing lands, not at the original Q cast -- unlike E, this is
    *not* snapshotted at cast on the server. No ``is Minion``/``is BaseTurret``
    branch anywhere in ``GarenQAttack`` (contrast E's 0.75x/0x): full damage
    to any target type, matching ``GarenQ.json``'s ``TextFlags`` including
    ``AffectMinions | AffectTurrets | AffectBuildings``.
    """
    r = jnp.maximum(rank.astype(attack_damage.dtype), 1.0)
    return 30.0 + 25.0 * (r - 1.0) + attack_damage * 1.4


def cast_q(buff_id, buff_elapsed, buff_duration, buff_power, spell_cooldown,
           want_cast, rank, slot=Q_BUFF_SLOT, haste_slot=Q_HASTE_BUFF_SLOT):
    """Open Garen's empowered-next-auto-attack window (Q.cs:76-85).

    Gated on rank, cooldown, and -- matching ``SealSpellSlot`` (Q.cs:84,
    ``GarenQ.cs:97``) -- on the window not already being open, since the real
    spell slot is locked for the cast bar the whole time ``GarenQ`` is active
    and cannot be recast regardless of what the cooldown timer shows.

    Sets the spell's cooldown to 0 immediately (Q.cs:85 overwrites the
    engine's default cast-time cooldown in the same event -- see the module
    docstring). :func:`step_buffs` is what sets the real 8 s cooldown, when
    the window closes.

    Does **not** deal Q's damage or apply its silence -- see the module
    docstring's Q section for exactly why and what is needed to add it.
    """
    already_open = buff_id[:, slot] == BuffId.GAREN_Q
    ready = (want_cast & (spell_cooldown[:, Slot.Q] <= 0) & (rank > 0)
             & ~already_open)
    haste_dur = q_haste_duration_at_rank(rank)

    buff_id = buff_id.at[:, slot].set(
        jnp.where(ready, jnp.int8(BuffId.GAREN_Q), buff_id[:, slot]))
    buff_id = buff_id.at[:, haste_slot].set(
        jnp.where(ready, jnp.int8(BuffId.GAREN_Q_HASTE), buff_id[:, haste_slot]))
    buff_elapsed = buff_elapsed.at[:, slot].set(
        jnp.where(ready, 0.0, buff_elapsed[:, slot]))
    buff_elapsed = buff_elapsed.at[:, haste_slot].set(
        jnp.where(ready, 0.0, buff_elapsed[:, haste_slot]))
    buff_duration = buff_duration.at[:, slot].set(
        jnp.where(ready, Q_BUFF_DURATION, buff_duration[:, slot]))
    buff_duration = buff_duration.at[:, haste_slot].set(
        jnp.where(ready, haste_dur, buff_duration[:, haste_slot]))
    spell_cooldown = spell_cooldown.at[:, Slot.Q].set(
        jnp.where(ready, 0.0, spell_cooldown[:, Slot.Q]))

    return buff_id, buff_elapsed, buff_duration, buff_power, spell_cooldown, ready


# --------------------------------------------------------------------- W ---
#: ``2 + spellLevel - 1`` -- Characters/Garen/W.cs:51. Rank 1..5 -> 2..6 s.
W_DURATIONS = tuple(rank + 1.0 for rank in range(1, 6))
#: ``Spells/GarenW/GarenW.json`` ``Cooldown1``-``Cooldown5``. Nothing in
#: ``W.cs``/``GarenW.cs`` calls ``SetCooldown``, so unlike Q and E this is the
#: engine's default cast-time cooldown (``Spell.cs:1017-1021``), unmodified --
#: it starts **when W is cast**, not when the active window ends.
W_COOLDOWNS = (24.0, 23.0, 22.0, 21.0, 20.0)
#: ``dmg.PostMitigationDamage *= 0.7f`` -- Buffs/Garen/GarenW.cs:54. Applies to
#: every source of incoming damage while the window is open: no attacker-type
#: or damage-type filter in ``PreTakeDamage``.
W_DAMAGE_MULT = 0.7
#: ``StatsModifier.Armor.PercentBonus += 0.2f`` -- Buffs/Garen/GarenWPassive.cs:34.
W_PASSIVE_ARMOR_PCT = 0.20
#: ``StatsModifier.MagicResist.PercentBonus += 0.2f`` -- GarenWPassive.cs:36.
W_PASSIVE_MR_PCT = 0.20


def w_duration_at_rank(rank: jax.Array) -> jax.Array:
    """``rank + 1`` seconds -- Characters/Garen/W.cs:51 (``2 + spellLevel - 1``)."""
    r = jnp.maximum(rank.astype(jnp.float32), 1.0)
    return r + 1.0


def cast_w(buff_id, buff_elapsed, buff_duration, buff_power, spell_cooldown,
           want_cast, rank, slot=W_BUFF_SLOT):
    """Open Garen's 0.7x-incoming-damage window (W.cs:49-53).

    Sets the cooldown to the rank's table value **immediately**, unlike
    :func:`cast_q` and E -- see :data:`W_COOLDOWNS`. Does not touch
    ``GarenWPassive``: that is granted on rank-up, not on cast, and is handled
    entirely in :func:`step_buffs` -- see the module docstring's W section for
    why casting W is the wrong trigger for it.
    """
    ready = want_cast & (spell_cooldown[:, Slot.W] <= 0) & (rank > 0)
    r = jnp.clip(rank.astype(jnp.int32), 1, len(W_COOLDOWNS))
    dur = r.astype(buff_duration.dtype) + 1.0
    cd_table = jnp.asarray(W_COOLDOWNS, spell_cooldown.dtype)

    buff_id = buff_id.at[:, slot].set(
        jnp.where(ready, jnp.int8(BuffId.GAREN_W), buff_id[:, slot]))
    buff_elapsed = buff_elapsed.at[:, slot].set(
        jnp.where(ready, 0.0, buff_elapsed[:, slot]))
    buff_duration = buff_duration.at[:, slot].set(
        jnp.where(ready, dur, buff_duration[:, slot]))
    spell_cooldown = spell_cooldown.at[:, Slot.W].set(
        jnp.where(ready, cd_table[r - 1], spell_cooldown[:, Slot.W]))

    return buff_id, buff_elapsed, buff_duration, buff_power, spell_cooldown, ready


# --------------------------------------------------------------------- R ---
#: ``Spells/GarenR/GarenR.json`` ``Cooldown1``-``Cooldown3`` (ranks 4-6 exist
#: in the JSON as padding but are unreachable -- R caps at 3 ranks). No
#: ``SetCooldown`` call anywhere in ``R.cs``: like W, this is the engine's
#: unmodified default, starting at cast.
R_COOLDOWNS = (160.0, 120.0, 80.0)
#: ``175f * spell.CastInfo.SpellLevel`` -- Characters/Garen/R.cs:29.
R_BASE_PER_RANK = 175.0
#: ``new[] { 0.2857f, 0.3333f, 0.4f }`` -- Characters/Garen/R.cs:28. Indexed by
#: ``rank - 1``, exactly as the C# array is.
R_MISSING_HP_FRAC = (0.2857, 0.3333, 0.4)
#: ``Spells/GarenR/GarenR.json`` ``"CastRange": "400.0000"``. Engine-level
#: ``SpellData`` targeting, not a content-script rule, so strictly outside the
#: audit's stated scope -- but real, and the same JSON that gives the cooldown
#: table above, so it is applied here rather than left as a silent gap.
R_CAST_RANGE = 400.0


def r_damage_at_rank(rank: jax.Array, missing_hp: jax.Array) -> jax.Array:
    """``175*rank + missingHpFrac[rank]*missingHP`` -- R.cs:28-29.

    Pre-mitigation. The server applies this as ``DAMAGE_TYPE_MAGICAL``
    (R.cs:33) -- worth stating plainly, because modern-patch Demacian Justice
    is physical and this server (patch 4.20) is not modern-patch League.
    :func:`step_buffs` mitigates it against Magic Resist, not Armor.
    """
    r = jnp.clip(rank.astype(jnp.int32), 1, len(R_COOLDOWNS))
    frac = jnp.asarray(R_MISSING_HP_FRAC, missing_hp.dtype)[r - 1]
    return R_BASE_PER_RANK * r.astype(missing_hp.dtype) + frac * jnp.maximum(
        missing_hp, 0.0)


def enemy_champion_index(n: int) -> jax.Array:
    """The only enemy champion in a fixed 2-champion lane: unit 0 <-> unit 1.

    Shared by :func:`cast_r` (who gets hit) and ``orders.apply_orders`` (is
    this cast's target even legal) so the fact "there is exactly one possible
    R target and it's the other champion slot" is not encoded twice and does
    not get to drift the way ``GAREN_SKILL_ORDER`` once did across three
    copies.
    """
    idx = jnp.arange(n)
    return jnp.where(idx == 0, jnp.int8(1),
                     jnp.where(idx == 1, jnp.int8(0), jnp.int8(-1)))


def cast_r(buff_id, buff_elapsed, buff_duration, buff_power, spell_cooldown,
           want_cast, rank, hp, max_hp, target, slot=R_PENDING_BUFF_SLOT):
    """Snapshot Demacian Justice's damage at cast and mark the target for it.

    Everything the server's synchronous ``OnSpellPostCast`` (R.cs:23-39) reads
    -- rank and the target's current/max HP -- is available where champion
    orders are decoded, so (mirroring E's "snapshot at cast" discipline) the
    raw pre-mitigation damage is computed here rather than deferred. Only the
    mitigation step (Magic Resist) waits for :func:`step_buffs`, the same way
    E's damage is computed here-ish and mitigated there.

    The pending hit is written onto the **target's** buff row (not the
    caster's) via :data:`enemy_champion_index`, not through a data-dependent
    scatter on ``target`` -- with exactly two champion slots, "the other
    champion" is a fixed permutation, and scattering through an
    attacker-chosen index would risk one champion's not-casting no-op write
    landing on the same destination row as the other's real write in the same
    ``.at[].set()`` call, with JAX's duplicate-index tie-break deciding which
    one survives.

    ``target`` is expected to already be validated by the caller (an enemy
    champion, in range) -- see ``orders.apply_orders``. Re-checked against
    :func:`enemy_champion_index` here anyway, so a caller that skips that gate
    fails closed (no cast) instead of open.
    """
    n = buff_id.shape[0]
    mirror = enemy_champion_index(n)
    valid_target = (target == mirror) & (mirror >= 0)
    ready = (want_cast & (spell_cooldown[:, Slot.R] <= 0) & (rank > 0)
             & (target >= 0) & valid_target)

    r = jnp.clip(rank.astype(jnp.int32), 1, len(R_COOLDOWNS))
    cd_table = jnp.asarray(R_COOLDOWNS, spell_cooldown.dtype)

    mirror_idx = jnp.clip(mirror, 0, n - 1).astype(jnp.int32)
    missing_hp = jnp.maximum(max_hp - hp, 0.0)          # each unit's own
    target_missing_hp = missing_hp[mirror_idx]          # its mirror's missing HP
    raw_by_caster = r_damage_at_rank(rank, target_missing_hp)

    hits_me = ready[mirror_idx]
    dmg_to_me = raw_by_caster[mirror_idx]

    buff_id = buff_id.at[:, slot].set(
        jnp.where(hits_me, jnp.int8(BuffId.GAREN_R_PENDING), buff_id[:, slot]))
    buff_elapsed = buff_elapsed.at[:, slot].set(
        jnp.where(hits_me, 0.0, buff_elapsed[:, slot]))
    buff_duration = buff_duration.at[:, slot].set(
        jnp.where(hits_me, 0.0, buff_duration[:, slot]))
    buff_power = buff_power.at[:, slot].set(
        jnp.where(hits_me, dmg_to_me, buff_power[:, slot]))

    spell_cooldown = spell_cooldown.at[:, Slot.R].set(
        jnp.where(ready, cd_table[r - 1], spell_cooldown[:, Slot.R]))

    return buff_id, buff_elapsed, buff_duration, buff_power, spell_cooldown, ready


class BuffStep(NamedTuple):
    buff_id: jax.Array
    buff_elapsed: jax.Array
    spell_cooldown: jax.Array
    damage_dealt: jax.Array     # (N,) post-mitigation damage received this tick
    dealt_by: jax.Array         # (N,) who dealt it, -1 if nobody
    suppress_attack: jax.Array  # (N,) bool: CanAttack cleared
    ghosted: jax.Array          # (N,) bool
    #: Multiply a unit's total incoming damage by this before subtracting HP.
    #: 1.0 normally, :data:`W_DAMAGE_MULT` while Garen's W window is open.
    #: NOT applied anywhere yet -- see the module docstring's INTEGRATION
    #: NEEDED note, hook 3.
    damage_multiplier: jax.Array
    #: Additive fraction to fold into ``Armor.Total`` before mitigation
    #: (``armor * (1 + this)``). 0.0 normally, :data:`W_PASSIVE_ARMOR_PCT`
    #: once ``GarenWPassive`` is granted. Hook 2.
    armor_pct_bonus: jax.Array
    #: Same, for ``MagicResist.Total``. Hook 2.
    mr_pct_bonus: jax.Array


def step_buffs(*, buff_id, buff_elapsed, buff_duration, buff_power,
               spell_cooldown, spell_level, x, y, kind, team, alive, armor,
               magic_resist=None, delta_ms: float = 1000.0 / 60.0,
               e_slot: int = E_BUFF_SLOT, w_slot: int = W_BUFF_SLOT,
               wp_slot: int = W_PASSIVE_BUFF_SLOT, q_slot: int = Q_BUFF_SLOT,
               qh_slot: int = Q_HASTE_BUFF_SLOT,
               r_slot: int = R_PENDING_BUFF_SLOT) -> BuffStep:
    """Advance every Garen buff one tick: E's spin, W's window and passive,
    Q's empowerment/haste windows, and R's one-shot pending hit.

    The tick boundary follows ``Buff.Update`` for all of them: elapsed
    advances first, then a buff's effect for this tick is evaluated, then it
    deactivates once ``TimeElapsed >= Duration``.

    ``magic_resist`` is optional and falls back to reusing ``armor`` if not
    given -- WRONG whenever Magic Resist != Armor, but harmless today because
    no existing caller casts R (the only consumer of this value): the
    fallback is dead code until ``step.py``'s call site is updated to pass
    ``magic_resist=P("magic_resist")``. Kept optional specifically so adding
    R here does not break every other test in this file, which do not go
    through ``apply_orders``'s R path at all.
    """
    n = x.shape[0]
    dt_s = delta_ms / 1000.0

    # ---- E: the spin (unchanged from before Q/W/R existed) ---------------
    e_active = (buff_id[:, e_slot] == BuffId.GAREN_E) & alive
    e_elapsed = jnp.where(e_active, buff_elapsed[:, e_slot] + dt_s,
                          buff_elapsed[:, e_slot])

    # `TimeSinceLastTick >= 500` -- a tick every 500 ms of buff life
    before = jnp.floor(buff_elapsed[:, e_slot] * 1000.0 / E_TICK_MS)
    after = jnp.floor(e_elapsed * 1000.0 / E_TICK_MS)
    fires = e_active & (after > before)

    d2 = (x[None, :] - x[:, None]) ** 2 + (y[None, :] - y[:, None]) ** 2
    hittable = alive & (kind != Kind.TURRET) & (kind != Kind.NONE)
    hit = (
        fires[:, None] & hittable[None, :]
        & (team[None, :] != team[:, None])
        & (d2 <= E_RADIUS * E_RADIUS)
    )
    mult = jnp.where(kind == Kind.LANE_MINION, E_MINION_MULTIPLIER, 1.0)
    raw = buff_power[:, e_slot][:, None] * mult[None, :]
    dealt = jnp.where(hit, post_mitigation_damage(raw, armor[None, :], jnp),
                      jnp.zeros_like(d2))
    damage_e = dealt.sum(axis=0)
    any_hit = jnp.any(hit, axis=0)
    dealt_by_e = jnp.where(any_hit, jnp.argmax(hit, axis=0), -1).astype(jnp.int8)

    e_expired = e_active & (e_elapsed >= buff_duration[:, e_slot])
    e_rank = jnp.clip(spell_level[:, Slot.E].astype(jnp.int32), 1,
                      len(E_COOLDOWNS)) - 1
    e_cd_table = jnp.asarray(E_COOLDOWNS, spell_cooldown.dtype)

    # ---- W: the 0.7x window ------------------------------------------------
    w_active = (buff_id[:, w_slot] == BuffId.GAREN_W) & alive
    w_elapsed = jnp.where(w_active, buff_elapsed[:, w_slot] + dt_s,
                          buff_elapsed[:, w_slot])
    w_expired = w_active & (w_elapsed >= buff_duration[:, w_slot])
    w_active_now = w_active & ~w_expired
    damage_multiplier = jnp.where(
        w_active_now, jnp.asarray(W_DAMAGE_MULT, x.dtype), jnp.ones_like(x))

    # ---- GarenWPassive: granted on RANK-UP, not on cast --------------------
    # See the module docstring's W section: `OnLevelUpSpell` is registered from
    # spell-object construction, independent of ever pressing W, and fires
    # once when SpellLevel first becomes 1. Reproduced the same way: as soon
    # as `spell_level[..., Slot.W] >= 1` and the passive isn't already marked,
    # grant it; once granted (`infiniteduration`) nothing here ever clears it.
    has_wp = buff_id[:, wp_slot] == BuffId.GAREN_W_PASSIVE
    grant_wp = alive & (spell_level[:, Slot.W] >= 1) & ~has_wp
    buff_id_wp = jnp.where(grant_wp, jnp.int8(BuffId.GAREN_W_PASSIVE),
                           buff_id[:, wp_slot])
    wp_active_now = alive & (buff_id_wp == BuffId.GAREN_W_PASSIVE)
    armor_pct_bonus = jnp.where(
        wp_active_now, jnp.asarray(W_PASSIVE_ARMOR_PCT, x.dtype),
        jnp.zeros_like(x))
    mr_pct_bonus = jnp.where(
        wp_active_now, jnp.asarray(W_PASSIVE_MR_PCT, x.dtype),
        jnp.zeros_like(x))

    # ---- Q: empowerment window + haste window ------------------------------
    # Neither deals damage or applies silence here -- see the module
    # docstring's Q section for why that needs the auto-attack system.
    q_active = (buff_id[:, q_slot] == BuffId.GAREN_Q) & alive
    q_elapsed = jnp.where(q_active, buff_elapsed[:, q_slot] + dt_s,
                          buff_elapsed[:, q_slot])
    q_expired = q_active & (q_elapsed >= buff_duration[:, q_slot])

    qh_active = (buff_id[:, qh_slot] == BuffId.GAREN_Q_HASTE) & alive
    qh_elapsed = jnp.where(qh_active, buff_elapsed[:, qh_slot] + dt_s,
                           buff_elapsed[:, qh_slot])
    qh_expired = qh_active & (qh_elapsed >= buff_duration[:, qh_slot])

    # ---- R: one-shot pending hit --------------------------------------------
    mr = armor if magic_resist is None else magic_resist
    r_active = (buff_id[:, r_slot] == BuffId.GAREN_R_PENDING) & alive
    # Fires on the tick that sees the INCOMING (pre-advance) elapsed still at
    # 0 -- i.e. the first step_buffs call after cast_r wrote it -- and then
    # expires unconditionally this same tick (duration was set to 0 at cast),
    # matching R.cs having no windup or duration of its own: cast and hit are
    # the same instant.
    r_fires = r_active & (buff_elapsed[:, r_slot] == 0.0)
    raw_r = buff_power[:, r_slot]
    damage_r = jnp.where(r_fires, post_mitigation_damage(raw_r, mr, jnp),
                         jnp.zeros_like(raw_r))
    mirror = enemy_champion_index(n)
    dealt_by_r = jnp.where(r_fires, mirror, -1).astype(jnp.int8)
    r_elapsed = jnp.where(r_active, buff_elapsed[:, r_slot] + dt_s,
                          buff_elapsed[:, r_slot])
    r_expired = r_active

    # ---- combine E's and R's directly-dealt damage for tick()'s kill
    # attribution cumsum. A victim can in principle take both in the same
    # tick (E's spin from a nearby enemy AND an R landing, from either
    # champion) -- rare, since R only ever targets the enemy champion, but
    # not impossible. Summed for HP; R's attacker wins the attribution tie,
    # an arbitrary but documented choice for a genuinely rare double-source
    # tick, rather than the two-row plumbing a fully general fix would need.
    damage_dealt = damage_e + damage_r
    dealt_by = jnp.where(damage_r > 0, dealt_by_r, dealt_by_e)

    # ---- cooldowns: generic per-tick decay on ALL FOUR spell slots, with
    # E and Q overriding their column to the rank's/flat table value on the
    # tick their buff expires. Before Q/W/R existed only E's column was ever
    # touched here; W's and R's cooldowns are set once at cast (`cast_w`,
    # `cast_r`) and, without this, would never count back down.
    decayed_cd = jnp.maximum(spell_cooldown - dt_s, 0.0)
    new_cd = decayed_cd.at[:, Slot.E].set(
        jnp.where(e_expired, e_cd_table[e_rank], decayed_cd[:, Slot.E]))
    new_cd = new_cd.at[:, Slot.Q].set(
        jnp.where(q_expired, Q_COOLDOWN, decayed_cd[:, Slot.Q]))

    buff_id_out = buff_id
    buff_id_out = buff_id_out.at[:, e_slot].set(
        jnp.where(e_expired, jnp.int8(BuffId.NONE), buff_id[:, e_slot]))
    buff_id_out = buff_id_out.at[:, w_slot].set(
        jnp.where(w_expired, jnp.int8(BuffId.NONE), buff_id[:, w_slot]))
    buff_id_out = buff_id_out.at[:, wp_slot].set(buff_id_wp)
    buff_id_out = buff_id_out.at[:, q_slot].set(
        jnp.where(q_expired, jnp.int8(BuffId.NONE), buff_id[:, q_slot]))
    buff_id_out = buff_id_out.at[:, qh_slot].set(
        jnp.where(qh_expired, jnp.int8(BuffId.NONE), buff_id[:, qh_slot]))
    buff_id_out = buff_id_out.at[:, r_slot].set(
        jnp.where(r_expired, jnp.int8(BuffId.NONE), buff_id[:, r_slot]))

    buff_elapsed_out = buff_elapsed
    buff_elapsed_out = buff_elapsed_out.at[:, e_slot].set(
        jnp.where(e_expired, 0.0, e_elapsed))
    buff_elapsed_out = buff_elapsed_out.at[:, w_slot].set(
        jnp.where(w_expired, 0.0, w_elapsed))
    buff_elapsed_out = buff_elapsed_out.at[:, q_slot].set(
        jnp.where(q_expired, 0.0, q_elapsed))
    buff_elapsed_out = buff_elapsed_out.at[:, qh_slot].set(
        jnp.where(qh_expired, 0.0, qh_elapsed))
    buff_elapsed_out = buff_elapsed_out.at[:, r_slot].set(
        jnp.where(r_expired, 0.0, r_elapsed))
    # wp_slot's elapsed/duration are left untouched: `infiniteduration` means
    # there is no countdown to track.

    return BuffStep(
        buff_id=buff_id_out,
        buff_elapsed=buff_elapsed_out,
        spell_cooldown=new_cd,
        damage_dealt=damage_dealt.astype(x.dtype),
        dealt_by=dealt_by,
        # `SetStatus(CanAttack, false)` and `SetStatus(Ghosted, true)` for the
        # duration -- E suppresses autos and passes through collision.
        suppress_attack=e_active & ~e_expired,
        ghosted=e_active & ~e_expired,
        damage_multiplier=damage_multiplier,
        armor_pct_bonus=armor_pct_bonus,
        mr_pct_bonus=mr_pct_bonus,
    )
