"""Garen's kit: Q (Decisive Strike), W (Courage), E (Judgment), R (Demacian Justice).

Why E first
-----------
``constants.GAREN_SKILL_ORDER`` takes it at level 1 and maxes it first, with the
reason stated: *"E first: it is the farming and trading spell."* A Garen with no
abilities cannot clear a wave the way the policy will be trained to, so this is
the first one that changes what the agent can do rather than how accurately it
does it. E was built and tested first for that reason; Q, W and R follow the
same pattern (rank tables, cooldowns, cast gating, buff slots), with their
combat effects integrated by ``step.py``.

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

The autoattack integration is implemented in ``step.py``: it consumes the
one skipped swing, replaces the following hit's damage, applies the ranked
silence to that hit's target, closes the empowerment window, and begins the
cooldown. This module supplies the formulas and buff bookkeeping, including
the ``SealSpellSlot`` recast-lock (Q.cs:84, ``GarenQ.cs:97``). The engine's default
``CurrentCooldown = GetCooldown()`` at cast (``Spell.cs:1017-1021``) is
overwritten to 0 in the same event by Q's own ``OnSpellPostCast``
(Q.cs:85, ``spell.SetCooldown(0)``), and the real 8 s cooldown is set only when
the empowerment window ends (``GarenQ.cs:98``, hardcoded, not rank-scaled --
``GarenQ.json``'s ``Cooldown1``-``Cooldown5`` are all ``"8.0000"`` too, so the
override and the JSON agree). If no empowered hit lands, expiry starts that
same cooldown after the full 4.5-second window.

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
* ``GarenWPassive`` (``Buffs/Garen/GarenWPassive.cs:34-37``): NOT a clean +20%
  to either stat -- ``StatsModifier.Armor.PercentBonus += 0.2f;
  Armor.PercentBaseBonus -= 0.2f;`` (mirrored for ``MagicResist``). Against
  ``Stat.cs:68``'s ``Total = ((BaseValue+BaseBonus)*(1+PercentBaseBonus) +
  FlatBonus)*(1+PercentBonus)``, writing ``B = BaseValue+BaseBonus`` and
  ``F = FlatBonus``, this composes to ``0.96*B + 1.2*F`` -- a **4% DECREASE**
  when ``F=0`` (Garen's MagicResist: no item/rune MR source in this project),
  and for Armor (``F = RUNE_ARMOR_BONUS = 9.0``, the only nonzero flat
  component either stat has here -- confirmed a rune lands in ``FlatBonus``,
  not ``BaseValue``/``BaseBonus``: ``Champion.OnAdded`` applies a rune page
  entry as an item, and ``ItemData : StatsModifier`` with
  ``Armor.FlatBonus = file.GetFloat("Data", "FlatArmorMod")``,
  ``ItemData.cs:70``) a small, net-POSITIVE-only-while ``F > 0.2*B`` result
  that inverts to net-negative around level 11 as base Armor outgrows the
  fixed rune term. Read ``W.cs:26-46`` carefully: the listener that grants it
  is registered once when the *spell object itself* is constructed
  (``OnActivate(ObjAIBase, Spell)``, the ``ISpellScript`` lifecycle hook --
  not the buff's ``OnActivate``), and it fires on ``OnLevelUpSpell`` the
  moment W's rank first becomes 1. **This is not tied to ever casting W.** A
  Garen who puts a point in W at level 3 and never presses W again still has
  this (real, not-flat-+20%) mitigation change from that level onward.
  Modelled here the same way: :func:`step_buffs` grants it the tick
  ``spell_level[..., Slot.W]`` first becomes >= 1, independent of
  :func:`cast_w`, and reports the RAW ``PercentBaseBonus``/``PercentBonus``
  pair (``-0.2``/``+0.2`` while granted, ``0``/``0`` otherwise) rather than a
  single pre-composed multiplier, so the caller can run the real
  ``combat.stat_total`` formula instead of a flat ``*1.2``.

What this module does: the buff bookkeeping for both (duration, expiry,
cooldown-at-cast for the active window, permanent-and-granted-once for the
passive), plus the pure constants (``W_DAMAGE_MULT``, ``W_PASSIVE_ARMOR_PCT``,
``W_PASSIVE_MR_PCT``) and, from :func:`step_buffs`, the *per-unit* values a
caller needs to actually apply these -- ``BuffStep.damage_multiplier``
(unconditionally 1.0, per the active-window bug-compat note above) and the
``armor_percent_base_bonus``/``armor_percent_bonus``/``mr_percent_base_bonus``/
``mr_percent_bonus`` quartet (the passive). Wired into ``step.py``: it passes
``magic_resist=P("magic_resist")`` into :func:`step_buffs`, then computes
``armor_eff``/``magic_resist_eff`` via ``combat.stat_total`` (not a flat
``* (1 + pct)``, which is exactly the bug this section replaced) and uses
those wherever mitigation is computed downstream that tick
(``step_autoattack``'s ``target_resist``, ``step_missiles``'s ``armor``).
``damage_multiplier`` is still multiplied into the final per-unit damage
total in ``step.py``, unconditionally a no-op now that it is always 1.0 --
left in place rather than removed so a future, different W fix does not need
to re-thread the multiply site.

R -- Demacian Justice, from ``Characters/Garen/R.cs``
------------------------------------------------------
R is a single-target spell whose script damage happens synchronously in
``OnSpellPostCast`` (R.cs:23-39), after the engine's 0.435-second cast time::

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
Cooldown (``GarenR.json`` ``Cooldown1``-``Cooldown3``: 160/120/80 s) is the
unmodified engine default and begins when casting finishes. Target health is
read at that finish event, exactly when ``OnSpellPostCast`` runs, and damage
is mitigated against Magic Resist.

All three hooks this section used to ask ``step.py`` to add are now wired
there (``magic_resist=P("magic_resist")`` into :func:`step_buffs`;
``armor_eff``/``magic_resist_eff`` via ``combat.stat_total`` right after; the
``damage_multiplier`` multiply before ``hp = jnp.maximum(state.hp - dealt,
...)``) -- left named here rather than deleted so the next person can find
where each one lives without re-reading ``step.py`` end to end.

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
    "Q_HASTE_MULTIPLIER", "q_haste_duration_at_rank", "q_silence_duration_at_rank",
    "q_damage_at_rank", "cast_q", "consume_q_on_hit",
    "W_DURATIONS", "W_COOLDOWNS", "W_DAMAGE_MULT", "W_PASSIVE_ARMOR_PCT",
    "W_PASSIVE_MR_PCT", "w_duration_at_rank", "cast_w",
    "R_COOLDOWNS", "R_BASE_PER_RANK", "R_MISSING_HP_FRAC", "R_CAST_RANGE",
    "R_CAST_TIME_S",
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

# `Buffs/Garen/GarenQHaste.cs:34`: `MoveSpeed.PercentBonus += 0.35f`.
Q_HASTE_MULTIPLIER = 1.35

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
    # `already_open` is NOT optional, and leaving it out gave the RL agent a
    # permanent damage aura.
    #
    # E is the one Garen spell whose cooldown starts when the spin ENDS, not
    # when it is cast (`orders.py`: "cast_e never touches cooldown; step_buffs
    # does"). So mid-spin `spell_cooldown[E]` is 0, and without this guard a
    # re-cast passed `ready` and reset `buff_elapsed` to 0.0 below. A policy
    # casting E every decision therefore held the spin at elapsed 0.0 for the
    # whole episode: it never reached `E_DURATION_S`, never expired, never
    # started its cooldown, and kept dealing its 500 ms periodic damage.
    #
    # Measured, on the RL-006 checkpoint: E cast on 80-83% of decisions and the
    # cooldown never rose ONCE in 300 s of game time, against 44 spins in the
    # C# server over the same window under the same orders. `Spell.Cast` there
    # is gated by `champ.CanCast(sp)`, which refuses a cast while the spell is
    # active -- `LanerlControl`'s cast handler returns on it silently, because
    # a refused cast is ordinary gameplay.
    #
    # `cast_q` has always had this guard, in this exact shape, for the same
    # reason (Q's cooldown starts after its empowerment window). `cast_w` and
    # `cast_r` do not need it: both write their cooldown at cast time, so the
    # `spell_cooldown <= 0` term already refuses a re-cast.
    #
    # The observation builder ALREADY modelled the correct rule -- its
    # `cast_locked` term reports E unavailable while `buff_id[E] ==
    # BuffId.GAREN_E`, "unavailable while active despite a zero countdown". So
    # the sim was telling the policy E was unavailable and then casting it
    # anyway, which is the worst of both: the feature could not explain the
    # reward, and the reward taught the policy to press the button regardless.
    already_open = buff_id[:, slot] == BuffId.GAREN_E
    ready = (want_cast & (spell_cooldown[:, Slot.E] <= 0) & (rank > 0)
             & ~already_open)
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
    # `GarenQ.OnActivate` calls `SkipNextAutoAttack()` after cancelling the
    # current swing. The fixed Q buff lane carries that one-bit state until
    # `step_autoattack` consumes it at the next swing gate.
    buff_power = buff_power.at[:, slot].set(
        jnp.where(ready, jnp.ones_like(buff_power[:, slot]), buff_power[:, slot]))

    return buff_id, buff_elapsed, buff_duration, buff_power, spell_cooldown, ready


def consume_q_on_hit(buff_id, spell_cooldown, q_landed, slot=Q_BUFF_SLOT):
    """End Q and begin its fixed cooldown when ``GarenQAttack`` lands.

    `GarenQAttack.OnSpellPostCast` calls `OnSpellEnd`, which deactivates the
    still-live `GarenQ` buff. Its `OnDeactivate` restores the ordinary attack
    spell and sets cooldown 8 immediately; this is earlier than the natural
    4.5-second expiry in the usual successful-hit case.
    """
    buff_id = buff_id.at[:, slot].set(
        jnp.where(q_landed, jnp.int8(BuffId.NONE), buff_id[:, slot]))
    spell_cooldown = spell_cooldown.at[:, Slot.Q].set(
        jnp.where(q_landed, jnp.asarray(Q_COOLDOWN, spell_cooldown.dtype),
                  spell_cooldown[:, Slot.Q]))
    return buff_id, spell_cooldown


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
# `SpellData.GetCastTime() = (1 + DelayCastOffsetPercent) * 0.5`; GarenR's
# data has `DelayCastOffsetPercent = -0.13` and lacks `InstantCast`, so its
# script's post-cast damage lands after this real engine windup.
R_CAST_TIME_S = 0.435


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
    """Start R's windup and mark its target for a delayed hit.

    `GarenR.OnSpellPostCast` reads the target's health, not
    `OnSpellPreCast`. The pending lane therefore stores the caster's rank --
    not a precomputed damage snapshot -- and :func:`step_buffs` reads current
    HP when the 0.435-second engine cast timer completes. ``hp``/``max_hp``
    remain accepted for source-compatible callers but are intentionally not
    read here.

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
    mirror_idx = jnp.clip(mirror, 0, n - 1).astype(jnp.int32)
    valid_target = (target == mirror) & (mirror >= 0)
    already_casting = buff_id[mirror_idx, slot] == BuffId.GAREN_R_PENDING
    ready = (want_cast & (spell_cooldown[:, Slot.R] <= 0) & (rank > 0)
             & (target >= 0) & valid_target & ~already_casting)

    hits_me = ready[mirror_idx]
    rank_to_me = rank[mirror_idx]

    buff_id = buff_id.at[:, slot].set(
        jnp.where(hits_me, jnp.int8(BuffId.GAREN_R_PENDING), buff_id[:, slot]))
    buff_elapsed = buff_elapsed.at[:, slot].set(
        jnp.where(hits_me, 0.0, buff_elapsed[:, slot]))
    buff_duration = buff_duration.at[:, slot].set(
        jnp.where(hits_me, R_CAST_TIME_S, buff_duration[:, slot]))
    buff_power = buff_power.at[:, slot].set(
        jnp.where(hits_me, rank_to_me.astype(buff_power.dtype),
                  buff_power[:, slot]))

    return buff_id, buff_elapsed, buff_duration, buff_power, spell_cooldown, ready


class BuffStep(NamedTuple):
    buff_id: jax.Array
    buff_elapsed: jax.Array
    buff_power: jax.Array
    spell_cooldown: jax.Array
    damage_dealt: jax.Array     # (N,) post-mitigation damage received this tick
    dealt_by: jax.Array         # (N,) who dealt it, -1 if nobody
    suppress_attack: jax.Array  # (N,) bool: CanAttack cleared
    ghosted: jax.Array          # (N,) bool
    #: Multiply a unit's total incoming damage by this before subtracting HP.
    #: Unconditionally **1.0** -- see :func:`step_buffs`'s W-active section for
    #: why: on this server build, ``GarenW``'s 0.7x never reaches real HP loss
    #: (a stale-local-vs-mutated-field bug in ``AttackableUnit.TakeDamage``),
    #: so bug-compatibility means this field does nothing, not that it holds
    #: :data:`W_DAMAGE_MULT`. Kept as a field (rather than deleted) so a
    #: caller does not need special-casing, and so the window's own
    #: (unaffected) duration/cooldown tracking has an obvious place to grow a
    #: real consumer later (e.g. an observation) without re-plumbing.
    damage_multiplier: jax.Array
    #: ``Stat.Armor.PercentBaseBonus`` contribution from ``GarenWPassive``:
    #: ``-0.2`` once granted, ``0`` otherwise (``GarenWPassive.cs:34``). See
    #: :func:`step_buffs`'s W-passive section for why this is reported
    #: separately from ``armor_percent_bonus`` rather than pre-composed into
    #: one multiplier -- they compose around ``FlatBonus`` differently.
    armor_percent_base_bonus: jax.Array
    #: ``Stat.Armor.PercentBonus`` contribution: ``+0.2`` once granted, ``0``
    #: otherwise (``GarenWPassive.cs:34``).
    armor_percent_bonus: jax.Array
    #: Same pair, for ``MagicResist`` (``GarenWPassive.cs:36``).
    mr_percent_base_bonus: jax.Array
    mr_percent_bonus: jax.Array
    #: Q remains a live spell-swapped auto after its one skipped swing.
    q_empowered: jax.Array
    #: The one-bit `SkipNextAutoAttack` marker carried in Q's buff-power lane.
    q_skip_next: jax.Array


def step_buffs(*, buff_id, buff_elapsed, buff_duration, buff_power,
               spell_cooldown, spell_level, x, y, kind, team, alive, armor,
               magic_resist=None, hp=None, max_hp=None,
               delta_ms: float = 1000.0 / 60.0,
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

    # `GarenE.TimeSinceLastTick` is constructed at 500 ms, so its first
    # `OnUpdate(diff)` fires immediately; only later ticks wait another 500.
    # `buff_elapsed == 0` is the fixed-shape equivalent of that private
    # script-field initial condition.
    before = jnp.floor(buff_elapsed[:, e_slot] * 1000.0 / E_TICK_MS)
    after = jnp.floor(e_elapsed * 1000.0 / E_TICK_MS)
    first_update = (buff_elapsed[:, e_slot] == 0.0) & (e_elapsed > 0.0)
    fires = e_active & (first_update | (after > before))

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

    # ---- W active: the 0.7x window (real buff, but a no-op on this server) --
    # `GarenW.cs:47-55`'s `PreTakeDamage` listener does
    # `dmg.PostMitigationDamage *= 0.7f`, but `AttackableUnit.TakeDamage`
    # (`AttackableUnit.cs:551,558,585,606,612-616`) already copied
    # `PostMitigationDamage` into a stale LOCAL float BEFORE publishing
    # `OnPreTakeDamage` -- both the real HP subtraction (`:585`) and lifesteal
    # (`:612-616`) read that stale local, never the mutated `damageData`
    # field. Only the cosmetic floating-damage-number packet (`:606`) ever
    # sees the 0.7x. Verified directly from both files, not inferred from the
    # bug's plausibility. So on THIS server build, W's active window changes
    # NOTHING about real damage taken -- reproduced as bug-compatibility, not
    # "fixed" to the intended mechanic our sim used to implement. The window
    # itself (duration, cooldown-at-cast) is still real and tracked
    # (`w_active_now`) since the buff genuinely exists and genuinely expires
    # on schedule; only `damage_multiplier` is unconditionally 1.0.
    w_active = (buff_id[:, w_slot] == BuffId.GAREN_W) & alive
    w_elapsed = jnp.where(w_active, buff_elapsed[:, w_slot] + dt_s,
                          buff_elapsed[:, w_slot])
    w_expired = w_active & (w_elapsed >= buff_duration[:, w_slot])
    w_active_now = w_active & ~w_expired
    damage_multiplier = jnp.ones_like(x)

    # ---- GarenWPassive: granted on RANK-UP, not on cast --------------------
    # See the module docstring's W section: `OnLevelUpSpell` is registered from
    # spell-object construction, independent of ever pressing W, and fires
    # once when SpellLevel first becomes 1. Reproduced the same way: as soon
    # as `spell_level[..., Slot.W] >= 1` and the passive isn't already marked,
    # grant it; once granted (`infiniteduration`) nothing here ever clears it.
    #
    # Reports the RAW `PercentBaseBonus`/`PercentBonus` pair the server writes
    # (`-0.2`/`+0.2`, `GarenWPassive.cs:34-37`) rather than a single combined
    # multiplier: `Stat.Total`'s `FlatBonus` term sits BETWEEN these two
    # percent terms (`combat.stat_total`), so a caller with a nonzero
    # `FlatBonus` (Garen's Armor, via the rune page) needs both terms
    # separately to compose the formula correctly -- collapsing them into one
    # multiplier here would silently re-introduce the flat `*1.2` bug this
    # replaces.
    has_wp = buff_id[:, wp_slot] == BuffId.GAREN_W_PASSIVE
    grant_wp = alive & (spell_level[:, Slot.W] >= 1) & ~has_wp
    buff_id_wp = jnp.where(grant_wp, jnp.int8(BuffId.GAREN_W_PASSIVE),
                           buff_id[:, wp_slot])
    wp_active_now = alive & (buff_id_wp == BuffId.GAREN_W_PASSIVE)
    armor_percent_base_bonus = jnp.where(
        wp_active_now, jnp.asarray(-W_PASSIVE_ARMOR_PCT, x.dtype),
        jnp.zeros_like(x))
    armor_percent_bonus = jnp.where(
        wp_active_now, jnp.asarray(W_PASSIVE_ARMOR_PCT, x.dtype),
        jnp.zeros_like(x))
    mr_percent_base_bonus = jnp.where(
        wp_active_now, jnp.asarray(-W_PASSIVE_MR_PCT, x.dtype),
        jnp.zeros_like(x))
    mr_percent_bonus = jnp.where(
        wp_active_now, jnp.asarray(W_PASSIVE_MR_PCT, x.dtype),
        jnp.zeros_like(x))

    # ---- Q: empowerment window + haste window ------------------------------
    q_active = (buff_id[:, q_slot] == BuffId.GAREN_Q) & alive
    q_elapsed = jnp.where(q_active, buff_elapsed[:, q_slot] + dt_s,
                          buff_elapsed[:, q_slot])
    q_expired = q_active & (q_elapsed >= buff_duration[:, q_slot])
    q_live = q_active & ~q_expired
    q_skip_next = q_live & (buff_power[:, q_slot] > 0)
    q_empowered = q_live & ~q_skip_next

    qh_active = (buff_id[:, qh_slot] == BuffId.GAREN_Q_HASTE) & alive
    qh_elapsed = jnp.where(qh_active, buff_elapsed[:, qh_slot] + dt_s,
                           buff_elapsed[:, qh_slot])
    qh_expired = qh_active & (qh_elapsed >= buff_duration[:, qh_slot])

    # ---- R: one-shot pending hit --------------------------------------------
    mr = armor if magic_resist is None else magic_resist
    r_pending = buff_id[:, r_slot] == BuffId.GAREN_R_PENDING
    # `GarenR` has `CantCancelWhileWindingUp=1`: a target dying during the
    # cast does not cancel its owner's spell or prevent its cooldown from
    # beginning. Owner death *does* take the generic CastCancelCheck path.
    # The mailbox lives on the target row, so look the owner up through the
    # fixed champion mirror rather than incorrectly using target `alive`.
    mirror = enemy_champion_index(n)
    r_caster = jnp.clip(mirror, 0, n - 1).astype(jnp.int32)
    r_cancelled = r_pending & ~alive[r_caster]
    r_active = r_pending & ~r_cancelled
    r_elapsed = jnp.where(r_active, buff_elapsed[:, r_slot] + dt_s,
                          buff_elapsed[:, r_slot])
    r_fires = r_active & (r_elapsed >= buff_duration[:, r_slot])
    if hp is None or max_hp is None:
        # Compatibility only for direct, non-R callers. A real pending R
        # requires the target-health snapshot supplied by `step.tick`.
        missing_hp = jnp.zeros_like(x)
    else:
        missing_hp = jnp.maximum(max_hp - hp, 0.0)
    raw_r = r_damage_at_rank(buff_power[:, r_slot].astype(jnp.int32), missing_hp)
    damage_r = jnp.where(r_fires & alive, post_mitigation_damage(raw_r, mr, jnp),
                         jnp.zeros_like(raw_r))
    dealt_by_r = jnp.where(r_fires, mirror, -1).astype(jnp.int8)
    # The mailbox survives through the real cast windup. Clearing it on the
    # first `UpdateBuffs` call would silently restore the old synchronous
    # behavior by deleting the queued hit before `Spell.Update` can finish it.
    r_expired = r_fires | r_cancelled

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
    r_cast_finished = r_fires[r_caster]
    r_rank = jnp.clip(buff_power[r_caster, r_slot].astype(jnp.int32), 1,
                      len(R_COOLDOWNS))
    r_cd_table = jnp.asarray(R_COOLDOWNS, spell_cooldown.dtype)
    new_cd = new_cd.at[:, Slot.R].set(
        jnp.where(r_cast_finished, r_cd_table[r_rank - 1], new_cd[:, Slot.R]))

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
        buff_power=buff_power,
        spell_cooldown=new_cd,
        damage_dealt=damage_dealt.astype(x.dtype),
        dealt_by=dealt_by,
        # `SetStatus(CanAttack, false)` and `SetStatus(Ghosted, true)` for the
        # duration -- E suppresses autos and passes through collision.
        suppress_attack=e_active & ~e_expired,
        ghosted=e_active & ~e_expired,
        damage_multiplier=damage_multiplier,
        armor_percent_base_bonus=armor_percent_base_bonus,
        armor_percent_bonus=armor_percent_bonus,
        mr_percent_base_bonus=mr_percent_base_bonus,
        mr_percent_bonus=mr_percent_bonus,
        q_empowered=q_empowered,
        q_skip_next=q_skip_next,
    )
