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
  Modelled here the same way: :func:`grant_w_passive` grants it the tick
  ``spell_level[..., Slot.W]`` first becomes >= 1, independent of
  :func:`cast_w`, and :func:`w_passive_modifiers` reports the RAW
  ``PercentBaseBonus``/``PercentBonus`` pair (``-0.2``/``+0.2`` while
  granted, ``0``/``0`` otherwise) rather than a single pre-composed
  multiplier, so the caller can run the real ``combat.stat_total`` formula
  instead of a flat ``*1.2``.

What this module does: the buff bookkeeping for both (duration, expiry,
cooldown-at-cast for the active window, permanent-and-granted-once for the
passive), plus the pure constants (``W_DAMAGE_MULT``, ``W_PASSIVE_ARMOR_PCT``,
``W_PASSIVE_MR_PCT``) and, from :func:`step_buffs`, the *per-unit* values a
caller needs to actually apply these -- ``BuffStep.damage_multiplier``
(unconditionally 1.0, per the active-window bug-compat note above) and the
``armor_percent_base_bonus``/``armor_percent_bonus``/``mr_percent_base_bonus``/
``mr_percent_bonus`` quartet (the passive). Wired into ``step.py``: it
grants the passive and computes ``armor_eff``/``magic_resist_eff`` via
``combat.stat_total`` (not a flat ``* (1 + pct)``, which is exactly the bug
this section replaced) BEFORE :func:`step_buffs`, and uses those wherever
mitigation is computed that tick -- E's and R's damage inside
:func:`step_buffs` (`SPELL-007`: they used to see the pre-passive resists),
``step_autoattack``'s ``target_resist``, ``step_missiles``'s ``armor``.
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

The buff lifecycle (`STRUCT-001`)
---------------------------------
State: ``LaneState.buffs``, one typed record per buff kind
(:class:`~lanerl_jax.sim.state.Buffs`) -- not a lane table. Every buff has
exactly one START (its ``cast_*``, or :func:`grant_w_passive`) and exactly one
END, ``end_<kind>``, which is the C# ``OnDeactivate`` and is called by EVERY
removal path:

    ==============  ===========================================================
    end_e           expiry at 3.0 s (step_buffs); cancel by re-press >= 1.0 s
                    (cast_e). Writes the rank cooldown.
    end_q           expiry at 4.5 s (step_buffs); the empowered swing landing
                    (step.py). Writes the 8 s cooldown.
    end_q_haste     expiry (step_buffs). No cooldown.
    end_w           expiry (step_buffs). No cooldown (W's starts at cast).
    end_r_pending   the windup completing (step_buffs; writes the CASTER's R
                    cooldown) or the caster dying (step_buffs; no cooldown --
                    the generic CastCancelCheck).
    ==============  ===========================================================

**Death is not a removal path** (`SPELL-006`, fixed): nothing on the server
removes a buff on death -- ``AttackableUnit.UpdateBuffs`` keeps ticking them
and neither ``Champion.Die`` nor ``Champion.Respawn`` touches one -- so a
corpse's E and Q run out normally and their ``OnDeactivate`` writes the
cooldown, and ``GarenE.OnUpdate`` (no ``IsDead`` check) keeps dealing spin
damage from the corpse. The sim used to wipe E/Q/W on death with no cooldown,
which made dying a free E and Q reset.

The writer set of ``spell_cooldown`` is :data:`COOLDOWN_WRITERS` and nothing
else in the simulator writes it (``tests/test_buff_lane_lint.py`` enforces
it).

What a unit may DO under its buffs is :func:`status` -- ``ghosted`` (collision),
``can_attack`` (the auto-attack gate), ``cast_locked``/``can_cast`` (the cast
gate in ``orders.apply_orders`` AND the observation's spell-availability
feature). One function, so the observation and the cast gate cannot disagree
again: E is locked for the first ``E_CANCEL_MIN_S`` of its spin and then
pressable -- as a CANCEL -- in both.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .combat import post_mitigation_damage
from .state import Buffs, Kind

__all__ = [
    "Slot", "E_RADIUS", "E_DURATION_S", "E_TICK_MS",
    "E_MINION_MULTIPLIER", "E_COOLDOWNS", "SKILL_ORDER", "RANKS_BY_LEVEL",
    "e_damage_at_rank", "cast_e", "end_e", "E_CANCEL_MIN_S",
    "Q_BUFF_DURATION", "Q_COOLDOWN",
    "Q_HASTE_MULTIPLIER", "q_haste_duration_at_rank", "q_silence_duration_at_rank",
    "q_damage_at_rank", "cast_q", "end_q", "end_q_haste", "consume_q_skip",
    "W_DURATIONS", "W_COOLDOWNS", "W_DAMAGE_MULT", "W_PASSIVE_ARMOR_PCT",
    "W_PASSIVE_MR_PCT", "w_duration_at_rank", "cast_w", "end_w",
    "grant_w_passive", "WPassiveModifiers", "w_passive_modifiers",
    "R_COOLDOWNS", "R_BASE_PER_RANK", "R_MISSING_HP_FRAC", "R_CAST_RANGE",
    "R_CAST_TIME_S",
    "r_damage_at_rank", "cast_r", "end_r_pending", "enemy_champion_index",
    "decay_cooldowns", "COOLDOWN_WRITERS", "BUFF_NAMES", "active_by_name",
    "Status", "status", "status_of", "cast_locked", "e_cancellable",
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
    ``Data``: ``Spell1=GarenQ, Spell2=GarenW, Spell3=GarenE, Spell4=GarenR``.

    Indexes ``spell_level``/``spell_cooldown`` and :class:`Status`'s
    ``cast_locked``/``can_cast`` -- and NOTHING about buffs, which are
    records found by name (`STRUCT-001`, `SPELL-005`)."""
    Q, W, E, R = 0, 1, 2, 3


#: Server buff name (``AddBuff("...")`` in the vendored Garen scripts) ->
#: the :class:`~lanerl_jax.sim.state.Buffs` field that models it. R's pending
#: hit is a sim-only mailbox and has no server name.
BUFF_NAMES = {
    "GarenE": "e",
    "GarenW": "w",
    "GarenWPassive": "w_passive",
    "GarenQ": "q",
    "GarenQHaste": "q_haste",
}


def active_by_name(buffs: Buffs) -> dict:
    """``{server buff name: (N,) bool}`` -- which named buffs each unit has.
    For renderers and injectors that speak the server's names."""
    out = {}
    for name, field in BUFF_NAMES.items():
        rec = getattr(buffs, field)
        out[name] = rec if field == "w_passive" else rec.active
    return out


# `Buffs/Garen/GarenQHaste.cs:34`: `MoveSpeed.PercentBonus += 0.35f`.
Q_HASTE_MULTIPLIER = 1.35

#: ``GetUnitsInRange(Owner.Position, 330f, true)``
E_RADIUS = 330.0
#: ``AddBuff("GarenE", 3f, ...)``
E_DURATION_S = 3.0
#: ``TimeSinceLastTick >= 500.0f``. E's accumulator is the one buff clock kept
#: in MILLISECONDS (``EBuff.tick_acc_ms``), because the server's is: it is
#: initialised to 500, RESET TO 0 on every fire, so the period drifts with the
#: frame time and the fire times are 0.0167, 0.5333, 1.050, 1.5667, 2.0833,
#: 2.600 -- SIX ticks, the seventh falling at 3.117 s, past the 3.0 s expiry
#: (`SPELL-003`). An absolute `floor(elapsed / 500)` grid fires SEVEN times.
E_TICK_MS = 500.0
#: minions take three quarters
E_MINION_MULTIPLIER = 0.75
#: `GetSpell("GarenECancel").SetCooldown(1f, true)` -- Characters/Garen/E.cs.
#: A re-cast before this is silently refused (`Spell.Cast` sees a spell that is
#: not `STATE_READY`) and the spin continues; at or after it, the spin ends and
#: the full rank cooldown starts. Real League: "can be recast after 1 second
#: while active".
E_CANCEL_MIN_S = 1.0
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


def _where(mask, new, old):
    return jnp.where(mask, jnp.asarray(new, old.dtype), old)


def _cast_rank(rank, cap):
    """The rank a cast records on its buff: 1..cap, int8."""
    return jnp.clip(rank.astype(jnp.int32), 1, cap).astype(jnp.int8)


# ------------------------------------------------------------ cooldowns ---
def decay_cooldowns(spell_cooldown, dt_s):
    """``Spell.Update``'s per-tick countdown on all four slots, floored at 0.
    The only cooldown writer that is not a cast or a buff end."""
    return jnp.maximum(spell_cooldown - dt_s, 0.0)


#: EVERY function that writes ``spell_cooldown`` in the simulator. Nothing
#: else may (`STRUCT-001`; ``tests/test_buff_lane_lint.py`` rule 4). Casts
#: write their cast-time value, ends write the ``OnDeactivate`` value:
#:
#: * ``cast_q``  -- 0 (``Q.cs:85`` overwrites the engine's cast-time cooldown);
#: * ``cast_w``  -- the rank cooldown, at cast (engine default, unmodified);
#: * ``end_e``   -- the rank cooldown, on expiry OR cancel (``GarenE.OnDeactivate``);
#: * ``end_q``   -- 8 s, on expiry OR the empowered hit (``GarenQ.OnDeactivate``);
#: * ``end_r_pending`` -- the caster's rank cooldown when R's windup FINISHES
#:   (``Spell.FinishCasting``); none when the caster dies mid-windup;
#: * ``decay_cooldowns`` -- the per-tick countdown.
#:
#: ``cast_e`` and ``cast_r`` write none themselves (E's cooldown starts when the
#: spin ends, R's when the windup finishes); ``cast_e``'s cancel goes through
#: ``end_e``. State constructors (``empty_state``, parity injection, the eval's
#: ``StateRebuilder``) build the array from outside the simulation and are not
#: writers in this sense.
COOLDOWN_WRITERS = ("cast_q", "cast_w", "end_e", "end_q", "end_r_pending",
                    "decay_cooldowns")


# --------------------------------------------------------------------- E ---
def e_damage_at_rank(rank: jax.Array, attack_damage: jax.Array) -> jax.Array:
    """``10 + 12.5*(rank-1) + AD*(0.35 + 0.05*(rank-1))``, snapshotted at cast."""
    r = jnp.maximum(rank.astype(attack_damage.dtype), 1.0)
    return 10.0 + 12.5 * (r - 1.0) + attack_damage * (0.35 + 0.05 * (r - 1.0))


def e_cancellable(buffs: Buffs) -> jax.Array:
    """``GarenECancel`` is ``STATE_READY``: the spin is live and at least
    :data:`E_CANCEL_MIN_S` old, so a press ENDS it (``SPELL-001``)."""
    return buffs.e.active & (buffs.e.elapsed_s >= E_CANCEL_MIN_S)


def _e_can_start(buffs: Buffs, cooldown_e, rank):
    return (cooldown_e <= 0) & (rank > 0) & ~buffs.e.active


def end_e(buffs: Buffs, spell_cooldown, ended, rank):
    """``GarenE.OnDeactivate``: restore ``CanAttack``, clear ``Ghosted``, swap
    ``GarenE`` back into the slot and ``SetCooldown`` the FULL rank cooldown.
    Called on expiry (:func:`step_buffs`) and on cancel (:func:`cast_e`).

    ``power`` and ``tick_acc_ms`` belong to the ended script instance and are
    left as they are: nothing reads them while the spin is inactive, the
    accumulator is zeroed by the next :func:`step_buffs` and both are
    re-primed by the next cast.
    """
    r = jnp.clip(rank.astype(jnp.int32), 1, len(E_COOLDOWNS))
    cd_table = jnp.asarray(E_COOLDOWNS, spell_cooldown.dtype)
    spell_cooldown = spell_cooldown.at[:, Slot.E].set(
        jnp.where(ended, cd_table[r - 1], spell_cooldown[:, Slot.E]))
    e = buffs.e.replace(active=buffs.e.active & ~ended,
                        elapsed_s=_where(ended, 0.0, buffs.e.elapsed_s))
    return buffs.replace(e=e), spell_cooldown


def cast_e(buffs: Buffs, spell_cooldown, want_cast, rank, attack_damage):
    """A press of E: start the spin, or cancel the running one, or nothing.

    Returns ``(buffs, spell_cooldown, started)``; ``started`` is a fresh cast
    only (a cancel is not reported as a cast).
    """
    # E HAS THREE OUTCOMES, not two, and modelling it with two gave an RL
    # policy a permanent damage aura it learned to live off.
    #
    # The server does not "refuse" a re-cast -- it SWAPS THE SLOT.
    # `Characters/Garen/E.cs`'s `OnSpellPostCast` does
    # `SetSpell("GarenECancel", 2, true)`, which replaces `Spells[2]` with a
    # different spell object and deactivates the GarenE one (discarding the
    # cooldown `FinishCasting` had just written), then puts GarenECancel on a
    # 1 s cooldown. So a "cast slot 2" during the spin reaches GarenECancel,
    # whose `OnSpellPostCast` removes the GarenE buff; the buff's
    # `OnDeactivate` (`end_e`) restores `CanAttack`, clears `Ghosted`, swaps
    # GarenE back and sets the FULL rank cooldown. Inside the first second
    # GarenECancel is not `STATE_READY`, so `Spell.Cast` returns false and the
    # spin simply continues.
    #
    #     re-cast at elapsed < 1.0 s   ignored, spin continues
    #     re-cast at elapsed >= 1.0 s  spin ENDS NOW, full cooldown starts
    #     no re-cast                   spin ends at 3.0 s, same cooldown
    #
    # Real League agrees: "Recast: Judgment is ended early", "can be recast
    # after 1 second while active". (Real League also starts the cooldown on
    # CAST; this 4.20 build starts it when the spin ends. The server is the
    # authority here, so cooldown-on-end stays.)
    #
    # What the old two-outcome model cost (`SPELL-001`): with neither the
    # cancel nor a guard, mid-spin `spell_cooldown[E]` is 0, the re-cast
    # passed `ready`, and it reset the elapsed clock to 0 -- so a policy
    # casting E every decision held the spin at elapsed 0 forever. Measured on
    # the trained checkpoint: E cast on 80-83% of decisions and the sim's E
    # cooldown never rose ONCE in 300 s. An intermediate "always refuse" guard
    # was wrong in the other direction: a spin that cannot be cancelled runs
    # its full 3 s before the cooldown even begins.
    #
    # The observation reads the SAME rule through `status` (`cast_locked[E]`
    # is "live and younger than E_CANCEL_MIN_S"), so a policy is told E is
    # pressable exactly when a press does something.
    start = want_cast & _e_can_start(buffs, spell_cooldown[:, Slot.E], rank)
    cancel = want_cast & e_cancellable(buffs)
    # A cancelled spin deals no tick on the tick it is cancelled: orders run
    # before `step_buffs`, which is also the server's order (`RemoveBuff` in
    # the order phase, `UpdateBuffs` after).
    buffs, spell_cooldown = end_e(buffs, spell_cooldown, cancel, rank)
    dmg = e_damage_at_rank(rank, attack_damage)
    e = buffs.e
    e = e.replace(
        active=e.active | start,
        elapsed_s=_where(start, 0.0, e.elapsed_s),
        power=jnp.where(start, dmg.astype(e.power.dtype), e.power),
        # prime the accumulator to E_TICK_MS: the server's
        # `TimeSinceLastTick = 500` initial condition, so the first
        # `OnUpdate` after the cast fires immediately.
        tick_acc_ms=_where(start, E_TICK_MS, e.tick_acc_ms))
    return buffs.replace(e=e), spell_cooldown, start


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


def _q_can_start(buffs: Buffs, cooldown_q, rank):
    return (cooldown_q <= 0) & (rank > 0) & ~buffs.q.active


def cast_q(buffs: Buffs, spell_cooldown, want_cast, rank):
    """Open Garen's empowered-next-auto-attack window (Q.cs:76-85).

    Gated on rank, cooldown, and on the window not already being open
    (``SPELL-008``: the server would RENEW the buff -- ``SealSpellSlot`` is
    inert -- and real League refuses; the sim keeps the League-correct
    refusal as a recorded deviation).

    Sets the spell's cooldown to 0 immediately (Q.cs:85 overwrites the
    engine's default cast-time cooldown in the same event). :func:`end_q`
    sets the real 8 s when the window closes.

    Does **not** deal Q's damage or apply its silence -- the empowered swing
    does, in ``step.py``.
    """
    ready = want_cast & _q_can_start(buffs, spell_cooldown[:, Slot.Q], rank)
    q = buffs.q.replace(
        active=buffs.q.active | ready,
        elapsed_s=_where(ready, 0.0, buffs.q.elapsed_s),
        # `GarenQ.OnActivate` calls `SkipNextAutoAttack()` after cancelling
        # the current swing; `step_autoattack` consumes it at the next gate.
        skip_next=buffs.q.skip_next | ready)
    qh = buffs.q_haste.replace(
        active=buffs.q_haste.active | ready,
        elapsed_s=_where(ready, 0.0, buffs.q_haste.elapsed_s),
        rank=jnp.where(ready, _cast_rank(rank, 5), buffs.q_haste.rank))
    spell_cooldown = spell_cooldown.at[:, Slot.Q].set(
        jnp.where(ready, 0.0, spell_cooldown[:, Slot.Q]))
    return buffs.replace(q=q, q_haste=qh), spell_cooldown, ready


def end_q(buffs: Buffs, spell_cooldown, ended):
    """``GarenQ.OnDeactivate``: restore the ordinary attack spell, unseal the
    slot and ``SetCooldown(8)``. Called on expiry (:func:`step_buffs`) and when
    the empowered ``GarenQAttack`` lands (``step.py``: its
    ``OnSpellPostCast`` calls ``OnSpellEnd``, which deactivates the buff --
    earlier than the natural 4.5 s in the usual successful-hit case)."""
    spell_cooldown = spell_cooldown.at[:, Slot.Q].set(
        jnp.where(ended, jnp.asarray(Q_COOLDOWN, spell_cooldown.dtype),
                  spell_cooldown[:, Slot.Q]))
    q = buffs.q.replace(active=buffs.q.active & ~ended,
                        elapsed_s=_where(ended, 0.0, buffs.q.elapsed_s),
                        skip_next=buffs.q.skip_next & ~ended)
    return buffs.replace(q=q), spell_cooldown


def consume_q_skip(buffs: Buffs, consumed):
    """``SkipNextAutoAttack`` spent at the swing gate (``step_autoattack``)."""
    return buffs.replace(q=buffs.q.replace(skip_next=buffs.q.skip_next & ~consumed))


def end_q_haste(buffs: Buffs, ended):
    """``GarenQHaste.OnDeactivate``: the +35% move speed goes. No cooldown."""
    qh = buffs.q_haste
    return buffs.replace(q_haste=qh.replace(
        active=qh.active & ~ended, elapsed_s=_where(ended, 0.0, qh.elapsed_s),
        rank=_where(ended, 0, qh.rank)))


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


def cast_w(buffs: Buffs, spell_cooldown, want_cast, rank):
    """Open Garen's 0.7x-incoming-damage window (W.cs:49-53).

    Sets the cooldown to the rank's table value **immediately**, unlike
    :func:`cast_q` and E -- see :data:`W_COOLDOWNS`. Does not touch
    ``GarenWPassive``: that is granted on rank-up, not on cast
    (:func:`grant_w_passive`).
    """
    ready = want_cast & (spell_cooldown[:, Slot.W] <= 0) & (rank > 0)
    r = jnp.clip(rank.astype(jnp.int32), 1, len(W_COOLDOWNS))
    cd_table = jnp.asarray(W_COOLDOWNS, spell_cooldown.dtype)
    w = buffs.w.replace(
        active=buffs.w.active | ready,
        elapsed_s=_where(ready, 0.0, buffs.w.elapsed_s),
        rank=jnp.where(ready, _cast_rank(rank, len(W_COOLDOWNS)), buffs.w.rank))
    spell_cooldown = spell_cooldown.at[:, Slot.W].set(
        jnp.where(ready, cd_table[r - 1], spell_cooldown[:, Slot.W]))
    return buffs.replace(w=w), spell_cooldown, ready


def end_w(buffs: Buffs, ended):
    """``GarenW.OnDeactivate``: the window closes. No cooldown -- W's started
    at cast."""
    w = buffs.w
    return buffs.replace(w=w.replace(
        active=w.active & ~ended, elapsed_s=_where(ended, 0.0, w.elapsed_s),
        rank=_where(ended, 0, w.rank)))


def grant_w_passive(buffs: Buffs, spell_level, alive):
    """``W.OnLevelUpSpell``: ``GarenWPassive`` the moment W first has a rank.

    Registered from spell-object construction, independent of ever pressing W
    (see the module docstring's W section). ``infiniteduration``: never ended.
    """
    grant = alive & (spell_level[:, Slot.W] >= 1)
    return buffs.replace(w_passive=buffs.w_passive | grant)


class WPassiveModifiers(NamedTuple):
    """The RAW ``PercentBaseBonus``/``PercentBonus`` pairs ``GarenWPassive``
    writes (``-0.2``/``+0.2`` while it is up, ``0``/``0`` otherwise) -- not one
    pre-composed multiplier, because ``Stat.Total``'s ``FlatBonus`` term sits
    BETWEEN them (``combat.stat_total``) and Garen's Armor has a flat rune
    bonus."""
    armor_percent_base_bonus: jax.Array
    armor_percent_bonus: jax.Array
    mr_percent_base_bonus: jax.Array
    mr_percent_bonus: jax.Array


def w_passive_modifiers(buffs: Buffs, alive, dtype=jnp.float32) -> WPassiveModifiers:
    """Per-unit resist modifiers from ``GarenWPassive``, for anything that
    mitigates damage against a unit's Armor/MR (``step.py``) or reports it
    (the observation)."""
    on = alive & buffs.w_passive
    z = jnp.zeros(on.shape, dtype)
    return WPassiveModifiers(
        armor_percent_base_bonus=jnp.where(on, jnp.asarray(-W_PASSIVE_ARMOR_PCT, dtype), z),
        armor_percent_bonus=jnp.where(on, jnp.asarray(W_PASSIVE_ARMOR_PCT, dtype), z),
        mr_percent_base_bonus=jnp.where(on, jnp.asarray(-W_PASSIVE_MR_PCT, dtype), z),
        mr_percent_bonus=jnp.where(on, jnp.asarray(W_PASSIVE_MR_PCT, dtype), z))


# --------------------------------------------------------------------- R ---
#: ``Spells/GarenR/GarenR.json`` ``Cooldown1``-``Cooldown3`` (ranks 4-6 exist
#: in the JSON as padding but are unreachable -- R caps at 3 ranks). No
#: ``SetCooldown`` call anywhere in ``R.cs``: like W, this is the engine's
#: unmodified default, starting when casting finishes.
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


def cast_r(buffs: Buffs, spell_cooldown, want_cast, rank, target):
    """Start R's windup and mark its target for a delayed hit.

    `GarenR.OnSpellPostCast` reads the target's health, not
    `OnSpellPreCast`. The pending record therefore stores the caster's rank --
    not a precomputed damage snapshot -- and :func:`step_buffs` reads current
    HP when the 0.435-second engine cast timer completes. Writes no cooldown:
    R's starts when the windup finishes (:func:`end_r_pending`).

    The pending hit is written onto the **target's** row (not the caster's)
    via :data:`enemy_champion_index`, not through a data-dependent scatter on
    ``target`` -- with exactly two champion slots, "the other champion" is a
    fixed permutation, and scattering through an attacker-chosen index would
    risk one champion's not-casting no-op write landing on the same
    destination row as the other's real write in the same ``.at[].set()``
    call, with JAX's duplicate-index tie-break deciding which one survives.

    ``target`` is expected to already be validated by the caller (an enemy
    champion, in range) -- see ``orders.apply_orders``. Re-checked against
    :func:`enemy_champion_index` here anyway, so a caller that skips that gate
    fails closed (no cast) instead of open.
    """
    n = rank.shape[0]
    mirror = enemy_champion_index(n)
    mirror_idx = jnp.clip(mirror, 0, n - 1).astype(jnp.int32)
    valid_target = (target == mirror) & (mirror >= 0)
    already_casting = buffs.r_pending.active[mirror_idx]
    ready = (want_cast & (spell_cooldown[:, Slot.R] <= 0) & (rank > 0)
             & (target >= 0) & valid_target & ~already_casting)

    hits_me = ready[mirror_idx]
    rank_to_me = rank[mirror_idx]
    rp = buffs.r_pending
    rp = rp.replace(
        active=rp.active | hits_me,
        elapsed_s=_where(hits_me, 0.0, rp.elapsed_s),
        rank=jnp.where(hits_me, rank_to_me.astype(jnp.int8), rp.rank))
    return buffs.replace(r_pending=rp), spell_cooldown, ready


def end_r_pending(buffs: Buffs, spell_cooldown, fired, cancelled):
    """R's pending hit leaves the victim's row. ``fired`` is
    ``Spell.FinishCasting`` -- it starts the CASTER's rank cooldown (the
    caster is the victim's :func:`enemy_champion_index` mirror, and the rank is
    the one the pending record carries); ``cancelled`` is the caster dying
    mid-windup, the generic ``CastCancelCheck``, which starts none."""
    n = fired.shape[0]
    mirror = enemy_champion_index(n)
    r_caster = jnp.clip(mirror, 0, n - 1).astype(jnp.int32)
    r_cast_finished = fired[r_caster]
    r_rank = jnp.clip(buffs.r_pending.rank[r_caster].astype(jnp.int32), 1,
                      len(R_COOLDOWNS))
    r_cd_table = jnp.asarray(R_COOLDOWNS, spell_cooldown.dtype)
    spell_cooldown = spell_cooldown.at[:, Slot.R].set(
        jnp.where(r_cast_finished, r_cd_table[r_rank - 1],
                  spell_cooldown[:, Slot.R]))
    ended = fired | cancelled
    rp = buffs.r_pending
    rp = rp.replace(active=rp.active & ~ended,
                    elapsed_s=_where(ended, 0.0, rp.elapsed_s),
                    rank=_where(ended, 0, rp.rank))
    return buffs.replace(r_pending=rp), spell_cooldown


# ---------------------------------------------------------------- status ---
class Status(NamedTuple):
    """What each unit may do under its buffs and cast state, ONE derivation
    consumed by collision, the auto-attack gate, the cast gate and the
    observation (`STRUCT-001`)."""
    #: ``StatusFlags.Ghosted`` (GarenE): passes through unit collision.
    ghosted: jax.Array      # (N,) bool
    #: ``StatusFlags.CanAttack`` and no cast in progress: GarenE clears the
    #: flag for its spin (`SPELL-002`), a recall wind-up/channel or R's
    #: uncancellable windup holds the attack.
    can_attack: jax.Array   # (N,) bool
    #: alive, not silenced, and no ordinary cast in progress (recall wind-up,
    #: R's windup) -- the unit-level half of ``can_cast``.
    may_cast: jax.Array     # (N,) bool
    #: by ``Slot``: the slot is not ``STATE_READY`` despite a zero cooldown --
    #: Q while its window is open, E for the first ``E_CANCEL_MIN_S`` of its
    #: spin. What the observation's cooldown feature reports as locked.
    cast_locked: jax.Array  # (N, 4) bool
    #: by ``Slot``: a press does something (for E, starting OR cancelling the
    #: spin). R's target and range are order properties, checked by the order.
    can_cast: jax.Array     # (N, 4) bool


def cast_locked(buffs: Buffs) -> jax.Array:
    """``(N, 4)`` by :class:`Slot`: locked by a live buff despite a zero
    cooldown. Q: the empowerment window is open (``SPELL-008``). E: the spin
    is live and ``GarenECancel`` is not yet ready (``SPELL-001``)."""
    none = jnp.zeros_like(buffs.q.active)
    by_slot = {Slot.Q: buffs.q.active, Slot.W: none,
               Slot.E: buffs.e.active & ~e_cancellable(buffs), Slot.R: none}
    return jnp.stack([by_slot[s] for s in range(4)], axis=1)


def status(buffs: Buffs, *, alive, spell_level, spell_cooldown, silenced_ms,
           recall_windup_ms, recall_channel_ms, r_cast_ms) -> Status:
    """:class:`Status` from a buff state and the cast-lock timers.

    Takes arrays rather than a ``LaneState`` because ``step.py`` asks it at
    two points of a tick: collision reads the INCOMING buffs (``UpdateBuffs``
    runs after ``Map.Update``), the auto-attack gate the post-``UpdateBuffs``
    ones with this tick's recall/R timers. :func:`status_of` is the
    whole-state form every other caller uses.
    """
    ghosted = buffs.e.active & alive
    no_cast_in_progress = (recall_windup_ms <= 0) & (r_cast_ms <= 0)
    can_attack = (alive & ~buffs.e.active & no_cast_in_progress
                  & (recall_channel_ms <= 0))
    may_cast = alive & (silenced_ms <= 0) & no_cast_in_progress
    ranked = spell_level > 0
    off_cd = spell_cooldown <= 0
    ready = {
        Slot.Q: _q_can_start(buffs, spell_cooldown[:, Slot.Q],
                             spell_level[:, Slot.Q]),
        Slot.W: ranked[:, Slot.W] & off_cd[:, Slot.W],
        Slot.E: (_e_can_start(buffs, spell_cooldown[:, Slot.E],
                              spell_level[:, Slot.E])
                 | e_cancellable(buffs)),
        Slot.R: ranked[:, Slot.R] & off_cd[:, Slot.R],
    }
    can_cast = may_cast[:, None] & jnp.stack([ready[s] for s in range(4)], axis=1)
    return Status(ghosted=ghosted, can_attack=can_attack, may_cast=may_cast,
                  cast_locked=cast_locked(buffs), can_cast=can_cast)


def status_of(state) -> Status:
    """:func:`status` of a whole ``LaneState``."""
    return status(state.buffs, alive=state.alive, spell_level=state.spell_level,
                  spell_cooldown=state.spell_cooldown,
                  silenced_ms=state.silenced_ms,
                  recall_windup_ms=state.recall_windup_ms,
                  recall_channel_ms=state.recall_channel_ms,
                  r_cast_ms=state.r_cast_ms)


# ------------------------------------------------------------ step_buffs ---
class BuffStep(NamedTuple):
    buffs: Buffs
    spell_cooldown: jax.Array
    damage_dealt: jax.Array     # (N,) post-mitigation damage received this tick
    dealt_by: jax.Array         # (N,) who dealt it, -1 if nobody
    #: Multiply a unit's total incoming damage by this before subtracting HP.
    #: Unconditionally **1.0** -- see :func:`step_buffs`'s W-active section for
    #: why: on this server build, ``GarenW``'s 0.7x never reaches real HP loss
    #: (a stale-local-vs-mutated-field bug in ``AttackableUnit.TakeDamage``),
    #: so bug-compatibility means this field does nothing, not that it holds
    #: :data:`W_DAMAGE_MULT`. Kept as a field (rather than deleted) so a
    #: caller does not need special-casing.
    damage_multiplier: jax.Array
    #: ``GarenWPassive``'s modifiers after this tick's grant -- see
    #: :class:`WPassiveModifiers`.
    armor_percent_base_bonus: jax.Array
    armor_percent_bonus: jax.Array
    mr_percent_base_bonus: jax.Array
    mr_percent_bonus: jax.Array
    #: Q remains a live spell-swapped auto after its one skipped swing.
    q_empowered: jax.Array
    #: ``SkipNextAutoAttack`` still pending on a live window.
    q_skip_next: jax.Array


def step_buffs(*, buffs: Buffs, spell_cooldown, spell_level, x, y, kind, team,
               alive, armor, magic_resist=None, hp=None, max_hp=None,
               collision_radius=None,
               delta_ms: float = 1000.0 / 60.0) -> BuffStep:
    """Advance every Garen buff one tick: E's spin, W's window and passive,
    Q's empowerment/haste windows, and R's one-shot pending hit.

    The tick boundary follows ``Buff.Update`` for all of them: elapsed
    advances first, then a buff's effect for this tick is evaluated, then it
    deactivates -- through its ``end_*`` -- once ``TimeElapsed >= Duration``.

    A DEAD unit's buffs keep running (`SPELL-006`): ``UpdateBuffs`` has no
    death check, so a corpse's spin expires and starts its cooldown on
    schedule, and ``GarenE.OnUpdate`` keeps dealing its periodic damage from
    the corpse. Only the targets must be alive. The exception is R's pending
    hit, whose CASTER dying cancels it (``CastCancelCheck``).

    ``armor``/``magic_resist`` are the TARGETS' effective resists, i.e. with
    ``GarenWPassive`` applied from the incoming buff state (`SPELL-007`);
    ``step.py`` computes them before calling this. ``magic_resist`` falls back
    to ``armor`` if omitted (a direct-caller convenience; ``step.py`` passes
    it).
    """
    n = x.shape[0]
    dt_s = delta_ms / 1000.0

    # ---- E: the spin -------------------------------------------------------
    e = buffs.e
    e_active = e.active
    e_elapsed = jnp.where(e_active, e.elapsed_s + dt_s, e.elapsed_s)
    # THE SERVER'S ACCUMULATOR, not an absolute grid. `GarenE.OnUpdate` keeps
    # `TimeSinceLastTick` in ms, primed to 500 by `cast_e` and reset to 0 on
    # every fire, so the period drifts by one frame each time: 0.0167, 0.5333,
    # 1.050, 1.5667, 2.0833, 2.600 -- six ticks, the seventh at 3.117 s never
    # arriving because the buff expires at 3.0 (`SPELL-003`).
    tick_acc = e.tick_acc_ms + jnp.where(e_active, delta_ms, 0.0)
    fires = e_active & (tick_acc >= E_TICK_MS)
    tick_acc = jnp.where(fires, 0.0, tick_acc)

    d2 = (x[None, :] - x[:, None]) ** 2 + (y[None, :] - y[:, None]) ** 2
    hittable = alive & (kind != Kind.TURRET) & (kind != Kind.NONE)
    # E's 330 is CENTRE-TO-EDGE on the server, not centre-to-centre
    # (`SPELL-004`). `GetUnitsInRange` goes through
    # `CollisionHandler.GetNearestObjects`, whose quadtree nodes are circles
    # of the unit's own collision radius, and `Circle.IntersectsWith(Circle)`
    # tests `dist^2 < (330 + r_unit)^2`. Minions are 40 and champions 30, so
    # the effective radius is 370 and 360. Strict `<`, matching
    # `IntersectsWith`.
    reach = E_RADIUS + (jnp.zeros_like(x) if collision_radius is None
                        else collision_radius)
    hit = (
        fires[:, None] & hittable[None, :]
        & (team[None, :] != team[:, None])
        & (d2 < (reach ** 2)[None, :])
    )
    mult = jnp.where(kind == Kind.LANE_MINION, E_MINION_MULTIPLIER, 1.0)
    raw = e.power[:, None] * mult[None, :]
    dealt = jnp.where(hit, post_mitigation_damage(raw, armor[None, :], jnp),
                      jnp.zeros_like(d2))
    damage_e = dealt.sum(axis=0)
    any_hit = jnp.any(hit, axis=0)
    dealt_by_e = jnp.where(any_hit, jnp.argmax(hit, axis=0), -1).astype(jnp.int8)
    e_expired = e_active & (e_elapsed >= E_DURATION_S)

    # ---- W active: the 0.7x window (real buff, but a no-op on this server) --
    # `GarenW.cs:47-55`'s `PreTakeDamage` listener does
    # `dmg.PostMitigationDamage *= 0.7f`, but `AttackableUnit.TakeDamage`
    # (`AttackableUnit.cs:551,558,585,606,612-616`) already copied
    # `PostMitigationDamage` into a stale LOCAL float BEFORE publishing
    # `OnPreTakeDamage` -- both the real HP subtraction (`:585`) and lifesteal
    # (`:612-616`) read that stale local, never the mutated `damageData`
    # field. Only the cosmetic floating-damage-number packet (`:606`) ever
    # sees the 0.7x. So on THIS server build, W's active window changes
    # NOTHING about real damage taken -- reproduced as bug-compatibility. The
    # window itself (duration, cooldown-at-cast) is still real and tracked.
    w = buffs.w
    w_active = w.active
    w_elapsed = jnp.where(w_active, w.elapsed_s + dt_s, w.elapsed_s)
    w_expired = w_active & (w_elapsed >= w_duration_at_rank(w.rank))
    damage_multiplier = jnp.ones_like(x)

    # ---- GarenWPassive: granted on RANK-UP, not on cast --------------------
    buffs = grant_w_passive(buffs, spell_level, alive)
    wp = w_passive_modifiers(buffs, alive, x.dtype)

    # ---- Q: empowerment window + haste window ------------------------------
    q = buffs.q
    q_active = q.active
    q_elapsed = jnp.where(q_active, q.elapsed_s + dt_s, q.elapsed_s)
    q_expired = q_active & (q_elapsed >= Q_BUFF_DURATION)
    q_live = q_active & ~q_expired
    q_skip_next = q_live & q.skip_next
    q_empowered = q_live & ~q_skip_next

    qh = buffs.q_haste
    qh_active = qh.active
    qh_elapsed = jnp.where(qh_active, qh.elapsed_s + dt_s, qh.elapsed_s)
    qh_expired = qh_active & (qh_elapsed >= q_haste_duration_at_rank(qh.rank))

    # ---- R: one-shot pending hit, on the victim's row ----------------------
    mr = armor if magic_resist is None else magic_resist
    rp = buffs.r_pending
    r_pending = rp.active
    # `GarenR` has `CantCancelWhileWindingUp=1`: a target dying during the
    # cast does not cancel its owner's spell or prevent its cooldown from
    # beginning. Owner death *does* take the generic CastCancelCheck path.
    # The record lives on the target row, so look the owner up through the
    # fixed champion mirror rather than incorrectly using target `alive`.
    mirror = enemy_champion_index(n)
    r_caster = jnp.clip(mirror, 0, n - 1).astype(jnp.int32)
    r_cancelled = r_pending & ~alive[r_caster]
    r_active = r_pending & ~r_cancelled
    r_elapsed = jnp.where(r_active, rp.elapsed_s + dt_s, rp.elapsed_s)
    r_fires = r_active & (r_elapsed >= R_CAST_TIME_S)
    if hp is None or max_hp is None:
        # Compatibility only for direct, non-R callers. A real pending R
        # requires the target-health snapshot supplied by `step.tick`.
        missing_hp = jnp.zeros_like(x)
    else:
        missing_hp = jnp.maximum(max_hp - hp, 0.0)
    raw_r = r_damage_at_rank(rp.rank.astype(jnp.int32), missing_hp)
    damage_r = jnp.where(r_fires & alive, post_mitigation_damage(raw_r, mr, jnp),
                         jnp.zeros_like(raw_r))
    dealt_by_r = jnp.where(r_fires, mirror, -1).astype(jnp.int8)

    # ---- combine E's and R's directly-dealt damage for tick()'s kill
    # attribution cumsum. A victim can in principle take both in the same
    # tick; summed for HP, R's attacker wins the attribution tie -- an
    # arbitrary but documented choice for a genuinely rare double-source tick.
    damage_dealt = damage_e + damage_r
    dealt_by = jnp.where(damage_r > 0, dealt_by_r, dealt_by_e)

    # ---- write the advanced clocks, then END what expired ------------------
    # Every end goes through its `end_*` (= `OnDeactivate`). A cooldown a
    # buff end writes replaces this tick's countdown value for that slot, as
    # before.
    buffs = buffs.replace(
        # the accumulator carries the drift across ticks; zeroed while no
        # spin is live, so a later cast starts clean (cast_e primes it).
        e=buffs.e.replace(elapsed_s=e_elapsed,
                          tick_acc_ms=jnp.where(e_active, tick_acc, 0.0)),
        w=buffs.w.replace(elapsed_s=w_elapsed),
        q=buffs.q.replace(elapsed_s=q_elapsed),
        q_haste=buffs.q_haste.replace(elapsed_s=qh_elapsed),
        r_pending=buffs.r_pending.replace(elapsed_s=r_elapsed))
    cd = decay_cooldowns(spell_cooldown, dt_s)
    buffs, cd = end_e(buffs, cd, e_expired, spell_level[:, Slot.E])
    buffs = end_w(buffs, w_expired)
    buffs, cd = end_q(buffs, cd, q_expired)
    buffs = end_q_haste(buffs, qh_expired)
    buffs, cd = end_r_pending(buffs, cd, fired=r_fires, cancelled=r_cancelled)

    return BuffStep(
        buffs=buffs,
        spell_cooldown=cd,
        damage_dealt=damage_dealt.astype(x.dtype),
        dealt_by=dealt_by,
        damage_multiplier=damage_multiplier,
        armor_percent_base_bonus=wp.armor_percent_base_bonus,
        armor_percent_bonus=wp.armor_percent_bonus,
        mr_percent_base_bonus=wp.mr_percent_base_bonus,
        mr_percent_bonus=wp.mr_percent_bonus,
        q_empowered=q_empowered,
        q_skip_next=q_skip_next,
    )
