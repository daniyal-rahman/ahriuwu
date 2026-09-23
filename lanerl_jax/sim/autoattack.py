"""The auto-attack clock: when a unit swings, and when the damage lands.

This is the mechanic the whole task rests on.  Last-hitting is deciding to swing
at the moment a minion's health will be below your damage *when the swing
resolves*, and that is two numbers: the cooldown until you may swing, and the
wind-up before the swing connects.  Both are in ``UNOBSERVABLE`` -- the state
dump exposes neither -- so this port is checked against the C# source and
against `lanerl_rl/constants.py`'s independent derivation, not against a dump.

Where each piece comes from
---------------------------

``ObjAIBase.Update`` (`ObjAIBase.cs:1099-1105`)::

    UpdateAssistMarkers();
    UpdateTarget();                                             // the swing gate

    if (_autoAttackCurrentCooldown > 0)
    {
        _autoAttackCurrentCooldown -= diff / 1000.0f;            // AFTER it
    }

so the cooldown is in **seconds** while ``diff`` is in milliseconds, and it is
decremented **after** the target logic has already run in the same tick.  This
docstring asserted the opposite order as established fact until `AA-001` was
measured, and :func:`step_autoattack` implemented what it said.

The consequence is one tick of phase, in the direction that makes the sim
*faster*: a unit may **not** swing on the tick its cooldown expires, because
the gate at that tick still reads the pre-decrement value.  The period is
unchanged either way (the gate at tick T+k sees ``period - k*step`` under both
orderings, so 97 ticks for a 1.6 s cooldown), which is exactly why every
timing test here passed while the phase was wrong.

The observable that settles it is the cooldown dumped on the tick a unit is
seen to swing.  Measured on minion net 1073743551: ``aacd`` 17 -> 0 -> 0 ->
**802** with ``attacking`` 0,0,0,1, and 802/1024 = 0.7832 = 0.8 - 1/60.  The
server's swing tick ends at ``period - one tick``, not at ``period``: the
cooldown it just set is decremented by the very next statement.  Firing before
the decrement and then decrementing is the only arrangement that reproduces
that, and it is why the decrement below is the LAST thing this function does.

The *two* zeros in that sequence are a different fact and are not fixable
here: 17/1024 minus one tick leaves a **positive** residue smaller than the
dump's own 1/2048 rounding threshold, so a still-running cooldown is reported
as 0 for one extra tick.  A one-step harness that injects that 0 therefore
swings one tick before the server on 39.9% of swings no matter what this
function does -- see `RESET-002` and the gate-1 row of
`docs/JAX_FIDELITY_LEDGER.md`.  Do not "fix" that by reintroducing the
decrement-first order: it papers over an unobservable with a wrong one.

``ObjAIBase.UpdateTarget``, the swing gate::

    idealRange = Stats.Range.Total + TargetUnit.CollisionRadius      // edge-to-edge
    if (DistanceSquared(Position, TargetUnit.Position) <= idealRange^2
        && MovementParameters == null
        && AutoAttackSpell.State == STATE_READY
        && CanAttack()
        && _autoAttackCurrentCooldown <= 0) {
            HasAutoAttacked = false;
            IsAttacking = true;
            AutoAttackSpell.Cast(...);
            _autoAttackCurrentCooldown = 1.0f / Stats.GetTotalAttackSpeed();
    }

Note **``idealRange`` adds the target's collision radius** but not the
attacker's: the server's own comment says attacks are edge-to-edge while spells
are centre-to-centre. Getting this wrong shifts every last-hit by a minion
radius.

``Spell.cs:541``, the wind-up::

    DesignerCastTime = autoAttackTotalTime
                     * (AttackDelayCastPercent + AttackDelayCastOffsetPercent)

i.e. the attack period times (0.3 + the character's offset). Garen's offset is
negative, so he commits slightly earlier in the swing than the global default.

``ObjAIBase.AutoAttackHit`` applies ``Stats.AttackDamage.Total`` as physical
damage through ``GetPostMitigationDamage``. Crit is rolled at swing *start*
(``IsNextAutoCrit``) and multiplies at hit time; Garen's ``BaseCritChance`` is 0
and nothing in the lane grants crit, so the roll is modelled as always-false and
the RNG it would consume is not drawn. **That is a deliberate simplification
with a condition attached**: it stops being correct the moment a crit source
enters the kit, and `crit_chance` is threaded through so the assumption is
visible rather than buried.

Windup cancellation (`ObjAIBase.cs:1183-1199`)
-----------------------------------------------
A swing already in progress is aborted -- and the cooldown reset to 0, since
``HasAutoAttacked`` is false for the whole windup -- the instant its target
dies, goes untargetable, leaves vision (`:1183-1191`, unconditional), or
leaves ``idealRange`` while the spell is still ``STATE_CASTING`` and the
attack allows it (`:1193-1199`, gated on ``!CantCancelWhileWindingUp`` --
confirmed ``"0"`` for Garen's, every Map1 lane-minion model's, and the outer
turret's basic attack, so unconditional in every reachable case here).
``CancelAutoAttack(reset=!HasAutoAttacked, fullCancel=true)`` zeroes both the
cooldown and the windup and drops ``IsAttacking`` -- a free, immediate
re-engage opportunity, not merely "the swing whiffs".

Per the tick order (`step.py`'s module docstring): the naturally-completing
tick is not retroactively cancelled, because ``Spell.Update`` (which resolves
a completing swing via ``FinishCasting``) runs before ``UpdateTarget`` in the
same tick -- by the time the cancellation check would run, the spell is
already back to ``STATE_READY``, not ``STATE_CASTING``. So only a swing that
is **still** winding up after this tick's decrement is a cancellation
candidate; :func:`step_autoattack` reproduces exactly that ordering, not a
same-tick race between "completes" and "cancels".

Only ``in_range``/``has_target`` are read for this, both already computed by
the caller from this tick's post-movement, post-acquisition state -- the
exact values ``ObjAIBase.UpdateTarget`` itself would see. This module owns the
clock; ``step.py`` (the target-update path) owns those two conditions, per
the split this docstring used to describe as "not modelled here" before this
was implemented -- see `docs/PORT_AUDIT_AI.md` row 10.4.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from .combat import post_mitigation_damage

__all__ = ["AutoAttackOut", "ideal_attack_range", "step_autoattack"]


class AutoAttackOut(NamedTuple):
    aa_cooldown: Any     # seconds until the next swing may start
    aa_windup: Any       # seconds left in the current swing, 0 when idle
    is_attacking: Any
    has_auto_attacked: Any
    hit: Any             # True on the tick the damage lands
    damage: Any          # post-mitigation damage on that tick, else 0
    consumed_skip: Any   # True where a pending SkipNextAutoAttack was consumed
    start: Any           # True on the tick a swing begins (its target is
                         # the one the hit will land on -- see `step.py`)


def ideal_attack_range(attack_range: Any, target_collision_radius: Any) -> Any:
    """``Stats.Range.Total + TargetUnit.CollisionRadius`` -- edge to edge.

    The attacker's own radius is **not** added. That asymmetry is the server's,
    stated in ``UpdateTarget``'s comment: "Spell casts usually do not take into
    account collision radius, thus range is center -> center VS edge -> edge for
    attacks."
    """
    return attack_range + target_collision_radius


def step_autoattack(
    aa_cooldown: Any,
    aa_windup: Any,
    is_attacking: Any,
    has_auto_attacked: Any,
    *,
    in_range: Any,
    can_attack: Any,
    has_target: Any,
    attack_period: Any,
    windup_time: Any,
    attack_damage: Any,
    target_resist: Any,
    delta_ms: float = 1000.0 / 60.0,
    crit_chance: Any = 0.0,
    crit_damage: Any = 2.0,
    empowered_attack: Any = False,
    empowered_damage: Any = None,
    skip_next_autoattack: Any = False,
    may_engage: Any = True,
    xp: Any = np,
) -> AutoAttackOut:
    """One tick of the auto-attack clock for a batch of units.

    Order of operations follows ``ObjAIBase.Update``: the swing gate
    (``UpdateTarget``) is evaluated first against the cooldown as it stands on
    entry, and the cooldown is decremented *afterwards*.  So a cooldown that
    reaches zero only as a result of **this** tick's decrement does not permit
    a swing until the **next** tick, and a swing started this tick has already
    paid one tick of its new cooldown by the time the tick ends.
    """
    # The decrement must be computed in the SAME precision as the state.
    #
    # The server does `_autoAttackCurrentCooldown -= diff / 1000.0f` with every
    # term a C# float, and the residue after 96 ticks of a 1.6 s cooldown is
    # ~1.2e-7 -- positive, so a 97th tick is needed and the real attack rate is
    # 0.600/s, not the nominal 0.625. That 4% is not rounding noise: over a ten
    # minute lane it is dozens of swings.
    #
    # Leaving `dt_s` as a Python float silently promotes the subtraction to
    # float64 even when the state is float32, which is a *different* accumulation
    # from the server's and can land on the other side of the boundary.
    dt_s = xp.asarray(delta_ms, aa_cooldown.dtype) / xp.asarray(1000.0, aa_cooldown.dtype)

    # 1. the cooldown enters the tick UNCHANGED. `UpdateTarget` -- steps 2-3
    #    below -- is what the server runs next; the decrement is step 4, after
    #    the gate. See this module's docstring for the dumped evidence.
    cd = aa_cooldown

    # 2. advance an in-flight swing; the hit lands when the wind-up runs out
    winding = is_attacking & (aa_windup > 0)
    windup = xp.where(winding, aa_windup - xp.asarray(dt_s, aa_windup.dtype), aa_windup)
    hit = winding & (windup <= 0)

    # `ObjAIBase.cs:1183-1199`: a swing still winding up after this tick's
    # decrement (i.e. it did NOT complete this tick -- see the module
    # docstring for why a completing tick is never retroactively cancelled)
    # is aborted the instant its target is gone or out of range.
    # `CancelAutoAttack(!HasAutoAttacked, true)`: `HasAutoAttacked` is false
    # for the whole windup, so this is always a `reset=true` cancel -- cooldown
    # and windup both zero, immediately re-engageable.
    still_casting = winding & (windup > 0)
    # `~can_attack` belongs here, and leaving it out was exploitable.
    #
    # `can_attack` carries E's `suppress_attack`, but it only gated the START
    # of a swing below -- so a swing already winding up when E was cast kept
    # decrementing and `hit` fired. A policy that times E to land on the last
    # frame of a windup therefore got the auto AND the spin, straight into the
    # last-hit reward channel.
    #
    # The server cancels it: `GarenE` clears `StatusFlags.CanAttack`, Garen's
    # basic attacks carry `CantCancelWhileWindingUp = 0`, so `CastCancelCheck`
    # reaches `Spell.cs`'s `(CastInfo.IsAutoAttack && ... ||
    # !status.HasFlag(StatusFlags.CanAttack))` and calls `ResetSpellCast()`.
    # Real League agrees -- E makes you unable to declare basic attacks.
    cancel_lost_target = still_casting & (~has_target | ~in_range)
    cancel_suppressed = still_casting & ~can_attack
    cancel = cancel_lost_target | cancel_suppressed

    # A skipped auto sets `IsAttacking` but never calls `Spell.Cast` or starts
    # its cooldown. On the following server update `UpdateTarget` observes the
    # ready spell, clears `IsAttacking`, and returns; it cannot begin another
    # swing until the *next* update. This otherwise odd one-tick state is how
    # Garen Q's `SkipNextAutoAttack()` hands the following swing to
    # `GarenQAttack`.
    skipped_ready = is_attacking & (aa_windup <= 0) & ~has_auto_attacked

    raw_normal = attack_damage * xp.where(
        xp.asarray(crit_chance) > 0, crit_damage, xp.ones_like(attack_damage))
    if empowered_damage is None:
        empowered_damage = attack_damage
    raw = xp.where(empowered_attack, empowered_damage, raw_normal)
    dmg = xp.where(hit, post_mitigation_damage(raw, target_resist, xp),
                   xp.zeros_like(attack_damage))

    attacking = xp.where(hit | cancel | skipped_ready,
                         xp.zeros_like(is_attacking, dtype=bool),
                         is_attacking)
    hit_done = has_auto_attacked | hit
    windup = xp.where(hit | cancel, xp.zeros_like(windup), windup)
    # ONLY the lost-target cancel zeroes the cooldown. That path is
    # `CancelAutoAttack(!HasAutoAttacked, true)` -- a `reset=true` cancel.
    # `ResetSpellCast` on the suppressed path leaves `_autoAttackCurrentCooldown`
    # alone, so a suppressed swing does not hand back a free re-engage.
    cd = xp.where(cancel_lost_target, xp.zeros_like(cd), cd)

    # 3. the swing gate. `AutoAttackSpell.State == STATE_READY` is "not already
    #    winding up", which is `~attacking` here.
    #    `may_engage` is `TargetUnit.Team != Team` (`ObjAIBase.cs:1285`): the
    #    swing branch is inside that test, the cancel branch above is not.
    #    A held ALLY target therefore neither starts a swing nor cancels
    #    one already in flight (which lands on the unit it started on).
    start = (has_target & in_range & can_attack & may_engage & (~attacking)
             & (cd <= 0) & ~skipped_ready)
    consumed_skip = start & skip_next_autoattack
    cd = xp.where(start, xp.where(consumed_skip, xp.zeros_like(cd), attack_period), cd)
    windup = xp.where(start,
                      xp.where(consumed_skip, xp.zeros_like(windup), windup_time),
                      windup)
    attacking = attacking | start
    # `HasAutoAttacked = false;` on every swing start
    hit_done = xp.where(start, xp.zeros_like(hit_done, dtype=bool), hit_done)

    # 4. `if (_autoAttackCurrentCooldown > 0) _autoAttackCurrentCooldown -= diff/1000`
    #    -- the LAST statement of `ObjAIBase.Update`, so it also drains one tick
    #    off a cooldown that step 3 has just set. A swing tick therefore ends on
    #    `attack_period - delta`, which is the dumped 802/1024 for a 0.8 s
    #    period. The `> 0` guard is the server's: a cooldown left at exactly 0
    #    (an idle unit, or a `reset=true` cancel) is never pushed negative.
    cd = xp.where(cd > 0, cd - dt_s, cd)

    return AutoAttackOut(cd, windup, attacking, hit_done, hit, dmg, consumed_skip,
                         start)
