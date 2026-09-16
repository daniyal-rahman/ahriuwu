"""The auto-attack clock: when a unit swings, and when the damage lands.

This is the mechanic the whole task rests on.  Last-hitting is deciding to swing
at the moment a minion's health will be below your damage *when the swing
resolves*, and that is two numbers: the cooldown until you may swing, and the
wind-up before the swing connects.  Both are in ``UNOBSERVABLE`` -- the state
dump exposes neither -- so this port is checked against the C# source and
against `lanerl_rl/constants.py`'s independent derivation, not against a dump.

Where each piece comes from
---------------------------

``ObjAIBase.Update``::

    if (_autoAttackCurrentCooldown > 0) _autoAttackCurrentCooldown -= diff / 1000.0f;

so the cooldown is in **seconds** while ``diff`` is in milliseconds, and it is
decremented *before* the target logic runs in the same tick.

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
    xp: Any = np,
) -> AutoAttackOut:
    """One tick of the auto-attack clock for a batch of units.

    Order of operations follows ``ObjAIBase.Update``: the cooldown is
    decremented first, *then* the swing gate is evaluated, so a cooldown that
    expires this tick permits a swing on the same tick.
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

    # 1. `if (_autoAttackCurrentCooldown > 0) _autoAttackCurrentCooldown -= diff/1000`
    cd = xp.where(aa_cooldown > 0, aa_cooldown - dt_s, aa_cooldown)

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
    cancel = still_casting & (~has_target | ~in_range)

    raw = attack_damage * xp.where(
        xp.asarray(crit_chance) > 0, crit_damage, xp.ones_like(attack_damage))
    dmg = xp.where(hit, post_mitigation_damage(raw, target_resist, xp),
                   xp.zeros_like(attack_damage))

    attacking = xp.where(hit | cancel, xp.zeros_like(is_attacking, dtype=bool),
                         is_attacking)
    hit_done = has_auto_attacked | hit
    windup = xp.where(hit | cancel, xp.zeros_like(windup), windup)
    cd = xp.where(cancel, xp.zeros_like(cd), cd)

    # 3. the swing gate. `AutoAttackSpell.State == STATE_READY` is "not already
    #    winding up", which is `~attacking` here.
    start = has_target & in_range & can_attack & (~attacking) & (cd <= 0)
    cd = xp.where(start, attack_period, cd)
    windup = xp.where(start, windup_time, windup)
    attacking = attacking | start
    # `HasAutoAttacked = false;` on every swing start
    hit_done = xp.where(start, xp.zeros_like(hit_done, dtype=bool), hit_done)

    return AutoAttackOut(cd, windup, attacking, hit_done, hit, dmg)
