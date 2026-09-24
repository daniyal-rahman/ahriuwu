"""Basic-attack missiles: the reason a ranged minion's damage can be wasted.

WHO FIRES ONE, EXACTLY
----------------------
``Spell.FinishCasting``, for an auto-attack::

    if (!CastInfo.Owner.IsMelee)
    {
        if (HasEmptyScript)
        {
            CreateSpellMissile(new MissileParameters { Type = MissileType.Target });
        }
    }
    else
    {
        ApplyEffects(CastInfo.Targets[0].Unit);
        CastInfo.Owner.AutoAttackHit(CastInfo.Targets[0].Unit);
    }

with ``HasEmptyScript = Script.GetType() == typeof(SpellScriptEmpty)``. This
looked like two conditions that separate every unit, with the lane turret as
the interesting case that needs both -- ranged, so a naive "ranged units fire
missiles" rule would give it one, but carrying a real ``BasicAttack.cs``
script (at 0.7x damage against minions) that applies damage itself and leaves
``MissileParameters`` null. That script is real, and exists in Content --
it is just for the wrong turret. ``lanerl/cfg/garen1v1.json`` pins
``"map": 1``, whose outer turrets are ``OrderTurretNormal`` (blue) and
``ChaosTurretWorm`` (red) (``Maps/Map1/LevelScriptObjects.cs:77-89``), and
**neither has a script anywhere in ``Characters/``**. The scripted turret,
``SRUAP_Turret_Order3``/``Chaos3``, is a Map11 unit this map never spawns --
the same wrong-map mistake the turret AD/armour/regen numbers made (see
``data/patch.TURRET_MODELS``), just discovered a second time in a different
column. So the second condition is never false in this slice, and the rule
collapses to the one everything else already implies:

===============  ========  ==============  =========================
unit             IsMelee   BasicAttack.cs  basic attack
===============  ========  ==============  =========================
Garen             true     none            instant
melee minion      true     none            instant
caster minion     false    none            **MISSILE**
cannon minion     false    none            **MISSILE**
lane turret       false    none            **MISSILE**
===============  ========  ==============  =========================

``fires_missile`` is therefore just ``not IsMelee``. Kept as an explicit
per-profile column rather than inlined at the call site anyway, because the
turret history above is exactly why a "this unit is obviously an exception"
shortcut is not trustworthy without checking which model is actually on the
field.

WHY IT MATTERS MORE THAN IT LOOKS
---------------------------------
``SpellData.MissileSpeed`` defaults to **500** units/second, and that default
is a fallback nothing in this slice actually uses -- every basic attack here
overrides it in its own ``Spells/<name>BasicAttack/<name>BasicAttack.json``:
caster minion 650, cannon minion and both outer turrets 1200, melee minion 0
(never read, since melee never fires one). A caster's attack range is 550, so
its missile is in flight for ``550 / 650`` ~= **0.85 seconds** against a
1.49-second attack period -- more than half the attack cycle, not the ~1.1 s
a flat 500 u/s would suggest. And ``SpellMissile.Update`` is unforgiving about
what happens in that window::

    if (HasTarget() && !TargetUnit.IsDead && TargetUnit.Status.HasFlag(Targetable))
    {
        Move(diff);
    }
    else
    {
        SetToRemove();          // target died in flight: NO damage
    }

So a ranged minion whose target dies before its missile lands has wasted the
whole attack. Modelling the damage as instant does not merely shift it in time,
it **creates damage the server never deals** -- and it creates more of it for
the side that is winning, because that side's targets die faster. That is a
negative feedback the sim did not have, and its absence is visible: the sim's
lane runs away to one side while the server's oscillates around the middle,
flipping which team leads every minute or two.

Casters and cannons are 58.4% of a wave, and now so are both outer turrets --
a turret can whiff, too, against a minion that dies to something else first.

The missile homes: ``GetTargetPosition`` returns ``TargetUnit.Position`` every
tick, so it chases a moving target rather than flying at where it was fired.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from .combat import post_mitigation_damage

__all__ = ["DEFAULT_MISSILE_SPEED", "MissileOut", "step_missiles"]

#: ``SpellData.MissileSpeed``'s engine default -- used only when a spell does
#: not override it. Nothing in this slice is in that position: every basic
#: attack that can fire a missile has its own ``MissileSpeed`` in its
#: ``Spells/<name>BasicAttack/<name>BasicAttack.json`` (see
#: ``data.patch.UnitStats.missile_speed``), so this constant exists as a named
#: fallback and a place for that fact to be written down, not as the speed
#: anything actually travels at.
DEFAULT_MISSILE_SPEED = 500.0


class MissileOut(NamedTuple):
    alive: jax.Array
    x: jax.Array
    y: jax.Array
    target: jax.Array
    source: jax.Array
    damage: jax.Array
    speed: jax.Array
    #: ``(N, N)`` post-mitigation damage landing this tick, attacker x victim.
    damage_ij: jax.Array
    #: launches that found no free slot -- must stay 0; see the test
    overflow: jax.Array
    #: the shooter's identity per missile slot (`SLOT-003`); passed through
    #: unchanged, or -1/0, when the caller does not track it
    source_seq: Any = None
    source_model: Any = None
    #: ``(M,)`` post-mitigation damage landed THIS tick by a missile whose
    #: shooter no longer exists (its slot was recycled), indexed by the
    #: missile slot it flew in. NOT in ``damage_ij``: that row now belongs to
    #: a different unit. ``victim`` is the unit it landed on.
    orphan_damage: Any = None
    victim: Any = None


def step_missiles(*, m_alive, m_x, m_y, m_target, m_source, m_damage, m_speed,
                  launches, raw_damage, launch_speed, x, y, alive, targetable,
                  armor, target, delta_ms: float,
                  m_source_seq=None, m_source_model=None, spawn_seq=None,
                  model=None) -> MissileOut:
    """One tick of ``SpellMissile.Update`` for every missile, then the launches.

    Advance-then-spawn, so a missile created at the end of a cast does not also
    travel on the tick it was created -- the server creates it in
    ``FinishCasting`` and moves it in the following ``Update``.

    ``raw_damage`` is pre-mitigation on purpose. ``AutoAttackHit`` computes
    ``GetPostMitigationDamage`` when the missile *lands*, not when it is fired.
    Nothing in this slice changes armour mid-flight so the two agree today, but
    storing the raw value means an armour shred later does not silently read
    from the wrong instant.

    ``launch_speed`` is per-FIRING-UNIT (shape ``(N,)``, gathered by the caller
    from ``data.patch.UnitStats.missile_speed`` through the unit's profile),
    not a single constant -- a caster's missile and a turret's travel at
    650 and 1200 u/s respectively, not the same speed. Read only for units in
    ``launches``; every other unit's entry is irrelevant.

    ``spawn_seq``/``model`` (per unit) and ``m_source_seq``/``m_source_model``
    (per missile) track the SHOOTER's identity (`SLOT-003`). The server's
    missile holds its owner as an object (``CastInfo.Owner``) and lands
    whether or not that owner is still alive: ``Update`` tests only the
    target (`SpellMissile.cs:70-84`), ``CheckFlagsForUnit`` tests the target
    and ``IsValidTarget(Owner, target)`` -- teams and flags, not the owner's
    death (`SpellMissile.cs:177-196`, `SpellData.cs:191-222`) -- and then
    ``Owner.AutoAttackHit`` credits the dead owner. Here the owner is a slot
    index, and a recycled slot is a different unit. A landing whose shooter
    slot's ``spawn_seq`` no longer matches the one recorded at launch goes to
    ``orphan_damage`` instead of ``damage_ij``'s row. Without the four
    arguments nothing is tracked and every landing is credited to the slot.
    """
    n = x.shape[0]
    dt_s = delta_ms / 1000.0
    dtype = m_x.dtype

    # ---- advance the ones already in the air ------------------------------
    t = jnp.clip(m_target, 0, n - 1)
    target_ok = alive[t] & targetable[t] & (m_target >= 0)
    # `else SetToRemove()` -- the target died or became untargetable
    dropped = m_alive & ~target_ok

    dx = x[t] - m_x
    dy = y[t] - m_y
    dist = jnp.sqrt(dx * dx + dy * dy)
    step = m_speed * dt_s
    arrives = m_alive & target_ok & (dist <= step)
    moving = m_alive & target_ok & ~arrives

    frac = step / jnp.maximum(dist, 1e-6)
    m_x = jnp.where(moving, m_x + dx * frac, m_x)
    m_y = jnp.where(moving, m_y + dy * frac, m_y)

    src = jnp.clip(m_source, 0, n - 1)
    landed = jnp.where(arrives, post_mitigation_damage(m_damage, armor[t], jnp),
                       jnp.zeros_like(m_damage))
    tracked = m_source_seq is not None and spawn_seq is not None
    if tracked:
        # `SLOT-003`: the shooter is gone when its slot now holds a different
        # unit. -1 = not recorded (hand-built/injected): trust the slot.
        orphaned = (m_source_seq >= 0) & (spawn_seq[src] != m_source_seq)
    else:
        orphaned = jnp.zeros_like(m_alive)
    orphan_damage = jnp.where(orphaned, landed, jnp.zeros_like(landed))
    landed = jnp.where(orphaned, jnp.zeros_like(landed), landed)
    damage_ij = jnp.zeros((n, n), dtype).at[src, t].add(landed.astype(dtype))
    victim = t

    m_alive = m_alive & ~arrives & ~dropped

    # ---- then launch this tick's shots into the free slots -----------------
    # Lowest firing unit into the lowest free slot, matching `spawn_minion`'s
    # convention: allocation order is part of the parity surface because
    # `argmin` tie-breaks read it.
    M = m_alive.shape[0]
    free = ~m_alive
    order = jnp.argsort(jnp.where(free, 0, 1), stable=True)   # free slots first
    n_free = jnp.sum(free)
    rank = jnp.cumsum(launches) - 1                            # (N,)
    ok = launches & (rank >= 0) & (rank < n_free)
    slot = jnp.where(ok, order[jnp.clip(rank, 0, M - 1)], M)   # M == drop

    def put(arr, vals):
        return arr.at[slot].set(vals.astype(arr.dtype), mode="drop")

    m_alive = m_alive.at[slot].set(True, mode="drop")
    m_x = put(m_x, x)
    m_y = put(m_y, y)
    m_target = put(m_target, target)
    m_source = put(m_source, jnp.arange(n))
    m_damage = put(m_damage, raw_damage)
    m_speed = put(m_speed, launch_speed)
    if tracked:
        m_source_seq = put(m_source_seq, spawn_seq)
        if m_source_model is not None and model is not None:
            m_source_model = put(m_source_model, model)

    return MissileOut(alive=m_alive, x=m_x, y=m_y, target=m_target,
                      source=m_source, damage=m_damage, speed=m_speed,
                      damage_ij=damage_ij,
                      overflow=jnp.sum(launches) - jnp.sum(ok),
                      source_seq=m_source_seq, source_model=m_source_model,
                      orphan_damage=orphan_damage.astype(dtype), victim=victim)
