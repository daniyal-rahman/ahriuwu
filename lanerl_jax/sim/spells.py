"""Garen's kit. Currently: **E, Judgment** -- the farming and trading spell.

Why E first
-----------
``constants.GAREN_SKILL_ORDER`` takes it at level 1 and maxes it first, with the
reason stated: *"E first: it is the farming and trading spell."* A Garen with no
abilities cannot clear a wave the way the policy will be trained to, so this is
the first one that changes what the agent can do rather than how accurately it
does it.

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
    "e_damage_at_rank", "cast_e", "step_buffs", "ranks_for_level",
]

#: ``constants.GAREN_SKILL_ORDER`` -- one entry per champion level, 0=Q 1=W 2=E 3=R.
#:
#: **THE canonical copy is `lanerl_rl/constants.py`**, which records that this
#: order previously existed in three places that disagreed
#: (``LanerlConfig.SkillOrder``, ``lanerl_bot.build``, and ``obs.AbilityBook``,
#: which encoded a fourth order again). That is not cosmetic: the action mask
#: then forbids a spell the champion HAS and offers one it does not -- and
#: casting an unlearned spell is **not** a no-op on the server, because nothing
#: in ``Spell.Cast`` checks the level, so it grants the effect anyway.
#:
#: Mirrored here rather than imported so the sim has no import-time dependency
#: on the torch-bearing package; `test_skill_order_matches_constants` fails if
#: they drift.
SKILL_ORDER = (2, 0, 1, 2, 2, 3, 2, 2, 0, 0, 3, 0, 0, 1, 1, 3, 1, 1)


class Slot:
    Q, W, E, R = 0, 1, 2, 3


class BuffId:
    """0 is "empty slot", so a zeroed buff table holds no buffs."""
    NONE = 0
    GAREN_E = 1


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
           spell_cooldown, want_cast, rank, attack_damage, slot=0):
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


class BuffStep(NamedTuple):
    buff_id: jax.Array
    buff_elapsed: jax.Array
    spell_cooldown: jax.Array
    damage_dealt: jax.Array     # (N,) post-mitigation damage received this tick
    dealt_by: jax.Array         # (N,) who dealt it, -1 if nobody
    suppress_attack: jax.Array  # (N,) bool: CanAttack cleared
    ghosted: jax.Array          # (N,) bool


def step_buffs(*, buff_id, buff_elapsed, buff_duration, buff_power,
               spell_cooldown, spell_level, x, y, kind, team, alive, armor,
               delta_ms: float = 1000.0 / 60.0, slot: int = 0) -> BuffStep:
    """Advance buffs one tick and apply Judgment's area damage.

    The tick boundary follows ``Buff.Update``: elapsed advances first, the
    script's ``OnUpdate`` runs, and the buff deactivates once
    ``TimeElapsed >= Duration``.
    """
    n = x.shape[0]
    dt_s = delta_ms / 1000.0
    active = (buff_id[:, slot] == BuffId.GAREN_E) & alive
    elapsed = jnp.where(active, buff_elapsed[:, slot] + dt_s,
                        buff_elapsed[:, slot])

    # `TimeSinceLastTick >= 500` -- a tick every 500 ms of buff life
    before = jnp.floor(buff_elapsed[:, slot] * 1000.0 / E_TICK_MS)
    after = jnp.floor(elapsed * 1000.0 / E_TICK_MS)
    fires = active & (after > before)

    d2 = (x[None, :] - x[:, None]) ** 2 + (y[None, :] - y[:, None]) ** 2
    hittable = alive & (kind != Kind.TURRET) & (kind != Kind.NONE)
    hit = (
        fires[:, None] & hittable[None, :]
        & (team[None, :] != team[:, None])
        & (d2 <= E_RADIUS * E_RADIUS)
    )
    mult = jnp.where(kind == Kind.LANE_MINION, E_MINION_MULTIPLIER, 1.0)
    raw = buff_power[:, slot][:, None] * mult[None, :]
    dealt = jnp.where(hit, post_mitigation_damage(raw, armor[None, :], jnp),
                      jnp.zeros_like(d2))
    damage = dealt.sum(axis=0)
    any_hit = jnp.any(hit, axis=0)
    dealt_by = jnp.where(any_hit, jnp.argmax(hit, axis=0), -1).astype(jnp.int8)

    expired = active & (elapsed >= buff_duration[:, slot])
    rank = jnp.clip(spell_level[:, Slot.E].astype(jnp.int32), 1,
                    len(E_COOLDOWNS)) - 1
    cd_table = jnp.asarray(E_COOLDOWNS, spell_cooldown.dtype)
    new_cd = spell_cooldown.at[:, Slot.E].set(
        jnp.where(expired, cd_table[rank],
                  jnp.maximum(spell_cooldown[:, Slot.E] - dt_s, 0.0)))

    return BuffStep(
        buff_id=buff_id.at[:, slot].set(
            jnp.where(expired, jnp.int8(BuffId.NONE), buff_id[:, slot])),
        buff_elapsed=buff_elapsed.at[:, slot].set(
            jnp.where(expired, 0.0, elapsed)),
        spell_cooldown=new_cd,
        damage_dealt=damage.astype(x.dtype),
        dealt_by=dealt_by,
        # `SetStatus(CanAttack, false)` and `SetStatus(Ghosted, true)` for the
        # duration -- E suppresses autos and passes through collision.
        suppress_attack=active & ~expired,
        ghosted=active & ~expired,
    )
