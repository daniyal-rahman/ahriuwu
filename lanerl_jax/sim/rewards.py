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
           "level_for_xp", "ambient_gold",
           "HIT_FLAG_MS", "CHAMPION_BASE_GOLD", "CHAMPION_MAX_GOLD",
           "CHAMPION_MIN_GOLD", "FIRST_BLOOD_EXTRA_GOLD",
           "LEVEL_DIFFERENCE_EXP_MULTIPLE", "MINIMUM_EXP_MULTIPLE",
           "GOLD_FROM_MINIONS_THRESHOLD", "TURRET_RANGE_MULTIPLE",
           "update_hit_flag", "ChampionKillRewards", "champion_kill_rewards",
           "turret_kill_rewards", "minion_gold_deathspree_decay"]

#: ``Champion._championHitFlagTimer = 15 * 1000`` -- ``Champion.cs:573``. Reset
#: on every ``TakeDamage`` this champion takes, from ANY source (no exemption
#: for ordinary minion damage, unlike ``ms_since_damaged``/the passive-regen
#: interrupt in ``step.py``, which is a completely different mechanic).
HIT_FLAG_MS = 15_000.0

#: ``MapScriptMetadata`` defaults (``GameServerLib/Scripting/CSharp/
#: MapScriptMetadata.cs:5-9``), confirmed unoverridden by Map1's own
#: ``LevelScript`` (`Content/LeagueSandbox-Scripts/Maps/Map1/LevelScript.cs`
#: only ever touches `MinionSpawnEnabled` on this object) -- these really are
#: the numbers Map1 runs, not merely the compiled defaults.
CHAMPION_BASE_GOLD = 300.0
CHAMPION_MAX_GOLD = 500.0
CHAMPION_MIN_GOLD = 50.0
FIRST_BLOOD_EXTRA_GOLD = 100.0

#: Map1's ``ExpCurve.json``, ``Values.ExpGrantedOnDeath`` block. The third
#: field, ``BaseExpMultiple`` (0.55), is baked directly into
#: ``profiles.build_profile_tables``'s ``champion_kill_exp`` table rather than
#: kept here, since it multiplies a per-level table rather than standing alone.
LEVEL_DIFFERENCE_EXP_MULTIPLE = 0.08
MINIMUM_EXP_MULTIPLE = 0.15

#: ``Champion.OnKill`` (`Champion.cs:384`): crossing this many cumulative
#: minion-gold-while-on-a-death-spree knocks one stack off `DeathSpree`.
GOLD_FROM_MINIONS_THRESHOLD = 1000.0

#: ``LaneTurret.Die`` (`LaneTurret.cs:44`): ``Stats.Range.Total * 1.5f``.
TURRET_RANGE_MULTIPLE = 1.5

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

    Restricted to non-champion victims: this is ``AttackableUnit.Die``'s own
    proximity-XP block (`AttackableUnit.cs:646-660`), and ``Champion`` overrides
    ``Die`` entirely and never calls ``base.Die`` -- so this path never executes
    for a champion death on the server at all. See
    :func:`champion_kill_rewards` for what actually pays out when a champion
    dies. (Numerically a no-op either way in this project: Garen.json carries
    no ``ExpGivenOnDeath`` key, so a champion's own ``xp_on_death`` is already
    0 -- excluded explicitly anyway so a future content change can't silently
    reopen this path.)
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
        died[:, None] & ~is_champ[:, None] & is_champ[None, :] & alive[None, :]
        & (team[None, :] != team[:, None])
        & (d2 <= exp_radius * exp_radius)
    )
    share_count = jnp.sum(eligible, axis=1)
    per = jnp.where(share_count > 0, xp_on_death / jnp.maximum(share_count, 1),
                    jnp.zeros_like(xp_on_death))
    xp = jnp.sum(jnp.where(eligible, per[:, None], jnp.zeros_like(d2)), axis=0)

    return DeathRewards(gold=gold, xp=xp.astype(x.dtype), cs=cs)


def update_hit_flag(*, kind: jax.Array, damage_ij: jax.Array,
                    buff_damage: jax.Array, buff_dealt_by: jax.Array,
                    hit_flag_ms: jax.Array, hit_flag_by: jax.Array,
                    delta_ms: float):
    """One tick of ``Champion._championHitFlagTimer``/``_playerHitId``.

    ``Champion.TakeDamage`` (`Champion.cs:569-575`) resets the timer to 15000
    and records the attacker's id on **every** hit, unconditionally -- no
    source filter, unlike the passive-regen combat gate. The decrement itself
    is `Champion.Update`'s own block (`Champion.cs:267-273`), unconditional
    every tick. Modelled with the same reset-beats-decay shape already used for
    `ms_since_damaged` (`step.py`), since the same cross-unit-ordering
    approximation applies to both: which of several same-tick hits is "last"
    depends on the server's serial `ObjectManager` iteration order, which this
    vectorized sim does not reproduce (see `docs/PORT_AUDIT_COMBAT.md`'s tick
    order note). Where more than one attacker lands on the same champion in the
    same tick, this picks the **highest unit index** among them as the
    tie-break -- an explicit, documented choice, not a guess -- with Judgment's
    (buff-row) damage treated as if it landed before every ordinary attacker's,
    matching `step.py`'s own note that Judgment's damage is applied earlier in
    the tick than the auto-attack/missile resolution it is prepended to for
    kill attribution.

    ``damage_ij`` is ``(attacker, victim)``, already the real (non-Judgment)
    hits for this tick (``step.py``'s ``dmg_ij``, post-mitigation). Only
    champions carry a meaningful hit-flag on the server; this returns updated
    arrays for every unit, which is harmless since nothing reads them for a
    non-champion.
    """
    n = kind.shape[0]
    idx = jnp.arange(n)
    hit_mask = damage_ij > 0
    last_ordinary = jnp.max(jnp.where(hit_mask, idx[:, None], -1), axis=0)
    any_ordinary = last_ordinary >= 0
    last_attacker = jnp.where(
        any_ordinary, last_ordinary,
        jnp.where(buff_damage > 0, buff_dealt_by.astype(idx.dtype), -1))
    any_hit = any_ordinary | (buff_damage > 0)

    new_timer = jnp.where(
        any_hit, jnp.asarray(HIT_FLAG_MS, hit_flag_ms.dtype),
        jnp.maximum(hit_flag_ms - delta_ms, 0.0))
    new_by = jnp.where(any_hit, last_attacker.astype(hit_flag_by.dtype),
                       hit_flag_by)
    return new_timer, new_by


class ChampionKillRewards(NamedTuple):
    gold: Any            # (N,) gold paid to whoever gets credit
    xp: Any              # (N,) XP paid to whoever gets credit
    kills: Any           # (N,) int16, +1 to whoever gets credit
    kill_spree: Any      # (N,) updated KillSpree
    death_spree: Any     # (N,) updated DeathSpree
    gold_from_minions: Any  # (N,) updated GoldFromMinions
    first_blood_done: Any   # () updated flag


def champion_kill_rewards(*, died: jax.Array, kind: jax.Array,
                          level: jax.Array, killer: jax.Array,
                          hit_flag_ms: jax.Array, hit_flag_by: jax.Array,
                          kill_spree: jax.Array, death_spree: jax.Array,
                          gold_from_minions: jax.Array,
                          first_blood_done: jax.Array,
                          kill_exp_table: jax.Array) -> ChampionKillRewards:
    """``Champion.Die`` (`Champion.cs:392-461`) -- the champion-kill formula.

    Not modelled by :func:`death_rewards`: ``Champion`` overrides ``Die``
    entirely (confirmed: it never calls ``base.Die``), so this is the *only*
    path that pays anything for a champion death.

    ``cKiller`` resolution (`Champion.cs:403-416`): the actual killing blow's
    attacker if it was a champion, else -- **only** if this champion was hit by
    a champion within the last 15 s (`hit_flag_ms > 0`, and only if THAT hit
    was itself from a champion, since `_playerHitId` is overwritten by every
    hit regardless of source) -- that champion instead. If neither resolves,
    the method returns having paid nothing and, critically, **without touching
    `KillSpree`/`DeathSpree` at all** (`:411-416`) -- a champion killed by a
    minion/turret/monster it was never recently hit by feeds nobody and its own
    spree counters freeze.

    The gold formula's `KillSpree`/`DeathSpree` (`:421-436`) are the **victim's
    own**, not the killer's -- `Die()` runs on the victim (`this`), so bare
    references inside it mean `this.KillSpree`/`this.DeathSpree`. This is
    League's shutdown/feeding-discount mechanic: killing a champion who is
    themselves on a kill streak (`KillSpree>1`) is worth a streak-scaled bonus;
    killing one already on a death spree (`KillSpree==0 && DeathSpree>=1`) is
    worth a discount that compounds with repeat deaths. `KillSpree==1` exactly
    hits neither branch (`>1` is false, `==0` is false) -- flat base gold,
    regardless of `DeathSpree` -- reproduced literally, not smoothed over.

    `DeathSpree` is incremented **twice** in the feeding-discount branch: once
    inside it (`:434`, using the OLD value for the `Math.Pow` above it) and
    once more, unconditionally, at the very end (`:496`) -- a real bug in the
    server (confirmed by reading both lines independently), reproduced here as
    `death_spree + 2` for that branch and `+ 1` for every other paid death.

    First blood (`:437-441`) is a MAP-level flag, checked and (if unset) both
    awarded and latched in the same statement, regardless of which branch
    above fired.

    EXP (`:444-454`) uses champion LEVEL, not spree: `ExpCurve[victim.Level-1]
    * BaseExpMultiple`, adjusted up to +-`min(0.08*|levelDiff|, 0.15)` of itself
    toward the under-levelled side. `kill_exp_table` is
    `profiles.build_profile_tables()["champion_kill_exp"]`, already carrying
    the `* BaseExpMultiple` and the `ExpCurve[Level-1] == xp_for_level(Level+1)`
    index shift -- see that table's own docstring.

    Killer-side bookkeeping (`:487-496`, `cKiller.*`) is scattered onto
    whichever unit `cKiller` resolves to; victim-side bookkeeping is written to
    the victim's own row. For this project's fixed 2-champion lane the two
    are always disjoint indices whenever both could ever collide (a champion
    can only ever be credited as the OTHER champion's killer), so the
    scatter-add/scatter-max below is exact, not merely an approximation, for
    every reachable configuration. The one genuinely rare case -- a champion
    that is simultaneously credited as a killer (of the other) AND dies itself
    this same tick (to a third source) -- resolves killer-credit-wins for that
    unit's own `death_spree`/`gold_from_minions` (the killer-side reset is
    applied after the victim-side write below); this is the same class of
    cross-unit tick-ordering approximation already accepted elsewhere in this
    project, not a new kind of imprecision.
    """
    n = kind.shape[0]
    dtype = kill_exp_table.dtype
    is_champ = kind == Kind.CHAMPION
    died_champ = died & is_champ

    k = jnp.clip(killer, 0, n - 1).astype(jnp.int32)
    direct_ok = (killer >= 0) & is_champ[k]
    hf = jnp.clip(hit_flag_by, 0, n - 1).astype(jnp.int32)
    fallback_ok = (~direct_ok) & (hit_flag_ms > 0) & (hit_flag_by >= 0) \
        & is_champ[hf]
    c_killer = jnp.where(direct_ok, killer,
                         jnp.where(fallback_ok, hit_flag_by, jnp.int8(-1)))
    paid = died_champ & (c_killer >= 0)
    ck = jnp.clip(c_killer, 0, n - 1).astype(jnp.int32)

    kswin = kill_spree.astype(dtype)
    dswin = death_spree.astype(dtype)

    streak_gold = jnp.minimum(
        CHAMPION_BASE_GOLD * jnp.power(7.0 / 6.0, jnp.maximum(kswin - 1.0, 0.0)),
        CHAMPION_MAX_GOLD)
    half = jnp.floor_divide(jnp.maximum(death_spree, 0), 2).astype(dtype)
    feed_gold = jnp.asarray(CHAMPION_BASE_GOLD * (11.0 / 12.0), dtype)
    feed_gold = jnp.where(
        dswin > 1.0,
        jnp.maximum(feed_gold * jnp.power(0.8, half), CHAMPION_MIN_GOLD),
        feed_gold)

    is_streak = kswin > 1.0
    is_feed = (kswin == 0.0) & (dswin >= 1.0)
    gold = jnp.where(is_streak, streak_gold,
                     jnp.where(is_feed, feed_gold,
                               jnp.asarray(CHAMPION_BASE_GOLD, dtype)))

    fb_award = paid & (~first_blood_done)
    gold = jnp.where(fb_award, gold + FIRST_BLOOD_EXTRA_GOLD, gold)

    victim_level_i = jnp.clip(level.astype(jnp.int32), 1, kill_exp_table.shape[0] - 1)
    exp_base = kill_exp_table[victim_level_i]
    killer_level = level[ck]
    ldiff = jnp.abs(killer_level.astype(dtype) - level.astype(dtype))
    exp_mult = jnp.minimum(LEVEL_DIFFERENCE_EXP_MULTIPLE * ldiff,
                           MINIMUM_EXP_MULTIPLE)
    exp_adj = exp_base * exp_mult
    exp_adj = jnp.where(killer_level > level, -exp_adj, exp_adj)
    exp = jnp.where(killer_level != level, exp_base + exp_adj, exp_base)

    gold_out = jnp.zeros((n,), dtype).at[ck].add(jnp.where(paid, gold, 0.0))
    xp_out = jnp.zeros((n,), dtype).at[ck].add(jnp.where(paid, exp, 0.0))
    kills_out = jnp.zeros((n,), jnp.int16).at[ck].add(
        jnp.where(paid, 1, 0).astype(jnp.int16))

    # victim-side: `KillSpree = 0` always on a paid death; `DeathSpree` +1, or
    # +2 in the feeding-discount branch (see docstring).
    new_kill_spree = jnp.where(paid, jnp.zeros_like(kill_spree), kill_spree)
    new_death_spree = jnp.where(
        paid & is_feed, death_spree + 2,
        jnp.where(paid, death_spree + 1, death_spree))

    # killer-side: `KillSpree++`, `DeathSpree = 0`, `GoldFromMinions = 0`.
    new_kill_spree = new_kill_spree.at[ck].add(jnp.where(paid, 1, 0))
    killer_credited = jnp.zeros((n,), bool).at[ck].max(paid)
    new_death_spree = jnp.where(killer_credited, 0, new_death_spree)
    new_gold_from_minions = jnp.where(killer_credited, 0.0, gold_from_minions)

    new_first_blood = first_blood_done | jnp.any(fb_award)

    return ChampionKillRewards(
        gold=gold_out, xp=xp_out.astype(dtype), kills=kills_out,
        kill_spree=new_kill_spree, death_spree=new_death_spree,
        gold_from_minions=new_gold_from_minions,
        first_blood_done=new_first_blood)


def turret_kill_rewards(*, died: jax.Array, kind: jax.Array, team: jax.Array,
                        alive: jax.Array, x: jax.Array, y: jax.Array,
                        local_gold: jax.Array, global_gold: jax.Array,
                        global_xp: jax.Array, attack_range: jax.Array):
    """``LaneTurret.Die`` (`LaneTurret.cs:37-88`) -- turret destruction reward.

    ``championsInRange = GetChampionsInRange(Position, Range*1.5, onlyAlive=
    true)`` (`:44`) is **not team-filtered** -- it can include the turret's own
    team's champion, who is then skipped by the `continue` inside the loop but
    still counted toward the split's denominator, diluting the enemy's local
    share. Reproduced exactly (`count` below sums BOTH teams' alive champions
    in range, not just enemies).

    If `LocalGoldGivenOnDeath <= 0` OR no champion (either team) is in range at
    the moment of death, every enemy champion instead gets exactly
    `GlobalGoldGivenOnDeath` (no local share at all) -- true unconditionally
    for the INHIBITOR/NEXUS tiers on this map, whose `LocalGoldGivenOnDeath` is
    0 in Content. Otherwise, an in-range enemy gets `local/count + global`, and
    an out-of-range enemy gets `global` alone. Every enemy champion gets
    `GlobalExpGivenOnDeath`, unconditionally, regardless of range.

    Distance is measured from the turret's position at the moment of death
    against `alive` as of this tick's own resolution (post this-tick's damage,
    same convention `death_rewards`'s XP-share already uses) -- consistent
    rather than independently re-derived.
    """
    n = kind.shape[0]
    dtype = x.dtype
    is_turret = kind == Kind.TURRET
    is_champ = kind == Kind.CHAMPION
    died_turret = died & is_turret

    d2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    rng2 = (TURRET_RANGE_MULTIPLE * attack_range) ** 2
    in_range = (d2 <= rng2[:, None]) & alive[None, :] & is_champ[None, :]
    count = jnp.sum(in_range, axis=1)

    enemy = team[None, :] != team[:, None]
    use_local = died_turret & (local_gold > 0) & (count > 0)
    local_share = jnp.where(count > 0, local_gold / jnp.maximum(count, 1),
                            jnp.zeros_like(local_gold))

    gold_tc = jnp.where(
        died_turret[:, None] & enemy,
        jnp.where(use_local[:, None] & in_range,
                 local_share[:, None] + global_gold[:, None],
                 global_gold[:, None]),
        jnp.zeros_like(d2))
    xp_tc = jnp.where(died_turret[:, None] & enemy, global_xp[:, None],
                      jnp.zeros_like(d2))

    gold_out = jnp.sum(gold_tc, axis=0).astype(dtype)
    xp_out = jnp.sum(xp_tc, axis=0).astype(dtype)
    return gold_out, xp_out


def minion_gold_deathspree_decay(*, minion_gold: jax.Array,
                                 death_spree: jax.Array,
                                 gold_from_minions: jax.Array):
    """``Champion.OnKill``'s ``GoldFromMinions``/``DeathSpree`` decay
    (`Champion.cs:379-388`) -- the farming-catch-up half of the death-spree
    state machine :func:`champion_kill_rewards` reads. Accumulates
    ``minion_gold`` (this tick's minion-kill gold -- exactly
    :func:`death_rewards`'s ``gold`` output, which has no other source) into
    ``GoldFromMinions`` only while ``DeathSpree > 0`` (the incoming value, read
    before this tick's own mutation, matching the C# reading its own field
    before writing it), then knocks exactly **one** stack off ``DeathSpree``
    per tick if the accumulator has crossed 1000 -- an ``if``, not a ``while``
    (`:384-388`), so gold from several minions in one tick that would cross
    1000 more than once still only pays off one stack, and the excess rolls
    over rather than triggering twice. No clamp to 0 on the decrement
    (`:387`, a bare ``DeathSpree -= 1``) -- not reachable in practice, since
    ``GoldFromMinions`` only ever grows while ``DeathSpree >= 1`` and is zeroed
    the instant its owner lands a champion kill, but reproduced literally
    rather than defensively clamped.
    """
    accum = jnp.where(death_spree > 0, gold_from_minions + minion_gold,
                      gold_from_minions)
    crossed = accum >= GOLD_FROM_MINIONS_THRESHOLD
    new_gold_from_minions = jnp.where(
        crossed, accum - GOLD_FROM_MINIONS_THRESHOLD, accum)
    new_death_spree = jnp.where(crossed, death_spree - 1, death_spree)
    return new_gold_from_minions, new_death_spree


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
