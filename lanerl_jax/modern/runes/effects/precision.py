"""Precision tree 8000 (RUNES.md §3). Values from client data (``ea``); level values extrapolate past 18 (U-01).

DMG.40 amps (PTA, Coup de Grace, Cut Down, Last Stand) apply to holder packets on enemy champions of every damage
type, never with PROP_NO_DAMAGE_MOD, TAG_NON_AMPABLE or PROP_SUMMONER (§1.4).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import (CLASS_CHAMPION, CLASS_MINION, PROP_NO_DAMAGE_MOD, PROP_SUMMONER, TAG_ACTIVE_SPELL,
                            TAG_BASIC_ATTACK, TAG_NON_AMPABLE, TAG_ON_HIT, TAG_PET, TAG_PROC, concat_packets, has,
                            packets)
from ...items.catalog import ItemStats
from .core import (BIG, adaptive_damage_type, breakpoints, by_range, ea, effects, first_instance, has_rune,
                   level_table, lin, lin_growth, rune_catalog, rune_item, target_class)

PTA, LETHAL_TEMPO, FLEET, CONQUEROR = 8005, 8008, 8021, 8010
ABSORB_LIFE, TRIUMPH, PRESENCE_OF_MIND = 9101, 9111, 8009
ALACRITY, HASTE, BLOODLINE = 9104, 9105, 9103
COUP_DE_GRACE, CUT_DOWN, LAST_STAND = 8014, 8017, 8299

COVERAGE = {
    PTA: "3 on-hit stacks per target (swap clears); proc + 8% amp until 5 s out of champion combat (INFERRED-M); "
         "damage in the tick the amp is gained is not amped; one stack per attack",
    LETHAL_TEMPO: "stacks on launch vs champions; decay 1 at expiry then 1 per 0.3 s; bolt on attacks launched at "
                  "max stacks after the gain, landing with that attack (PerfectlyTimedForgivance ignored, U-05)",
    FLEET: "energy per attack, ability on-hit instance and distance moved; energized hit heals and hastes",
    CONQUEROR: "melee/ranged basic and per-cast-instance spell stacks (4 s same-spell ring, Ignite once), level "
               "lock (U-03); heals from post-mitigation damage once the 12th stack is reached in packet order "
               "(U-04); cannot tell invulnerable targets from 0-damage hits",
    ABSORB_LIFE: "breakpoint heal per champion/minion/large-monster kill",
    TRIUMPH: "1 s after a champion takedown (even if dead, U-06): max HP + newest missing HP %, gold",
    PRESENCE_OF_MIND: "mana/energy on champion damage (8 s cd); takedown restores % max mana after 1 s",
    ALACRITY: "legend stacks -> bonus AS",
    HASTE: "legend stacks -> basic ability haste",
    BLOODLINE: "legend stacks -> life steal, max HP at max stacks",
    COUP_DE_GRACE: "amp vs champions below 40% HP (tick-start HP)",
    CUT_DOWN: "amp vs champions above 60% HP (tick-start HP)",
    LAST_STAND: "amp vs champions by own missing HP",
}

CONQ_MIN, CONQ_MAX = ea(CONQUEROR, "MinAdaptivePerStack"), ea(CONQUEROR, "MaxAdaptivePerStack")
CONQ_MAX_STACKS = ea(CONQUEROR, "MaxStacks")
CONQ_DURATION = ea(CONQUEROR, "BuffDuration")
CONQ_SAME_SPELL = ea(CONQUEROR, "TimeUntilNextStackFromSameSpell")
CONQ_HEAL, CONQ_HEAL_RANGED = ea(CONQUEROR, "HealingPercent"), ea(CONQUEROR, "RangedHealingPercent")
CONQ_RING = 8                  # cast instances remembered for the 4 s rule
CONQ_MELEE_HIT, CONQ_RANGED_HIT, CONQ_SPELL = 2.0, 1.0, 2.0   # wiki, RUNES §3.1

PTA_MIN, PTA_MAX = ea(PTA, "MinDamage"), ea(PTA, "MaxDamage")
PTA_HITS = ea(PTA, "HitsRequired")
PTA_STACK_TIME = ea(PTA, "TimeBetweenHits")
PTA_AMP = ea(PTA, "BonusPercentDamage")
PTA_OOC = ea(PTA, "OutOfCombatTimer")
PTA_COOLDOWN = ea(PTA, "Cooldown")

_LT = rune_catalog()[LETHAL_TEMPO].calculations
LT_DURATION, LT_MAX = ea(LETHAL_TEMPO, "Duration"), ea(LETHAL_TEMPO, "MaxStacks")
LT_AS = _LT["ASPerStack"]["mFormulaParts"][0]["mNumber"]
LT_AS_RANGED = _LT["ASPerStack"]["mRangedMultiplier"]["mNumber"]
LT_BOLT_MIN = _LT["NoteDamage"]["mFormulaParts"][0]["mStartValue"]
LT_BOLT_MAX = _LT["NoteDamage"]["mFormulaParts"][0]["mEndValue"]
LT_BOLT_RANGED = _LT["NoteDamage"]["mRangedMultiplier"]["mNumber"]
LT_DECAY = 0.3                 # wiki, RUNES §3.3

FLEET_HEAL_MIN, FLEET_HEAL_MAX = ea(FLEET, "HealBase"), ea(FLEET, "HealMax")
FLEET_AD, FLEET_AP = ea(FLEET, "HealBonusADRatio"), ea(FLEET, "HealAPRatio")
FLEET_RANGED_HEAL, FLEET_MINION = ea(FLEET, "RangedHealMod"), ea(FLEET, "MinionHealMod")
FLEET_MS, FLEET_MS_TIME, FLEET_RANGED_MS = ea(FLEET, "MSBuff"), ea(FLEET, "MSDuration"), ea(FLEET, "RangedMSMod")
FLEET_FULL, FLEET_PER_HIT, FLEET_UNITS_PER_CHARGE = 100.0, 6.0, 24.0   # Energized template, RUNES §3.4

_AL = rune_catalog()[ABSORB_LIFE].calculations["HealAmount"]["mFormulaParts"][0]
ABSORB_L1, ABSORB_PER = _AL["mLevel1Value"], _AL["mInitialBonusPerLevel"]
ABSORB_POINTS = tuple((b["mLevel"], b.get("mBonusPerLevelAtAndAfter", 0.0)) for b in _AL["mBreakpoints"])

TRIUMPH_MISSING, TRIUMPH_MAX = ea(TRIUMPH, "MissingHealthRestored"), ea(TRIUMPH, "TriumphMaxHealthRestored")
TRIUMPH_GOLD = ea(TRIUMPH, "BonusGold")
DELAY = 1.0                    # Triumph / PoM takedown restore delay, RUNES §3.6-3.7
QUEUE = 4

POM_TABLE = tuple(rune_catalog()[PRESENCE_OF_MIND].calculations["RegenAmount"]["mFormulaParts"][0]["values"])
POM_CD, POM_ENERGY = ea(PRESENCE_OF_MIND, "CooldownDuration"), ea(PRESENCE_OF_MIND, "EnergyRestore")
POM_TAKEDOWN = ea(PRESENCE_OF_MIND, "PercentManaRestore")
POM_RANGED = 0.8               # wiki, RUNES §3.7 (not in the client calc)

LEGEND = (ALACRITY, HASTE, BLOODLINE)
_hashed = {ea(ALACRITY, k) for k in ("{3e1fdd4a}", "{8fb410e2}", "{f6570baa}")}
assert len(_hashed) == 1, "Legend takedown values differ; map the hashed keys"
LEGEND_TAKEDOWN = _hashed.pop()        # champion and epic takedowns
LEGEND_MINION, LEGEND_LARGE = ea(ALACRITY, "MinionKillValue"), ea(ALACRITY, "LargeMonsterKillValue")
LEGEND_PER_STACK = 100.0               # RUNES §3.8

COUP_BELOW, COUP_AMP = ea(COUP_DE_GRACE, "EnemyHealthPercentageThreshold"), ea(COUP_DE_GRACE, "BonusPercentDamage")
CUT_ABOVE, CUT_AMP = ea(CUT_DOWN, "EnemyHealthPercentageThreshold"), ea(CUT_DOWN, "BonusPercentDamage")
LS_MIN, LS_MAX = ea(LAST_STAND, "MinBonusDamagePercent"), ea(LAST_STAND, "MaxBonusDamagePercent")
LS_START, LS_END = ea(LAST_STAND, "HealthThresholdStart"), ea(LAST_STAND, "HealthThresholdEnd")


class State(NamedTuple):
    conq_stacks: Any        # (C,)
    conq_expire: Any
    conq_level: Any         # level locked when stacks leave 0 (U-03)
    conq_seen_id: Any       # (C, CONQ_RING) int32 cast ids
    conq_seen_t: Any        # (C, CONQ_RING) last stack time per id
    pta_target: Any         # int32 unit carrying the holder's stacks
    pta_stacks: Any
    pta_expire: Any
    pta_cd: Any
    pta_amp: Any            # bool
    pta_amp_since: Any
    lt_stacks: Any          # stacks at the last gain
    lt_expire: Any
    lt_bolt: Any            # bool: launched attack carries a bolt
    lt_bolt_target: Any
    lt_bolt_raw: Any
    lt_bolt_dtype: Any
    fleet_energy: Any
    fleet_armed: Any        # bool: launched attack is Energized
    fleet_ms_until: Any
    tri_due: Any            # (C, QUEUE)
    tri_n: Any
    tri_missing: Any        # newest missing HP at a takedown
    pom_cd: Any
    pom_due: Any            # (C, QUEUE)
    pom_n: Any
    legend_points: Any


def init(n_champions: int, n_units: int) -> State:
    z = jnp.zeros((n_champions,), jnp.float32)
    f, i = jnp.zeros((n_champions,), bool), jnp.zeros((n_champions,), jnp.int32)
    ring = jnp.zeros((n_champions, CONQ_RING), jnp.float32)
    q = jnp.zeros((n_champions, QUEUE), jnp.float32)
    return State(z, z - BIG, z + 1.0, ring.astype(jnp.int32), ring - BIG,
                 i - 1, z, z - BIG, z - BIG, f, z - BIG,
                 z, z - BIG, f, i, z, i,
                 z, f, z - BIG,
                 q + BIG, q, z,
                 z - BIG, q + BIG, q, z)


def _enemy_champion_target(ctx, units, idx):
    i = jnp.clip(idx, 0, units.cls.shape[0] - 1)
    return (idx >= 0) & (units.cls[i] == CLASS_CHAMPION) & (units.team[i] != ctx.team)


def _to_enemy_champions(ctx, units, p):
    """(C, P): holder's packets on enemy champions (dead or alive)."""
    d = jnp.clip(p.dst, 0, units.cls.shape[0] - 1)
    champ = p.valid & (units.cls[d] == CLASS_CHAMPION)
    return champ[None, :] & (p.src[None, :] == ctx.unit[:, None]) & (units.team[d][None, :] != ctx.team[:, None])


def _ampable(ctx, units, p):
    ok = ~has(p.flags, PROP_NO_DAMAGE_MOD) & ~has(p.flags, TAG_NON_AMPABLE) & ~has(p.flags, PROP_SUMMONER)
    return _to_enemy_champions(ctx, units, p) & ok[None, :]


def _lt_current(state: State, now):
    lost = jnp.where(now >= state.lt_expire, 1.0 + jnp.floor((now - state.lt_expire) / LT_DECAY), 0.0)
    return jnp.maximum(state.lt_stacks - lost, 0.0)


def _legend_stacks(points, perk):
    return jnp.minimum(jnp.floor(points / LEGEND_PER_STACK), ea(perk, "MaxLegendStacks"))


def _enqueue(due, cnt, add, at):
    free = due >= BIG / 2
    slot = jnp.argmax(free, axis=1)
    put = (jnp.arange(due.shape[1])[None, :] == slot[:, None]) & ((add > 0) & jnp.any(free, axis=1))[:, None]
    return jnp.where(put, at, due), jnp.where(put, add[:, None], cnt)


def _pop(due, cnt, now):
    fire = due <= now
    return jnp.where(fire, BIG, due), jnp.where(fire, 0.0, cnt), jnp.sum(jnp.where(fire, cnt, 0.0), axis=1)


def stats(state: State, page, ctx, ev) -> ItemStats:
    now = ctx.now
    conq = has_rune(page, CONQUEROR) & (now < state.conq_expire)
    af = jnp.where(conq, state.conq_stacks * lin(CONQ_MIN, CONQ_MAX, state.conq_level), 0.0)
    lt = jnp.where(has_rune(page, LETHAL_TEMPO), _lt_current(state, now) * LT_AS * by_range(ctx, 1.0, LT_AS_RANGED),
                   0.0)
    ms = jnp.where(has_rune(page, FLEET) & (now < state.fleet_ms_until),
                   FLEET_MS * by_range(ctx, 1.0, FLEET_RANGED_MS), 0.0)
    pts = state.legend_points
    a_n, h_n, b_n = (_legend_stacks(pts, p) for p in LEGEND)
    alacrity = jnp.where(has_rune(page, ALACRITY),
                         ea(ALACRITY, "AttackSpeedBase") + ea(ALACRITY, "AttackSpeedPerStack") * a_n, 0.0)
    haste = jnp.where(has_rune(page, HASTE), ea(HASTE, "HasteBase") + ea(HASTE, "HastePerStack") * h_n, 0.0)
    blood = has_rune(page, BLOODLINE)
    ls = jnp.where(blood, (ea(BLOODLINE, "LifeStealBase") + ea(BLOODLINE, "LifeStealPerStack") * b_n) / 100.0, 0.0)
    hp = jnp.where(blood & (b_n >= ea(BLOODLINE, "MaxLegendStacks")), ea(BLOODLINE, "BonusHealth"), 0.0)
    return ItemStats(adaptive_force=af, attack_speed=lt + alacrity, percent_move_speed=ms,
                     basic_ability_haste=haste, life_steal=ls, health=hp)


def on_attack(state: State, page, ctx, units, ev):
    c, n = ctx.level.shape[0], units.x.shape[0]
    a, now = ev.attack, ctx.now
    launched = a.launched & ctx.alive
    # Lethal Tempo.
    lt_go = launched & has_rune(page, LETHAL_TEMPO) & _enemy_champion_target(ctx, units, a.target)
    lt_new = jnp.minimum(_lt_current(state, now) + 1.0, LT_MAX)
    bolt = lt_go & (lt_new >= LT_MAX)
    raw = lin(LT_BOLT_MIN, LT_BOLT_MAX, ctx.level) * (1.0 + ev.bonus_attack_speed) * by_range(ctx, 1.0, LT_BOLT_RANGED)
    # Fleet: an attack launched at full energy is Energized.
    fleet = launched & has_rune(page, FLEET)
    armed = jnp.where(fleet, state.fleet_energy >= FLEET_FULL, state.fleet_armed)
    energy = jnp.where(fleet, jnp.minimum(state.fleet_energy + FLEET_PER_HIT, FLEET_FULL), state.fleet_energy)
    state = state._replace(
        lt_stacks=jnp.where(lt_go, lt_new, state.lt_stacks),
        lt_expire=jnp.where(lt_go, now + LT_DURATION, state.lt_expire),
        lt_bolt=jnp.where(lt_go, bolt, state.lt_bolt),
        lt_bolt_target=jnp.where(bolt, a.target, state.lt_bolt_target),
        lt_bolt_raw=jnp.where(bolt, raw, state.lt_bolt_raw),
        lt_bolt_dtype=jnp.where(bolt, adaptive_damage_type(ev), state.lt_bolt_dtype),
        fleet_energy=energy, fleet_armed=armed)
    return state, effects(c, n)


def on_hit(state: State, page, ctx, units, ev):
    c, n = ctx.level.shape[0], units.x.shape[0]
    a, now = ev.attack, ctx.now
    hit = a.hit & ctx.alive
    tgt = jnp.maximum(a.target, 0)

    # Press the Attack.
    pta_go = hit & has_rune(page, PTA) & _enemy_champion_target(ctx, units, a.target) & (now >= state.pta_cd)
    keep = (state.pta_target == a.target) & (now < state.pta_expire)
    stacks = jnp.where(keep, state.pta_stacks, 0.0) + 1.0
    burst = pta_go & (stacks >= PTA_HITS)
    p_pta = packets(burst, ctx.unit, tgt, lin(PTA_MIN, PTA_MAX, ctx.level), adaptive_damage_type(ev), TAG_PROC,
                    item=rune_item(PTA))

    # Lethal Tempo bolt.
    bolt = hit & state.lt_bolt & has_rune(page, LETHAL_TEMPO)
    p_lt = packets(bolt, ctx.unit, jnp.maximum(state.lt_bolt_target, 0), state.lt_bolt_raw, state.lt_bolt_dtype,
                   TAG_PROC, item=rune_item(LETHAL_TEMPO))

    # Fleet energized hit.
    fleet = hit & state.fleet_armed & has_rune(page, FLEET)
    minion = target_class(units, a.target) == CLASS_MINION
    heal = (lin_growth(FLEET_HEAL_MIN, FLEET_HEAL_MAX, ctx.level) + FLEET_AD * ev.bonus_ad + FLEET_AP * ev.ap) \
        * by_range(ctx, 1.0, FLEET_RANGED_HEAL) * jnp.where(minion, FLEET_MINION, 1.0)

    state = state._replace(
        pta_target=jnp.where(pta_go, a.target, state.pta_target),
        pta_stacks=jnp.where(pta_go, jnp.where(burst, 0.0, stacks), state.pta_stacks),
        pta_expire=jnp.where(pta_go, now + PTA_STACK_TIME, state.pta_expire),
        pta_cd=jnp.where(burst, now + PTA_COOLDOWN, state.pta_cd),
        pta_amp=state.pta_amp | burst,
        pta_amp_since=jnp.where(burst, now, state.pta_amp_since),
        lt_bolt=jnp.where(hit, False, state.lt_bolt),
        fleet_armed=jnp.where(fleet, False, state.fleet_armed),
        fleet_energy=jnp.where(fleet, 0.0, state.fleet_energy),
        fleet_ms_until=jnp.where(fleet, now + FLEET_MS_TIME, state.fleet_ms_until))
    return state, effects(c, n, packets=concat_packets(p_pta, p_lt), heal=jnp.where(fleet, heal, 0.0))


def periodic(state: State, page, ctx, units, ev):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    conq_end = now >= state.conq_expire
    pta_live = now - jnp.maximum(ev.clocks.last_champion_combat, state.pta_amp_since) < PTA_OOC
    move = has_rune(page, FLEET) & ctx.alive
    energy = jnp.where(move, jnp.minimum(state.fleet_energy + ctx.moved / FLEET_UNITS_PER_CHARGE, FLEET_FULL),
                       state.fleet_energy)
    tri_due, tri_n, tri = _pop(state.tri_due, state.tri_n, now)
    tri = jnp.where(has_rune(page, TRIUMPH), tri, 0.0)
    tri_heal = tri * (TRIUMPH_MAX * ctx.max_hp + TRIUMPH_MISSING * state.tri_missing)
    pom_due, pom_n, pom = _pop(state.pom_due, state.pom_n, now)
    pom_mana = jnp.where(has_rune(page, PRESENCE_OF_MIND), pom * POM_TAKEDOWN * ctx.max_mana, 0.0)
    state = state._replace(
        conq_stacks=jnp.where(conq_end, 0.0, state.conq_stacks),
        pta_stacks=jnp.where(now >= state.pta_expire, 0.0, state.pta_stacks),
        pta_amp=state.pta_amp & pta_live,
        fleet_energy=energy, tri_due=tri_due, tri_n=tri_n, pom_due=pom_due, pom_n=pom_n)
    return state, effects(c, n, heal=tri_heal, gold=tri * TRIUMPH_GOLD, mana=pom_mana)


def packet_amp(state: State, page, ctx, units, ev, p):
    sel = _ampable(ctx, units, p)
    d = jnp.clip(p.dst, 0, units.cls.shape[0] - 1)
    frac = (units.hp[d] / jnp.maximum(units.max_hp[d], 1.0))[None, :]
    own = ctx.hp / jnp.maximum(ctx.max_hp, 1.0)
    pta_on = state.pta_amp & (ctx.now > state.pta_amp_since) \
        & (ctx.now - jnp.maximum(ev.clocks.last_champion_combat, state.pta_amp_since) < PTA_OOC)
    ls = jnp.where(own < LS_START,
                   LS_MIN + (LS_MAX - LS_MIN) * jnp.clip((LS_START - own) / (LS_START - LS_END), 0.0, 1.0), 0.0)
    amp = jnp.where((has_rune(page, PTA) & pta_on)[:, None], PTA_AMP, 0.0) \
        + jnp.where(has_rune(page, COUP_DE_GRACE)[:, None] & (frac < COUP_BELOW), COUP_AMP, 0.0) \
        + jnp.where(has_rune(page, CUT_DOWN)[:, None] & (frac > CUT_ABOVE), CUT_AMP, 0.0) \
        + jnp.where(has_rune(page, LAST_STAND), ls, 0.0)[:, None]
    return jnp.sum(jnp.where(sel, amp, 0.0), axis=0).astype(jnp.float32)


def _conqueror(state: State, page, ctx, units, rep):
    p, r = rep.packets, rep.resolved
    now = ctx.now
    sel = _to_enemy_champions(ctx, units, p) & (has_rune(page, CONQUEROR) & ctx.alive)[:, None]
    proc = has(p.flags, TAG_PROC) & ~has(p.flags, TAG_PET)
    basic = has(p.flags, TAG_BASIC_ATTACK) & ~proc
    spell = sel & (~basic & ~proc)[None, :]
    cid = p.cast_id
    P = cid.shape[0]
    earlier = jnp.arange(P)[None, :] < jnp.arange(P)[:, None]                     # [p, q]: q before p
    same = (cid[:, None] == cid[None, :]) & (cid[:, None] != 0) & earlier
    first = spell & ~(jnp.einsum("pq,cq->cp", same.astype(jnp.float32), spell.astype(jnp.float32)) > 0.0)
    recent = (now - state.conq_seen_t < CONQ_SAME_SPELL)[:, None, :] | has(p.flags, PROP_SUMMONER)[None, :, None]
    seen = jnp.any((state.conq_seen_id[:, None, :] == cid[None, :, None]) & recent, axis=2) & (cid != 0)[None, :]
    gain_spell = first & ~seen
    gain_p = jnp.where(sel & basic[None, :], by_range(ctx, CONQ_MELEE_HIT, CONQ_RANGED_HIT)[:, None], 0.0) \
        + jnp.where(gain_spell, CONQ_SPELL, 0.0)
    stacks0 = jnp.where(now < state.conq_expire, state.conq_stacks, 0.0)
    cum = jnp.minimum(stacks0[:, None] + jnp.cumsum(gain_p, axis=1), CONQ_MAX_STACKS)
    total = jnp.sum(gain_p, axis=1)
    new = jnp.minimum(stacks0 + total, CONQ_MAX_STACKS)
    refresh = jnp.any(sel & ~proc[None, :], axis=1) & (new > 0)
    heal = jnp.sum(jnp.where(sel & (cum >= CONQ_MAX_STACKS), r.final[None, :], 0.0), axis=1) \
        * by_range(ctx, CONQ_HEAL, CONQ_HEAL_RANGED)
    # Write the r-th new instance into the r-th oldest ring slot.
    ring = gain_spell & (cid != 0)[None, :]
    rank = jnp.cumsum(ring, axis=1) - 1
    order = jnp.argsort(state.conq_seen_t, axis=1)
    ids, ts = state.conq_seen_id, state.conq_seen_t
    for k in range(CONQ_RING):
        pick = ring & (rank == k)
        put = (jnp.arange(CONQ_RING)[None, :] == order[:, k:k + 1]) & jnp.any(pick, axis=1)[:, None]
        ids = jnp.where(put, jnp.sum(jnp.where(pick, cid[None, :], 0), axis=1)[:, None].astype(ids.dtype), ids)
        ts = jnp.where(put, now, ts)
    state = state._replace(
        conq_stacks=new, conq_expire=jnp.where(refresh, now + CONQ_DURATION, state.conq_expire),
        conq_level=jnp.where((stacks0 == 0) & (total > 0), ctx.level, state.conq_level),
        conq_seen_id=ids, conq_seen_t=ts.astype(jnp.float32))
    return state, heal


def on_damage(state: State, page, ctx, units, ev):
    c, n = ctx.level.shape[0], units.x.shape[0]
    rep = ev.report
    p = rep.packets
    state, heal = _conqueror(state, page, ctx, units, rep)
    # Presence of Mind.
    hit = jnp.any(_to_enemy_champions(ctx, units, p), axis=1)
    pom = hit & has_rune(page, PRESENCE_OF_MIND) & ctx.alive & (ctx.now >= state.pom_cd)
    restore = jnp.where(ev.uses_energy, POM_ENERGY,
                        level_table(POM_TABLE, ctx.level) * by_range(ctx, 1.0, POM_RANGED))
    # Fleet: per ability instance that applies on-hit.
    own = p.valid[None, :] & (p.src[None, :] == ctx.unit[:, None]) \
        & (has(p.flags, TAG_ACTIVE_SPELL) & has(p.flags, TAG_ON_HIT))[None, :]
    k = jnp.sum(first_instance(p, own), axis=1).astype(jnp.float32)
    fleet = has_rune(page, FLEET) & ctx.alive
    state = state._replace(
        pom_cd=jnp.where(pom, ctx.now + POM_CD, state.pom_cd),
        fleet_energy=jnp.where(fleet, jnp.minimum(state.fleet_energy + FLEET_PER_HIT * k, FLEET_FULL),
                               state.fleet_energy))
    return state, effects(c, n, heal=heal, mana=jnp.where(pom, restore, 0.0))


def on_takedown(state: State, page, ctx, units, ev):
    c, n = ctx.level.shape[0], units.x.shape[0]
    kl, now = ev.kills, ctx.now
    takedowns = (kl.champion_kill + kl.champion_assist).astype(jnp.float32)
    kills = (kl.champion_kill + kl.minion_kill + ev.large_monster_kill).astype(jnp.float32)
    absorb = jnp.where(has_rune(page, ABSORB_LIFE),
                       kills * breakpoints(ABSORB_L1, ABSORB_PER, ABSORB_POINTS, ctx.level), 0.0)
    tri = jnp.where(has_rune(page, TRIUMPH), takedowns, 0.0)
    tri_due, tri_n = _enqueue(state.tri_due, state.tri_n, tri, now + DELAY)
    pom = jnp.where(has_rune(page, PRESENCE_OF_MIND), takedowns, 0.0)
    pom_due, pom_n = _enqueue(state.pom_due, state.pom_n, pom, now + DELAY)
    legend = has_rune(page, ALACRITY) | has_rune(page, HASTE) | has_rune(page, BLOODLINE)
    pts = LEGEND_TAKEDOWN * (takedowns + ev.epic_takedown) + LEGEND_LARGE * ev.large_monster_kill \
        + LEGEND_MINION * kl.minion_kill
    state = state._replace(
        tri_due=tri_due, tri_n=tri_n,
        tri_missing=jnp.where(tri > 0, jnp.maximum(ctx.max_hp - ctx.hp, 0.0), state.tri_missing),
        pom_due=pom_due, pom_n=pom_n,
        legend_points=jnp.where(legend, state.legend_points + pts, state.legend_points).astype(jnp.float32))
    return state, effects(c, n, heal=absorb)
