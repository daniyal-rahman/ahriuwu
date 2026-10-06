"""Inspiration tree 8300 (RUNES.md §7). Values from client data (``ea``) and the item catalog.

The world owns summoner-spell effects (Flash, the Hexflash blink, swapped-in spells) and inventory placement;
this module owns the runes' timing and state machines and reports through ``outputs``. Item grants go through a
per-holder queue: ``outputs.grant_item`` repeats the head until ``ev.granted`` acknowledges it. Hexflash and
Approach Velocity run in ``post_tick`` because they read this tick's clocks and impairments.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ...core.damage import CLASS_CHAMPION, CLASS_STRUCTURE, TAG_INDIRECT, TAG_PROC, TRUE, packets
from ...items.catalog import DATA_PATH, STAT_INDEX, ItemStats, catalog
from .core import (BIG, Effects, RuneEvents, RuneOutputs, ea, effects, enemy_champions, has_rune, lin, no_outputs,
                   rune_item)

GLACIAL, SPELLBOOK, FIRST_STRIKE = 8351, 8360, 8369
FLASHTRAPTION, FOOTWEAR, CASH_BACK = 8306, 8304, 8321
TRIPLE_TONIC, TIME_WARP, BISCUITS = 8313, 8352, 8345
COSMIC, APPROACH, JACK = 8347, 8410, 8316

BISCUIT_ITEM, BOOTS_ITEM = 2010, 2422
AVARICE, FORCE, SKILL = 2151, 2152, 2150
HEALTH_POTION, REFILLABLE = 2003, 2031

COVERAGE = {
    GLACIAL: "immobilizing a champion -> slowing ray zones (spare rays fan, INFERRED-L); enemies inside deal less "
             "damage to the holder's allied champions (never the holder)",
    SPELLBOOK: "swap availability: first time, unique-scaled cooldown, out of combat, no repeat of recent picks; "
               "the swap itself and its slot cooldowns are world-owned (U-16)",
    FIRST_STRIKE: "struck first early in a champion-combat episode -> gold + a buff turning % of post-mitigation "
                  "champion damage into delayed true packets, paid back partly as gold at the end; struck first "
                  "by a champion -> full cooldown (clamped at L18)",
    FLASHTRAPTION: "Hexflash channel while Flash is on cooldown: blink scaled by channel time, MS; champion combat "
                   "or early release -> combat cooldown; cooldowns scaled by summoner haste",
    FOOTWEAR: "boots granted at a takedown-reduced time (queued if the inventory is full); Boots purchases "
              "forbidden until then; flat MS while owning Boots",
    CASH_BACK: "Legendary purchases refund a % of cost; selling a refunded unit takes it back",
    TRIPLE_TONIC: "elixirs at three levels via the grant queue; Elixir of Skill auto-consumed if inventory is full",
    TIME_WARP: "Health / Refillable Potion heals a % of its total instantly (biscuits excluded, U-17)",
    BISCUITS: "biscuits at fixed minutes via the grant queue; selling one still gives its silent max HP",
    COSMIC: "summoner haste and item haste",
    APPROACH: "MS facing enemy champions the holder impairs, half facing visible impaired ones in range",
    JACK: "AH per unique eligible item stat type (incl. gold generation), AF at 5 and 10; item-effect stats not "
          "counted",
}

GA_RAYS = int(ea(GLACIAL, "RayCount"))
GA_LENGTH = ea(GLACIAL, "SlowZoneLength")
GA_WIDTH = ea(GLACIAL, "SlowZoneWidth")
GA_DURATION = ea(GLACIAL, "SlowZoneDuration")
GA_CC_CARRY = ea(GLACIAL, "CCCarryOverRatio") / 100.0
GA_COOLDOWN = ea(GLACIAL, "Cooldown")
GA_REDUCTION = ea(GLACIAL, "DmgReduction")
GA_INDENT = ea(GLACIAL, "BeamPosIndent")          # zone starts behind the target (INFERRED-L)
GA_SLOW_BASE = ea(GLACIAL, "SlowZoneSlowBase") / 100.0
GA_SLOW_BAD = ea(GLACIAL, "SlowZoneSlowbADRatio") / 100.0      # per bonus AD
GA_SLOW_AP = ea(GLACIAL, "SlowZoneSlowAPRatio") / 100.0        # per AP
GA_SLOW_HSP = ea(GLACIAL, "SlowZoneSlowHealShieldRatio") / 100.0  # per 1.0 HSP
GA_ALLY_RANGE = 1000.0      # INFERRED-L
GA_FAN = np.deg2rad(30.0)   # rays without an ally to aim at (INFERRED-L)

# Unsealed Spellbook (wiki T:3960876).
SB_FIRST = ea(SPELLBOOK, "ShardFirstMinutes") * 60.0
SB_BASE = ea(SPELLBOOK, "ShardRechargeMinutes") * 60.0
SB_PER_UNIQUE = ea(SPELLBOOK, "ShardRechargeReductionSeconds")
SB_CAP_SWAPS = ea(SPELLBOOK, "{0bb7b933}")         # 6 (wiki cdcap)
SB_MIN = SB_BASE - SB_PER_UNIQUE * SB_CAP_SWAPS    # 120 s
SB_NO_REPEAT = int(ea(SPELLBOOK, "NumSummonersBeforeRepeat"))
SB_OOC = ea(SPELLBOOK, "{b7e0131f}")               # 5 s out of combat

FS_GRACE = ea(FIRST_STRIKE, "GraceWindow")
FS_DURATION = ea(FIRST_STRIKE, "Duration")
FS_AMP = ea(FIRST_STRIKE, "DamageAmp")
FS_GOLD_FLAT = ea(FIRST_STRIKE, "GoldProcBonus")
FS_GOLD_MELEE = ea(FIRST_STRIKE, "GoldPercentBonus")
FS_GOLD_RANGED = ea(FIRST_STRIKE, "GoldPercentBonusRanged")
FS_CD_START = ea(FIRST_STRIKE, "TooltipOnlyCooldownStartAmount")
FS_CD_END = ea(FIRST_STRIKE, "TooltipOnlyCooldownEndAmount")
FS_MODE_CD = ea(FIRST_STRIKE, "ModesCooldownReduction")
FS_DELAY = 0.4              # wiki: missile travel
FS_SLOTS = 24               # delay-queue slots (>= FS_DELAY of FS_PUSH per pass at 30 Hz)
FS_PUSH = 2                 # distinct enemy champions queued per damage pass
FS = rune_item(FIRST_STRIKE)

# Hextech Flashtraption (SUMMONER_SPELLS.md §11).
HX_CHANNEL = ea(FLASHTRAPTION, "ChannelDuration")
HX_MIN = ea(FLASHTRAPTION, "MinimumChannelDuration")
HX_COOLDOWN = ea(FLASHTRAPTION, "CooldownTime")
HX_COMBAT_CD = ea(FLASHTRAPTION, "ChampionCombatCooldown")
HX_MS = ea(FLASHTRAPTION, "{d6487c09}")            # bonus MS after the blink
HX_MS_DURATION = 0.25       # wiki estimate
HX_FLASH_GATE = 2.0         # Flash remaining cooldown must exceed this
HX_RANGE0, HX_RANGE_STEP, HX_RANGE_PERIOD, HX_RANGE_MAX = 200.0, 40.0, 0.3, 400.0   # wiki T:4011828

MF_AT = ea(FOOTWEAR, "GiveBootsAtMinute") * 60.0
MF_PER_TAKEDOWN = ea(FOOTWEAR, "SecondsSoonerPerTakedown")
MF_MS = ea(FOOTWEAR, "AdditionalMovementSpeed")

CB_REFUND = ea(CASH_BACK, "PercentRefund")
TONICS = ((int(ea(TRIPLE_TONIC, "FirstElixirLevel")), AVARICE), (int(ea(TRIPLE_TONIC, "SecondElixirLevel")), FORCE),
          (int(ea(TRIPLE_TONIC, "ThirdElixirLevel")), SKILL))
TWT_PCT = ea(TIME_WARP, "RestorationPercentage")
BISCUIT_EVERY = ea(BISCUITS, "BiscuitMinuteInterval") * 60.0
BISCUIT_LAST = ea(BISCUITS, "SwapOverMinute") * 60.0
BISCUIT_COUNT = int(round(BISCUIT_LAST / BISCUIT_EVERY))
BISCUIT_HP = ea(BISCUITS, "PermanentHP")
CI_SUMMONER = ea(COSMIC, "SummonerHaste")
CI_ITEM = ea(COSMIC, "ItemHaste")
AV_OWN = ea(APPROACH, "MovementSpeedPercentBonus")
AV_OTHER = AV_OWN / 2.0     # wiki, RUNES §7.4: impairments from any source
AV_RANGE = ea(APPROACH, "ActivationDistance")
JACK_AH = ea(JACK, "HastePerStack")
JACK_AF5, JACK_AF10 = ea(JACK, "{1b48f5ea}"), ea(JACK, "{55d14eea}")   # AF at 5 / 10 stacks

GRANT_SLOTS = 8             # 3 biscuits + 3 elixirs + boots, plus slack

# Jack of All Trades stat types (wiki 4047523), one tuple per type; gold generation is the GoldPer category.
JACK_TYPES = (
    ("attack_damage",), ("attack_speed", "multiplicative_attack_speed"),
    ("ability_haste", "basic_ability_haste", "ultimate_haste"), ("ability_power",), ("armor",),
    ("percent_armor_pen",), ("crit_chance",), ("crit_damage",), ("heal_shield_power",), ("health",),
    ("health_regen", "percent_base_health_regen"), ("life_steal",), ("lethality",), ("magic_pen",),
    ("percent_magic_pen",), ("magic_resist",), ("mana",), ("mana_regen", "percent_base_mana_regen"),
    ("move_speed",), ("percent_move_speed",), ("omnivamp",), ("tenacity",),
)


@lru_cache(maxsize=1)
def _tables():
    """Static per-item-row NumPy tables."""
    cat = catalog()
    payload = json.loads(DATA_PATH.read_text())["items"]
    ids = np.asarray(cat.arrays.item_id, np.int32)
    groups = cat.group_names
    boots = np.asarray(cat.arrays.groups[:, groups.index("Boots")], bool)
    # Legendary = client epicness 5; the wiki's "Guardian's" exclusion (ARAM items) is a guard only.
    legendary = np.asarray([s.epicness == 5 and not s.name.startswith("Guardian's") for s in cat.specs], bool)
    total = np.asarray(cat.arrays.total, np.float32)
    stats = np.asarray(cat.arrays.stats, np.float32) > 0.0
    jack = np.zeros((len(ids), len(JACK_TYPES) + 1), bool)
    for k, names in enumerate(JACK_TYPES):
        for name in names:
            jack[:, k] |= stats[:, STAT_INDEX[name]]
    jack[:, -1] = [("GoldPer" in payload[str(i)].get("categories", ())) for i in ids]
    slots = np.where(np.asarray(cat.arrays.trinket), 0, 1).astype(np.float32)
    max_stack = np.maximum(np.asarray(cat.arrays.max_stack, np.float32), 1.0)
    heal = {HEALTH_POTION: cat.dv(HEALTH_POTION, "HealAmount"), REFILLABLE: cat.dv(REFILLABLE, "HealAmount")}
    return dict(ids=ids, boots=boots, legendary=legendary, total=total, jack=jack.astype(np.float32),
                slots=slots, max_stack=max_stack, heal=heal)


def _row_of(item_id: Any) -> tuple[Any, Any]:
    """(row, valid) for an int32 item-id array (0 or unknown = invalid)."""
    ids = jnp.asarray(_tables()["ids"])
    r = jnp.clip(jnp.searchsorted(ids, item_id), 0, ids.shape[0] - 1)
    return r, (item_id > 0) & (ids[r] == item_id)


def _slots_used(own: Any) -> Any:
    """(C,) occupied main inventory slots (stacks packed), from owned counts."""
    t = _tables()
    per = jnp.ceil(jnp.asarray(own, jnp.float32) / jnp.asarray(t["max_stack"])[None, :])
    return jnp.sum(per * jnp.asarray(t["slots"])[None, :], axis=1)


def first_strike_cooldown(level: Any) -> Any:
    """``lin(25, 15)`` with ``mScalePastDefaultMaxLevel=false`` (clamped at 18)."""
    return lin(FS_CD_START, FS_CD_END, level, scale_past_18=False) * FS_MODE_CD


def glacial_slow(bonus_ad: Any, ap: Any, hsp: Any) -> Any:
    return GA_SLOW_BASE + GA_SLOW_BAD * bonus_ad + GA_SLOW_AP * ap + GA_SLOW_HSP * hsp


def spellbook_cooldown(unique: Any) -> Any:
    return jnp.maximum(SB_BASE - SB_PER_UNIQUE * jnp.asarray(unique, jnp.float32), SB_MIN)


def hexflash_range(elapsed: Any) -> Any:
    return jnp.minimum(HX_RANGE0 + HX_RANGE_STEP * jnp.floor(elapsed / HX_RANGE_PERIOD + 1e-4), HX_RANGE_MAX)


class State(NamedTuple):
    grant_q: Any            # (C, Q) int32 pending item grants, head first, 0 = empty
    biscuits_sched: Any     # (C,) int32 biscuits queued so far
    tonic_bits: Any         # (C,) int32 Triple Tonic elixirs handed out (bit k)
    boots_queued: Any       # (C,) bool
    boots_received: Any     # (C,) bool
    takedowns: Any          # (C,) champion takedowns (Magical Footwear)
    skill_now: Any          # (C,) int32 Elixir of Skill auto-consumed this tick
    biscuits_sold: Any      # (C,)
    refunds: Any            # (C, I) int32 refunded units per catalog row (Cash Back)
    fs_cd_until: Any        # (C,)
    fs_until: Any           # (C,) buff end
    fs_due: Any             # (C, K) bonus-packet due time, BIG = free
    fs_dst: Any             # (C, K) int32
    fs_amt: Any             # (C, K)
    fs_gold_acc: Any        # (C,) bonus damage dealt during the current activation
    fs_gold_tick: Any       # (C,) First Strike gold awarded this tick
    ga_cd_until: Any        # (C,)
    ga_until: Any           # (C,) zone end
    ga_x0: Any              # (C, 3) ray start
    ga_y0: Any
    ga_dx: Any              # (C, 3) unit direction
    ga_dy: Any
    sb_ready_at: Any        # (C,) next swap time
    sb_mask: Any            # (C,) int32 bitmask of summoner ids swapped to
    sb_recent: Any          # (C, 3) int32 last picks, newest first
    hx_channel: Any         # (C,) bool
    hx_start: Any           # (C,)
    hx_cd_until: Any        # (C,)
    hx_blink: Any           # (C,) bool blink released this tick
    hx_range: Any           # (C,)
    hx_ms_until: Any        # (C,)
    av_bonus: Any           # (C,) Approach Velocity percent MS


def init(n_champions: int, n_units: int) -> State:
    c = n_champions
    z, zi, f = jnp.zeros((c,), jnp.float32), jnp.zeros((c,), jnp.int32), jnp.zeros((c,), bool)
    z3 = jnp.zeros((c, GA_RAYS), jnp.float32)
    zk = jnp.zeros((c, FS_SLOTS), jnp.float32)
    return State(
        grant_q=jnp.zeros((c, GRANT_SLOTS), jnp.int32), biscuits_sched=zi, tonic_bits=zi, boots_queued=f,
        boots_received=f, takedowns=z, skill_now=zi, biscuits_sold=z,
        refunds=jnp.zeros((c, len(catalog().ids)), jnp.int32),
        fs_cd_until=z - BIG, fs_until=z - BIG, fs_due=zk + BIG, fs_dst=zk.astype(jnp.int32), fs_amt=zk,
        fs_gold_acc=z, fs_gold_tick=z,
        ga_cd_until=z - BIG, ga_until=z - BIG, ga_x0=z3, ga_y0=z3, ga_dx=z3 + 1.0, ga_dy=z3,
        sb_ready_at=z + SB_FIRST, sb_mask=zi, sb_recent=jnp.zeros((c, SB_NO_REPEAT), jnp.int32),
        hx_channel=f, hx_start=z - BIG, hx_cd_until=z - BIG, hx_blink=f, hx_range=z, hx_ms_until=z - BIG,
        av_bonus=z)


def _push(q: Any, go: Any, item_id: int) -> Any:
    """Append ``item_id`` at the first empty slot where ``go`` (C,)."""
    empty = q == 0
    first = jnp.argmax(empty, axis=1)
    put = (jnp.arange(q.shape[1])[None, :] == first[:, None]) & go[:, None] & jnp.any(empty, axis=1)[:, None]
    return jnp.where(put, item_id, q).astype(jnp.int32)


def _pop(q: Any, go: Any) -> Any:
    shifted = jnp.concatenate([q[:, 1:], jnp.zeros((q.shape[0], 1), jnp.int32)], axis=1)
    return jnp.where(go[:, None], shifted, q)


def jack_stacks(own: Any) -> Any:
    """(C,) unique eligible stat types granted by owned items."""
    present = (jnp.asarray(own, jnp.float32) > 0).astype(jnp.float32) @ jnp.asarray(_tables()["jack"])
    return jnp.sum(present > 0, axis=1).astype(jnp.float32)


def stats(state: State, page, ctx, ev: RuneEvents) -> ItemStats:
    c = ctx.level.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    own = ev.own
    if own is not None:
        boots = jnp.any((own > 0) & jnp.asarray(_tables()["boots"])[None, :], axis=1)
        jack = jnp.where(has_rune(page, JACK), jack_stacks(own), 0.0)
    else:
        boots, jack = jnp.zeros((c,), bool), z
    ms = jnp.where(has_rune(page, FOOTWEAR) & boots, MF_MS, 0.0)
    hx_ms = jnp.where(has_rune(page, FLASHTRAPTION) & (ctx.now < state.hx_ms_until), HX_MS, 0.0)
    af = jnp.where(jack >= 10, JACK_AF10, jnp.where(jack >= 5, JACK_AF5, 0.0))
    biscuit_hp = jnp.where(has_rune(page, BISCUITS), BISCUIT_HP * state.biscuits_sold, 0.0)
    cosmic = has_rune(page, COSMIC)
    return ItemStats(move_speed=ms, percent_move_speed=jnp.where(has_rune(page, APPROACH), state.av_bonus, 0.0) + hx_ms,
                     ability_haste=JACK_AH * jack, adaptive_force=af, health=biscuit_hp, silent_health=biscuit_hp,
                     summoner_haste=jnp.where(cosmic, CI_SUMMONER, 0.0), item_haste=jnp.where(cosmic, CI_ITEM, 0.0))


def zone_mask(state: State, ctx, units) -> Any:
    """(C, N) units whose hitbox touches any of the holder's active icy zones."""
    active = (ctx.now < state.ga_until)[:, None]
    rx = units.x[None, None, :] - state.ga_x0[:, :, None]
    ry = units.y[None, None, :] - state.ga_y0[:, :, None]
    along = rx * state.ga_dx[:, :, None] + ry * state.ga_dy[:, :, None]
    across = jnp.abs(-rx * state.ga_dy[:, :, None] + ry * state.ga_dx[:, :, None])
    r = units.radius[None, None, :]
    inside = (along >= -r) & (along <= GA_LENGTH + r) & (across <= GA_WIDTH / 2.0 + r)
    return jnp.any(inside, axis=1) & active


def on_cc(state: State, page, ctx, units, ev: RuneEvents) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    champs = enemy_champions(ctx, units)
    imm = ev.cc.immobilized & champs
    go = has_rune(page, GLACIAL) & ctx.alive & (ctx.now >= state.ga_cd_until) & jnp.any(imm, axis=1)
    tgt = jnp.argmax(imm, axis=1)
    tx, ty = units.x[tgt], units.y[tgt]
    dur = GA_DURATION + GA_CC_CARRY * jnp.take_along_axis(ev.cc_duration, tgt[:, None], axis=1)[:, 0]
    # Ray 0 aims at the holder; the others at its nearest other allied champions near the target, else fan.
    hx, hy = units.x[ctx.unit], units.y[ctx.unit]
    d0x, d0y = hx - tx, hy - ty
    norm = jnp.sqrt(d0x ** 2 + d0y ** 2)
    d0x = jnp.where(norm > 1e-3, d0x / jnp.maximum(norm, 1e-3), -ctx.facing_x)
    d0y = jnp.where(norm > 1e-3, d0y / jnp.maximum(norm, 1e-3), -ctx.facing_y)
    allies = (units.cls[None, :] == CLASS_CHAMPION) & (units.team[None, :] == ctx.team[:, None]) \
        & units.alive[None, :] & (jnp.arange(n)[None, :] != ctx.unit[:, None])
    dist = jnp.sqrt((units.x[None, :] - tx[:, None]) ** 2 + (units.y[None, :] - ty[:, None]) ** 2)
    ok = allies & (dist <= GA_ALLY_RANGE) & (dist > 1e-3)
    key = jnp.where(ok, dist, jnp.inf)
    order = jnp.argsort(key, axis=1)
    dxs, dys = [d0x], [d0y]
    for k in range(1, GA_RAYS):
        j = order[:, k - 1] if n >= k else jnp.zeros((c,), jnp.int32)
        has_ally = jnp.isfinite(jnp.take_along_axis(key, j[:, None], axis=1)[:, 0])
        ax, ay = units.x[j] - tx, units.y[j] - ty
        an = jnp.maximum(jnp.sqrt(ax ** 2 + ay ** 2), 1e-3)
        ang = GA_FAN * (1.0 if k % 2 else -1.0) * ((k + 1) // 2)
        fx = d0x * np.cos(ang) - d0y * np.sin(ang)
        fy = d0x * np.sin(ang) + d0y * np.cos(ang)
        dxs.append(jnp.where(has_ally, ax / an, fx))
        dys.append(jnp.where(has_ally, ay / an, fy))
    dx, dy = jnp.stack(dxs, axis=1), jnp.stack(dys, axis=1)
    g = go[:, None]
    state = state._replace(
        ga_x0=jnp.where(g, tx[:, None] - GA_INDENT * dx, state.ga_x0),
        ga_y0=jnp.where(g, ty[:, None] - GA_INDENT * dy, state.ga_y0),
        ga_dx=jnp.where(g, dx, state.ga_dx), ga_dy=jnp.where(g, dy, state.ga_dy),
        ga_until=jnp.where(go, ctx.now + dur, state.ga_until),
        ga_cd_until=jnp.where(go, ctx.now + GA_COOLDOWN, state.ga_cd_until))
    # Slow enemy non-structures inside a zone, for one tick at a time.
    targets = zone_mask(state, ctx, units) & (units.team[None, :] != ctx.team[:, None]) \
        & units.alive[None, :] & (units.cls[None, :] != CLASS_STRUCTURE) & has_rune(page, GLACIAL)[:, None]
    strength = glacial_slow(ev.bonus_ad, ev.ap, ctx.heal_shield_power)
    slow = jnp.max(jnp.where(targets, strength[:, None], 0.0), axis=0)
    dur_n = jnp.where(slow > 0.0, ctx.dt, 0.0) * jnp.ones((n,), jnp.float32)
    return state, effects(c, n, slow=slow.astype(jnp.float32), slow_duration=dur_n.astype(jnp.float32))


def packet_amp(state: State, page, ctx, units, ev: RuneEvents, p) -> Any:
    """Glacial: enemies in a zone deal less damage to the holder's allied champions (not the holder)."""
    n = units.x.shape[0]
    zone = zone_mask(state, ctx, units) & (units.team[None, :] != ctx.team[:, None]) & has_rune(page, GLACIAL)[:, None]
    src, dst = jnp.clip(p.src, 0, n - 1), jnp.clip(p.dst, 0, n - 1)
    from_zone = zone[:, src]                                                        # (C, P)
    to_ally = (units.team[dst][None, :] == ctx.team[:, None]) & (units.cls[dst] == CLASS_CHAMPION)[None, :] \
        & (p.dst[None, :] != ctx.unit[:, None])
    hit = jnp.any(from_zone & to_ally, axis=0) & p.valid
    return jnp.where(hit, -GA_REDUCTION, 0.0).astype(jnp.float32)


def periodic(state: State, page, ctx, units, ev: RuneEvents) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    t = _tables()
    now, gt = ctx.now, ev.game_time

    # Grant queue: pop the acknowledged head, push due grants.
    q = state.grant_q
    acked = (q[:, 0] != 0) & (ev.granted == q[:, 0])
    boots_received = state.boots_received | (acked & (q[:, 0] == BOOTS_ITEM))
    q = _pop(q, acked)
    due_b = has_rune(page, BISCUITS) & (state.biscuits_sched < BISCUIT_COUNT) \
        & (gt >= BISCUIT_EVERY * (state.biscuits_sched + 1).astype(jnp.float32))
    q = _push(q, due_b, BISCUIT_ITEM)
    sched = state.biscuits_sched + due_b.astype(jnp.int32)
    full = (_slots_used(ev.own) >= 6.0) if ev.own is not None else jnp.zeros((c,), bool)
    bits, skill_now = state.tonic_bits, jnp.zeros((c,), jnp.int32)
    for k, (lvl, item) in enumerate(TONICS):
        go = has_rune(page, TRIPLE_TONIC) & (ctx.level >= lvl) & (((bits >> k) & 1) == 0)
        bits = jnp.where(go, bits | (1 << k), bits)
        if item == SKILL:
            skill_now = jnp.where(go & full, 1, skill_now)
            go = go & ~full
        q = _push(q, go, item)
    takedowns = state.takedowns + ev.kills.champion_kill + ev.kills.champion_assist
    boots_due = MF_AT - MF_PER_TAKEDOWN * takedowns
    go_boots = has_rune(page, FOOTWEAR) & ~state.boots_queued & (gt >= boots_due)
    q = _push(q, go_boots, BOOTS_ITEM)

    # Cash Back.
    cb = has_rune(page, CASH_BACK)
    hot = jnp.arange(state.refunds.shape[1])[None, :]
    rs, vs = _row_of(ev.sold)
    sold_back = cb & vs & (jnp.take_along_axis(state.refunds, rs[:, None], axis=1)[:, 0] > 0)
    refunds = state.refunds - ((hot == rs[:, None]) & sold_back[:, None]).astype(jnp.int32)
    rb, vb = _row_of(ev.purchased)
    bought = cb & vb & jnp.asarray(t["legendary"])[rb]
    refunds = refunds + ((hot == rb[:, None]) & bought[:, None]).astype(jnp.int32)
    total = jnp.asarray(t["total"])
    gold = jnp.where(bought, CB_REFUND * total[rb], 0.0) - jnp.where(sold_back, CB_REFUND * total[rs], 0.0)

    # Time Warp Tonic.
    pot = ev.potion_drunk
    heal = jnp.where(pot == HEALTH_POTION, TWT_PCT * t["heal"][HEALTH_POTION],
                     jnp.where(pot == REFILLABLE, TWT_PCT * t["heal"][REFILLABLE], 0.0))
    heal = jnp.where(has_rune(page, TIME_WARP) & ctx.alive, heal, 0.0)

    sold_biscuit = has_rune(page, BISCUITS) & (ev.sold == BISCUIT_ITEM)

    # First Strike: emit due packets; pay the gold once the buff and its missiles are done.
    due = state.fs_due <= now + 1e-4
    alive_dst = units.alive[jnp.clip(state.fs_dst, 0, n - 1)]
    p = packets(due & alive_dst & (state.fs_amt > 0.0), ctx.unit[:, None], state.fs_dst, state.fs_amt, TRUE,
                TAG_PROC | TAG_INDIRECT, item=FS)
    fs_due = jnp.where(due, BIG, state.fs_due)
    fs_amt = jnp.where(due, 0.0, state.fs_amt)
    done = (now > state.fs_until) & jnp.all(fs_due >= BIG / 2, axis=1) & ~jnp.any(due, axis=1)
    pct = jnp.where(ctx.is_ranged, FS_GOLD_RANGED, FS_GOLD_MELEE)
    pay = jnp.where(done & (state.fs_gold_acc > 0.0), pct * state.fs_gold_acc, 0.0)
    fs_acc = jnp.where(done, 0.0, state.fs_gold_acc)

    # Unsealed Spellbook.
    req = ev.spellbook_request
    ready = spellbook_ready(state, page, ctx, ev)
    allowed = ready & (req > 0) & ~jnp.any(state.sb_recent == req[:, None], axis=1)
    mask = jnp.where(allowed, state.sb_mask | (1 << jnp.clip(req, 0, 30)), state.sb_mask)
    unique = jax.lax.population_count(mask)
    recent = jnp.where(allowed[:, None], jnp.concatenate([req[:, None], state.sb_recent[:, :-1]], axis=1),
                       state.sb_recent)
    ready_at = jnp.where(allowed, now + spellbook_cooldown(unique), state.sb_ready_at)

    state = state._replace(
        grant_q=q, biscuits_sched=sched, tonic_bits=bits, boots_queued=state.boots_queued | go_boots,
        boots_received=boots_received, takedowns=takedowns, skill_now=skill_now,
        biscuits_sold=state.biscuits_sold + sold_biscuit, refunds=refunds,
        fs_due=fs_due, fs_amt=fs_amt, fs_gold_acc=fs_acc, fs_gold_tick=pay,
        sb_mask=mask, sb_recent=recent.astype(jnp.int32), sb_ready_at=ready_at)
    return state, effects(c, n, packets=p, heal_plain=heal.astype(jnp.float32),
                          gold=(gold + pay).astype(jnp.float32))


def spellbook_ready(state: State, page, ctx, ev: RuneEvents) -> Any:
    """(C,) a swap may be selected now (the TP-channel check is world-side)."""
    ooc = ctx.now - ev.clocks.last_combat >= SB_OOC
    return has_rune(page, SPELLBOOK) & ctx.alive & (ev.game_time >= state.sb_ready_at) & ooc


def on_damage(state: State, page, ctx, units, ev: RuneEvents) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    rep = ev.report
    p, r = rep.packets, rep.resolved
    dst = jnp.clip(p.dst, 0, n - 1)
    mine = p.valid[None, :] & (p.src[None, :] == ctx.unit[:, None])                       # (C, P)
    to_champ = (units.cls[dst] == CLASS_CHAMPION)[None, :] & (units.team[dst][None, :] != ctx.team[:, None])
    fs_pkt = (p.item == FS)[None, :]
    has_fs = has_rune(page, FIRST_STRIKE)
    now = ctx.now
    clk = ev.clocks

    # Gold ledger: own bonus packets resolving this pass.
    acc = state.fs_gold_acc + jnp.sum(jnp.where(mine & fs_pkt, r.final[None, :], 0.0), axis=1)

    ready = has_fs & (now >= state.fs_cd_until)
    new_episode = clk.champion_combat_start >= now - 1e-6
    lockout = ready & new_episode & ~clk.struck_first
    hit = jnp.any(mine & to_champ & ~fs_pkt, axis=1)
    fire = ready & ~lockout & clk.struck_first & (now - clk.champion_combat_start <= FS_GRACE + 1e-6) & hit
    cd = first_strike_cooldown(ctx.level)
    cd_until = jnp.where(fire | lockout, now + cd, state.fs_cd_until)
    fs_until = jnp.where(fire, now + FS_DURATION, state.fs_until)
    flat = jnp.where(fire, FS_GOLD_FLAT, 0.0)

    # Queue a share of post-mitigation champion damage while active.
    active = has_fs & (now <= fs_until)
    sel = mine & to_champ & ~fs_pkt & (r.final > 0.0)[None, :] & active[:, None]
    onehot = (p.dst[:, None] == jnp.arange(n)[None, :]).astype(jnp.float32)
    bonus = jnp.where(sel, FS_AMP * r.final[None, :], 0.0) @ onehot                       # (C, N)
    due, dsts, amt = state.fs_due, state.fs_dst, state.fs_amt
    t_new = now + FS_DELAY
    slots = jnp.arange(FS_SLOTS)[None, :]
    for _ in range(FS_PUSH):
        j = jnp.argmax(bonus, axis=1).astype(jnp.int32)
        b = jnp.take_along_axis(bonus, j[:, None], axis=1)[:, 0]
        go = b > 0.0
        same = (jnp.abs(due - t_new) < 1e-6) & (dsts == j[:, None])
        free = due >= BIG / 2
        any_same, any_free = jnp.any(same, axis=1), jnp.any(free, axis=1)
        latest_same_dst = jnp.argmax(jnp.where(dsts == j[:, None], jnp.where(free, -BIG, due), -2 * BIG), axis=1)
        k = jnp.where(any_same, jnp.argmax(same, axis=1),
                      jnp.where(any_free, jnp.argmax(free, axis=1), latest_same_dst))
        put = (slots == k[:, None]) & go[:, None]
        fresh = put & free
        due = jnp.where(fresh, t_new, due)
        dsts = jnp.where(fresh, j[:, None], dsts)
        amt = jnp.where(put & (dsts == j[:, None]), amt + b[:, None], amt)
        bonus = jnp.where(jnp.arange(n)[None, :] == j[:, None], 0.0, bonus)

    state = state._replace(fs_cd_until=cd_until, fs_until=fs_until, fs_due=due, fs_dst=dsts.astype(jnp.int32),
                           fs_amt=amt, fs_gold_acc=acc, fs_gold_tick=state.fs_gold_tick + flat)
    return state, effects(c, n, gold=flat.astype(jnp.float32))


def post_tick(state: State, page, ctx, units, ev: RuneEvents) -> State:
    now = ctx.now
    # Hextech Flashtraption; ev.summoner_haste is 0 when the event is not filled.
    hx = has_rune(page, FLASHTRAPTION)
    haste = jnp.maximum(ev.summoner_haste, jnp.where(has_rune(page, COSMIC), CI_SUMMONER, 0.0))
    haste_mult = 100.0 / (100.0 + haste)
    combat = hx & (ev.clocks.last_champion_combat >= now - 1e-6)
    avail = hx & ctx.alive & (ev.flash_cooldown > HX_FLASH_GATE) & (now >= state.hx_cd_until)
    req = ev.hexflash_request
    start = ~state.hx_channel & (req == 1) & avail & ~combat
    channel = state.hx_channel | start
    t0 = jnp.where(start, now, state.hx_start)
    elapsed = now - t0
    interrupted = channel & (combat | ~ctx.alive)
    release = channel & ~interrupted & ~start & ((req == 2) | (elapsed >= HX_CHANNEL - 1e-6))
    early = release & (elapsed < HX_MIN - 1e-6)
    blink = release & ~early
    cd_until = state.hx_cd_until
    cd_until = jnp.where(combat, jnp.maximum(cd_until, now + HX_COMBAT_CD * haste_mult), cd_until)
    cd_until = jnp.where(interrupted | early, now + HX_COMBAT_CD * haste_mult, cd_until)
    cd_until = jnp.where(blink, now + HX_COOLDOWN * haste_mult, cd_until)
    state = state._replace(
        hx_channel=channel & ~interrupted & ~release, hx_start=t0, hx_cd_until=cd_until, hx_blink=blink,
        hx_range=jnp.where(blink, hexflash_range(elapsed), 0.0),
        hx_ms_until=jnp.where(blink, now + HX_MS_DURATION, state.hx_ms_until))

    # Approach Velocity (facing within 90 deg, also while standing still; wiki).
    champs = enemy_champions(ctx, units)
    dx, dy = units.x[None, :] - ctx.x[:, None], units.y[None, :] - ctx.y[:, None]
    facing = dx * ctx.facing_x[:, None] + dy * ctx.facing_y[:, None] >= 0.0
    dist = jnp.sqrt(dx ** 2 + dy ** 2)
    own = jnp.any(champs & facing & ev.impaired_by_holder, axis=1)
    other = jnp.any(champs & facing & ev.visible & ev.movement_impaired[None, :] & (dist <= AV_RANGE), axis=1)
    bonus = jnp.where(own, AV_OWN, jnp.where(other, AV_OTHER, 0.0))
    return state._replace(av_bonus=jnp.where(has_rune(page, APPROACH) & ctx.alive, bonus, 0.0).astype(jnp.float32))


def outputs(state: State, page, ctx, ev: RuneEvents) -> RuneOutputs:
    c, i = ctx.level.shape[0], len(catalog().ids)
    out = no_outputs(c, i)
    forbid = (has_rune(page, FOOTWEAR) & ~state.boots_received)[:, None] & jnp.asarray(_tables()["boots"])[None, :]
    return out._replace(
        grant_item=state.grant_q[:, 0], forbid_purchase=forbid, skill_points=state.skill_now,
        move_locked=state.hx_channel, blink=state.hx_blink, blink_range=state.hx_range.astype(jnp.float32),
        spellbook_swap_ready=spellbook_ready(state, page, ctx, ev),
        first_strike_gold=state.fs_gold_tick.astype(jnp.float32))
