"""Summoner's Rift summoner spells, 26.19 (docs/modern/SUMMONER_SPELLS.md).

C champions with loadout slots D = 0, F = 1 and the top-quest free Unleashed Teleport slot 2 (ROLE_QUESTS §4.2).
Values: client 16.19.8230722 summoner objects with the 26.1-26.19 patch changes; quest numbers from ``role_quest``.
``step`` returns ``Effects`` (Ignite, Heal, Barrier, the TP quest shield) and a ``SummonerOut`` of movement, CC and
rune events; the world owns positions, CC, shields and HP and resolves Flash against walls (U-S12). Smite is
``jungle.camps.smite_step``; Hexflash reads ``flash_cooldown``. ``U-S*`` tags mark the spec §15 defaults for
unresolved rules.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from .. import role_quest as RQ
from ..core.damage import PROP_NO_DAMAGE_MOD, PROP_NO_OMNIVAMP, PROP_SUMMONER, TAG_PERIODIC, TRUE, packets
from ..core.stat_pipeline import cooldown, rescale_cooldown
from ..core.types import KIND_CHAMPION, KIND_INHIBITOR, KIND_MINION, KIND_NEXUS, KIND_TURRET, Dash, WorldUnits
from ..items.effects.core import BIG, Effects, ShieldGrant, effects, shield_grants
from ..runes.catalog import lin

SUMMONERS = {"flash": 4, "teleport": 12, "ignite": 14, "exhaust": 3, "barrier": 21, "heal": 7, "ghost": 6,
             "cleanse": 1, "smite": 11}
FLASH, TELEPORT, IGNITE, EXHAUST, BARRIER, HEAL, GHOST, CLEANSE, SMITE = (
    SUMMONERS[k] for k in ("flash", "teleport", "ignite", "exhaust", "barrier", "heal", "ghost", "cleanse", "smite"))
DEFERRED = {SMITE: "run by jungle.camps.smite_step (charges, upgrades, pets); step() ignores Smite requests"}
QUEST_SLOT = 2

COOLDOWN = {FLASH: 300.0, TELEPORT: 300.0, IGNITE: 180.0, EXHAUST: 240.0, BARRIER: 180.0, HEAL: 240.0,
            GHOST: 240.0, CLEANSE: 240.0, SMITE: 15.0}
START_COOLDOWN = 15.0          # §1.3, unhasted (U-S2)
READY_EPS = 1e-4

FLASH_RANGE = 400.0            # mEffectAmount[0]
TP_CHANNEL = 3.0               # ChannelDuration (both)
TP_UPGRADE_TIME = 600.0        # UpgradeMinute 10
TP_UPGRADE_FLOOR = 2.0         # "placed on a 2 s cooldown" (§3.3)
TP_DASH = (0.5, 4.5, 5000.0)   # base: 0.5 + 4.5·min(d, 5000)/5000 (U-S7)
UTP_DASH = (0.5, 3.5, 18000.0)  # Unleashed: 0.5 + 3.5·min(d, 18000)/18000
UTP_MS, UTP_MS_DURATION = 0.5, 4.0    # MSAmount / MSDuration (U-S8)
TP_FORGIVE_DIST, TP_FORGIVE_RADIUS = 2000.0, 400.0
TP_KINDS = (KIND_MINION, KIND_TURRET, KIND_INHIBITOR, KIND_NEXUS)   # U-S5
IDLE, CHANNEL, DASHING = 0, 1, 2

IGNITE_RANGE = 600.0           # centre range (WIKI)
IGNITE_TICKS = 5
IGNITE_FIRST = 0.25            # U-S3
IGNITE_PERIOD = 1.056          # WIKI
GRIEVOUS_DURATION = 5.0        # DotDuration; GrievousAmount 0.4 lives in core.damage
IGNITE_FLAGS = TAG_PERIODIC | PROP_SUMMONER | PROP_NO_DAMAGE_MOD | PROP_NO_OMNIVAMP

EXHAUST_RANGE, EXHAUST_DURATION = 650.0, 3.0   # centre range, like Ignite (INFERRED)
EXHAUST_SLOW, EXHAUST_REDUCTION = 0.40, 0.35

BARRIER_DURATION = 2.5
HEAL_ALLY_RANGE, HEAL_CURSOR = 900.0, 200.0
HEAL_MS, HEAL_MS_DURATION = 0.30, 1.0
HEAL_REPEAT, HEAL_DEBUFF = 0.5, 30.0  # U-S10
GHOST_DURATION = 10.0
CLEANSE_TENACITY, CLEANSE_DURATION = 0.75, 3.0


def ignite_total(level: Any) -> Any:
    """Breakpoints(70, +20/level, at L6 +25/level): 70 … 150 (L5), 175 (L6), 475 (L18)."""
    lv = jnp.maximum(jnp.asarray(level, jnp.float32), 1.0)
    return 70.0 + 20.0 * (jnp.minimum(lv, 5.0) - 1.0) + 25.0 * jnp.maximum(lv - 5.0, 0.0)


def barrier_amount(level: Any) -> Any:
    return lin(100.0, 460.0, level)


def heal_amount(level: Any) -> Any:
    return lin(80.0, 318.0, level)


def ghost_ms(level: Any) -> Any:
    return lin(0.24, 0.48, level)                 # the unused lin(4, 7) calc is ignored (U-S11)


def tp_dash_time(dist: Any, unleashed: Any) -> Any:
    b0, b1, cap = (jnp.where(unleashed, u, t) for u, t in zip(UTP_DASH, TP_DASH))
    return b0 + b1 * jnp.minimum(dist, cap) / cap


class State(NamedTuple):
    """Per champion (C,) unless noted."""
    spell: Any              # (C, 2) int32 loadout (D, F)
    ready_at: Any           # (C, 3) absolute time each slot is ready (slot 2 = quest TP)
    haste: Any              # summoner haste seen last step (-1 = not yet seen)
    quest_tp: Any           # bool: bonus Unleashed TP granted (quest done, no own TP)
    upgraded: Any           # bool: own Teleport became Unleashed (10:00)
    tp_phase: Any           # int32 IDLE / CHANNEL / DASHING
    tp_slot: Any            # int32 slot that cast the Teleport
    tp_unleashed: Any       # bool: the cast was Unleashed
    tp_t_end: Any           # end of the current phase
    tp_dash_time: Any       # dash seconds, fixed at cast (§3.1)
    tp_x: Any
    tp_y: Any
    tp_target: Any          # int32 unit
    utp_ms_until: Any
    ig_target: Any          # int32 unit the caster's Ignite burns (-1 none)
    ig_next: Any            # next tick time
    ig_left: Any            # ticks left
    ig_per_tick: Any
    ex_target: Any          # int32 unit the caster's Exhaust is on (-1 none)
    ex_until: Any
    heal_debuff_until: Any  # repeat-Heal debuff on this champion
    heal_ms_until: Any
    ghost_until: Any
    ghost_pct: Any          # Ghost MS fraction at cast
    cleanse_until: Any      # Cleanse tenacity window
    now: Any                # () time of the last step (``flash_cooldown``)


class SummonerOut(NamedTuple):
    dash: Dash              # Flash blink (speed inf, blink); the world resolves walls
    teleport_start: Any     # (C,) bool channel began this tick
    teleport_channel: Any   # (C,) bool channelling: move/attack/cast locked
    teleport_dash: Any      # (C,) bool travelling: untargetable, cannot act
    teleport_arrive: Any    # (C,) bool arrived this tick at (teleport_x, teleport_y)
    teleport_x: Any
    teleport_y: Any
    teleport_target: Any    # (C,) int32 unit of the active/last Teleport
    arrival_shield: Any     # (C,) quest shield amount granted this tick (already in Effects.shields)
    ghosted: Any            # (C,) ignores unit collision
    bonus_ms_pct: Any       # (C,) Ghost + Heal + Unleashed arrival MS (fractions, additive)
    tenacity: Any           # (C,) Cleanse 0.75 while active
    exhaust_reduction: Any  # (N,) Offense.dealt_reduction (not on true damage)
    exhaust_slow: Any       # (N,) slow applied this tick (before slow resist)
    exhaust_slow_duration: Any
    cleanse: Any            # (C,) bool: world removes cleansable CC this tick
    cast_event: Any         # (C,) bool: summoner cast (TP: channel completed) - Nimbus Cloak
    cast_cooldown: Any      # (C,) hasted cooldown of that spell
    cast_spell: Any         # (C,) int32 spell id of the event (0 none)
    is_teleport: Any        # (C,) bool
    blinked: Any            # (C,) bool: Flash or TP arrival - Sudden Impact
    ignite_target: Any      # (C,) int32 unit Ignited this tick (-1 none) - Conqueror/Electrocute
    cooldowns: Any          # (C, 2) remaining D/F cooldowns
    quest_cooldown: Any     # (C,) remaining quest-TP cooldown (inf when not granted)


def validate_loadout(loadout) -> np.ndarray:
    """Host-side §1.1 check: two distinct SR spells per champion; returns (C, 2) deferred flags."""
    lo = np.asarray(loadout)
    if lo.ndim != 2 or lo.shape[1] != 2:
        raise ValueError(f"loadout must be (C, 2), got {lo.shape}")
    valid = set(SUMMONERS.values())
    for c, (a, b) in enumerate(lo.tolist()):
        if a not in valid or b not in valid:
            raise ValueError(f"champion {c}: spell not on Summoner's Rift 26.19: {(a, b)}")
        if a == b:
            raise ValueError(f"champion {c}: duplicate summoner spell {a}")
    return np.isin(lo, list(DEFERRED))


def init(loadout) -> State:
    """All slots start on the 15 s start-of-game cooldown (§1.3)."""
    try:
        validate_loadout(loadout)
    except TypeError:       # traced loadout: validate host-side before jit
        pass
    spell = jnp.asarray(loadout, jnp.int32)
    c = spell.shape[0]
    z, f, i = jnp.zeros((c,), jnp.float32), jnp.zeros((c,), bool), jnp.full((c,), -1, jnp.int32)
    ready = jnp.full((c, 3), START_COOLDOWN, jnp.float32).at[:, QUEST_SLOT].set(BIG)
    return State(spell, ready, z - 1.0, f, f, jnp.zeros((c,), jnp.int32), jnp.zeros((c,), jnp.int32), f, z, z,
                 z, z, i, z - BIG, i, z, z, z, i, z - BIG, z - BIG, z - BIG, z - BIG, z, z - BIG,
                 jnp.zeros((), jnp.float32))


def flash_cooldown(state: State, now: Any = None) -> Any:
    """(C,) remaining Flash cooldown (0 without Flash) for Hexflash availability."""
    now = state.now if now is None else now
    rem = jnp.maximum(state.ready_at[:, :2] - now, 0.0)
    return jnp.sum(jnp.where(state.spell == FLASH, rem, 0.0), axis=1)


def _f32(x):
    return jnp.asarray(x, jnp.float32)


def step(state: State, ctx, units: WorldUnits, *, request, now, dt, summoner_haste, can_cast,
         channel_interrupted, quest_complete, took_champion_damage=None, rooted=None, suppressed=None,
         nearsighted=None) -> tuple[State, Effects, SummonerOut]:
    """One summoner tick: haste upkeep, the 10:00 upgrade, TP phases, casts, Ignite ticks.

    ``can_cast`` gates Flash and Teleport; ``rooted`` (Flash, Teleport), ``suppressed`` (every spell) and
    ``nearsighted`` (Teleport) default to False. ``took_champion_damage`` is unused (TP ignores damage, S-F20)."""
    del took_champion_damage
    c, n = state.spell.shape[0], units.x.shape[0]
    now, dt = _f32(now), _f32(dt)
    fc = jnp.zeros((c,), bool)
    rooted = fc if rooted is None else jnp.asarray(rooted, bool)
    suppressed = fc if suppressed is None else jnp.asarray(suppressed, bool)
    nearsighted = fc if nearsighted is None else jnp.asarray(nearsighted, bool)
    can_cast = jnp.asarray(can_cast, bool)
    quest_complete = jnp.asarray(quest_complete, bool)
    haste = _f32(summoner_haste) * jnp.ones((c,), jnp.float32)
    level, alive = ctx.level, ctx.alive
    cd = lambda base: cooldown(_f32(base), haste)

    # ---- haste rescale (U-S1) -----------------------------------------------
    seen = state.haste >= 0.0
    rem = jnp.maximum(state.ready_at - now, 0.0)
    ratio = jnp.where(seen, rescale_cooldown(1.0, state.haste, haste), 1.0)
    ready = jnp.where(rem > 0.0, now + rem * ratio[:, None], state.ready_at)

    own_tp = state.spell == TELEPORT                                   # (C, 2)
    has_tp = jnp.any(own_tp, axis=1)
    # ---- quest free TP (§3.4, U-S13 ready on grant) ---------------------------
    grant = quest_complete & ~has_tp & ~state.quest_tp
    quest_tp = state.quest_tp | grant
    ready = ready.at[:, QUEST_SLOT].set(jnp.where(grant, now, ready[:, QUEST_SLOT]))

    # ---- 10:00 Unleashed transformation (§3.3, U-S9) ---------------------------
    upgrade = has_tp & ~state.upgraded & (now >= TP_UPGRADE_TIME)
    cap = cd(RQ.unleashed_tp_cooldown(1.0, quest_complete))
    rem = jnp.maximum(ready[:, :2] - now, 0.0)
    idle_tp = own_tp & (state.tp_phase == IDLE)[:, None]
    up_rem = jnp.maximum(TP_UPGRADE_FLOOR, jnp.minimum(rem, cap[:, None]))
    ready = ready.at[:, :2].set(jnp.where(upgrade[:, None] & idle_tp, now + up_rem, ready[:, :2]))
    upgraded = state.upgraded | upgrade

    def tp_cd(slot, unleashed):                                     # every Teleport is hasted (INFERRED)
        unl = cd(RQ.unleashed_tp_cooldown(level, quest_complete & (slot < QUEST_SLOT)))
        return jnp.where(slot == QUEST_SLOT, cd(RQ.FREE_TP_COOLDOWN), jnp.where(unleashed, unl, cd(COOLDOWN[TELEPORT])))

    slot_hot = lambda slot: jnp.arange(3)[None, :] == slot[:, None]   # (C, 3)

    # ---- TP phase transitions (§3.1) -------------------------------------------
    phase, t_end = state.tp_phase, state.tp_t_end
    this_cd = tp_cd(state.tp_slot, state.tp_unleashed)
    in_channel = phase == CHANNEL
    done_channel = in_channel & (now >= t_end - READY_EPS)
    interrupted = in_channel & ~done_channel & (jnp.asarray(channel_interrupted, bool) | ~alive | rooted | suppressed)
    ready = jnp.where(slot_hot(state.tp_slot) & interrupted[:, None], (now + this_cd)[:, None], ready)  # U-S6
    channel_end = t_end
    t_end = jnp.where(done_channel, channel_end + state.tp_dash_time, t_end)
    phase = jnp.where(interrupted, IDLE, jnp.where(done_channel, DASHING, phase))
    arrive = (phase == DASHING) & (now >= t_end - READY_EPS)
    ready = jnp.where(slot_hot(state.tp_slot) & arrive[:, None], (t_end + this_cd)[:, None], ready)
    phase = jnp.where(arrive, IDLE, phase).astype(jnp.int32)
    shield_tp = jnp.where(arrive, RQ.tp_arrival_shield(ctx.max_hp, quest_complete, state.tp_slot < QUEST_SLOT), 0.0)
    utp_ms_until = jnp.where(arrive & state.tp_unleashed, now + UTP_MS_DURATION, state.utp_ms_until)

    # ---- casts (§1.4) ----------------------------------------------------------
    slot = jnp.asarray(request.slot, jnp.int32)
    has_quest_slot = quest_tp & (slot == QUEST_SLOT)
    spell = jnp.where(slot == QUEST_SLOT, jnp.where(quest_tp, TELEPORT, 0),
                      jnp.where((slot == 0) | (slot == 1), state.spell[jnp.arange(c), jnp.clip(slot, 0, 1)], 0))
    slot_ready = jnp.take_along_axis(ready, jnp.clip(slot, 0, 2)[:, None], axis=1)[:, 0] <= now + READY_EPS
    busy = phase != IDLE
    base_ok = (slot >= 0) & ((slot < QUEST_SLOT) | has_quest_slot) & slot_ready & alive & ~busy & ~suppressed
    mobile_ok = base_ok & can_cast & ~rooted         # Flash / Teleport

    tgt = jnp.asarray(request.target, jnp.int32)
    ti = jnp.clip(tgt, 0, n - 1)
    dist_t = jnp.sqrt((units.x[ti] - ctx.x) ** 2 + (units.y[ti] - ctx.y) ** 2)
    enemy_champ = (tgt >= 0) & (units.kind[ti] == KIND_CHAMPION) & (units.team[ti] != ctx.team) \
        & units.alive[ti] & units.targetable[ti]

    # Flash: blink min(|cursor|, 400) toward the cursor.
    flash = mobile_ok & (spell == FLASH)
    rx, ry = _f32(request.x), _f32(request.y)
    vx, vy = rx - ctx.x, ry - ctx.y
    d = jnp.sqrt(vx ** 2 + vy ** 2)
    scale = jnp.where(d > FLASH_RANGE, FLASH_RANGE / jnp.maximum(d, 1e-6), 1.0)
    dash = Dash(flash, jnp.where(flash, ctx.x + vx * scale, 0.0), jnp.where(flash, ctx.y + vy * scale, 0.0),
                jnp.where(flash, jnp.inf, 0.0).astype(jnp.float32), jnp.full((c,), -1, jnp.int32), flash)

    # Teleport: allied minion/structure target (forgiveness snap near a far click).
    ally_ok = (units.team[None, :] == ctx.team[:, None]) & units.alive[None, :] & units.targetable[None, :]
    kind_ok = jnp.zeros((n,), bool)
    for k in TP_KINDS:
        kind_ok = kind_ok | (units.kind == k)
    tp_valid = ally_ok & kind_ok[None, :]                                        # (C, N)
    direct = (tgt >= 0) & tp_valid[jnp.arange(c), ti]
    click_far = jnp.sqrt(vx ** 2 + vy ** 2) >= TP_FORGIVE_DIST
    dclick = jnp.sqrt((units.x[None, :] - rx[:, None]) ** 2 + (units.y[None, :] - ry[:, None]) ** 2)
    snap_key = jnp.where(tp_valid & (dclick <= TP_FORGIVE_RADIUS), dclick, jnp.inf)
    snap = jnp.argmin(snap_key, axis=1)
    can_snap = click_far & jnp.isfinite(jnp.min(snap_key, axis=1))
    tp_tgt = jnp.where(direct, ti, jnp.where(can_snap, snap, -1)).astype(jnp.int32)
    tp = mobile_ok & (spell == TELEPORT) & ~nearsighted & (tp_tgt >= 0)
    tpi = jnp.clip(tp_tgt, 0, n - 1)
    tx, ty = units.x[tpi], units.y[tpi]
    unleashed_cast = jnp.where(slot == QUEST_SLOT, True, upgraded)
    tp_d = jnp.sqrt((tx - ctx.x) ** 2 + (ty - ctx.y) ** 2)
    phase = jnp.where(tp, CHANNEL, phase).astype(jnp.int32)
    t_end = jnp.where(tp, now + TP_CHANNEL, t_end)

    ignite = base_ok & (spell == IGNITE) & enemy_champ & (dist_t <= IGNITE_RANGE)
    exhaust = base_ok & (spell == EXHAUST) & enemy_champ & (dist_t <= EXHAUST_RANGE)
    barrier = base_ok & (spell == BARRIER)
    heal = base_ok & (spell == HEAL)
    ghost = base_ok & (spell == GHOST)
    cleanse = base_ok & (spell == CLEANSE)
    instant = flash | ignite | exhaust | barrier | heal | ghost | cleanse
    spell_cd = jnp.zeros((c,), jnp.float32)
    for sid in (FLASH, IGNITE, EXHAUST, BARRIER, HEAL, GHOST, CLEANSE):
        spell_cd = jnp.where(spell == sid, cd(COOLDOWN[sid]), spell_cd)
    ready = jnp.where(slot_hot(slot) & instant[:, None], (now + spell_cd)[:, None], ready)

    # Ignite: recast overrides; GW 40% for 5 s on cast.
    ig_target = jnp.where(ignite, tgt, state.ig_target)
    ig_next = jnp.where(ignite, now + jnp.where(dt >= IGNITE_FIRST, 0.0, IGNITE_FIRST), state.ig_next)
    ig_left = jnp.where(ignite, float(IGNITE_TICKS), state.ig_left)
    ig_per = jnp.where(ignite, ignite_total(level) / IGNITE_TICKS, state.ig_per_tick)
    grievous = jnp.zeros((n,), jnp.float32).at[jnp.where(ignite, ti, n)].max(
        jnp.float32(GRIEVOUS_DURATION), mode="drop")

    # Exhaust: refresh, no stacking; slow emitted on the cast tick.
    ex_target = jnp.where(exhaust, tgt, state.ex_target)
    ex_until = jnp.where(exhaust, now + EXHAUST_DURATION, state.ex_until)
    ex_idx = jnp.where(exhaust, ti, n)
    ex_slow = jnp.zeros((n,), jnp.float32).at[ex_idx].max(jnp.float32(EXHAUST_SLOW), mode="drop")
    ex_slow_dur = jnp.zeros((n,), jnp.float32).at[ex_idx].max(jnp.float32(EXHAUST_DURATION), mode="drop")

    # Cleanse removes Ignite DoTs and Exhaust on the caster (GW stays, §9).
    cleansed_unit = jnp.zeros((n,), bool).at[jnp.where(cleanse, ctx.unit, n)].set(True, mode="drop")
    ig_target = jnp.where((ig_target >= 0) & cleansed_unit[jnp.clip(ig_target, 0, n - 1)], -1, ig_target)
    ex_target = jnp.where((ex_target >= 0) & cleansed_unit[jnp.clip(ex_target, 0, n - 1)], -1, ex_target)
    cleanse_until = jnp.where(cleanse, now + CLEANSE_DURATION, state.cleanse_until)

    # Ghost.
    ghost_until = jnp.where(ghost, now + GHOST_DURATION, state.ghost_until)
    ghost_pct = jnp.where(ghost, ghost_ms(level), state.ghost_pct)

    # Heal: self + one ally champion (cursor within 200, else lowest %HP within 900).
    same = (ctx.team[:, None] == ctx.team[None, :]) & ~jnp.eye(c, dtype=bool) & alive[None, :]
    d_ally = jnp.sqrt((ctx.x[None, :] - ctx.x[:, None]) ** 2 + (ctx.y[None, :] - ctx.y[:, None]) ** 2)
    ally = same & (d_ally <= HEAL_ALLY_RANGE)
    d_cur = jnp.sqrt((ctx.x[None, :] - rx[:, None]) ** 2 + (ctx.y[None, :] - ry[:, None]) ** 2)
    near = ally & (d_cur <= HEAL_CURSOR)
    pct = ctx.hp / jnp.maximum(ctx.max_hp, 1.0)
    key = jnp.where(jnp.any(near, axis=1, keepdims=True), jnp.where(near, d_cur, jnp.inf),
                    jnp.where(ally, pct[None, :], jnp.inf))
    pick = jnp.argmin(key, axis=1)
    has_ally = heal & jnp.isfinite(jnp.min(key, axis=1))
    ally_hit = (jnp.arange(c)[None, :] == pick[:, None]) & has_ally[:, None]        # (caster, ally)
    debuffed = state.heal_debuff_until > now
    rep = jnp.where(debuffed, HEAL_REPEAT, 1.0)
    base_heal = heal_amount(level)
    self_heal = jnp.where(heal, base_heal * rep, 0.0)
    # Ally heal uses the caster's heal power; the integrator applies the
    # recipient's, so pre-divide by it.
    hsp = ctx.heal_shield_power
    ally_heal = jnp.sum(jnp.where(ally_hit, (base_heal * (1.0 + hsp))[:, None], 0.0), axis=0) \
        * rep / (1.0 + hsp)
    healed = heal | jnp.any(ally_hit, axis=0)
    heal_debuff_until = jnp.where(healed, now + HEAL_DEBUFF, state.heal_debuff_until)
    heal_ms_until = jnp.where(healed, now + HEAL_MS_DURATION, state.heal_ms_until)

    # ---- Ignite ticks (after casts so a dt >= 0.25 cast ticks at once) ---------
    ig_ti = jnp.clip(ig_target, 0, n - 1)
    ig_alive = (ig_target >= 0) & units.alive[ig_ti]
    ig_left = jnp.where(ig_alive, ig_left, 0.0)
    due = (ig_left > 0) & (now >= ig_next - READY_EPS)
    k = jnp.where(due, jnp.minimum(jnp.floor((now - ig_next) / IGNITE_PERIOD + READY_EPS) + 1.0, ig_left), 0.0)
    pk = packets(k > 0, ctx.unit, ig_ti, k * ig_per, TRUE, IGNITE_FLAGS, item=0)
    ig_left = ig_left - k
    ig_next = ig_next + k * IGNITE_PERIOD
    ig_target = jnp.where(ig_left > 0, ig_target, -1).astype(jnp.int32)

    # ---- outputs ---------------------------------------------------------------
    ex_live = (ex_target >= 0) & (ex_until > now)
    ex_red = jnp.zeros((n,), jnp.float32).at[jnp.where(ex_live, jnp.clip(ex_target, 0, n - 1), n)].max(
        jnp.float32(EXHAUST_REDUCTION), mode="drop")
    barrier_amt = jnp.where(barrier, barrier_amount(level), 0.0)
    sh_b, sh_t = shield_grants(barrier_amt, duration=BARRIER_DURATION), \
        shield_grants(shield_tp, duration=RQ.TP_SHIELD_DURATION)
    shields = ShieldGrant(*(jnp.concatenate([a, b], axis=1) for a, b in zip(sh_b, sh_t)))
    eff = effects(c, n, packets=pk, heal=(self_heal + ally_heal).astype(jnp.float32), shields=shields,
                  grievous=grievous)

    ghosted = now < ghost_until
    bonus_ms = jnp.where(ghosted, ghost_pct, 0.0) + jnp.where(now < heal_ms_until, HEAL_MS, 0.0) \
        + jnp.where(now < utp_ms_until, UTP_MS, 0.0)
    tp_event_cd = tp_cd(state.tp_slot, state.tp_unleashed)
    cast_event = instant | done_channel
    cast_spell = jnp.where(instant, spell, jnp.where(done_channel, TELEPORT, 0)).astype(jnp.int32)
    new_state = State(
        state.spell, ready.astype(jnp.float32), haste, quest_tp, upgraded, phase,
        jnp.where(tp, slot, state.tp_slot).astype(jnp.int32), jnp.where(tp, unleashed_cast, state.tp_unleashed),
        t_end.astype(jnp.float32), jnp.where(tp, tp_dash_time(tp_d, unleashed_cast), state.tp_dash_time),
        jnp.where(tp, tx, state.tp_x), jnp.where(tp, ty, state.tp_y),
        jnp.where(tp, tp_tgt, state.tp_target).astype(jnp.int32), utp_ms_until,
        ig_target, ig_next, ig_left, ig_per, ex_target.astype(jnp.int32), ex_until, heal_debuff_until,
        heal_ms_until, ghost_until, ghost_pct, cleanse_until, now)
    new_state = State(*(_f32(v) if jnp.asarray(v).dtype == jnp.float64 else v for v in new_state))
    rem_q = jnp.where(quest_tp, jnp.maximum(ready[:, QUEST_SLOT] - now, 0.0), jnp.inf)
    out = SummonerOut(
        dash=dash, teleport_start=tp, teleport_channel=phase == CHANNEL, teleport_dash=phase == DASHING,
        teleport_arrive=arrive, teleport_x=new_state.tp_x, teleport_y=new_state.tp_y,
        teleport_target=new_state.tp_target, arrival_shield=_f32(shield_tp), ghosted=ghosted,
        bonus_ms_pct=_f32(bonus_ms), tenacity=jnp.where(now < cleanse_until, CLEANSE_TENACITY, 0.0).astype(jnp.float32),
        exhaust_reduction=ex_red, exhaust_slow=ex_slow, exhaust_slow_duration=ex_slow_dur, cleanse=cleanse,
        cast_event=cast_event,
        cast_cooldown=_f32(jnp.where(instant, spell_cd, jnp.where(done_channel, tp_event_cd, 0.0))),
        cast_spell=cast_spell, is_teleport=done_channel, blinked=flash | arrive,
        ignite_target=jnp.where(ignite, tgt, -1).astype(jnp.int32),
        cooldowns=_f32(jnp.maximum(ready[:, :2] - now, 0.0)), quest_cooldown=_f32(rem_q))
    return new_state, eff, out
