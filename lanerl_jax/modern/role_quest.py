"""Patch-26.19 Top role quest (docs/modern/ROLE_QUESTS.md), per champion (C,).

Quest numbers are server-side: Riot notes 26.1/26.9/26.19, wiki oldid 4064833 and the §8 defaults. Role is a
scenario parameter; only ``ROLE_TOP`` earns points.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

ROLE_NONE, ROLE_TOP, ROLE_JUNGLE, ROLE_MID, ROLE_BOT, ROLE_SUPPORT = range(6)
THRESHOLD = 1200.0
PASSIVE_START = 65.0
PASSIVE_BASE = 1.0 / 3.0            # anywhere
PASSIVE_IN_LANE = 1.5               # replaces the base rate (26.9)
RECALL_LOCK = 12.0                  # passive off after a recall completes (U-RQ-8)
ROAM_FILL = 0.5                     # bank seconds per second in lane
ROAM_CAP_EARLY, ROAM_CAP = 5.0, 60.0  # before / from level 3
POINTS = {"minion": 2.0, "turret": 50.0, "plate": 40.0, "takedown": 15.0, "epic": 30.0}
COMPLETION_XP = 600.0
XP_BONUS = 0.11                     # all non-takedown XP (26.9)
TAKEDOWN_XP = 80.0                  # flat per champion takedown (26.9)
EARLY_PENALTY_LEVEL = 3             # -25% minion gold/XP outside top before level 3 (U-RQ-5)
EARLY_PENALTY = 0.25
FREE_TP_COOLDOWN = 390.0            # 26.19
TP_SHIELD_FRACTION, TP_SHIELD_DURATION = 0.35, 10.0   # 26.12
QUEST_TP_REDUCTION = 30.0           # 26.19, client BuffCounter x -30


class QuestState(NamedTuple):
    role: Any               # int32
    points: Any
    complete: Any           # bool
    roam_bank: Any          # seconds
    recall_lock_until: Any


def init_quest(roles) -> QuestState:
    r = jnp.asarray(roles, jnp.int32)
    z = jnp.zeros(r.shape, jnp.float32)
    return QuestState(r, z, jnp.zeros(r.shape, bool), z, z - 1e9)


def out_of_lane_mult(points: Any) -> Any:
    """§2.1: 25% at 0 progress rising linearly to 100% at completion."""
    return 0.25 + 0.75 * jnp.clip(points / THRESHOLD, 0.0, 1.0)


class QuestEvents(NamedTuple):
    """Credits this tick (C,), split by whether the object is in the top lane."""
    minions_in_lane: Any
    minions_out: Any
    turrets_in_lane: Any
    turrets_out: Any
    plates_in_lane: Any
    plates_out: Any
    takedowns: Any
    epic: Any


def no_quest_events(n: int) -> QuestEvents:
    z = jnp.zeros((n,), jnp.float32)
    return QuestEvents(z, z, z, z, z, z, z, z)


class QuestStep(NamedTuple):
    state: QuestState
    completed_now: Any      # (C,) bool: grant +600 XP and raise the cap this tick
    level_cap: Any          # (C,) 18 or 20


def quest_step(state: QuestState, ev: QuestEvents, *, now: Any, dt: Any, in_lane: Any, alive: Any,
               level: Any, recalled: Any) -> QuestStep:
    """Event points, passive points, then the completion check (§6). Each event type scales with the progress
    before its grant, in gold-distribution order (minions, plates, turrets, takedowns, epic)."""
    top = state.role == ROLE_TOP
    pts = state.points
    for n_in, n_out, value in ((ev.minions_in_lane, ev.minions_out, POINTS["minion"]),
                               (ev.plates_in_lane, ev.plates_out, POINTS["plate"]),
                               (ev.turrets_in_lane, ev.turrets_out, POINTS["turret"])):
        pts = pts + value * (n_in + n_out * out_of_lane_mult(pts))
    pts = pts + POINTS["takedown"] * ev.takedowns + POINTS["epic"] * ev.epic
    # Roam bank (§2.2.5) and passive rate.
    cap = jnp.where(level >= 3, ROAM_CAP, ROAM_CAP_EARLY)
    lane = in_lane & alive
    bank = jnp.where(lane, jnp.minimum(state.roam_bank + ROAM_FILL * dt, cap), state.roam_bank)
    spend = ~lane & alive & (bank > 0.0)
    bank = jnp.where(spend, jnp.maximum(bank - dt, 0.0), bank)
    lock_until = jnp.where(recalled, now + RECALL_LOCK, state.recall_lock_until)
    rate = jnp.where(lane | spend, PASSIVE_IN_LANE, PASSIVE_BASE)    # dead: base rate (U-RQ-4)
    pts = pts + jnp.where((now >= PASSIVE_START) & (now >= lock_until), rate * dt, 0.0)
    pts = jnp.where(top & ~state.complete, pts, state.points)
    done = top & ~state.complete & (pts >= THRESHOLD)
    complete = state.complete | done
    return QuestStep(QuestState(state.role, pts, complete, jnp.where(top, bank, state.roam_bank), lock_until),
                     done, jnp.where(complete, 20, 18).astype(jnp.int32))


def xp_bonus(state: QuestState) -> Any:
    """(C,) additive XP modifier for non-takedown XP (+11% after completion)."""
    return jnp.where(state.complete & (state.role == ROLE_TOP), XP_BONUS, 0.0)


def takedown_xp(state: QuestState) -> Any:
    return jnp.where(state.complete & (state.role == ROLE_TOP), TAKEDOWN_XP, 0.0)


def minion_penalty(state: QuestState, level: Any, minion_in_lane: Any) -> Any:
    """(C, M) gold/XP multiplier: -25% for top outside its lane before level 3 (§2.3)."""
    early = (state.role == ROLE_TOP)[:, None] & (jnp.asarray(level)[:, None] < EARLY_PENALTY_LEVEL)
    return jnp.where(early & ~minion_in_lane, 1.0 - EARLY_PENALTY, 1.0)


def unleashed_tp_cooldown(level: Any, quest_complete: Any = False) -> Any:
    """§4.3 (client S12_SummonerTeleportUpgrade): 330 - 10(min(L,9) - 1) - (10 at L >= 10), -30 with the quest."""
    lv = jnp.asarray(level, jnp.float32)
    cd = 330.0 - 10.0 * (jnp.minimum(lv, 9.0) - 1.0) - jnp.where(lv >= 10, 10.0, 0.0)
    return cd - jnp.where(quest_complete, QUEST_TP_REDUCTION, 0.0)


def tp_arrival_shield(max_hp: Any, quest_complete: Any, has_teleport: Any) -> Any:
    """§4.2: 35% max HP for 10 s on Teleport arrival (quest complete, own TP)."""
    return jnp.where(quest_complete & has_teleport, TP_SHIELD_FRACTION * max_hp, 0.0)
