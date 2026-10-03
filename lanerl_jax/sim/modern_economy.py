"""Patch-26.19 SR economy and progression (docs/modern/ECONOMY_PROGRESSION.md).

Pure, fixed-shape JAX over C champions (holder ``c`` = world unit
``unit[c]``) and N world units. Client values come from ``economy_client.json``
(``lanerl_jax.data.build_modern_economy``, client 16.19.8230722) and the
minion barracks config (``minions.json``); defaults for unresolved rules cite
the ECONOMY U-E ids.

Contents, in the doc's order:
  §2 ambient gold      ``ambient_payments``
  §3 levels            ``level_for_xp``, ``decimal_level``, ``skill_points``
  §4 minion rewards    ``minion_rewards`` (1500 split table, comeback bonus, last-hit gold)
  §5 kill credit, XP   ``Credit``/``credit_update``/``kill_credit``, ``kill_xp``
  §6 kill gold/bounty  ``Bounty``, ``kill_gold``, ``assist_gold``, ``bounty_*``
  §7 structure gold    ``structure_gold``
  §8 level-up          ``level_up_sync``
  §9–10 death/respawn  ``death_time``, ``respawn_due``
  §11 recall/Homeguard ``Recall``/``recall_step``, ``homeguard_bonus_ms``, ``Homeguard``/``homeguard_step``
  §12 fountain         ``fountain_regen``
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from . import modern_damage as D

DATA = Path(__file__).resolve().parents[1] / "data" / "modern" / "26.19"
BIG = 1e9
LEVEL_CAP, QUEST_LEVEL_CAP, SKILL_POINT_LEVELS = 18, 20, 18
AMBIENT_TICK = 0.5                 # wiki: paid every 0.5 s (§2.1)
STRUCTURE_RADIUS = 1200.0          # plate/turret local-gold proximity (§7.1)
STRUCTURE_WINDOW = 10.0            # damage participation window (U-E-5 default)
FIRST_TURRET_BONUS = 300.0         # RN 26.1 (§7.3)
HOMEGUARD_SWITCH = 840.0           # 14:00
HOMEGUARD_DECAY = 4.0
HOMEGUARD_LOCKOUT = 8.0
HOMEGUARD_FOUNTAIN_HEAL = 0.08     # missing HP and mana every 0.5 s (§11.2.5)
DEATHGUARD_MS = 0.75               # Respawn Homeguard before 14:00 (§10.3)
RECALL_CHANNEL = 8.0
RECALL_DAMAGE_GRACE = 0.1          # damage in the last 0.1 s does not interrupt (§11.1.3)


@lru_cache(maxsize=1)
def econ() -> dict:
    payload = json.loads((DATA / "economy_client.json").read_text())
    if payload.get("schema") != "lanerl-client-sr-economy-v1" or payload.get("patch") != "26.19":
        raise RuntimeError("modern economy table has wrong schema or patch")
    return payload


def _const(name):
    return float(econ()["constants"][name])


@lru_cache(maxsize=1)
def _tables():
    e = econ()
    need = np.zeros(QUEST_LEVEL_CAP + 2, np.float32)               # need[L] = XP to reach level L
    need[2:QUEST_LEVEL_CAP + 1] = e["xp_required"][:QUEST_LEVEL_CAP - 1]
    need[QUEST_LEVEL_CAP + 1] = e["xp_required"][QUEST_LEVEL_CAP - 1]  # level 21 boundary (decimal level)
    kill_xp = np.asarray([0.0] + e["kill_xp"][:QUEST_LEVEL_CAP], np.float32)
    shared = e["shared_kill_xp_mult"]
    share = np.asarray([shared[0]] + [shared[min(v - 1, len(shared) - 1)] for v in range(1, QUEST_LEVEL_CAP + 1)],
                       np.float32)
    brw = e["death_time_per_level"]
    death = np.asarray([brw[0]] + [brw[min(v - 1, len(brw) - 1)] for v in range(1, QUEST_LEVEL_CAP + 1)],
                       np.float32)                                    # levels 19–20 clamp (U-E-6)
    gold = e["base_kill_gold"]
    base_gold = np.asarray([gold[0]] + [gold[min(v - 1, len(gold) - 1)] for v in range(1, QUEST_LEVEL_CAP + 1)],
                           np.float32)
    split = np.asarray(e["minion_split_xp"], np.float32)
    barracks = json.loads((DATA / "minions.json").read_text())
    radius = float(_find(barracks, "ExpRadius"))
    return dict(need=need, kill_xp=kill_xp, share=share, death=death, base_gold=base_gold, split=split,
                minion_xp_radius=radius)


def _find(x, key):
    if isinstance(x, dict):
        if key in x:
            return x[key]
        for v in x.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(x, list):
        for v in x:
            r = _find(v, key)
            if r is not None:
                return r
    return None


def table(name: str) -> Any:
    return jnp.asarray(_tables()[name])


def _lv(level):
    return jnp.clip(jnp.asarray(level, jnp.int32), 0, QUEST_LEVEL_CAP)


# ---- §2 ambient gold ---------------------------------------------------------

def ambient_payments(t0: Any, t1: Any) -> Any:
    """Gold paid in (t0, t1]: 1.02 g at 65.0 s and every 0.5 s after.

    U-E-1 resolved by the 16.9 replay oracle (145 games): the first payment
    lands at the start time itself (65.0 s), not one tick later."""
    start = _const("mission_AmbientGoldStartTime")
    per = _const("ai_AmbientGoldAmount") / _const("ai_AmbientGoldInterval") * AMBIENT_TICK
    k = lambda t: jnp.where(jnp.asarray(t, jnp.float32) >= start,
                            jnp.floor((jnp.asarray(t, jnp.float32) - start) / AMBIENT_TICK + 1e-6) + 1.0, 0.0)
    return per * (k(t1) - k(t0))


# ---- §3 levels ---------------------------------------------------------------

def level_for_xp(xp: Any, cap: Any = LEVEL_CAP) -> Any:
    """``1 + #{L in 2..cap : xp >= need[L]}`` (multi-level jumps allowed)."""
    need = table("need")[2:QUEST_LEVEL_CAP + 1]                       # levels 2..20
    levels = jnp.arange(2, QUEST_LEVEL_CAP + 1)
    xp = jnp.asarray(xp, jnp.float32)[..., None]
    reached = (xp >= need) & (levels <= jnp.asarray(cap)[..., None])
    return 1 + jnp.sum(reached, axis=-1).astype(jnp.int32)


def decimal_level(xp: Any, cap: Any = LEVEL_CAP) -> Any:
    """``L + (xp − need[L]) / (need[L+1] − need[L])``; exactly ``L`` at the cap (§3.3)."""
    lv = level_for_xp(xp, cap)
    need = table("need")
    lo, hi = need[lv], need[jnp.minimum(lv + 1, QUEST_LEVEL_CAP + 1)]
    frac = jnp.clip((jnp.asarray(xp, jnp.float32) - lo) / jnp.maximum(hi - lo, 1.0), 0.0, 0.999999)
    return jnp.where(lv >= jnp.asarray(cap), lv.astype(jnp.float32), lv + frac)


def skill_points(level: Any) -> Any:
    """One point per level 1–18; levels 19–20 grant stats only (§8.3)."""
    return jnp.minimum(jnp.asarray(level, jnp.int32), SKILL_POINT_LEVELS)


def max_rank(level: Any, ultimate: bool = False) -> Any:
    """Basic ranks ≤ ceil(level/2) up to 5; R at 6/11/16 (§8.3)."""
    lv = jnp.minimum(jnp.asarray(level, jnp.int32), SKILL_POINT_LEVELS)
    if ultimate:
        return jnp.sum(lv[..., None] >= jnp.asarray([6, 11, 16]), axis=-1)
    return jnp.minimum((lv + 1) // 2, 5)


# ---- §4 minion rewards -------------------------------------------------------

def minion_comeback_mult(minion_level: Any, receiver_decimal_level: Any) -> Any:
    """§4.3 underleveled bonus vs lane minions (negative variant disabled)."""
    ml = jnp.asarray(minion_level, jnp.float32)
    d = ml - jnp.asarray(receiver_decimal_level, jnp.float32)
    start = _const("aiExp_bonusExpLaneLevelStart")
    dmin = _const("aiExp_bonusExpLaneLevelDeltaMin")
    c1, c1_ub = _const("aiExp_bonusExpPercentPerLaneMinionLevelC1"), _const("aiExp_bonusExpPercentPerLaneMinionLevelC1UBound")
    c2, cap = _const("aiExp_bonusExpPercentPerLaneMinionLevelC2"), _const("aiExp_bonusExpLevelDeltaCap")
    bonus = jnp.where(d < c1_ub, c1 * d, c2 * jnp.minimum(d, cap))
    return 1.0 + jnp.where((ml > start) & (d > dmin), bonus, 0.0)


def xp_modifier(sum_of_bonuses: Any) -> Any:
    """§3.4 global clamp on summed XP modifiers: multiplier in [0, 6]."""
    return 1.0 + jnp.clip(sum_of_bonuses, _const("gcd_PercentEXPBonusMinimum"), _const("gcd_PercentEXPBonusMaximum"))


def _cm(v: Any, c: int, m: int) -> Any:
    """Broadcast a scalar, (C,) or (C, M) value to (C, M)."""
    v = jnp.asarray(v, jnp.float32)
    return jnp.broadcast_to(v[:, None] if v.ndim == 1 else v, (c, m))


class MinionDeaths(NamedTuple):
    """Lane minions that died this tick, shape (M,) (padded with ``valid``)."""
    valid: Any
    x: Any
    y: Any
    team: Any               # owning team of the minion
    gold: Any               # bounty (modern_minions.gold_bounty)
    xp: Any                 # XP value per minion type
    level: Any              # minion level at spawn (1v1: owning champion's level, U-E-11)
    last_hitter: Any        # int32 holder index c that last-hit it, -1 = not a champion
    unit: Any = None        # int32 world unit index (for Kills.killed_units), optional


def minion_rewards(deaths: MinionDeaths, cx: Any, cy: Any, cteam: Any, calive: Any, cdec_level: Any,
                   xp_bonus: Any = 0.0, gold_mult: Any = 1.0, xp_mult: Any = 1.0) -> tuple[Any, Any, Any]:
    """(gold (C,), xp (C,), last_hits (C,)) from this tick's minion deaths (§4).

    XP: enemy champions alive within the barracks ExpRadius (1500) of the death
    plus the last hitter always, each ``xp × split[n] × comeback × (1 + bonus)``.
    Gold: full bounty to the champion last hitter only (§4.5). ``xp_bonus``
    (C,) are additive regular XP modifiers (quest +11%), ``gold_mult``/``xp_mult``
    (C, M) or (C,) extra multipliers (top quest out-of-lane −25%, ROLE_QUESTS §2.3).
    """
    r = _tables()["minion_xp_radius"]
    c = cx.shape[0]
    dist = jnp.sqrt((cx[:, None] - deaths.x[None, :]) ** 2 + (cy[:, None] - deaths.y[None, :]) ** 2)
    hitter = (jnp.arange(c)[:, None] == deaths.last_hitter[None, :]) & deaths.valid[None, :]
    near = (cteam[:, None] != deaths.team[None, :]) & calive[:, None] & (dist <= r) & deaths.valid[None, :]
    elig = near | hitter
    n = jnp.sum(elig, axis=0)
    split = table("split")[jnp.clip(n - 1, 0, table("split").shape[0] - 1)]
    m = deaths.valid.shape[0]
    comeback = minion_comeback_mult(deaths.level[None, :], cdec_level[:, None])
    per = deaths.xp[None, :] * split[None, :] * comeback * _cm(xp_modifier(_cm(xp_bonus, c, m)), c, m) \
        * _cm(xp_mult, c, m)
    xp = jnp.sum(jnp.where(elig, per, 0.0), axis=1)
    gold = jnp.sum(jnp.where(hitter, deaths.gold[None, :] * _cm(gold_mult, c, m), 0.0), axis=1)
    return gold, xp, jnp.sum(hitter, axis=1)


# ---- §5 kill credit and champion-kill XP --------------------------------------

class Credit(NamedTuple):
    """Last time each holder affected each unit (damage incl. 0, CC), (C, N)."""
    last_affect: Any
    last_structure_damage: Any      # (C, N) last damage to structures (plate/turret share, §7)


def init_credit(n_champions: int, n_units: int) -> Credit:
    z = jnp.full((n_champions, n_units), -BIG, jnp.float32)
    return Credit(z, z)


def credit_update(credit: Credit, report, unit: Any, now: Any, units_cls: Any, cc=None) -> Credit:
    """Record this tick's packets from holders (and CC applied by holders)."""
    p = report.packets
    n = credit.last_affect.shape[1]
    src = p.valid[None, :] & (p.src[None, :] == unit[:, None])                    # (C, P)
    onehot = (p.dst[:, None] == jnp.arange(n)[None, :]) & ~D.has(p.flags, D.PROP_REACTIVE)[:, None]
    touched = (src.astype(jnp.float32) @ onehot.astype(jnp.float32)) > 0.0
    if cc is not None:
        touched = touched | cc.slowed | cc.immobilized
    structure = touched & (units_cls[None, :] == D.CLASS_STRUCTURE)
    return Credit(jnp.where(touched, now, credit.last_affect),
                  jnp.where(structure, now, credit.last_structure_damage))


def kill_credit(credit: Credit, victim: Any, now: Any, killer_hint: Any = None,
                window: Any = None) -> tuple[Any, Any, Any]:
    """(killer (int32, -1 = execution), assisters (C,) bool, any_credit ()) for unit ``victim``.

    The last holder that affected the victim within 15 s gets the kill, even
    if a non-champion dealt the final blow; others in the window assist
    (§5.1.1). ``killer_hint`` (int32, -1 none) is the holder that dealt the
    final blow this tick, which wins ties.
    """
    w = econ()["assist_window"] if window is None else window
    t = credit.last_affect[:, victim]
    ok = (now - t) <= w
    key = jnp.where(ok, t, -BIG)
    if killer_hint is not None:
        key = jnp.where(jnp.arange(t.shape[0]) == killer_hint, jnp.where(ok, key + 1e-3, now + 1e-3), key)
        ok = ok | (jnp.arange(t.shape[0]) == killer_hint)
    killer = jnp.where(jnp.any(ok), jnp.argmax(key), -1).astype(jnp.int32)
    assisters = ok & (jnp.arange(t.shape[0]) != killer)
    return killer, assisters, jnp.any(ok)


def level_difference_xp_mult(victim_dec: Any, recipient_dec: Any) -> Any:
    """§5.2.3: 0 for |Δ| ≤ 1, then ±20% per level; overleveled floor 40%."""
    slope = econ()["level_difference_xp"][1]
    delta = jnp.asarray(victim_dec) - jnp.asarray(recipient_dec)
    m = slope * jnp.maximum(jnp.abs(delta) - 1.0, 0.0)
    return jnp.where(delta > 0, 1.0 + m, 1.0 - jnp.minimum(m, 0.6))


def kill_xp(victim_level: Any, victim_dec: Any, recipient_dec: Any, eligible: Any,
            takedown: Any = None, quest_flat: Any = 0.0) -> Any:
    """(C,) XP for one champion death (§5.2).

    ``eligible`` (C,): killer, assisters, and enemies of the victim alive within
    1600 (or dead < 10 s ago, near their corpse). ``takedown`` (C,) marks kill
    or assist (quest +80 flat applies to those, §5.2.4).
    """
    v = _lv(victim_level)
    n = jnp.sum(eligible)
    pool = table("kill_xp")[v] * jnp.where(n >= 2, table("share")[v], 1.0)
    each = pool / jnp.maximum(n, 1)
    xp = jnp.where(eligible, each * level_difference_xp_mult(victim_dec, recipient_dec), 0.0)
    td = eligible if takedown is None else takedown
    return xp + jnp.where(td, quest_flat, 0.0)


def kill_xp_eligible(victim_x, victim_y, victim_team, cx, cy, cteam, calive, cdead_since, now, takedown):
    """Eligibility for champion-kill XP (§5.2.1)."""
    r = _const("ai_ExpRadius2")
    dist = jnp.sqrt((cx - victim_x) ** 2 + (cy - victim_y) ** 2)
    recently_dead = ~calive & (now - cdead_since <= _const("aiExp_timeForKillCreditAfterDeath"))
    return (cteam != victim_team) & ((calive | recently_dead) & (dist <= r) | takedown)


# ---- §6 kill gold and the bounty system ---------------------------------------

class Bounty(NamedTuple):
    """Per champion, shape (C,)."""
    b: Any                  # applied bounty offset from base (may be negative)
    buf: Any                # positive-entry buffer consumed (0..100)
    carry: Any              # extended bounty restored at respawn
    pending: Any            # deferred change (applied 5 s out of champion combat)


def init_bounty(n_champions: int) -> Bounty:
    z = jnp.zeros((n_champions,), jnp.float32)
    return Bounty(z, z, z, z)


def _b(name):
    return float(econ()["bounty"][name])


def base_kill_gold(victim_level: Any) -> Any:
    return table("base_gold")[_lv(victim_level)]


def kill_gold(victim_b: Any, victim_level: Any, first_blood: Any = False) -> Any:
    """§6.2.1: ``clamp(base + min(max(B,0), 700) + min(B,0), 50, base + 700)`` (+100 first blood)."""
    base = base_kill_gold(victim_level)
    cap = _b("max_above_base")
    k = jnp.clip(base + jnp.minimum(jnp.maximum(victim_b, 0.0), cap) + jnp.minimum(victim_b, 0.0),
                 _b("min_kill_gold"), base + cap)
    return k + jnp.where(first_blood, float(econ()["first_blood_bonus"]), 0.0)


def early_assist_factor(t: Any) -> Any:
    s, e, m = _b("early_assist_start"), _b("early_assist_end"), _b("early_assist_mult")
    return jnp.clip(m + (1.0 - m) * (jnp.asarray(t) - s) / (e - s), m, 1.0)


def assist_gold(k_without_fb: Any, victim_level: Any, t: Any, n_assisters: Any, first_blood: Any = 0.0) -> Any:
    """§6.2.2: assist pool ``(min(0.5·K, 0.5·base) + 0.5·FB)·early(t)``, split equally.

    The first-blood bonus sits outside the 50%-of-base cap and is shared at
    50% too: replay oracle (16.9, 145 first bloods) pays 200 to a lone
    assister after 175 s, where ``min(0.5·K, 0.5·base)`` alone gives 150;
    4,589 later assisted kills confirm the cap (pool / base median 0.5 on
    shutdowns)."""
    total = (jnp.minimum(0.5 * k_without_fb, 0.5 * base_kill_gold(victim_level)) + 0.5 * first_blood) \
        * early_assist_factor(t)
    return jnp.where(n_assisters > 0, total / jnp.maximum(n_assisters, 1), 0.0), \
        jnp.where(n_assisters > 0, total, 0.0)


def _accrue(b: Any, buf: Any, delta: Any) -> tuple[Any, Any]:
    """Apply a bounty change with the 100-point positive-entry buffer (§6.2.4)."""
    total = b + delta
    pos_gain = jnp.maximum(total, 0.0) - jnp.maximum(b, 0.0)
    fill = jnp.clip(jnp.minimum(pos_gain, _b("positive_buffer") - buf), 0.0, None)
    nb = total - fill
    nbuf = jnp.where(nb <= 0.0, 0.0, buf + fill)
    return nb, jnp.where(total <= 0.0, 0.0, nbuf)


def bounty_champion_gold(state: Bounty, gold: Any, shutdown_part: Any = 0.0) -> Bounty:
    """§6.2.4: +1 per 3 g of kill/assist gold, deferred; shutdown gold above
    base+100 is ignored while the earner is positive."""
    counted = gold - jnp.where(state.b > 0.0, shutdown_part, 0.0)
    return state._replace(pending=state.pending + counted / _b("kill_gold_per_bounty"))


def bounty_gv(state: Bounty, gv_gold: Any) -> Bounty:
    """§6.2.5: minion/monster gold, 1:20 while B ≥ 0, 1:7 while B < 0 (deferred)."""
    rate = jnp.where(state.b + state.pending >= 0.0, _b("gv_gold_per_bounty_positive"),
                     _b("gv_gold_per_bounty_negative"))
    return state._replace(pending=state.pending + gv_gold / rate)


def bounty_on_death(state: Bounty, victim_level: Any, paid_total: Any, died: Any) -> Bounty:
    """§6.2.3 victim depreciation (immediate). ``paid_total`` = K + all assist gold."""
    base = base_kill_gold(victim_level)
    cap = _b("max_above_base")
    pos = state.b > 0.0
    carry = jnp.where(pos, jnp.maximum(state.b - cap, 0.0), state.carry)
    neg_b = jnp.maximum(state.b - paid_total / _b("devalue_gold_per_bounty"), _b("min_kill_gold") - base)
    nb = jnp.where(pos, 0.0, neg_b)
    return Bounty(jnp.where(died, nb, state.b), jnp.where(died, 0.0, state.buf),
                  jnp.where(died, carry, state.carry), state.pending)


def bounty_apply_pending(state: Bounty, out_of_champion_combat: Any) -> Bounty:
    """§6.2.6: deferred changes land after 5 s out of champion combat."""
    nb, nbuf = _accrue(state.b, state.buf, state.pending)
    go = out_of_champion_combat >= _b("deferral_out_of_combat")
    return Bounty(jnp.where(go, nb, state.b), jnp.where(go, nbuf, state.buf), state.carry,
                  jnp.where(go, 0.0, state.pending))


def bounty_on_respawn(state: Bounty, respawned: Any) -> Bounty:
    """§6.2.3: extended bounty above the 700 cap is restored at respawn."""
    return state._replace(b=jnp.where(respawned, state.b + state.carry, state.b),
                          carry=jnp.where(respawned, 0.0, state.carry))


class KillPayout(NamedTuple):
    gold: Any               # (C,) gold to each holder
    bounty: Bounty
    first_blood_done: Any
    killer: Any
    assisters: Any


def champion_kill(bounty: Bounty, victim: Any, victim_level: Any, killer: Any, assisters: Any,
                  now: Any, first_blood_done: Any, credited: Any) -> KillPayout:
    """Gold and bounty for the death of holder ``victim`` (§6.2 steps 1–4).

    ``killer`` is the credited holder (-1 execution: nothing is paid and the
    victim's bounty is unchanged, §5.1.2).
    """
    c = bounty.b.shape[0]
    fb = credited & ~first_blood_done
    k = kill_gold(bounty.b[victim], victim_level, fb)
    k_nofb = k - jnp.where(fb, float(econ()["first_blood_bonus"]), 0.0)
    n_ast = jnp.sum(assisters)
    each, total_a = assist_gold(k_nofb, victim_level, now, n_ast, k - k_nofb)
    is_killer = (jnp.arange(c) == killer) & credited
    gold = jnp.where(is_killer, k, 0.0) + jnp.where(assisters & credited, each, 0.0)
    shutdown = jnp.where(is_killer, jnp.maximum(k_nofb - base_kill_gold(victim_level) - 100.0, 0.0), 0.0)
    b = bounty_champion_gold(bounty, gold, shutdown)
    died = (jnp.arange(c) == victim) & credited
    b = bounty_on_death(b, victim_level, k_nofb + total_a, died)
    return KillPayout(gold, b, first_blood_done | credited, killer, assisters)


# ---- §7 structure gold --------------------------------------------------------

def structure_eligible(structure: Any, sx: Any, sy: Any, structure_team: Any, credit: Credit, now: Any,
                       cx: Any, cy: Any, cteam: Any, calive: Any) -> Any:
    """(C,) local-gold (and quest-credit) eligibility for a plate or turret (§7.1)."""
    enemy = cteam != structure_team
    recent = (now - credit.last_structure_damage[:, structure]) <= STRUCTURE_WINDOW
    near = calive & (jnp.sqrt((cx - sx) ** 2 + (cy - sy) ** 2) <= STRUCTURE_RADIUS)
    return enemy & (recent | near)


def structure_gold(local_gold: Any, global_gold: Any, structure: Any, sx: Any, sy: Any, structure_team: Any,
                   credit: Credit, now: Any, cx: Any, cy: Any, cteam: Any, calive: Any,
                   first_turret: Any = False) -> Any:
    """(C,) gold for one plate or turret event (§7.1–7.3).

    Local gold (+300 first-turret share) splits equally among enemy champions
    that damaged it in the last 10 s (any range, alive or dead) or are alive
    within 1200; global gold goes to every enemy champion.
    """
    enemy = cteam != structure_team
    elig = structure_eligible(structure, sx, sy, structure_team, credit, now, cx, cy, cteam, calive)
    local = local_gold + jnp.where(first_turret, FIRST_TURRET_BONUS, 0.0)
    share = jnp.where(elig, local / jnp.maximum(jnp.sum(elig), 1), 0.0)
    return share + jnp.where(enemy, global_gold, 0.0)


# ---- §8 level-up --------------------------------------------------------------

def level_up_sync(hp: Any, max_hp_old: Any, max_hp_new: Any, mana: Any = 0.0, max_mana_old: Any = 0.0,
                  max_mana_new: Any = 0.0) -> tuple[Any, Any]:
    """§8.2: current HP (and mana, INF M) gain the full max increase."""
    gain = _const("ai_levelUp_healthGainNetGain")
    penalty = _const("ai_levelUp_healthGainPercentMissingPenalty")
    missing = 1.0 - hp / jnp.maximum(max_hp_old, 1.0)
    dh = jnp.maximum(max_hp_new - max_hp_old, 0.0) * (gain - penalty * missing)
    dm = jnp.maximum(max_mana_new - max_mana_old, 0.0)
    return jnp.minimum(hp + dh, max_hp_new), jnp.minimum(mana + dm, max_mana_new)


# ---- §9–10 death timer and respawn -----------------------------------------------

def time_increase_factor(t: Any) -> Any:
    """§9 TIF: from 15:00, each scaling point adds its percent per 30 s,
    accrued **continuously** (not in 30 s steps), capped at +50%.

    Both shapes were checked against 6,025 replay deaths (16.9 oracle): the
    wiki's per-segment ``ceil`` steps put 47% of post-15:00 deaths within
    0.1 s, continuous accrual 94%. No scaling applies before 15:00 (U-E-7
    resolved: deaths at 10:00–15:00 match the unscaled table).
    """
    e = econ()
    inc = float(e["death_scaling_increment"])
    cap = float(e["death_scaling_cap"]) - 1.0
    t = jnp.asarray(t, jnp.float32)
    pts = e["death_scaling_points"]
    total = jnp.zeros_like(t)
    for i, (start, pct) in enumerate(pts):
        end = pts[i + 1][0] if i + 1 < len(pts) else 1e9
        total = total + pct * jnp.clip(jnp.minimum(t, end) - start, 0.0, None) / inc
    return jnp.minimum(total, cap)


def death_time(level: Any, t: Any, reduction: Any = 0.0) -> Any:
    """BRW[level at death] × (1 + TIF(t)), respawn-time mods clamped at −95%."""
    mod = jnp.maximum(-jnp.asarray(reduction, jnp.float32), _const("gcd_PercentRespawnTimeModMinimum"))
    return table("death")[_lv(level)] * (1.0 + time_increase_factor(t)) * (1.0 + mod)


def respawn_due(dead: Any, respawn_at: Any, now: Any) -> Any:
    return dead & (now >= respawn_at)


def deathguard_ms(game_time: Any) -> Any:
    """§10.3: respawn Homeguard 75% bonus MS before 14:00."""
    return jnp.where(jnp.asarray(game_time) < HOMEGUARD_SWITCH, DEATHGUARD_MS, 0.0)


# ---- §11 recall and Homeguard ------------------------------------------------------

class Recall(NamedTuple):
    channeling: Any         # (C,) bool
    start: Any              # (C,) seconds


def init_recall(n_champions: int) -> Recall:
    return Recall(jnp.zeros((n_champions,), bool), jnp.full((n_champions,), -BIG, jnp.float32))


def recall_step(state: Recall, now: Any, *, request: Any, cancel_action: Any, health_damage: Any,
                disabled: Any, dead: Any, channel: Any = None) -> tuple[Recall, Any]:
    """§11.1: 8 s channel; returns (state, completed (C,)).

    ``cancel_action``: the holder moved/attacked/cast; ``health_damage``: damage
    > 0 reached health this tick (shield-absorbed damage does not count);
    ``disabled``: silence, ground, root or a stun-class CC.
    """
    start = request & ~state.channeling & ~dead
    ch = state.channeling | start
    t0 = jnp.where(start, now, state.start)
    elapsed = now - t0
    grace = elapsed >= RECALL_CHANNEL - RECALL_DAMAGE_GRACE
    interrupted = ch & ~start & (cancel_action | (health_damage & ~grace) | disabled | dead)
    done = ch & ~interrupted & (elapsed >= (RECALL_CHANNEL if channel is None else channel))
    return Recall(ch & ~interrupted & ~done, t0), done


def homeguard_bonus_ms(game_time: Any, since_leaving_fountain: Any) -> Any:
    """§11.2.2: 80% → 40% (150% → 65% after 14:00) linearly over 4 s, then flat."""
    late = jnp.asarray(game_time) >= HOMEGUARD_SWITCH
    hi, lo = jnp.where(late, 1.5, 0.8), jnp.where(late, 0.65, 0.4)
    f = jnp.clip(jnp.asarray(since_leaving_fountain) / HOMEGUARD_DECAY, 0.0, 1.0)
    return hi + (lo - hi) * f


class Homeguard(NamedTuple):
    active: Any             # (C,) bool
    left_at: Any            # (C,) when the holder left the fountain (+inf while inside)
    lockout_until: Any      # (C,)


def init_homeguard(n_champions: int) -> Homeguard:
    z = jnp.zeros((n_champions,), jnp.float32)
    return Homeguard(jnp.zeros((n_champions,), bool), z + BIG, z - BIG)


def homeguard_step(state: Homeguard, now: Any, game_time: Any, *, in_fountain: Any, combat: Any,
                   reached_endpoint: Any, in_jungle: Any, teleported: Any, recalled: Any) -> tuple[Homeguard, Any]:
    """§11.2: (state, bonus MS (C,)). Available from 0:20; lockout 8 s after
    losing it to combat/jungle; Recall removes the lockout."""
    lock = jnp.where(recalled, -BIG, state.lockout_until)
    gain = in_fountain & (game_time >= 20.0) & (now >= lock)
    active = state.active | gain
    left_at = jnp.where(in_fountain, BIG, jnp.where(state.left_at >= BIG, now, state.left_at))
    lose_combat = active & (combat | in_jungle) & ~in_fountain
    lose = active & ((reached_endpoint | teleported) & ~in_fountain) | lose_combat
    lock = jnp.where(lose_combat, now + HOMEGUARD_LOCKOUT, lock)
    active = active & ~lose
    ms = jnp.where(active & ~in_fountain, homeguard_bonus_ms(game_time, now - left_at), 0.0)
    return Homeguard(active, left_at, lock), ms


# ---- §12 fountain -------------------------------------------------------------------

def fountain_regen(hp: Any, max_hp: Any, mana: Any, max_mana: Any, in_fountain: Any, t0: Any, t1: Any,
                   homeguard: Any = False) -> tuple[Any, Any]:
    """§12.1 (+§11.2.5): +2% max HP and +2.5% max mana every 0.25 s within
    1100 of the fountain; with Homeguard also +8% missing HP/mana every 0.5 s
    (flat pulses first on coincident ticks, INF)."""
    period = _const("sp_RegenTickInterval")
    pulses = lambda p: jnp.floor(jnp.asarray(t1) / p + 1e-6) - jnp.floor(jnp.asarray(t0) / p + 1e-6)
    k = jnp.where(in_fountain, pulses(period), 0.0)
    hp = jnp.minimum(hp + k * _const("sp_HealthRegenPercent") * max_hp, max_hp)
    mana = jnp.minimum(mana + k * _const("sp_ManaRegenPercent") * max_mana, max_mana)
    kh = jnp.where(in_fountain & homeguard, pulses(0.5), 0.0)
    keep = (1.0 - HOMEGUARD_FOUNTAIN_HEAL) ** kh
    return max_hp - (max_hp - hp) * keep, max_mana - (max_mana - mana) * keep


def in_fountain(x: Any, y: Any, fountain_x: Any, fountain_y: Any) -> Any:
    return jnp.sqrt((x - fountain_x) ** 2 + (y - fountain_y) ** 2) <= _const("sp_RegenRadius")


def starting_gold() -> float:
    return _const("ai_StartingGold")


# ---- reference per-tick economy step (§13 ordering) ----------------------------------

class EconomyState(NamedTuple):
    gold: Any               # (C,) current gold
    gold_total: Any         # (C,) lifetime gold earned (starting gold included)
    xp: Any                 # (C,)
    level: Any              # (C,) int32
    bounty: Bounty
    credit: Credit
    recall: Recall
    homeguard: Homeguard
    quest: Any              # modern_role_quest.QuestState
    dead: Any               # (C,) bool
    dead_since: Any         # (C,)
    respawn_at: Any         # (C,)
    first_blood_done: Any   # () bool
    first_turret_done: Any  # () bool
    last_t: Any             # () game time of the previous step


def init_economy(n_champions: int, n_units: int, roles) -> EconomyState:
    from .modern_role_quest import init_quest
    z = jnp.zeros((n_champions,), jnp.float32)
    g = z + starting_gold()
    return EconomyState(g, g, z, jnp.ones((n_champions,), jnp.int32), init_bounty(n_champions),
                        init_credit(n_champions, n_units), init_recall(n_champions), init_homeguard(n_champions),
                        init_quest(roles), jnp.zeros((n_champions,), bool), z - BIG, z - BIG,
                        jnp.asarray(False), jnp.asarray(False), jnp.float32(0.0))


class StructureEvents(NamedTuple):
    """Plates/turrets destroyed this tick, shape (S,)."""
    valid: Any
    unit: Any               # structure unit index
    x: Any
    y: Any
    team: Any
    local_gold: Any
    global_gold: Any
    is_turret: Any          # turret destroyed (vs plate)
    in_top_lane: Any
    is_structure: Any = None  # (S,) bool: the unit is a structure every tick (§7 damage-credit marking,
                              # independent of ``valid``); None = ``valid``


class EconomyInputs(NamedTuple):
    now: Any
    unit: Any               # (C,) champion unit index
    x: Any
    y: Any
    team: Any
    hp: Any                 # (C,) after this tick's combat
    max_hp: Any
    report: Any             # combat Report (main + follow-up concatenated is fine)
    cc: Any                 # CC or None
    final_blow: Any         # (C,) int32 holder that dealt the final blow to holder c this tick (-1 none)
    minion_deaths: MinionDeaths
    minion_in_lane: Any     # (M,) bool: the minion belongs to the top lane
    structures: StructureEvents
    last_champion_combat: Any   # (C,) from modern_combat clocks
    in_fountain: Any
    in_quest_lane: Any
    recall_request: Any
    cancel_action: Any
    health_damage: Any      # (C,) damage > 0 reached health this tick
    disabled: Any
    reached_endpoint: Any
    in_jungle: Any
    teleported: Any
    extra_gold: Any = None  # (C,) gold from other systems this tick (jungle, objectives, wards)
    extra_xp: Any = None    # (C,) XP from other systems this tick (monsters, objectives)
    epic: Any = None        # (C,) epic-monster takedowns this tick (role quest points)
    recall_channel: Any = None  # (C,) recall channel seconds (Empowered Recall 4 s), default 8
    minion_gold_delta: Any = None   # (C,) gold change per lane-minion last hit (jungle-pet holders)
    minion_xp_mult: Any = None      # (C,) lane-minion XP multiplier (jungle-pet holders)


class EconomyOut(NamedTuple):
    state: EconomyState
    gold_gained: Any        # (C,) this tick, all sources
    xp_gained: Any
    levels_gained: Any      # (C,) int32
    kills: Any              # item/rune Kills for the next combat tick
    respawned: Any          # (C,) bool: move to the fountain, full HP/mana
    recalled: Any           # (C,) bool: teleport to the fountain
    death_duration: Any     # (C,) seconds for champions that died this tick (0 otherwise)
    homeguard_ms: Any       # (C,) bonus MS
    quest_completed: Any    # (C,) bool


def economy_step(state: EconomyState, inp: EconomyInputs) -> EconomyOut:
    """One tick of gold, XP, bounty, death and quest bookkeeping (§13)."""
    from . import modern_role_quest as Q
    from .modern_item_effects.core import Kills
    c = state.gold.shape[0]
    now = jnp.asarray(inp.now, jnp.float32)
    dt = now - state.last_t
    n_units = state.credit.last_affect.shape[1]
    cls = jnp.full((n_units,), D.CLASS_MINION, jnp.int32).at[inp.unit].set(D.CLASS_CHAMPION)
    credit = state.credit
    if inp.report is not None:
        st_mark = inp.structures.valid if inp.structures.is_structure is None else inp.structures.is_structure
        cls = cls.at[inp.structures.unit].set(jnp.where(st_mark, D.CLASS_STRUCTURE, cls[inp.structures.unit]))
        credit = credit_update(credit, inp.report, inp.unit, now, cls, inp.cc)

    # 2. Ambient gold (paid while dead, §2.4).
    gold_gain = jnp.broadcast_to(ambient_payments(state.last_t, now), (c,))
    # 4–5. Champion deaths: credit, kill gold, bounty.
    died = ~state.dead & (inp.hp <= 0.0)
    bounty, fb_done = state.bounty, state.first_blood_done
    kills = jnp.zeros((c,), jnp.float32)
    assists = jnp.zeros((c,), jnp.float32)
    killed_units = jnp.zeros((c, n_units), bool)
    xp_gain = jnp.zeros((c,), jnp.float32)
    cap = jnp.where(state.quest.complete, QUEST_LEVEL_CAP, LEVEL_CAP)
    dec = decimal_level(state.xp, cap)
    for v in range(c):
        killer, ast, any_credit = kill_credit(credit, inp.unit[v], now, inp.final_blow[v])
        ast = ast & (inp.team != inp.team[v])
        is_dead = died[v]
        valid_kill = is_dead & any_credit & (killer >= 0) & (inp.team[jnp.maximum(killer, 0)] != inp.team[v])
        pay = champion_kill(bounty, v, state.level[v], killer, ast, now, fb_done, valid_kill)
        bounty = Bounty(*(jnp.where(is_dead, a, b) for a, b in zip(pay.bounty, bounty)))
        fb_done = jnp.where(valid_kill, True, fb_done)
        gold_gain = gold_gain + jnp.where(is_dead, pay.gold, 0.0)
        is_killer = (jnp.arange(c) == killer) & valid_kill
        kills = kills + is_killer
        assists = assists + (ast & valid_kill)
        takedown = is_killer | (ast & valid_kill)
        killed_units = killed_units.at[:, inp.unit[v]].set(killed_units[:, inp.unit[v]] | takedown)
        elig = kill_xp_eligible(inp.x[v], inp.y[v], inp.team[v], inp.x, inp.y, inp.team, ~state.dead & ~died,
                                state.dead_since, now, takedown)
        xp_gain = xp_gain + jnp.where(is_dead, kill_xp(state.level[v], dec[v], dec, elig, takedown,
                                                       Q.takedown_xp(state.quest)), 0.0)
    # Minion gold (last hit) and XP, with the top-quest early out-of-lane penalty.
    md = inp.minion_deaths
    penalty = Q.minion_penalty(state.quest, state.level, inp.minion_in_lane)
    pet_xp = 1.0 if inp.minion_xp_mult is None else _cm(inp.minion_xp_mult, c, 1)
    m_gold, m_xp, last_hits = minion_rewards(md, inp.x, inp.y, inp.team, ~state.dead & ~died, dec,
                                             Q.xp_bonus(state.quest), gold_mult=penalty, xp_mult=penalty * pet_xp)
    if inp.minion_gold_delta is not None:
        m_gold = jnp.maximum(m_gold + jnp.asarray(inp.minion_gold_delta, jnp.float32) * last_hits, 0.0)
    gold_gain = gold_gain + m_gold
    xp_gain = xp_gain + m_xp
    if md.unit is not None:
        hit = (jnp.arange(c)[:, None] == md.last_hitter[None, :]) & md.valid[None, :]
        killed_units = killed_units.at[:, jnp.clip(md.unit, 0, n_units - 1)].max(hit)
    bounty = bounty_gv(bounty, m_gold)
    # Structures (plates, turrets; first turret +300).
    st = inp.structures
    first_turret = state.first_turret_done
    s_gold = jnp.zeros((c,), jnp.float32)
    s_elig = []
    for k in range(st.valid.shape[0]):
        s_elig.append(structure_eligible(st.unit[k], st.x[k], st.y[k], st.team[k], credit, now, inp.x, inp.y,
                                         inp.team, ~state.dead & ~died) & st.valid[k])
        ft = st.valid[k] & st.is_turret[k] & ~first_turret
        g = structure_gold(st.local_gold[k], st.global_gold[k], st.unit[k], st.x[k], st.y[k], st.team[k],
                           credit, now, inp.x, inp.y, inp.team, ~state.dead & ~died, ft)
        s_gold = s_gold + jnp.where(st.valid[k], g, 0.0)
        first_turret = first_turret | (st.valid[k] & st.is_turret[k])
    gold_gain = gold_gain + s_gold
    if inp.extra_gold is not None:
        gold_gain = gold_gain + jnp.asarray(inp.extra_gold, jnp.float32)
    if inp.extra_xp is not None:
        xp_gain = xp_gain + jnp.asarray(inp.extra_xp, jnp.float32)   # eligibility is the source's job
    # 7. Quest points and completion (before level-up).
    # Quest credit = local-gold eligibility (ROLE_QUESTS §2.1).
    near_struct = (jnp.stack(s_elig, axis=1) if s_elig else jnp.zeros((c, 0), bool)).astype(jnp.float32)
    qe = Q.QuestEvents(
        minions_in_lane=jnp.sum(jnp.where(inp.minion_in_lane[None, :] & (jnp.arange(c)[:, None] == md.last_hitter[None, :])
                                          & md.valid[None, :], 1.0, 0.0), axis=1),
        minions_out=jnp.sum(jnp.where(~inp.minion_in_lane[None, :] & (jnp.arange(c)[:, None] == md.last_hitter[None, :])
                                      & md.valid[None, :], 1.0, 0.0), axis=1),
        turrets_in_lane=jnp.sum(near_struct * (st.is_turret & st.in_top_lane)[None, :], axis=1),
        turrets_out=jnp.sum(near_struct * (st.is_turret & ~st.in_top_lane)[None, :], axis=1),
        plates_in_lane=jnp.sum(near_struct * (~st.is_turret & st.in_top_lane)[None, :], axis=1),
        plates_out=jnp.sum(near_struct * (~st.is_turret & ~st.in_top_lane)[None, :], axis=1),
        takedowns=kills + assists,
        epic=jnp.zeros((c,), jnp.float32) if inp.epic is None else jnp.asarray(inp.epic, jnp.float32))
    # 11. Recall / Homeguard.
    recall, recalled = recall_step(state.recall, now, request=inp.recall_request, cancel_action=inp.cancel_action,
                                   health_damage=inp.health_damage, disabled=inp.disabled, dead=state.dead | died,
                                   channel=inp.recall_channel)
    qs = Q.quest_step(state.quest, qe, now=now, dt=dt, in_lane=inp.in_quest_lane, alive=~state.dead & ~died,
                      level=state.level, recalled=recalled)
    xp_gain = xp_gain + jnp.where(qs.completed_now, Q.COMPLETION_XP, 0.0)
    # 8. Level-up with the (possibly raised) cap.
    xp = state.xp + xp_gain
    level = level_for_xp(xp, qs.level_cap)
    # 10. Death timers (level at death) and respawn.
    duration = jnp.where(died, death_time(state.level, now), 0.0)
    respawn_at = jnp.where(died, now + duration, state.respawn_at)
    dead_since = jnp.where(died, now, state.dead_since)
    dead = state.dead | died
    respawned = respawn_due(dead, respawn_at, now) & ~died
    dead = dead & ~respawned
    bounty = bounty_on_respawn(bounty, respawned)
    # 11. Deferred bounty changes.
    bounty = bounty_apply_pending(bounty, now - inp.last_champion_combat)
    hg, hg_ms = homeguard_step(state.homeguard, now, now, in_fountain=inp.in_fountain & ~dead,
                               combat=(now - inp.last_champion_combat) <= 0.0, reached_endpoint=inp.reached_endpoint,
                               in_jungle=inp.in_jungle, teleported=inp.teleported, recalled=recalled)
    hg_ms = jnp.where(respawned, jnp.maximum(hg_ms, deathguard_ms(now)), hg_ms)
    gold_cap = _const("Gold_Max")
    new = EconomyState(jnp.minimum(state.gold + gold_gain, gold_cap), state.gold_total + gold_gain, xp, level,
                       bounty, credit, recall, hg, qs.state, dead, dead_since, respawn_at, fb_done, first_turret, now)
    kill_struct = Kills(kills, assists, last_hits, died, killed_units)
    return EconomyOut(new, gold_gain, xp_gain, level - state.level, kill_struct, respawned, recalled, duration,
                      hg_ms, qs.completed_now)
