"""Wards, trinkets, stealth and true sight for the 26.19 modern world (WARDS.md).

Pure fixed-shape JAX. ``ward_step`` owns the ward slots and the per-champion
trinket state; it returns events (placements, kills with the killer, gold,
consumed Control Wards) and never writes world arrays. ``ward_view`` and
``vision_kwargs`` turn the state into the per-slot world view the tick writes
onto its ``KIND_WARD`` units and the optional ``vision.visibility``
inputs (sight radius, stealth, true sight, unobstructed, exposed).

Slots: ``S = 2 * MAX_WARDS_PER_TEAM``; team t owns slots ``[t*8, (t+1)*8)``.
Champion c is holder c (team ``team[c]``), as everywhere in the modern world.

Rules and evidence (WARDS.md has the full table):

* **Totem Ward** (trinket 3340, "Stealth Ward"): client ``YellowTrinket``
  3 HP, sight 900; 2 charges (``Effect5Amount``), recharge 210 -> 90 s and
  duration 90 -> 120 s by average champion level (client data values, PATCH
  26.3, WIKI); 3 placed per player (``MaxWardsPlaced``); range 625; stealthed
  2 s after placement; 10 gold bounty; no XP (V14.19).
* **Control Ward** (2055): 75 g item consumed on placement (client), 1 placed
  per player (client ``maxNumberOfUnits`` 1), range 625, 4 HP (client
  ``JammerDevice``), regen 1 HP / 3 s after 6 s undamaged (WIKI), sight 900,
  true sight of stealthed units in its sight radius, disables enemy wards in
  it, cannot be disabled, visible, exposed while revealing a stealthed ward;
  30 gold bounty.
* **Farsight Ward** (trinket 3363, level 9 by the shop): range 4000, 1 HP
  (client ``BlueTrinket``), sight 500 (800 for the 2 s placement reveal and
  once it spots an enemy champion, then it dies 3 s later), sees over walls
  and into brush, visible, indefinite, no per-player limit; 15 gold; 1 charge,
  recharge 198 -> 99 s by average level.
* **Oracle Lens** (trinket 3364): 2 charges, recharge 160 -> 100 s by average
  level, 8 s sweep around the user, radius 600 (level 1-4), 630 at 5, +30 per
  3 levels to 750 at 17, edge range; reveals and disables stealthed enemy
  wards (2 s linger after leaving the radius); while active, wards it hits are
  revealed 2 s.
* **Hits**: wards take 1 damage per champion basic attack hit, whatever the
  damage; minions, turrets and abilities do not damage wards. A ward's first
  hit within 10 s of placement pays 5 g and removes 5 g from its bounty.
* **Trinket haste** (Grisly Mementos, Cosmic Insight item haste) shortens the
  recharge: ``R * 100 / (100 + haste)``.
* **Swap** (shop): the new trinket keeps the time-equivalent of the old one's
  charges and progress (V9.24 "charges are converted into cooldown").
* **Vision runes**: Deep Ward and Sixth Sense (domination kernels).
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from . import vision as MV
from .core import types as W
from .runes.effects import domination as DOM

__all__ = ["WardType", "TOTEM_ITEM", "CONTROL_ITEM", "FARSIGHT_ITEM", "ORACLE_ITEM", "REQ_NONE", "REQ_TRINKET",
           "REQ_CONTROL", "WardGrid", "ward_grid", "WardState", "TrinketState", "Wards", "WardRequest",
           "no_request", "WardEvents", "WardView", "init_wards", "ward_step", "ward_view", "oracle_radius",
           "vision_kwargs", "COVERAGE"]


class WardType(IntEnum):
    TOTEM = 0          # trinket "Stealth Ward" (3340): stealthed Totem Ward
    CONTROL = 1        # Control Ward (2055)
    FARSIGHT = 2       # Farsight Ward (3363)


TOTEM_ITEM, CONTROL_ITEM, FARSIGHT_ITEM, ORACLE_ITEM = 3340, 2055, 3363, 3364
TRINKETS = (TOTEM_ITEM, FARSIGHT_ITEM, ORACLE_ITEM)
REQ_NONE, REQ_TRINKET, REQ_CONTROL = -1, 0, 1

# Result codes (WardEvents.code)
OK, ERR_NONE, ERR_DEAD, ERR_NO_ITEM, ERR_NO_CHARGE, ERR_LOCKED, ERR_RANGE, ERR_TERRAIN = 0, 1, 2, 3, 4, 5, 6, 7

# ---- constants (evidence in WARDS.md) ---------------------------------------
WARD_HP = (3.0, 4.0, 1.0)                  # CLIENT YellowTrinket/JammerDevice/BlueTrinket baseHP
WARD_SIGHT = (900.0, 900.0, 500.0)         # CLIENT perceptionBubbleRadius
WARD_BOUNTY = (10.0, 30.0, 15.0)           # WIKI Ward tip data
WARD_XP = (0.0, 0.0, 0.0)                  # V14.19 removed ward XP (Control Ward: U-W-5)
CONTROL_WARD_XP_HISTORIC = 40.0            # WIKI V6.22 value, not used (U-W-5)
WARD_RADIUS = 1.0                          # CLIENT overrideGameplayCollisionRadius
TOTEM_RANGE = CONTROL_RANGE = 625.0        # CLIENT castRange TrinketTotemLvl1 / JammerDevice
FARSIGHT_RANGE = 4000.0                    # CLIENT castRange TrinketOrbLvl3
TOTEM_CAP = 3                              # CLIENT 3340 MaxWardsPlaced
CONTROL_CAP = 1                            # CLIENT JammerDevice maxNumberOfUnits
TOTEM_STEALTH_DELAY = 2.0                  # WIKI Totem Ward "stealthed after 2 seconds"
TOTEM_DURATION = (90.0, 120.0)             # CLIENT 3340 Effect1/Effect3 (avg level)
TOTEM_RECHARGE = (210.0, 90.0)             # CLIENT 3340 Starting/EndingSingleChargeTime; PATCH 26.3
ORACLE_RECHARGE = (160.0, 100.0)           # CLIENT 3364 Starting/EndingSingleChargeTime
FARSIGHT_RECHARGE = (198.0, 99.0)          # CLIENT 3363 Effect10/Effect11
MAX_AMMO = {TOTEM_ITEM: 2, ORACLE_ITEM: 2, FARSIGHT_ITEM: 1}   # CLIENT Effect5 / MaxAmmo; WIKI Farsight
INIT_CHARGES = {TOTEM_ITEM: 1, ORACLE_ITEM: 1, FARSIGHT_ITEM: 0}  # INFERRED-L (U-W-2)
TOTEM_LOCKOUT = 1.25                       # CLIENT TrinketTotemLvl1 cooldownTime (wiki: 2 s)
ORACLE_LOCKOUT = 5.0                       # CLIENT TrinketSweeperLvl3 cooldownTime; WIKI
ORACLE_DURATION = 8.0                      # CLIENT 3364 Duration; PATCH 26.1
ORACLE_RADIUS = (600.0, 750.0)             # CLIENT 3364 Starting/EndingRadius; WIKI breakpoints
ORACLE_LINGER = 2.0                        # WIKI "disabled, lingering for 2 seconds"
ORACLE_HIT_REVEAL = 2.0                    # WIKI "wards hit ... 2 seconds of vision"
FARSIGHT_REVEAL_SIGHT = 800.0              # CLIENT 3363 ScryerVisionRange
FARSIGHT_REVEAL_TIME = 2.0                 # CLIENT 3363 Effect2Amount
FARSIGHT_TRIGGER_LIFE = 3.0                # WIKI "destroys itself after 3 seconds"
CONTROL_REGEN_DELAY = 6.0                  # WIKI
CONTROL_REGEN_PERIOD = 3.0                 # WIKI
EARLY_BONUS_WINDOW = 10.0                  # WIKI "within 10 seconds of placement, grants 5 gold"
EARLY_BONUS_GOLD = 5.0

# Map regions for Deep Ward (NGRID v7 region bytes; FrankTheBoxMonster NavGridCell.cs enums).
REGION_OTHER, REGION_BLUE_JUNGLE, REGION_RED_JUNGLE, REGION_RIVER = 0, 1, 2, 3

COVERAGE = {
    TOTEM_ITEM: "Totem Ward trinket", CONTROL_ITEM: "Control Ward", FARSIGHT_ITEM: "Farsight Alteration",
    ORACLE_ITEM: "Oracle Lens",
}
INF = jnp.float32(jnp.inf)


def _lerp(ab, level):
    a, b = ab
    lv = jnp.clip(jnp.asarray(level, jnp.float32), 1.0, 18.0)
    return jnp.float32(a) + jnp.float32(b - a) * (lv - 1.0) / 17.0


def oracle_radius(level) -> Any:
    """600 at levels 1-4, 630 at 5, +30 every 3 levels, 750 from 17 (CLIENT castRadius / WIKI)."""
    lv = jnp.asarray(level, jnp.int32)
    steps = jnp.where(lv >= 5, 1 + (lv - 5) // 3, 0)
    return jnp.minimum(ORACLE_RADIUS[0] + 30.0 * steps, ORACLE_RADIUS[1]).astype(jnp.float32)


# ---- placement grid -----------------------------------------------------------

class WardGrid(NamedTuple):
    """Ward placement terrain and Deep Ward regions, arrays [z, x]."""
    walkable: Any           # (H, W) bool, team gates closed
    region: Any             # (H, W) int32 REGION_*
    cell_size: float
    min_x: float
    min_z: float


def ward_grid(grid) -> WardGrid:
    """Host: ``data.modern_map.ModernMapGrid`` -> ``WardGrid``.

    Region bytes (NGRID v7 ``regions[..., 1]``): high nibble MainRegion
    (5/6 top/bot-side jungle, 7/8 top/bot-side river), low nibble
    JungleQuadrant (1 north, 2 east = red side; 3 west, 4 south = blue side).
    """
    main = np.asarray(grid.regions[..., 1] >> 4, np.int32)
    quad = np.asarray(grid.regions[..., 1] & 15, np.int32)
    jungle = (main == 5) | (main == 6)
    region = np.where((main == 7) | (main == 8), REGION_RIVER,
                      np.where(jungle & ((quad == 3) | (quad == 4)), REGION_BLUE_JUNGLE,
                               np.where(jungle & ((quad == 1) | (quad == 2)), REGION_RED_JUNGLE, REGION_OTHER)))
    return WardGrid(jnp.asarray(grid.walkable(None)), jnp.asarray(region, jnp.int32), float(grid.cell_size),
                    float(grid.min_bounds[0]), float(grid.min_bounds[2]))


def _lookup(g: WardGrid, x, y):
    h, w = g.walkable.shape
    cx = jnp.floor((x - g.min_x) / g.cell_size).astype(jnp.int32)
    cz = jnp.floor((y - g.min_z) / g.cell_size).astype(jnp.int32)
    ok = (cx >= 0) & (cz >= 0) & (cx < w) & (cz < h)
    cx, cz = jnp.clip(cx, 0, w - 1), jnp.clip(cz, 0, h - 1)
    return ok & g.walkable[cz, cx], jnp.where(ok, g.region[cz, cx], REGION_OTHER)


# ---- state ----------------------------------------------------------------------

class WardState(NamedTuple):
    """Ward slots (S,)."""
    alive: Any
    type: Any               # int32 WardType
    owner: Any              # int32 champion
    x: Any
    y: Any
    placed_at: Any
    expires_at: Any         # inf = until killed (Control, Farsight before trigger)
    hp: Any                 # hits remaining
    max_hp: Any
    bounty: Any             # gold left for the killer
    early_paid: Any         # bool: 5 g early-detection bonus already taken
    last_damaged: Any
    regen_at: Any           # next Control Ward regen tick (inf none)
    revealed_until: Any     # exposed to the enemy team until (Sixth Sense, Oracle hit)
    disabled_until: Any     # Oracle linger
    triggered_at: Any       # Farsight spotted a champion (inf none)
    tracked: Any            # bool: tracked by the enemy team (Sixth Sense)
    deep: Any               # bool: Deep Ward
    seq: Any                # int32 placement order (oldest replaced)


class TrinketState(NamedTuple):
    """Per champion (C,)."""
    trinket: Any            # int32 item id held at the end of last tick (0 none)
    charges: Any            # int32
    progress: Any           # fraction of the next charge
    lock_until: Any         # activation lockout (1.25 s Totem, 5 s Oracle)
    oracle_until: Any       # sweep active until
    sixth_cd_until: Any     # Sixth Sense cooldown


class Wards(NamedTuple):
    slots: WardState
    trinket: TrinketState
    next_seq: Any           # () int32


class WardRequest(NamedTuple):
    """One ward action per champion (C,). ``kind``: -1 none, 0 trinket (place or sweep), 1 Control Ward."""
    kind: Any
    x: Any
    y: Any


def no_request(c: int) -> WardRequest:
    z = jnp.zeros((c,), jnp.float32)
    return WardRequest(jnp.full((c,), REQ_NONE, jnp.int32), z, z)


class WardEvents(NamedTuple):
    code: Any               # (C,) int32 result of the request (OK / ERR_*), ERR_NONE when no request
    placed: Any             # (C,) bool
    placed_slot: Any        # (C,) int32 (-1)
    placed_type: Any        # (C,) int32
    sweep_started: Any      # (C,) bool Oracle activation
    trinket_used: Any       # (C,) bool a trinket charge was spent
    consumed_control: Any   # (C,) bool: remove one 2055 from the inventory
    killed: Any             # (S,) bool killed by a champion
    killer: Any             # (S,) int32 (-1)
    expired: Any            # (S,) bool timed out / Farsight self-destruct
    replaced: Any           # (S,) bool removed by a newer placement (cap)
    gold: Any               # (C,) kill bounties + early-detection bonus
    xp: Any                 # (C,)
    sensed: Any             # (C,) bool Sixth Sense triggered


class WardView(NamedTuple):
    """Per-slot world view (S,) for the tick's KIND_WARD units."""
    alive: Any
    x: Any
    y: Any
    team: Any
    sub: Any                # WardType
    owner: Any
    hp: Any
    max_hp: Any
    sight_radius: Any       # 0 when disabled
    stealthed: Any
    true_sight: Any         # radius (Control Ward)
    unobstructed: Any       # Farsight
    exposed: Any            # shown to the enemy team through fog
    disabled: Any
    tracked: Any
    expires_at: Any


def _slot_team(s: int) -> Any:
    return (jnp.arange(s) // W.MAX_WARDS_PER_TEAM).astype(jnp.int32)


def init_wards(n_champions: int, trinket_ids=None) -> Wards:
    """Empty slots; trinkets from ``trinket_ids`` (C,) item ids (default Stealth Ward)."""
    c, s = n_champions, 2 * W.MAX_WARDS_PER_TEAM
    ids = [TOTEM_ITEM] * c if trinket_ids is None else [int(i) for i in trinket_ids]
    zf, zb, zi = jnp.zeros((s,), jnp.float32), jnp.zeros((s,), bool), jnp.zeros((s,), jnp.int32)
    ninf, inf = jnp.full((s,), -jnp.inf, jnp.float32), jnp.full((s,), jnp.inf, jnp.float32)
    slots = WardState(zb, zi, zi - 1, zf, zf, zf, inf, zf, zf, zf, zb, ninf, inf, ninf, ninf, inf, zb, zb, zi)
    cz = jnp.zeros((c,), jnp.float32)
    tr = TrinketState(jnp.asarray(ids, jnp.int32), jnp.asarray([INIT_CHARGES.get(i, 0) for i in ids], jnp.int32),
                      cz, cz, jnp.full((c,), -jnp.inf, jnp.float32), cz)
    return Wards(slots, tr, jnp.int32(0))


def _recharge(trinket, avg_level, haste):
    base = jnp.where(trinket == TOTEM_ITEM, _lerp(TOTEM_RECHARGE, avg_level),
                     jnp.where(trinket == ORACLE_ITEM, _lerp(ORACLE_RECHARGE, avg_level),
                               _lerp(FARSIGHT_RECHARGE, avg_level)))
    return base * 100.0 / (100.0 + jnp.maximum(haste, 0.0))


def _max_ammo(trinket):
    return jnp.where(trinket == FARSIGHT_ITEM, MAX_AMMO[FARSIGHT_ITEM],
                     jnp.where((trinket == TOTEM_ITEM) | (trinket == ORACLE_ITEM), 2, 0)).astype(jnp.int32)


def _is_trinket(t):
    return (t == TOTEM_ITEM) | (t == FARSIGHT_ITEM) | (t == ORACLE_ITEM)


def _control_cover(sl: WardState, now):
    """(S, S) bool: alive Control Ward i covers ward j of the other team (900 centre to centre)."""
    team = _slot_team(sl.x.shape[0])
    ctrl = sl.alive & (sl.type == WardType.CONTROL)
    d2 = (sl.x[:, None] - sl.x[None, :]) ** 2 + (sl.y[:, None] - sl.y[None, :]) ** 2
    return ctrl[:, None] & sl.alive[None, :] & (team[:, None] != team[None, :]) \
        & (d2 <= WARD_SIGHT[WardType.CONTROL] ** 2)


def _oracle_cover(sl: WardState, tr: TrinketState, now, cx, cy, cteam, calive, level):
    """(C, S) bool: champion c's active sweep covers enemy ward j (edge range)."""
    r = oracle_radius(level) + WARD_RADIUS
    team = _slot_team(sl.x.shape[0])
    act = (now < tr.oracle_until) & calive
    d2 = (sl.x[None, :] - cx[:, None]) ** 2 + (sl.y[None, :] - cy[:, None]) ** 2
    return act[:, None] & sl.alive[None, :] & (team[None, :] != cteam[:, None]) & (d2 <= (r ** 2)[:, None])


def _stealthed(sl: WardState, now):
    return sl.alive & (sl.type == WardType.TOTEM) & (now >= sl.placed_at + TOTEM_STEALTH_DELAY)


def ward_step(w: Wards, *, now, dt, request: WardRequest, x, y, team, alive, level, trinket_id,
              control_count, grid: WardGrid, can_use=None, trinket_haste=None, hits=None, hitter=None,
              rune_pages=None, ward_visible=None) -> tuple[Wards, WardEvents]:
    """Advance wards and trinkets by one tick (call after DEATH, before FOG).

    Champion inputs are (C,) at the tick's final positions: ``x, y, team,
    alive, level``; ``trinket_id`` item id in the trinket slot (0/-1 none);
    ``control_count`` Control Wards (2055) held; ``can_use`` item actives
    allowed (alive and not stunned/suppressed; default ``alive``);
    ``trinket_haste`` item haste + trinket haste. Ward inputs (S,): ``hits``
    champion basic-attack hits landed this tick, ``hitter`` champion that
    landed them (-1). ``rune_pages`` (C, R) page counts (Deep Ward, Sixth
    Sense); ``ward_visible`` (2, S) last tick's ``visible`` on the ward slots.
    """
    sl, tr = w.slots, w.trinket
    s, c = sl.x.shape[0], x.shape[0]
    now = jnp.float32(now)
    level = jnp.asarray(level, jnp.int32)
    avg = jnp.mean(level.astype(jnp.float32))
    can_use = alive if can_use is None else can_use & alive
    haste = jnp.zeros((c,), jnp.float32) if trinket_haste is None else jnp.asarray(trinket_haste, jnp.float32)
    hits = jnp.zeros((s,), jnp.int32) if hits is None else jnp.asarray(hits, jnp.int32)
    hitter = jnp.full((s,), -1, jnp.int32) if hitter is None else jnp.asarray(hitter, jnp.int32)
    steam = _slot_team(s)
    arange_c = jnp.arange(c, dtype=jnp.int32)

    # 1. Trinket swap: carry the time-equivalent of charges + progress (V9.24).
    tid = jnp.where(_is_trinket(trinket_id), trinket_id, 0).astype(jnp.int32)
    swapped = tid != tr.trinket
    old_worth = (tr.charges.astype(jnp.float32) + tr.progress) * _recharge(tr.trinket, avg, haste)
    total = jnp.where(_is_trinket(tr.trinket), old_worth, 0.0) / _recharge(tid, avg, haste)
    amax = _max_ammo(tid)
    ch_sw = jnp.minimum(jnp.floor(total).astype(jnp.int32), amax)
    pr_sw = jnp.where(ch_sw < amax, total - jnp.floor(total), 0.0)
    charges = jnp.where(swapped, ch_sw, tr.charges)
    progress = jnp.where(swapped, pr_sw, tr.progress)
    oracle_until = jnp.where(swapped & (tr.trinket == ORACLE_ITEM), now, tr.oracle_until)

    # 2. Recharge.
    charging = (charges < amax) & (tid != 0)
    progress = jnp.where(charging, progress + dt / _recharge(tid, avg, haste), 0.0)
    gain = jnp.floor(progress).astype(jnp.int32)
    charges = jnp.minimum(charges + jnp.where(charging, gain, 0), amax)
    progress = jnp.where(charges < amax, progress - gain, 0.0).astype(jnp.float32)

    # 3. Hits: 1 damage per champion basic attack; early-detection bonus; Oracle hit reveal.
    hc = jnp.clip(hitter, 0, c - 1)
    hit = sl.alive & (hits > 0) & (hitter >= 0) & (team[hc] != steam)
    early = hit & ~sl.early_paid & (now - sl.placed_at <= EARLY_BONUS_WINDOW)
    early_gold = jnp.where(early, jnp.minimum(EARLY_BONUS_GOLD, sl.bounty), 0.0)
    bounty = sl.bounty - early_gold
    hp = jnp.where(hit, sl.hp - hits.astype(jnp.float32), sl.hp)
    killed = hit & (hp <= 0.0)
    killer = jnp.where(killed, hitter, -1).astype(jnp.int32)
    oracle_hit = hit & (now < oracle_until[hc])
    revealed_until = jnp.where(oracle_hit, jnp.maximum(sl.revealed_until, now + ORACLE_HIT_REVEAL),
                               sl.revealed_until)
    last_damaged = jnp.where(hit, now, sl.last_damaged)
    regen_at = jnp.where(hit, now + CONTROL_REGEN_DELAY + CONTROL_REGEN_PERIOD, sl.regen_at)
    gold = jnp.zeros((c,), jnp.float32).at[hc].add(jnp.where(early, early_gold, 0.0))
    gold = gold.at[hc].add(jnp.where(killed, bounty, 0.0))
    wtype = sl.type
    xp_s = jnp.where(wtype == WardType.CONTROL, WARD_XP[1], jnp.where(wtype == WardType.TOTEM, WARD_XP[0], WARD_XP[2]))
    xp = jnp.zeros((c,), jnp.float32).at[hc].add(jnp.where(killed, xp_s, 0.0))

    # 4. Expiry (Totem duration, Farsight 3 s after spotting) and Control Ward regen.
    expired = sl.alive & ~killed & ((now >= sl.expires_at) | (now >= sl.triggered_at + FARSIGHT_TRIGGER_LIFE))
    alive_s = sl.alive & ~killed & ~expired
    regen = alive_s & (wtype == WardType.CONTROL) & (now >= regen_at) & (hp < sl.max_hp)
    hp = jnp.where(regen, hp + 1.0, hp)
    regen_at = jnp.where(regen, regen_at + CONTROL_REGEN_PERIOD, regen_at)
    sl = sl._replace(alive=alive_s, hp=hp, bounty=bounty, early_paid=sl.early_paid | early,
                     last_damaged=last_damaged, regen_at=regen_at, revealed_until=revealed_until)

    # 5. Requests: trinket (place Totem/Farsight, or Oracle sweep) and Control Ward.
    req = request.kind
    dist = jnp.sqrt((request.x - x) ** 2 + (request.y - y) ** 2)
    walk, region = _lookup(grid, request.x, request.y)
    is_sweep = (req == REQ_TRINKET) & (tid == ORACLE_ITEM)
    place_type = jnp.where(req == REQ_CONTROL, WardType.CONTROL,
                           jnp.where(tid == FARSIGHT_ITEM, WardType.FARSIGHT, WardType.TOTEM)).astype(jnp.int32)
    rng = jnp.where(place_type == WardType.FARSIGHT, FARSIGHT_RANGE, TOTEM_RANGE)
    has_item = jnp.where(req == REQ_CONTROL, control_count > 0, tid != 0)
    needs_charge = req == REQ_TRINKET
    code = jnp.where(req == REQ_NONE, ERR_NONE,
           jnp.where(~can_use, ERR_DEAD,
           jnp.where(~has_item, ERR_NO_ITEM,
           jnp.where(needs_charge & (charges < 1), ERR_NO_CHARGE,
           jnp.where(needs_charge & (now < tr.lock_until), ERR_LOCKED,
           jnp.where(~is_sweep & (dist > rng), ERR_RANGE,
           jnp.where(~is_sweep & ~walk, ERR_TERRAIN, OK)))))))
    ok = code == OK
    sweep = ok & is_sweep
    place = ok & ~is_sweep
    used = ok & needs_charge
    charges = charges - used.astype(jnp.int32)
    lock_until = jnp.where(used, now + jnp.where(is_sweep, ORACLE_LOCKOUT, TOTEM_LOCKOUT), tr.lock_until)
    oracle_until = jnp.where(sweep, now + ORACLE_DURATION, oracle_until)

    # Deep Ward (domination kernel) on Totem placements.
    page = jnp.zeros((c, 1), jnp.int32) if rune_pages is None else rune_pages
    enemy_jungle = jnp.where(team == W.BLUE, region == REGION_RED_JUNGLE, region == REGION_BLUE_JUNGLE)
    if rune_pages is None:
        deep_hp, deep_dur, deep = jnp.zeros((c,), jnp.float32), jnp.zeros((c,), jnp.float32), jnp.zeros((c,), bool)
    else:
        deep_hp, deep_dur, deep = DOM.deep_ward(page, arange_c, level, avg, enemy_jungle, region == REGION_RIVER,
                                                place_type == WardType.TOTEM)
    hp_new = jnp.asarray(WARD_HP, jnp.float32)[place_type] + deep_hp
    dur = jnp.where(place_type == WardType.TOTEM, _lerp(TOTEM_DURATION, avg) + deep_dur, jnp.inf)

    replaced = jnp.zeros((s,), bool)
    placed_slot = jnp.full((c,), -1, jnp.int32)
    seq = w.next_seq
    big = jnp.int32(2 ** 30)
    for i in range(c):                                   # static, C = 2
        typ = place_type[i]
        mine = sl.alive & (sl.owner == i) & (sl.type == typ)
        cap = jnp.where(typ == WardType.TOTEM, TOTEM_CAP, jnp.where(typ == WardType.CONTROL, CONTROL_CAP, s))
        over = jnp.sum(mine) >= cap
        in_team = steam == team[i]
        free = in_team & ~sl.alive
        oldest_mine = jnp.argmin(jnp.where(mine, sl.seq, big))
        oldest_team = jnp.argmin(jnp.where(in_team & sl.alive, sl.seq, big))
        k = jnp.where(over, oldest_mine, jnp.where(jnp.any(free), jnp.argmax(free), oldest_team)).astype(jnp.int32)
        p = place[i]
        replaced = replaced.at[k].set(replaced[k] | (p & sl.alive[k]))
        put = lambda arr, v: arr.at[k].set(jnp.where(p, jnp.asarray(v, arr.dtype), arr[k]))
        sl = WardState(
            alive=put(sl.alive, True), type=put(sl.type, typ), owner=put(sl.owner, i),
            x=put(sl.x, request.x[i]), y=put(sl.y, request.y[i]), placed_at=put(sl.placed_at, now),
            expires_at=put(sl.expires_at, now + dur[i]), hp=put(sl.hp, hp_new[i]), max_hp=put(sl.max_hp, hp_new[i]),
            bounty=put(sl.bounty, jnp.asarray(WARD_BOUNTY, jnp.float32)[typ]), early_paid=put(sl.early_paid, False),
            last_damaged=put(sl.last_damaged, -jnp.inf), regen_at=put(sl.regen_at, jnp.inf),
            revealed_until=put(sl.revealed_until, -jnp.inf), disabled_until=put(sl.disabled_until, -jnp.inf),
            triggered_at=put(sl.triggered_at, jnp.inf), tracked=put(sl.tracked, False), deep=put(sl.deep, deep[i]),
            seq=put(sl.seq, seq))
        placed_slot = placed_slot.at[i].set(jnp.where(p, k, -1))
        seq = seq + p.astype(jnp.int32)

    tr = TrinketState(tid, charges.astype(jnp.int32), progress, lock_until.astype(jnp.float32),
                      oracle_until.astype(jnp.float32), tr.sixth_cd_until)

    # 6. Farsight trigger: a live enemy champion within its current (unobstructed) sight.
    fr = jnp.where(now < sl.placed_at + FARSIGHT_REVEAL_TIME, FARSIGHT_REVEAL_SIGHT, WARD_SIGHT[2])
    d2c = (sl.x[:, None] - x[None, :]) ** 2 + (sl.y[:, None] - y[None, :]) ** 2           # (S, C)
    spot = jnp.any((d2c <= (fr ** 2)[:, None]) & alive[None, :] & (team[None, :] != steam[:, None]), axis=1)
    disabled_now, _, _ = _disable(sl, tr, now, x, y, team, alive, level)
    trig = sl.alive & (sl.type == WardType.FARSIGHT) & spot & ~disabled_now & ~jnp.isfinite(sl.triggered_at)
    sl = sl._replace(triggered_at=jnp.where(trig, now, sl.triggered_at))

    # 7. Oracle linger: wards inside an enemy sweep stay disabled 2 s after leaving it.
    in_oracle = jnp.any(_oracle_cover(sl, tr, now, x, y, team, alive, level), axis=0) & _stealthed(sl, now)
    sl = sl._replace(disabled_until=jnp.where(in_oracle, now + ORACLE_LINGER, sl.disabled_until))

    # 8. Sixth Sense (domination kernel).
    sensed = jnp.zeros((c,), bool)
    if rune_pages is not None:
        vis = jnp.zeros((2, s), bool) if ward_visible is None else ward_visible
        unseen = ~vis[jnp.clip(team, 0, 1)]
        cd, pick, rev = DOM.sixth_sense(rune_pages, tr.sixth_cd_until, now, level, alive, x, y, team,
                                        sl.alive, sl.x, sl.y, steam, unseen, sl.tracked)
        sensed = jnp.any(pick, axis=1)
        picked = jnp.any(pick, axis=0)
        rev_s = jnp.any(pick & rev[:, None], axis=0)
        sl = sl._replace(tracked=sl.tracked | picked,
                         revealed_until=jnp.where(rev_s, jnp.maximum(sl.revealed_until,
                                                                          now + DOM.ea(DOM.SIXTH_SENSE, "RevealDuration")),
                                                  sl.revealed_until))
        tr = tr._replace(sixth_cd_until=cd)

    ev = WardEvents(code=code.astype(jnp.int32), placed=place, placed_slot=placed_slot, placed_type=place_type,
                    sweep_started=sweep, trinket_used=used, consumed_control=place & (req == REQ_CONTROL),
                    killed=killed, killer=killer, expired=expired, replaced=replaced, gold=gold, xp=xp,
                    sensed=sensed)
    return Wards(sl, tr, seq), ev


def _disable(sl: WardState, tr: TrinketState, now, cx, cy, cteam, calive, level):
    """(disabled (S,), control_exposing (S,), oracle cover (C, S))."""
    cover = _control_cover(sl, now)                                   # (S, S)
    oc = _oracle_cover(sl, tr, now, cx, cy, cteam, calive, level)     # (C, S)
    stealth = _stealthed(sl, now)
    not_control = sl.type != WardType.CONTROL
    by_control = jnp.any(cover, axis=0) & not_control
    by_oracle = (jnp.any(oc, axis=0) | (now < sl.disabled_until)) & stealth
    disabled = sl.alive & (by_control | by_oracle)
    exposing = sl.alive & (sl.type == WardType.CONTROL) & jnp.any(cover & stealth[None, :], axis=1)
    return disabled, exposing, oc


def ward_view(w: Wards, *, now, x, y, team, alive, level) -> tuple[WardView, Any]:
    """Per-slot world view (S,) and the Oracle true-sight radius per champion (C,).

    Champion inputs are those passed to the visibility call (final positions)."""
    sl, tr = w.slots, w.trinket
    now = jnp.float32(now)
    disabled, exposing, _ = _disable(sl, tr, now, x, y, team, alive, level)
    far = sl.type == WardType.FARSIGHT
    far_r = jnp.where((now < sl.placed_at + FARSIGHT_REVEAL_TIME) | jnp.isfinite(sl.triggered_at),
                      FARSIGHT_REVEAL_SIGHT, WARD_SIGHT[2])
    r = jnp.where(far, far_r, WARD_SIGHT[0])
    r = jnp.where(sl.alive & ~disabled, r, 0.0).astype(jnp.float32)
    ctrl = sl.alive & (sl.type == WardType.CONTROL)
    view = WardView(alive=sl.alive, x=sl.x, y=sl.y, team=_slot_team(sl.x.shape[0]), sub=sl.type, owner=sl.owner,
                    hp=sl.hp, max_hp=sl.max_hp, sight_radius=r, stealthed=_stealthed(sl, now),
                    true_sight=jnp.where(ctrl, WARD_SIGHT[1], 0.0).astype(jnp.float32),
                    unobstructed=sl.alive & far, exposed=sl.alive & ((now < sl.revealed_until) | exposing),
                    disabled=disabled, tracked=sl.tracked & sl.alive, expires_at=sl.expires_at)
    oracle = jnp.where((now < tr.oracle_until) & alive, oracle_radius(level) + WARD_RADIUS, 0.0).astype(jnp.float32)
    return view, oracle


def vision_kwargs(view: WardView, oracle, kind, sub, alive, *, ward_start: int) -> dict:
    """Optional ``vision.visibility`` inputs (N,) with the ward slots at
    ``[ward_start, ward_start + S)`` and champions at ``[0, C)``. ``kind``,
    ``sub`` and ``alive`` are the world arrays after the ward units are written."""
    n = kind.shape[0]
    s = view.alive.shape[0]
    c = oracle.shape[0]
    sl = slice(ward_start, ward_start + s)
    radius = MV.sight_radius(kind, sub, alive).at[sl].set(view.sight_radius)
    stealthed = jnp.zeros((n,), bool).at[sl].set(view.stealthed)
    true_sight = jnp.zeros((n,), jnp.float32).at[sl].set(view.true_sight).at[:c].set(oracle)
    unobstructed = jnp.zeros((n,), bool).at[sl].set(view.unobstructed)
    exposed = jnp.zeros((n,), bool).at[sl].set(view.exposed)
    return dict(radius=radius, stealthed=stealthed, true_sight=true_sight, unobstructed=unobstructed,
                exposed=exposed)
