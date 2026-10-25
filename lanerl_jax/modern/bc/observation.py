"""Replay-only v4 observations; shared layout, no simulator state/config. BC.md section 14."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from .. import obs as OB
from ..data import PATCH_DIR
from ..data import champion_table as CT
from ..frame import make_lane_frame
from ..screen import target_on_screen
from . import fog
from .replay import BLUE, RED, MINION_SUB, lane_line


@lru_cache(maxsize=1)
def geometry():
    return json.loads((PATCH_DIR / "geometry.json").read_text())


def frames(g, lane):
    """Outer turrets from replay; orientation from the pinned client spawn anchors."""
    outer = [g.turret_xy[(team, lane, "outer")] for team in (BLUE, RED)]
    spawn = geometry()["spawn_gate_anchors"]
    return (make_lane_frame(outer[0], outer[1], spawn["OrderSpawnGate"]),
            make_lane_frame(outer[1], outer[0], spawn["ChaosSpawnGate"]))


def corridor(lane):
    spawn = geometry()["spawn_gate_anchors"]
    return (spawn["OrderSpawnGate"], *lane_line(lane), spawn["ChaosSpawnGate"])


def build(g, frame, col, role, tier, extra_champions, timers):
    """Return entities/padding/self/global and replay unit indices per observation row.

    Missing scalar evidence stays NaN. Empty entity slots are padding, not claims
    that undecoded entities or structures do not exist. Only replay units enter
    the selection; this function never initializes a world or composes kit stats.
    """
    T, U = g.x.shape
    me, opponent = g.index(g.me), g.index(g.opponent)
    team = g.units[me].team
    xy = np.stack([g.x[:, me], g.y[:, me]], -1)
    dx, dy = g.x - xy[:, :1], g.y - xy[:, 1:]
    axis, normal = np.asarray(frame.axis), np.asarray(frame.normal)
    ds, dn = dx * axis[0] + dy * axis[1], dx * normal[0] + dy * normal[1]
    screen = np.asarray(target_on_screen(ds, dn))
    all_slots = np.broadcast_to(np.arange(U), (T, U))
    visible = fog.team_visibility(g.units, g.x, g.y, g.alive, all_slots)[:, team]
    base = visible & g.alive & screen
    base[:, me] = False
    families = np.asarray([u.family for u in g.units])
    ally = np.asarray([u.team == team for u in g.units])
    champion, minion, turret = (families == f for f in ("champion", "minion", "turret"))
    eligible = base.copy()
    if not extra_champions:
        eligible[:, champion & (np.arange(U) != opponent)] = False
    selected = np.full((T, OB.N_SLOTS), -1, np.int32)
    entities = np.zeros((T, OB.N_SLOTS, OB.MODERN_ENTITY_DIM), np.float32)
    distance = dx * dx + dy * dy
    for k in range(T):
        taken = np.zeros(U, bool)
        def choose(mask, count):
            ids = np.flatnonzero(eligible[k] & mask & ~taken)
            ids = ids[np.argsort(distance[k, ids], kind="stable")[:count]]
            taken[ids] = True
            return ids
        blocks = [(0, choose(np.arange(U) == opponent, 1)),
                  (1, choose(minion & ally, 12)), (13, choose(minion & ~ally, 12)),
                  (25, choose(turret, 2))]
        spare = choose(champion, 5) if extra_champions else np.zeros(0, int)
        spare = np.concatenate([spare, choose(np.ones(U, bool), 5 - len(spare))])
        blocks.append((27, spare))
        for offset, ids in blocks:
            selected[k, offset:offset + len(ids)] = ids
        for slot in np.flatnonzero(selected[k] >= 0):
            u = int(selected[k, slot])
            row = entities[k, slot]
            row[:3] = (1., ds[k, u] / OB.NORM_DIST, dn[k, u] / OB.NORM_DIST)
            row[3] = (np.round(g.hp[k, u] / g.max_hp[k, u] * OB.HP_BAR_STEPS) / OB.HP_BAR_STEPS
                      if np.isfinite(g.max_hp[k, u]) and g.max_hp[k, u] > 0 else np.nan)
            row[4 + {"champion": 0, "minion": 1, "turret": 2}.get(families[u], 5)] = 1.
            row[10 + (0 if ally[u] else 1)] = 1.
            if minion[u]:
                row[13 + min(MINION_SUB[g.units[u].subtype], 2)] = 1.
    pad = selected < 0
    sv = np.full((T, OB.self_dim(OB.REPLAY_PROFILE)), np.nan, np.float32)
    def put(name, value):
        sv[:, col[name]] = value
    put("own_id", CT.index_of(g.units[me].subtype))
    put("enemy_id", CT.index_of(g.opponent_name))
    put("tier", tier)
    delta = xy - np.asarray(frame.origin)
    put("s", (delta @ axis) / OB.NORM_XY)
    put("n", (delta @ normal) / OB.NORM_XY)
    hp = np.where(g.max_hp[:, me] > 0, g.hp[:, me] / np.maximum(g.max_hp[:, me], 1e-6), np.nan)
    put("hp", np.round(hp * OB.HP_BAR_STEPS) / OB.HP_BAR_STEPS)
    put("dead", np.where(np.isfinite(g.hp[:, me]), ~g.alive[:, me], np.nan))
    put("level", g.level / 20.)
    put("gold", g.gold / OB.NORM_GOLD)
    put("cs", g.cs / OB.NORM_CS)
    stats = g.champion_stats[:, 0]
    for name, index in (("ad", 0), ("ap", 1), ("armor", 2), ("mr", 3)):
        put(name, stats[:, index] / OB.NORM_AD)
    put("ms", stats[:, 5] / 500.)
    put("range", stats[:, 9] / 600.)
    # A known zero maximum resource is a real no-resource observation; missing is not zero.
    put("mana", np.where(g.max_mana > 0, g.mana / np.maximum(g.max_mana, 1.),
                          np.where(g.max_mana == 0, 0., np.nan)))
    put("shield", g.shield / np.where(g.max_hp[:, me] > 0, g.max_hp[:, me], np.nan))
    put("in_combat", np.where(np.isfinite(g.last_champion_combat),
                              (g.t - g.last_champion_combat) < 5., np.nan))
    for slot, name in enumerate(("q", "w", "e", "r")):
        put("cd_" + name, timers[:, slot] / 60.)
        put("rank_" + name, g.spell_rank[:, 0, slot] / 5.)
    for raw, canonical in g.summoner_slot.items():
        put(("summ_d", "summ_f")[canonical], timers[:, raw] / 300.)
    for r in OB.ROLES:
        put("role_" + r, float(role == r))
    # Recall, XP fraction, skill points, quest, trinket and descriptors remain unavailable.
    gv = np.full((T, OB.GLOBAL_DIM), np.nan, np.float32)
    gv[:, 0] = g.t / 600.
    gv[:, 1] = selected[:, 0] >= 0
    for tc, slot in sorted(g.opponent_casts):
        if slot not in range(4):
            continue
        # Choose the last grid state at/before the event, never the following state.
        k = np.searchsorted(g.t, tc, side="right") - 1
        if k >= 0 and visible[k, opponent]:
            later = g.t >= tc
            gv[later, 2 + slot] = np.clip((g.t[later] - tc) / 60., 0., 1.)
    return entities, pad, sv, gv, selected, visible
