"""Observation for the 26.19 modern world (``world.tick.ModernState``).

Profile ``modern-world-v1`` (MODERN-005): the same ``Observation`` container,
entity layout, slot blocks and global vector as ``obs.builder`` so the policy
architecture is unchanged, with a modern ``self`` block of
``MODERN_WORLD_SELF_DIM`` columns. Checkpoints trained on the legacy 16- or
28-column ``self`` profiles are not compatible (different column meanings);
``PROFILE`` names the contract so callers can reject a mismatched resume.

    entities (32, 20)  valid, ds, dn, hp_frac, 6 type (champion, minion, turret,
                       inhibitor, nexus, other), 3 team (ally, enemy, neutral),
                       3 minion subtype (melee, caster, siege; super minions use
                       siege), then monster, epic monster, ward, control ward
    self     (32,)     lane_s, lane_n, hp_frac, level/20, gold, cs, 4 cooldown
                       fractions, ad, ap, armor, mr, is_dead, recalling,
                       is_garen, is_jax, enemy_is_garen, enemy_is_jax,
                       mana_frac, shield/max_hp, summoner D/F cooldown fractions,
                       fraction of the way to the next level, quest progress, quest complete,
                       in_combat, move_speed/500, attack_range/600,
                       unspent skill points/4, trinket charges/2
    inventory (7,)     catalog row per inventory slot (-1 empty), ``inventory_stack`` (7,)
    affordable (I,)    in-store catalog rows the shop would sell this champion now
    global   (6,)      clock, enemy_visible, 4x time-since-observed-cast

The rules of ``obs.builder`` carry over: only visible units are slotted
(``state.visible`` from ``vision``, on screen; a fogged unit is
absent, never kept with a stale position),
there is no slot-index feature, health is quantised to the bar, max health is
not fed, and the enemy's cast memory is witnessed events normalised by rank-1
cooldowns so neither the enemy's level nor ranks leak.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_rl.projection import target_on_screen

from ..obs.builder import (GLOBAL_DIM, HP_BAR_STEPS, N_SLOTS, NORM_AD, NORM_CS, NORM_DIST, NORM_GOLD, NORM_XY,
                           Observation, _topk_slots)
from ..obs.frame import LaneFrame, delta_to_lane, make_lane_frame, to_lane
from . import champions as K
from . import economy as E
from .champions.core import cooldown_row
from .core import damage as D
from .core import types as W
from .items import inventory as I
from .items.catalog import catalog
from .role_quest import THRESHOLD
from .world.views import champion_stats

__all__ = ["PROFILE", "MODERN_WORLD_SELF_DIM", "MODERN_ENTITY_DIM", "ModernObservation", "modern_frames",
           "build_modern_observation"]

PROFILE = "modern-world-v1"
MODERN_WORLD_SELF_DIM = 32
MODERN_ENTITY_DIM = 20


class ModernObservation(NamedTuple):
    """``obs.builder.Observation`` fields plus the shop/inventory view (profile modern-world-v1)."""
    entities: Any
    entity_pad_mask: Any
    self_vec: Any
    global_vec: Any
    slot_unit: Any
    inventory: Any          # (7,) int32 catalog row per slot, -1 empty
    inventory_stack: Any    # (7,) int32
    affordable: Any         # (I,) bool
_TYPES = (W.KIND_CHAMPION, W.KIND_MINION, W.KIND_TURRET, W.KIND_INHIBITOR, W.KIND_NEXUS)


def modern_frames(cfg) -> tuple[LaneFrame, LaneFrame]:
    """Per-team lane frames from the modern world's top outer turrets and Nexuses."""
    kind, team, sub, lane = (np.asarray(a) for a in (cfg.unit_kind, cfg.unit_team, cfg.unit_sub, cfg.unit_lane))
    xy = np.stack([np.asarray(cfg.unit_x), np.asarray(cfg.unit_y)], -1)
    outer = [xy[(kind == W.KIND_TURRET) & (team == t) & (sub == 0) & (lane == 2)][0] for t in (0, 1)]
    nexus = [xy[(kind == W.KIND_NEXUS) & (team == t)][0] for t in (0, 1)]
    return (make_lane_frame(outer[0], outer[1], nexus[0]), make_lane_frame(outer[1], outer[0], nexus[1]))


def build_modern_observation(state, me: int, frame: LaneFrame, cfg, *, horizon_s: float = 1200.0) -> Observation:
    """One champion's observation of the modern world. ``me`` is its unit index (0 or 1)."""
    n = state.kind.shape[0]
    other = 1 - me
    my_team = state.team[me]
    dx, dy = state.x - state.x[me], state.y - state.y[me]
    d2 = dx * dx + dy * dy
    not_me = jnp.arange(n) != me
    kind = state.kind
    ally = state.team == my_team
    enemy = (state.team != my_team) & (kind != W.KIND_NONE)
    screen_ds, screen_dn = delta_to_lane(frame, dx, dy)
    vis = state.visible[my_team] & state.alive & (kind != W.KIND_NONE)   # team fog (vision)
    base = vis & not_me & target_on_screen(screen_ds, screen_dn)
    structure = (kind == W.KIND_TURRET) | (kind == W.KIND_INHIBITOR) | (kind == W.KIND_NEXUS)
    enemy_champ = _topk_slots(d2, base & (kind == W.KIND_CHAMPION) & enemy, 1)
    ally_minion = _topk_slots(d2, base & (kind == W.KIND_MINION) & ally, 12)
    enemy_minion = _topk_slots(d2, base & (kind == W.KIND_MINION) & enemy, 12)
    turret = _topk_slots(d2, base & structure, 2)
    taken = jnp.zeros((n,), bool)
    for sel in (enemy_champ, ally_minion, enemy_minion, turret):
        taken = taken | jnp.any(jnp.arange(n)[:, None] == jnp.where(sel >= 0, sel, -1)[None, :], axis=1)
    spare = _topk_slots(d2, base & ~taken, 5)
    slot_unit = jnp.concatenate([enemy_champ, ally_minion, enemy_minion, turret, spare])
    assert slot_unit.shape[0] == N_SLOTS
    valid = slot_unit >= 0
    u = jnp.clip(slot_unit, 0, n - 1)
    ds, dn = delta_to_lane(frame, dx[u], dy[u])
    hp_frac = jnp.where(state.max_hp[u] > 0, state.hp[u] / state.max_hp[u], 0.0)
    hp_frac = jnp.round(hp_frac * HP_BAR_STEPS) / HP_BAR_STEPS
    k = kind[u]
    type_1h = jnp.stack([k == t for t in _TYPES] + [(k == W.KIND_MONSTER) | (k == W.KIND_WARD)],
                        axis=-1).astype(jnp.float32)
    neutral = state.team[u] == W.NEUTRAL
    team_1h = jnp.stack([state.team[u] == my_team, (state.team[u] != my_team) & ~neutral,
                         neutral], axis=-1).astype(jnp.float32)
    sub = jnp.minimum(state.sub[u], 2)                                # super -> siege column
    sub_1h = jnp.where((k == W.KIND_MINION)[:, None],
                       jnp.stack([sub == 0, sub == 1, sub == 2], axis=-1).astype(jnp.float32), 0.0)
    lay = cfg.layout
    is_mon = k == W.KIND_MONSTER
    epic = is_mon & (u >= lay.epic0) & (u < lay.ward0)
    is_ward = k == W.KIND_WARD
    control = is_ward & (state.sub[u] == 1)                          # wards.WardType.CONTROL
    extra = jnp.stack([is_mon, epic, is_ward, control], axis=-1).astype(jnp.float32)
    entities = jnp.concatenate([valid[:, None].astype(jnp.float32), (ds / NORM_DIST)[:, None],
                                (dn / NORM_DIST)[:, None], hp_frac[:, None], type_1h, team_1h, sub_1h, extra],
                               axis=-1)
    entities = jnp.where(valid[:, None], entities, 0.0)

    # ---- self ---------------------------------------------------------------------
    c = state.champ
    level = state.econ.level
    st = champion_stats(state, cfg)
    s_, n_ = to_lane(frame, state.x[me], state.y[me])
    my_hp = jnp.where(state.max_hp[me] > 0, state.hp[me] / state.max_hp[me], 0.0)
    cid = cfg.champion_ids
    def rank_cd(who, ranks):
        """(4,) base cooldowns at ``ranks`` of the champion in slot ``who`` (kit registry order)."""
        out = cooldown_row(K.KITS[-1].NAME, ranks[None])
        for k in reversed(K.KITS[:-1]):
            out = jnp.where(cid[who] == k.ID, cooldown_row(k.NAME, ranks[None]), out)
        return out[0]
    base_cd = rank_cd(me, jnp.maximum(c.ranks[me], 1))
    cooldowns = jnp.where(c.ranks[me] > 0, jnp.clip(c.cooldowns[me] / jnp.maximum(base_cd, 1e-3), 0.0, 1.0), 1.0)
    shield = D.total_shield(state.shields, state.t)[me]
    summ_cd = jnp.clip((state.summoners.ready_at[me, :2] - state.t) / 300.0, 0.0, 1.0)
    cap = jnp.where(state.econ.quest.complete[me], E.QUEST_LEVEL_CAP, E.LEVEL_CAP)
    nxt = E.decimal_level(state.econ.xp[me], cap) - level[me]
    in_combat = (state.t - state.combat.clocks.last_champion_combat[me]) < 5.0
    self_vec = jnp.stack([
        s_ / NORM_XY, n_ / NORM_XY, jnp.round(my_hp * HP_BAR_STEPS) / HP_BAR_STEPS,
        level[me].astype(jnp.float32) / 20.0, state.econ.gold[me] / NORM_GOLD, c.cs[me].astype(jnp.float32) / NORM_CS,
        cooldowns[0], cooldowns[1], cooldowns[2], cooldowns[3],
        (st.base_ad[me] + st.bonus_ad[me]) / NORM_AD, st.ap[me] / NORM_AD,
        (st.base_armor[me] + st.bonus_armor[me]) / NORM_AD, (st.base_mr[me] + st.bonus_mr[me]) / NORM_AD,
        (~state.alive[me]).astype(jnp.float32), state.econ.recall.channeling[me].astype(jnp.float32),
        *((cid[me] == k.ID).astype(jnp.float32) for k in K.KITS),
        *((cid[other] == k.ID).astype(jnp.float32) for k in K.KITS),
        c.mana[me] / jnp.maximum(st.max_mana[me], 1.0), shield / jnp.maximum(state.max_hp[me], 1.0),
        summ_cd[0], summ_cd[1], nxt, jnp.clip(state.econ.quest.points[me] / THRESHOLD, 0.0, 1.0),
        state.econ.quest.complete[me].astype(jnp.float32), in_combat.astype(jnp.float32),
        st.move_speed[me] / 500.0, st.attack_range[me] / 600.0,
        (E.skill_points(level[me]) + c.bonus_points[me] - jnp.sum(c.ranks[me])).astype(jnp.float32) / 4.0,
        state.wards.trinket.charges[me].astype(jnp.float32) / 2.0])
    assert self_vec.shape[0] == MODERN_WORLD_SELF_DIM

    # ---- global -------------------------------------------------------------------
    enemy_visible = (enemy_champ[0] >= 0).astype(jnp.float32)
    rank1 = rank_cd(other, jnp.ones((4,), jnp.int32))
    # Witnessed casts only: the enemy was visible to this team when it cast (``seen_cast``).
    since = state.t - c.seen_cast[other]
    since_observed = jnp.where(c.seen_cast[other] > -1e8, jnp.clip(since / jnp.maximum(rank1, 1e-3), 0.0, 1.0), 1.0)
    global_vec = jnp.concatenate([jnp.stack([state.t / horizon_s, enemy_visible]), since_observed])
    assert global_vec.shape[0] == GLOBAL_DIM
    cat = catalog()
    in_shop = I.in_shop_area(state.x[me], state.y[me], my_team, ~state.alive[me])
    inv_me = I.Inventory(c.inventory.item[me], c.inventory.stack[me])
    rows = jnp.arange(len(cat.ids), dtype=jnp.int32)
    # Exactly the shop's own check (gold after owned components, groups, level/ranged gates).
    ok = jax.vmap(lambda r: I.buy(inv_me, state.econ.gold[me], r, can_shop=in_shop, level=level[me],
                                  is_ranged=st.attack_range[me] > 300.0, now=state.t,
                                  group_cd_until=c.group_cd[me]).ok)(rows)
    affordable = jnp.asarray(cat.arrays.in_store, bool) & ok & ~c.forbid[me]
    return ModernObservation(entities=entities, entity_pad_mask=~valid, self_vec=self_vec, global_vec=global_vec,
                       slot_unit=slot_unit, inventory=c.inventory.item[me], inventory_stack=c.inventory.stack[me],
                       affordable=affordable)
