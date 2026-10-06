"""Read-only views of a ``ModernState`` shared by the phases and the observation: kit/item contexts, composed
champion stats, fog (``visibility``) and terrain lookups (brush, river)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from .. import champions as K
from .. import economy as E
from .. import mechanics as M
from .. import vision as MV
from .. import wards as WD
from ..core import types as W
from ..core.stat_pipeline import ChampionStats, compose
from ..items import inventory as I
from ..items.catalog import ItemStats, combine_stats, zero_stats
from ..items.effects.core import Ctx
from ..items.loadout import stat_shard_stats
from ..jungle import camps as J
from ..jungle import objectives as OBJ
from ..map import regions as REG
from ..map.rift import vision_for
from ..runes.catalog import RunePage
from .config import N_CHAMPIONS, WorldConfig

if TYPE_CHECKING:
    from .state import ModernState


def shard_stats(cfg: WorldConfig, level) -> ItemStats:
    """Static shard stats per champion (adaptive left unresolved for STAT.50)."""
    parts = [stat_shard_stats(lo.rune_page.shards, level=level[c], adaptive_to_ad=None)
             if isinstance(lo.rune_page, RunePage) else zero_stats(()) for c, lo in enumerate(cfg.loadouts)]
    return ItemStats(*(jnp.stack([jnp.asarray(getattr(p, k), jnp.float32) for p in parts])
                       for k in ItemStats._fields))


def visibility(cfg: WorldConfig, x, y, kind, sub, team, alive, reveal, now, *, wards=None, level=None,
                variant=None, jungle=None):
    """``(visible (2, N), sight (C, N), dropped)``: team visibility, each champion's own sight and the sight
    rays past ``layout.ray_capacity``; everything live is visible when ``cfg.vision`` is None. ``variant``
    selects the Rift / Baron-pit brush layout."""
    n = x.shape[0]
    if cfg.vision is None:
        live = alive & (kind != W.KIND_NONE)
        return (jnp.broadcast_to(live[None, :], (2, n)), jnp.broadcast_to(live[None, :], (N_CHAMPIONS, n)),
                jnp.int32(0))
    grid = cfg.vision
    if cfg.rift is not None and variant is not None:
        grid = vision_for(cfg.rift, variant, cfg.vision)
    kw = {}
    lay = cfg.layout
    if wards is not None:
        c = N_CHAMPIONS
        lv = jnp.ones((c,), jnp.int32) if level is None else level
        view, oracle = WD.ward_view(wards, now=now, x=x[:c], y=y[:c], team=team[:c], alive=alive[:c], level=lv)
        kw = WD.vision_kwargs(view, oracle, kind, sub, alive, ward_start=lay.ward0)
    if jungle is not None:                                            # Scuttle Speed Shrines (525 sight)
        on = jungle.shrine_until > now
        kw["sources"] = (jnp.asarray(J.SHRINE_POS, jnp.float32)[:, 0], jnp.asarray(J.SHRINE_POS, jnp.float32)[:, 1],
                         jnp.where(on, J.SHRINE_SIGHT, 0.0), jungle.shrine_team)
    visible, sight, dropped = MV.visibility(x, y, kind, sub, team, alive, reveal, now, grid, n_fogged=lay.struct0,
                                            ray_capacity=lay.ray_capacity, sight_rows=N_CHAMPIONS, **kw)
    return visible, sight, dropped


def kit_attack_target(kit_out) -> Any:
    """(C,) unit a kit asks the champion to attack (Jax Q), -1 none."""
    at = kit_out.attack_target
    return jnp.full((N_CHAMPIONS,), -1, jnp.int32) if at is None else at


def refresh_visibility(s: ModernState, cfg: WorldConfig) -> ModernState:
    """Recompute ``visible``/``sight`` after editing a state by hand (``step`` does this every tick)."""
    vis, sight, _ = visibility(cfg, s.x, s.y, s.kind, s.sub, s.team, s.alive, s.reveal, s.t, wards=s.wards,
                             level=s.econ.level, variant=s.terrain_variant, jungle=s.jungle)
    return s._replace(visible=vis, sight=sight)


def monster_buff_stats(s: ModernState, cfg: WorldConfig, st0: ChampionStats, now) -> ItemStats:
    """Static bonuses from jungle buffs (Blue, Red, shrine) and team objective buffs (drakes, soul, Baron)."""
    c = N_CHAMPIONS
    parts = []
    if cfg.jungle is not None:
        b = J.buff_stats(s.jungle, now=now, level=s.econ.level, max_mana=st0.max_mana, max_hp=s.max_hp[:c],
                         x=s.x[:c], y=s.y[:c], team=s.team[:c],
                         champion_combat_recent=(now - s.combat.clocks.last_champion_combat) < 5.0)
        parts.append(zero_stats((c,))._replace(ability_haste=b.ability_haste, mana_regen=b.mana_per_s,
                                               health_regen=b.hp_regen_per_s, move_speed=b.shrine_ms))
    if cfg.objectives is not None:
        parts.append(OBJ.team_buff_stats(s.obj, cfg.objectives, s.team[:c], s.alive[:c],
                                         st0.base_ad + st0.bonus_ad, st0.ap, now=now,
                                         out_of_combat=(now - s.combat.clocks.last_combat) >= 5.0))
    out = zero_stats((c,))
    for p in parts:
        out = combine_stats(out, p)
    return out


def in_brush(cfg: WorldConfig, x, y, variant) -> Any:
    """(C,) bool: a brush cell (vision-grid flag bit 0x1, Rift variant aware); False without fog."""
    if cfg.vision is None:
        return jnp.zeros(x.shape, bool)
    grid = cfg.vision
    if cfg.rift is not None:
        grid = vision_for(cfg.rift, variant, cfg.vision)
    h, w = grid.flags.shape
    ix = jnp.floor((x - grid.min_x) / grid.cell_size).astype(jnp.int32)
    iy = jnp.floor((y - grid.min_y) / grid.cell_size).astype(jnp.int32)
    ok = (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h)
    return ok & ((grid.flags[jnp.clip(iy, 0, h - 1), jnp.clip(ix, 0, w - 1)] & 1) != 0)


def in_river(cfg: WorldConfig, x, y) -> Any:
    if cfg.regions is None:
        return jnp.zeros(x.shape, bool)
    return REG.in_river(x, y, cfg.regions)


def skill_order(cfg: WorldConfig, c: int) -> tuple:
    lo = cfg.loadouts[c]
    return tuple(lo.skill_order) or K.kit(lo.champion).SKILL_ORDER


def static_stats(s: ModernState, cfg: WorldConfig, caps: dict, now, dt) -> tuple[ItemStats, ChampionStats]:
    """``(static, composed)``: items + shards + monster buffs + kit stats, without dynamic stats or slows."""
    level = s.econ.level
    base_static = combine_stats(I.inventory_stats(s.champ.inventory), shard_stats(cfg, level))
    st0 = compose(cfg.champion_base, level, base_static, adaptive_physical=cfg.adaptive_physical)
    base_static = combine_stats(base_static, monster_buff_stats(s, cfg, st0, now))
    st0 = compose(cfg.champion_base, level, base_static, adaptive_physical=cfg.adaptive_physical)
    static = combine_stats(base_static, K.stats(s.kits, kit_ctx(s, cfg, st0, caps, now, dt)))
    return static, compose(cfg.champion_base, level, static, adaptive_physical=cfg.adaptive_physical)


def champion_stats(s: ModernState, cfg: WorldConfig) -> ChampionStats:
    """Champion stats of the current state as the tick uses them for casts (observation helper)."""
    caps = M.capabilities(s.cc, s.t)
    static, _ = static_stats(s, cfg, caps, s.t, cfg.dt)
    return compose(cfg.champion_base, s.econ.level, combine_stats(static, s.champ.dyn),
                   adaptive_physical=cfg.adaptive_physical, slow=caps["slow"][:N_CHAMPIONS])


def kit_ctx(s: ModernState, cfg: WorldConfig, st: ChampionStats, caps: dict, now, dt) -> K.KitCtx:
    c = N_CHAMPIONS
    lv = s.econ.level
    return K.KitCtx(unit=jnp.arange(c, dtype=jnp.int32), champion_id=cfg.champion_ids, team=s.team[:c],
                    alive=s.alive[:c], level=lv, ranks=s.champ.ranks, x=s.x[:c], y=s.y[:c], mana=s.champ.mana,
                    max_mana=st.max_mana, hp=s.hp[:c], max_hp=s.max_hp[:c], base_ad=st.base_ad, bonus_ad=st.bonus_ad,
                    ap=st.ap, bonus_hp=s.max_hp[:c] - st.base_hp, armor=st.base_armor + st.bonus_armor,
                    magic_resist=st.base_mr + st.bonus_mr, bonus_attack_speed=st.bonus_attack_speed,
                    crit_chance=st.crit_chance, crit_damage=st.crit_damage, ability_haste=st.basic_ability_haste,
                    ultimate_haste=st.ultimate_haste, cooldowns=s.champ.cooldowns, now=now, dt=jnp.float32(dt),
                    silenced=caps["silenced"][:c], stunned=caps["stunned"][:c],
                    in_combat_ms_since_damaged=now - s.champ.last_damaged,
                    attack_target_kind=jnp.where(s.champ.attack_order >= 0,
                                                 s.kind[jnp.clip(s.champ.attack_order, 0, s.kind.shape[0] - 1)],
                                                 W.KIND_NONE).astype(jnp.int32),
                    rooted=s.cc.root_until[:c] > now)


def item_ctx(s: ModernState, cfg: WorldConfig, st: ChampionStats, now, dt) -> Ctx:
    c = N_CHAMPIONS
    return Ctx(now=now, dt=jnp.float32(dt), unit=jnp.arange(c, dtype=jnp.int32), team=s.team[:c], alive=s.alive[:c],
               level=s.econ.level.astype(jnp.float32), is_ranged=st.attack_range > 300.0, x=s.x[:c], y=s.y[:c],
               facing_x=s.champ.facing[:, 0], facing_y=s.champ.facing[:, 1], moved=jnp.zeros((c,), jnp.float32),
               base_ad=st.base_ad, bonus_ad=st.bonus_ad, ap=st.ap, base_hp=st.base_hp, max_hp=s.max_hp[:c],
               hp=s.hp[:c], base_armor=st.base_armor, bonus_armor=st.bonus_armor, base_mr=st.base_mr,
               bonus_mr=st.bonus_mr, mana=s.champ.mana, max_mana=st.max_mana,
               base_ms=cfg.champion_base.base_ms, move_speed=st.move_speed, crit_chance=st.crit_chance,
               crit_damage=st.crit_damage, life_steal=st.life_steal, bonus_attack_speed=st.bonus_attack_speed,
               ability_haste=st.basic_ability_haste, lethality=st.lethality, heal_shield_power=st.heal_shield_power,
               attack_windup=st.attack_windup, in_combat=(now - s.combat.clocks.last_combat) < 5.0,
               in_shop=I.in_shop_area(s.x[:c], s.y[:c], s.team[:c], ~s.alive[:c]),
               base_mana=cfg.champion_base.base_mana, attack_range=st.attack_range)


def decimal_team_level(s: ModernState) -> Any:
    """(2,) decimal champion level per team (one champion per team in the lane world)."""
    return E.decimal_level(s.econ.xp, jnp.where(s.econ.quest.complete, 20, 18))[jnp.asarray([0, 1])]
