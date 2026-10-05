"""Read-only views of a ``ModernState`` shared by the tick phases and the observation:
unit/kit/item contexts, composed champion stats, fog (``visibility``), terrain lookups (brush,
river), slot resets and small derived quantities. Nothing here writes the state."""
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
from ..items.effects.core import Ctx, Units
from ..items.loadout import stat_shard_stats
from ..jungle import camps as J
from ..jungle import objectives as OBJ
from ..map import regions as REG
from ..map.rift import vision_for
from ..runes.catalog import RunePage
from .config import N_CHAMPIONS, WorldConfig
from .config import layout as MW_layout

if TYPE_CHECKING:
    from .state import ModernState

SKILL_ORDERS = {86: (2, 0, 1, 2, 2, 3, 2, 0, 2, 0, 3, 0, 0, 1, 1, 3, 1, 1),     # Garen E>Q>W (modern.skill_ranks)
                24: (2, 0, 1, 1, 1, 3, 1, 2, 1, 2, 3, 2, 2, 0, 0, 3, 0, 0)}     # Jax W>E>Q


def shard_stats(cfg: WorldConfig, level) -> ItemStats:
    """Static shard stats per champion (adaptive left unresolved for STAT.50)."""
    parts = []
    for c, lo in enumerate(cfg.loadouts):
        if isinstance(lo.rune_page, RunePage):
            sh = stat_shard_stats(lo.rune_page.shards, level=level[c], adaptive_to_ad=None)
        else:
            sh = zero_stats(())
        parts.append(sh)
    return ItemStats(*(jnp.stack([jnp.asarray(getattr(p, k), jnp.float32) for p in parts])
                       for k in ItemStats._fields))


def visibility(cfg: WorldConfig, x, y, kind, sub, team, alive, reveal, now, *, wards=None, level=None,
                variant=None, jungle=None):
    """``(visible (2, N), sight (N, N))``; everything live is visible when ``cfg.vision`` is None.

    Wards (``wards``) add sight, stealth and true sight; ``variant`` selects the Elemental
    Rift / Baron-pit brush layout."""
    n = x.shape[0]
    if cfg.vision is None:
        live = alive & (kind != W.KIND_NONE)
        return jnp.broadcast_to(live[None, :], (2, n)), jnp.broadcast_to(live[None, :], (n, n))
    grid = cfg.vision
    if cfg.rift is not None and variant is not None:
        grid = vision_for(cfg.rift, variant, cfg.vision)
    kw = {}
    lay = MW_layout()
    if wards is not None:
        c = N_CHAMPIONS
        lv = jnp.ones((c,), jnp.int32) if level is None else level
        view, oracle = WD.ward_view(wards, now=now, x=x[:c], y=y[:c], team=team[:c], alive=alive[:c], level=lv)
        kw = WD.vision_kwargs(view, oracle, kind, sub, alive, ward_start=lay["ward0"])
    if jungle is not None:                                            # Scuttle Speed Shrines (525 sight)
        on = jungle.shrine_until > now
        kw["sources"] = (jnp.asarray(J.SHRINE_POS, jnp.float32)[:, 0], jnp.asarray(J.SHRINE_POS, jnp.float32)[:, 1],
                         jnp.where(on, J.SHRINE_SIGHT, 0.0), jungle.shrine_team)
    return MV.visibility(x, y, kind, sub, team, alive, reveal, now, grid, n_fogged=lay["struct0"], **kw)


def kit_attack_target(kit_out) -> Any:
    """(C,) unit a kit asks the champion to attack (Jax Q landing on a champion), -1 none."""
    at = kit_out.attack_target
    return jnp.full((N_CHAMPIONS,), -1, jnp.int32) if at is None else at


def refresh_visibility(s: ModernState, cfg: WorldConfig) -> ModernState:
    """Recompute ``visible``/``sight`` from the current positions (after editing a state by hand,
    e.g. scenario resets); ``step`` does this itself at the end of every tick."""
    vis, sight = visibility(cfg, s.x, s.y, s.kind, s.sub, s.team, s.alive, s.reveal, s.t, wards=s.wards,
                             level=s.econ.level, variant=s.terrain_variant, jungle=s.jungle)
    return s._replace(visible=vis, sight=sight)


def monster_buff_stats(s: ModernState, cfg: WorldConfig, st0: ChampionStats, now) -> ItemStats:
    """Static bonuses from jungle buffs (Blue/Red/Scuttle shrine) and team objective buffs
    (drake stacks and soul, Hand of Baron)."""
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
    """(C,) bool: the position is a brush cell (vision-grid flag bit 0x1, Rift variant aware); False
    without fog (``cfg.vision`` None: no grid in the config)."""
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
    return tuple(lo.skill_order) or SKILL_ORDERS[{"Garen": 86, "Jax": 24}[lo.champion]]


def units_view(s: ModernState) -> W.WorldUnits:
    return W.WorldUnits(kind=s.kind, sub=s.sub, team=s.team, alive=s.alive, targetable=s.targetable & s.alive,
                        x=s.x, y=s.y, radius=s.radius, hp=s.hp, max_hp=s.max_hp, armor=s.armor,
                        magic_resist=s.mr, attack_damage=s.ad, attack_range=s.arange, attack_speed=s.aspeed,
                        move_speed=s.mspeed, spawn_seq=s.spawn_seq, spawn_time=s.spawn_time)


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
    """Displayed champion stats for the current state (items, shards, monster buffs, kit, last
    tick's dynamic stats and slows): the stats the tick uses for casts (observation helper)."""
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


def reset_slots(s: ModernState, mask) -> ModernState:
    """Fresh attack state and CC timers for (re)spawned slots."""
    put = lambda arr, v: jnp.where(mask, jnp.asarray(v, arr.dtype), arr)   # noqa: E731
    return s._replace(att=s.att._replace(target=put(s.att.target, -1), windup_left=put(s.att.windup_left, 0.0),
                                         cooldown_left=put(s.att.cooldown_left, 0.0)),
                      cc=M.CCTimers(*(put(v, 0.0) for v in s.cc)))


def item_units(s: ModernState) -> Any:
    """``items.effects.core.Units`` view of the world."""
    return Units(x=s.x, y=s.y, team=s.team, cls=W.damage_class(s.kind), alive=s.alive, hp=s.hp, max_hp=s.max_hp,
                 radius=s.radius, targetable=s.targetable & s.alive & (s.kind != W.KIND_WARD),
                 is_siege_or_super=(s.kind == W.KIND_MINION) & (s.sub >= 2),
                 bonus_hp=jnp.zeros_like(s.hp), armor=s.armor, magic_resist=s.mr)


def owned_items(inv) -> Any:
    return I.owned_counts(inv)


def decimal_team_level(s: ModernState) -> Any:
    """(2,) decimal champion level per team (one champion per team in the lane world)."""
    return E.decimal_level(s.econ.xp, jnp.where(s.econ.quest.complete, 20, 18))[jnp.asarray([0, 1])]
