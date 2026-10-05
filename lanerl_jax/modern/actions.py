"""Decode screen-click actions ``(button, screen_x, screen_y[, choice])`` into ``ModernOrders``
(profile ``modern-world-v2``). The policy names no unit: a click hit-tests the client selection circles of
units its team can see (nearest centre wins), in the acting champion's lane frame.

    move / attack_move  hostile under the cursor -> attack it, else move / attack-move to the point
    q, w, e, r          cast; target = the hostile under the cursor (or -1), point = the cursor
    summoner_d / f      cast; target = any unit under the cursor, point = the cursor
    recall, stop        start Recall; clear move/attack orders
    level_q..r          spend a skill point (otherwise the default skill order spends it)
    buy / sell          catalog row / inventory slot ``choice`` (the world checks shop rules)
    use_item            activate the item in slot ``choice``, aimed at the cursor
    ward, control_ward  trinket / Control Ward at the cursor

Clicks off screen or over the minimap are no-ops for every cursor button.
"""
from __future__ import annotations

import jax.numpy as jnp

from .core import types as W
from .items.catalog import catalog
from .screen import MINIMAP_X_MIN, MINIMAP_Y_MIN, N_SCREEN_X, N_SCREEN_Y, screen_to_lane
from .world.config import N_CHAMPIONS
from .world.state import no_orders

PROFILE = "modern-world-v2"
MODERN_BUTTONS = ("noop", "move", "attack_move", "q", "w", "e", "r", "recall", "summoner_d", "summoner_f",
                  "level_q", "level_w", "level_e", "level_r", "buy", "sell", "use_item", "ward", "control_ward", "stop")
MODERN_BUTTON_INDEX = {name: i for i, name in enumerate(MODERN_BUTTONS)}
N_INVENTORY_SLOTS = 7
# Buttons whose order reads the cursor; the PPO likelihood counts the click heads only for these.
SCREEN_BUTTONS = ("move", "attack_move", "q", "w", "e", "r", "summoner_d", "summoner_f", "use_item", "ward",
                  "control_ward")
CHOICE_BUTTONS = ("buy", "sell", "use_item")
# Client selection radii (cdragon selectionRadius). Wards (world radius 1) get a click circle about one
# click cell wide.
CHAMPION_SELECTION_RADIUS = 120.0
MINION_SELECTION_RADIUS = (115.0, 115.0, 140.0, 145.0)        # melee, caster, siege, super
WARD_CLICK_RADIUS = 65.0


def selection_radius(state):
    """(N,) click hit-test radius: selection radii for champions/minions, the ward circle, else the world radius."""
    sub = jnp.clip(state.sub, 0, 3)
    r = jnp.where(state.kind == W.KIND_CHAMPION, CHAMPION_SELECTION_RADIUS, state.radius)
    r = jnp.where(state.kind == W.KIND_MINION, jnp.asarray(MINION_SELECTION_RADIUS, jnp.float32)[sub], r)
    return jnp.where(state.kind == W.KIND_WARD, jnp.maximum(state.radius, WARD_CLICK_RADIUS), r)


def screen_usage(button):
    """(...) float32: 1 where ``button`` uses the click."""
    used = jnp.zeros(jnp.shape(button), bool)
    for name in SCREEN_BUTTONS:
        used = used | (button == MODERN_BUTTON_INDEX[name])
    return used.astype(jnp.float32)


def shop_choice_mask():
    """(I,) bool: catalog rows a ``buy`` choice may name (in-store items)."""
    return jnp.asarray(catalog().arrays.in_store, bool)


def modern_orders_from(action, state, frames, *, cfg_x: int = N_SCREEN_X, cfg_y: int = N_SCREEN_Y):
    """``action = (button, sx, sy[, choice])``, each (C,) int; ``frames`` = one LaneFrame per champion."""
    if len(action) not in (3, 4):
        raise ValueError("screen-click actions are (button, screen_x, screen_y[, choice])")
    c = N_CHAMPIONS
    button, sx, sy = (jnp.asarray(a, jnp.int32) for a in action[:3])
    choice = jnp.asarray(action[3], jnp.int32) if len(action) == 4 else jnp.zeros_like(button)
    screen_x = (sx + 0.5) / cfg_x
    screen_y = (sy + 0.5) / cfg_y
    ds, dn = screen_to_lane(screen_x.astype(jnp.float32), screen_y.astype(jnp.float32))
    axis = jnp.stack([jnp.asarray(f.axis, jnp.float32) for f in frames])
    normal = jnp.stack([jnp.asarray(f.normal, jnp.float32) for f in frames])
    x = state.x[:c] + ds * axis[:, 0] + dn * normal[:, 0]
    y = state.y[:c] + ds * axis[:, 1] + dn * normal[:, 1]

    n = state.kind.shape[0]
    d2 = (state.x[None, :] - x[:, None]) ** 2 + (state.y[None, :] - y[:, None]) ** 2
    seen = state.visible[state.team[:c]]                                  # (C, N) team fog
    pick_r = selection_radius(state)
    hit = seen & (state.alive & state.targetable & (state.kind != W.KIND_NONE))[None, :] \
        & (jnp.arange(n)[None, :] != jnp.arange(c)[:, None]) & (d2 <= pick_r[None, :] ** 2)
    any_hit = jnp.any(hit, axis=1)
    picked = jnp.argmin(jnp.where(hit, d2, jnp.inf), axis=1).astype(jnp.int32)
    hostile = any_hit & (state.team[picked] != state.team[:c])
    enemy_target = jnp.where(hostile, picked, -1)

    invalid = (sx < 0) | (sx >= cfg_x) | (sy < 0) | (sy >= cfg_y) \
        | ((screen_x >= MINIMAP_X_MIN) & (screen_y >= MINIMAP_Y_MIN))
    b = MODERN_BUTTON_INDEX
    is_click = (button == b["move"]) | (button == b["attack_move"])
    attack = is_click & hostile & ~invalid
    move = (button == b["move"]) & ~hostile & ~invalid
    attack_move = (button == b["attack_move"]) & ~hostile & ~invalid
    ward_kind = jnp.where((button == b["ward"]) & ~invalid, 0,
                          jnp.where((button == b["control_ward"]) & ~invalid, 1, -1))
    cast = (button >= b["q"]) & (button <= b["r"])
    cast_slot = jnp.where(cast & ~invalid, button - b["q"], -1)
    summ = (button == b["summoner_d"]) | (button == b["summoner_f"])
    summ_slot = jnp.where(summ & ~invalid, button - b["summoner_d"], -1)
    ids = jnp.asarray(catalog().ids, jnp.int32)
    n_rows = ids.shape[0]
    row_ok = (choice >= 0) & (choice < n_rows) & shop_choice_mask()[jnp.clip(choice, 0, n_rows - 1)]
    buy = jnp.where((button == b["buy"]) & row_ok, ids[jnp.clip(choice, 0, n_rows - 1)], 0)
    slot = jnp.clip(choice, 0, N_INVENTORY_SLOTS - 1)
    held = state.champ.inventory.item[jnp.arange(c), slot]
    held_id = jnp.where((held >= 0) & (choice >= 0) & (choice < N_INVENTORY_SLOTS), ids[jnp.clip(held, 0, n_rows - 1)], 0)
    sell = jnp.where(button == b["sell"], held_id, 0)
    use = jnp.where(button == b["use_item"], held_id, 0)
    level = (button >= b["level_q"]) & (button <= b["level_r"])
    o = no_orders(c)
    return o._replace(
        buy=buy.astype(jnp.int32), sell=sell.astype(jnp.int32), item_active=use.astype(jnp.int32),
        attack_move=attack_move, ward_kind=ward_kind.astype(jnp.int32), ward_x=x, ward_y=y,
        level_up=jnp.where(level, button - b["level_q"], -1).astype(jnp.int32),
        move=move, move_x=x, move_y=y,
        attack=jnp.where(attack, enemy_target, -1).astype(jnp.int32),
        cast_slot=cast_slot.astype(jnp.int32), cast_target=enemy_target.astype(jnp.int32), cast_x=x, cast_y=y,
        summoner_slot=summ_slot.astype(jnp.int32),
        summoner_target=jnp.where(any_hit, picked, -1).astype(jnp.int32), summoner_x=x, summoner_y=y,
        recall=button == b["recall"], stop=button == b["stop"])
