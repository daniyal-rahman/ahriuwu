"""26.19 SR inventory and shop rules as fixed-shape JAX kernels.

ITEMS.md §3–5. One champion's inventory is ``item (7,)`` catalog rows
(``EMPTY`` = -1; slot 6 is the trinket) and ``stack (7,)``. Kernels are
written per champion; ``jax.vmap`` them over champions/environments.

Buying consumes owned recipe components depth-first in pre-order (an owned
component claims its whole subtree), pays ``total - Σ total(claimed)``, then
checks item-group limits on the remaining inventory (combining a recipe
therefore bypasses the limit for the consumed member, ITEMS.md §3), slot
availability, gold, shop range/death, level, ranged-only and purchase-buff
gates and the group purchase cooldown (Elixirs, 5 s).

Known simplifications (ITEMS.md §4.2): no undo action; selling a stack sells
one unit; champion-locked and Smite-gated items are not purchasable.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .catalog import BUFF_CURRENCIES, EMPTY, MAX_RECIPE_NODES, N_SLOTS, STAT_FIELDS, TRINKET_SLOT, ItemStats, catalog

STARTING_GOLD = 500.0
SHOP_RADIUS = 1000.0
# ShopAreaCenter locators from map11 geometry (ITEMS.md §4.1), (x, z) by team.
SHOP_CENTER = ((412.9, 416.2), (14297.2, 14388.3))
STEALTH_WARD = 3340

OK = 0
ERR_NOT_IN_SHOP = 1
ERR_NOT_PURCHASABLE = 2
ERR_LEVEL = 3
ERR_GOLD = 4
ERR_GROUP = 5
ERR_NO_SLOT = 6
ERR_COOLDOWN = 7
ERR_EMPTY_SLOT = 8
ERR_NOT_SELLABLE = 9


class Inventory(NamedTuple):
    item: Any       # (..., 7) int32 catalog rows
    stack: Any      # (..., 7) int32


def empty_inventory(n_champions: int = 2, *, trinket: bool = True) -> Inventory:
    cat = catalog()
    item = np.full((n_champions, N_SLOTS), EMPTY, np.int32)
    stack = np.zeros((n_champions, N_SLOTS), np.int32)
    if trinket:
        item[:, TRINKET_SLOT] = cat.row(STEALTH_WARD)
        stack[:, TRINKET_SLOT] = 1
    return Inventory(jnp.asarray(item), jnp.asarray(stack))


def inventory_from_ids(item_ids, *, trinket: bool = True) -> Inventory:
    """Host helper: per champion a list of item ids (stackables may repeat)."""
    cat = catalog()
    inv = empty_inventory(len(item_ids), trinket=trinket)
    item, stack = np.asarray(inv.item).copy(), np.asarray(inv.stack).copy()
    for c, ids in enumerate(item_ids):
        slot = 0
        for iid in ids:
            row = cat.row(iid)
            if cat.arrays.trinket[row]:
                item[c, TRINKET_SLOT], stack[c, TRINKET_SLOT] = row, 1
                continue
            same = np.nonzero((item[c, :TRINKET_SLOT] == row)
                              & (stack[c, :TRINKET_SLOT] < cat.arrays.max_stack[row]))[0]
            if cat.arrays.max_stack[row] > 1 and same.size:
                stack[c, same[0]] += 1
                continue
            if slot >= TRINKET_SLOT:
                raise ValueError("more than six inventory items")
            item[c, slot], stack[c, slot] = row, 1
            slot += 1
    return Inventory(jnp.asarray(item), jnp.asarray(stack))


def owned_counts(inv: Inventory, n_items: int | None = None) -> Any:
    """(..., I) units owned of each catalog row (stacks counted)."""
    n_items = len(catalog().ids) if n_items is None else n_items
    onehot = (inv.item[..., None] == jnp.arange(n_items)) & (inv.item[..., None] >= 0)
    return jnp.sum(onehot * inv.stack[..., None], axis=-2)


def owns(inv: Inventory, item_id: int) -> Any:
    """(...,) bool: champion holds at least one ``item_id``."""
    return jnp.any(inv.item == catalog().row(item_id), axis=-1)


def inventory_stats(inv: Inventory) -> ItemStats:
    """Static item stats of each inventory (STAT.20 contributions).

    Additive fields sum over slots; tenacity, slow resist and %pen stack as
    ``1 - prod(1 - x)``. Stacked items contribute stats once per slot.
    """
    a = catalog().arrays
    rows = jnp.clip(inv.item, 0, a.stats.shape[0] - 1)
    present = (inv.item >= 0) & (inv.stack > 0)
    per_slot = jnp.where(present[..., None], jnp.asarray(a.stats)[rows], 0.0)
    additive = jnp.sum(per_slot, axis=-2)
    multiplicative = 1.0 - jnp.prod(1.0 - per_slot, axis=-2)
    total = jnp.where(jnp.asarray(a.multiplicative), multiplicative, additive)
    return ItemStats(*(total[..., i] for i in range(len(STAT_FIELDS))))


def in_shop_area(x: Any, z: Any, team: Any, dead: Any) -> Any:
    centre = jnp.asarray(SHOP_CENTER, jnp.float32)[jnp.clip(team, 0, 1)]
    d2 = (x - centre[..., 0]) ** 2 + (z - centre[..., 1]) ** 2
    return dead | (d2 <= SHOP_RADIUS ** 2)


class ShopResult(NamedTuple):
    inv: Inventory
    gold: Any
    group_cd_until: Any
    ok: Any
    code: Any
    spent: Any


def buy(inv: Inventory, gold: Any, row: Any, *, can_shop: Any, level: Any, is_ranged: Any,
        now: Any = 0.0, group_cd_until: Any = None, buff_currency: Any = None) -> ShopResult:
    """Buy catalog ``row`` for one champion; failed purchases change nothing."""
    a = catalog().arrays
    can_shop, is_ranged = jnp.asarray(can_shop, bool), jnp.asarray(is_ranged, bool)
    n_groups = a.groups.shape[1]
    group_cd_until = jnp.zeros((n_groups,), jnp.float32) if group_cd_until is None else group_cd_until
    buff_currency = jnp.zeros((len(BUFF_CURRENCIES),), bool) if buff_currency is None else buff_currency
    row = jnp.asarray(row, jnp.int32)
    items, stacks = inv.item, inv.stack
    main = jnp.arange(N_SLOTS) < TRINKET_SLOT

    # Recipe consumption over the static pre-order tree.
    node_item = jnp.asarray(a.node_item)[row]
    node_parent = jnp.asarray(a.node_parent)[row]
    node_total = jnp.asarray(a.node_total)[row]

    def claim(carry, k):
        taken, covered, refund = carry
        it, parent = node_item[k], node_parent[k]
        parent_cov = jnp.where(parent >= 0, covered[jnp.maximum(parent, 0)], False)
        avail = (items == it) & (it >= 0) & ~taken & main & (stacks > 0)
        slot = jnp.argmax(avail)
        got = (it >= 0) & ~parent_cov & jnp.any(avail)
        taken = taken.at[slot].set(taken[slot] | got)
        covered = covered.at[k].set(parent_cov | got)
        return (taken, covered, refund + jnp.where(got, node_total[k], 0)), None

    (taken, _, discount), _ = jax.lax.scan(
        claim, (jnp.zeros((N_SLOTS,), bool), jnp.zeros((MAX_RECIPE_NODES,), bool), jnp.int32(0)),
        jnp.arange(MAX_RECIPE_NODES))
    cost = (jnp.asarray(a.total)[row] - discount).astype(jnp.float32)
    remaining_items = jnp.where(taken, EMPTY, items)
    remaining_stacks = jnp.where(taken, 0, stacks)

    groups = jnp.asarray(a.groups)
    member = groups[row]
    trinket = jnp.asarray(a.trinket)[row]
    max_stack = jnp.asarray(a.max_stack)[row]
    stackable = (remaining_items == row) & (remaining_stacks < max_stack) & main & (max_stack > 1)
    free = (remaining_items < 0) & main
    has_stack = jnp.any(stackable)

    # Group limits count occupied slots (a stack is one owned item); the
    # trinket slot is replaced by a trinket purchase, never added to.
    present = (remaining_items >= 0) & ~(trinket & (jnp.arange(N_SLOTS) == TRINKET_SLOT))
    slot_groups = jnp.where(present[:, None], groups[jnp.clip(remaining_items, 0)], False)
    held = jnp.sum(slot_groups, axis=0)
    gmax = jnp.asarray(a.group_max)
    adds = jnp.where(has_stack, 0, 1)
    group_ok = jnp.all(~member | (gmax < 0) | (held + adds <= gmax))
    cd_ok = jnp.all(~member | (now >= group_cd_until))
    has_free = jnp.any(free)
    slot = jnp.where(trinket, TRINKET_SLOT, jnp.where(has_stack, jnp.argmax(stackable), jnp.argmax(free)))
    slot_ok = trinket | has_stack | has_free

    need_buff = jnp.asarray(a.required_buff)[row]
    buff_ok = (need_buff == -1) | ((need_buff >= 0) & buff_currency[jnp.maximum(need_buff, 0)])
    purchasable = jnp.asarray(a.in_store)[row] & ~jnp.asarray(a.blocked)[row] & buff_ok \
        & (~jnp.asarray(a.ranged_only)[row] | is_ranged)
    level_ok = level >= jnp.asarray(a.required_level)[row]
    gold_ok = gold >= cost

    code = jnp.select(
        [~can_shop, ~purchasable, ~level_ok, ~cd_ok, ~group_ok, ~slot_ok, ~gold_ok],
        [ERR_NOT_IN_SHOP, ERR_NOT_PURCHASABLE, ERR_LEVEL, ERR_COOLDOWN, ERR_GROUP, ERR_NO_SLOT, ERR_GOLD],
        OK)
    ok = code == OK
    new_items = remaining_items.at[slot].set(row)
    new_stacks = remaining_stacks.at[slot].set(jnp.where(has_stack & ~trinket, remaining_stacks[slot] + 1, 1))
    gcd = jnp.asarray(a.group_purchase_cd)
    new_cd = jnp.where(member & (gcd > 0), now + gcd, group_cd_until)
    return ShopResult(
        Inventory(jnp.where(ok, new_items, items), jnp.where(ok, new_stacks, stacks)),
        jnp.where(ok, gold - cost, gold), jnp.where(ok, new_cd, group_cd_until), ok, code,
        jnp.where(ok, cost, 0.0))


def sell(inv: Inventory, gold: Any, slot: Any, *, can_shop: Any) -> ShopResult:
    """Sell one unit from ``slot`` at round_half_up(total × sellBackModifier)."""
    a = catalog().arrays
    can_shop = jnp.asarray(can_shop, bool)
    row = inv.item[slot]
    occupied = row >= 0
    sellable = occupied & jnp.asarray(a.can_be_sold)[jnp.maximum(row, 0)]
    code = jnp.select([~can_shop, ~occupied, ~sellable], [ERR_NOT_IN_SHOP, ERR_EMPTY_SLOT, ERR_NOT_SELLABLE], OK)
    ok = code == OK
    value = jnp.where(ok, jnp.asarray(a.sell_value)[jnp.maximum(row, 0)], 0).astype(jnp.float32)
    left = inv.stack[slot] - 1
    new_item = inv.item.at[slot].set(jnp.where(left > 0, row, EMPTY))
    new_stack = inv.stack.at[slot].set(jnp.maximum(left, 0))
    out = Inventory(jnp.where(ok, new_item, inv.item), jnp.where(ok, new_stack, inv.stack))
    return ShopResult(out, gold + value, None, ok, code, -value)


def replace_item(inv: Inventory, from_row: Any, to_row: Any, enabled: Any = True) -> Inventory:
    """Transform/distribute in place (Tear -> Muramana etc., ITEMS.md §3)."""
    hit = (inv.item == from_row) & jnp.asarray(enabled)
    first = jnp.argmax(hit)
    do = jnp.any(hit)
    return inv._replace(item=jnp.where(do, inv.item.at[first].set(to_row), inv.item))


def consume_one(inv: Inventory, slot: Any, enabled: Any = True) -> Inventory:
    """Use one charge/unit of a consumed item (potions, elixirs)."""
    do = jnp.asarray(enabled) & (inv.item[slot] >= 0) & (inv.stack[slot] > 0)
    left = inv.stack[slot] - 1
    item = inv.item.at[slot].set(jnp.where(do & (left <= 0), EMPTY, inv.item[slot]))
    stack = inv.stack.at[slot].set(jnp.where(do, jnp.maximum(left, 0), inv.stack[slot]))
    return Inventory(item, stack)


def validate_item_loadout(item_ids) -> None:
    """Host check of a fixed loadout against store, group and slot rules."""
    cat = catalog()
    for iid in item_ids:
        if not cat[iid].in_store:
            raise ValueError(f"item {iid} {cat[iid].name} is not purchasable on Map11 in patch 26.19")
    inv = inventory_from_ids([list(item_ids)], trinket=False)  # raises past six slots
    counts: dict[str, int] = {}
    for row in np.asarray(inv.item)[0]:
        if row < 0:
            continue
        for g in cat.specs[row].groups:
            counts[g] = counts.get(g, 0) + 1
            limit = int(cat.group_info[g]["max_ownable"])
            if 0 <= limit < counts[g]:
                raise ValueError(f"item-limit group {g!r} allows {limit}: {list(item_ids)}")
