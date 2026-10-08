"""Catalog arrays the inventory/shop kernels read (lanerl_jax/modern/items/inventory.py), by catalog row."""


def consts():
    import numpy as np

    from lanerl_jax.modern.items.catalog import catalog
    a = catalog().arrays
    out = {f"econ.inventory.{f}": np.asarray(getattr(a, f)).astype(np.float64)
           for f in ("item_id", "stats", "multiplicative", "total", "sell_value", "can_be_sold", "in_store",
                     "max_stack", "groups", "group_max", "group_purchase_cd", "trinket", "required_level",
                     "ranged_only", "blocked", "required_buff", "node_item", "node_parent", "node_total")}
    # group_purchase_cd is float32 already: keep its exact value.
    out["econ.inventory.group_purchase_cd"] = np.asarray(a.group_purchase_cd, np.float32)
    out["econ.inventory.stats"] = np.asarray(a.stats, np.float32)
    return out
