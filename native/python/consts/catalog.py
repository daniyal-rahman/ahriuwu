"""Catalog row orders (items, runes) shared by every native module."""


def consts():
    from lanerl_jax.modern.items.catalog import catalog
    from lanerl_jax.modern.runes.catalog import rune_catalog
    return {"catalog.ids": list(catalog().ids), "runes.ids": list(rune_catalog().ids)}
