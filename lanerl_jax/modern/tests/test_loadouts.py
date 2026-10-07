"""Item allow-lists: the closure over transforms and rune grants, compile-time ``holds``, the shop gate, the loader."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.data import loadouts as DL
from lanerl_jax.modern.items.catalog import catalog
from lanerl_jax.modern.items.effects import core as IC
from lanerl_jax.modern.items.effects.starters import ARCHANGELS, SERAPHS
from lanerl_jax.modern.items.inventory import STEALTH_WARD
from lanerl_jax.modern.items.loadout import RUNE_GRANTS, acquirable_rows, validate_rune_page
from lanerl_jax.modern.runes import catalog as RD
from lanerl_jax.modern.tests import world_harness as H
from lanerl_jax.modern.world import config as MW

LONG_SWORD, DORAN_BLADE, HEALTH_POTION, BOOTS = 1036, 1055, 2003, 1001


def test_acquirable_rows_close_over_transforms_and_rune_grants():
    cat = catalog()
    rows = acquirable_rows((ARCHANGELS,))
    assert {i for i in cat.ids if rows[cat.row(i)]} == {ARCHANGELS, SERAPHS, STEALTH_WARD, *RUNE_GRANTS}


def test_holds_is_a_compile_time_false_for_items_no_holder_can_hold():
    cat = catalog()
    counts = jnp.zeros((2, len(cat.ids)), jnp.int32).at[0, cat.row(LONG_SWORD)].set(1)
    allowed = np.zeros(counts.shape, bool)
    allowed[0, cat.row(DORAN_BLADE)] = True
    held = jax.make_jaxpr(lambda c: IC.holds(IC.Owned(c, allowed), LONG_SWORD))(counts)
    assert all(e.primitive.name == "broadcast_in_dim" for e in held.jaxpr.eqns)        # no read of the counts
    assert not IC.can_hold(IC.Owned(counts, allowed), (LONG_SWORD,))
    np.testing.assert_array_equal(IC.holds(IC.Owned(counts, allowed), DORAN_BLADE), [False, False])
    np.testing.assert_array_equal(IC.holds(counts, LONG_SWORD), [True, False])        # bare counts: unrestricted


@pytest.mark.skipif(not DL.DEFAULT_LOADOUTS.exists(), reason="allow-list research file not present")
def test_research_allowlist_loads_items_and_valid_pages():
    items = DL.allowed_items("Garen")
    assert 6631 in items and BOOTS in items and HEALTH_POTION in items          # Stridebreaker, basic Boots
    assert 3157 not in items                                                    # Zhonya's: not a Garen item
    pages = DL.rune_pages("Garen")
    assert 2 <= len(pages) <= 3 and pages[0].keystone == 8010                   # Conqueror first
    for page in pages:
        validate_rune_page(page)


@pytest.mark.skipif(not H.artifacts_present(), reason="modern map/route artifacts not present")
def test_shop_refuses_items_outside_the_allowlist():
    from lanerl_jax.modern import world as MS
    cfg = MW.build_config((MW.Loadout("Garen", items=H.ITEMS, rune_page=RD.GAREN_DEFAULT_PAGE,
                                      allowed_items=(DORAN_BLADE, HEALTH_POTION, BOOTS)),
                           MW.Loadout("Jax", items=H.ITEMS, rune_page=H.JAX_PAGE)),
                          lanes=(2,), jungle=False, objectives=False)
    s, e = jax.jit(lambda s, o: MS.step(s, o, cfg))(MS.init_state(cfg), H.orders(buy=[LONG_SWORD, LONG_SWORD]))
    rows = np.asarray(s.champ.inventory.item)
    assert catalog().row(LONG_SWORD) not in rows[0].tolist() and float(s.econ.gold[0]) == 500.0
    assert catalog().row(LONG_SWORD) in rows[1].tolist()                        # Jax is unrestricted
    assert int(e.item_overflow) == 0
