"""Combined item + rune tick (combat): STAT.70 sync, clocks, carry."""
import jax
import jax.numpy as jnp
import pytest

from lanerl_jax.modern import combat as M
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items import inventory as I
from lanerl_jax.modern.items.catalog import catalog
from lanerl_jax.modern.items.effects import runtime as R
from lanerl_jax.modern.tests import item_harness as H

IRON, BISCUIT = 2138, 2010


def setup(loadouts=((), ())):
    u = H.units(H.champions(x1=150.) + [dict(x=2000, y=0, team=1)])
    n = u.x.shape[0]
    inv = I.inventory_from_ids([list(l) for l in loadouts])
    own, item = I.owned_counts(inv), I.inventory_stats(inv)
    hp = jnp.asarray([1000., 1000., 500.], jnp.float32)
    u = u._replace(hp=hp, max_hp=hp)
    ctx = H.ctx(base_hp=1000., max_hp=1000.)
    dfn = D.default_defense(n)._replace(unit_class=u.cls)
    off = D.default_offense(n)._replace(unit_class=u.cls)
    return u, own, item, ctx, dfn, off


def run(state, own, item, ctx, u, dfn, off, hp, max_hp, *, now, request=(0, 0), base=None):
    n = u.x.shape[0]
    ctx = ctx._replace(now=jnp.float32(now), hp=hp[:2], max_hp=max_hp[:2])
    u = u._replace(hp=hp, max_hp=max_hp)
    return M.combat_tick(state, own, M.empty_page(2), ctx, u, attack=H.attack(hit=(False, False)),
                         cast=H.cast(started=(False, False)), request=jnp.asarray(request, jnp.int32),
                         base_packets=D.empty_packets(0) if base is None else base, base_offense=off,
                         base_defense=dfn, hp=hp, max_hp=max_hp, shields=D.init_shields(n),
                         status=R.init_status(n), kills=H.kills(n), holder_stats=item)


def test_dynamic_max_health_sync_heals_on_gain_except_silent_health():
    u, own, item, ctx, dfn, off = setup(([IRON, BISCUIT], ()))
    state = M.init_combat(2, u.x.shape[0])
    hp, mx = u.hp.at[0].set(600.), u.max_hp
    out = run(state, own, item, ctx, u, dfn, off, hp, mx, now=0.0, request=(IRON, 0))
    assert int(out.consume_row[0]) == catalog().row(IRON)
    out = run(out.state, own, item, ctx, u, dfn, off, out.hp, out.max_hp, now=0.1)
    # Elixir of Iron +300 max HP: a stat gain raises current HP by the same delta (STAT.70).
    assert float(out.max_hp[0]) == pytest.approx(1300.) and float(out.hp[0]) == pytest.approx(900.)
    out = run(out.state, own, item, ctx, u, dfn, off, out.hp, out.max_hp, now=0.2, request=(BISCUIT, 0))
    out = run(out.state, own, item, ctx, u, dfn, off, out.hp, out.max_hp, now=0.3)
    # A biscuit's +30 permanent max HP does not raise current HP (RUNES §7.3).
    assert float(out.max_hp[0]) == pytest.approx(1330.) and float(out.hp[0]) == pytest.approx(900.)
    # Elixir expiry: the loss only clamps (current HP is below the new max).
    late = run(out.state, own, item, ctx, u, dfn, off, out.hp, out.max_hp, now=200.0)
    assert float(late.max_hp[0]) == pytest.approx(1030.)
    assert float(late.hp[0]) <= 1030.0


def test_combat_clocks_and_struck_first():
    u, own, item, ctx, dfn, off = setup()
    state = M.init_combat(2, u.x.shape[0])
    hit = D.packets(jnp.ones(1, bool), 0, 1, 50.0, D.PHYSICAL, D.BASIC_ATTACK)
    out = run(state, own, item, ctx, u, dfn, off, u.hp, u.max_hp, now=3.0, base=hit)
    c = out.state.clocks
    assert [float(x) for x in c.last_champion_combat] == [3.0, 3.0]
    assert [bool(x) for x in c.struck_first] == [True, False]
    assert float(c.last_hit_by_champion[1]) == 3.0 and float(c.last_hit_by_champion[0]) < 0
    # A minion hit is combat but not champion combat.
    mhit = D.packets(jnp.ones(1, bool), 2, 0, 10.0, D.PHYSICAL, D.BASIC_ATTACK)
    out = run(out.state, own, item, ctx, u, dfn, off, out.hp, out.max_hp, now=4.0, base=mhit)
    assert float(out.state.clocks.last_combat[0]) == 4.0
    assert float(out.state.clocks.last_champion_combat[0]) == 3.0


def test_combat_tick_compiles():
    u, own, item, ctx, dfn, off = setup(([IRON], ()))
    state = M.init_combat(2, u.x.shape[0])
    f = jax.jit(lambda st, hp: run(st, own, item, ctx, u, dfn, off, hp, u.max_hp, now=1.0))
    out = f(state, u.hp)
    assert int(out.packet_overflow) == 0


def test_every_rune_is_classified():
    from lanerl_jax.modern.runes import effects as RE
    from lanerl_jax.modern.runes.catalog import rune_catalog
    report = RE.coverage_report()
    assert set(report) == set(rune_catalog().ids)
    assert {p for p, v in report.items() if v.startswith("DEFERRED")} == set(RE.DEFERRED)


def test_all_runes_together_through_jitted_combat_tick():
    """Stress: holder 0 owns every rune at once (no legality), holder 1 the
    Garen page; several ticks of mutual basic attacks stay finite and bounded."""
    import numpy as np
    from lanerl_jax.modern.runes import catalog as RDm
    from lanerl_jax.modern.tests import rune_harness as RH
    u, own, item, ctx, dfn, off = setup(([3071, 3053], [3075]))
    n = u.x.shape[0]
    page = RH.perks(list(RDm.rune_catalog().runes), list(RDm.GAREN_DEFAULT_PAGE.perks))
    state = M.init_combat(2, n)

    def step(st, hp, mx, shields, status, now):
        c = ctx._replace(now=now, hp=hp[:2], max_hp=mx[:2], in_combat=jnp.ones((2,), bool))
        uu = u._replace(hp=hp, max_hp=mx)
        base = D.packets(jnp.ones(2, bool), jnp.asarray([0, 1]), jnp.asarray([1, 0]), 60.0, D.PHYSICAL,
                         D.BASIC_ATTACK, cast_id=jnp.asarray([1, 2]) + (now * 30).astype(jnp.int32) * 2)
        att = H.attack(hit=(True, True), target=(1, 0), raw=(60.0, 60.0))
        ev = M.rune_events(c, n, game_time=now + 800.0, attack_started=jnp.ones((2,), bool),
                           attack_start_target=jnp.asarray([1, 0], jnp.int32))
        return M.combat_tick(st, own, page, c, uu, attack=att, cast=H.cast(started=(True, True)),
                             request=jnp.zeros((2,), jnp.int32), base_packets=base, base_offense=off,
                             base_defense=dfn, hp=hp, max_hp=mx, shields=shields, status=status,
                             kills=H.kills(n), holder_stats=item, ev=ev)

    f = jax.jit(step)
    hp, mx, sh, stt = u.hp.at[:2].set(3000.), u.max_hp.at[:2].set(3000.), D.init_shields(n), R.init_status(n)
    for k in range(8):
        out = f(state, hp, mx, sh, stt, jnp.float32(0.6 * k))
        state, hp, mx, sh, stt = out.state, out.hp, out.max_hp, out.shields, out.status
        assert int(out.packet_overflow) == 0
        assert np.all(np.isfinite(np.asarray(hp))) and np.all(np.asarray(hp) <= np.asarray(mx) + 1e-3)
    assert float(hp[1]) < 3000.0 and float(hp[0]) < 3000.0
