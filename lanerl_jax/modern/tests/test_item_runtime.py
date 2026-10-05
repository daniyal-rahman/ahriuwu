"""End-to-end item tick: modules, dispatch, damage pipeline and effect application together."""
import jax
import jax.numpy as jnp
import pytest

from lanerl_jax.modern.combat import item_tick
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items import effects as E
from lanerl_jax.modern.items import inventory as I
from lanerl_jax.modern.items.catalog import catalog
from lanerl_jax.modern.items.effects import runtime as R
from lanerl_jax.modern.tests import item_harness as H

BLUE = [3071, 3053, 3078]          # Black Cleaver, Sterak's Gage, Trinity Force
RED = [3075, 3068]                 # Thornmail, Sunfire Aegis


def world():
    rows = H.champions(x1=150.) + [dict(x=150, y=300, team=1), dict(x=2000, y=0, team=0)]
    return H.units(rows)


def setup():
    u = world()
    n = u.x.shape[0]
    inv = I.inventory_from_ids([BLUE, RED])
    own = I.owned_counts(inv)
    item = I.inventory_stats(inv)
    base_armor, base_mr, base_hp = 30.0, 30.0, 1000.0
    max_hp = base_hp + item.health
    ctx = H.ctx(base_ad=60., bonus_ad=item.attack_damage, base_hp=base_hp, max_hp=max_hp,
                base_armor=base_armor, bonus_armor=item.armor, base_mr=base_mr, bonus_mr=item.magic_resist,
                in_combat=True)
    hp = jnp.concatenate([max_hp, jnp.full((n - 2,), 1000.0)]).astype(jnp.float32)
    u = u._replace(hp=hp, max_hp=hp)
    cls = u.cls
    dfn = D.default_defense(n)._replace(unit_class=cls,
                                         armor=jnp.concatenate([ctx.armor, jnp.zeros((n - 2,))]).astype(jnp.float32),
                                         magic_resist=jnp.concatenate([ctx.magic_resist, jnp.zeros((n - 2,))]).astype(jnp.float32))
    off = D.default_offense(n)._replace(unit_class=cls)
    return u, own, item, ctx, dfn, off, hp


def tick(state, own, item, ctx, u, dfn, off, hp, shields, status, *, attack, cast, base, now):
    ctx = ctx._replace(now=jnp.float32(now), hp=hp[:2])
    u = u._replace(hp=hp)
    return item_tick(state, own, ctx, u, attack=attack, cast=cast, request=jnp.zeros((2,), jnp.int32),
                       base_packets=base, base_offense=off, base_defense=dfn, hp=hp, max_hp=u.max_hp,
                       shields=shields, status=status, kills=H.kills(u.x.shape[0]), holder_stats=item)


def test_duel_tick_spellblade_carve_thorns_sunfire_lifeline():
    u, own, item, ctx, dfn, off, hp = setup()
    n = u.x.shape[0]
    state, shields, status = E.init(2, n), D.init_shields(n), R.init_status(n)
    # Tick 0: blue casts Q (arms Spellblade) - no damage yet.
    no_attack = H.attack(hit=(False, False))
    out = tick(state, own, item, ctx, u, dfn, off, hp, shields, status, attack=no_attack,
               cast=H.cast(), base=D.empty_packets(0), now=0.0)
    # Tick 1: blue's basic attack (raw 100 physical) lands on red.
    t1 = 0.5
    base = D.packets(jnp.ones(1, bool), 0, 1, 100.0, D.PHYSICAL, D.BASIC_ATTACK)
    out = tick(out.state, own, item, ctx, u, dfn, off, out.hp, out.shields, out.status,
               attack=H.attack(), cast=H.cast(started=(False, False)), base=base, now=t1)
    red_armor = float(ctx.armor[1])
    expect_attack = 100 * 100 / (100 + red_armor)
    expect_trinity = 2.0 * 60 * 100 / (100 + red_armor)
    red_loss = float(item.health[1] + 1000 - out.hp[1])
    assert red_loss == pytest.approx(expect_attack + expect_trinity, rel=1e-4)
    # Thornmail reflects 20 + 10% bonus armor magic onto blue and applies Grievous Wounds.
    thorns = (20 + 0.1 * float(ctx.bonus_armor[1])) * 100 / (100 + float(ctx.magic_resist[0]))
    blue_loss = float(item.health[0] + 1000 - out.hp[0])
    assert blue_loss == pytest.approx(thorns, rel=1e-4)
    assert float(out.status.grievous_until[0]) == pytest.approx(t1 + 3.0)
    # Carve: the basic attack and the separate Spellblade proc instance each add a
    # stack (wiki: only non-basic damage is limited to one stack per frame).
    deb = E.target_debuffs(out.state, own, ctx._replace(now=jnp.float32(t1 + 0.1)), u)
    assert float(deb.percent_armor_reduction[1]) == pytest.approx(0.12, rel=1e-4)
    # Tick 2 at +1 s: Sunfire (activated by red taking damage) burns blue within 325.
    t2 = t1 + 1.0
    before = float(out.hp[0])
    out = tick(out.state, own, item, ctx, u, dfn, off, out.hp, out.shields, out.status,
               attack=no_attack, cast=H.cast(started=(False, False)), base=D.empty_packets(0), now=t2)
    burn = (20 + 0.015 * float(item.health[1])) * 100 / (100 + float(ctx.magic_resist[0]))
    assert before - float(out.hp[0]) == pytest.approx(burn, rel=1e-3)


def test_sterak_lifeline_through_runtime_and_jit():
    u, own, item, ctx, dfn, off, hp = setup()
    n = u.x.shape[0]
    low = hp.at[0].set(0.35 * hp[0])
    hit = D.packets(jnp.ones(1, bool), 1, 0, 400.0, D.TRUE)
    run = jax.jit(lambda st, h: tick(st, own, item, ctx, u, dfn, off, h, D.init_shields(n), R.init_status(n),
                                     attack=H.attack(hit=(False, False)), cast=H.cast(started=(False, False)),
                                     base=hit, now=5.0))
    out = run(E.init(2, n), low)
    assert bool(out.report.resolved.lifeline_fired[0])
    shield = 0.60 * float(item.health[0])                       # 60% bonus HP (items only here)
    assert float(out.report.resolved.absorbed[0]) == pytest.approx(min(400.0, shield), rel=1e-4)
    assert float(out.state.defense.lifeline_cd[0]) == pytest.approx(95.0)
    # Second hit right after: Lifeline is on cooldown.
    out2 = run(out.state, out.hp)
    assert not bool(out2.report.resolved.lifeline_fired[0])


def test_every_catalog_item_is_classified():
    report = E.coverage_report()
    assert set(report) == set(catalog().ids)
    deferred = {i for i, v in report.items() if v.startswith("DEFERRED")}
    assert deferred == set(E.DEFERRED)
