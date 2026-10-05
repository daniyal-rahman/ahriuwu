"""Patch-26.19 summoner spells (docs/modern/SUMMONER_SPELLS.md §16 fixtures S-F1..S-F21)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.champions import summoners as S
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core.types import KIND_CHAMPION, KIND_MINION, KIND_TURRET, CastOrder, WorldUnits
from lanerl_jax.modern.tests import item_harness as H

FL, TP, IG, EX, BA, HE, GH, CL, SM = (S.SUMMONERS[k] for k in
                                      ("flash", "teleport", "ignite", "exhaust", "barrier", "heal", "ghost",
                                       "cleanse", "smite"))


def world(extra=(), *, x1=300.0, team1=1, alive=(True, True)):
    """Units 0/1 = champions (holders 0/1); extras are (kind, team, x, y)."""
    rows = [(KIND_CHAMPION, 0, 0.0, 0.0), (KIND_CHAMPION, team1, x1, 0.0)] + list(extra)
    n = len(rows)
    f = lambda i: jnp.asarray([r[i] for r in rows], jnp.float32)
    z = jnp.zeros((n,), jnp.float32)
    al = jnp.asarray(list(alive) + [True] * (n - 2))
    return WorldUnits(kind=jnp.asarray([r[0] for r in rows], jnp.int32), sub=jnp.zeros((n,), jnp.int32),
                      team=jnp.asarray([r[1] for r in rows], jnp.int32), alive=al, targetable=jnp.ones((n,), bool),
                      x=f(2), y=f(3), radius=z + 65.0, hp=z + 1000.0, max_hp=z + 1000.0, armor=z, magic_resist=z,
                      attack_damage=z, attack_range=z, attack_speed=z, move_speed=z + 340.0,
                      spawn_seq=jnp.zeros((n,), jnp.int32), spawn_time=z)


def ctx(now, *, x1=300.0, team=(0, 1), **kw):
    return H.ctx(2, now=now, x=jnp.asarray([0.0, x1]), team=team, **kw)


def run(state, now, units=None, c=None, *, slot=(-1, -1), target=(-1, -1), x=(0.0, 0.0), y=(0.0, 0.0), dt=1 / 30,
        haste=0.0, can_cast=True, interrupted=False, quest=False, damaged=False, **kw):
    units = world() if units is None else units
    c = ctx(now) if c is None else c
    req = CastOrder(jnp.asarray(slot, jnp.int32), jnp.asarray(target, jnp.int32), jnp.asarray(x, jnp.float32),
                    jnp.asarray(y, jnp.float32))
    b = lambda v: jnp.broadcast_to(jnp.asarray(v), (2,))
    return S.step(state, c, units, request=req, now=now, dt=dt, summoner_haste=b(haste), can_cast=b(can_cast),
                  channel_interrupted=b(interrupted), quest_complete=b(quest), took_champion_damage=b(damaged), **kw)


def ready(loadout=((FL, IG), (FL, EX)), t=20.0):
    """State with every slot ready at time ``t``."""
    st = S.init(np.asarray(loadout))
    return st._replace(ready_at=st.ready_at.at[:, :2].set(0.0))


# ---- shared rules ------------------------------------------------------------

def test_sf1_start_of_game_cooldown():
    st = S.init(np.asarray([[FL, IG], [FL, EX]]))
    st, _, out = run(st, 0.0)
    np.testing.assert_allclose(out.cooldowns, 15.0)
    _, _, out = run(st, 14.9, slot=(0, -1), x=(1000.0, 0.0))
    assert not bool(out.dash.active[0])
    _, _, out = run(st, 15.0, slot=(0, -1), x=(1000.0, 0.0))
    assert bool(out.dash.active[0])


def test_start_cooldown_not_hasted_u_s2():
    st = S.init(np.asarray([[FL, IG], [FL, EX]]))
    _, _, out = run(st, 0.0, haste=18.0)
    np.testing.assert_allclose(out.cooldowns, 15.0)


def test_loadout_validation_and_smite_deferred():
    with pytest.raises(ValueError):
        S.validate_loadout([[FL, FL]])
    with pytest.raises(ValueError):
        S.validate_loadout([[FL, 13]])          # Clarity: ARAM only
    flags = S.validate_loadout([[SM, FL], [FL, TP]])
    assert flags.tolist() == [[True, False], [False, False]]
    st = ready(((SM, FL), (FL, TP)))
    st2, eff, out = run(st, 20.0, slot=(0, -1), target=(1, -1))
    assert not bool(out.cast_event[0]) and float(out.cooldowns[0, 0]) == 0.0
    assert not bool(jnp.any(eff.packets.valid))


def test_haste_rescales_remaining_cooldown_u_s1():
    st, _, out = run(ready(), 20.0, slot=(0, -1), x=(1000.0, 0.0))
    assert float(out.cooldowns[0, 0]) == pytest.approx(300.0)
    _, _, out = run(st, 120.0, haste=10.0)          # 200 s left -> 200 * 100/110
    assert float(out.cooldowns[0, 0]) == pytest.approx(200.0 / 1.1, rel=1e-5)


# ---- Flash -------------------------------------------------------------------

def test_sf2_sf3_flash_distance_and_cooldown():
    _, _, out = run(ready(), 20.0, slot=(0, -1), x=(1000.0, 0.0))
    d = out.dash
    assert bool(d.active[0]) and bool(d.blink[0]) and np.isinf(float(d.speed[0]))
    assert (float(d.to_x[0]), float(d.to_y[0])) == pytest.approx((400.0, 0.0))
    assert float(out.cooldowns[0, 0]) == pytest.approx(300.0)
    assert bool(out.blinked[0]) and bool(out.cast_event[0]) and int(out.cast_spell[0]) == FL
    _, _, out = run(ready(), 20.0, slot=(0, -1), x=(200.0, 0.0))
    assert (float(out.dash.to_x[0]), float(out.dash.to_y[0])) == pytest.approx((200.0, 0.0))


def test_sf4_flash_summoner_haste():
    _, _, out = run(ready(), 20.0, slot=(0, -1), x=(1000.0, 0.0), haste=28.0)
    assert float(out.cooldowns[0, 0]) == pytest.approx(234.375)
    assert float(out.cast_cooldown[0]) == pytest.approx(234.375)


def test_sf5_flash_rejected_rooted_or_stunned():
    for kw in (dict(rooted=jnp.asarray([True, False])), dict(can_cast=False)):
        _, _, out = run(ready(), 20.0, slot=(0, -1), x=(1000.0, 0.0), **kw)
        assert not bool(out.dash.active[0]) and float(out.cooldowns[0, 0]) == 0.0


def test_casting_rules_disabled_vs_suppressed():
    # Ignite is canCastWhileDisabled (stun/root ok) but not under suppression.
    _, _, out = run(ready(), 20.0, slot=(1, -1), target=(1, -1), can_cast=False, rooted=jnp.asarray([True, False]))
    assert int(out.ignite_target[0]) == 1
    _, _, out = run(ready(), 20.0, slot=(1, -1), target=(1, -1), suppressed=jnp.asarray([True, False]))
    assert int(out.ignite_target[0]) == -1
    # Dead champions cast nothing.
    _, _, out = run(ready(), 20.0, None, ctx(20.0, alive=jnp.asarray([False, True])), slot=(0, -1), x=(500.0, 0.0))
    assert not bool(out.dash.active[0])


def test_flash_cooldown_helper_for_hexflash():
    st, _, _ = run(ready(), 20.0, slot=(-1, 0), x=(0.0, 1000.0))
    np.testing.assert_allclose(S.flash_cooldown(st), [0.0, 300.0])
    np.testing.assert_allclose(S.flash_cooldown(st, 120.0), [0.0, 200.0])
    st2 = S.init(np.asarray([[IG, EX], [HE, GH]]))
    np.testing.assert_allclose(S.flash_cooldown(st2, 0.0), 0.0)


# ---- Ignite ------------------------------------------------------------------

def _ignite_trace(level, t_end=6.0, dt=1 / 30):
    st, eff, out = run(ready(), 20.0, None, ctx(20.0, level=level), slot=(1, -1), target=(1, -1), dt=dt)
    times, dmg = [], []
    first = (eff, 20.0)
    for i in range(1, int(t_end / dt) + 1):
        now = 20.0 + i * dt
        st, e, _ = run(st, now, None, ctx(now, level=level), dt=dt)
        v = np.asarray(e.packets.valid)
        if v.any():
            times.append(now - 20.0)
            dmg.append(float(np.asarray(e.packets.raw)[v].sum()))
            p = e.packets
            assert int(p.dst[v][0]) == 1 and int(p.src[v][0]) == 0 and int(p.dtype[v][0]) == D.TRUE
            assert int(p.flags[v][0]) == S.IGNITE_FLAGS and int(p.item[v][0]) == 0
    return first, times, dmg, out


@pytest.mark.parametrize("level,total", [(1, 70.0), (5, 150.0), (6, 175.0), (9, 250.0), (18, 475.0)])
def test_sf6_ignite_damage_ticks_gw(level, total):
    (eff0, _), times, dmg, out = _ignite_trace(level)
    assert len(dmg) == 5
    np.testing.assert_allclose(dmg, total / 5, rtol=1e-5)
    assert times[0] == pytest.approx(0.25, abs=1 / 30 + 1e-6)
    np.testing.assert_allclose(np.diff(times), 1.056, atol=1 / 30 + 1e-6)
    assert float(eff0.grievous[1]) == pytest.approx(5.0) and float(eff0.grievous[0]) == 0.0
    assert float(out.cooldowns[0, 1]) == pytest.approx(180.0)
    assert int(out.ignite_target[0]) == 1


def test_ignite_first_tick_same_tick_with_large_dt():
    _, eff, _ = run(ready(), 20.0, slot=(1, -1), target=(1, -1), dt=0.25)
    assert int(jnp.sum(eff.packets.valid)) == 1


def test_ignite_range_and_target_rules():
    _, _, out = run(ready(), 20.0, world(x1=600.0), ctx(20.0, x1=600.0), slot=(1, -1), target=(1, -1))
    assert int(out.ignite_target[0]) == 1
    _, _, out = run(ready(), 20.0, world(x1=601.0), ctx(20.0, x1=601.0), slot=(1, -1), target=(1, -1))
    assert int(out.ignite_target[0]) == -1 and float(out.cooldowns[0, 1]) == 0.0
    # An allied champion / a minion is not a valid target.
    _, _, out = run(ready(), 20.0, world(team1=0), ctx(20.0, team=(0, 0)), slot=(1, -1), target=(1, -1))
    assert int(out.ignite_target[0]) == -1
    _, _, out = run(ready(), 20.0, world([(KIND_MINION, 1, 100.0, 0.0)]), slot=(1, -1), target=(2, -1))
    assert int(out.ignite_target[0]) == -1


def test_sf7_grievous_wounds_heal():
    _, eff, _ = run(ready(), 20.0, slot=(1, -1), target=(1, -1))
    gw = float(eff.grievous[1]) > 0.0
    assert float(D.heal_amount(100.0, grievous=gw)) == pytest.approx(60.0)


# ---- Exhaust -----------------------------------------------------------------

def test_sf8_exhaust_reduction_and_slow():
    st, _, out = run(ready(), 20.0, slot=(-1, 1), target=(-1, 0))
    assert float(out.exhaust_slow[0]) == pytest.approx(0.4) and float(out.exhaust_slow_duration[0]) == 3.0
    assert float(out.exhaust_reduction[0]) == pytest.approx(0.35) and float(out.exhaust_reduction[1]) == 0.0
    assert float(out.cooldowns[1, 1]) == pytest.approx(240.0)
    off = D.default_offense(2, unit_class=D.CLASS_CHAMPION)._replace(dealt_reduction=out.exhaust_reduction)
    dfn = D.default_defense(2, unit_class=D.CLASS_CHAMPION)
    p = D.packets(True, jnp.asarray([0, 0]), jnp.asarray([1, 1]), 200.0, jnp.asarray([D.PHYSICAL, D.TRUE]))
    np.testing.assert_allclose(D.premitigation_to_final(p, off, dfn), [130.0, 200.0], rtol=1e-6)
    # Lasts 3 s; slow only emitted on the cast tick.
    st, _, out = run(st, 22.9)
    assert float(out.exhaust_reduction[0]) == pytest.approx(0.35) and float(out.exhaust_slow[0]) == 0.0
    _, _, out = run(st, 23.01)
    assert float(out.exhaust_reduction[0]) == 0.0


def test_exhaust_range():
    _, _, out = run(ready(), 20.0, world(x1=651.0), ctx(20.0, x1=651.0), slot=(-1, 1), target=(-1, 0))
    assert float(out.exhaust_reduction[0]) == 0.0 and float(out.cooldowns[1, 1]) == 0.0


# ---- Barrier / Heal / Ghost --------------------------------------------------

@pytest.mark.parametrize("level,amount", [(1, 100.0), (6, 205.88), (9, 269.41), (13, 354.12), (18, 460.0)])
def test_sf9_barrier(level, amount):
    _, eff, out = run(ready(((BA, FL), (FL, EX))), 20.0, None, ctx(20.0, level=level), slot=(0, -1))
    sh = eff.shields
    assert float(sh.amount[0].sum()) == pytest.approx(amount, abs=0.01)
    assert float(sh.duration[0, int(jnp.argmax(sh.amount[0]))]) == 2.5
    assert float(sh.amount[1].sum()) == 0.0
    assert float(out.cooldowns[0, 0]) == pytest.approx(180.0)


def test_sf10_heal_amount_ms_and_repeat_debuff():
    # Two allied champions 300 apart; holder 0 heals both, holder 1 heals again 10 s later.
    load = ((HE, FL), (HE, FL))
    team = (0, 0)
    w = world(team1=0)
    st, eff, out = run(ready(load), 20.0, w, ctx(20.0, level=9, team=team, hp=jnp.asarray([300.0, 500.0])),
                       slot=(0, -1), x=(0.0, 0.0))
    np.testing.assert_allclose(eff.heal, [192.0, 192.0], rtol=1e-5)
    np.testing.assert_allclose(out.bonus_ms_pct, [0.3, 0.3])
    assert float(out.cooldowns[0, 0]) == pytest.approx(240.0)
    _, _, out2 = run(st, 21.01, w, ctx(21.01, level=9, team=team))
    np.testing.assert_allclose(out2.bonus_ms_pct, 0.0)
    _, eff, _ = run(st, 30.0, w, ctx(30.0, level=9, team=team), slot=(-1, 0))
    np.testing.assert_allclose(eff.heal, [96.0, 96.0], rtol=1e-5)
    # Outside the 30 s window: full heal.
    _, eff, _ = run(st, 50.1, w, ctx(50.1, level=9, team=team), slot=(-1, 0))
    np.testing.assert_allclose(eff.heal, [192.0, 192.0], rtol=1e-5)


@pytest.mark.parametrize("level,amount", [(1, 80.0), (6, 150.0), (13, 248.0), (18, 318.0)])
def test_heal_levels_solo(level, amount):
    _, eff, _ = run(ready(((HE, FL), (FL, EX))), 20.0, None, ctx(20.0, level=level), slot=(0, -1))
    np.testing.assert_allclose(eff.heal, [amount, 0.0], rtol=1e-4)   # the enemy is not an ally


def test_heal_ally_choice_and_caster_heal_power():
    # Three blue champions: cursor near holder 2 picks it over the more wounded holder 1.
    u = world([(KIND_CHAMPION, 0, 0.0, 600.0)], team1=0)
    c3 = H.ctx(3, now=20.0, level=1, x=jnp.asarray([0.0, 300.0, 0.0]), y=jnp.asarray([0.0, 0.0, 600.0]),
               team=(0, 0, 0), hp=jnp.asarray([600.0, 100.0, 500.0]), hsp=jnp.asarray([0.2, 0.0, 0.5]))
    st = S.init(np.asarray([[HE, FL], [FL, EX], [FL, IG]]))
    st = st._replace(ready_at=st.ready_at.at[:, :2].set(0.0))
    req = lambda x, y: CastOrder(jnp.asarray([0, -1, -1], jnp.int32), jnp.full((3,), -1, jnp.int32),
                                 jnp.asarray([x, 0.0, 0.0], jnp.float32), jnp.asarray([y, 0.0, 0.0], jnp.float32))
    kw = dict(now=20.0, dt=1 / 30, summoner_haste=jnp.zeros(3), can_cast=jnp.ones(3, bool),
              channel_interrupted=jnp.zeros(3, bool), quest_complete=jnp.zeros(3, bool),
              took_champion_damage=jnp.zeros(3, bool))
    _, eff, _ = S.step(st, c3, u, request=req(0.0, 650.0), **kw)
    # Integrator multiplies by recipient HSP: 80*(1+0.2) for both after that.
    np.testing.assert_allclose(np.asarray(eff.heal) * (1.0 + np.asarray(c3.heal_shield_power)), [96.0, 0.0, 96.0],
                               rtol=1e-5)
    _, eff, _ = S.step(st, c3, u, request=req(5000.0, 5000.0), **kw)       # no cursor ally: lowest %HP
    np.testing.assert_allclose(np.asarray(eff.heal) * (1.0 + np.asarray(c3.heal_shield_power)), [96.0, 96.0, 0.0],
                               rtol=1e-5)


@pytest.mark.parametrize("level,pct", [(1, 0.24), (6, 0.3106), (9, 0.3529), (18, 0.48)])
def test_sf11_ghost(level, pct):
    st, _, out = run(ready(((GH, FL), (FL, EX))), 20.0, None, ctx(20.0, level=level), slot=(0, -1))
    assert float(out.bonus_ms_pct[0]) == pytest.approx(pct, abs=1e-4) and bool(out.ghosted[0])
    assert not bool(out.ghosted[1])
    _, _, out = run(st, 29.9, None, ctx(29.9, level=level))
    assert bool(out.ghosted[0])
    _, _, out = run(st, 30.01, None, ctx(30.01, level=level))
    assert not bool(out.ghosted[0]) and float(out.bonus_ms_pct[0]) == 0.0
    # Not interrupted by CC: castable while stunned.
    _, _, out = run(ready(((GH, FL), (FL, EX))), 20.0, slot=(0, -1), can_cast=False)
    assert bool(out.ghosted[0])


# ---- Cleanse -----------------------------------------------------------------

def test_sf12_cleanse_removes_ignite_and_exhaust_keeps_gw():
    st = ready(((CL, FL), (IG, EX)))
    st, eff_ig, _ = run(st, 20.0, slot=(-1, 0), target=(-1, 0))
    assert float(eff_ig.grievous[0]) == pytest.approx(5.0)
    st, _, out = run(st, 20.1, slot=(-1, 1), target=(-1, 0))
    assert float(out.exhaust_reduction[0]) == pytest.approx(0.35)
    st, eff, out = run(st, 20.5, slot=(0, -1), can_cast=False)    # stunned
    assert bool(out.cleanse[0]) and not bool(out.cleanse[1])
    assert float(out.exhaust_reduction[0]) == 0.0
    assert float(out.tenacity[0]) == pytest.approx(0.75) and float(out.tenacity[1]) == 0.0
    assert float(eff.grievous[0]) == 0.0           # GW is the world's status: nothing removes it here
    total = float(jnp.sum(jnp.where(eff.packets.valid, eff.packets.raw, 0.0)))
    for i in range(1, 200):
        now = 20.5 + i / 30
        st, eff, out = run(st, now)
        total += float(jnp.sum(jnp.where(eff.packets.valid, eff.packets.raw, 0.0)))
        if now < 23.49:
            assert float(out.tenacity[0]) == pytest.approx(0.75)
    assert total == 0.0
    assert float(out.tenacity[0]) == 0.0
    # Not castable under suppression.
    _, _, out = run(ready(((CL, FL), (IG, EX))), 20.0, slot=(0, -1), suppressed=jnp.asarray([True, False]))
    assert not bool(out.cleanse[0])


# ---- Teleport ----------------------------------------------------------------

def tp_world(dist):
    return world([(KIND_MINION, 0, dist, 0.0), (KIND_TURRET, 0, -dist, 0.0), (KIND_MINION, 1, 100.0, 0.0)])


def tp_run(st, t0, dist, *, level=1, quest=False, slot=0, dt=0.25, until=12.0, haste=0.0):
    """Cast TP at t0 on the allied minion ``dist`` away; step until t0+until. Returns per-tick outputs."""
    u = tp_world(dist)
    sl = (slot, -1)
    st, eff, out = run(st, t0, u, ctx(t0, level=level), slot=sl, target=(2, -1), dt=dt, quest=quest, haste=haste)
    trace = [(t0, eff, out)]
    for i in range(1, int(until / dt) + 1):
        now = t0 + i * dt
        st, eff, out = run(st, now, u, ctx(now, level=level), dt=dt, quest=quest, haste=haste)
        trace.append((now, eff, out))
    return st, trace


def _arrival(trace):
    hits = [(t, e, o) for t, e, o in trace if bool(o.teleport_arrive[0])]
    assert len(hits) == 1
    return hits[0]


def test_sf13_teleport_channel_dash_cooldown():
    st, trace = tp_run(ready(((TP, FL), (FL, EX))), 300.0, 5000.0)
    assert bool(trace[0][2].teleport_start[0]) and bool(trace[0][2].teleport_channel[0])
    chan_end = [t for t, _, o in trace if bool(o.cast_event[0])]
    assert chan_end == [pytest.approx(303.0)]
    ev = [o for _, _, o in trace if bool(o.cast_event[0])][0]
    assert bool(ev.is_teleport[0]) and float(ev.cast_cooldown[0]) == pytest.approx(300.0)
    for t, _, o in trace:
        if t < 303.0:
            assert bool(o.teleport_channel[0])
        elif t < 308.0:
            assert bool(o.teleport_dash[0]) and not bool(o.teleport_channel[0])
    t, _, o = _arrival(trace)
    assert t == pytest.approx(308.0)
    assert (float(o.teleport_x[0]), float(o.teleport_y[0])) == (5000.0, 0.0)
    assert bool(o.blinked[0]) and float(o.cooldowns[0, 0]) == pytest.approx(300.0)
    assert float(o.bonus_ms_pct[0]) == 0.0 and float(o.arrival_shield[0]) == 0.0


def test_sf14_teleport_dash_time():
    st, trace = tp_run(ready(((TP, FL), (FL, EX))), 300.0, 2500.0)
    t, _, _ = _arrival(trace)
    assert t == pytest.approx(303.0 + 2.75)


def test_teleport_targets_and_forgiveness():
    u = tp_world(3000.0)
    base = ready(((TP, FL), (FL, EX)))
    _, _, out = run(base, 300.0, u, slot=(0, -1), target=(4, -1))          # enemy minion
    assert not bool(out.teleport_start[0])
    _, _, out = run(base, 300.0, u, slot=(0, -1), target=(3, -1))          # allied turret
    assert bool(out.teleport_start[0]) and int(out.teleport_target[0]) == 3
    _, _, out = run(base, 300.0, u, slot=(0, -1), target=(-1, -1), x=(2700.0, 0.0))   # snap within 400
    assert bool(out.teleport_start[0]) and int(out.teleport_target[0]) == 2
    _, _, out = run(base, 300.0, u, slot=(0, -1), target=(-1, -1), x=(2500.0, 0.0))
    assert not bool(out.teleport_start[0])
    _, _, out = run(base, 300.0, u, slot=(0, -1), target=(2, -1), rooted=jnp.asarray([True, False]))
    assert not bool(out.teleport_start[0])


def test_sf15_unleashed_teleport():
    st, _, out = run(ready(((TP, FL), (FL, EX))), 700.0)        # 10:00 passed: upgrade, 2 s floor
    assert float(out.cooldowns[0, 0]) == pytest.approx(2.0)
    st, trace = tp_run(st, 702.0, 9000.0, level=6)
    t, _, o = _arrival(trace)
    assert t == pytest.approx(705.0 + 2.25)
    assert float(o.cooldowns[0, 0]) == pytest.approx(280.0)
    assert float(o.bonus_ms_pct[0]) == pytest.approx(0.5)
    later = [o for tt, _, o in trace if t + 3.7 < tt < t + 3.8]
    assert later and float(later[0].bonus_ms_pct[0]) == pytest.approx(0.5)
    after = [o for tt, _, o in trace if tt > t + 4.1]
    assert after and float(after[0].bonus_ms_pct[0]) == 0.0


def test_sf16_unleashed_with_top_quest_shield():
    st, _, _ = run(ready(((TP, FL), (FL, EX))), 700.0, quest=True)
    st, trace = tp_run(st, 702.0, 9000.0, level=12, quest=True)
    _, eff, o = _arrival(trace)
    assert float(o.cooldowns[0, 0]) == pytest.approx(210.0)
    assert float(o.arrival_shield[0]) == pytest.approx(0.35 * 600.0)
    sh = eff.shields
    k = int(jnp.argmax(sh.amount[0]))
    assert float(sh.amount[0, k]) == pytest.approx(210.0) and float(sh.duration[0, k]) == 10.0
    assert float(sh.amount[1].sum()) == 0.0


def test_sf16_no_shield_when_interrupted():
    st, trace = tp_run(ready(((TP, FL), (FL, EX))), 300.0, 3000.0, quest=True, until=1.0)
    st, eff, out = run(st, 301.5, tp_world(3000.0), interrupted=True, quest=True)
    assert not bool(out.teleport_channel[0]) and float(out.arrival_shield[0]) == 0.0


def test_sf17_quest_free_teleport():
    st = ready()
    st, _, out = run(st, 400.0, quest=True)
    assert bool(st.quest_tp[0]) and float(out.quest_cooldown[0]) == 0.0
    assert float(S.init(np.asarray([[FL, IG], [FL, EX]])).ready_at[0, 2]) >= S.BIG    # no slot before grant
    st, trace = tp_run(st, 400.0, 9000.0, quest=True, slot=S.QUEST_SLOT)
    t, eff, o = _arrival(trace)
    assert t == pytest.approx(403.0 + 2.25)               # Unleashed before 10:00
    assert float(o.quest_cooldown[0]) == pytest.approx(390.0)
    assert float(o.bonus_ms_pct[0]) == pytest.approx(0.5)
    assert float(o.arrival_shield[0]) == 0.0              # shield is for an own TP only
    np.testing.assert_allclose(o.cooldowns[0], [0.0, 0.0])  # D/F untouched
    # Champions with Teleport equipped get no bonus slot.
    st2, _, out = run(ready(((TP, FL), (FL, EX))), 400.0, quest=True)
    assert not bool(st2.quest_tp[0]) and np.isinf(float(out.quest_cooldown[0]))


def test_sf18_upgrade_keeps_remaining_cooldown():
    st = ready(((TP, FL), (FL, EX)))
    for rem, expect in ((180.0, 180.0), (0.0, 2.0), (500.0, 330.0)):
        s = st._replace(ready_at=st.ready_at.at[0, 0].set(599.9 + rem))
        s, _, _ = run(s, 599.9, None, ctx(599.9, level=7))
        assert not bool(s.upgraded[0])
        _, _, out = run(s, 600.0, None, ctx(600.0, level=7))
        assert float(out.cooldowns[0, 0]) == pytest.approx(expect, abs=0.11)
    # Hasted level-1 cap with the quest: 300 * 100/110.
    s = st._replace(ready_at=st.ready_at.at[0, 0].set(1100.0))
    _, _, out = run(s, 600.0, haste=10.0, quest=True)
    assert float(out.cooldowns[0, 0]) == pytest.approx(300.0 / 1.1, rel=1e-5)


def test_unleashed_cooldown_by_level():
    for lv, cd in ((1, 330), (2, 320), (5, 290), (6, 280), (9, 250), (10, 240), (20, 240)):
        st, _, _ = run(ready(((TP, FL), (FL, EX))), 700.0)
        _, trace = tp_run(st, 702.0, 9000.0, level=lv)   # dash 2.25 s: arrival on a tick
        assert float(_arrival(trace)[2].cooldowns[0, 0]) == pytest.approx(cd)


def test_sf19_channel_interrupted_by_stun():
    st, _ = tp_run(ready(((TP, FL), (FL, EX))), 300.0, 5000.0, until=1.25)
    st, _, out = run(st, 301.5, tp_world(5000.0), interrupted=True)
    assert not bool(out.teleport_channel[0]) and not bool(out.teleport_dash[0])
    assert float(out.cooldowns[0, 0]) == pytest.approx(300.0)        # U-S6: full from interrupt
    assert not bool(out.cast_event[0])
    for i in range(1, 40):
        st, _, out = run(st, 301.5 + 0.25 * i, tp_world(5000.0))
        assert not bool(out.teleport_arrive[0]) and not bool(out.teleport_dash[0])


def test_sf20_damage_does_not_interrupt():
    st, _ = tp_run(ready(((TP, FL), (FL, EX))), 300.0, 5000.0, until=1.0)
    st, _, out = run(st, 301.5, tp_world(5000.0), damaged=True)
    assert bool(out.teleport_channel[0])


def test_channel_blocks_other_summoners():
    st, _ = tp_run(ready(((TP, FL), (FL, EX))), 300.0, 5000.0, until=1.0)
    _, _, out = run(st, 301.5, tp_world(5000.0), slot=(1, -1), x=(500.0, 0.0))
    assert not bool(out.dash.active[0]) and float(out.cooldowns[0, 1]) == 0.0


# ---- rune events, isolation, jit ---------------------------------------------

def test_sf21_rune_event_cooldowns():
    bracket = lambda cd: 0.15 if cd < 100 else (0.35 if cd <= 250 else 0.45)
    got = {}
    for load, slot, target in (((FL, IG), 0, -1), ((FL, IG), 1, 1), ((EX, FL), 0, 1)):
        _, _, out = run(ready((load, (FL, EX))), 20.0, slot=(slot, -1), target=(target, -1), x=(500.0, 0.0))
        assert bool(out.cast_event[0]) and not bool(out.is_teleport[0])
        got[int(out.cast_spell[0])] = bracket(float(out.cast_cooldown[0]))
    assert got == {FL: 0.45, IG: 0.35, EX: 0.35}
    _, trace = tp_run(ready(((TP, FL), (FL, EX))), 300.0, 2000.0, until=4.0)
    ev = [o for _, _, o in trace if bool(o.cast_event[0])]
    assert len(ev) == 1 and bool(ev[0].is_teleport[0])


def test_holder_isolation():
    st, eff, out = run(ready(), 20.0, slot=(-1, 0), x=(0.0, 300.0), y=(0.0, 1000.0))
    np.testing.assert_allclose(out.cooldowns[0], [0.0, 0.0])
    assert not bool(out.dash.active[0]) and bool(out.dash.active[1])
    assert float(out.dash.to_x[1]) == pytest.approx(300.0, abs=1e-4) and float(out.dash.to_y[1]) == pytest.approx(400.0)
    assert not bool(out.cast_event[0])


def test_step_under_jit():
    st = ready(((FL, IG), (TP, EX)))
    u = tp_world(3000.0)
    c = ctx(20.0, level=6)
    req = CastOrder(jnp.asarray([1, 0], jnp.int32), jnp.asarray([1, 4], jnp.int32), jnp.zeros(2, jnp.float32),
                    jnp.zeros(2, jnp.float32))
    b = jnp.zeros(2, bool)

    @jax.jit
    def f(st, c, u, req, now):
        return S.step(st, c, u, request=req, now=now, dt=jnp.float32(1 / 30), summoner_haste=jnp.zeros(2),
                      can_cast=~b, channel_interrupted=b, quest_complete=b, took_champion_damage=b)
    st2, eff, out = f(st, c, u, req, jnp.float32(20.0))
    assert int(out.ignite_target[0]) == 1 and bool(out.teleport_start[1])
    st3, _, _ = f(st2, ctx(20.5, level=6), u, req._replace(slot=jnp.asarray([-1, -1], jnp.int32)), jnp.float32(20.5))
    assert jax.tree_util.tree_structure(st3) == jax.tree_util.tree_structure(st)
    for a, b_ in zip(st3, st):
        assert a.shape == b_.shape and a.dtype == b_.dtype
