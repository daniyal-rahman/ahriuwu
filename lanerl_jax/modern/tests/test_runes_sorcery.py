"""Sorcery tree 8200 (runes.effects.sorcery; RUNES.md §5, §13 F-29)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.core import stat_pipeline as SP
from lanerl_jax.modern.items.effects.core import CC
from lanerl_jax.modern.runes.effects import sorcery as S
from lanerl_jax.modern.tests import item_harness as H
from lanerl_jax.modern.tests import rune_harness as R
from lanerl_jax.modern.tests.rune_harness import world

SPELL = D.TAG_ACTIVE_SPELL


def hit(state, page, ctx, u, raw, flags=D.BASIC_ATTACK, src=0, dst=1, **evkw):
    p = R.hit_packet(src, dst, raw, flags=flags)
    rep = R.report(p, u)
    return S.on_damage(state, page, ctx, u, R.ev(ctx, u.x.shape[0], report=rep, **evkw))


def run_periodic(state, page, u, t0, t1, dt=1 / 30, **ctxkw):
    """Step periodic from t0 to t1; returns (state, list of (t, packets))."""
    out = []
    steps = int(round((t1 - t0) / dt))
    for i in range(steps + 1):
        t = t0 + i * dt
        c = H.ctx(now=t, **ctxkw)
        state, eff = S.periodic(state, page, c, u, R.ev(c, u.x.shape[0]))
        out.append((t, eff.packets))
    return state, out


def total(out, rune, dst=None):
    return sum(R.total(p, rune=rune, dst=dst) for _, p in out)


def test_coverage_lists_every_sorcery_rune():
    ids = {8214, 8229, 8230, 8992, 8224, 8226, 8275, 8210, 8234, 8233, 8237, 8232, 8236}
    assert set(S.COVERAGE) == ids
    assert all(isinstance(v, str) and v for v in S.COVERAGE.values())
    assert "Phase Rush" not in S.COVERAGE[8230].replace("not Phase Rush", "")


# ---- Stormraider's Surge (F-29) ---------------------------------------------

def test_f29_stormraider_trigger_l9():
    u = world()
    page = R.page(S.STORMRAIDER)
    st = S.init(2, 2)
    st, _ = hit(st, page, H.ctx(now=0.0, level=9), u, 130.0)
    c = H.ctx(now=2.5, level=9)
    st, _ = hit(st, page, c, u, 130.0)
    assert float(st.sr_until[0]) == pytest.approx(6.5)
    assert float(st.sr_cd_until[0] - 2.5) == pytest.approx(15.2941, abs=1e-4)
    s = S.stats(st, page, H.ctx(now=3.0, level=9), R.ev(c, 2))
    assert float(s.percent_move_speed[0]) == pytest.approx(0.48, abs=1e-6)
    assert float(s.slow_resist[0]) == pytest.approx(0.5)
    assert float(s.percent_move_speed[1]) == 0.0
    # Expires after 4 s.
    s = S.stats(st, page, H.ctx(now=6.6, level=9), R.ev(c, 2))
    assert float(s.percent_move_speed[0]) == 0.0 and float(s.slow_resist[0]) == 0.0


def test_f29_stormraider_below_threshold_and_window():
    u = world()
    page = R.page(S.STORMRAIDER)
    st = S.init(2, 2)
    st, _ = hit(st, page, H.ctx(now=0.0, level=9), u, 120.0)
    st, _ = hit(st, page, H.ctx(now=2.5, level=9), u, 120.0)
    assert float(st.sr_until[0]) < 0.0                      # 240 < 250
    st2 = S.init(2, 2)
    st2, _ = hit(st2, page, H.ctx(now=0.0, level=9), u, 130.0)
    st2, _ = hit(st2, page, H.ctx(now=3.5, level=9), u, 130.0)
    assert float(st2.sr_until[0]) < 0.0                     # outside the 3 s window
    # Procs, DoTs and the follow-up pass all count (same tick, two reports).
    st3 = S.init(2, 2)
    st3, _ = hit(st3, page, H.ctx(now=1.0, level=9), u, 200.0)
    st3, _ = hit(st3, page, H.ctx(now=1.0, level=9), u, 60.0, flags=D.TAG_PROC | D.TAG_PERIODIC)
    assert float(st3.sr_until[0]) == pytest.approx(5.0)


def test_stormraider_cooldown_and_ranged():
    u = world()
    page = R.page(S.STORMRAIDER)
    st = S.init(2, 2)
    st, _ = hit(st, page, H.ctx(now=0.0, level=1, ranged=True), u, 300.0)
    assert float(st.sr_cd_until[0]) == pytest.approx(20.0)
    s = S.stats(st, page, H.ctx(now=1.0, ranged=True), R.ev(H.ctx(), 2))
    assert float(s.percent_move_speed[0]) == pytest.approx(0.36, abs=1e-6)
    st, _ = hit(st, page, H.ctx(now=10.0), u, 300.0)
    assert float(st.sr_until[0]) == pytest.approx(4.0)      # on cooldown: no retrigger
    st, _ = hit(st, page, H.ctx(now=20.0), u, 300.0)
    assert float(st.sr_until[0]) == pytest.approx(24.0)
    assert float(S.stormraider_cooldown(18.0)) == pytest.approx(10.0)


# ---- Summon Aery ------------------------------------------------------------

def test_aery_damage_delay_and_return():
    u = world()
    page = R.page(S.AERY)
    st = S.init(2, 2)
    st, _ = hit(st, page, H.ctx(now=0.0, bonus_ad=20.0), u, 50.0)
    assert float(st.aery_due[0]) == pytest.approx(0.45)
    st, out = run_periodic(st, page, u, 0.0, 0.6, bonus_ad=20.0)
    assert total(out, S.AERY, dst=1) == pytest.approx(12.0)            # lin(10,50)@1 + 10% of 20
    landed = [t for t, p in out if R.total(p, rune=S.AERY) > 0]
    assert landed[0] == pytest.approx(0.4667, abs=0.02)
    p = [p for t, p in out if R.total(p, rune=S.AERY) > 0][0]
    sel = np.asarray(p.valid)
    assert np.all(np.asarray(p.dtype)[sel] == D.PHYSICAL)             # adaptive: bonus AD > AP
    assert np.all(np.asarray(p.flags)[sel] & D.TAG_PROC)
    free = float(st.aery_free_t[0])
    assert free == pytest.approx(0.45 + 2.0 + float(S._aery_return_time(jnp.float32(300.0), 1.0)))
    assert 2.6 < free < 4.5
    # Cannot be re-sent before she returns.
    st2, _ = hit(st, page, H.ctx(now=1.0), u, 50.0)
    assert float(st2.aery_due[0]) > 1e8
    st3, _ = hit(st, page, H.ctx(now=free + 0.01), u, 50.0)
    assert float(st3.aery_due[0]) == pytest.approx(free + 0.46, abs=1e-3)


def test_aery_magic_on_ap_and_ignores_persistent_damage():
    u = world()
    page = R.page(S.AERY)
    st, _ = hit(S.init(2, 2), page, H.ctx(now=0.0, ap=40.0), u, 50.0, flags=SPELL)
    assert int(st.aery_dtype[0]) == D.MAGIC
    assert float(st.aery_raw[0]) == pytest.approx(12.0)
    st, _ = hit(S.init(2, 2), page, H.ctx(now=0.0), u, 50.0, flags=SPELL | D.TAG_PERIODIC)
    assert float(st.aery_due[0]) > 1e8
    # Ignite's first tick (summoner damage after a gap) sends Aery.
    st, _ = hit(S.init(2, 2), page, H.ctx(now=0.0), u, 20.0,
                flags=D.PROP_SUMMONER | D.TAG_PERIODIC | D.PROP_NO_OMNIVAMP)
    assert float(st.aery_due[0]) == pytest.approx(0.45)


# ---- Arcane Comet -----------------------------------------------------------

def test_comet_damage_distance_amp_and_cooldown():
    u = world(x1=300.0)
    page = R.page(S.COMET)
    st = S.init(2, 2)
    st, _ = hit(st, page, H.ctx(now=0.0), u, 50.0, flags=D.BASIC_ATTACK)
    assert float(st.comet_due[0]) > 1e8                                 # attacks don't trigger
    st, _ = hit(st, page, H.ctx(now=0.0), u, 0.0, flags=SPELL)
    assert float(st.comet_due[0]) > 1e8                                 # 0 damage excluded
    st, _ = hit(st, page, H.ctx(now=0.0), u, 50.0, flags=SPELL)
    assert float(st.comet_raw[0]) == pytest.approx(15.0 * 1.4)
    assert int(st.comet_dtype[0]) == D.MAGIC                            # zero contributions -> magic
    assert float(st.comet_cd_until[0]) == pytest.approx(20.0)
    st, out = run_periodic(st, page, u, 0.0, 1.0)
    assert total(out, S.COMET, dst=1) == pytest.approx(21.0)
    t_land = [t for t, p in out if R.total(p, rune=S.COMET) > 0]
    assert t_land == [pytest.approx(0.8, abs=0.04)]
    assert float(S.comet_cooldown(18.0)) == pytest.approx(8.0)
    assert float(S.comet_cooldown(9.0)) == pytest.approx(20 - 12 * 8 / 17, abs=1e-4)


def test_comet_physical_bonus_ad_max_range_and_dodge():
    u = world(x1=1000.0)
    page = R.page(S.COMET)
    st, _ = hit(S.init(2, 2), page, H.ctx(now=0.0, level=18, bonus_ad=100.0), u, 50.0, flags=SPELL)
    assert float(st.comet_raw[0]) == pytest.approx((100.0 + 10.0) * 2.0)
    assert int(st.comet_dtype[0]) == D.PHYSICAL
    # Target walks out of the 140 radius before landing.
    moved = world(x1=1300.0)
    _, out = run_periodic(st, page, moved, 0.0, 1.0, level=18)
    assert total(out, S.COMET) == 0.0


# ---- Deathfire Touch --------------------------------------------------------

def test_deathfire_spell_burn_total_and_amp():
    u = world()
    page = R.page(S.DEATHFIRE)
    st, _ = hit(S.init(2, 2), page, H.ctx(now=0.0), u, 50.0, flags=SPELL)
    st, out = run_periodic(st, page, u, 1 / 30, 5.0)
    ticks = [R.total(p, rune=S.DEATHFIRE) for _, p in out if R.total(p, rune=S.DEATHFIRE) > 0]
    assert ticks == pytest.approx([1.5] * 5 + [2.625] * 3)
    assert sum(ticks) == pytest.approx(15.375)
    p = [p for _, p in out if R.total(p, rune=S.DEATHFIRE) > 0][0]
    sel = np.asarray(p.valid)
    assert np.all(np.asarray(p.dtype)[sel] == D.MAGIC)
    assert np.all((np.asarray(p.flags)[sel] & (D.TAG_PROC | D.TAG_PERIODIC)) == (D.TAG_PROC | D.TAG_PERIODIC))


def test_deathfire_durations_snapshot_and_refresh_rule():
    u = world()
    page = R.page(S.DEATHFIRE)
    st, _ = hit(S.init(2, 2), page, H.ctx(now=0.0, level=18, ap=40.0), u, 50.0, flags=SPELL | D.TAG_AOE)
    assert float(st.dft_end[0, 1]) == pytest.approx(2.0)
    assert float(st.dft_dmg[0, 1]) == pytest.approx((12.0 + 1.0) / 2)
    st, out = run_periodic(st, page, u, 1 / 30, 3.0, level=18)          # stats snapshotted (ctx has 0 AP)
    assert total(out, S.DEATHFIRE) == pytest.approx(4 * 6.5)
    # A 2 s AoE application does not overwrite a 4 s spell burn with 3 s left.
    st, _ = hit(S.init(2, 2), page, H.ctx(now=0.0), u, 50.0, flags=SPELL)
    st, _ = hit(st, page, H.ctx(now=1.0), u, 50.0, flags=SPELL | D.TAG_AOE)
    assert float(st.dft_end[0, 1]) == pytest.approx(4.0)
    # ... but does once fewer than 2 s remain; continuity (amp clock) is kept.
    st, _ = hit(st, page, H.ctx(now=2.5), u, 50.0, flags=SPELL | D.TAG_AOE)
    assert float(st.dft_end[0, 1]) == pytest.approx(4.5)
    assert float(st.dft_start[0, 1]) == pytest.approx(0.0)
    # Pet damage: 1 s.
    st, _ = hit(S.init(2, 2), page, H.ctx(now=0.0), u, 50.0, flags=D.TAG_PET)
    assert float(st.dft_end[0, 1]) == pytest.approx(1.0)
    # Basic attacks and procs do not apply it.
    st, _ = hit(S.init(2, 2), page, H.ctx(now=0.0), u, 50.0, flags=D.ON_HIT_ITEM)
    assert float(st.dft_end[0, 1]) < 0.0


# ---- Scorch -----------------------------------------------------------------

def test_scorch_delay_and_cooldown():
    u = world()
    page = R.page(S.SCORCH)
    st, _ = hit(S.init(2, 2), page, H.ctx(now=0.0, level=18), u, 50.0, flags=SPELL)
    st, out = run_periodic(st, page, u, 0.0, 1.2, level=18)
    hits = [(t, R.total(p, rune=S.SCORCH)) for t, p in out if R.total(p, rune=S.SCORCH) > 0]
    assert len(hits) == 1 and hits[0][0] == pytest.approx(1.0, abs=0.04) and hits[0][1] == pytest.approx(40.0)
    st2, _ = hit(st, page, H.ctx(now=5.0), u, 50.0, flags=SPELL)
    assert float(st2.scorch_due[0]) > 1e8
    st3, _ = hit(st, page, H.ctx(now=10.0), u, 50.0, flags=SPELL)
    assert float(st3.scorch_due[0]) == pytest.approx(11.0)
    assert float(st3.scorch_raw[0]) == pytest.approx(20.0)


# ---- Axiom Arcanist / Transcendence ----------------------------------------

def test_axiom_packet_amp():
    u = world()
    page = R.page(S.AXIOM)
    c = H.ctx()
    p = D.packets(jnp.ones(4, bool), jnp.asarray([0, 0, 0, 1]), jnp.asarray([1, 1, 1, 0]), 100.0, D.PHYSICAL,
                  jnp.asarray([SPELL | D.PROP_ULTIMATE, SPELL | D.PROP_ULTIMATE | D.TAG_AOE, SPELL,
                               SPELL | D.PROP_ULTIMATE]))
    amp = S.packet_amp(S.init(2, 2), page, c, u, R.ev(c, 2), p)
    assert np.asarray(amp) == pytest.approx([0.12, 0.08, 0.0, 0.0])


def test_takedown_refunds_axiom_transcendence():
    u = world()
    page = R.perks([S.AXIOM, S.TRANSCENDENCE], [S.TRANSCENDENCE])
    st = S.init(2, 2)
    c = H.ctx(level=11)
    st, _ = S.on_takedown(st, page, c, u, R.ev(c, 2, kills=H.kills(2, champion_kill=(1, 1))))
    out = S.outputs(st, page, c, R.ev(c, 2))
    assert np.asarray(out.ult_cd_refund) == pytest.approx([0.07, 0.0])
    assert np.asarray(out.basic_cd_refund) == pytest.approx([0.2, 0.2])
    st, _ = S.on_takedown(st, page, c, u, R.ev(c, 2))                   # next tick: nothing
    out = S.outputs(st, page, c, R.ev(c, 2))
    assert float(out.ult_cd_refund[0]) == 0.0 and float(out.basic_cd_refund[0]) == 0.0
    c10 = H.ctx(level=10)
    st, _ = S.on_takedown(st, page, c10, u, R.ev(c10, 2, kills=H.kills(2, champion_assist=(1, 0))))
    out = S.outputs(st, page, c10, R.ev(c10, 2))
    assert float(out.basic_cd_refund[0]) == 0.0 and float(out.ult_cd_refund[0]) == pytest.approx(0.07)


def test_transcendence_haste_by_level():
    page = R.page(S.TRANSCENDENCE)
    for lv, ah in ((4, 0.0), (5, 5.0), (7, 5.0), (8, 10.0), (18, 10.0)):
        c = H.ctx(level=lv)
        assert float(S.stats(S.init(2, 2), page, c, R.ev(c, 2)).ability_haste[0]) == pytest.approx(ah)


# ---- movement / AF minors ---------------------------------------------------

def test_celerity_composes_with_move_speed():
    page = R.page(S.CELERITY)
    c = H.ctx()
    s = S.stats(S.init(2, 2), page, c, R.ev(c, 2))
    ms = SP.move_speed(340.0, s.move_speed[0], s.percent_move_speed[0], bonus_ms_amp=s.bonus_ms_amp[0])
    assert float(ms) == pytest.approx(340.0 * 1.01, abs=1e-3)
    ms = SP.move_speed(340.0, 0.0, s.percent_move_speed[0] + 0.025, bonus_ms_amp=s.bonus_ms_amp[0])
    assert float(ms) == pytest.approx(340.0 * (1.0 + 0.025 * 1.07 + 0.01), abs=1e-3)
    assert float(s.percent_move_speed[1]) == 0.0 and float(s.bonus_ms_amp[1]) == 0.0


def test_absolute_focus_threshold():
    page = R.page(S.ABSOLUTE_FOCUS)
    for hp, lv, af in ((800.0, 1, 3.0), (700.0, 1, 0.0), (701.0, 18, 30.0), (1000.0, 9, 3 + 27 * 8 / 17)):
        c = H.ctx(level=lv, max_hp=1000.0, hp=hp)
        assert float(S.stats(S.init(2, 2), page, c, R.ev(c, 2)).adaptive_force[0]) == pytest.approx(af, abs=1e-4)


def test_gathering_storm_steps():
    page = R.page(S.GATHERING_STORM)
    for t, af in ((599.0, 0.0), (600.0, 8.0), (1199.0, 8.0), (1200.0, 24.0), (1800.0, 48.0)):
        c = H.ctx(now=t)
        assert float(S.stats(S.init(2, 2), page, c, R.ev(c, 2, game_time=jnp.float32(t))).adaptive_force[0]) \
            == pytest.approx(af)


def test_waterwalking_river_and_decay():
    u = world()
    page = R.page(S.WATERWALKING)
    st = S.init(2, 2)
    c = H.ctx(now=5.0)
    river = jnp.asarray([True, True])
    s = S.stats(st, page, c, R.ev(c, 2, in_river=river))
    assert float(s.move_speed[0]) == pytest.approx(10.0) and float(s.adaptive_force[0]) == pytest.approx(13.0)
    assert float(s.move_speed[1]) == 0.0
    st, _ = S.periodic(st, page, c, u, R.ev(c, 2, in_river=river))
    c2 = H.ctx(now=5.5)
    s = S.stats(st, page, c2, R.ev(c2, 2))
    assert float(s.move_speed[0]) == pytest.approx(5.0, abs=1e-4) and float(s.adaptive_force[0]) == 0.0
    c3 = H.ctx(now=6.2)
    assert float(S.stats(st, page, c3, R.ev(c3, 2)).move_speed[0]) == 0.0


def test_nimbus_brackets_decay_and_ghosting():
    u = world()
    page = R.page(S.NIMBUS)
    yes = jnp.asarray([True, False])
    for cd, tp, ms in ((99.0, False, 0.15), (100.0, False, 0.35), (250.0, False, 0.35), (254.0, False, 0.45),
                       (300.0, False, 0.45), (50.0, True, 0.45)):
        c = H.ctx(now=10.0)
        ev = R.ev(c, 2, summoner_cast=yes, summoner_cooldown=jnp.full((2,), cd, jnp.float32),
                  summoner_is_teleport=jnp.asarray([tp, tp]))
        st, _ = S.on_cast(S.init(2, 2), page, c, u, ev)
        s = S.stats(st, page, c, ev)
        assert float(s.percent_move_speed[0]) == pytest.approx(ms, abs=1e-6)
        assert float(s.percent_move_speed[1]) == 0.0
    c1 = H.ctx(now=11.0)
    assert float(S.stats(st, page, c1, R.ev(c1, 2)).percent_move_speed[0]) == pytest.approx(0.225, abs=1e-6)
    assert bool(S.outputs(st, page, c1, R.ev(c1, 2)).ghosted[0])
    c2 = H.ctx(now=12.01)
    assert float(S.stats(st, page, c2, R.ev(c2, 2)).percent_move_speed[0]) == 0.0
    assert not bool(S.outputs(st, page, c2, R.ev(c2, 2)).ghosted[0])
    # A weaker cast does not replace a stronger running one.
    c3 = H.ctx(now=10.5)
    st2, _ = S.on_cast(st, page, c3, u, R.ev(c3, 2, summoner_cast=yes, summoner_cooldown=jnp.full((2,), 15.0)))
    assert float(st2.nim_ms[0]) == pytest.approx(0.45)


def test_manaflow_stacks_cap_and_restore():
    u = world()
    page = R.page(S.MANAFLOW)
    st = S.init(2, 2)
    for i in range(12):
        st, _ = hit(st, page, H.ctx(now=15.0 * i), u, 50.0, flags=SPELL)
    assert int(st.mf_stacks[0]) == 10
    c = H.ctx(now=200.0, mana=100.0, max_mana=300.0)
    assert float(S.stats(st, page, c, R.ev(c, 2)).mana[0]) == pytest.approx(250.0)
    st, eff = S.periodic(st, page, H.ctx(now=135.0 + 5.0, mana=100.0, max_mana=300.0), u, R.ev(c, 2))
    assert float(eff.mana[0]) == pytest.approx(0.01 * (550.0 - 100.0))
    _, eff = S.periodic(st, page, H.ctx(now=141.0, mana=100.0, max_mana=300.0), u, R.ev(c, 2))
    assert float(eff.mana[0]) == 0.0
    # Cooldown: two ability hits 5 s apart give one stack; CC also stacks.
    st = S.init(2, 2)
    st, _ = hit(st, page, H.ctx(now=0.0), u, 50.0, flags=SPELL)
    st, _ = hit(st, page, H.ctx(now=5.0), u, 50.0, flags=SPELL)
    assert int(st.mf_stacks[0]) == 1
    c = H.ctx(now=15.0)
    slowed = jnp.asarray([[False, True], [False, False]])
    st, _ = S.on_cc(st, page, c, u, R.ev(c, 2, cc=CC(slowed, jnp.zeros((2, 2), bool))))
    assert int(st.mf_stacks[0]) == 2


# ---- isolation and jit ------------------------------------------------------

def test_holder_isolation():
    u = world()
    page = R.perks([], [S.COMET, S.STORMRAIDER])
    st = S.init(2, 2)
    st, _ = hit(st, page, H.ctx(now=0.0), u, 400.0, flags=SPELL)       # holder 0 has no runes
    assert float(st.comet_due[0]) > 1e8 and float(st.sr_until[0]) < 0.0
    assert float(st.comet_due[1]) > 1e8 and float(st.sr_until[1]) < 0.0   # holder 1 was the victim
    st, _ = hit(st, page, H.ctx(now=0.0), u, 400.0, flags=SPELL, src=1, dst=0)
    assert float(st.comet_due[1]) == pytest.approx(0.8) and float(st.sr_until[1]) == pytest.approx(4.0)
    assert float(st.comet_due[0]) > 1e8


def test_hooks_under_jit():
    u = world()
    page = R.perks([S.COMET, S.DEATHFIRE, S.SCORCH], [S.AERY])
    p = R.hit_packet(0, 1, 50.0, flags=SPELL)
    rep = R.report(p, u)

    @jax.jit
    def go(st, now):
        c = H.ctx(now=now)
        st, _ = S.on_damage(st, page, c, u, R.ev(c, 2, report=rep))
        c2 = H.ctx(now=now + 1.0)
        st, eff = S.periodic(st, page, c2, u, R.ev(c2, 2))
        return st, eff.packets

    st, pk = go(S.init(2, 2), jnp.float32(0.0))
    assert R.total(pk, rune=S.COMET, dst=1) == pytest.approx(21.0)
    assert R.total(pk, rune=S.SCORCH, dst=1) == pytest.approx(20.0)
    assert R.total(pk, rune=S.DEATHFIRE, dst=1) == pytest.approx(1.5)
    assert R.total(pk, rune=S.AERY) == 0.0
