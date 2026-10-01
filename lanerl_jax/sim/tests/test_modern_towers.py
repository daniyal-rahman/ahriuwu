import jax
import jax.numpy as jnp
import numpy as np
from lanerl_jax.sim import modern_towers as t


def exposed(now=100.):
    return t.advance(t.init_outer_turret(), now, True, False)


def test_plate_thresholds_and_independent_bulwark():
    s = exposed()
    h = t.apply_turret_damage(s, 100., 0., 0., 900.)
    assert h.plates == 1 and h.local_gold == 120
    assert t.resistance(h.state, 100., 1) == 90
    h2 = t.apply_turret_damage(h.state, 110., 0., 0., 6750.) # backdoor scales to1350
    assert h2.plates == 1
    assert t.resistance(h2.state, 119., 1) == 120
    assert t.resistance(h2.state, 120., 1) == 90
    assert t.resistance(h2.state, 130., 1) == 60
    assert t.resistance(h2.state, 119., 5) == 160


def test_decay_and_ad_boundaries():
    times=jnp.array([0., 29.99, 30., 90., 810., 900.])
    np.testing.assert_allclose(jax.jit(t.outer_attack_damage)(times), [182,182,194,206,350,350])
    np.testing.assert_allclose(t.plate_value(jnp.array([659.9,660.,720.,780.,840.,1000.])), [120,110,100,90,80,80])


def test_backdoor_grace_and_true_damage():
    s=t.advance(t.init_outer_turret(), 10., True, True)
    assert t.apply_turret_damage(s,12.999,0,0,100).damage == 100
    assert t.apply_turret_damage(s,13.,0,0,100).damage == 20
    s=t.advance(s,13.,True,True)
    assert t.apply_turret_damage(s,13.,0,0,100).damage == 100


def test_growth_suppression_fast_forward_consumption():
    s=t.advance(t.init_outer_turret(),100.,False,True)
    assert not s.growth_active
    s=t.advance(s,400.,True,False)
    assert s.growth_active
    h=t.apply_turret_damage(s,400.,0,0,0,champion_attack=True,growth_min_fraction=.02,growth_max_fraction=.033)
    np.testing.assert_allclose(h.overgrowth_damage,297.,rtol=1e-6)
    assert not h.state.growth_active
    assert h.state.growth_since == 400
    assert not t.advance(h.state,489.99,False,False).growth_active
    assert t.advance(h.state,490.,False,False).growth_active
    blocked=t.apply_turret_damage(s,404.,0,0,0,champion_attack=True,growth_min_fraction=.02,growth_max_fraction=.033)
    assert blocked.overgrowth_damage == 0 and blocked.state.growth_active


def test_growth_clock_and_separate_melee_packet():
    s=exposed(100.)
    np.testing.assert_allclose([t.overgrowth_damage(s,x,.02,.033) for x in [100.,160.,280.,400.]], [180,180,238.5,297])
    h=t.apply_turret_damage(s,100.,160,0,0,champion_attack=True,melee_champion=True,growth_min_fraction=.02,growth_max_fraction=.033)
    assert h.damage == 300 #100physical*1.2 plus180 crystal


def test_ap_attack_uses_both_contributions():
    damage,magic=t.champion_structure_attack(60.,40.,100.)
    assert damage == 160 and magic
    assert not t.champion_structure_attack(60.,60.,100.)[1]


def test_shot_ramp_across_targets_and_expiry():
    s=t.init_outer_turret()
    expected=[194,291,388,485,485]
    for now,d in zip([30.,31.,32.,33.,34.],expected):
        s,actual=t.champion_shot_impact(s,now,0.)
        assert actual == d
    _,actual=t.champion_shot_impact(s,39.,100.)
    np.testing.assert_allclose(actual,194/1.7)
    dead=s._replace(hp=jnp.float32(0))
    assert t.champion_shot_impact(dead,35.,0.)[1] == 0


def test_minion_hits_and_target_lock():
    np.testing.assert_allclose(t.minion_shot_damage(1000.,jnp.arange(4)),[450,700,140,50])
    e=jnp.array([True,True,True]); d=jnp.array([100.,200.,300.]); p=jnp.array([t.CHAMPION,t.MELEE,t.CANNON_SUPER])
    none=jnp.zeros(3,bool)
    assert t.select_target(-1,e,d,p,none)==2
    assert t.select_target(1,e,d,p,none)==1
    assert t.select_target(1,e,d,p,jnp.array([True,False,False]))==0
    assert t.select_target(1,~e,d,p,none)==-1


def test_multiplate_kill_is_paid_once_and_jit_vmap():
    f=jax.jit(jax.vmap(lambda x:t.apply_turret_damage(exposed(),100.,0.,0.,x)))
    hits=f(jnp.array([900.,9000.,18000.]))
    np.testing.assert_array_equal(hits.plates,[1,5,5])
    assert hits.local_gold[1] == 600
    dead=t.apply_turret_damage(exposed(),100.,0,0,10000).state
    assert t.apply_turret_damage(dead,100.,0,0,10000).local_gold==0


def test_reward_assist_ignores_death_and_range():
    assert t.local_reward_eligible(False,5000.,90.,100.)
    assert not t.local_reward_eligible(False,5000.,89.99,100.)
    assert t.local_reward_eligible(True,1200.,-jnp.inf,100.)
