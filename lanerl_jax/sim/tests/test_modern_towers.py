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
    np.testing.assert_allclose(h.damage,300.) #100physical*1.2 plus180 crystal


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
    # Changed: super shot is 7 % of max HP (client item 1511), not 5 % (TOWERS D3).
    np.testing.assert_allclose(t.minion_shot_damage(1000.,jnp.arange(4)),[450,700,140,70])
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


def test_all_tiers_hp_damage_and_plate_rules():
    states=jax.vmap(t.init_turret)(jnp.arange(4))
    np.testing.assert_allclose(states.hp,[9000,5000,4750,3500])
    np.testing.assert_allclose(jax.vmap(t.attack_damage,in_axes=(0,None))(jnp.arange(4),180.),[218,203,203,181])
    np.testing.assert_allclose(jax.vmap(t.resistance,in_axes=(0,None,None))(states,1000.,1),[0,60,60,60])
    for tier in [t.INNER,t.INHIBITOR]:
        state=t.advance(t.init_turret(tier),1000.,True,False)
        hit=t.apply_turret_damage(state,1000.,0,0,state.hp)
        assert hit.plates==5 and hit.local_gold==600
    nexus=t.advance(t.init_turret(t.NEXUS),1000.,True,False)
    hit=t.apply_turret_damage(nexus,1000.,0,0,nexus.hp)
    assert hit.plates==0 and hit.local_gold==0 and not nexus.growth_active


def test_base_regen_segments_and_nexus_respawn():
    base=t.init_turret(t.INHIBITOR)
    for frac,cap in [(.2,.3),(.5,.75),(.9,1.)]:
        state=base._replace(hp=base.max_hp*frac)
        assert t.regenerate_and_respawn(state,500.,10000.).hp==base.max_hp*cap
    nexus=t.advance(t.init_turret(t.NEXUS),1000.,True,False)
    hit=t.apply_turret_damage(nexus,1000.,0,0,nexus.hp)
    assert t.regenerate_and_respawn(hit.state,1179.,1.).hp==0
    returned=t.regenerate_and_respawn(hit.state,1180.,1.)
    assert returned.hp==1400 and jnp.isinf(returned.respawn_at)
    assert t.regenerate_and_respawn(returned,1181.,1000.).hp==1400


def test_locked_lane_growth_clock_and_minion_tier_damage():
    inner=t.init_turret(t.INNER,jnp.inf)
    assert not t.advance(inner,500.,False,False).growth_active
    inner=t.unlock(inner,500.)
    assert not t.advance(inner,589.,False,False).growth_active
    assert t.advance(inner,590.,False,False).growth_active
    np.testing.assert_allclose(t.minion_shot_damage(1000.,2,tier=jnp.arange(4)),[140,110,80,80])
    # Changed: 7 % super shot, and by default the %-max-HP amount is not
    # reduced by armor (README X-3, MINIONS U-14); the TOWERS §4.2 mitigated
    # reading stays available behind mitigated=True.
    np.testing.assert_allclose(t.minion_shot_damage(1000.,3,100.),70.)
    np.testing.assert_allclose(t.minion_shot_damage(1000.,3,100.,mitigated=True),70/1.7)
    np.testing.assert_allclose(t.minion_shot_damage(2000.,3,100.,mitigated=True),82.3529,rtol=1e-5)


def test_explicit_growth_approximation_endpoints_and_finite_runtime():
    # Changed: client item-1524 curve (TOWERS §6.2/D1/D2) replaces the linear
    # max interpolation and the 1-18 clamp: max = (1.6+0.4L)% x
    # (1.65 + 0.5 clip((L-1)/17)); L9.5 high 0.111 -> 0.1026, L20 extends.
    low,high=jax.jit(t.overgrowth_level_fractions)(jnp.array([1.,9.5,18.,20.]))
    np.testing.assert_allclose(low,[.02,.054,.088,.096],rtol=1e-6)
    np.testing.assert_allclose(high,[.033,.1026,.1892,.2064],rtol=1e-5)
    state=exposed(100.)
    hit=jax.jit(lambda s:t.apply_turret_damage(s,100.,0.,0.,0.,champion_attack=True,average_team_level=9.5))(state)
    np.testing.assert_allclose(hit.damage,486.,rtol=1e-5)
    assert jnp.isfinite(hit.state.hp)
    assert hit.state.plates.dtype == state.plates.dtype


def test_jitted_scan_preserves_state_dtypes_through_plate_and_growth():
    def body(state,now):
        state=t.advance(state,now,jnp.bool_(True),jnp.bool_(False))
        hit=t.apply_turret_damage(state,now,jnp.float32(40),jnp.float32(0),jnp.float32(0),
                                  champion_attack=True,average_team_level=jnp.float32(3))
        return hit.state,hit.damage
    result,damage=jax.jit(lambda state:jax.lax.scan(body,state,jnp.arange(30,120,dtype=jnp.float32)))(t.init_outer_turret())
    assert result.plates==2 and jnp.all(jnp.isfinite(damage))
    assert result.growth_since==100


def test_overgrowth_client_curve_fixtures():
    # TOWERS §13.3: outer 9000 HP, og(L, g) at g<=60 / 180 / >=300.
    s=t.init_outer_turret()   # cooldown start 10 -> appears 100
    for level,lo,mid,hi in [(1,180.,238.5,297.),(3,252.,341.3,430.6),(6,360.,503.5,646.9),
                            (9,468.,675.2,882.3),(12,576.,856.4,1136.8),(18,792.,1247.4,1702.8),
                            (20,864.,1360.8,1857.6)]:
        a,b=t.overgrowth_level_fractions(float(level))
        got=[t.overgrowth_damage(s,100.+g,a,b) for g in (60.,180.,300.)]
        np.testing.assert_allclose(got,[lo,mid,hi],atol=.06)
    inner=t.init_turret(t.INNER,100.)
    a,b=t.overgrowth_level_fractions(6.)
    np.testing.assert_allclose([t.overgrowth_damage(inner,250.,a,b),t.overgrowth_damage(inner,490.,a,b)],
                               [200.,359.4],atol=.06)


def test_negative_resist_is_kept_not_clamped():
    # Decayed outer turret (0 base resist) shredded below zero by flat pen
    # stays at 0 (flat pen cannot cross zero), but a resist that is already
    # negative keeps the negative mitigation branch (DAMAGE D1, README item 4).
    np.testing.assert_allclose(t.resistance_multiplier(-20.),2-100/120)
    s=t.advance(t.init_outer_turret(),900.,True,False)
    hit=t.apply_turret_damage(s,900.,100.,0.,0.,armor_pen_flat=30.)
    np.testing.assert_allclose(hit.damage,100.)


def test_buildings_regen_respawn_and_no_plates():
    inhib=t.init_turret(t.INHIBITOR_BUILDING); nexus=t.init_turret(t.NEXUS_BUILDING)
    assert inhib.max_hp==4000 and nexus.max_hp==5500
    np.testing.assert_allclose(t.regenerate_and_respawn(inhib._replace(hp=jnp.float32(1000.)),0.,10.).hp,1150.)
    np.testing.assert_allclose(t.regenerate_and_respawn(nexus._replace(hp=jnp.float32(1000.)),0.,10.).hp,1200.)
    hit=t.apply_turret_damage(t.advance(inhib,100.,False,False),100.,0.,0.,4000.)
    assert hit.destroyed and hit.plates==0 and hit.damage==4000   # no backdoor DR on buildings
    assert hit.state.respawn_at==400.
    assert t.regenerate_and_respawn(hit.state,399.,1.).hp==0
    assert t.regenerate_and_respawn(hit.state,400.,1.).hp==4000
    np.testing.assert_allclose(t.ATTACK_PERIOD,1.20048,rtol=1e-5)
    np.testing.assert_allclose(t.WINDUP_S,.1669,rtol=1e-3)
