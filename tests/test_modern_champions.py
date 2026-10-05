"""Independent 26.19 combat examples, including mixed-champion interactions."""
import jax
import jax.numpy as jnp
import numpy as np
from lanerl_jax.sim import modern_bridge as m
from lanerl_jax.sim.orders import Orders, OrderKind as O, apply_orders
from lanerl_jax.sim.autoattack import AutoAttackOut
from lanerl_jax.sim.state import Kind


def lane():
    s=m.init_lane();p=m.make_params()
    s=s.replace(x=s.x.at[:2].set(jnp.array([5000.,5100.],jnp.float32)),y=s.y.at[:2].set(5000.),
                spell_level=s.spell_level.at[:2].set(jnp.ones((2,4),jnp.int8)))
    return s,p


def cast(s,p,side,kind,target=-1):
    return apply_orders(s,Orders(jnp.array([kind if i==side else O.NOOP for i in range(2)],jnp.int8),jnp.zeros(2),jnp.zeros(2),jnp.array([target if i==side else -1 for i in range(2)],jnp.int8)),p)


def advance(s,p,dt=100.):
    return m.advance(s,p,jnp.zeros_like(s.hp),jnp.zeros_like(s.hp),dt)


def test_stats_and_identity():
    s,p=lane()
    np.testing.assert_allclose(s.hp[:2],[690,650]);np.testing.assert_array_equal(s.champion.id[:2],[86,24])
    np.testing.assert_allclose(p['attack_damage'][:2],[69,68]);assert float(s.champion.mana[1])==339


def test_jax_mana_and_invalid_target_do_not_burn_cooldown():
    s,p=lane();bad=cast(s,p,1,O.CAST_Q,1)
    np.testing.assert_array_equal(bad.spell_cooldown,s.spell_cooldown)
    assert float(bad.champion.mana[1])==339
    s=s.replace(champion=s.champion.replace(mana=s.champion.mana.at[1].set(49)))
    assert not bool(m.status(s).can_cast[1,0])
    np.testing.assert_array_equal(cast(s,p,1,O.CAST_Q,0).champion.dash_ms,s.champion.dash_ms)


def test_leap_strike_consumes_w_only_on_enemy():
    s,p=lane();s=cast(s,p,1,O.CAST_W);s=cast(s,p,1,O.CAST_Q,0)
    s,b=advance(s,p,100.)
    np.testing.assert_allclose(b.damage_dealt[0],115.) # Q65 + W50, zero resistance
    assert float(s.champion.jax_w_ms[1])==0
    assert float(s.spell_cooldown[1,1])==7
    np.testing.assert_allclose(s.champion.mana[1],259.164,atol=.001)


def test_counterstrike_recast_costs_no_mana_and_uses_e_rank():
    s,p=lane();s=cast(s,p,1,O.CAST_E)
    assert float(s.champion.mana[1])==289
    s=s.replace(champion=s.champion.replace(jax_e_ms=s.champion.jax_e_ms.at[1].set(1000),jax_e_dodges=s.champion.jax_e_dodges.at[1].set(5)))
    s=cast(s,p,1,O.CAST_E);s,b=advance(s,p)
    np.testing.assert_allclose(b.damage_dealt[0],(40+.04*690)*2)
    assert float(s.spell_cooldown[1,2])==17
    assert float(s.champion.stun_ms[0])==1000


def test_garen_w_modern_reduction_shield_and_duration():
    s,p=lane();s=cast(s,p,0,O.CAST_W)
    assert float(s.champion.shield[0])==65
    s=cast(s,p,1,O.CAST_R);s,b=advance(s,p,250)
    np.testing.assert_allclose(b.damage_dealt[0],75) # rank1 R100, W25% DR
    assert float(s.champion.jax_r_ms[1])==8000
    assert float(s.champion.jax_r_armor[1])==45


def test_garen_r_true_damage_and_target_lock():
    s,p=lane();s=s.replace(hp=s.hp.at[1].set(450));s=cast(s,p,0,O.CAST_R,1)
    s,b=m.advance(s,p,jnp.full_like(s.hp,1000),jnp.full_like(s.hp,1000),435)
    np.testing.assert_allclose(b.damage_dealt[1],175) #125 +25% missing200


def test_garen_spin_tick_count_nearest_and_shred():
    s,p=lane();s=cast(s,p,0,O.CAST_E)
    total=0
    for _ in range(30):
        s,b=advance(s,p,100);total+=float(b.damage_dealt[1])
    np.testing.assert_allclose(total,7*(4+.4*69)*1.25,rtol=1e-6)
    assert float(s.champion.garen_ticks[0])==7
    assert float(s.champion.garen_shred_ms[1])>0
    assert not bool(s.buffs.e.active[0]);assert float(s.spell_cooldown[0,2])==9


def test_jax_has_no_garen_passive():
    s,p=lane();s=s.replace(hp=s.hp.at[:2].add(-100),ms_since_damaged=jnp.full_like(s.hp,9000))
    out,_=advance(s,p,1000)
    np.testing.assert_allclose(out.hp[:2],[592.07,550],atol=.001)


def test_jitted_mixed_lane_tick():
    from lanerl_jax.sim.step import tick
    s,p=lane();s=cast(s,p,0,O.CAST_W)
    out=jax.jit(lambda state:tick(state,p,enable_collision=False))(s)
    assert bool(jnp.isfinite(out.hp).all())
    assert out.modern


def hit_fixture(s, attacker=0, amount=100.):
    n=len(s.hp);z=jnp.zeros(n);f=jnp.zeros(n,bool);hit=f.at[attacker].set(True)
    return AutoAttackOut(z,z,f,f,hit,z.at[attacker].set(amount),f,hit)


def test_jax_dodge_consumes_garen_q_without_silence_or_damage():
    s,p=lane();s=cast(s,p,0,O.CAST_Q);s=cast(s,p,1,O.CAST_E)
    target=jnp.zeros(len(s.hp),jnp.int32).at[0].set(1)
    out,damage,silence,b=m.auto_hits(s,p,hit_fixture(s),target,jnp.zeros_like(s.hp),jnp.zeros_like(s.hp),s.buffs)
    assert float(damage[0])==0 and not bool(silence[0])
    assert not bool(b.q.active[0])
    assert float(out.champion.jax_e_dodges[1])==1


def test_counterstrike_does_not_dodge_turrets():
    s,p=lane();s=cast(s,p,1,O.CAST_E)
    turret=int(jnp.argmax(s.kind==Kind.TURRET));target=jnp.ones(len(s.hp),jnp.int32)
    out,damage,_,_=m.auto_hits(s,p,hit_fixture(s,turret),target,jnp.zeros_like(s.hp),jnp.zeros_like(s.hp),s.buffs)
    assert float(damage[turret])==100
    assert float(out.champion.jax_e_dodges[1])==0


def test_jax_r_every_third_hit_then_every_second_during_active():
    s,p=lane();target=jnp.zeros(len(s.hp),jnp.int32);damages=[]
    for _ in range(3):
        s,d,_,_=m.auto_hits(s,p,hit_fixture(s,1),target,jnp.zeros_like(s.hp),jnp.zeros_like(s.hp),s.buffs);damages.append(float(d[1]))
    np.testing.assert_allclose(damages,[100,100,175])
    s=s.replace(champion=s.champion.replace(jax_r_ms=s.champion.jax_r_ms.at[1].set(8000)))
    for expected in (100,175):
        s,d,_,_=m.auto_hits(s,p,hit_fixture(s,1),target,jnp.zeros_like(s.hp),jnp.zeros_like(s.hp),s.buffs)
        assert float(d[1])==expected


def test_death_clears_spins_and_starts_full_cooldowns():
    s,p=lane();s=cast(s,p,0,O.CAST_E);s=cast(s,p,1,O.CAST_E);s=cast(s,p,1,O.CAST_W)
    died=jnp.zeros(len(s.hp),bool).at[:2].set(True)
    out=m.finish(s,s.alive,died,jnp.zeros_like(died),jnp.full(len(s.hp),-1))
    assert not bool(out.buffs.e.active.any())
    np.testing.assert_allclose(out.champion.jax_e_ms,0)
    np.testing.assert_allclose(out.spell_cooldown[:2,2],[9,17]);assert float(out.spell_cooldown[1,1])==7


def test_vmap_casts_keep_champion_resources_independent():
    s,p=lane();batch=jax.tree.map(lambda x:jnp.stack([x,x]),s)
    result=jax.jit(jax.vmap(lambda state:cast(state,p,1,O.CAST_E)))(batch)
    np.testing.assert_allclose(result.champion.mana[:,1],[289,289])


def test_modern_observation_exposes_identity_mana_and_own_buffs():
    from lanerl_jax.obs.builder import build_observation,MODERN_SELF_DIM
    from lanerl_jax.obs.frame import make_lane_frame
    s,p=lane();s=cast(s,p,1,O.CAST_E)
    frame=make_lane_frame(jnp.array([0.,0.]),jnp.array([10000.,0.]),jnp.array([-100.,0.]))
    obs=build_observation(s,1,frame,params=p)
    assert obs.self_vec.shape==(MODERN_SELF_DIM,)
    np.testing.assert_allclose(obs.self_vec[16:21],[0,1,1,0,289/339])
    assert float(obs.self_vec[24])==1


def test_modern_scan_preserves_carry_dtypes_and_cooldowns():
    from lanerl_jax.sim.step import step_decision
    s=m.init_lane();p=m.make_params()
    s=cast(s,p,0,O.CAST_E);s=cast(s,p,1,O.CAST_E)
    result=jax.jit(lambda state:step_decision(state,p,step_ticks=240,enable_collision=False))(s)
    assert int(result.tick)==int(s.tick)+240
    assert not bool(result.buffs.e.active.any())
    assert float(result.champion.jax_e_ms[1])==0
    assert float(result.spell_cooldown[0,2])>0 and float(result.spell_cooldown[1,2])>0


def test_garen_regen_breakpoints_apply_at_level_seven_and_fourteen():
    s,p=lane();s=s.replace(hp=s.hp.at[0].set(500),ms_since_damaged=s.ms_since_damaged.at[0].set(9000))
    for level,percent in ((6,2.5),(7,3.3),(13,8.1),(14,8.5),(18,10.1)):
        out,_=advance(s.replace(level=s.level.at[0].set(level)),p,1000)
        np.testing.assert_allclose(out.hp[0],500+690*percent/500,atol=.001)


def test_legacy_wire_rebuilder_rejects_modern_champions():
    import pytest
    from lanerl_jax.parity.policy_driver import StateRebuilder
    with pytest.raises(ValueError,match="requires a modern collector"):
        StateRebuilder().rebuild({"u":[{"k":"Champion","modern":{"patch":"26.19","id":24}}]})
