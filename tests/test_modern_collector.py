"""Collector profile and wire regressions; no server process required."""
import json
import numpy as np
import pytest
from lanerl_jax.parity.policy_driver import StateRebuilder, pending_rank_up
from lanerl_jax.train import champion_profile as profile


def wire(names=('Garen','Jax')):
    units=[]
    for i,name in enumerate(names):
        m=dict(patch='26.19',schema=2,castCounts=[0,0,0,0],id=86 if name=='Garen' else 24,
               mana=150,maxMana=400,q=0,haste=0,w=0,shield=0,shieldTime=0,
               e=0,eElapsed=0,spinTicks=0,spinCount=0,dodges=0,
               passiveStacks=0,passiveTime=0,r=0,rArmor=0,rHits=0,
               jumpTime=0,kills=0,stunned=False,silenced=False,casting=False,rPending=0)
        units.append(dict(k='Champion',id=i+1,tm=100+100*i,x=5000+100*i,y=5000,
                          hp=650,mhp=690,dead=False,lvl=6,gold=475,cs=0,
                          sl=[1,1,3,1],ad=80,ap=0,ar=40,mr=40,se=[1]*4,modern=m))
    return dict(t=1000,u=units)


def test_wire_resources_buffs_and_identity():
    f=wire();g,j=[u['modern'] for u in f['u']]
    g.update(q=3,e=2,eElapsed=1,shield=55,shieldTime=.5,kills=25)
    j.update(w=4,e=1.5,passiveStacks=6,r=7,rArmor=45,stunned=True)
    s,ids=StateRebuilder(('Garen','Jax')).rebuild(f)
    assert s.modern
    np.testing.assert_array_equal(s.champion.id[:2],[86,24])
    assert s.champion.mana[1]==150 and s.champion.max_mana[1]==400
    assert s.champion.jax_e_ms[1]==1500 and s.champion.jax_w_ms[1]==4000
    assert s.champion.jax_stacks[1]==6 and s.champion.jax_r_ms[1]==7000
    assert s.champion.shield[0]==55 and s.champion.garen_w_stacks[0]==25
    assert s.champion.stun_ms[1]>0 and s.buffs.q.active[0] and s.buffs.e.active[0]
    assert not s.buffs.e.active[1] and not s.buffs.w_passive.any()
    # Reconstruction starts clean each frame, including after an episode reset.
    s,_=StateRebuilder(('Garen','Jax')).rebuild(wire())
    assert not s.buffs.e.active.any() and not s.champion.jax_e_ms.any()


@pytest.mark.parametrize('pair',[('Jax','Garen'),('Jax','Jax'),('Garen','Garen')])
def test_all_pairings(pair):
    s,_=StateRebuilder(pair).rebuild(wire(pair))
    assert tuple(s.champion.id[:2])==tuple(24 if n=='Jax' else 86 for n in pair)


def test_mismatched_or_old_wire_rejected():
    r=StateRebuilder(('Garen','Jax'))
    for change in ({'id':86},{'patch':'26.18'},{'schema':1}):
        f=wire();f['u'][1]['modern'].update(change)
        with pytest.raises(ValueError,match='modern wire requires'):r.rebuild(f)
    with pytest.raises(ValueError):r.rebuild(dict(t=0,u=[]))


def test_jax_rank_order():
    j=wire()['u'][1];j.update(lvl=4,sl=[1,1,1,0])
    assert pending_rank_up(j)==1
    j['sl']=[1,2,1,0]
    assert pending_rank_up(j) is None


def test_config_and_checkpoint_contract(tmp_path):
    pair=('Jax','Garen')
    cfg=json.loads(profile.game_config(pair,tmp_path/'server/bin/DeadProbe',tmp_path).read_text())
    assert [p['champion'] for p in cfg['players']]==list(pair)
    assert all(not p['runes'] and not p['talents'] for p in cfg['players'])
    modern={'collector':{'modern_champions':pair},'train':{'policy':{'self_dim':28}}}
    profile.validate_checkpoint(modern,pair)
    for wanted in (None,('Garen','Jax')):
        with pytest.raises(ValueError):profile.validate_checkpoint(modern,wanted)
    with pytest.raises(ValueError):profile.validate_checkpoint({},pair)
    with pytest.raises(ValueError):profile.game_config(pair,None,tmp_path)
    assert profile.self_dim(None)==16


def test_jax_farming_profile_observation_and_reset(tmp_path):
    from lanerl_jax.train.jax_farm import JaxFarmCollector
    from lanerl_jax.sim.config import SimConfig
    c=JaxFarmCollector(1,tmp_path,teams=(0,1),modern_champions=('Jax','Garen'),
                       sim_config=SimConfig.training(route_artifact=None))
    try:
        obs,_=c.observe()
        assert obs.self_vec.shape==(2,28)
        np.testing.assert_array_equal(np.asarray(obs.self_vec)[:,16:20],[[0,1,1,0],[1,0,0,1]])
        c.restart_done(np.array([True,True]))
        np.testing.assert_array_equal(c.states.champion.id[0,:2],[24,86])
        np.testing.assert_allclose(c.states.max_hp[0,:2],[650,690])
        assert c.states.modern
    finally:
        c.close()


def test_cast_memory_uses_events_not_delayed_cooldowns():
    f=wire();r=StateRebuilder(('Garen','Jax'))
    for u in f['u']:u.update(vb=True,vr=True)
    r.rebuild(f)
    f['t']=1100;f['u'][1]['modern']['castCounts'][1]=1
    s,_=r.rebuild(f)
    assert s.observed_enemy_cast_ms[0,1]==0
    f['t']=1200;f['u'][1]['cd1']=7000
    s,_=r.rebuild(f)
    assert s.observed_enemy_cast_ms[0,1]==100
    # Hidden events must not refresh enemy cast memory.
    f['u'][1]['vb']=False;r.rebuild(f)
    f['t']=1300;f['u'][1]['modern']['castCounts'][1]=2
    s,_=r.rebuild(f)
    assert s.observed_enemy_cast_ms[0,1]==200


def test_worktree_vendor_override(monkeypatch,tmp_path):
    from lanerl_train import paths
    monkeypatch.setenv('LANERL_VENDOR_ROOT',str(tmp_path/'vendor'))
    assert paths.vendor_root()==tmp_path/'vendor'
    assert paths.dotnet_root()==tmp_path/'vendor/dotnet'
