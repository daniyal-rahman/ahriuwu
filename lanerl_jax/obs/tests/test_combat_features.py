"""Combat schema provenance: direct HP, visibility and own readiness."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from lanerl_jax.obs.builder import build_observation
from lanerl_jax.obs.combat_features import append_combat_features, ENTITY_FEATURES, SELF_FEATURES
from lanerl_jax.parity.policy_driver import _lane_frames
from lanerl_jax.sim.init import lane_params
from lanerl_jax.train.wave_scenario import raw_state


def scene():
    p = lane_params()
    s = raw_state(SimpleNamespace(params=p), 0)
    s = s.replace(x=s.x.at[2].set(s.x[0]+100), y=s.y.at[2].set(s.y[0]),
                  hp=s.hp.at[2].set(1.7))
    return s, p


def observe(s, p, visibility=None):
    f = _lane_frames()[0]
    o = build_observation(s, 0, f, params=p, visibility=visibility)
    return o, append_combat_features(o, s, 0, f, p)


def test_combat_direct_hp_preserves_base_columns_and_masks_padding():
    s, p = scene()
    old, new = observe(s, p, jnp.ones_like(s.alive))
    row = int(np.flatnonzero(np.asarray(old.slot_unit)==2)[0])
    hp = 16+ENTITY_FEATURES.index('hp_points')
    maxhp = 16+ENTITY_FEATURES.index('max_hp')
    np.testing.assert_allclose(new.entities[row,hp], s.hp[2]/3000.)
    assert float(old.entities[row,3]) == 0. and float(new.entities[row,hp]) > 0.
    np.testing.assert_allclose(new.self_vec[16+SELF_FEATURES.index('hp_points')], s.hp[0]/3000.)
    np.testing.assert_allclose(new.entities[row,maxhp], s.max_hp[2]/3000.)
    np.testing.assert_array_equal(new.entities[:,:16], old.entities)
    np.testing.assert_array_equal(new.self_vec[:16], old.self_vec)
    assert np.isfinite(new.entities).all() and np.isfinite(new.self_vec).all()
    assert not np.asarray(new.entities[old.entity_pad_mask]).any()


def test_combat_hidden_units_targets_and_enemy_clocks_do_not_leak():
    s, p = scene()
    visibility = jnp.ones_like(s.alive).at[1].set(False)
    _, ref = observe(s, p, visibility)
    changed = s.replace(hp=s.hp.at[1].set(7), max_hp=s.max_hp.at[1].set(9999),
        level=s.level.at[1].set(15), aa_cooldown=s.aa_cooldown.at[1:].set(9),
        aa_windup=s.aa_windup.at[1:].set(3), target=jnp.full_like(s.target, 7),
        aa_target=jnp.full_like(s.aa_target, 8))
    _, got = observe(changed, p, visibility)
    np.testing.assert_array_equal(ref.entities, got.entities)
    np.testing.assert_array_equal(ref.self_vec, got.self_vec)


def test_combat_own_readiness_and_permutation():
    s, p = scene()
    old, ready = observe(s, p, jnp.ones_like(s.alive))
    ix = 16+SELF_FEATURES.index('attack_available')
    assert float(ready.self_vec[ix]) == 1.
    for changed in [s.replace(aa_cooldown=s.aa_cooldown.at[0].set(.2)),
                    s.replace(is_attacking=s.is_attacking.at[0].set(True)),
                    s.replace(alive=s.alive.at[0].set(False))]:
        assert float(observe(changed,p,jnp.ones_like(s.alive))[1].self_vec[ix]) == 0.
    perm = jnp.arange(31,-1,-1)
    shuffled = old._replace(entities=old.entities[perm], entity_pad_mask=old.entity_pad_mask[perm],
                           slot_unit=old.slot_unit[perm])
    got = append_combat_features(shuffled,s,0,_lane_frames()[0],p)
    np.testing.assert_array_equal(got.entities, ready.entities[perm])


def test_combat_zero_period_profile_has_finite_encoding():
    s,p = scene()
    # Fountain attack period is zero in the real profile table. Force the
    # visible test unit's period to zero to test the observation boundary.
    changed = dict(p)
    changed['attack_period'] = p['attack_period'].at[s.model[2]].set(0.)
    old,new = observe(s,changed,jnp.ones_like(s.alive))
    row = int(np.flatnonzero(np.asarray(old.slot_unit)==2)[0])
    assert np.isfinite(new.entities).all() and np.isfinite(new.self_vec).all()
    assert float(new.entities[row,16+ENTITY_FEATURES.index('attack_speed')]) == 5.


def test_movement_and_recall_mask_follow_own_cast_locks():
    s,p=scene()
    from lanerl_jax.train.action_masks import available_buttons
    from lanerl_rl.constants import BUTTON_INDEX
    for field in ('r_cast_ms','recall_windup_ms'):
        locked=s.replace(**{field:getattr(s,field).at[0].set(100.)})
        mask=np.asarray(available_buttons(observe(locked,p)[1].self_vec))
        assert mask[BUTTON_INDEX['noop']] and mask.sum()==1
    cooldown=s.replace(aa_cooldown=s.aa_cooldown.at[0].set(1.))
    assert bool(available_buttons(observe(cooldown,p)[1].self_vec)[BUTTON_INDEX['attack_move']])
