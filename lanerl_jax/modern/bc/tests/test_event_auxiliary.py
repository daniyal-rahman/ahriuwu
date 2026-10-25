import numpy as np

from lanerl_jax.modern.bc.event_auxiliary import future_targets


def example():
    n = 64
    own = np.zeros((n, 40), np.float32)
    own[:, 0] = np.arange(n)*.01
    own[:, 2] = .8
    entities = np.zeros((n, 32, 25), np.float32)
    entities[:, 0, 1] = 100/3000
    entities[:, 0, 3] = 1
    ids = np.full((n, 32), -1)
    ids[:, 0] = 9
    return dict(t=np.arange(n)*.1, self=own, entities=entities,
                pad=np.zeros((n, 32), bool), valid=np.ones(n, bool), unit_ids=ids)


def test_future_positions_use_current_anchor_and_never_modify_inputs():
    a = example()
    before = {k: v.copy() for k, v in a.items()}
    b = future_targets(a)
    assert b['own_future'][5] == 7*24+14
    assert b['enemy_future'][5] == 7*24+13
    assert b['own_future_valid'][:44].all() and not b['own_future_valid'][44:].any()
    np.testing.assert_allclose(b['own_hp_future'][:34], .8)
    assert b['own_hp_future_valid'][:34].all()
    a['self'][:, :2] += 4
    shifted = future_targets(a)
    np.testing.assert_array_equal(b['own_future'], shifted['own_future'])
    np.testing.assert_array_equal(b['enemy_future'], shifted['enemy_future'])
    for k in a:
        if k != 'self':
            np.testing.assert_array_equal(a[k], before[k])
    np.testing.assert_array_equal(a['self'][:, 2:], before['self'][:, 2:])


def test_future_masks_cover_entire_horizon_and_enemy_visibility_identity():
    a = example()
    a['valid'][12] = False
    a['pad'][25, 0] = True
    a['unit_ids'][40:, 0] = 10
    b = future_targets(a)
    assert not b['own_future_valid'][:13].any()
    assert b['own_future_valid'][13:44].all()
    assert not b['enemy_future_valid'][15:26].any()
    assert not b['enemy_future_valid'][30:41].any()
    assert b['enemy_future_valid'][41:54].all()
    a['self'][35, 10] = 1
    a['self'][48:, 0] += 1
    a['t'][28:] += .1
    c = future_targets(a)
    assert not c['own_future_valid'][8:44].any()
    assert not c['own_hp_future_valid'].any()


def test_missing_hp_and_short_sequences_do_not_fabricate_targets():
    a = example()
    a['self'][:, 2] = np.nan
    b = future_targets(a)
    assert not b['own_hp_future_valid'].any()
    short = {k: v[:8] for k, v in a.items()}
    c = future_targets(short)
    assert all(not v.any() for k, v in c.items() if k.endswith('_valid'))


def test_auxiliary_loss_has_expected_scale_and_hp_gradient(monkeypatch):
    from types import SimpleNamespace
    import jax
    import jax.numpy as jnp
    from lanerl_jax.modern.bc import event_loss as loss_module
    from lanerl_jax.modern.bc.event_policy import EventLogits

    def forward(params, batch):
        return EventLogits(jnp.zeros((1, 2, 9)), jnp.zeros((1, 2, 32)),
                           jnp.zeros((1, 2, 6)), jnp.zeros((1, 2, 4)),
                           None, None, None, jnp.zeros((1, 2, 336)),
                           jnp.zeros((1, 2, 336)), jnp.full((1, 2), params['hp']))
    monkeypatch.setattr(loss_module, 'make_event_forward', lambda policy: forward)
    policy = SimpleNamespace(cfg=SimpleNamespace(grid_x=3, grid_y=2, n_angles=4))
    zeros = jnp.zeros((1, 2), jnp.int32)
    batch = dict(button=zeros, weight=jnp.ones((1, 2)), aim_valid=zeros.astype(bool),
                 aim_mode=zeros, screen_x=zeros, screen_y=zeros, target_row=zeros,
                 angle=jnp.zeros((1, 2)), own_future=zeros, enemy_future=zeros,
                 own_future_valid=jnp.ones((1, 2), bool), enemy_future_valid=jnp.ones((1, 2), bool),
                 own_hp_future=jnp.full((1, 2), .8), own_hp_future_valid=jnp.ones((1, 2), bool))
    batch.update(entities=jnp.zeros((1, 2, 32, 25)), pad=jnp.ones((1, 2, 32), bool),
                 own_id=jnp.zeros((1, 2), jnp.int32))
    params = dict(hp=jnp.asarray(0.))
    base, _ = loss_module.make_event_loss(policy)(params, batch)
    auxiliary = loss_module.make_event_loss(policy, auxiliary_weight=.1)
    value, info = auxiliary(params, batch)
    np.testing.assert_allclose(value-base, .1*(2*np.log(336)+.075), rtol=1e-5)
    np.testing.assert_allclose(jax.grad(lambda p: auxiliary(p, batch)[0])(params)['hp'], -.01)
    assert info['own_future_count'] == 2
