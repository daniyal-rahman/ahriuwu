import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.modern.bc.event_next import next_event_targets, immediate_logits


def test_next_targets_include_now_and_mask_censoring_and_gaps():
    b = np.array([0, 0, 3, 0, 0, 0, 5, 0, 0, 0])
    valid = np.ones(10, bool)
    valid[4] = False
    a = next_event_targets(b, valid, np.arange(10)*.1, horizon=2)
    np.testing.assert_array_equal(a['next_button'], [3, 3, 3, 0, 0, 5, 5, 0, 0, 0])
    np.testing.assert_array_equal(a['next_delay'], [2, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    assert not a['next_valid'][3:5].any()
    assert a['next_valid'][7] and not a['next_observed'][7]
    assert not a['next_valid'][8:].any()
    t = np.arange(10)*.1
    t[2:] += 1
    broken = next_event_targets(b, np.ones(10, bool), t, horizon=2)
    assert not broken['next_valid'][:2].any()


def test_immediate_distribution_retains_all_remaining_mass_as_continue():
    event = np.log(np.array([.2, .1, .1, .1, .1, .1, .1, .1, .1]))
    delay = np.tile(np.log([.25, .75]), (8, 1))
    p = np.asarray(jax.nn.softmax(immediate_logits(jnp.asarray(event), jnp.asarray(delay))))
    np.testing.assert_allclose(p[0], .8, atol=1e-6)
    np.testing.assert_allclose(p[1:], .025, atol=1e-6)
    np.testing.assert_allclose(p.sum(), 1., atol=1e-6)
    derivative = jax.grad(lambda x: -immediate_logits(jnp.asarray(event), x)[3])(jnp.asarray(delay))
    assert np.isfinite(derivative).all() and np.linalg.norm(derivative) > 0


def test_next_event_targets_do_not_modify_observations_or_current_aim_labels():
    button = np.array([1, 0, 0, 5, 0, 0, 0, 0])
    copy = button.copy()
    a = next_event_targets(button, np.ones(8, bool), np.arange(8)*.1, horizon=2)
    np.testing.assert_array_equal(button, copy)
    assert a['next_button'][1] == 5 and a['next_delay'][1] == 2
    assert a['next_button'][0] == 1 and a['next_delay'][0] == 0


def test_next_event_loss_trains_delay_for_next_type_not_current_aim_button(monkeypatch):
    from types import SimpleNamespace
    from lanerl_jax.modern.bc import event_loss as module
    from lanerl_jax.modern.bc.event_policy import EventLogits
    def forward(params, batch):
        return EventLogits(jnp.zeros((1, 2, 9)), jnp.zeros((1, 2, 32)),
                           jnp.zeros((1, 2, 6)), jnp.zeros((1, 2, 4)),
                           None, None, None, None, None, None, params['delay'])
    monkeypatch.setattr(module, 'make_event_forward', lambda policy: forward)
    policy = SimpleNamespace(cfg=SimpleNamespace(grid_x=3, grid_y=2, n_angles=4, next_event_horizon=1))
    zero = jnp.zeros((1, 2), jnp.int32)
    batch = dict(button=jnp.ones((1, 2), jnp.int32), weight=jnp.ones((1, 2)), aim_valid=zero.astype(bool),
                 aim_mode=zero, screen_x=zero, screen_y=zero, target_row=zero, angle=zero,
                 next_button=jnp.full((1, 2), 3), next_delay=jnp.ones((1, 2), jnp.int32),
                 next_valid=jnp.ones((1, 2), bool), next_observed=jnp.ones((1, 2), bool))
    batch.update(entities=jnp.zeros((1, 2, 32, 25)), pad=jnp.ones((1, 2, 32), bool),
                 own_id=jnp.zeros((1, 2), jnp.int32))
    params = dict(delay=jnp.zeros((1, 2, 8, 2)))
    loss = module.make_event_loss(policy)
    value, info = loss(params, batch)
    np.testing.assert_allclose(value, np.log(9)+np.log(2), rtol=1e-5)
    grad = np.asarray(jax.grad(lambda p: loss(p, batch)[0])(params)['delay'])
    assert np.linalg.norm(grad[..., 2, :]) > 0
    assert not grad[..., [0, 1, 3, 4, 5, 6, 7], :].any()
