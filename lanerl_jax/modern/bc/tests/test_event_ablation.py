from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from lanerl_jax.modern.bc.event_batches import ahri_angle_targets
from lanerl_jax.modern.bc import event_loss as L
from lanerl_jax.modern.bc.event_policy import EventLogits


def test_angle_ablation_keeps_the_original_supervision_set_and_inputs():
    a = dict(button=np.array([3, 5, 3, 4]), valid=np.ones(4, bool),
             aim_valid=np.array([True, False, True, False]), aim_mode=np.array([2, 2, 2, 0]),
             angle=np.array([0., 1., np.nan, np.nan]), self=np.arange(8).reshape(4, 2))
    before = a['self'].copy()
    ahri_angle_targets([dict(row=dict(champion='Ahri'), arrays=a)])
    np.testing.assert_array_equal(a['aim_mode'], [3, 2, 2, 0])
    np.testing.assert_array_equal(a['aim_valid'], [True, False, True, False])
    np.testing.assert_array_equal(a['self'], before)


def test_cast_aim_weight_changes_only_cast_target_loss(monkeypatch):
    logits = EventLogits(jnp.zeros((1, 2, 9)), jnp.zeros((1, 2, 32)),
                        jnp.zeros((1, 2, 6)), jnp.zeros((1, 2, 4)), *([None]*6))
    monkeypatch.setattr(L, 'make_event_forward', lambda policy: lambda params, batch: logits)
    policy = SimpleNamespace(cfg=SimpleNamespace(grid_x=3, grid_y=2, n_angles=4))
    batch = dict(button=jnp.array([[1, 3]]), weight=jnp.ones((1, 2)),
                 aim_valid=jnp.ones((1, 2), bool), aim_mode=jnp.full((1, 2), 2),
                 screen_x=jnp.zeros((1, 2), jnp.int32), screen_y=jnp.zeros((1, 2), jnp.int32),
                 target_row=jnp.zeros((1, 2), jnp.int32), angle=jnp.zeros((1, 2)))
    a, ai = L.make_event_loss(policy)({}, batch)
    b, bi = L.make_event_loss(policy, cast_aim_weight=3.)({}, batch)
    np.testing.assert_allclose(b-a, np.log(6), rtol=1e-5)
    np.testing.assert_allclose(ai['button_nll'], bi['button_nll'])
    np.testing.assert_allclose(ai['noncast_aim_loss_mass'], bi['noncast_aim_loss_mass'])


def test_previous_event_dropout_is_grouped_reproducible_and_preserves_cast_history():
    from lanerl_jax.modern.bc.event_batches import drop_previous_event_history
    batch = dict(self=np.ones((64, 8, 40), np.float32),
                 entities=np.ones((64, 8, 32, 25), np.float32),
                 button=np.arange(512).reshape(64, 8))
    assert drop_previous_event_history(batch, np.random.default_rng(1), 0.) is batch
    a = drop_previous_event_history(batch, np.random.default_rng(1), .3)
    b = drop_previous_event_history(batch, np.random.default_rng(1), .3)
    np.testing.assert_array_equal(a['self'], b['self'])
    selected = np.isnan(a['self'][:, 0, 28])
    assert selected.any() and not selected.all()
    assert np.isnan(a['self'][selected, ..., 28:]).all()
    assert np.isnan(a['entities'][selected, ..., 24:]).all()
    np.testing.assert_array_equal(a['self'][..., :28], batch['self'][..., :28])
    np.testing.assert_array_equal(a['entities'][..., :24], batch['entities'][..., :24])
    np.testing.assert_array_equal(a['self'][~selected], batch['self'][~selected])
    assert a['button'] is batch['button']
    assert np.isfinite(batch['self']).all() and np.isfinite(batch['entities']).all()
