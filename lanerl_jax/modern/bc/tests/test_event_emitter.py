import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.modern.bc.event_eval import make_predict
from lanerl_jax.modern.bc.event_loss import INPUTS


class FakePolicy:
    def encode(self):
        pass

    def buttons(self):
        pass

    def targets(self):
        pass

    def apply(self, params, *args, method, **kwargs):
        if method.__name__ == 'encode':
            return jnp.zeros((1, 1)), args[-1]+1, jnp.zeros((1, 1, 1))
        if method.__name__ == 'buttons':
            return jnp.full((1, 9), -100.).at[:, 1].set(0.).at[:, 3].set(0.)
        target = jax.nn.one_hot(args[-1], 9)*10.
        return target, target, target


def test_sampled_targets_follow_emitted_button_and_rng_survives_chunking():
    predict = make_predict(FakePolicy(), sampled=True)
    batch = {k: jnp.zeros((1, 16)) for k in INPUTS}
    batch['entities'] = jnp.zeros((1, 16, 32, 25))
    batch['button'] = jnp.full((1, 16), 2, jnp.int32)
    start, key, bias = jnp.zeros((1, 1)), jax.random.key(0), jnp.zeros(9)
    state, full = predict({}, start, batch, bias, key)
    first = {k: v[:, :7] for k, v in batch.items()}
    second = {k: v[:, 7:] for k, v in batch.items()}
    (carry, middle), a = predict({}, start, first, bias, key)
    end, b = predict({}, carry, second, bias, middle)
    np.testing.assert_array_equal(full['button'], jnp.concatenate([a['button'], b['button']], 1))
    np.testing.assert_array_equal(full['actual_grid'], full['button'])
    np.testing.assert_array_equal(full['teacher_grid'], 2)
    assert set(np.asarray(full['button']).ravel()) == {1, 3}
    np.testing.assert_array_equal(state[0], end[0])


def test_periodic_memory_reset_preserves_emitter_rng_stream(monkeypatch):
    from types import SimpleNamespace
    from lanerl_jax.modern.bc import event_eval as module
    monkeypatch.setattr(module, 'event_batch', lambda *args, **kwargs: dict(button=np.zeros((1, 2))))
    game = dict(row=dict(game='fixed-game'), arrays=dict(button=np.zeros(6)))
    policy = SimpleNamespace(initial_carry=lambda shape: np.zeros((1, 1)))
    keys = []
    def predict(params, carry, batch, bias, key):
        key, draw = jax.random.split(key)
        keys.append(np.asarray(jax.random.key_data(draw)))
        return (carry+2, key), dict(button=np.full((1, 2), float(carry[0, 0])))
    a = module.predict_game({}, policy, predict, game, chunk=2, seed=0)
    first = keys.copy()
    keys.clear()
    b = module.predict_game({}, policy, predict, game, chunk=2, seed=0, reset_every=4)
    np.testing.assert_array_equal(a['button'], [0, 0, 2, 2, 4, 4])
    np.testing.assert_array_equal(b['button'], [0, 0, 2, 2, 0, 0])
    np.testing.assert_array_equal(keys, first)
