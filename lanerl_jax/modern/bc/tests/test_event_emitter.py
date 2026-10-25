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

    def apply(self, params, *args, method):
        if method.__name__ == 'encode':
            return args[0], args[-1]+1, jnp.zeros((1, 1, 1))
        if method.__name__ == 'buttons':
            return jnp.full((1, 9), -100.).at[:, 1].set(0.).at[:, 3].set(0.)
        target = jax.nn.one_hot(args[-1], 9)*10.
        return target, target, target


def test_sampled_targets_follow_emitted_button_and_rng_survives_chunking():
    predict = make_predict(FakePolicy(), sampled=True)
    batch = {k: jnp.zeros((1, 16)) for k in INPUTS}
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
