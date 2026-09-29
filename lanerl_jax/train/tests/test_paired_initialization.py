"""Exploration initialization changes only the requested button biases."""
import numpy as np
import jax
import jax.numpy as jnp
from lanerl_jax.train.paired_vec_train import adjust_button_bias


def test_button_shift_preserves_farming_conditional_and_other_weights():
    bias=jnp.array([-5.,2.,1.,-15.,-15.,-15.,-15.,-10.])
    params={'params':{'button':{'bias':bias,'kernel':jnp.ones((3,8))},
                      'screen_x':{'bias':jnp.arange(4.)},'other':jnp.array([3.])}}
    altered=adjust_button_bias(params,{'w':10.,'e':10.})
    np.testing.assert_array_equal(params['params']['button']['bias'],bias)
    expected=np.asarray(bias).copy();expected[[4,5]]+=10
    np.testing.assert_array_equal(altered['params']['button']['bias'],expected)
    for k in ('screen_x','other'):
        for a,b in zip(jax.tree.leaves(params['params'][k]),jax.tree.leaves(altered['params'][k])):
            np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(altered['params']['button']['kernel'],params['params']['button']['kernel'])
    old=np.asarray(jax.nn.softmax(bias));new=np.asarray(jax.nn.softmax(altered['params']['button']['bias']))
    np.testing.assert_allclose(new[1]/new[2],old[1]/old[2],rtol=1e-6)
    assert new[4]>old[4]*1000
    assert type(altered) is type(params)
