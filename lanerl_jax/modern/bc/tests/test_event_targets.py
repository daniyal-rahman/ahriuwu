import jax.numpy as jnp
import numpy as np

from lanerl_jax.modern.bc.event_targets import GAREN, pointer_candidates, emission_logits
from lanerl_jax.modern.bc.events import BUTTON_INDEX as B


def test_known_eligibility_and_empty_emission():
    entities = jnp.zeros((4, 25)).at[0, 4].set(1).at[1, 5].set(1).at[2, 4].set(1)
    entities = entities.at[:2, 11].set(1).at[2, 10].set(1)
    pad = jnp.array([False, False, False, True])
    np.testing.assert_array_equal(pointer_candidates(entities, pad, jnp.array(GAREN), jnp.array(B['attack'])), [True, True, False, False])
    np.testing.assert_array_equal(pointer_candidates(entities, pad, jnp.array(GAREN), jnp.array(B['r'])), [True, False, False, False])
    # Unknown spell eligibility stays permissive; it must not silently lose ally targets.
    np.testing.assert_array_equal(pointer_candidates(entities, pad, jnp.array(0), jnp.array(B['r'])), ~pad)
    logits = emission_logits(jnp.arange(9.), entities, jnp.ones(4, bool), jnp.array(GAREN))
    assert logits[B['attack']] < -1000 and logits[B['r']] < -1000


def test_grid_forbids_minimap_and_soft_targets_remain_normalized():
    from lanerl_jax.modern.bc.event_targets import grid_candidates
    from lanerl_jax.modern.bc.event_loss import grid_ce
    mask = grid_candidates()
    assert mask[0, 0] and not mask[-1, -1]
    logits = jnp.where(mask.reshape(-1), 0., -1e4)
    value = grid_ce(logits, jnp.array(75), jnp.array(36))
    np.testing.assert_allclose(value, np.log(int(mask.sum())), rtol=1e-5)
