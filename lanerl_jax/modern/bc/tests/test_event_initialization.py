import json
import hashlib

from flax import serialization
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.bc.event_policy import EventPolicyConfig
from lanerl_jax.modern.bc.event_train import initialize_parameters


def checkpoint(tmp_path, *, shape=(2, 3), config=None):
    cfg = EventPolicyConfig() if config is None else config
    legacy = cfg._asdict()
    legacy.pop('next_event_horizon')
    legacy.pop('previous_event_history')
    saved = dict(params=dict(weight=np.ones(shape, np.float32)), opt_state=dict(bogus=np.nan),
                 step=1000, contract_json=json.dumps(dict(config=legacy, champion='all')))
    path = tmp_path/'donor.msgpack'
    path.write_bytes(serialization.to_bytes(saved))
    return path


def test_initialization_transfers_weights_and_records_donor_without_optimizer(tmp_path):
    path = checkpoint(tmp_path)
    params, receipt = initialize_parameters(path, EventPolicyConfig(), dict(weight=jnp.zeros((2, 3))))
    np.testing.assert_array_equal(params['weight'], 1.)
    assert set(params) == {'weight'}
    assert receipt['step'] == 1000
    assert receipt['checkpoint_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt['source_contract']['champion'] == 'all'
    assert receipt['optimizer'].startswith('fresh')


def test_initialization_rejects_architecture_and_shape_mismatches(tmp_path):
    path = checkpoint(tmp_path)
    with pytest.raises(ValueError, match='architecture'):
        initialize_parameters(path, EventPolicyConfig(core_dim=256), dict(weight=jnp.zeros((2, 3))))
    with pytest.raises(ValueError, match='shapes'):
        initialize_parameters(path, EventPolicyConfig(), dict(weight=jnp.zeros((3, 2))))
    with pytest.raises(ValueError, match='structure'):
        initialize_parameters(path, EventPolicyConfig(), dict(other=jnp.zeros((2, 3))))
