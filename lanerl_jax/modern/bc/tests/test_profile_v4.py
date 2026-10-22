import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import obs as OB, world as MS
from lanerl_jax.modern.rl.policy import LanePolicy, make_policy_config, restore_policy_config
from lanerl_jax.modern.train import default_loadouts
from lanerl_jax.modern.bc.sources import SOURCE_CONTRACT, require_replay_only, ReplayOnlyUnavailable


def test_missingness_is_finite_distinct_from_zero_and_has_finite_parameter_gradients():
    cfg = make_policy_config(OB.REPLAY_PROFILE, d_model=16, n_heads=2, n_layers=1, ffn_dim=16,
                             ctx_dim=16, core_dim=16, mlp_hidden=16, mlp_layers=1,
                             unsupported_self=(9,))
    policy = LanePolicy(cfg)
    ent = jnp.zeros((32, 20))
    pad = jnp.ones((32,), bool).at[0].set(False)
    sv, gv = jnp.zeros(cfg.self_dim), jnp.zeros(6)
    params = policy.init(jax.random.key(0), ent, pad, sv, gv)
    known = policy.apply(params, ent, pad, sv, gv)
    unknown = policy.apply(params, ent.at[0, 3].set(jnp.nan), pad,
                           sv.at[7].set(jnp.nan), gv.at[2].set(jnp.nan))
    assert np.isfinite(np.asarray(unknown.button)).all()
    assert not np.array_equal(np.asarray(known.button), np.asarray(unknown.button))
    # A feature never observed during BC is masked even if the sim supplies it.
    ignored = policy.apply(params, ent, pad, sv.at[9].set(5.), gv)
    np.testing.assert_array_equal(np.asarray(known.button), np.asarray(ignored.button))
    grads = jax.grad(lambda p: policy.apply(p, ent.at[0, 3].set(jnp.nan), pad,
                                          sv.at[7].set(jnp.nan), gv).button.sum())(params)
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grads))
    restored = restore_policy_config(json.loads(json.dumps(cfg._asdict())))
    assert restored == cfg and hash(restored)


def test_v4_sim_inputs_use_fixed_scaling_and_mask_unsupported_fields(monkeypatch):
    cfg = MS.build_config(default_loadouts(("Garen", "Jax")), lanes=(2,), jungle=False,
                          objectives=False, lane_structures=True)
    s = MS.init_state(cfg)
    s = s._replace(champ=s.champ._replace(cooldowns=s.champ.cooldowns.at[0].set(jnp.array([60., 30., 0., 120.]))))
    def forbidden(*args, **kwargs):
        raise AssertionError("v4 must not use kit/catalog cooldown or descriptor tables")
    monkeypatch.setattr(OB.CT, "cooldowns", forbidden)
    monkeypatch.setattr(OB.CT, "descriptors", forbidden)
    out = OB.build_modern_observation(s, 0, OB.modern_frames(cfg)[0], cfg, profile=OB.REPLAY_PROFILE)
    sv = np.asarray(out.self_vec)
    np.testing.assert_allclose(sv[9:13], [1., .5, 0., 2.])
    assert np.isnan(sv[[18, 23, 24, 25, 29, 30]]).all()
    assert np.isnan(sv[35:35 + 2 * OB.CT.SIDE_DIM]).all()
    assert np.isnan(np.asarray(out.global_vec)[2:]).all()


def test_contract_rejects_old_profile_even_with_a_copied_stamp():
    with pytest.raises(ReplayOnlyUnavailable):
        require_replay_only("old", dict(source_contract=SOURCE_CONTRACT, profile="modern-world-v3"))
    require_replay_only("new", dict(source_contract=SOURCE_CONTRACT, profile=OB.REPLAY_PROFILE))
