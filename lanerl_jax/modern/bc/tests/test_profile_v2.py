"""The ``modern-world-v2`` observation layout and the policy's identity embeddings (docs/modern/MULTICHAMP.md)."""
import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.modern import obs as OB
from lanerl_jax.modern import world as MS
from lanerl_jax.modern.train import default_loadouts
from lanerl_jax.modern.data import champion_table as CT
from lanerl_jax.modern.rl.policy import LanePolicy, PolicyConfig, make_policy_config


def _world(blue="Garen", red="Jax"):
    cfg = MS.build_config(default_loadouts((blue, red)), lanes=(2,), jungle=False,
                          objectives=False, lane_structures=True)
    state = MS.init_state(cfg)
    return cfg, state, OB.modern_frames(cfg)


def test_v2_carries_the_ids_the_ranks_and_the_descriptor_blocks_and_keeps_the_v1_columns():
    cfg, state, frames = _world()
    v1 = OB.build_modern_observation(state, 0, frames[0], cfg, profile="modern-world-v1")
    v2 = OB.build_modern_observation(state, 0, frames[0], cfg, profile="modern-world-v2")
    a, b = np.asarray(v1.self_vec), np.asarray(v2.self_vec)
    assert a.shape == (OB.MODERN_WORLD_SELF_DIM,) and b.shape == (OB.V2_SELF_DIM,)
    k = len(OB.K.KITS)
    assert b[0] == CT.index_of("Garen") and b[1] == CT.index_of("Jax") and b[2] == CT.ARCHIVE_TIER
    head, tail = a[:16], a[16 + 2 * k:]
    np.testing.assert_allclose(b[3:19], head)
    np.testing.assert_allclose(b[19:31], tail)
    ranks = b[31:35]
    assert ranks.shape == (4,) and np.all(ranks <= 1.0)
    own, enemy = b[35:35 + CT.SIDE_DIM], b[35 + CT.SIDE_DIM:]
    np.testing.assert_allclose(own, CT.descriptors()[CT.index_of("Garen")])
    np.testing.assert_allclose(enemy, CT.descriptors()[CT.index_of("Jax")])
    assert enemy[0] == 125.0 / 600.0                                        # Jax attack range
    assert own[9 + 0 * len(CT.SLOT_COLUMNS)] == 1.0                         # Garen Q is a self-cast
    assert enemy[9 + 0 * len(CT.SLOT_COLUMNS) + 1] == 1.0                   # Jax Q is unit-targeted
    # The other side sees itself as Jax and the enemy as Garen.
    c = np.asarray(OB.build_modern_observation(state, 1, frames[1], cfg, profile="modern-world-v2").self_vec)
    assert c[0] == CT.index_of("Jax") and c[1] == CT.index_of("Garen")
    # Entities and the global vector are the same in both profiles.
    np.testing.assert_allclose(np.asarray(v1.entities), np.asarray(v2.entities))
    np.testing.assert_allclose(np.asarray(v1.global_vec), np.asarray(v2.global_vec))


def test_v2_cooldown_fractions_use_the_table():
    cfg, state, frames = _world()
    champ = state.champ._replace(ranks=state.champ.ranks.at[0].set(jnp.asarray([1, 1, 1, 1])),
                                 cooldowns=state.champ.cooldowns.at[0].set(jnp.asarray([4.0, 11.0, 4.5, 60.0])))
    state = state._replace(champ=champ)
    v2 = np.asarray(OB.build_modern_observation(state, 0, frames[0], cfg, profile="modern-world-v2").self_vec)
    table = CT.cooldowns()[CT.index_of("Garen")][:, 0]                      # rank-1 base cooldowns: 8, 22, 9, 120
    np.testing.assert_allclose(v2[3 + 6:3 + 10], np.clip(np.array([4.0, 11.0, 4.5, 60.0]) / table, 0, 1), atol=1e-6)


def test_policy_embeds_the_ids_only_for_v2():
    cfg, state, frames = _world()
    obs = OB.build_modern_observation(state, 0, frames[0], cfg, profile="modern-world-v2")
    pc = make_policy_config("modern-world-v2", core="mlp", d_model=32, ffn_dim=32, ctx_dim=32, core_dim=32,
                            mlp_hidden=32, mlp_layers=1)
    assert pc.self_dim == OB.V2_SELF_DIM
    policy = LanePolicy(pc)
    params = policy.init(jax.random.key(0), obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
    assert "champion_embed" in params["params"] and "tier_embed" in params["params"]
    assert params["params"]["champion_embed"]["embedding"].shape == (pc.n_champion_ids, pc.champion_embed)
    logits = policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
    assert np.all(np.isfinite(np.asarray(logits.button)))
    # v1 policies have no embedding and reject a v2 observation.
    v1 = LanePolicy(make_policy_config("modern-world-v1", core="mlp", d_model=32, ffn_dim=32, ctx_dim=32, core_dim=32,
                                       mlp_hidden=32, mlp_layers=1))
    o1 = OB.build_modern_observation(state, 0, frames[0], cfg, profile="modern-world-v1")
    p1 = v1.init(jax.random.key(0), o1.entities, o1.entity_pad_mask, o1.self_vec, o1.global_vec)
    assert "champion_embed" not in p1["params"]
    try:
        v1.apply(p1, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
    except ValueError as exc:
        assert "observation widths" in str(exc)
    else:
        raise AssertionError("a v1 policy accepted a v2 observation")


def test_old_checkpoint_configs_without_a_profile_are_read_as_v1():
    from lanerl_jax.modern.rl.policy import restore_policy_config
    saved = PolicyConfig(core="gru")._asdict(); del saved["profile"]; saved["self_dim"] = OB.MODERN_WORLD_SELF_DIM
    pc = restore_policy_config(saved)
    assert pc.profile == "modern-world-v1" and pc.self_dim == OB.MODERN_WORLD_SELF_DIM
    assert restore_policy_config(PolicyConfig()._asdict()).profile == "modern-world-v2"
