"""The modern-world scan trainer (``train.modern_vec_train``).

Cheap checks (geometry, masked policy, reward, decoder on a modern state) use
the shared ``tests.world_harness`` world without compiling the tick. The
end-to-end test builds a tiny bank and runs a rollout plus two PPO updates; it
compiles the full tick twice (bank and training program), minutes on CPU, so
run it on the desktop through Slurm.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.tests import world_harness as H

if not H.artifacts_present():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.modern import obs as OB  # noqa: E402
from lanerl_jax.modern import world as MS  # noqa: E402
from lanerl_jax.modern import train as MV  # noqa: E402
from lanerl_jax.modern.actions import MODERN_BUTTON_INDEX, MODERN_BUTTONS, modern_orders_from  # noqa: E402
from lanerl_jax.train.ppo import PPOConfig, factored_log_prob  # noqa: E402

SMALL = MV.modern_policy_config(core="gru", core_norm=True, core_residual=True, **MV.SMALL_POLICY)


@lru_cache(maxsize=1)
def env():
    return MV.make_env(H.world())


def test_lane_segment_spans_the_outer_turrets_and_corridor_distance():
    e = env()
    cfg = e.wcfg
    seg = np.asarray(e.segment)
    xy = np.stack([np.asarray(cfg.unit_x), np.asarray(cfg.unit_y)], -1)[np.asarray(e.outer)]
    # Ends are the turrets' projections onto the lane path, blue first.
    assert np.hypot(*(seg[0] - xy[0])) < 1000 and np.hypot(*(seg[-1] - xy[1])) < 1000
    d = MV.lane_path_distance(jnp.asarray(seg[:, 0]), jnp.asarray(seg[:, 1]), e.segment)
    np.testing.assert_allclose(np.asarray(d), 0.0)
    f = np.asarray(cfg.fountain)
    far = np.asarray(MV.lane_path_distance(jnp.asarray(f[:, 0]), jnp.asarray(f[:, 1]), e.segment))
    assert (far > 2000).all()                     # both fountains are well outside the corridor


def test_masked_policy_shares_params_and_never_samples_masked_buttons():
    pol = MV.MaskedLanePolicy(SMALL, tuple(MODERN_BUTTON_INDEX[b] for b in MV.DEFAULT_BUTTONS_OFF))
    plain = MV.LanePolicy(SMALL)
    args = (jnp.zeros((4, 32, OB.MODERN_ENTITY_DIM)), jnp.zeros((4, 32), bool),
            jnp.zeros((4, OB.MODERN_WORLD_SELF_DIM)), jnp.zeros((4, 6)), jnp.zeros((4, SMALL.core_dim)))
    p = plain.init(jax.random.key(0), *args)
    assert jax.tree.structure(pol.init(jax.random.key(0), *args)) == jax.tree.structure(p)
    lg, _ = pol.apply(p, *args)
    probs = jax.nn.softmax(lg.button, -1)
    for b in MV.DEFAULT_BUTTONS_OFF:
        assert float(probs[:, MODERN_BUTTON_INDEX[b]].max()) == 0.0
    a = jax.random.categorical(jax.random.key(1), lg.button[0], shape=(4000,))
    assert not np.isin(np.asarray(a), [MODERN_BUTTON_INDEX[b] for b in MV.DEFAULT_BUTTONS_OFF]).any()


def test_relative_reward_zero_sum_terms_and_tower():
    e = env()
    cfg = MV.ModernVecConfig(health_loss_gold=100.0, tower_damage_gold=900.0)
    prev = MS.init_state(e.wcfg)
    econ = prev.econ._replace(gold_total=prev.econ.gold_total + jnp.asarray([30.0, 10.0]),
                              xp=prev.econ.xp + jnp.asarray([77.0, 51.0]),
                              gold=prev.econ.gold - 500.0)        # spending is not a reward
    red_t = int(e.outer[1])
    nxt = prev._replace(econ=econ, hp=prev.hp.at[0].add(-prev.max_hp[0] * 0.1).at[red_t].add(-prev.max_hp[red_t] / 6))
    r, terms, credited = MV.modern_relative_reward(prev, nxt, cfg, e)
    assert set(terms) == {"cs", "death", "approach", "xp", "health", "tower"}
    np.testing.assert_allclose(np.asarray(terms["cs"]), [1.0, -1.0], atol=1e-5)          # (30-10)/20
    np.testing.assert_allclose(np.asarray(terms["xp"]), [0.008 * 26, -0.008 * 26], atol=1e-5)
    assert abs(float((terms["cs"] + terms["xp"]).sum())) < 1e-5                          # zero-sum mirror
    np.testing.assert_allclose(float(terms["health"][0]), -0.5, atol=1e-5)
    np.testing.assert_allclose(float(terms["tower"][0]), 7.5, atol=1e-4)
    assert float(terms["tower"][1]) == 0.0 and float(credited[0]) > 0
    np.testing.assert_allclose(np.asarray(r), sum(np.asarray(v) for v in terms.values()), atol=1e-5)
    # Approach: walking from the fountain onto the lane is positive shaping.
    seg = np.asarray(e.segment)
    onto = prev._replace(x=prev.x.at[0].set(seg[1, 0]), y=prev.y.at[0].set(seg[1, 1]))
    _, t2, _ = MV.modern_relative_reward(prev, onto, MV.ModernVecConfig(), e)
    assert float(t2["approach"][0]) > 0 and float(t2["approach"][1]) == 0.0


def test_action_round_trip_on_the_modern_state():
    """Policy-sampled actions decode into ModernOrders for both champions; a click on the enemy
    attacks it, the screen centre moves, masked/no-op buttons give no order."""
    e = env()
    s = MS.init_state(e.wcfg)
    seg = np.asarray(e.segment)
    mx, my = seg[len(seg) // 2]
    s = H.refresh(s._replace(x=s.x.at[0].set(mx).at[1].set(mx + 250.0), y=s.y.at[0].set(my).at[1].set(my)))
    decode = jax.jit(lambda a, st: modern_orders_from(a, st, e.frames))
    b = MODERN_BUTTON_INDEX
    noop = decode((jnp.asarray([b["noop"]] * 2), jnp.asarray([48, 48]), jnp.asarray([27, 27])), s)
    assert not np.asarray(noop.move).any() and (np.asarray(noop.attack) == -1).all()
    mv = decode((jnp.asarray([b["move"]] * 2), jnp.asarray([48, 48]), jnp.asarray([5, 5])), s)
    assert np.asarray(mv.move).all()
    # Click where the enemy is, in each champion's own lane frame.
    from lanerl_jax.obs.frame import delta_to_lane
    from lanerl_rl.constants import N_SCREEN_X, N_SCREEN_Y
    from lanerl_jax.train.actions import _screen_to_centred_lane
    gx, gy = np.meshgrid(np.arange(N_SCREEN_X), np.arange(N_SCREEN_Y), indexing="ij")
    ds, dn = _screen_to_centred_lane(jnp.asarray((gx + 0.5) / N_SCREEN_X, jnp.float32),
                                     jnp.asarray((gy + 0.5) / N_SCREEN_Y, jnp.float32))
    clicks = []
    for me in (0, 1):
        tds, tdn = delta_to_lane(e.frames[me], s.x[1 - me] - s.x[me], s.y[1 - me] - s.y[me])
        i = int(np.argmin(np.asarray((ds - tds) ** 2 + (dn - tdn) ** 2)))
        clicks.append(np.unravel_index(i, gx.shape))
    at = decode((jnp.asarray([b["attack_move"]] * 2), jnp.asarray([c[0] for c in clicks]),
                 jnp.asarray([c[1] for c in clicks])), s)
    np.testing.assert_array_equal(np.asarray(at.attack), [1, 0])
    # The sampler's screen usage follows the decoder: move/casts use the click, recall does not.
    from lanerl_jax.modern.actions import screen_usage
    np.testing.assert_array_equal(np.asarray(screen_usage(jnp.asarray([b["move"], b["recall"], b["q"],
                                                                         b["level_q"], b["ward"]]))),
                                  [1, 0, 1, 0, 1])


def test_observation_adapter_shapes():
    e = env()
    cfg = MV.ModernVecConfig(n_envs=2, policy=SMALL)
    bank = jax.tree.map(lambda a: jnp.stack([a, a]), MV.ModernEnvState(MS.init_state(e.wcfg), jnp.zeros((2,), jnp.int32),
                                                                        jnp.zeros((2,), jnp.float32)))
    built = MV.make_modern_vec_train(cfg, e, bank)
    obs = jax.jit(built["obs"])(MS.init_state(e.wcfg))
    assert obs.entities.shape == (2, 32, OB.MODERN_ENTITY_DIM) and obs.self_vec.shape == (2, OB.MODERN_WORLD_SELF_DIM)
    assert obs.global_vec.shape == (2, 6) and obs.entity_pad_mask.shape == (2, 32)
    with pytest.raises(ValueError):
        MV.make_modern_vec_train(cfg._replace(policy=MV.PolicyConfig()), e, bank)   # legacy widths rejected
    with pytest.raises(ValueError):
        MV.make_modern_vec_train(cfg._replace(buttons_off=("noop",)), e, bank)


@pytest.mark.parametrize("opponent", ["mirror"])
def test_end_to_end_tiny_update(opponent):
    """Tiny bank (1 s of game), 2 envs x 4 decisions: actor/learner log-probs agree, five PPO
    updates run, metrics finite, episodes end and reset from the bank."""
    e = env()
    cfg = MV.ModernVecConfig(n_envs=2, rollout_steps=4, n_updates=2, n_minibatches=2, bank_size=2, start_s=1.0,
                             episode_s=2.0, step_ticks=3, opponent=opponent, policy=SMALL,
                             ppo=PPOConfig.standard(decision_hz=10.0))
    bank = MV.prepare_modern_bank(cfg, e, seed=0)
    assert bank.world.t.shape == (2,) and np.allclose(np.asarray(bank.world.t), 1.0, atol=1e-3)
    assert not np.array_equal(np.asarray(bank.world.key[0]), np.asarray(bank.world.key[1]))
    built = MV.make_modern_vec_train(cfg, e, bank)
    runner = built["initial_runner"](jax.random.key(0))
    runner2, tr, batch = jax.jit(built["rollout"])(runner)
    assert tr.obs_self.shape == (4, 2, 2, OB.MODERN_WORLD_SELF_DIM)
    assert not np.isin(np.asarray(tr.action[0]), [MODERN_BUTTON_INDEX[b] for b in MV.DEFAULT_BUTTONS_OFF]).any()
    lg = built["loss"].forward(runner.params, batch)
    lp = factored_log_prob((lg.button, lg.screen_x, lg.screen_y), batch["action"], batch["uses_screen"])
    np.testing.assert_allclose(np.asarray(lp), np.asarray(batch["log_prob"]), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(runner2.env_state.world.t),
                               np.asarray(runner.env_state.world.t) + 4 * 3 / 30, atol=1e-3)
    # The staggered first episode ends by t=2 s (10 decisions); the next, full one needs 10 more.
    runner3, m = jax.jit(built["run_chunk"], static_argnums=1)(runner2, 5)
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "post_kl", "reward", "reward_cs",
              "reward_xp", "reward_approach", "lane_dist", "button_summoner_d"):
        assert k in m and np.isfinite(np.asarray(m[k])).all(), k
    assert float(np.asarray(m["cs_episodes"]).sum()) > 0          # a full episode ended
    assert int(runner3.step) == 5 * cfg.n_envs * 2 * cfg.rollout_steps
    assert float(np.asarray(runner3.env_state.world.t).max()) <= 2.0 + 1e-3   # reset from the bank
    assert len(MODERN_BUTTONS) == 20 and MODERN_BUTTONS[-1] == "stop"
