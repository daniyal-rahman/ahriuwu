"""The vectorised trainer's invariants: its on-device relative reward is the
collector path's `server_train.relative_reward`; the loop runs for gru and mlp
cores and a scripted opponent; and the rollout's log-probs equal the learner's
recomputation (actor/learner agreement, the silent-ratio bug class)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.sim.config import SimConfig
from lanerl_jax.sim.init import init_lane
from lanerl_jax.train.policy import PolicyConfig
from lanerl_jax.train.ppo import PPOConfig, factored_log_prob
from lanerl_jax.train.vec_train import VecConfig, _relative_reward, make_vec_train


def _bank(k=2):
    return jax.tree.map(lambda *a: jnp.stack(a), *(init_lane(seed=i) for i in range(k)))


def _sim():
    return SimConfig.training().replace(step_ticks=6)


def test_relative_reward_matches_collector_path():
    from lanerl_jax.train import server_train as st
    from lanerl_jax.train.reward import lane_corridor_distance
    cfg = VecConfig(gold_scale=20.0, xp_scale=0.008, enemy_scale=1.0)
    prev = init_lane(seed=0)
    nxt = prev.replace(gold=prev.gold.at[0].add(30.).at[1].add(10.),
                       xp=prev.xp.at[0].add(77.).at[1].add(51.),
                       x=prev.x.at[0].add(400.), y=prev.y.at[1].add(-300.))
    r, terms = _relative_reward(prev, nxt, cfg)
    def stats(s):
        pot = -lane_corridor_distance(s.x[:2], s.y[:2]) / 10000.
        return np.stack([np.asarray(s.cs[:2], np.float32), np.ones(2, np.float32),
                         np.asarray(pot), np.asarray(s.xp[:2], np.float32),
                         np.asarray(s.gold[:2], np.float32)], -1)
    st.RELATIVE_REWARD.update(mode="relative", gold_scale=20.0, xp_scale=0.008, enemy_scale=1.0)
    sb, sa = stats(prev), stats(nxt)
    ref, ref_terms = st.relative_reward(sb, sa, sb[::-1], sa[::-1])
    np.testing.assert_allclose(np.asarray(r), np.asarray(ref), rtol=1e-5, atol=1e-6)
    for k in ("cs", "xp", "approach"):
        np.testing.assert_allclose(np.asarray(terms[k]), np.asarray(ref_terms[k]), rtol=1e-5, atol=1e-6)
    assert float(jnp.abs(terms["death"]).sum()) == 0.0
    # zero-sum in a mirror: the gold and xp terms sum to zero over the two champions
    assert abs(float((terms["cs"] + terms["xp"]).sum())) < 1e-5


@pytest.mark.parametrize("core,opponent", [("gru", "mirror"), ("mlp", "mirror"), ("gru", "lasthit")])
def test_loop_runs_and_actor_learner_agree(core, opponent):
    pcfg = PolicyConfig(core=core, core_norm=(core == "gru"), core_residual=(core == "gru"),
                        d_model=32, n_layers=1, ffn_dim=32, ctx_dim=32, core_dim=32,
                        mlp_hidden=32, mlp_layers=1)
    cfg = VecConfig(n_envs=2, rollout_steps=6, n_updates=2, n_minibatches=2, episode_s=2.0,
                    opponent=opponent, ppo=PPOConfig.standard(decision_hz=10.0), policy=pcfg)
    built = make_vec_train(cfg, _sim(), _bank())
    runner = built["initial_runner"](jax.random.key(0))
    runner2, tr, batch = jax.jit(built["rollout"])(runner)
    lg = built["loss"].forward(runner.params, batch)
    lp = factored_log_prob((lg.button, lg.screen_x, lg.screen_y), batch["action"],
                           batch["uses_screen"], click_mask=batch.get("click_mask"))
    np.testing.assert_allclose(np.asarray(lp), np.asarray(batch["log_prob"]), rtol=1e-4, atol=1e-5)
    # Deadlines are floored at two rollouts from the start, so no done inside
    # the first rollout by design; four more updates (2.4 s of game at 10 Hz)
    # must end every 2-s episode at least once; `cs_episodes` counts only FULL
    # episodes (the staggered first one is excluded), so run 8 updates = 4.8 s.
    runner3, m = jax.jit(built["run_chunk"], static_argnums=1)(runner2, 8)
    assert float(np.asarray(m["cs_episodes"]).sum()) > 0, "no full episode ended in 4.8 s of game"
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "post_kl", "reward",
              "reward_cs", "reward_xp", "reward_approach", "lane_dist", "cs_episodes"):
        assert k in m and np.isfinite(np.asarray(m[k])).all(), k
    assert int(runner3.step) == 8 * cfg.n_envs * cfg.learn_agents * cfg.rollout_steps
    if core == "gru":
        # the carry is zeroed on done, both for the env and the batch's carry0
        assert batch["carry0"].shape == (cfg.n_envs * cfg.learn_agents, 32)
