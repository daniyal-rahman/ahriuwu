"""Source-server collector contracts, independent of live server availability."""
import numpy as np
import jax.numpy as jnp
from lanerl_rl.constants import BUTTON_INDEX
from lanerl_jax.parity.policy_driver import _lane_frames, wire_visibility
from lanerl_jax.train.server_train import screen_order, farm_reward


def test_cursor_command_is_independent_of_entity_identity():
    frame = _lane_frames()[0]
    me = {"x": 3000., "y": 12000., "sl": [1, 1, 1, 1], "id": 111, "dead": False}
    action = (BUTTON_INDEX["attack_move"], 50, 25)
    first = screen_order(action, me, frame)
    second = screen_order(action, {**me, "id": 999, "target": 12345}, frame)
    assert first == second
    assert set(first) == {"t", "button", "x", "y"}


def test_fog_is_server_authoritative_and_missing_flags_fail_closed():
    raw = {"u": [{"id": 11, "vb": 0, "vr": 1}, {"id": 12}]}
    np.testing.assert_array_equal(wire_visibility(raw, [11, 12, 0], 0), [False]*3)
    np.testing.assert_array_equal(wire_visibility(raw, [11, 12, 0], 1), [True, False, False])


def test_shaping_telescopes_independently_of_path_and_pays_nothing_when_still():
    # Same start and terminal potentials, two different intermediate paths.
    totals = []
    for path in ([-.8, -.5, -.2, 0.], [-.8, -.9, -.7, 0.]):
        rewards = []
        for i in range(3):
            r, _ = farm_reward(0., 0., jnp.array(True), jnp.array(True),
                               path[i], path[i+1], i == 2, .9)
            rewards.append(float(r))
        totals.append(sum(rewards))
    np.testing.assert_allclose(totals, [4., 4.], atol=1e-6)
    # `REW-11`: parked far from the lane, at the episode's last step or not,
    # the shaping is exactly zero -- the discounted/zero-terminal form paid
    # (1-gamma)*|Phi| per step plus |Phi| at the cutoff for sitting in base.
    for done in (False, True):
        r, terms = farm_reward(0., 0., jnp.array(True), jnp.array(True), -.8, -.8, done, .9)
        assert float(r) == 0. and float(terms["approach"]) == 0.


def test_own_sealed_spell_is_not_reported_ready_at_zero_cooldown():
    import pytest
    from lanerl_jax.sim.init import init_lane, lane_params
    from lanerl_jax.obs.builder import build_observation
    from lanerl_jax.parity.policy_driver import wire_own_hud, apply_own_hud
    state = init_lane()
    state = state.replace(spell_level=state.spell_level.at[0, 0].set(1))
    obs = build_observation(state, 0, _lane_frames()[0], params=lane_params())
    assert float(obs.self_vec[6]) == 0.
    me = dict(k='Champion', tm=100, dead=False, ad=78., ap=0., ar=30., mr=30., se=[0, 1, 1, 1])
    corrected = apply_own_hud(obs, wire_own_hud({'u': [me]}, 0))
    assert float(corrected.self_vec[6]) == 1.
    del me['se']
    with pytest.raises(ValueError, match='HUD enablement'):
        wire_own_hud({'u': [me]}, 0)


def test_shared_learner_preserves_terminal_rewards_and_budget_checkpoint(tmp_path):
    """Asynchronous resets cannot erase earned CS or cross-contaminate envs."""
    import json
    from lanerl_jax.obs.builder import Observation, N_SLOTS, ENTITY_DIM, SELF_DIM, GLOBAL_DIM
    from lanerl_jax.train.policy import LanePolicy, PolicyConfig
    from lanerl_jax.train.ppo import PPOConfig
    from lanerl_jax.train.run_manifest import RunDir
    from lanerl_jax.train.server_train import run_farming_learner

    class Collector:
        n = 2

        def __init__(self):
            self.episodes = [0, 0]
            self.age = np.zeros(2, dtype=np.int32)
            self.closed = False

        def observe(self):
            obs = Observation(jnp.zeros((2, N_SLOTS, ENTITY_DIM)),
                jnp.ones((2, N_SLOTS), bool), jnp.zeros((2, SELF_DIM)),
                jnp.zeros((2, GLOBAL_DIM)).at[:, 0].set(self.age / 3.),
                jnp.full((2, N_SLOTS), -1, dtype=jnp.int32))
            stats = np.stack([self.age * [1., 2.], np.ones(2), np.zeros(2)], axis=1)
            return obs, stats

        def spell_ranks(self):
            return np.zeros((2, 4), dtype=np.int32)

        def step(self, actions):
            assert actions.shape == (2, 3)
            self.age += 1
            return self.age >= [2, 3]

        def restart_done(self, done):
            for i in np.flatnonzero(done):
                self.episodes[i] += 1
                self.age[i] = 0

        def close(self):
            self.closed = True

    collector = Collector()
    policy = LanePolicy(PolicyConfig(d_model=8, n_layers=1, n_heads=1, ffn_dim=8,
        ctx_dim=8, core_dim=8, mlp_hidden=8, mlp_layers=1))
    run = RunDir(tmp_path, 'shared-learner', {})
    run_farming_learner(collector, policy, PPOConfig(lr=1e-3, critic_lr=1e-3), run,
                        seed=4, rollout=4, updates=2, save_updates=[1])
    assert collector.closed
    assert collector.episodes == [4, 2]
    assert run.manifest['results']['parameters_changed']
    assert run.manifest['results']['status'] == 'complete'
    assert (run.path / 'budget_000000008.msgpack').is_file()
    rows = [json.loads(line) for line in (run.path / 'metrics.jsonl').read_text().splitlines()]
    updates = [row for row in rows if 'steps' in row]
    assert [row['steps'] for row in updates] == [8, 16]
    for row in updates:
        assert row['reward_cs'] == 1.5
        assert sum(row['sampled_buttons'].values()) == 8
        assert row['sampled_r_unranked'] == row['sampled_buttons']['r']
    assert [row['cs'] for row in rows if 'episode' in row].count(2.) == 4
    assert [row['cs'] for row in rows if 'episode' in row].count(6.) == 2


def test_shared_learner_records_initial_observation_failure_and_closes(tmp_path):
    import pytest
    from lanerl_jax.train.ppo import PPOConfig
    from lanerl_jax.train.run_manifest import RunDir
    from lanerl_jax.train.server_train import run_farming_learner

    class BrokenCollector:
        closed = False

        def observe(self):
            raise RuntimeError('collector failed before first action')

        def close(self):
            self.closed = True

    collector = BrokenCollector()
    run = RunDir(tmp_path, 'broken-collector', {})
    with pytest.raises(RuntimeError, match='before first action'):
        run_farming_learner(collector, None, PPOConfig(), run,
                            seed=0, rollout=1, updates=1)
    assert collector.closed
    assert run.manifest['results']['status'] == 'failed'


def test_authoritative_positive_hp_death_respawn_and_reward_once():
    from lanerl_jax.parity.policy_driver import (StateRebuilder, wire_own_hud,
                                                apply_own_hud)
    from lanerl_jax.obs.builder import build_observation
    from lanerl_jax.sim.init import lane_params
    from lanerl_jax.train.server_train import source_farm_stats, WaveStart, WAVE_START_POS
    from lanerl_jax.train.server_eval import summarize_frames
    import pytest

    def champion(team, dead, hp):
        return dict(id=team, k='Champion', tm=team, dead=dead, hp=hp, mhp=672,
                    x=WAVE_START_POS[0]+(team-100)*4, y=WAVE_START_POS[1],
                    cs=0, lvl=1, sl=[0,0,1,0], se=[1,1,1,1],
                    ad=78., ap=0., ar=30., mr=30., vb=1, vr=1, mo=10)
    # HP regeneration during one death must not create respawns or new deaths.
    life = [(False,600), (True,0), (True,16), (True,40), (False,672)]
    rebuilder, stats, frames = StateRebuilder(), [], []
    for i, (dead, hp) in enumerate(life):
        me = champion(100, dead, hp)
        raw = dict(t=i*1000, u=[me, champion(200, True, 35)])
        frames.append(raw)
        state, _ = rebuilder.rebuild(raw)
        assert bool(state.alive[0]) is (not dead)
        assert not bool(state.alive[1])  # Enemy corpse cannot enter visible units.
        obs = build_observation(state, 0, _lane_frames()[0], params=lane_params())
        obs = apply_own_hud(obs, wire_own_hud(raw, 0))
        assert float(obs.self_vec[14]) == float(dead)
        assert not np.any(np.asarray(obs.slot_unit)[~np.asarray(obs.entity_pad_mask)] == 1)
        stats.append(source_farm_stats(me, 0.))
        if dead:
            assert screen_order((BUTTON_INDEX['e'],50,25), me, _lane_frames()[0]) == {'t':'noop'}
            with pytest.raises(RuntimeError, match='died'):
                WaveStart().order(me, 120000)
    penalties = []
    for before, after in zip(stats, stats[1:]):
        _, terms = farm_reward(before[0], after[0], np.bool_(before[1]), np.bool_(after[1]),
                               0., 0., False, .99)
        penalties.append(float(terms['death']))
    assert penalties == [-2., 0., 0., 0.]
    assert summarize_frames(frames)['champions']['100']['deaths'] == 1


def test_missing_or_non_boolean_dead_fails_before_life_inference():
    import pytest
    from lanerl_jax.parity.policy_driver import champion_dead, StateRebuilder, wire_own_hud
    for value in (None, 0, 1, 'false'):
        me = dict(k='Champion', tm=100, hp=600, dead=value)
        with pytest.raises(ValueError, match='dead Boolean'):
            champion_dead(me)
        with pytest.raises(ValueError, match='dead Boolean'):
            StateRebuilder().rebuild({'u':[me]})
        with pytest.raises(ValueError, match='dead Boolean'):
            wire_own_hud({'u':[me]}, 0)


def test_shared_learner_learns_rewarded_action_and_zero_lr_is_invariant(tmp_path, monkeypatch):
    """A one-step bandit tests actor credit, not League skill or critic movement.

    Every transition is terminal: the independently known return is exactly
    one iff the recorded button earned CS. The optimizer batch must preserve
    that action/return association despite rollout stacking and resets.
    """
    import jax
    from flax.serialization import msgpack_restore
    from lanerl_jax.obs.builder import Observation, N_SLOTS, ENTITY_DIM, SELF_DIM, GLOBAL_DIM
    from lanerl_jax.train.policy import LanePolicy, PolicyConfig
    from lanerl_jax.train.ppo import PPOConfig
    from lanerl_jax.train.run_manifest import RunDir
    from lanerl_jax.train import server_train

    rewarded_button = BUTTON_INDEX['w']  # Arbitrary label; no game semantics.

    class BanditCollector:
        n = 8

        def __init__(self):
            self.episodes = [0] * self.n
            self.cs = np.zeros(self.n)
            self.closed = False
            self.earned = 0
            self.obs = Observation(jnp.zeros((self.n, N_SLOTS, ENTITY_DIM)),
                jnp.ones((self.n, N_SLOTS), bool),
                jnp.zeros((self.n, SELF_DIM)).at[:, 0].set(.75),
                jnp.zeros((self.n, GLOBAL_DIM)),
                jnp.full((self.n, N_SLOTS), -1, dtype=jnp.int32))

        def observe(self):
            return self.obs, np.stack([self.cs, np.ones(self.n), np.zeros(self.n)], axis=1)

        def spell_ranks(self):
            return np.ones((self.n, 4), dtype=np.int32)

        def step(self, actions):
            self.cs = (actions[:, 0] == rewarded_button).astype(float)
            self.earned += int(self.cs.sum())
            return np.ones(self.n, bool)

        def restart_done(self, done):
            assert done.all()
            self.episodes = [n + 1 for n in self.episodes]
            self.cs[:] = 0

        def close(self):
            self.closed = True

    # Observe the actual optimizer batch, not a separately assembled replica.
    make_learner = server_train.make_learner
    checked_batches = []

    def checked_learner(policy, cfg):
        tx, loss = make_learner(policy, cfg)

        def check(action, returns, advantages, values, context):
            expected = (np.asarray(action) == rewarded_button).astype(float)
            np.testing.assert_allclose(returns, expected, atol=2e-6, rtol=0)
            np.testing.assert_allclose(advantages, expected - np.asarray(values), atol=2e-6, rtol=0)
            np.testing.assert_array_equal(context, .75)
            checked_batches.append(len(expected))

        def checked_loss(params, batch, ppo):
            jax.debug.callback(check, batch['action'][0], batch['returns'],
                batch['adv'], batch['value'], batch['self'][:, 0], ordered=True)
            return loss(params, batch, ppo)
        return tx, checked_loss

    monkeypatch.setattr(server_train, 'make_learner', checked_learner)
    policy = LanePolicy(PolicyConfig(d_model=8, n_layers=1, n_heads=1, ffn_dim=8,
        ctx_dim=8, core_dim=8, mlp_hidden=8, mlp_layers=1))
    initial_trees = []
    for lr in (0., .01):
        collector = BanditCollector()
        run = RunDir(tmp_path, f'actor-credit-lr{lr}', {})
        cfg = PPOConfig(lr=lr, critic_lr=0., value_coef=0., entropy_coef=0.,
                        epochs=1, normalize_advantage=False)
        server_train.run_farming_learner(collector, policy, cfg, run,
                                        seed=7, rollout=8, updates=3)
        jax.effects_barrier()
        before = msgpack_restore((run.path/'initial.msgpack').read_bytes())['params']
        after = msgpack_restore((run.path/'ckpt_latest.msgpack').read_bytes())['params']
        initial_trees.append(before)
        assert collector.closed and collector.episodes == [24] * 8
        assert 0 < collector.earned < 192  # Both rewarded and unrewarded actions exercised.
        if lr == 0:
            for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
                np.testing.assert_array_equal(a, b)
            assert run.manifest['results']['parameters_changed'] is False
        else:
            def probability(params):
                logits = policy.apply(params, collector.obs.entities,
                    collector.obs.entity_pad_mask, collector.obs.self_vec,
                    collector.obs.global_vec)
                return float(jax.nn.softmax(logits.button)[0, rewarded_button])
            assert probability(after) > probability(before) + .005
    for a, b in zip(jax.tree.leaves(initial_trees[0]), jax.tree.leaves(initial_trees[1])):
        np.testing.assert_array_equal(a, b)
    assert checked_batches == [64] * 6
