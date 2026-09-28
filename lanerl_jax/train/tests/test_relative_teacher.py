"""Relative reward and matched heuristic comparison boundary regressions."""
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from lanerl_jax.train import server_train as st
from lanerl_jax.obs.builder import Observation


def frozen_fixture():
    # Two games, learner blue in game 0 and red in game 1.
    collector = object.__new__(st.FrozenOpponentCollector)
    collector.n = collector.n_envs = 2
    collector.T, collector.teams = 1, (0,)
    collector.side = np.array([0, 1])
    collector.rows, collector.opp_rows = np.array([0, 3]), np.array([1, 2])
    obs = Observation(jnp.zeros((4, 1, 16)), jnp.zeros((4, 1), bool),
                      jnp.zeros((4, 16)), jnp.zeros((4, 6)), jnp.zeros((4, 1), jnp.int32))
    return collector, obs


def test_frozen_relative_reward_keeps_both_snapshots_and_opponent_credit(monkeypatch):
    monkeypatch.setitem(st.RELATIVE_REWARD, 'mode', 'relative')
    c, obs = frozen_fixture()
    before = np.array([[0,1,-.2,100,500]] * 4, float)
    after = before.copy()
    # Enemy earns a last hit/XP in BOTH games, learner earns nothing.
    after[c.opp_rows, 4] += 20
    after[c.opp_rows, 3] += 60
    _, b = c._split(obs, before)
    _, a = c._split(obs, after)
    reward, terms = st.step_reward(c, b, a, np.ones(2, bool), .99)
    np.testing.assert_allclose(reward, [-1.48, -1.48], atol=1e-6)
    np.testing.assert_array_equal(terms['death'], [0,0])
    # Calling _split again cannot overwrite the historical enemy snapshot.
    c._split(obs, after + 900)
    np.testing.assert_array_equal(st.enemy_rows(c, b), before[c.opp_rows])
    np.testing.assert_array_equal(st.agent_teams(c), [0,1])


def test_only_relative_gold_xp_and_lane_change_pay(monkeypatch):
    monkeypatch.setitem(st.RELATIVE_REWARD, 'mode', 'relative')
    before = np.array([[0,1,-.2,100,500]], float)
    after = np.array([[100,0,-.1,160,540]], float)  # CS/death cannot add reward.
    enemy_before = before.copy()
    enemy_after = before.copy(); enemy_after[:,3:] += [20,20]
    reward, terms = st.relative_reward(before, after, enemy_before, enemy_after, [True])
    np.testing.assert_allclose(reward, [1 + .008*40 + .5], atol=1e-6)
    assert float(terms['death'][0]) == 0
    # Matching ambient gold/XP increments cancel.
    reward, _ = st.relative_reward(before, before + [0,0,0,60,20], before, before + [0,0,0,60,20])
    np.testing.assert_array_equal(reward, [0])


def test_scripted_opponent_distribution_exactly_reproduces_teacher():
    from lanerl_jax.parity.policy_driver import ScriptedPolicy
    from lanerl_jax.train.scripted_policy import scripted_act
    from lanerl_jax.train.trainer import _sample
    from lanerl_jax.obs.builder import build_observation
    from lanerl_jax.sim.init import init_lane, lane_params
    obs = build_observation(init_lane(), 0, st._lane_frames()[0], params=lane_params())
    batched = jax.tree.map(lambda x: x[None], obs)
    lg = ScriptedPolicy(scripted_act).apply({}, batched.entities, batched.entity_pad_mask, batched.self_vec, batched.global_vec)
    got, _, _ = _sample(lg, jax.random.key(42), None)
    want = scripted_act(obs, None)
    for a,b in zip(got,want): np.testing.assert_array_equal(a, jnp.asarray(b)[None])


def test_evaluation_keeps_sides_terminal_returns_and_exact_episode_counts(monkeypatch):
    monkeypatch.setitem(st.RELATIVE_REWARD, 'mode', 'relative')
    class Collector:
        n, T, teams = 2, 1, (0,)
        side = np.array([0,1])
        enemy_stats = st.FrozenOpponentCollector.enemy_stats
        def __init__(self): self.age=np.zeros(2,int); self.episodes=[0,0]
        def observe(self):
            _, obs = frozen_fixture()
            obs = jax.tree.map(lambda x:x[:2], obs)
            own = np.stack([self.age, np.ones(2), np.zeros(2), self.age*60, self.age*20],1)
            enemy = np.stack([self.age*0, np.ones(2), np.zeros(2), self.age*0, self.age*0],1)
            return obs, np.concatenate([own,enemy],1)
        def step(self,a): self.age+=1; return self.age >= [1,2]
        def restart_done(self,done):
            self.age[done]=0
            for i in np.flatnonzero(done): self.episodes[i]+=1
        def close(self): pass
    run=SimpleNamespace(log=lambda r:None, set_results=lambda **kw:None)
    records=st.evaluate_frozen(Collector(), None, {}, run, seed=2, episodes_per_env=1,
        act_fn=lambda obs,key: (jnp.int32(0),jnp.int32(0),jnp.int32(0)), diag_steps=False)
    assert len(records)==2
    assert [r['team'] for r in records]==[0,1]
    np.testing.assert_allclose([r['reward_return'] for r in records],[1.48,2.96],atol=1e-6)
    assert [r['gold_diff'] for r in records]==[20,40]


def test_improvement_gate_cannot_pass_on_training_metrics_or_one_good_baseline():
    from ops.heuristic_improvement import verdict
    keys=[(s,e,e%2,0) for s in (2,3) for e in range(8)]
    def cohort(reward,gold,cs): return {k:dict(reward_return=reward,gold_diff=gold,cs=cs) for k in keys}
    data=dict(teacher=cohort(0,0,40),clone=cohort(1,10,45),ppo=cohort(2,20,46))
    assert verdict(data)['improved_beyond_teacher_and_clone']
    data['ppo']=cohort(.5,5,42)
    assert not verdict(data)['improved_beyond_teacher_and_clone']
    data['ppo']=cohort(2,20,39)
    assert not verdict(data)['improved_beyond_teacher_and_clone']
    del data['ppo'][keys[0]]
    with pytest.raises(ValueError,match='identical 16'): verdict(data)


def test_eval_canary_preserves_checkpoint_and_checks_eval_completion(monkeypatch):
    from ops import launch
    calls=[]
    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0,stdout='{"mean_cs": 1.0}',stderr='')
    monkeypatch.setattr(launch.subprocess,'run',run)
    launch.canary({'envs':8,'port-base':28980,'out':'example','eval-episodes':1,
                   'resume':'final.msgpack','opponent':'scripted','start-jitter-s':20}, 'test')
    command=calls[0][-1]
    assert '--init-from final.msgpack' in command
    assert '--eval-episodes 1' in command and '--episode-s 130' in command
    assert '--start-jitter-s 0' in command


def test_final_eval_requires_checkpoint_and_training_honors_time_limit(monkeypatch,capsys):
    from ops import launch
    monkeypatch.setattr(launch,'must_exist',lambda label,path:str(path))
    spec={'id':'final','port_base':28900,'args':{},'require_checkpoint':True}
    args=SimpleNamespace(seed=2,init_from=None,resume=None,opponent_ckpt=None)
    with pytest.raises(SystemExit,match='requires --resume'): launch.build_args(spec,args)
    launch.submit({'slurm':{'partition':'gpuhog','cpus':8,'mem':'12G','time':'02:00:00'}}, {}, 'test',True)
    output=capsys.readouterr().out
    assert '--partition=gpuhog' in output and '--time=02:00:00' in output


def test_short_completed_eval_is_not_reported_as_startup_failure(tmp_path, monkeypatch, capsys):
    import json
    from ops import launch
    run = tmp_path/'server-farm-fixture'; run.mkdir()
    (run/'manifest.json').write_text(json.dumps({'results':{'status':'complete'}}))
    monkeypatch.setattr(launch,'live_jobs',lambda:[])
    launch.watch_startup('fixture','unused',completed_dir=tmp_path)
    assert 'completed successfully' in capsys.readouterr().out
